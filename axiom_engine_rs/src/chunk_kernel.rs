use candle_core::{bail, DType, Result, Tensor};
use std::fmt;

/// Chunk-fused TTT prefill kernel utilities.
pub struct ChunkFusedTTT;

#[derive(Debug)]
pub enum EngineError {
    InvalidTensorBounds(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InvalidTensorBounds(msg) => write!(f, "{msg}"),
        }
    }
}

impl From<EngineError> for candle_core::Error {
    fn from(value: EngineError) -> Self {
        candle_core::Error::Msg(value.to_string())
    }
}

impl ChunkFusedTTT {
    /// Perform chunk-wise fused prefill over Q/K/V tensors.
    ///
    /// Supports:
    /// - `[B, T, D]` (single implicit head)
    /// - `[B, H, T, D]` (multi-head explicit)
    ///
    /// Returns `(output, global_session_weight)`.
    pub fn forward_chunk_fused(
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        block_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        if block_size == 0 {
            return Err(EngineError::InvalidTensorBounds(
                "block_size must be greater than zero".to_string(),
            )
            .into());
        }
        if queries.dims() != keys.dims() || queries.dims() != values.dims() {
            bail!(
                "Q/K/V shape mismatch: q={:?} k={:?} v={:?}",
                queries.dims(),
                keys.dims(),
                values.dims()
            );
        }

        let original_dtype = queries.dtype();
        let rank = queries.dims().len();

        let (q4, k4, v4, squeeze_head) = match rank {
            3 => (
                queries.unsqueeze(1)?.to_dtype(DType::F32)?,
                keys.unsqueeze(1)?.to_dtype(DType::F32)?,
                values.unsqueeze(1)?.to_dtype(DType::F32)?,
                true,
            ),
            4 => (
                queries.to_dtype(DType::F32)?,
                keys.to_dtype(DType::F32)?,
                values.to_dtype(DType::F32)?,
                false,
            ),
            _ => bail!(
                "forward_chunk_fused expects rank-3 or rank-4 tensors, got rank {}",
                rank
            ),
        };

        let (b, h, t, d) = q4.dims4()?;
        if t == 0 || d == 0 || b == 0 || h == 0 {
            return Err(EngineError::InvalidTensorBounds(format!(
                "invalid Q tensor bounds: (b={b}, h={h}, t={t}, d={d})"
            ))
            .into());
        }
        let mut global_session_weight = Tensor::zeros((b, h, d, d), DType::F32, queries.device())?;
        let inv_sqrt_d = Tensor::new(1f32 / (d as f32).sqrt(), queries.device())?;
        let mut cached_mask: Option<(usize, Tensor)> = None;
        let mut cached_inv_len: Option<(usize, Tensor)> = None;

        let mut chunk_outputs: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        while start < t {
            let len = (t - start).min(block_size);
            if len == 0 || start + len > t {
                return Err(EngineError::InvalidTensorBounds(format!(
                    "invalid chunk bounds start={start} len={len} t={t}"
                ))
                .into());
            }
            let q_chunk = q4.narrow(2, start, len)?.contiguous()?;
            let k_chunk = k4.narrow(2, start, len)?.contiguous()?;
            let v_chunk = v4.narrow(2, start, len)?.contiguous()?;
            let (_, _, q_len, q_dim) = q_chunk.dims4()?;
            if q_len != len || q_dim != d {
                return Err(EngineError::InvalidTensorBounds(format!(
                    "q_chunk bounds mismatch expected len={len},d={d} got len={q_len},d={q_dim}"
                ))
                .into());
            }

            // Intra-chunk Gram matrix: [B, H, C, C]
            let gram = q_chunk
                .matmul(&k_chunk.transpose(2, 3)?.contiguous()?)?
                .broadcast_mul(&inv_sqrt_d)?;

            // Causal lower-triangular mask for local chunk processing.
            let local_mask = match &cached_mask {
                Some((cached_len, mask)) if *cached_len == len => mask.clone(),
                _ => {
                    let mask = Tensor::tril2(len, DType::F32, queries.device())?
                        .unsqueeze(0)?
                        .unsqueeze(0)?;
                    cached_mask = Some((len, mask.clone()));
                    mask
                }
            };
            let softmax = gram.broadcast_mul(&local_mask)?;

            // Fused local context: [B, H, C, D]
            let local_context = softmax.matmul(&v_chunk)?;

            // Dense local update in chunk-local memory layout bounds.
            // Update shape: [B, H, D, D]
            let inv_len = match &cached_inv_len {
                Some((cached_len, inv)) if *cached_len == len => inv.clone(),
                _ => {
                    let inv = Tensor::new(1f32 / len as f32, queries.device())?;
                    cached_inv_len = Some((len, inv.clone()));
                    inv
                }
            };
            let local_grads = v_chunk
                .transpose(2, 3)?
                .contiguous()?
                .matmul(&k_chunk)?
                .broadcast_mul(&inv_len)?;
            global_session_weight = global_session_weight.add(&local_grads)?;

            // Carry chunk boundary state to next chunk via global matrix.
            let boundary_carry = q_chunk.matmul(&global_session_weight)?;
            let chunk_output = local_context.add(&boundary_carry)?;
            chunk_outputs.push(chunk_output);

            drop(boundary_carry);
            drop(local_grads);
            drop(local_context);
            drop(softmax);
            drop(local_mask);
            drop(gram);
            drop(q_chunk);
            drop(k_chunk);
            drop(v_chunk);
            start += len;
        }

        if chunk_outputs.is_empty() {
            return Err(EngineError::InvalidTensorBounds(
                "chunk-fused forward produced empty output".to_string(),
            )
            .into());
        }
        let chunk_output_refs: Vec<&Tensor> = chunk_outputs.iter().collect();
        let fused_output = Tensor::cat(&chunk_output_refs, 2)?;

        if squeeze_head {
            Ok((
                fused_output.squeeze(1)?.to_dtype(original_dtype)?,
                global_session_weight.squeeze(1)?,
            ))
        } else {
            Ok((
                fused_output.to_dtype(original_dtype)?,
                global_session_weight,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ChunkFusedTTT;
    use candle_core::{Device, Result, Tensor};

    #[test]
    fn chunk_fused_preserves_expected_shapes() -> Result<()> {
        let device = Device::Cpu;
        let b = 1usize;
        let h = 2usize;
        let t = 129usize;
        let d = 8usize;
        let len = b * h * t * d;
        let data: Vec<f32> = (0..len).map(|i| ((i % 97) as f32) / 97.0).collect();
        let q = Tensor::from_vec(data.clone(), (b, h, t, d), &device)?;
        let k = Tensor::from_vec(data.clone(), (b, h, t, d), &device)?;
        let v = Tensor::from_vec(data, (b, h, t, d), &device)?;

        let (out, w) = ChunkFusedTTT::forward_chunk_fused(&q, &k, &v, 64)?;
        assert_eq!(out.dims4()?, (b, h, t, d));
        assert_eq!(w.dims4()?, (b, h, d, d));
        Ok(())
    }
}
