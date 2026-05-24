use candle_core::{bail, DType, Result, Tensor};

const NF4_BLOCK_SIZE: usize = 64;
const NF4_PACKED_BLOCK_BYTES: usize = NF4_BLOCK_SIZE / 2;

/// Standard 16-level NormalFloat lookup table (zero-mean Gaussian quantiles).
pub const NF4_TABLE: [f32; 16] = [
    -1.0, -0.6961947, -0.5250761, -0.3949174, -0.2844413, -0.1847734, -0.0910502, 0.0, 0.0795802,
    0.1609302, 0.246115, 0.3379152, 0.4407098, 0.562617, 0.7229568, 1.0,
];

/// 4-bit NF4 quantizer for idle session state compression.
pub struct NF4Quantizer;

impl NF4Quantizer {
    fn nearest_nf4_index(value: f32) -> u8 {
        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        for (i, q) in NF4_TABLE.iter().enumerate() {
            let dist = (value - q).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        best_idx as u8
    }

    /// Quantize a weight tensor into packed NF4 indices + block scales.
    ///
    /// Returns:
    /// - `packed_indices`: `u8` tensor shaped `[num_blocks, 32]`
    /// - `scale`: `f16` tensor shaped `[num_blocks]`
    pub fn quantize_state(weight: &Tensor) -> Result<(Tensor, Tensor)> {
        let flat = weight
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if flat.is_empty() {
            bail!("cannot quantize an empty tensor");
        }

        let num_blocks = flat.len().div_ceil(NF4_BLOCK_SIZE);
        let mut packed = vec![0u8; num_blocks * NF4_PACKED_BLOCK_BYTES];
        let mut scales = vec![0f32; num_blocks];

        for block_idx in 0..num_blocks {
            let block_start = block_idx * NF4_BLOCK_SIZE;
            let block_end = (block_start + NF4_BLOCK_SIZE).min(flat.len());
            let block = &flat[block_start..block_end];
            let abs_max = block.iter().fold(0f32, |acc, x| acc.max(x.abs()));
            let scale = if abs_max > 0f32 { abs_max } else { 1f32 };
            scales[block_idx] = scale;

            for pair in 0..NF4_PACKED_BLOCK_BYTES {
                let i0 = pair * 2;
                let i1 = i0 + 1;

                let val0 = if block_start + i0 < block_end {
                    flat[block_start + i0] / scale
                } else {
                    0f32
                };
                let val1 = if block_start + i1 < block_end {
                    flat[block_start + i1] / scale
                } else {
                    0f32
                };

                let lo = Self::nearest_nf4_index(val0) & 0x0F;
                let hi = (Self::nearest_nf4_index(val1) & 0x0F) << 4;
                packed[block_idx * NF4_PACKED_BLOCK_BYTES + pair] = lo | hi;
            }
        }

        let packed_tensor = Tensor::from_vec(
            packed,
            (num_blocks, NF4_PACKED_BLOCK_BYTES),
            weight.device(),
        )?;
        let scale_tensor =
            Tensor::from_vec(scales, (num_blocks,), weight.device())?.to_dtype(DType::F16)?;
        Ok((packed_tensor, scale_tensor))
    }

    /// Dequantize packed NF4 indices with block scales back to f32 tensor.
    ///
    /// Returns shape `[num_blocks, 64]`.
    pub fn dequantize_state(packed_indices: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let (num_blocks, packed_width) = packed_indices.dims2()?;
        if packed_width == 0 {
            bail!("packed indices tensor must have non-zero packed width");
        }

        let packed = packed_indices
            .to_dtype(DType::U8)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let scales = scale
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if scales.len() != num_blocks {
            bail!(
                "scale size mismatch: expected {num_blocks}, got {}",
                scales.len()
            );
        }

        let mut out = vec![0f32; num_blocks * packed_width * 2];
        for block_idx in 0..num_blocks {
            let s = scales[block_idx];
            for i in 0..packed_width {
                let byte = packed[block_idx * packed_width + i];
                let lo_idx = (byte & 0x0F) as usize;
                let hi_idx = ((byte >> 4) & 0x0F) as usize;
                let out_base = (block_idx * packed_width + i) * 2;
                out[out_base] = NF4_TABLE[lo_idx] * s;
                out[out_base + 1] = NF4_TABLE[hi_idx] * s;
            }
        }

        Tensor::from_vec(out, (num_blocks, packed_width * 2), packed_indices.device())
    }
}

#[cfg(test)]
mod tests {
    use super::NF4Quantizer;
    use candle_core::{Device, Result, Tensor};

    #[test]
    fn nf4_roundtrip_reconstructs_expected_shape() -> Result<()> {
        let device = Device::Cpu;
        let src: Vec<f32> = (0..130)
            .map(|i| (i as f32 * 0.03125).sin() * 0.75)
            .collect();
        let w = Tensor::from_vec(src, (1, 130), &device)?;
        let (packed, scales) = NF4Quantizer::quantize_state(&w)?;
        let deq = NF4Quantizer::dequantize_state(&packed, &scales)?;
        let (_, recovered_width) = deq.dims2()?;
        assert_eq!(packed.dims2()?.1, 32);
        assert_eq!(recovered_width, 64);
        Ok(())
    }
}
