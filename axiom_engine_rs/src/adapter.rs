use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};

use crate::config::AxiomConfig;
use crate::ttt_layer::TTTLinearLayer;

const DEFAULT_CHUNK_FUSED_INNER_STEPS: usize = 4;

pub struct ChunkFusedTTT {
    layer: TTTLinearLayer,
    _varmap: VarMap,
}

impl ChunkFusedTTT {
    pub fn new(config: AxiomConfig, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let layer = TTTLinearLayer::new(vb.pp("chunk_fused_ttt"), config)?;
        Ok(Self {
            layer,
            _varmap: varmap,
        })
    }

    pub fn forward_chunk_fused(
        &self,
        x: &Tensor,
        initial_state: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_, seq_len, _) = x.dims3()?;
        let mut state = initial_state.clone();
        let mut outputs = Vec::with_capacity(seq_len);
        for token_index in 0..seq_len {
            let token = x.narrow(1, token_index, 1)?;
            let (output, next_state) =
                self.layer
                    .forward_decode(&token, &state, Some(DEFAULT_CHUNK_FUSED_INNER_STEPS))?;
            outputs.push(output);
            state = next_state;
        }
        let output_refs = outputs.iter().collect::<Vec<_>>();
        Ok((Tensor::cat(&output_refs, 1)?, state))
    }
}

pub struct TTTTransformerAdapter {
    frozen_projection: Tensor,
    chunk_fused_ttt: ChunkFusedTTT,
    layer_index: usize,
}

impl TTTTransformerAdapter {
    pub fn new(config: AxiomConfig, device: &Device, layer_index: usize) -> Result<Self> {
        if layer_index >= config.n_layers {
            candle_core::bail!(
                "adapter layer index {layer_index} is out of range for {} layers",
                config.n_layers
            );
        }
        let frozen_projection = Tensor::eye(config.d_model, DType::F32, device)?;
        let chunk_fused_ttt = ChunkFusedTTT::new(config, device)?;
        Ok(Self {
            frozen_projection,
            chunk_fused_ttt,
            layer_index,
        })
    }

    /// `alpha` interpolates between the frozen base-model path (`1.0`) and the
    /// adaptive TTT memory path (`0.0`) for the returned final-token state.
    pub fn forward_hybrid(
        &self,
        x: &Tensor,
        sliding_window_tokens: &Tensor,
        session_states: &mut Vec<Tensor>,
        alpha: f32,
    ) -> Result<Tensor> {
        let (batch, seq_len, d_model) = x.dims3()?;
        let (memory_batch, memory_seq_len, memory_d_model) = sliding_window_tokens.dims3()?;
        if batch != memory_batch {
            candle_core::bail!(
                "batch mismatch between base activations ({batch}) and sliding window ({memory_batch})"
            );
        }
        if d_model != memory_d_model {
            candle_core::bail!(
                "d_model mismatch between base activations ({d_model}) and sliding window ({memory_d_model})"
            );
        }
        if self.layer_index >= session_states.len() {
            candle_core::bail!(
                "session state count {} does not contain adapter layer {}",
                session_states.len(),
                self.layer_index
            );
        }
        if memory_seq_len == 0 || seq_len == 0 {
            candle_core::bail!("adapter inputs must contain at least one token");
        }

        let gated_alpha = alpha.clamp(0.0, 1.0);
        let flat_x = x.reshape((batch * seq_len, d_model))?;
        let base_output = flat_x
            .matmul(&self.frozen_projection)?
            .reshape((batch, seq_len, d_model))?;

        let (ttt_output, updated_state) = self
            .chunk_fused_ttt
            .forward_chunk_fused(sliding_window_tokens, &session_states[self.layer_index])?;
        session_states[self.layer_index] = updated_state;

        let base_last = base_output.narrow(1, seq_len - 1, 1)?;
        let ttt_last = ttt_output.narrow(1, memory_seq_len - 1, 1)?;
        let alpha_tensor = Tensor::new(gated_alpha, x.device())?;
        let beta_tensor = Tensor::new(1.0f32 - gated_alpha, x.device())?;

        base_last
            .broadcast_mul(&alpha_tensor)?
            .add(&ttt_last.broadcast_mul(&beta_tensor)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_hybrid_returns_last_hidden_state() {
        let device = Device::Cpu;
        let config = AxiomConfig {
            d_model: 16,
            n_layers: 2,
            num_heads: 2,
            head_dim: 8,
            vocab_size: 32,
            lr_inner: 1e-3,
            rms_norm_eps: 1e-6,
            use_log_scan: false,
            log_scan_auto_threshold: 128,
        };
        let adapter = TTTTransformerAdapter::new(config.clone(), &device, 0).unwrap();
        let x = Tensor::zeros((1usize, 3usize, 16usize), DType::F32, &device).unwrap();
        let sliding = Tensor::zeros((1usize, 4usize, 16usize), DType::F32, &device).unwrap();
        let mut session_states = (0..config.n_layers)
            .map(|_| {
                Tensor::zeros(
                    (1usize, config.num_heads, config.head_dim, config.head_dim),
                    DType::F32,
                    &device,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let output = adapter
            .forward_hybrid(&x, &sliding, &mut session_states, 0.25)
            .unwrap();

        assert_eq!(output.dims(), &[1, 1, 16]);
        assert_eq!(session_states.len(), config.n_layers);
    }
}
