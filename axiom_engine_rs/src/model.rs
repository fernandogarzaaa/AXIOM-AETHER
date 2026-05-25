//! Standalone autoregressive TTT language model backbone.
//!
//! `AxiomTTTLM` is a fully autonomous sequence model that acts as a **direct
//! replacement** for the Transformer architecture.  It contains no attention
//! layers, no KV-cache, and no adapter wrappers — only an embedding table,
//! a stack of [`NativeTTTBlock`] layers, a final RMSNorm, and an LM head.
//!
//! Autoregressive inference runs in true O(1) time per token: each call to
//! `forward_lm` advances the per-layer fast-weight `W̃` states for one token
//! and returns logits for that position.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::config::AxiomConfig;
use crate::kernel::RMSNorm;
use crate::ttt_block::NativeTTTBlock;

/// Full autoregressive TTT language model.
///
/// Architecture:
/// ```text
/// input_ids [B, T]
///     ↓  Embedding
/// hidden [B, T, d_model]
///     ↓  × n_layers of NativeTTTBlock (sequential, updating W̃ per token)
/// hidden [B, T, d_model]
///     ↓  RMSNorm
///     ↓  LM Head (linear, no bias)
/// logits [B, T, vocab_size]
/// ```
///
/// Session state: `Vec<Tensor>` of length `n_layers`, each `[B, H, D, D]`.
/// Initialise with [`AxiomTTTLM::init_native_states`] for identity matrices,
/// which gives the fast-weight kernel a stable, pass-through starting point.
pub struct AxiomTTTLM {
    embeddings: Embedding,
    layers: Vec<NativeTTTBlock>,
    ln_f: RMSNorm,
    lm_head: Linear,
    config: AxiomConfig,
}

impl AxiomTTTLM {
    /// Construct the model and register all parameters with `vs`.
    pub fn new(vs: VarBuilder, config: AxiomConfig) -> Result<Self> {
        let embeddings =
            candle_nn::embedding(config.vocab_size, config.d_model, vs.pp("embeddings"))?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(NativeTTTBlock::new(
                vs.pp(format!("layer_{i}")),
                config.clone(),
            )?);
        }

        let ln_f = RMSNorm::new(config.d_model, config.rms_norm_eps, vs.pp("ln_f"))?;
        let lm_head =
            candle_nn::linear_no_bias(config.d_model, config.vocab_size, vs.pp("lm_head"))?;

        Ok(Self {
            embeddings,
            layers,
            ln_f,
            lm_head,
            config,
        })
    }

    /// Autoregressive forward pass with O(1)-per-token inference.
    ///
    /// Processes `input_ids` token-by-token, updating each layer's fast-weight
    /// state `W̃` inline for every position.  No KV cache or attention overhead
    /// is incurred — the entire context is compressed into the `D²·H`-dimensional
    /// session state.
    ///
    /// # Arguments
    /// * `input_ids`      – Integer token indices, shape `[B, T]`.
    /// * `session_states` – Per-layer `W̃` tensors, each `[B, H, D, D]`.
    ///                      Each tensor is updated in-place as tokens are consumed.
    ///
    /// # Returns
    /// Logit tensor, shape `[B, T, vocab_size]`.
    pub fn forward_lm(
        &self,
        input_ids: &Tensor,
        session_states: &mut Vec<Tensor>,
    ) -> Result<Tensor> {
        let (_, t) = input_ids.dims2()?;

        // Embed the full token sequence at once: [B, T, d_model].
        let x_all = self.embeddings.forward(input_ids)?;

        let mut token_outputs: Vec<Tensor> = Vec::with_capacity(t);

        for ti in 0..t {
            // Extract single-position slice → [B, 1, d_model].
            let x_t = x_all.narrow(1, ti, 1)?;
            let mut h = x_t;

            // Sequentially route through each TTT block, carrying the
            // per-layer fast-weight state forward.  Causality is preserved
            // because each block's W̃ is updated by the current token before
            // the next block processes it.
            for (i, layer) in self.layers.iter().enumerate() {
                h = layer.forward_native(&h, &mut session_states[i])?;
            }

            token_outputs.push(h);
        }

        // Reassemble the per-token outputs into [B, T, d_model].
        let out = if token_outputs.len() == 1 {
            token_outputs.remove(0)
        } else {
            let refs: Vec<&Tensor> = token_outputs.iter().collect();
            Tensor::cat(&refs, 1)?
        };

        // Final norm and LM head projection → [B, T, vocab_size].
        let normed = self.ln_f.forward(&out)?;
        self.lm_head.forward(&normed)
    }

    /// Allocate per-layer identity-matrix `W̃` states.
    ///
    /// Identity initialisation ensures `f_{W̃}(k) ≈ k` at the start of every
    /// session, providing a stable gradient signal from the very first token
    /// without any warm-up required.
    ///
    /// # Returns
    /// `n_layers` tensors, each `[batch, num_heads, head_dim, head_dim]`,
    /// with every `[head_dim × head_dim]` sub-matrix equal to `I_D`.
    pub fn init_native_states(&self, batch: usize, device: &Device) -> Result<Vec<Tensor>> {
        let h = self.config.num_heads;
        let d = self.config.head_dim;

        (0..self.config.n_layers)
            .map(|_| {
                // Build batch * H identity matrices of shape [1, D, D] each,
                // concatenate along dim 0 → [batch*H, D, D], then reshape.
                let mut slices: Vec<Tensor> = Vec::with_capacity(batch * h);
                for _ in 0..(batch * h) {
                    slices.push(Tensor::eye(d, DType::F32, device)?.unsqueeze(0)?);
                }
                let slice_refs: Vec<&Tensor> = slices.iter().collect();
                Tensor::cat(&slice_refs, 0)?.reshape((batch, h, d, d))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    fn make_model() -> (AxiomTTTLM, Device) {
        let config = AxiomConfig {
            d_model: 16,
            n_layers: 2,
            num_heads: 2,
            head_dim: 8,
            vocab_size: 64,
            lr_inner: 1e-3,
            rms_norm_eps: 1e-6,
            use_log_scan: false,
            log_scan_auto_threshold: 128,
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = AxiomTTTLM::new(vb, config).unwrap();
        (model, device)
    }

    #[test]
    fn forward_lm_single_token_shape() {
        let (model, device) = make_model();
        let mut states = model.init_native_states(1, &device).unwrap();
        let input = Tensor::zeros((1usize, 1usize), DType::U32, &device).unwrap();
        let logits = model.forward_lm(&input, &mut states).unwrap();
        assert_eq!(logits.dims(), &[1, 1, 64]);
    }

    #[test]
    fn forward_lm_multi_token_sequence() {
        let (model, device) = make_model();
        let mut states = model.init_native_states(1, &device).unwrap();
        let input = Tensor::zeros((1usize, 4usize), DType::U32, &device).unwrap();
        let logits = model.forward_lm(&input, &mut states).unwrap();
        assert_eq!(logits.dims(), &[1, 4, 64]);
    }

    #[test]
    fn init_native_states_count_and_shape() {
        let (model, device) = make_model();
        let states = model.init_native_states(1, &device).unwrap();
        // n_layers = 2
        assert_eq!(states.len(), 2);
        for state in &states {
            // [batch=1, H=2, D=8, D=8]
            assert_eq!(state.dims(), &[1, 2, 8, 8]);
        }
    }

    #[test]
    fn init_native_states_are_identity() {
        let (model, device) = make_model();
        let states = model.init_native_states(1, &device).unwrap();
        let eye = Tensor::eye(8usize, DType::F32, &device).unwrap();
        for state in &states {
            // Flatten to [H, D, D] then check each head sub-matrix equals I_8.
            // Shape: [1, H=2, D=8, D=8] → compare head_0 via narrow operations.
            let head_0 = state
                .narrow(0, 0, 1)
                .unwrap()
                .narrow(1, 0, 1)
                .unwrap()
                .reshape((8usize, 8usize))
                .unwrap();
            let diff = head_0.sub(&eye).unwrap();
            let diff_norm = diff
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff_norm < 1e-6, "initial state must be identity matrix");
        }
    }

    #[test]
    fn states_updated_after_forward_lm() {
        let (model, device) = make_model();
        let mut states = model.init_native_states(1, &device).unwrap();
        let before: Vec<Tensor> = states.iter().map(|t| t.clone()).collect();
        // Use token id 1 so the input is non-zero after embedding.
        let input = Tensor::from_vec(vec![1u32], (1usize, 1usize), &device).unwrap();
        let _ = model.forward_lm(&input, &mut states).unwrap();
        // At least the first layer's state should differ from identity.
        let diff = states[0].sub(&before[0]).unwrap();
        let diff_norm = diff
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff_norm > 0.0,
            "session states must be updated by forward_lm"
        );
    }
}
