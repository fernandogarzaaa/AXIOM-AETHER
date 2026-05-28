//! Standalone autoregressive TTT language model.
//!
//! `AxiomTTTLM` is a complete sequence model that strings `NativeTTTBlock`s
//! together.  It is a direct replacement for Transformer-based architectures and
//! requires **no** attention caching: each layer carries a single
//! `[d_model, d_model]` fast-weight matrix as its recurrent state, giving O(1)
//! memory per inference step.
//!
//! Architecture: Embedding → N × NativeTTTBlock → RMSNorm → LM Head

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::config::AxiomConfig;
use crate::kernel::RMSNorm;
use crate::ttt_block::NativeTTTBlock;

/// Full autoregressive TTT language model.
///
/// Session state layout: one `[d_model, d_model]` identity-initialised
/// fast-weight tensor per layer.
pub struct AxiomTTTLM {
    embeddings: candle_nn::Embedding,
    layers: Vec<NativeTTTBlock>,
    ln_f: RMSNorm,
    lm_head: candle_nn::Linear,
    pub config: AxiomConfig,
    /// Shared inner test-time learning rate η (raw f32 bits) read by every
    /// `NativeTTTBlock`. Adjusting it retunes the whole stack at once.
    inner_lr: Arc<AtomicU32>,
}

impl AxiomTTTLM {
    /// Construct the full model, allocating all weights under the given
    /// `VarBuilder` scope.
    pub fn new(vs: VarBuilder, config: AxiomConfig) -> Result<Self> {
        let embeddings =
            candle_nn::embedding(config.vocab_size, config.d_model, vs.pp("embeddings"))?;

        // One shared inner-lr cell, initialised from config, handed to every
        // layer so meta-training can decay η across the whole stack.
        let inner_lr = Arc::new(AtomicU32::new(config.lr_inner.to_bits()));

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(NativeTTTBlock::new_with_shared_lr(
                vs.pp(format!("native_block_{i}")),
                config.clone(),
                inner_lr.clone(),
            )?);
        }

        let ln_f = RMSNorm::new(config.d_model, config.norm_eps, vs.pp("ln_f"))?;
        let lm_head =
            candle_nn::linear_no_bias(config.d_model, config.vocab_size, vs.pp("lm_head"))?;

        Ok(Self {
            embeddings,
            layers,
            ln_f,
            lm_head,
            config,
            inner_lr,
        })
    }

    /// Set the inner test-time learning rate η used by every TTT layer on
    /// subsequent `forward_native` / `forward_lm` calls.
    pub fn set_inner_lr(&self, eta: f32) {
        self.inner_lr.store(eta.to_bits(), Ordering::Relaxed);
    }

    /// Current inner test-time learning rate η.
    pub fn inner_lr(&self) -> f32 {
        f32::from_bits(self.inner_lr.load(Ordering::Relaxed))
    }

    /// Allocate per-layer identity fast-weight matrices as initial session states.
    ///
    /// Each state is a `[d_model, d_model]` identity matrix: the neutral starting
    /// point before any test-time training has occurred.
    pub fn init_states(&self, device: &candle_core::Device) -> Result<Vec<Tensor>> {
        (0..self.config.n_layers)
            .map(|_| Tensor::eye(self.config.d_model, DType::F32, device))
            .collect()
    }

    /// Autoregressive forward pass over a token sequence.
    ///
    /// Tokens are processed one at a time.  For each token every `NativeTTTBlock`
    /// performs a self-supervised fast-weight update and produces its hidden
    /// representation.  This preserves strict causality without any attention
    /// caching overhead.
    ///
    /// # Arguments
    /// * `input_ids`      – `[1, T]` integer token indices.
    /// * `session_states` – Per-layer `[d_model, d_model]` fast-weight tensors;
    ///   updated in-place after every token.
    ///
    /// # Returns
    /// Logits `[1, T, vocab_size]`.
    pub fn forward_lm(&self, input_ids: &Tensor, session_states: &mut [Tensor]) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;

        // Embed all tokens: [1, T, d_model].
        let embeddings = self.embeddings.forward(input_ids)?;

        let mut token_outputs: Vec<Tensor> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Single-token activation: [1, d_model] (batch=1, no sequence dim).
            let token_emb = embeddings.narrow(1, t, 1)?.squeeze(1)?;

            // Pass through all blocks sequentially, updating each layer's state.
            let mut hidden = token_emb;
            for (i, block) in self.layers.iter().enumerate() {
                hidden = block.forward_native(&hidden, &mut session_states[i])?;
            }

            token_outputs.push(hidden);
        }

        // Reassemble sequence: each token_output is [1, d_model].
        // Unsqueeze to [1, 1, d_model] then concatenate along dim 1 → [1, T, d_model].
        let unsqueezed: Vec<Tensor> = token_outputs
            .iter()
            .map(|t| t.unsqueeze(1))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&Tensor> = unsqueezed.iter().collect();
        let sequence_output = Tensor::cat(&refs, 1)?;

        // Final RMSNorm + LM head → [1, T, vocab_size].
        let normed = self.ln_f.forward(&sequence_output)?;
        self.lm_head.forward(&normed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, D};
    use candle_nn::{VarBuilder, VarMap};

    fn make_model(n_layers: usize) -> (AxiomTTTLM, Device) {
        let device = Device::Cpu;
        let config = AxiomConfig {
            d_model: 16,
            n_layers,
            vocab_size: 32,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = AxiomTTTLM::new(vb, config).unwrap();
        (model, device)
    }

    #[test]
    fn test_init_states_shape() {
        let (model, device) = make_model(2);
        let states = model.init_states(&device).unwrap();
        assert_eq!(states.len(), 2);
        for state in &states {
            assert_eq!(state.dims(), &[16, 16]);
        }
    }

    #[test]
    fn test_forward_lm_single_token_shape() {
        let (model, device) = make_model(2);
        let mut states = model.init_states(&device).unwrap();
        let input_ids = Tensor::zeros((1usize, 1usize), DType::U32, &device).unwrap();
        let logits = model.forward_lm(&input_ids, &mut states[..]).unwrap();
        assert_eq!(logits.dims(), &[1, 1, 32]);
    }

    #[test]
    fn test_forward_lm_sequence_shape() {
        let (model, device) = make_model(1);
        let mut states = model.init_states(&device).unwrap();
        let input_ids = Tensor::zeros((1usize, 5usize), DType::U32, &device).unwrap();
        let logits = model.forward_lm(&input_ids, &mut states[..]).unwrap();
        assert_eq!(logits.dims(), &[1, 5, 32]);
    }

    #[test]
    fn test_forward_lm_updates_states() {
        let (model, device) = make_model(1);
        let mut states = model.init_states(&device).unwrap();
        let eye_data: Vec<f32> = states[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let input_ids = Tensor::ones((1usize, 1usize), DType::U32, &device).unwrap();
        let _ = model.forward_lm(&input_ids, &mut states[..]).unwrap();
        let updated_data: Vec<f32> = states[0].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_ne!(
            eye_data, updated_data,
            "session states must change after forward"
        );
    }

    #[test]
    fn test_logits_are_finite() {
        let (model, device) = make_model(2);
        let mut states = model.init_states(&device).unwrap();
        let input_ids = Tensor::zeros((1usize, 3usize), DType::U32, &device).unwrap();
        let logits = model.forward_lm(&input_ids, &mut states[..]).unwrap();
        let values: Vec<f32> = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_argmax_decode() {
        let (model, device) = make_model(1);
        let mut states = model.init_states(&device).unwrap();
        let input_ids = Tensor::zeros((1usize, 1usize), DType::U32, &device).unwrap();
        let logits = model.forward_lm(&input_ids, &mut states[..]).unwrap();
        // logits: [1, 1, vocab_size] → squeeze(1) → [1, vocab_size] → argmax → [1]
        let next_id = logits
            .squeeze(1)
            .unwrap()
            .argmax(D::Minus1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        assert!(next_id < 32, "argmax token id must be in vocabulary range");
    }
}
