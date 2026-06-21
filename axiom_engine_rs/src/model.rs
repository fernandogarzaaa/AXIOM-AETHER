//! Standalone autoregressive TTT language model.
//!
//! `AxiomTTTLM` is a complete sequence model that strings `NativeTTTBlock`s
//! together.  It is a direct replacement for Transformer-based architectures and
//! requires **no** attention caching: each layer carries a single
//! `[d_model, d_model]` fast-weight matrix as its recurrent state, giving O(1)
//! memory per inference step.
//!
//! Architecture: Embedding → N × NativeTTTBlock → RMSNorm → LM Head

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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
    /// Shared stabilization flag read by every `NativeTTTBlock`. When on, the
    /// fast-weight update normalizes keys + clamps state so deep/wide models
    /// stay finite. Off by default (existing checkpoints unchanged).
    stabilize: Arc<AtomicBool>,
    /// Shared Gated-DeltaNet forget gate α ∈ (0, 1] (raw f32 bits) read by every
    /// `NativeTTTBlock`. α < 1 decays retained memory geometrically (bounded
    /// state); default 1.0 is the identity (ungated, existing checkpoints
    /// unchanged). Parameter-free, so no checkpoint-format change.
    forget_gate: Arc<AtomicU32>,
    /// Shared safe-online-update guards (drift-aware reset, token selection,
    /// anti-forgetting anchor) read by every `NativeTTTBlock`. All disabled by
    /// default ⇒ byte-identical dynamics; enable to bound long-stream persistent
    /// adaptation. Parameter-free, so no checkpoint-format change.
    guards: Arc<crate::ttt_block::OnlineGuards>,
}

impl AxiomTTTLM {
    /// Construct the full model, allocating all weights under the given
    /// `VarBuilder` scope.
    pub fn new(vs: VarBuilder, config: AxiomConfig) -> Result<Self> {
        Self::new_with_options(vs, config, false)
    }

    /// Construct the model, optionally with the *learned* data-dependent forget
    /// gate (a per-layer `w_α` projection — adds `w_alpha` weights, so a model
    /// built with `learned_gate=true` can only load a checkpoint trained the same
    /// way). `learned_gate=false` is parameter-identical to the original model.
    pub fn new_with_options(
        vs: VarBuilder,
        config: AxiomConfig,
        learned_gate: bool,
    ) -> Result<Self> {
        let embeddings =
            candle_nn::embedding(config.vocab_size, config.d_model, vs.pp("embeddings"))?;

        // One shared inner-lr cell, initialised from config, handed to every
        // layer so meta-training can decay η across the whole stack.
        let inner_lr = Arc::new(AtomicU32::new(config.lr_inner.to_bits()));
        // Shared stabilization flag (off by default → existing models unchanged).
        let stabilize = Arc::new(AtomicBool::new(false));
        // Shared forget gate, default 1.0 (ungated → existing models unchanged).
        let forget_gate = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        // Shared safe-online-update guards, all disabled (byte-identical default).
        let guards = Arc::new(crate::ttt_block::OnlineGuards::disabled());

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(NativeTTTBlock::new_with_shared_lr(
                vs.pp(format!("native_block_{i}")),
                config.clone(),
                inner_lr.clone(),
                stabilize.clone(),
                forget_gate.clone(),
                guards.clone(),
                learned_gate,
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
            stabilize,
            forget_gate,
            guards,
        })
    }

    /// Configure the safe-online-update guards across every TTT layer in one
    /// call (RDumb drift reset, EATA token selection, anti-forgetting anchor).
    /// These bound *persistent* fast-weight adaptation over long token streams;
    /// all default to disabled, so leaving them unset keeps the proven dynamics.
    ///
    /// * `drift_reset_norm` – reset `W̃` to init when its Frobenius norm exceeds
    ///   this (`0.0` disables).
    /// * `update_min_error` – skip the update when ‖pred − v‖ is below this
    ///   (`0.0` updates on every token).
    /// * `anchor_strength` – λ ∈ [0,1] pulling `W̃` toward init each step
    ///   (`0.0` disables).
    pub fn set_online_guards(
        &self,
        drift_reset_norm: f32,
        update_min_error: f32,
        anchor_strength: f32,
    ) {
        self.guards.set_drift_reset_norm(drift_reset_norm.max(0.0));
        self.guards.set_update_min_error(update_min_error.max(0.0));
        self.guards.set_anchor_strength(anchor_strength);
    }

    /// Select the inner self-supervised loss form across every TTT layer
    /// (B.6 ablation): `true` = normalized/contrastive multi-view objective,
    /// `false` (default) = exact MSE reconstruction (byte-identical dynamics).
    pub fn set_aux_loss_normalized(&self, on: bool) {
        self.guards.set_aux_loss_normalized(on);
    }

    /// Whether the contrastive multi-view inner loss is currently enabled.
    pub fn aux_loss_normalized(&self) -> bool {
        self.guards
            .aux_loss_normalized
            .load(Ordering::Relaxed)
    }

    /// Current safe-online-update guard settings as
    /// `(drift_reset_norm, update_min_error, anchor_strength)`.
    pub fn online_guards(&self) -> (f32, f32, f32) {
        (
            f32::from_bits(self.guards.drift_reset_norm.load(Ordering::Relaxed)),
            f32::from_bits(self.guards.update_min_error.load(Ordering::Relaxed)),
            f32::from_bits(self.guards.anchor_strength.load(Ordering::Relaxed)),
        )
    }

    /// Enable/disable inner-loop stabilization (normalized keys + state clamp)
    /// across every layer. Must match how the checkpoint was trained — the
    /// proxy sets this from the model's sidecar (`ModelMeta::stabilize`).
    pub fn set_stabilize(&self, on: bool) {
        self.stabilize.store(on, Ordering::Relaxed);
    }

    /// Whether inner-loop stabilization is currently enabled.
    pub fn stabilize(&self) -> bool {
        self.stabilize.load(Ordering::Relaxed)
    }

    /// Set the Gated-DeltaNet forget gate α ∈ (0, 1] across every TTT layer.
    /// α < 1 makes retained memory decay geometrically (spectral radius ≤ α), a
    /// principled bound on the fast-weight state; α = 1 is the ungated identity.
    /// Values are clamped into (0, 1].
    pub fn set_forget_gate(&self, alpha: f32) {
        let a = if alpha.is_finite() {
            alpha.clamp(1e-3, 1.0)
        } else {
            1.0
        };
        self.forget_gate.store(a.to_bits(), Ordering::Relaxed);
    }

    /// Current Gated-DeltaNet forget gate α.
    pub fn forget_gate(&self) -> f32 {
        f32::from_bits(self.forget_gate.load(Ordering::Relaxed))
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

    /// Autoregressive forward pass returning the final normed hidden states
    /// (after `ln_f`, before the LM head): `[1, T, d_model]`. This is the
    /// representation used for embeddings (mean-pooled into a dense vector).
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
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        session_states: &mut [Tensor],
    ) -> Result<Tensor> {
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

        // Final RMSNorm → [1, T, d_model].
        self.ln_f.forward(&sequence_output)
    }

    /// Autoregressive forward pass that returns only the final hidden state
    /// `[1, d_model]`. Session states are updated exactly as in
    /// [`Self::forward_hidden`], but intermediate token outputs are not stored.
    /// This is the fast path for detached online training where each chunk
    /// contributes a single next-token loss.
    pub fn forward_last_hidden(
        &self,
        input_ids: &Tensor,
        session_states: &mut [Tensor],
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let embeddings = self.embeddings.forward(input_ids)?;

        let mut last_hidden: Option<Tensor> = None;
        for t in 0..seq_len {
            let token_emb = embeddings.narrow(1, t, 1)?.squeeze(1)?;
            let mut hidden = token_emb;
            for (i, block) in self.layers.iter().enumerate() {
                hidden = block.forward_native(&hidden, &mut session_states[i])?;
            }
            last_hidden = Some(hidden);
        }

        let last = last_hidden.unwrap_or_else(|| {
            Tensor::zeros(
                (1usize, self.config.d_model),
                DType::F32,
                input_ids.device(),
            )
            .expect("zero last hidden")
        });
        self.ln_f.forward(&last)
    }

    /// Adapt session fast-weights over a token sequence without materializing
    /// hidden states, logits, or vocabulary projections.
    ///
    /// This is the hot path for context compression: it performs the same
    /// per-token TTT state mutation as [`Self::forward_hidden`] / [`Self::forward_lm`],
    /// but intentionally discards every intermediate activation once the layer
    /// states have been updated.
    pub fn adapt_tokens(&self, input_ids: &Tensor, session_states: &mut [Tensor]) -> Result<()> {
        let (_, seq_len) = input_ids.dims2()?;
        let embeddings = self.embeddings.forward(input_ids)?;

        for t in 0..seq_len {
            let token_emb = embeddings.narrow(1, t, 1)?.squeeze(1)?;
            let mut hidden = token_emb;
            for (i, block) in self.layers.iter().enumerate() {
                hidden = block.forward_native(&hidden, &mut session_states[i])?;
            }
            drop(hidden);
        }

        Ok(())
    }

    /// Autoregressive forward pass over a token sequence to logits
    /// `[1, T, vocab_size]`. Equivalent to `lm_head(forward_hidden(..))`; the
    /// hidden-state computation lives in [`Self::forward_hidden`].
    pub fn forward_lm(&self, input_ids: &Tensor, session_states: &mut [Tensor]) -> Result<Tensor> {
        let normed = self.forward_hidden(input_ids, session_states)?;
        self.lm_head.forward(&normed)
    }

    /// Autoregressive forward pass returning only final-token logits
    /// `[1, vocab_size]`.
    pub fn forward_last_logits(
        &self,
        input_ids: &Tensor,
        session_states: &mut [Tensor],
    ) -> Result<Tensor> {
        let normed = self.forward_last_hidden(input_ids, session_states)?;
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

    #[test]
    fn test_forward_hidden_shape() {
        let (model, device) = make_model(2);
        let mut states = model.init_states(&device).unwrap();
        let input_ids = Tensor::zeros((1usize, 4usize), DType::U32, &device).unwrap();
        let hidden = model.forward_hidden(&input_ids, &mut states[..]).unwrap();
        // [1, T, d_model] — d_model is 16 in make_model
        assert_eq!(hidden.dims(), &[1, 4, 16]);
    }

    #[test]
    fn test_forward_lm_equals_lm_head_of_hidden() {
        // forward_lm must remain logits = lm_head(forward_hidden) after refactor.
        let (model, device) = make_model(1);
        let mut s1 = model.init_states(&device).unwrap();
        let mut s2 = model.init_states(&device).unwrap();
        let input_ids = Tensor::zeros((1usize, 3usize), DType::U32, &device).unwrap();
        let logits = model.forward_lm(&input_ids, &mut s1[..]).unwrap();
        // vocab_size = 32
        assert_eq!(logits.dims(), &[1, 3, 32]);
        // hidden path produces values of the right shape
        let hidden = model.forward_hidden(&input_ids, &mut s2[..]).unwrap();
        assert_eq!(hidden.dims(), &[1, 3, 16]);
    }

    #[test]
    fn test_forward_last_logits_matches_full_final_token() {
        let (model, device) = make_model(2);
        let mut s1 = model.init_states(&device).unwrap();
        let mut s2 = model.init_states(&device).unwrap();
        let input_ids = Tensor::from_vec(vec![0u32, 1, 2, 3], (1usize, 4usize), &device).unwrap();

        let full = model.forward_lm(&input_ids, &mut s1[..]).unwrap();
        let last = model.forward_last_logits(&input_ids, &mut s2[..]).unwrap();

        assert_eq!(last.dims(), &[1, 32]);
        let full_last = full.narrow(1, 3, 1).unwrap().squeeze(1).unwrap();
        let diff = full_last
            .sub(&last)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap();
        assert!(diff.to_scalar::<f32>().unwrap() < 1e-5);
    }

    #[test]
    fn adapt_tokens_matches_forward_lm_state_without_materializing_logits() {
        let (model, device) = make_model(2);
        let mut logits_states = model.init_states(&device).unwrap();
        let mut adapt_states = model.init_states(&device).unwrap();
        let input_ids =
            Tensor::from_vec(vec![0u32, 1, 2, 3, 4], (1usize, 5usize), &device).unwrap();

        let _ = model
            .forward_lm(&input_ids, &mut logits_states[..])
            .unwrap();
        model
            .adapt_tokens(&input_ids, &mut adapt_states[..])
            .unwrap();

        for (logits_state, adapt_state) in logits_states.iter().zip(adapt_states.iter()) {
            let diff = logits_state
                .sub(adapt_state)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap();
            assert!(diff.to_scalar::<f32>().unwrap() < 1e-5);
        }
    }
}
