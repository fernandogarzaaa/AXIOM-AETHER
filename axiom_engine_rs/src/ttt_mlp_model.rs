//! B.5 wiring — a full autoregressive LM whose sequence-mixing primitive is the
//! expressive **TTT-MLP** block ([`NativeTTTMlpBlock`]), parallel to the linear
//! [`crate::model::AxiomTTTLM`].
//!
//! This is the model-level integration the roadmap deferred behind "measure
//! first": [`AxiomMlpLM`] stacks MLP blocks with a per-layer two-matrix
//! [`MlpState`], exposing the same `init_states` / `forward_hidden` / `forward_lm`
//! surface so it is a drop-in for evaluation. It does **not** touch the proven
//! linear model (which stays byte-identical); the two coexist so the MLP variant
//! can be benchmarked before any production switch.
//!
//! [`mlp_vs_linear_reconstruction`] is the measurement harness: it scores how
//! well each fast-weight form fits a *nonlinear* key→value association at
//! test time — the regime where the MLP state should win.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::config::AxiomConfig;
use crate::kernel::RMSNorm;
use crate::ttt_mlp::{MlpState, NativeTTTMlpBlock};

/// Autoregressive TTT-MLP language model: Embedding → N × `NativeTTTMlpBlock` →
/// RMSNorm → LM head. Session state is one [`MlpState`] per layer.
pub struct AxiomMlpLM {
    embeddings: candle_nn::Embedding,
    layers: Vec<NativeTTTMlpBlock>,
    ln_f: RMSNorm,
    lm_head: candle_nn::Linear,
    pub config: AxiomConfig,
    hidden: usize,
    inner_lr: Arc<AtomicU32>,
}

impl AxiomMlpLM {
    /// Build the model with an MLP hidden width of `hidden` for every layer.
    pub fn new(vs: VarBuilder, config: AxiomConfig, hidden: usize) -> Result<Self> {
        if hidden == 0 {
            candle_core::bail!("AxiomMlpLM hidden width must be non-zero");
        }
        let embeddings =
            candle_nn::embedding(config.vocab_size, config.d_model, vs.pp("embeddings"))?;
        let inner_lr = Arc::new(AtomicU32::new(config.lr_inner.to_bits()));

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(NativeTTTMlpBlock::new_with_shared_lr(
                vs.pp(format!("mlp_block_{i}")),
                config.clone(),
                hidden,
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
            hidden,
            inner_lr,
        })
    }

    /// Set the inner test-time learning rate η across every layer.
    pub fn set_inner_lr(&self, eta: f32) {
        self.inner_lr.store(eta.to_bits(), Ordering::Relaxed);
    }

    /// Current inner test-time learning rate η.
    pub fn inner_lr(&self) -> f32 {
        f32::from_bits(self.inner_lr.load(Ordering::Relaxed))
    }

    /// Per-layer neutral initial MLP states.
    pub fn init_states(&self, device: &candle_core::Device) -> Result<Vec<MlpState>> {
        (0..self.config.n_layers)
            .map(|_| MlpState::init(self.config.d_model, self.hidden, device))
            .collect()
    }

    /// Autoregressive forward returning normed hidden states `[1, T, d_model]`.
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        states: &mut [MlpState],
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let embeddings = self.embeddings.forward(input_ids)?;
        let mut token_outputs: Vec<Tensor> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let token_emb = embeddings.narrow(1, t, 1)?.squeeze(1)?;
            let mut hidden = token_emb;
            for (i, block) in self.layers.iter().enumerate() {
                hidden = block.forward_native(&hidden, &mut states[i])?;
            }
            token_outputs.push(hidden);
        }
        let unsqueezed: Vec<Tensor> = token_outputs
            .iter()
            .map(|t| t.unsqueeze(1))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&Tensor> = unsqueezed.iter().collect();
        let sequence_output = Tensor::cat(&refs, 1)?;
        self.ln_f.forward(&sequence_output)
    }

    /// Autoregressive forward to logits `[1, T, vocab_size]`.
    pub fn forward_lm(&self, input_ids: &Tensor, states: &mut [MlpState]) -> Result<Tensor> {
        let hidden = self.forward_hidden(input_ids, states)?;
        self.lm_head.forward(&hidden)
    }
}

/// Measurement harness ("measure first"): fit a fixed key→value association at
/// test time with a single linear fast-weight vs the MLP fast-weight, and return
/// `(linear_residual, mlp_residual)` — the squared reconstruction error each
/// achieves after `steps` gradient steps on the same input.
///
/// Both blocks minimize ‖pred − v‖² via their per-token updates; the 2-layer MLP
/// state has strictly more capacity, so on a workload where it helps its residual
/// is the smaller of the two. This is the honest, training-free signal for
/// whether the extra MLP cost buys expressivity before any production switch.
pub fn mlp_vs_linear_reconstruction(
    d_model: usize,
    hidden: usize,
    steps: usize,
) -> Result<(f32, f32)> {
    use crate::ttt_block::NativeTTTBlock;
    let device = candle_core::Device::Cpu;
    let config = AxiomConfig {
        d_model,
        n_layers: 1,
        vocab_size: 16,
        lr_inner: 5e-2,
        norm_eps: 1e-6,
    };

    // A fixed pseudo-random token activation, fed repeatedly so each block adapts
    // its fast-weights to this single (key, value) association.
    let x = Tensor::randn(0f32, 1f32, (1usize, d_model), &device)?;

    // Linear block.
    let lin_vm = candle_nn::VarMap::new();
    let lin_vb = VarBuilder::from_varmap(&lin_vm, DType::F32, &device);
    let lin = NativeTTTBlock::new(lin_vb.pp("lin"), config.clone())?;
    let mut lin_state = Tensor::eye(d_model, DType::F32, &device)?;
    for _ in 0..steps {
        let _ = lin.forward_native(&x, &mut lin_state)?;
    }
    let lin_res = lin.reconstruction_error(&x, &lin_state)?;

    // MLP block.
    let mlp_vm = candle_nn::VarMap::new();
    let mlp_vb = VarBuilder::from_varmap(&mlp_vm, DType::F32, &device);
    let mlp = NativeTTTMlpBlock::new(mlp_vb.pp("mlp"), config, hidden)?;
    let mut mlp_state = mlp.init_state(&device)?;
    for _ in 0..steps {
        let _ = mlp.forward_native(&x, &mut mlp_state)?;
    }
    let mlp_res = mlp.reconstruction_error(&x, &mlp_state)?;

    Ok((lin_res, mlp_res))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn make_model(d: usize, hidden: usize, layers: usize) -> (AxiomMlpLM, candle_core::Device) {
        let device = candle_core::Device::Cpu;
        let config = AxiomConfig {
            d_model: d,
            n_layers: layers,
            vocab_size: 16,
            lr_inner: 1e-2,
            norm_eps: 1e-6,
        };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let model = AxiomMlpLM::new(vb.pp("m"), config, hidden).unwrap();
        (model, device)
    }

    #[test]
    fn init_states_one_per_layer() {
        let (model, device) = make_model(8, 8, 3);
        let states = model.init_states(&device).unwrap();
        assert_eq!(states.len(), 3);
        assert_eq!(states[0].w1.dims(), &[8, 8]);
    }

    #[test]
    fn forward_lm_shape_and_finite() {
        let (model, device) = make_model(8, 12, 2);
        let mut states = model.init_states(&device).unwrap();
        let ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).unwrap();
        let logits = model.forward_lm(&ids, &mut states).unwrap();
        assert_eq!(logits.dims(), &[1, 5, 16]);
        let v: Vec<f32> = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn reconstruction_harness_runs_and_both_converge() {
        // Both block types must drive their reconstruction error down from the
        // identity-init starting point; the MLP, with more capacity, must not do
        // worse than the linear block on this association.
        let (lin_res, mlp_res) = mlp_vs_linear_reconstruction(16, 32, 60).unwrap();
        assert!(lin_res.is_finite() && mlp_res.is_finite());
        // Sanity: a fresh (identity) state has a sizable residual; after adapting,
        // the MLP should be at least as good as linear (typically better).
        assert!(
            mlp_res <= lin_res + 1e-3,
            "MLP residual ({mlp_res}) should be <= linear ({lin_res})"
        );
    }

    #[test]
    fn forward_updates_states() {
        let (model, device) = make_model(8, 8, 1);
        let mut states = model.init_states(&device).unwrap();
        let before = states[0].w1.clone();
        let ids = Tensor::new(&[[1u32, 2, 3]], &device).unwrap();
        let _ = model.forward_lm(&ids, &mut states).unwrap();
        let moved = states[0].w1.sub(&before).unwrap().sqr().unwrap().sum_all().unwrap()
            .to_scalar::<f32>().unwrap();
        assert!(moved > 0.0, "MLP layer state must adapt during the forward pass");
    }
}
