//! Native causal TTT block — standalone replacement for Multi-Head Attention.
//!
//! `NativeTTTBlock` maintains a single `[d_model, d_model]` fast-weight matrix
//! (W_tilde) as its recurrent session state.  For every incoming token the block
//! performs one self-supervised gradient step on W_tilde before producing output,
//! achieving O(1) memory per inference step.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use candle_core::{Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

use crate::config::AxiomConfig;

/// Element-magnitude backstop applied to the fast-weight matrix when
/// stabilization is enabled. Generous enough not to distort healthy dynamics
/// (states stay O(1) with normalized keys), tight enough that a runaway can
/// never reach `f32` overflow / NaN. Sync-free (no device round-trip).
const STAB_CLAMP: f32 = 10.0;

/// Standalone causal TTT block.
///
/// Replaces Multi-Head Attention as the core sequence-mixing primitive.
/// Projection weights `W_q`, `W_k`, `W_v` map hidden representations to
/// query/key/value spaces.  An embedded `RMSNorm` is applied to every output.
pub struct NativeTTTBlock {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    layer_norm: LayerNorm,
    /// Inner test-time learning rate η, stored as raw f32 bits so it can be
    /// adjusted at runtime (e.g. cosine decay during meta-training) and
    /// shared across all layers without rebuilding the model. Read on every
    /// `forward_native` step.
    inner_lr: Arc<AtomicU32>,
    /// When true, the fast-weight update L2-normalizes the key (bounding the
    /// per-step update to ~η regardless of weight scale) and element-clamps the
    /// state. This keeps deep/wide models (d384, d512+) from diverging to NaN.
    /// Off by default so the converged d256 path is byte-identical. Shared like
    /// `inner_lr` so one `set_stabilize` retunes the whole stack.
    stabilize: Arc<AtomicBool>,
}

impl NativeTTTBlock {
    /// Construct a new block with its own private inner-lr cell initialised
    /// from `config.lr_inner`.
    #[allow(dead_code)] // used by the lib path + tests; the bin builds via the model
    pub fn new(vs: VarBuilder, config: AxiomConfig) -> Result<Self> {
        let inner_lr = Arc::new(AtomicU32::new(config.lr_inner.to_bits()));
        let stabilize = Arc::new(AtomicBool::new(false));
        Self::new_with_shared_lr(vs, config, inner_lr, stabilize)
    }

    /// Construct a block that reads its inner learning rate from a shared
    /// atomic cell — used by `AxiomTTTLM` so a single `set_inner_lr` call
    /// retunes every layer at once.
    pub fn new_with_shared_lr(
        vs: VarBuilder,
        config: AxiomConfig,
        inner_lr: Arc<AtomicU32>,
        stabilize: Arc<AtomicBool>,
    ) -> Result<Self> {
        let d = config.d_model;
        Ok(Self {
            w_q: candle_nn::linear_no_bias(d, d, vs.pp("w_q"))?,
            w_k: candle_nn::linear_no_bias(d, d, vs.pp("w_k"))?,
            w_v: candle_nn::linear_no_bias(d, d, vs.pp("w_v"))?,
            layer_norm: candle_nn::layer_norm_no_bias(
                d,
                config.norm_eps as f64,
                vs.pp("layer_norm"),
            )?,
            inner_lr,
            stabilize,
        })
    }

    /// Autoregressive forward step for a single token.
    ///
    /// # Arguments
    /// * `x`             – `[1, d_model]` token activation.
    /// * `session_state` – `[d_model, d_model]` fast-weight matrix W_tilde,
    ///   updated in-place via one gradient descent step.
    ///
    /// # Returns
    /// `[1, d_model]` output after the TTT update and embedded layer normalisation.
    ///
    /// ## TTT update rule (MSE loss on key→value reconstruction)
    ///
    /// ```text
    /// q, k, v  = W_q(x),  W_k(x),  W_v(x)          [1, d_model] each
    /// pred     = W_tilde × k^T                       [d_model]
    /// error    = pred − v                            [d_model]
    /// grad     = error ⊗ k    (outer product)        [d_model, d_model]
    /// W_tilde  ← W_tilde − η · grad
    /// output   = q × W_tilde                         [1, d_model]
    /// ```
    pub fn forward_native(&self, x: &Tensor, session_state: &mut Tensor) -> Result<Tensor> {
        // Project input to query, key, value: each [1, d_model].
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        // --- Fast-weight gradient step ------------------------------------------
        let stabilize = self.stabilize.load(Ordering::Relaxed);

        // Effective key. When stabilization is on, L2-normalize it: this bounds
        // the per-step growth of ‖W_tilde‖ to ~(1+η) instead of (1+η·‖k‖²), which
        // is what makes d384/d512 diverge as the learned W_k weights grow.
        let k_eff = if stabilize {
            let norm = k.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?; // [1, 1]
            let norm = norm.affine(1.0, 1e-6)?; // + eps (avoid /0)
            k.broadcast_div(&norm)?
        } else {
            k.clone()
        };

        // k_col: [d_model, 1]  (transpose of the [1, d_model] key)
        let k_col = k_eff.t()?.contiguous()?;

        // pred: [d_model, d_model] × [d_model, 1] → [d_model, 1] → squeeze → [d_model]
        let pred = session_state.matmul(&k_col)?.squeeze(D::Minus1)?;

        // v_vec: [d_model]  (remove the leading batch-of-one dimension)
        let v_vec = v.squeeze(0)?;

        // error: [d_model]
        let error = pred.sub(&v_vec)?;

        // Outer product: [d_model, 1] × [1, d_model] → [d_model, d_model]
        let grad = error.unsqueeze(1)?.matmul(&k_eff)?;

        // W_tilde update: W_tilde ← W_tilde − η · grad
        // η is read live from the shared atomic so meta-training can decay it.
        let eta = f32::from_bits(self.inner_lr.load(Ordering::Relaxed));
        let lr = Tensor::new(eta, session_state.device())?;
        let mut updated_state = session_state.sub(&grad.broadcast_mul(&lr)?)?;
        if stabilize {
            // Sync-free element backstop: a hard ceiling that NaN can never breach.
            updated_state = updated_state.clamp(-STAB_CLAMP, STAB_CLAMP)?;
        }
        // The session state is an inference cache, not a BPTT tape. Detaching it
        // here prevents long prompts from retaining one Candle op node per token.
        *session_state = updated_state.detach();
        // ------------------------------------------------------------------------

        // output = q × W_tilde : [1, d_model] × [d_model, d_model] → [1, d_model]
        let output = q.matmul(&updated_state)?;

        // Embedded LayerNorm.
        self.layer_norm.forward(&output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn make_block(d_model: usize) -> (NativeTTTBlock, Device) {
        let device = Device::Cpu;
        let config = AxiomConfig {
            d_model,
            n_layers: 1,
            vocab_size: 16,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = NativeTTTBlock::new(vb.pp("block"), config).unwrap();
        (block, device)
    }

    #[test]
    fn test_forward_native_output_shape() {
        let d = 8usize;
        let (block, device) = make_block(d);
        let x = Tensor::zeros((1usize, d), DType::F32, &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        let output = block.forward_native(&x, &mut state).unwrap();
        assert_eq!(output.dims(), &[1, d]);
    }

    #[test]
    fn test_session_state_is_updated() {
        let d = 8usize;
        let (block, device) = make_block(d);
        let x = Tensor::ones((1usize, d), DType::F32, &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        let state_before: Vec<f32> = state.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let _ = block.forward_native(&x, &mut state).unwrap();
        let state_after: Vec<f32> = state.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_ne!(
            state_before, state_after,
            "session state must be updated after forward_native"
        );
    }

    #[test]
    fn test_forward_native_output_is_finite() {
        let d = 8usize;
        let (block, device) = make_block(d);
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        let output = block.forward_native(&x, &mut state).unwrap();
        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_stabilized_path_survives_blowup() {
        // Large-magnitude input + a long window: the raw (unnormalized) update
        // explodes here; the stabilized path must stay finite and bounded.
        let d = 32usize;
        let (block, device) = make_block(d);
        block.stabilize.store(true, Ordering::Relaxed);
        let x = Tensor::randn(0f32, 10f32, (1usize, d), &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        for _ in 0..512 {
            let out = block.forward_native(&x, &mut state).unwrap();
            let ov: Vec<f32> = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(
                ov.iter().all(|v| v.is_finite()),
                "stabilized output went non-finite"
            );
        }
        // The element clamp must hold the state inside [-STAB_CLAMP, STAB_CLAMP].
        let sv: Vec<f32> = state.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            sv.iter().all(|v| v.abs() <= STAB_CLAMP + 1e-3),
            "state exceeded the stabilization clamp"
        );
    }
}
