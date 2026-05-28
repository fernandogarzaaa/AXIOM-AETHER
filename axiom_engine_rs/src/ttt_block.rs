//! Native causal TTT block — standalone replacement for Multi-Head Attention.
//!
//! `NativeTTTBlock` maintains a single `[d_model, d_model]` fast-weight matrix
//! (W_tilde) as its recurrent session state.  For every incoming token the block
//! performs one self-supervised gradient step on W_tilde before producing output,
//! achieving O(1) memory per inference step.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use candle_core::{Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

use crate::config::AxiomConfig;

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
}

impl NativeTTTBlock {
    /// Construct a new block with its own private inner-lr cell initialised
    /// from `config.lr_inner`.
    #[allow(dead_code)] // used by the lib path + tests; the bin builds via the model
    pub fn new(vs: VarBuilder, config: AxiomConfig) -> Result<Self> {
        let inner_lr = Arc::new(AtomicU32::new(config.lr_inner.to_bits()));
        Self::new_with_shared_lr(vs, config, inner_lr)
    }

    /// Construct a block that reads its inner learning rate from a shared
    /// atomic cell — used by `AxiomTTTLM` so a single `set_inner_lr` call
    /// retunes every layer at once.
    pub fn new_with_shared_lr(
        vs: VarBuilder,
        config: AxiomConfig,
        inner_lr: Arc<AtomicU32>,
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
        // k_col: [d_model, 1]  (transpose of the [1, d_model] key)
        let k_col = k.t()?.contiguous()?;

        // pred: [d_model, d_model] × [d_model, 1] → [d_model, 1] → squeeze → [d_model]
        let pred = session_state.matmul(&k_col)?.squeeze(D::Minus1)?;

        // v_vec: [d_model]  (remove the leading batch-of-one dimension)
        let v_vec = v.squeeze(0)?;

        // error: [d_model]
        let error = pred.sub(&v_vec)?;

        // Outer product: [d_model, 1] × [1, d_model] → [d_model, d_model]
        let grad = error.unsqueeze(1)?.matmul(&k)?;

        // W_tilde update: W_tilde ← W_tilde − η · grad
        // η is read live from the shared atomic so meta-training can decay it.
        let eta = f32::from_bits(self.inner_lr.load(Ordering::Relaxed));
        let lr = Tensor::new(eta, session_state.device())?;
        let updated_state = session_state.sub(&grad.broadcast_mul(&lr)?)?;
        *session_state = updated_state.clone();
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
}
