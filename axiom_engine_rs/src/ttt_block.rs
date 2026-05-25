//! Standalone causal TTT block – the native replacement for a Transformer
//! Multi-Head Attention layer.
//!
//! For each incoming token activation the block performs three steps:
//!
//! 1. Map the input `x` to queries, keys, and values via static linear projections.
//! 2. Execute a self-supervised reconstruction gradient step on the fast-weight
//!    matrix `W̃` (the session state):
//!    `ΔW̃ = ∇L(f_{W̃}(k), v) = (W̃·k̂ − v) ⊗ k̂ᵀ`
//! 3. Evaluate the query against the freshly updated hidden state:
//!    `output = W̃_new × q`
//!
//! The output is passed through an embedded LayerNorm and output projection with a
//! residual connection, following the standard pre-LN pattern.

use candle_core::{Result, Tensor, D};
use candle_nn::{LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};

use crate::config::AxiomConfig;

/// Core native TTT block.
///
/// Replaces a Transformer Multi-Head Attention layer with an online
/// fast-weight learning kernel.  The per-layer `W̃` matrix (`session_state`)
/// accumulates knowledge token-by-token through gradient descent on a
/// self-supervised reconstruction objective.
pub struct NativeTTTBlock {
    /// Query projection: d_model → d_model.
    w_q: Linear,
    /// Key projection: d_model → d_model.
    w_k: Linear,
    /// Value projection: d_model → d_model.
    w_v: Linear,
    /// Output projection applied after querying the updated W̃.
    out_proj: Linear,
    /// Pre-norm applied to the input before the TTT core (standard pre-LN).
    /// Initialized with weight = ones so it behaves as a no-op at construction.
    norm: LayerNorm,
    config: AxiomConfig,
}

impl NativeTTTBlock {
    /// Construct a new `NativeTTTBlock` and register its parameters with `vs`.
    pub fn new(vs: VarBuilder, config: AxiomConfig) -> Result<Self> {
        let d = config.d_model;
        // Use LayerNorm with remove_mean=false (= RMSNorm) and affine=true so
        // that the weight is ones-initialized via get_with_hints(Init::Const(1.0)).
        let norm_config = LayerNormConfig {
            eps: config.rms_norm_eps as f64,
            remove_mean: false,
            affine: true,
        };
        Ok(Self {
            w_q: candle_nn::linear_no_bias(d, d, vs.pp("w_q"))?,
            w_k: candle_nn::linear_no_bias(d, d, vs.pp("w_k"))?,
            w_v: candle_nn::linear_no_bias(d, d, vs.pp("w_v"))?,
            out_proj: candle_nn::linear_no_bias(d, d, vs.pp("out_proj"))?,
            norm: candle_nn::layer_norm(d, norm_config, vs.pp("norm"))?,
            config,
        })
    }

    /// Forward pass over a (typically single-token) activation sequence.
    ///
    /// # Arguments
    /// * `x`             – Input activations, shape `[B, T, d_model]`.
    ///                     `T = 1` for standard autoregressive decoding.
    /// * `session_state` – Mutable fast-weight matrix `W̃`, shape `[B, H, D, D]`
    ///                     where `H = num_heads` and `D = head_dim`.
    ///                     Updated in-place via the reconstruction gradient step.
    ///
    /// # Returns
    /// Output activations, shape `[B, T, d_model]`.
    pub fn forward_native(&self, x: &Tensor, session_state: &mut Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let h = self.config.num_heads;
        let d = self.config.head_dim;

        let mut token_outputs: Vec<Tensor> = Vec::with_capacity(t);

        for ti in 0..t {
            // Extract single-token slice: [B, 1, d_model].
            let x_tok = x.narrow(1, ti, 1)?;

            // ── Pre-norm (standard pre-LN pattern) ────────────────────────────
            let x_normed = self.norm.forward(&x_tok)?;

            // ── Static linear projections → [B, H, D] ─────────────────────────
            let q = self.w_q.forward(&x_normed)?.reshape((b, h, d))?;
            let k = self.w_k.forward(&x_normed)?.reshape((b, h, d))?;
            let v = self.w_v.forward(&x_normed)?.reshape((b, h, d))?;

            // ── L2-normalise key: k̂ = k / √(‖k‖² + ε) ──────────────────────
            let eps = Tensor::new(self.config.rms_norm_eps, x.device())?;
            let k_sq_sum = k.sqr()?.sum_keepdim(D::Minus1)?;
            let k_norm = k.broadcast_div(&k_sq_sum.broadcast_add(&eps)?.sqrt()?)?;

            // Column view [B, H, D, 1] and row view [B, H, 1, D] for the
            // outer product that forms the rank-1 gradient matrix.
            let k_col = k_norm.unsqueeze(3)?;
            let k_row = k_norm.unsqueeze(2)?;

            // ── Fast-weight reconstruction update ──────────────────────────────
            //
            //   pred  = W̃ · k̂           →  [B, H, D]
            //   error = pred − v          →  [B, H, D]
            //   grad  = error ⊗ k̂ᵀ       →  [B, H, D, D]
            //   W̃_new = W̃ − η · grad
            //
            let pred = session_state.matmul(&k_col)?.squeeze(3)?;
            let error = pred.sub(&v)?;
            let grad = error.unsqueeze(3)?.matmul(&k_row)?;
            let lr = Tensor::new(self.config.lr_inner, x.device())?;
            *session_state = session_state.sub(&grad.broadcast_mul(&lr)?)?;

            // ── Query the updated state ────────────────────────────────────────
            //
            //   out_state = W̃_new · q    →  [B, H, D]
            //   ttt_out   = reshape      →  [B, 1, d_model]
            //
            let out_state = session_state.matmul(&q.unsqueeze(3)?)?.squeeze(3)?;
            let ttt_out = out_state.reshape((b, 1, self.config.d_model))?;

            // Output projection + residual connection.
            let projected = self.out_proj.forward(&ttt_out)?;
            token_outputs.push(x_tok.add(&projected)?);
        }

        if token_outputs.len() == 1 {
            Ok(token_outputs.remove(0))
        } else {
            let refs: Vec<&Tensor> = token_outputs.iter().collect();
            Tensor::cat(&refs, 1)
        }
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

    fn make_block(d: usize, h: usize) -> (NativeTTTBlock, Tensor, Device) {
        let config = AxiomConfig {
            d_model: d,
            n_layers: 1,
            num_heads: h,
            head_dim: d / h,
            vocab_size: 32,
            lr_inner: 1e-3,
            rms_norm_eps: 1e-6,
            use_log_scan: false,
            log_scan_auto_threshold: 128,
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = NativeTTTBlock::new(vb, config.clone()).unwrap();

        // Identity-matrix initial state for each head: [1, H, D, D].
        let mut heads: Vec<Tensor> = Vec::new();
        for _ in 0..h {
            let eye = Tensor::eye(config.head_dim, DType::F32, &device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            heads.push(eye);
        }
        let head_refs: Vec<&Tensor> = heads.iter().collect();
        let state = Tensor::cat(&head_refs, 0).unwrap().unsqueeze(0).unwrap(); // [1, H, D, D]

        (block, state, device)
    }

    #[test]
    fn forward_native_output_shape() {
        let (block, mut state, device) = make_block(16, 2);
        let x = Tensor::zeros((1usize, 1usize, 16usize), DType::F32, &device).unwrap();
        let out = block.forward_native(&x, &mut state).unwrap();
        assert_eq!(out.dims(), &[1, 1, 16]);
    }

    #[test]
    fn forward_native_multi_token_shape() {
        let (block, mut state, device) = make_block(16, 2);
        let x = Tensor::zeros((1usize, 4usize, 16usize), DType::F32, &device).unwrap();
        let out = block.forward_native(&x, &mut state).unwrap();
        assert_eq!(out.dims(), &[1, 4, 16]);
    }

    #[test]
    fn forward_native_updates_session_state() {
        let (block, mut state, device) = make_block(16, 2);
        let state_before: Vec<f32> = state.flatten_all().unwrap().to_vec1().unwrap();
        // Use a non-zero input so the reconstruction gradient is non-trivial.
        let x = Tensor::ones((1usize, 1usize, 16usize), DType::F32, &device).unwrap();
        let _ = block.forward_native(&x, &mut state).unwrap();
        let state_after: Vec<f32> = state.flatten_all().unwrap().to_vec1().unwrap();
        // With Kaiming-initialized (non-zero) projections, the gradient is non-zero
        // and W̃ must change.
        assert_ne!(
            state_before, state_after,
            "session state must be updated after a gradient step"
        );
    }
}
