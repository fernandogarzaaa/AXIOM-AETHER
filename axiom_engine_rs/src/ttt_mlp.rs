//! B.5 — TTT-MLP: an expressive (2-layer MLP) fast-weight hidden state.
//!
//! Where [`crate::ttt_block::NativeTTTBlock`] carries a single linear map `W̃` as
//! its recurrent state (TTT-Linear), this block carries a **two-layer MLP**
//! `pred(k) = W₂·φ(W₁·k)` as its state (TTT-MLP, Sun et al. 2024). The richer
//! hidden state can fit *nonlinear* key→value associations a single matrix
//! cannot, which is the paper's reported long-context advantage — at higher
//! memory/compute cost, so this is an **opt-in** primitive ("measure first").
//!
//! Each token still triggers exactly one self-supervised gradient step, now
//! backpropagated through both layers of the MLP state:
//!
//! ```text
//! z   = W₁ k          φ = tanh           a = φ(z)
//! pred = W₂ a         e = pred − v
//! dW₂ = e ⊗ a
//! da  = W₂ᵀ e         dz = da ⊙ (1 − a²)     (tanh′)
//! dW₁ = dz ⊗ k
//! W₂ ← W₂ − η dW₂      W₁ ← W₁ − η dW₁
//! out  = W₂ φ(W₁ q)
//! ```
//!
//! `tanh` is used as the inner nonlinearity precisely because its derivative
//! `1 − tanh²` is cheap and exact, keeping the per-token gradient step closed-form
//! (no autograd tape) just like the linear block.
//!
//! The state is **two** matrices, so this block is not a drop-in for the
//! single-`[d,d]`-tensor state plumbing of `AxiomTTTLM`; it is provided as a
//! standalone, fully-tested primitive plus its own [`MlpState`], to be wired into
//! a model behind a benchmark once long-context need is measured.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

use crate::config::AxiomConfig;

/// The recurrent MLP fast-weight state: `W₁ ∈ [h,d]`, `W₂ ∈ [d,h]`.
#[derive(Debug, Clone)]
pub struct MlpState {
    /// First layer weights `[hidden, d_model]`.
    pub w1: Tensor,
    /// Second layer weights `[d_model, hidden]`.
    pub w2: Tensor,
}

impl MlpState {
    /// Neutral, deterministic, parameter-free init: rectangular identities so the
    /// state starts non-degenerate (a zero `W₁` would stop `W₂` from ever
    /// learning, since `dW₂ = e ⊗ φ(0) = 0`).
    pub fn init(d_model: usize, hidden: usize, device: &Device) -> Result<Self> {
        let n = d_model.max(hidden);
        let eye = Tensor::eye(n, DType::F32, device)?;
        let w1 = eye.narrow(0, 0, hidden)?.narrow(1, 0, d_model)?.contiguous()?;
        let w2 = eye.narrow(0, 0, d_model)?.narrow(1, 0, hidden)?.contiguous()?;
        Ok(Self { w1, w2 })
    }
}

/// Standalone causal TTT-MLP block.
pub struct NativeTTTMlpBlock {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    layer_norm: LayerNorm,
    hidden: usize,
    d_model: usize,
    /// Shared inner test-time learning rate η (raw f32 bits), like the linear
    /// block — so a model can decay η across the whole stack at once.
    inner_lr: Arc<AtomicU32>,
}

impl NativeTTTMlpBlock {
    /// Construct a block with its own private inner-lr cell from `config.lr_inner`.
    #[allow(dead_code)]
    pub fn new(vs: VarBuilder, config: AxiomConfig, hidden: usize) -> Result<Self> {
        let inner_lr = Arc::new(AtomicU32::new(config.lr_inner.to_bits()));
        Self::new_with_shared_lr(vs, config, hidden, inner_lr)
    }

    /// Construct a block reading its inner learning rate from a shared cell.
    pub fn new_with_shared_lr(
        vs: VarBuilder,
        config: AxiomConfig,
        hidden: usize,
        inner_lr: Arc<AtomicU32>,
    ) -> Result<Self> {
        let d = config.d_model;
        Ok(Self {
            w_q: candle_nn::linear_no_bias(d, d, vs.pp("w_q"))?,
            w_k: candle_nn::linear_no_bias(d, d, vs.pp("w_k"))?,
            w_v: candle_nn::linear_no_bias(d, d, vs.pp("w_v"))?,
            layer_norm: candle_nn::layer_norm_no_bias(d, config.norm_eps as f64, vs.pp("layer_norm"))?,
            hidden,
            d_model: d,
            inner_lr,
        })
    }

    /// Hidden width of the MLP state.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Allocate a neutral initial [`MlpState`] for this block.
    pub fn init_state(&self, device: &Device) -> Result<MlpState> {
        MlpState::init(self.d_model, self.hidden, device)
    }

    /// Autoregressive forward step for a single token, updating the MLP state via
    /// one closed-form gradient step through both layers.
    ///
    /// * `x`     – `[1, d_model]` token activation.
    /// * `state` – the `[h,d]` / `[d,h]` MLP fast-weights, updated in place.
    ///
    /// Returns `[1, d_model]` after the update and embedded layer norm.
    pub fn forward_native(&self, x: &Tensor, state: &mut MlpState) -> Result<Tensor> {
        let q = self.w_q.forward(x)?; // [1,d]
        let k = self.w_k.forward(x)?; // [1,d]
        let v = self.w_v.forward(x)?; // [1,d]

        let k_vec = k.squeeze(0)?; // [d]
        let v_vec = v.squeeze(0)?; // [d]

        // Forward through the MLP state: z = W1 k, a = tanh(z), pred = W2 a.
        let k_col = k_vec.unsqueeze(1)?; // [d,1]
        let z = state.w1.matmul(&k_col)?.squeeze(D::Minus1)?; // [h]
        let a = z.tanh()?; // [h]
        let a_col = a.unsqueeze(1)?; // [h,1]
        let pred = state.w2.matmul(&a_col)?.squeeze(D::Minus1)?; // [d]
        let e = pred.sub(&v_vec)?; // [d]

        let eta = f32::from_bits(self.inner_lr.load(Ordering::Relaxed));
        let lr = Tensor::new(eta, x.device())?;

        // dW2 = e ⊗ a  ([d,1]·[1,h] = [d,h]).
        let dw2 = e.unsqueeze(1)?.matmul(&a.unsqueeze(0)?)?;
        // da = W2ᵀ e  ([h,d]·[d,1] = [h,1] → [h]).
        let da = state.w2.t()?.matmul(&e.unsqueeze(1)?)?.squeeze(D::Minus1)?;
        // dz = da ⊙ (1 − a²)   (tanh′).
        let one_minus_a2 = a.sqr()?.affine(-1.0, 1.0)?; // 1 − a²
        let dz = da.mul(&one_minus_a2)?; // [h]
        // dW1 = dz ⊗ k  ([h,1]·[1,d] = [h,d]).
        let dw1 = dz.unsqueeze(1)?.matmul(&k_vec.unsqueeze(0)?)?;

        // One gradient-descent step on both layers; detach (inference cache, not
        // a BPTT tape).
        state.w2 = state.w2.sub(&dw2.broadcast_mul(&lr)?)?.detach();
        state.w1 = state.w1.sub(&dw1.broadcast_mul(&lr)?)?.detach();

        // Output: read the (updated) MLP with the query. out = W2 tanh(W1 q).
        let q_col = q.squeeze(0)?.unsqueeze(1)?; // [d,1]
        let zq = state.w1.matmul(&q_col)?.squeeze(D::Minus1)?; // [h]
        let aq = zq.tanh()?.unsqueeze(1)?; // [h,1]
        let out = state.w2.matmul(&aq)?.squeeze(D::Minus1)?.unsqueeze(0)?; // [1,d]

        self.layer_norm.forward(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    fn make_block(d_model: usize, hidden: usize) -> (NativeTTTMlpBlock, Device) {
        let device = Device::Cpu;
        let config = AxiomConfig {
            d_model,
            n_layers: 1,
            vocab_size: 16,
            lr_inner: 1e-2,
            norm_eps: 1e-6,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = NativeTTTMlpBlock::new(vb.pp("mlp"), config, hidden).unwrap();
        (block, device)
    }

    #[test]
    fn state_init_has_expected_shapes() {
        let st = MlpState::init(8, 12, &Device::Cpu).unwrap();
        assert_eq!(st.w1.dims(), &[12, 8]);
        assert_eq!(st.w2.dims(), &[8, 12]);
    }

    #[test]
    fn forward_output_shape_and_finite() {
        let d = 16usize;
        let (block, device) = make_block(d, d);
        let mut st = block.init_state(&device).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();
        let out = block.forward_native(&x, &mut st).unwrap();
        assert_eq!(out.dims(), &[1, d]);
        let ov: Vec<f32> = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(ov.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn both_layers_update() {
        let d = 12usize;
        let (block, device) = make_block(d, d);
        let mut st = block.init_state(&device).unwrap();
        let w1_before = st.w1.clone();
        let w2_before = st.w2.clone();
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();
        let _ = block.forward_native(&x, &mut st).unwrap();
        let d1 = st.w1.sub(&w1_before).unwrap().sqr().unwrap().sum_all().unwrap()
            .to_scalar::<f32>().unwrap();
        let d2 = st.w2.sub(&w2_before).unwrap().sqr().unwrap().sum_all().unwrap()
            .to_scalar::<f32>().unwrap();
        assert!(d1 > 0.0, "W1 must update");
        assert!(d2 > 0.0, "W2 must update");
    }

    #[test]
    fn rectangular_hidden_is_supported() {
        // hidden != d_model must work (the paper uses a wider hidden state).
        let d = 8usize;
        let (block, device) = make_block(d, 20);
        assert_eq!(block.hidden(), 20);
        let mut st = block.init_state(&device).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();
        let out = block.forward_native(&x, &mut st).unwrap();
        assert_eq!(out.dims(), &[1, d]);
    }

    #[test]
    fn repeated_token_drives_reconstruction_error_down() {
        // Feeding the same token repeatedly, the inner reconstruction loss
        // ‖pred − v‖ for that token must decrease — proof the MLP state is
        // actually learning the key→value association via its gradient steps.
        let d = 16usize;
        let (block, device) = make_block(d, d);
        let mut st = block.init_state(&device).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();

        // Reconstruction error measured by re-deriving pred from the current state.
        let recon_err = |block: &NativeTTTMlpBlock, st: &MlpState| -> f32 {
            let k = block.w_k.forward(&x).unwrap().squeeze(0).unwrap();
            let v = block.w_v.forward(&x).unwrap().squeeze(0).unwrap();
            let z = st.w1.matmul(&k.unsqueeze(1).unwrap()).unwrap().squeeze(D::Minus1).unwrap();
            let a = z.tanh().unwrap();
            let pred = st.w2.matmul(&a.unsqueeze(1).unwrap()).unwrap().squeeze(D::Minus1).unwrap();
            pred.sub(&v).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap()
        };

        let before = recon_err(&block, &st);
        for _ in 0..40 {
            let _ = block.forward_native(&x, &mut st).unwrap();
        }
        let after = recon_err(&block, &st);
        assert!(
            after < before,
            "MLP state must reduce reconstruction error (before {before}, after {after})"
        );
    }
}
