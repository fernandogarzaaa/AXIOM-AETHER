use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::AxiomConfig;

/// Multi-head linear test-time training layer.
///
/// Supports two operational forms:
///
/// - **Parallel prefill** via a scaled causal Gram matrix (O(T·D²) total).
/// - **Step-wise decode** with an explicit per-token W_tilde gradient update.
pub struct TTTLinearLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    config: AxiomConfig,
}

impl TTTLinearLayer {
    pub fn new(vs: VarBuilder, config: AxiomConfig) -> Result<Self> {
        let d = config.d_model;
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(d, d, vs.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(d, d, vs.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(d, d, vs.pp("v_proj"))?,
            out_proj: candle_nn::linear_no_bias(d, d, vs.pp("out_proj"))?,
            config,
        })
    }

    /// Parallel dual-form prefill over a full sequence.
    ///
    /// Computes a causal Gram matrix `η · Q Kᵀ`, masks it causally, then
    /// multiplies by `V` to produce context-corrected output.
    ///
    /// # Arguments
    /// * `x` – `[B, T, d_model]`
    ///
    /// # Returns
    /// `[B, T, d_model]`
    pub fn forward_prefill(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let h = self.config.num_heads;
        let d = self.config.head_dim;

        // Linear projections → [B, H, T, D]
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, t, h, d))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, t, h, d))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, t, h, d))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled attention Gram: [B, H, T, T]
        let lr_scalar = Tensor::new(self.config.lr_inner, x.device())?;
        let gram = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .broadcast_mul(&lr_scalar)?;

        // Causal lower-triangular mask
        let mask = Tensor::tril2(t, DType::F32, x.device())?
            .unsqueeze(0)?
            .unsqueeze(0)?; // [1, 1, T, T]

        let masked_gram = gram.broadcast_mul(&mask)?;

        // Aggregate values: [B, H, T, D]
        let context = masked_gram.matmul(&v)?;

        // Merge heads and project: [B, T, d_model]
        let output = context.transpose(1, 2)?.reshape((b, t, c))?;
        self.out_proj.forward(&output)
    }

    /// Single-token decode with in-place W_tilde gradient step.
    ///
    /// 1. Reconstruct: `pred = W_tilde @ k_norm`
    /// 2. Gradient: `∂L/∂W = (pred − v) · k_normᵀ`
    /// 3. Update: `W_tilde_next = W_tilde − η · grad`
    /// 4. Query: `y = W_tilde_next @ q`
    ///
    /// # Arguments
    /// * `x`       – `[B, 1, d_model]`
    /// * `w_tilde` – `[B, H, D, D]` dynamic weight matrix
    ///
    /// # Returns
    /// `(output [B, 1, d_model], w_tilde_next [B, H, D, D])`
    pub fn forward_decode(&self, x: &Tensor, w_tilde: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, _, _) = x.dims3()?;
        let h = self.config.num_heads;
        let d = self.config.head_dim;

        // Project and reshape to [B, H, D]
        let q = self.q_proj.forward(x)?.reshape((b, h, d))?;
        let k = self.k_proj.forward(x)?.reshape((b, h, d))?;
        let v = self.v_proj.forward(x)?.reshape((b, h, d))?;

        // L2-normalise key vector: k_norm = k / (‖k‖² + ε)^0.5
        let eps = Tensor::new(self.config.rms_norm_eps, x.device())?;
        let k_squared_sum = k.sqr()?.sum_keepdim(D::Minus1)?;
        let k_norm = k.broadcast_div(&k_squared_sum.broadcast_add(&eps)?.sqrt()?)?;

        // Reconstruction and gradient: [B, H, D, D]
        let pred_v = w_tilde.matmul(&k_norm.unsqueeze(3)?)?.squeeze(3)?;
        let error = pred_v.sub(&v)?;
        let grad = error.unsqueeze(3)?.matmul(&k_norm.unsqueeze(2)?)?;

        // Gradient descent step on dynamic weights
        let lr_tensor = Tensor::new(self.config.lr_inner, x.device())?;
        let w_tilde_next = w_tilde.sub(&grad.broadcast_mul(&lr_tensor)?)?;

        // Query updated weight matrix and reshape to [B, 1, d_model]
        let out_state = w_tilde_next.matmul(&q.unsqueeze(3)?)?.squeeze(3)?;
        let output = out_state.reshape((b, 1, self.config.d_model))?;

        Ok((self.out_proj.forward(&output)?, w_tilde_next))
    }
}
