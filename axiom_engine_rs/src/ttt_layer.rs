use std::cell::RefCell;

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use crate::config::AxiomConfig;
use crate::log_scan::LogosAssociativeScanner;

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
    prefill_w_tilde: RefCell<Option<Tensor>>,
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
            prefill_w_tilde: RefCell::new(None),
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
        let (_, t, _) = x.dims3()?;
        if self.config.use_log_scan || t > self.config.log_scan_auto_threshold {
            return self.forward_prefill_logarithmic(x);
        }
        *self.prefill_w_tilde.borrow_mut() = None;

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

    /// Logarithmic prefill branch backed by associative tree-reduction.
    pub fn forward_prefill_logarithmic(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let h = self.config.num_heads;
        let d = self.config.head_dim;

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

        // Build associative per-token state.  Shape: [B, T, C].
        let token_state = k
            .mul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c))?;

        // Full Blelloch inclusive prefix scan → [B, T, C].
        // Position i carries the cumulative combination of states[0..=i],
        // giving each query access to its own causal context window.
        let compressed = LogosAssociativeScanner::parallel_prefix_reduce(&token_state)?;

        // Derive decode initial state [B, H, D, D] from the final prefix sum.
        // Take the last sequence position: [B, T, C] → [B, C] → [B, H, D].
        let final_state = compressed.narrow(1, t - 1, 1)?.squeeze(1)?;
        let compressed_heads = final_state.reshape((b, h, d))?;
        let eye = Tensor::eye(d, DType::F32, x.device())?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let w_tilde_init = compressed_heads.unsqueeze(3)?.broadcast_mul(&eye)?;
        *self.prefill_w_tilde.borrow_mut() = Some(w_tilde_init);

        // Causal residual path: each query position i attends only to the
        // prefix sum up to i (already causal from the scan).
        // compressed: [B, T, C];  q_reshaped: [B, T, C]
        let q_reshaped = q.transpose(1, 2)?.reshape((b, t, c))?;
        let output = q_reshaped.add(&compressed)?;
        self.out_proj.forward(&output)
    }

    pub fn take_prefill_state(&self) -> Option<Tensor> {
        self.prefill_w_tilde.borrow_mut().take()
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
    pub fn forward_decode(
        &self,
        x: &Tensor,
        w_tilde: &Tensor,
        inner_loop_steps: Option<usize>,
    ) -> Result<(Tensor, Tensor)> {
        let (output, w_tilde_next, _) =
            self.forward_decode_with_loss(x, w_tilde, inner_loop_steps)?;
        Ok((output, w_tilde_next))
    }

    /// Single-token decode with adaptive inner-loop updates and scalar reconstruction loss.
    ///
    /// Returns `(output, updated_w_tilde, reconstruction_loss)` where reconstruction loss
    /// is measured on the final adapted state.
    pub fn forward_decode_with_loss(
        &self,
        x: &Tensor,
        w_tilde: &Tensor,
        inner_loop_steps: Option<usize>,
    ) -> Result<(Tensor, Tensor, f32)> {
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

        let k_norm_col = k_norm.unsqueeze(3)?;
        let k_norm_row = k_norm.unsqueeze(2)?;

        // Adaptive gradient descent on dynamic weights.
        let lr_tensor = Tensor::new(self.config.lr_inner, x.device())?;
        let loss_threshold = 1e-4f32;
        let stability_epsilon = 1e-7f32;
        // Hard cap to 4 speculative inner-loop updates per token step.
        let max_additional_steps = inner_loop_steps.unwrap_or(4).min(4);

        let mut w_tilde_next = w_tilde.clone();

        // Baseline reconstruction loss and first update (always executed).
        let pred_v = w_tilde_next.matmul(&k_norm_col)?.squeeze(3)?;
        let error = pred_v.sub(&v)?;
        let initial_loss = error.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let grad = error.unsqueeze(3)?.matmul(&k_norm_row)?;
        w_tilde_next = w_tilde_next.sub(&grad.broadcast_mul(&lr_tensor)?)?;

        // Optional adaptive lookahead updates for hard tokens.
        if initial_loss > loss_threshold {
            let mut prev_loss = initial_loss;
            for _ in 0..max_additional_steps {
                let pred_v = w_tilde_next.matmul(&k_norm_col)?.squeeze(3)?;
                let error = pred_v.sub(&v)?;
                let loss = error.sqr()?.sum_all()?.to_scalar::<f32>()?;

                if loss <= loss_threshold || (prev_loss - loss).abs() <= stability_epsilon {
                    break;
                }

                let grad = error.unsqueeze(3)?.matmul(&k_norm_row)?;
                w_tilde_next = w_tilde_next.sub(&grad.broadcast_mul(&lr_tensor)?)?;
                prev_loss = loss;
            }
        }

        // Query updated weight matrix and reshape to [B, 1, d_model]
        let out_state = w_tilde_next.matmul(&q.unsqueeze(3)?)?.squeeze(3)?;
        let output = out_state.reshape((b, 1, self.config.d_model))?;
        let final_pred = w_tilde_next.matmul(&k_norm_col)?.squeeze(3)?;
        let final_error = final_pred.sub(&v)?;
        let final_loss = final_error.sqr()?.sum_all()?.to_scalar::<f32>()?;

        Ok((self.out_proj.forward(&output)?, w_tilde_next, final_loss))
    }
}
