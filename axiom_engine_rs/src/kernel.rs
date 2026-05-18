use candle_core::{Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::config::AxiomConfig;
use crate::ttt_layer::TTTLinearLayer;

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization.
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32, vs: VarBuilder) -> Result<Self> {
        let weight = vs.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let eps = Tensor::new(self.eps, x.device())?;
        let norm = x.broadcast_div(&variance.broadcast_add(&eps)?.sqrt()?)?;
        norm.broadcast_mul(&self.weight)
    }
}

// ---------------------------------------------------------------------------
// SwiGLU Feed-Forward Network
// ---------------------------------------------------------------------------

/// SwiGLU-activated Feed-Forward Network.
///
/// Hidden dimension: `int(2 * (d_model * 4 / 3) / 2)`
/// Gate: `silu(w1(x)) * w3(x)`
/// Down: `w2(gate)`
pub struct SwiGLUFFN {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl SwiGLUFFN {
    pub fn new(vs: VarBuilder, config: &AxiomConfig) -> Result<Self> {
        // Mirror the Python formula: int(2 * (d_model * 4 / 3) / 2)
        let hidden_dim = 2 * (config.d_model * 4 / 3) / 2;
        let d = config.d_model;
        Ok(Self {
            w1: candle_nn::linear_no_bias(d, hidden_dim, vs.pp("w1"))?,
            w2: candle_nn::linear_no_bias(hidden_dim, d, vs.pp("w2"))?,
            w3: candle_nn::linear_no_bias(d, hidden_dim, vs.pp("w3"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let silu = self.w1.forward(x)?.silu()?;
        let gated = silu.mul(&self.w3.forward(x)?)?;
        self.w2.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// AxiomBlock
// ---------------------------------------------------------------------------

/// Single Pre-LN residual block: RMSNorm → TTTLinearLayer → SwiGLUFFN.
pub struct AxiomBlock {
    ttt_norm: RMSNorm,
    ttt: TTTLinearLayer,
    ffn_norm: RMSNorm,
    ffn: SwiGLUFFN,
}

impl AxiomBlock {
    pub fn new(vs: VarBuilder, config: AxiomConfig) -> Result<Self> {
        Ok(Self {
            ttt_norm: RMSNorm::new(config.d_model, config.rms_norm_eps, vs.pp("ttt_norm"))?,
            ttt: TTTLinearLayer::new(vs.pp("ttt"), config.clone())?,
            ffn_norm: RMSNorm::new(config.d_model, config.rms_norm_eps, vs.pp("ffn_norm"))?,
            ffn: SwiGLUFFN::new(vs.pp("ffn"), &config)?,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x`          – `[B, T, d_model]` (prefill) or `[B, 1, d_model]` (decode).
    /// * `w_tilde`    – Per-layer dynamic weight `[B, H, D, D]`; required for decode.
    /// * `use_decode` – Select step-wise decode (true) or parallel prefill (false).
    ///
    /// # Returns
    /// `(output, Option<w_tilde_next>)` – next W_tilde is `Some` only during decode.
    pub fn forward(
        &self,
        x: &Tensor,
        w_tilde: Option<&Tensor>,
        use_decode: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        if use_decode {
            let state = w_tilde.expect("W_tilde required for decode phase.");
            let normed_x = self.ttt_norm.forward(x)?;
            let (ttt_out, w_tilde_next) = self.ttt.forward_decode(&normed_x, state)?;
            let x = x.add(&ttt_out)?;
            let ffn_out = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
            let x = x.add(&ffn_out)?;
            Ok((x, Some(w_tilde_next)))
        } else {
            let normed_x = self.ttt_norm.forward(x)?;
            let ttt_out = self.ttt.forward_prefill(&normed_x)?;
            let x = x.add(&ttt_out)?;
            let ffn_out = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
            let x = x.add(&ffn_out)?;
            Ok((x, None))
        }
    }
}

// ---------------------------------------------------------------------------
// AxiomTTTEngine
// ---------------------------------------------------------------------------

/// Full model stack: Embedding → N × AxiomBlock → RMSNorm → LM Head.
pub struct AxiomTTTEngine {
    embeddings: Embedding,
    layers: Vec<AxiomBlock>,
    ln_f: RMSNorm,
    output_head: Linear,
    config: AxiomConfig,
}

impl AxiomTTTEngine {
    pub fn new(vs: VarBuilder, config: AxiomConfig) -> Result<Self> {
        let embeddings =
            candle_nn::embedding(config.vocab_size, config.d_model, vs.pp("embeddings"))?;
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            layers.push(AxiomBlock::new(
                vs.pp(&format!("layer_{i}")),
                config.clone(),
            )?);
        }
        let ln_f = RMSNorm::new(config.d_model, config.rms_norm_eps, vs.pp("ln_f"))?;
        let output_head =
            candle_nn::linear_no_bias(config.d_model, config.vocab_size, vs.pp("output_head"))?;

        Ok(Self {
            embeddings,
            layers,
            ln_f,
            output_head,
            config,
        })
    }

    /// Full forward pass.
    ///
    /// # Arguments
    /// * `tokens`      – `[B, T]` integer token indices.
    /// * `states`      – Per-layer W_tilde tensors `[B, H, D, D]`; used in decode mode.
    /// * `use_decode`  – Route all blocks through step-wise decode when `true`.
    ///
    /// # Returns
    /// `(logits [B, T, vocab_size], Option<updated_states>)` – states are returned
    /// only when `use_decode = true`.
    pub fn forward(
        &self,
        tokens: &Tensor,
        states: Option<Vec<Tensor>>,
        use_decode: bool,
    ) -> Result<(Tensor, Option<Vec<Tensor>>)> {
        let mut x = self.embeddings.forward(tokens)?;
        let mut next_states: Vec<Tensor> = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let w_tilde = states.as_ref().map(|s| &s[i]);
            let (x_next, w_next) = layer.forward(&x, w_tilde, use_decode)?;
            x = x_next;
            if let Some(w) = w_next {
                next_states.push(w);
            }
        }

        x = self.ln_f.forward(&x)?;
        let logits = self.output_head.forward(&x)?;

        let out_states = if use_decode { Some(next_states) } else { None };
        Ok((logits, out_states))
    }

    /// Allocate zeroed W_tilde tensors for all layers.
    ///
    /// Shape per layer: `[batch, num_heads, head_dim, head_dim]`.
    pub fn init_states(&self, batch: usize, device: &candle_core::Device) -> Result<Vec<Tensor>> {
        let h = self.config.num_heads;
        let d = self.config.head_dim;
        (0..self.config.n_layers)
            .map(|_| Tensor::zeros((batch, h, d, d), candle_core::DType::F32, device))
            .collect()
    }
}
