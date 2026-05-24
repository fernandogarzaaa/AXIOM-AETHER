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
            let (ttt_out, w_tilde_next) = self.ttt.forward_decode(&normed_x, state, None)?;
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

    /// Prefill and return per-layer decode initial states.
    ///
    /// When logarithmic prefill is active in a layer, it emits a compressed
    /// `W_tilde` seed; otherwise a zero state is used for that layer.
    pub fn prefill_with_state_init(&self, tokens: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let (batch, _) = tokens.dims2()?;
        let mut x = self.embeddings.forward(tokens)?;
        let mut init_states: Vec<Tensor> = Vec::with_capacity(self.layers.len());
        let h = self.config.num_heads;
        let d = self.config.head_dim;

        for layer in self.layers.iter() {
            let normed_x = layer.ttt_norm.forward(&x)?;
            let ttt_out = layer.ttt.forward_prefill(&normed_x)?;
            let x_res = x.add(&ttt_out)?;
            let ffn_out = layer.ffn.forward(&layer.ffn_norm.forward(&x_res)?)?;
            x = x_res.add(&ffn_out)?;

            let state = match layer.ttt.take_prefill_state() {
                Some(state) => state,
                None => Tensor::zeros((batch, h, d, d), candle_core::DType::F32, tokens.device())?,
            };
            init_states.push(state);
        }

        x = self.ln_f.forward(&x)?;
        let logits = self.output_head.forward(&x)?;
        Ok((logits, init_states))
    }

    /// Prefill with a memory vector prepended to the token sequence.
    ///
    /// Behaves identically to [`prefill_with_state_init`] but injects a
    /// pre-computed `[1, d_model]` memory vector as position 0 of the input
    /// embedding sequence before any layer processing.  This makes the TTT
    /// attention mechanism condition every subsequent token on the compressed
    /// context carried by the memory vector.
    ///
    /// # Arguments
    /// * `tokens`        – `[1, T]` integer prompt token indices.
    /// * `memory_vector` – `[1, d_model]` compressed memory tensor to inject.
    ///
    /// # Returns
    /// `(logits [1, T+1, vocab_size], per-layer init states)`
    pub fn prefill_with_state_init_and_memory(
        &self,
        tokens: &Tensor,
        memory_vector: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let (batch, _) = tokens.dims2()?;
        if batch != 1 {
            return Err(candle_core::Error::Msg(format!(
                "prefill_with_state_init_and_memory only supports batch size 1, got {batch}"
            )));
        }

        // Embed prompt tokens: [1, T, d_model]
        let token_embeddings = self.embeddings.forward(tokens)?;

        let (mem_batch, mem_d_model) = memory_vector.dims2()?;
        if mem_batch != 1 {
            return Err(candle_core::Error::Msg(format!(
                "memory_vector must have batch size 1, got {mem_batch}"
            )));
        }
        if mem_d_model != self.config.d_model {
            return Err(candle_core::Error::Msg(format!(
                "memory_vector d_model mismatch: expected {}, got {mem_d_model}",
                self.config.d_model
            )));
        }

        // Insert sequence dimension to [1, 1, d_model] then prepend → [1, T+1, d_model].
        // The memory vector aligns with q_proj input space (d_model) in every layer.
        let mem_prefix = memory_vector.unsqueeze(1)?;
        let mut x = Tensor::cat(&[&mem_prefix, &token_embeddings], 1)?;

        let mut init_states: Vec<Tensor> = Vec::with_capacity(self.layers.len());
        let h = self.config.num_heads;
        let d = self.config.head_dim;

        for layer in self.layers.iter() {
            let normed_x = layer.ttt_norm.forward(&x)?;
            let ttt_out = layer.ttt.forward_prefill(&normed_x)?;
            let x_res = x.add(&ttt_out)?;
            let ffn_out = layer.ffn.forward(&layer.ffn_norm.forward(&x_res)?)?;
            x = x_res.add(&ffn_out)?;

            let state = match layer.ttt.take_prefill_state() {
                Some(state) => state,
                None => Tensor::zeros((batch, h, d, d), candle_core::DType::F32, tokens.device())?,
            };
            init_states.push(state);
        }

        x = self.ln_f.forward(&x)?;
        let logits = self.output_head.forward(&x)?;
        Ok((logits, init_states))
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

    /// Evaluate multiple speculative next-token branches and choose the path with
    /// the lowest aggregate per-layer reconstruction error.
    pub fn speculative_branch_evaluate(
        &self,
        tokens: &Tensor,
        states: &Vec<Tensor>,
        branch_candidates: Vec<Tensor>,
    ) -> Result<usize> {
        if branch_candidates.is_empty() {
            return Err(candle_core::Error::Msg(
                "speculative_branch_evaluate requires at least one candidate".into(),
            ));
        }
        if states.len() != self.layers.len() {
            return Err(candle_core::Error::Msg(format!(
                "state count mismatch: expected {}, got {}",
                self.layers.len(),
                states.len()
            )));
        }

        let (token_batch, _) = tokens.dims2()?;
        let mut best_idx = 0usize;
        let mut best_loss = f32::INFINITY;

        for (idx, candidate) in branch_candidates.into_iter().enumerate() {
            let (candidate_batch, _) = candidate.dims2()?;
            if candidate_batch != token_batch {
                return Err(candle_core::Error::Msg(format!(
                    "candidate batch mismatch: expected {token_batch}, got {candidate_batch}"
                )));
            }

            let mut x = self.embeddings.forward(&candidate)?;
            let mut branch_states: Vec<Tensor> = states
                .iter()
                .map(|state| state.clone().contiguous())
                .collect::<Result<Vec<_>>>()?;
            let mut aggregate_loss = 0f32;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let normed_x = layer.ttt_norm.forward(&x)?;
                let (ttt_out, w_next, layer_loss) = layer.ttt.forward_decode_with_loss(
                    &normed_x,
                    &branch_states[layer_idx],
                    Some(4),
                )?;
                branch_states[layer_idx] = w_next;
                aggregate_loss += layer_loss;

                let x_res = x.add(&ttt_out)?;
                let ffn_out = layer.ffn.forward(&layer.ffn_norm.forward(&x_res)?)?;
                x = x_res.add(&ffn_out)?;
            }

            if aggregate_loss < best_loss {
                best_loss = aggregate_loss;
                best_idx = idx;
            }
        }

        Ok(best_idx)
    }
}
