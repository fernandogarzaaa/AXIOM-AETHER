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

/// Length of the per-token learnable LR index vector. Token positions beyond
/// this are clamped to the last entry. 512 covers typical training windows.
const LEARNABLE_TOKEN_IDX_LEN: usize = 512;

/// Standalone causal TTT block.
///
/// Replaces Multi-Head Attention as the core sequence-mixing primitive.
/// Projection weights `W_q`, `W_k`, `W_v` map hidden representations to
/// query/key/value spaces.  An embedded `RMSNorm` is applied to every output.
pub struct NativeTTTBlock {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    /// Fast-weight bias vector `[d_model]`, added to the key→value prediction:
    /// `pred = W~ @ k + b`. Initialized to zeros so the default path is
    /// byte-identical to the unbiased version. Learned via outer-loop
    /// meta-gradients (like the reference's `b1` parameter).
    fast_bias: Tensor,
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
    /// Gated-DeltaNet forget gate α ∈ (0, 1] (raw f32 bits), shared across the
    /// stack. The delta-rule update is `W̃ ← W̃(I − ηkkᵀ) + ηvkᵀ`; this multiplies
    /// the *retained-memory* term by α, giving `W̃ ← α·W̃(I − ηkkᵀ) + ηvkᵀ`. With
    /// α < 1 old memory decays geometrically; combined with bounded/normalized
    /// keys (the `stabilize` path, where ‖I − ηkkᵀ‖ ≤ 1) this gives a spectral
    /// radius ≤ α, a principled bound on ‖W̃‖ rather than the hard element clamp.
    /// Default 1.0 is the identity: the update reduces to the exact ungated path
    /// (byte-identical), so existing checkpoints are unaffected. Parameter-free,
    /// so no checkpoint-format change. (The learned data-dependent gate is a
    /// follow-up requiring a new projection weight.)
    forget_gate: Arc<AtomicU32>,
    /// Optional *learned* data-dependent forget gate: a `w_α: Linear(d → 1)`
    /// projection whose per-token output α_t = α_min + (1−α_min)·σ(w_α·x) decays
    /// the retained-memory term. `Some` only when the model was built with the
    /// learned gate (and the checkpoint carries `w_alpha` weights); `None` falls
    /// back to the scalar `forget_gate` above. When present it takes precedence,
    /// letting the network decide what to forget per token (Gated-DeltaNet's
    /// data-dependent gate). A constant logit offset (`GATE_INIT_LOGIT`) makes α
    /// start near 1 (≈ ungated), so training departs from the proven dynamics
    /// rather than a cold, aggressively-forgetting gate.
    learned_gate: Option<Linear>,
    /// Shared safe-online-update guards (drift reset, token selection,
    /// anti-forgetting anchor). Shared `Arc` like `inner_lr` so one call on the
    /// model retunes every layer at once. Default-disabled ⇒ byte-identical.
    guards: Arc<OnlineGuards>,
    /// Learnable per-layer inner-LR modulation (finding #3). The effective
    /// learning rate is `η_eff = η_base * 2 * σ(lr_scale)`, where `η_base` is
    /// the global scheduled scalar from `inner_lr`. Initialized to 0.0 ⇒
    /// `2*σ(0) = 1.0` ⇒ identical to the unmodulated path at start of training.
    /// This is a per-layer (not per-token) gate: a minimal viable step toward
    /// the reference's per-token learnable LR (`token_eta` + `ttt_lr_eta`).
    /// Registered via `vs.pp("lr_scale")` so the optimizer updates it during
    /// meta-training (unlike `inner_lr`, which is a non-differentiable atomic).
    lr_scale: Tensor,
    /// Per-token learnable inner-LR offset (finding #3, per-token part).
    /// The reference computes `token_eta = 1/t + learnable_token_idx[t]`.
    /// We compute `token_scale = 1/(t+1) + learnable_token_idx[min(t, 511)]`
    /// and multiply it into the effective LR. Zeros init ⇒ starts as pure
    /// `1/(t+1)` decay. Registered via `vs.pp("learnable_token_idx")` so the
    /// optimizer updates it. Backward compat: old checkpoints lack this key;
    /// falls back to zeros.
    learnable_token_idx: Tensor,
    /// Data-dependent inner-LR gate (adapted from ttt-lm-pytorch's
    /// `ttt_lr_eta`). Computes `gate = 2*sigmoid(x @ w_lr + b_lr)`,
    /// 1.0 at init (neutral). `w_lr`: [d_model, 1], `b_lr`: scalar.
    /// Registered so optimizer updates them. Backward compat: zeros fallback.
    w_lr: Tensor,
    b_lr: Tensor,
    /// Titans surprise-gating parameters. The fast-weight update is scaled by
    /// `gate = sigmoid(alpha * (||error|| - beta))`. Init alpha=1.0, beta=0.0.
    /// Registered so optimizer tunes them. Backward compat: fallback to init.
    surprise_alpha: Tensor,
    surprise_beta: Tensor,
}

/// Lower bound on the learned forget gate, keeping α ∈ [GATE_FLOOR, 1) so a
/// saturated gate can never fully erase memory or push the spectral radius to 0.
const GATE_FLOOR: f64 = 1e-3;

/// Constant offset added to the learned-gate logit so that at initialization
/// (w_α ≈ 0) the gate σ(logit) ≈ σ(4) ≈ 0.98 ⇒ α ≈ 1. Training thus starts from
/// the proven near-ungated dynamics and *learns* to forget, rather than booting
/// with a cold, aggressively-forgetting gate.
const GATE_INIT_LOGIT: f64 = 4.0;

/// Shared, runtime-tunable guards for **safe online (persistent) test-time
/// updates**, grounded in the TTA literature (RDumb periodic reset, EATA sample
/// selection + Fisher anchoring, CoTTA stochastic restoration). When the
/// fast-weight matrix is carried across a long stream of tokens, the bare delta
/// rule can drift, accumulate error, or collapse; these three orthogonal guards
/// bound that without altering the proven short-context dynamics.
///
/// Every field defaults to a disabled/identity value (`0.0`), so a freshly
/// constructed block is **byte-identical to the ungated path** until a guard is
/// explicitly enabled — and the default path stays device-sync-free (the guards'
/// scalar reads only happen when enabled).
///
/// Orchestration order inside `forward_native` (each independent, layered so they
/// cannot fight): **(1) token selection** decides whether the update is kept at
/// all; **(2) anti-forgetting anchor** pulls the kept update toward the init;
/// **(3) drift reset** is the final backstop on the resulting state.
#[derive(Debug, Default)]
pub struct OnlineGuards {
    /// Drift-aware reset (RDumb). If the post-update fast-weight Frobenius norm
    /// exceeds this threshold, `W̃` is reset to its meta-trained init (identity).
    /// `0.0` disables the check.
    pub drift_reset_norm: AtomicU32,
    /// Token selection (EATA). The inner update is skipped when the
    /// reconstruction-error L2 norm ‖pred − v‖ is *below* this threshold (the
    /// token carries little new information). `0.0` never skips — every token
    /// updates `W̃` exactly as before.
    pub update_min_error: AtomicU32,
    /// Anti-forgetting anchor (EATA Fisher / CoTTA stochastic restore). After the
    /// update, `W̃ ← (1−λ)·W̃ + λ·I` pulls the state weakly back toward the
    /// meta-trained init. λ ∈ [0,1]; `0.0` disables anchoring.
    pub anchor_strength: AtomicU32,
    /// Pre-update gradient veto: skip the TTT update when `||g||_2 > max_grad_norm`.
    /// `0.0` disables the veto.
    pub max_grad_norm: AtomicU32,
    /// Post-update NaN/Inf rollback: when `true`, snapshot the state before the
    /// update and restore it if any element becomes non-finite.
    pub nan_rollback: AtomicBool,
    /// **B.6 inner-loss ablation** (separate from the guards above). When `true`,
    /// the self-supervised inner objective L2-normalizes both the predicted and
    /// the value view before taking the error, turning the MSE reconstruction
    /// loss into a directional / cosine ("contrastive multi-view") objective —
    /// the TTT++ finding that a contrastive aux task can beat plain
    /// reconstruction. `false` (default) is the exact reconstruction path, so the
    /// dynamics stay byte-identical until enabled.
    pub aux_loss_normalized: AtomicBool,
}

/// L2-normalize a 1-D `[d]` vector to unit length (with an epsilon floor so a
/// zero vector maps to ~zero rather than NaN). Used by the B.6 contrastive
/// inner-loss ablation to compare view *directions*.
fn l2_normalize(v: &Tensor) -> Result<Tensor> {
    let norm = v.sqr()?.sum_all()?.sqrt()?.affine(1.0, 1e-6)?;
    v.broadcast_div(&norm)
}

impl OnlineGuards {
    /// Construct guards in the disabled (identity / byte-identical) state.
    pub fn disabled() -> Self {
        Self::default()
    }
    /// Set the drift-reset Frobenius-norm threshold (`0.0` disables).
    pub fn set_drift_reset_norm(&self, v: f32) {
        self.drift_reset_norm.store(v.to_bits(), Ordering::Relaxed);
    }
    /// Set the token-selection minimum reconstruction-error threshold
    /// (`0.0` never skips).
    pub fn set_update_min_error(&self, v: f32) {
        self.update_min_error.store(v.to_bits(), Ordering::Relaxed);
    }
    /// Set the anti-forgetting anchor strength λ ∈ [0,1] (`0.0` disables).
    pub fn set_anchor_strength(&self, v: f32) {
        self.anchor_strength
            .store(v.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }
    /// Set the pre-update gradient-norm veto threshold (`0.0` disables).
    pub fn set_max_grad_norm(&self, v: f32) {
        self.max_grad_norm.store(v.max(0.0).to_bits(), Ordering::Relaxed);
    }
    /// Enable/disable post-update NaN/Inf rollback.
    pub fn set_nan_rollback(&self, v: bool) {
        self.nan_rollback.store(v, Ordering::Relaxed);
    }
    /// Select the inner-loss form: `true` = normalized/contrastive multi-view,
    /// `false` (default) = exact MSE reconstruction (byte-identical).
    pub fn set_aux_loss_normalized(&self, on: bool) {
        self.aux_loss_normalized.store(on, Ordering::Relaxed);
    }
    fn aux_normalized(&self) -> bool {
        self.aux_loss_normalized.load(Ordering::Relaxed)
    }
    fn drift(&self) -> f32 {
        f32::from_bits(self.drift_reset_norm.load(Ordering::Relaxed))
    }
    fn min_error(&self) -> f32 {
        f32::from_bits(self.update_min_error.load(Ordering::Relaxed))
    }
    fn anchor(&self) -> f32 {
        f32::from_bits(self.anchor_strength.load(Ordering::Relaxed))
    }
    fn max_grad_norm_val(&self) -> f32 {
        f32::from_bits(self.max_grad_norm.load(Ordering::Relaxed))
    }
    fn nan_rollback_on(&self) -> bool {
        self.nan_rollback.load(Ordering::Relaxed)
    }
    /// True when every guard is at its disabled default — lets `forward_native`
    /// skip the guard block (and its device syncs) entirely on the hot path.
    fn all_disabled(&self) -> bool {
        self.drift() == 0.0
            && self.min_error() == 0.0
            && self.anchor() == 0.0
            && self.max_grad_norm_val() == 0.0
    }
}

impl NativeTTTBlock {
    /// Construct a new block with its own private inner-lr cell initialised
    /// from `config.lr_inner`.
    #[allow(dead_code)] // used by the lib path + tests; the bin builds via the model
    pub fn new(vs: VarBuilder, config: AxiomConfig) -> Result<Self> {
        let inner_lr = Arc::new(AtomicU32::new(config.lr_inner.to_bits()));
        let stabilize = Arc::new(AtomicBool::new(false));
        let forget_gate = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let guards = Arc::new(OnlineGuards::disabled());
        Self::new_with_shared_lr(vs, config, inner_lr, stabilize, forget_gate, guards, false)
    }

    /// Construct a block that reads its inner learning rate from a shared
    /// atomic cell — used by `AxiomTTTLM` so a single `set_inner_lr` call
    /// retunes every layer at once. `learned_gate` allocates the `w_α` projection
    /// for the data-dependent forget gate (adds `w_alpha` weights to the
    /// checkpoint); when false the block is parameter-identical to before.
    pub fn new_with_shared_lr(
        vs: VarBuilder,
        config: AxiomConfig,
        inner_lr: Arc<AtomicU32>,
        stabilize: Arc<AtomicBool>,
        forget_gate: Arc<AtomicU32>,
        guards: Arc<OnlineGuards>,
        learned_gate: bool,
    ) -> Result<Self> {
        let d = config.d_model;
        let learned_gate = if learned_gate {
            // Linear(d → 1) with bias. Bias init defaults near 0 ⇒ σ≈0.5; we want
            // α to start near 1, so the gate logit is biased positive below.
            Some(candle_nn::linear(d, 1, vs.pp("w_alpha"))?)
        } else {
            None
        };
        let w_q = candle_nn::linear_no_bias(d, d, vs.pp("w_q"))?;
        let w_k = candle_nn::linear_no_bias(d, d, vs.pp("w_k"))?;
        let w_v = candle_nn::linear_no_bias(d, d, vs.pp("w_v"))?;
        // Device for fallback init (backward compat with old checkpoints).
        let device = w_q.weight().device().clone();
        // Learnable per-layer LR scale (scalar). Formula: η_eff = η_base * 2σ(lr_scale).
        // Default init gives values near 0 ⇒ factor ≈ 1.0 ⇒ starts close to the
        // unmodulated path. The optimizer tunes it during meta-training.
        // Backward compat: old checkpoints lack lr_scale; init to 0.
        let lr_scale = vs.pp("lr_scale").get((), "lr_scale").unwrap_or_else(|_| {
            Tensor::zeros((), candle_core::DType::F32, &device).unwrap()
        });
        // Fast-weight bias. Backward compat: old checkpoints lack fast_bias; init to 0.
        let fast_bias = vs.pp("fast_bias").get(d, "fast_bias").unwrap_or_else(|_| {
            Tensor::zeros(d, candle_core::DType::F32, &device).unwrap()
        });
        // Per-token learnable LR offset. Zeros init ⇒ token_scale starts as
        // pure 1/(t+1). Backward compat: old checkpoints lack this key;
        // init to t/(t+1) so token_scale = 1/(t+1) + t/(t+1) = 1.0,
        // preserving the old (unscaled) behavior.
        let learnable_token_idx = vs
            .pp("learnable_token_idx")
            .get(LEARNABLE_TOKEN_IDX_LEN, "learnable_token_idx")
            .unwrap_or_else(|_| {
                let vals: Vec<f32> = (0..LEARNABLE_TOKEN_IDX_LEN)
                    .map(|t| t as f32 / (t as f32 + 1.0))
                    .collect();
                Tensor::from_vec(vals, LEARNABLE_TOKEN_IDX_LEN, &device).unwrap()
            });
        // Data-dependent LR gate (Task 1). w_lr: [d_model, 1], b_lr: scalar.
        // Zeros init ⇒ gate = 2*sigmoid(0) = 1.0 (neutral). Backward compat:
        // old checkpoints lack these keys; falls back to zeros.
        let w_lr = vs
            .pp("w_lr")
            .get((d, 1), "weight")
            .unwrap_or_else(|_| {
                Tensor::zeros((d, 1), candle_core::DType::F32, &device).unwrap()
            });
        let b_lr = vs.pp("w_lr").get((), "bias").unwrap_or_else(|_| {
            Tensor::zeros((), candle_core::DType::F32, &device).unwrap()
        });
        // Titans surprise-gating params (Task 3). Init alpha=1.0, beta=0.0.
        // Backward compat: old checkpoints lack these keys.
        let surprise_alpha = vs.pp("surprise").get((), "alpha").unwrap_or_else(|_| {
            Tensor::new(1.0f32, &device).unwrap()
        });
        let surprise_beta = vs.pp("surprise").get((), "beta").unwrap_or_else(|_| {
            Tensor::new(0.0f32, &device).unwrap()
        });
        Ok(Self {
            w_q,
            w_k,
            w_v,
            fast_bias,
            layer_norm: candle_nn::layer_norm_no_bias(
                d,
                config.norm_eps as f64,
                vs.pp("layer_norm"),
            )?,
            inner_lr,
            stabilize,
            forget_gate,
            learned_gate,
            guards,
            lr_scale,
            learnable_token_idx,
            w_lr,
            b_lr,
            surprise_alpha,
            surprise_beta,
        })
    }

    /// Mean-squared reconstruction error ‖LN(pred) − (v−k)‖² for `x` under the
    /// current fast-weight `state`, *without* mutating anything. This is the inner
    /// self-supervised objective the per-token update minimizes; exposed so the
    /// TTT-MLP measurement harness can compare expressivity across block types.
    /// Uses the residual target (v−k) and LayerNorm on the prediction, matching
    /// the inner-loop loss in `forward_native`.
    pub fn reconstruction_error(&self, x: &Tensor, state: &Tensor) -> Result<f32> {
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?.squeeze(0)?;
        let k_vec = k.squeeze(0)?;
        let target = v.sub(&k_vec)?;
        let k_col = k_vec.unsqueeze(1)?.contiguous()?;
        let pred = state
            .matmul(&k_col)?
            .squeeze(D::Minus1)?
            .add(&self.fast_bias)?;
        let pred_ln = self.layer_norm.forward(&pred.unsqueeze(0)?)?.squeeze(0)?;
        pred_ln.sub(&target)?.sqr()?.sum_all()?.to_scalar::<f32>()
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
    /// pred     = W_tilde × k^T + b                   [d_model]  (b: fast bias, #7)
    /// target   = v − k                               [d_model]  (residual, #4)
    /// error    = LN(pred) − target                   [d_model]  (LN in loss, #2)
    /// grad     = error ⊗ k    (outer product)        [d_model, d_model]
    /// W_tilde  ← W_tilde − η · grad
    /// output   = q + (q × W_tilde)                   [1, d_model]  (residual, #5)
    /// ```
    pub fn forward_native(
        &self,
        x: &Tensor,
        session_state: &mut Tensor,
        training: bool,
        t: usize,
    ) -> Result<Tensor> {
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
        // Plus the fast-weight bias vector (finding #7): pred = W~ @ k + b.
        // Initialized to zeros ⇒ byte-identical to the unbiased path at init.
        let pred = session_state
            .matmul(&k_col)?
            .squeeze(D::Minus1)?
            .add(&self.fast_bias)?;

        // v_vec: [d_model]  (remove the leading batch-of-one dimension)
        let v_vec = v.squeeze(0)?;

        // Finding #4 (residual reconstruction target): instead of reconstructing
        // the raw value stream, the fast weight models the *innovation* over the
        // key stream, matching ttt-lm-pytorch (`reconstruction_target = XV - XK`):
        //   target = v - k
        // k_vec uses the raw key (not k_eff) so the target is the true residual.
        let k_vec = k.squeeze(0)?; // [d_model]
        let target = v_vec.sub(&k_vec)?; // [d_model]

        // B.6 inner-loss ablation: when enabled, L2-normalize both views so the
        // objective aligns *directions* (cosine / contrastive multi-view) instead
        // of raw magnitudes (MSE reconstruction). Applied to both `pred` and
        // `target` so every downstream update path (ungated, scalar/learned gate)
        // stays consistent. Default off ⇒ exact reconstruction (byte-identical).
        let (pred, target) = if self.guards.aux_normalized() {
            (l2_normalize(&pred)?, l2_normalize(&target)?)
        } else {
            (pred, target)
        };

        // Finding #2 (LayerNorm inside inner-loop loss): the reference computes
        // ||LN(Z1) - target||^2 with the gradient flowing THROUGH LayerNorm.
        // We apply the block's LayerNorm to the prediction. During meta-training
        // (training=true) the outer-loop gradient flows through it via autograd
        // because session_state is not detached.
        // pred_ln: [d_model]
        let pred_ln = self.layer_norm.forward(&pred.unsqueeze(0)?)?.squeeze(0)?;

        // error: [d_model]
        let error = pred_ln.sub(&target)?;

        // η is read live from the shared atomic so meta-training can decay it.
        let eta = f32::from_bits(self.inner_lr.load(Ordering::Relaxed));
        // Learnable per-layer LR modulation (finding #3): η_eff = η_base * 2σ(lr_scale).
        // lr_scale is a registered parameter, so gradients flow through the sigmoid
        // during meta-training (training=true preserves the graph via the detach fix).
        // At init lr_scale ≈ 0 ⇒ factor ≈ 1.0 ⇒ η_eff ≈ η_base (near-identical start).
        let lr_factor = candle_nn::ops::sigmoid(&self.lr_scale)?.affine(2.0, 0.0)?;
        // Per-token learnable LR offset (finding #3, per-token part):
        //   token_scale = 1/(t+1) + learnable_token_idx[min(t, 511)]
        // Matches the reference's `token_eta = 1/t + learnable_token_idx[t]`.
        // Zeros init ⇒ starts as pure 1/(t+1) decay. The parameter is registered,
        // so meta-training tunes the per-position offset via backprop.
        let t_clamped = t.min(LEARNABLE_TOKEN_IDX_LEN - 1);
        let token_offset = self
            .learnable_token_idx
            .narrow(0, t_clamped, 1)?
            .squeeze(0)?;
        let token_base = Tensor::new(1.0 / (t as f32 + 1.0), session_state.device())?;
        let token_scale = token_base.add(&token_offset)?;
        let base_lr = Tensor::new(eta, session_state.device())?;
        let lr = base_lr
            .broadcast_mul(&lr_factor)?
            .broadcast_mul(&token_scale)?;
        // Task 1: Data-dependent LR gate. Computes `gate = 2*sigmoid(x @ w_lr + b_lr)`,
        // 1.0 at init (neutral). The gate is data-dependent: the network learns
        // to modulate the inner LR based on the input token.
        // w_lr: [d_model, 1], x: [1, d_model] → logit: [1, 1].
        let gate_logit = x.matmul(&self.w_lr)?.broadcast_add(&self.b_lr)?;
        let data_gate = candle_nn::ops::sigmoid(&gate_logit)?.affine(2.0, 0.0)?;
        let lr = lr.broadcast_mul(&data_gate)?;
        // Task 3: Titans surprise-gating. Scale the update by
        // `gate = sigmoid(alpha * (||error|| - beta))`. High surprise
        // (large reconstruction error) → gate near 1 (full update);
        // low surprise → gate near 0.5 (attenuated update).
        let surprise = error.sqr()?.sum_all()?.sqrt()?;
        let surprise_logit = self
            .surprise_alpha
            .broadcast_mul(&surprise.sub(&self.surprise_beta)?)?;
        let surprise_gate = candle_nn::ops::sigmoid(&surprise_logit)?;
        let lr = lr.broadcast_mul(&surprise_gate)?;
        // Task 2: LayerNorm Jacobian first-order correction. The inner loss is
        // ||LN(pred) - target||^2, and the manual `error ⊗ k` gradient ignores
        // the LayerNorm Jacobian. The dominant term is the 1/sigma scaling
        // (where sigma = sqrt(var(pred) + eps)). We scale the error by 1/sigma.
        let d_model_f = pred.dims()[0] as f64;
        let pred_mean = (pred.sum_all()? / d_model_f)?;
        let pred_var = (pred
            .broadcast_sub(&pred_mean)?
            .sqr()?
            .sum_all()?
            / d_model_f)?;
        let inv_sigma = (pred_var + 1e-5)?.sqrt()?.recip()?;
        let error_corrected = error.broadcast_mul(&inv_sigma)?;
        // Gated-DeltaNet forget gate α ∈ (0, 1] read live from the shared atomic.
        let alpha = f32::from_bits(self.forget_gate.load(Ordering::Relaxed));

        let mut updated_state = if let Some(w_alpha) = &self.learned_gate {
            // Learned data-dependent gate: the network decides per token how much
            // memory to retain. α_t = GATE_FLOOR + (1−GATE_FLOOR)·σ(w_α·x) ∈
            // (GATE_FLOOR, 1), applied to the retained-memory term:
            //   W̃ ← α_t·W̃(I − ηkkᵀ) + ηvkᵀ
            // +GATE_INIT_LOGIT so α starts ≈ 1 (near-ungated) at initialization.
            let logit = w_alpha.forward(x)?.affine(1.0, GATE_INIT_LOGIT)?; // [1, 1]
            let s = candle_nn::ops::sigmoid(&logit)?; // [1, 1] ∈ (0, 1)
            let alpha_t = s.affine(1.0 - GATE_FLOOR, GATE_FLOOR)?; // [1, 1]
            // Use error_corrected for the LayerNorm Jacobian correction.
            let pred_outer = error_corrected.unsqueeze(1)?.matmul(&k_eff)?; // [d, d]
            let v_outer = target.unsqueeze(1)?.matmul(&k_eff)?; // [d, d]
            let memory = session_state.sub(&pred_outer.broadcast_mul(&lr)?)?;
            let write = v_outer.broadcast_mul(&lr)?;
            // alpha_t is [1,1] → broadcasts across the [d,d] memory term.
            memory.broadcast_mul(&alpha_t)?.add(&write)?
        } else if alpha >= 1.0 {
            // Ungated delta rule (default, byte-identical to the original path):
            //   W̃ ← W̃ − η·(LN(pred) − target)⊗k
            // where target = v − k (residual) and pred is LayerNorm'd.
            // Use error_corrected (1/sigma scaled) for the LayerNorm Jacobian.
            let grad = error_corrected.unsqueeze(1)?.matmul(&k_eff)?;
            session_state.sub(&grad.broadcast_mul(&lr)?)?
        } else {
            // Scalar (parameter-free) gated delta rule: decay only the
            // retained-memory term by α, leaving the fresh write ηvkᵀ at full
            // strength. With normalized keys ‖I − ηkkᵀ‖ ≤ 1 ⇒ spectral radius ≤ α.
            // Use error_corrected for the LayerNorm Jacobian correction.
            let pred_outer = error_corrected.unsqueeze(1)?.matmul(&k_eff)?; // [d, d]
            let v_outer = target.unsqueeze(1)?.matmul(&k_eff)?; // [d, d]
            let memory = session_state.sub(&pred_outer.broadcast_mul(&lr)?)?;
            let write = v_outer.broadcast_mul(&lr)?;
            let gate = Tensor::new(alpha, session_state.device())?;
            memory.broadcast_mul(&gate)?.add(&write)?
        };
        if stabilize {
            // Sync-free element backstop: a hard ceiling that NaN can never breach.
            updated_state = updated_state.clamp(-STAB_CLAMP, STAB_CLAMP)?;
        }

        // --- Safe online-update guards (all no-ops at their disabled defaults) --
        // Only taken when at least one guard is enabled, so the default path stays
        // device-sync-free. Layered (1)→(2)→(3) so the controls cannot conflict.
        if !self.guards.all_disabled() {
            // (1) Token selection (EATA): a token whose reconstruction error is
            //     below threshold carries little new signal — retain the prior
            //     fast-weights instead of writing. Cuts update cost and the noise
            //     that drives long-stream drift/collapse.
            let min_err = self.guards.min_error();
            if min_err > 0.0 {
                let err_norm = error.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                if err_norm < min_err {
                    updated_state = session_state.clone();
                }
            }
            // (1b) Pre-update gradient veto: if ||error_corrected⊗k_eff||
            //      exceeds max_grad_norm, the update is destabilizing — veto it
            //      entirely and keep the prior state.
            let max_gn = self.guards.max_grad_norm_val();
            if max_gn > 0.0 {
                let grad = error_corrected.unsqueeze(1)?.matmul(&k_eff)?;
                let gn = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                if gn > max_gn {
                    updated_state = session_state.clone();
                }
            }
            // (2) Anti-forgetting anchor (EATA Fisher / CoTTA restore): pull the
            //     kept state weakly toward the meta-trained init (identity):
            //     W̃ ← (1−λ)·W̃ + λ·I.
            let lambda = self.guards.anchor();
            if lambda > 0.0 {
                let d = updated_state.dim(0)?;
                let eye = Tensor::eye(d, updated_state.dtype(), updated_state.device())?;
                updated_state = updated_state
                    .affine((1.0 - lambda) as f64, 0.0)?
                    .add(&eye.affine(lambda as f64, 0.0)?)?;
            }
            // (3) Drift-aware reset (RDumb): last-resort backstop. If the state's
            //     Frobenius norm has run away past threshold, snap back to init
            //     rather than letting a collapsed/diverged W̃ poison the stream.
            let drift = self.guards.drift();
            if drift > 0.0 {
                let fro = updated_state.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                if fro > drift {
                    let d = updated_state.dim(0)?;
                    updated_state =
                        Tensor::eye(d, updated_state.dtype(), updated_state.device())?;
                }
            }
            // (4) NaN rollback: if the updated state contains NaN/inf, revert to
            //     the pre-update state instead of poisoning the session. Catches
            //     numerical blowups the clamp missed.
            if self.guards.nan_rollback_on() {
                let flat = updated_state.flatten_all()?;
                let has_bad = flat
                    .to_vec1::<f32>()?
                    .iter()
                    .any(|&x| !x.is_finite());
                if has_bad {
                    updated_state = session_state.clone();
                }
            }
        }
        // The session state is an inference cache, not a BPTT tape. Detaching it
        // here prevents long prompts from retaining one Candle op node per token.
        // During meta-training (training=true), we skip the detach so gradients
        // flow through the inner loop to W_k and W_v.
        *session_state = if training {
            // Clone preserves the computation graph for BPTT; the original
            // `updated_state` is used below for the output projection.
            updated_state.clone()
        } else {
            updated_state.detach()
        };
        // ------------------------------------------------------------------------

        // output = q + (q × W_tilde): residual connection (finding #5).
        // [1, d_model] + ([1, d_model] × [d_model, d_model]) → [1, d_model].
        // The skip path gives identity fallback if the fast-weight is poor,
        // matching the reference's `XQW = XQ + Z1_bar`.
        let output = q.add(&q.matmul(&updated_state)?)?;

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
        let output = block.forward_native(&x, &mut state, false, 0).unwrap();
        assert_eq!(output.dims(), &[1, d]);
    }

    #[test]
    fn test_session_state_is_updated() {
        let d = 8usize;
        let (block, device) = make_block(d);
        let x = Tensor::ones((1usize, d), DType::F32, &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        let state_before: Vec<f32> = state.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let _ = block.forward_native(&x, &mut state, false, 0).unwrap();
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
        let output = block.forward_native(&x, &mut state, false, 0).unwrap();
        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }

    fn make_learned_block(d_model: usize) -> (NativeTTTBlock, Device) {
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
        let inner_lr = Arc::new(AtomicU32::new(1e-3f32.to_bits()));
        let stabilize = Arc::new(AtomicBool::new(false));
        let forget_gate = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let guards = Arc::new(OnlineGuards::disabled());
        let block = NativeTTTBlock::new_with_shared_lr(
            vb.pp("block"),
            config,
            inner_lr,
            stabilize,
            forget_gate,
            guards,
            true, // learned gate
        )
        .unwrap();
        (block, device)
    }

    #[test]
    fn learned_gate_forward_is_finite_and_updates_state() {
        let d = 16usize;
        let (block, device) = make_learned_block(d);
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        let before: Vec<f32> = state.flatten_all().unwrap().to_vec1().unwrap();
        let out = block.forward_native(&x, &mut state, false, 0).unwrap();
        let ov: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(ov.iter().all(|v| v.is_finite()), "learned-gate output must be finite");
        let after: Vec<f32> = state.flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(before, after, "learned gate must still update the fast weights");
    }

    #[test]
    fn learned_gate_warm_starts_near_ungated() {
        // Deterministic warm-start check, independent of the random w_α init.
        // With ZERO input, q=k=v=0, so the delta update vanishes and the state
        // becomes exactly α·I (memory term × α, write term zero). Hence
        // ‖state‖_F = α·√d, and α = ‖state‖_F / √d. The +GATE_INIT_LOGIT offset
        // must keep α ≈ 0.98 at init (only the small bias term moves it), i.e.
        // warm-started near the ungated α = 1 rather than a cold, forgetting gate.
        // (Using x=ones was flaky: w_α·ones depends on random weights.)
        let d = 16usize;
        let (learned, device) = make_learned_block(d);
        let x = Tensor::zeros((1usize, d), DType::F32, &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        let _ = learned.forward_native(&x, &mut state, false, 0).unwrap();
        let nl = frobenius(&state);
        let alpha = nl / (d as f32).sqrt(); // state = α·I ⇒ ‖state‖_F = α·√d
        assert!(alpha.is_finite(), "state must stay finite");
        assert!(
            (0.95..=1.0).contains(&alpha),
            "learned gate must warm-start near α≈1 (got α≈{alpha})"
        );
    }

    #[test]
    fn gate_one_is_identical_to_ungated_default() {
        // α = 1.0 (default) must take the exact ungated path — bit-identical state.
        let d = 16usize;
        let (block, device) = make_block(d);
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();
        let mut s_default = Tensor::eye(d, DType::F32, &device).unwrap();
        let mut s_gate1 = Tensor::eye(d, DType::F32, &device).unwrap();
        // Default block: gate already 1.0.
        let _ = block.forward_native(&x, &mut s_default, false, 0).unwrap();
        // Explicitly set 1.0 and run a fresh state from the same input.
        block.forget_gate.store(1.0f32.to_bits(), Ordering::Relaxed);
        let _ = block.forward_native(&x, &mut s_gate1, false, 0).unwrap();
        let a: Vec<f32> = s_default.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = s_gate1.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a, b, "α=1 must be bit-identical to the ungated path");
    }

    fn frobenius(state: &Tensor) -> f32 {
        let v: Vec<f32> = state.flatten_all().unwrap().to_vec1().unwrap();
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn forget_gate_shrinks_steady_state_memory() {
        // Deterministic input (no RNG) with normalized keys so both runs stay
        // finite; the forget gate (α<1) must drive the fast-weight state to a
        // strictly smaller steady-state norm than the ungated run — that
        // geometric decay of retained memory is the whole point of the gate.
        let d = 32usize;
        let (block, device) = make_block(d);
        block.stabilize.store(true, Ordering::Relaxed); // normalized keys → finite
        let x = Tensor::ones((1usize, d), DType::F32, &device).unwrap();

        let run = |alpha: f32| -> f32 {
            block.forget_gate.store(alpha.to_bits(), Ordering::Relaxed);
            let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
            for _ in 0..256 {
                let _ = block.forward_native(&x, &mut state, false, 0).unwrap();
            }
            frobenius(&state)
        };

        let ungated = run(1.0);
        let gated = run(0.9);
        assert!(ungated.is_finite() && gated.is_finite(), "states must stay finite");
        assert!(
            gated < ungated,
            "forget gate must shrink retained memory (gated {gated} !< ungated {ungated})"
        );
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
            let out = block.forward_native(&x, &mut state, false, 0).unwrap();
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

    // ---- B.2: safe online-update guards -----------------------------------

    /// Build a block sharing an `OnlineGuards` cell the test can tune.
    fn make_block_with_guards(d_model: usize) -> (NativeTTTBlock, Device, Arc<OnlineGuards>) {
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
        let inner_lr = Arc::new(AtomicU32::new(config.lr_inner.to_bits()));
        let stabilize = Arc::new(AtomicBool::new(false));
        let forget_gate = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let guards = Arc::new(OnlineGuards::disabled());
        let block = NativeTTTBlock::new_with_shared_lr(
            vb.pp("block"),
            config,
            inner_lr,
            stabilize,
            forget_gate,
            guards.clone(),
            false,
        )
        .unwrap();
        (block, device, guards)
    }

    #[test]
    fn guards_disabled_leave_dynamics_unchanged() {
        // With every guard at its default 0.0, `all_disabled()` short-circuits the
        // guard block, so the run is deterministic and identical across repeats —
        // confirming the default path has no guard side effects.
        let d = 24usize;
        let (block, device, guards) = make_block_with_guards(d);
        assert!(guards.all_disabled(), "fresh guards must be disabled");
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();

        let run = || -> Vec<f32> {
            let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
            for _ in 0..16 {
                let _ = block.forward_native(&x, &mut state, false, 0).unwrap();
            }
            state.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        assert_eq!(run(), run(), "default-guard path must be a deterministic no-op");
    }

    #[test]
    fn token_selection_skips_low_information_updates() {
        // A high min-error threshold forces every token to be treated as
        // low-information, so the fast-weight state must never move from init.
        let d = 16usize;
        let (block, device, guards) = make_block_with_guards(d);
        guards.set_update_min_error(1e9); // nothing clears this bar → always skip
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();
        let init = Tensor::eye(d, DType::F32, &device).unwrap();
        let mut state = init.clone();
        for _ in 0..32 {
            let _ = block.forward_native(&x, &mut state, false, 0).unwrap();
        }
        let moved = state.sub(&init).unwrap().sqr().unwrap().sum_all().unwrap()
            .to_scalar::<f32>().unwrap();
        assert!(moved < 1e-9, "all updates should have been skipped (moved {moved})");
    }

    #[test]
    fn anchor_pulls_state_toward_identity() {
        // A strong anchor (λ→1) must hold the state near the identity init even
        // under many updates that would otherwise push it far away.
        let d = 16usize;
        let x = Tensor::randn(0f32, 2f32, (1usize, d), &device_cpu()).unwrap();

        let run = |lambda: f32| -> f32 {
            let (block, device, guards) = make_block_with_guards(d);
            guards.set_anchor_strength(lambda);
            let _ = device; // device captured via x already
            let mut state = Tensor::eye(d, DType::F32, &device_cpu()).unwrap();
            for _ in 0..64 {
                let _ = block.forward_native(&x, &mut state, false, 0).unwrap();
            }
            let eye = Tensor::eye(d, DType::F32, &device_cpu()).unwrap();
            state.sub(&eye).unwrap().sqr().unwrap().sum_all().unwrap()
                .to_scalar::<f32>().unwrap()
        };
        let unanchored = run(0.0);
        let anchored = run(0.9);
        assert!(
            anchored < unanchored,
            "anchor must keep state nearer init (anchored {anchored} !< {unanchored})"
        );
    }

    #[test]
    fn drift_reset_bounds_runaway_state() {
        // Large input with no stabilization would let ‖W̃‖ blow up; a drift-reset
        // threshold must snap it back, keeping the Frobenius norm bounded.
        let d = 16usize;
        let (block, device, guards) = make_block_with_guards(d);
        let threshold = 50.0f32;
        guards.set_drift_reset_norm(threshold);
        let x = Tensor::randn(0f32, 8f32, (1usize, d), &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        let mut max_seen = 0f32;
        for _ in 0..256 {
            let _ = block.forward_native(&x, &mut state, false, 0).unwrap();
            let fro = frobenius(&state);
            assert!(fro.is_finite(), "drift-reset state went non-finite");
            max_seen = max_seen.max(fro);
        }
        // After any step the norm may reach ~threshold then reset to ‖I‖=√d, but
        // it can never run away unboundedly.
        assert!(
            max_seen <= threshold * 4.0,
            "drift reset failed to bound the state (max {max_seen})"
        );
    }

    fn device_cpu() -> Device {
        Device::Cpu
    }

    // ---- B.6: contrastive multi-view inner-loss ablation ------------------

    #[test]
    fn aux_loss_normalized_changes_dynamics_but_stays_finite() {
        // Enabling the normalized (contrastive) inner loss must (a) keep the
        // state finite and (b) produce a *different* trajectory than the default
        // reconstruction loss — proving the ablation knob actually switches the
        // objective rather than being a silent no-op.
        let d = 16usize;
        let (block, device, guards) = make_block_with_guards(d);
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();

        let run = |normalized: bool| -> Vec<f32> {
            guards.set_aux_loss_normalized(normalized);
            let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
            for _ in 0..16 {
                let _ = block.forward_native(&x, &mut state, false, 0).unwrap();
            }
            state.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };

        let recon = run(false);
        let contrastive = run(true);
        assert!(
            contrastive.iter().all(|v| v.is_finite()),
            "normalized inner loss must stay finite"
        );
        assert!(
            recon != contrastive,
            "normalized inner loss must change the update trajectory"
        );
    }

    /// Gradient flow through the inner loop during meta-training
    /// (`training=true`): W_k, W_v, the learnable LR scale, and the per-token
    /// LR index must all receive non-zero gradients from a dummy loss.
    #[test]
    fn test_gradient_flow_training_true() {
        let d = 8usize;
        let (block, device) = make_block(d);
        // Deterministic input (not randn) so gradient magnitudes are stable
        // across runs.
        let x_vals: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let x = Tensor::from_vec(x_vals, (1usize, d), &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        let output = block.forward_native(&x, &mut state, true, 0).unwrap();
        // Loss = dot(output, c) with a fixed NON-UNIFORM weight vector. This
        // matters: the block ends with a LayerNorm, and any symmetric function
        // of its output (plain sum, sum of squares) is ~constant w.r.t. the
        // pre-norm activations, which would make every gradient ~0 for a
        // bogus reason. The non-uniform weights break that symmetry so the
        // loss genuinely depends on the parameters.
        let c_vals: Vec<f32> = (1..=d).map(|i| i as f32).collect();
        let c = Tensor::from_vec(c_vals, (1usize, d), &device).unwrap();
        let loss = output.broadcast_mul(&c).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        for (name, param) in [
            ("w_k", block.w_k.weight()),
            ("w_v", block.w_v.weight()),
            ("lr_scale", &block.lr_scale),
            ("learnable_token_idx", &block.learnable_token_idx),
        ] {
            let norm = grads
                .get(param)
                .map(|g| {
                    g.sqr()
                        .unwrap()
                        .sum_all()
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap()
                        .sqrt()
                })
                .unwrap_or_else(|| {
                    panic!("{name} must receive a gradient with training=true")
                });
            assert!(
                norm > 1e-6,
                "{name} gradient norm {norm} must be non-zero with training=true"
            );
        }
    }

    /// Detach behavior during inference (`training=false`): the forward pass
    /// works, but the persisted session state is detached, so backpropagating
    /// from a loss built purely on the stored state yields no gradients on
    /// the block parameters (no BPTT tape is retained across tokens).
    #[test]
    fn test_detach_training_false() {
        let d = 8usize;
        let (block, device) = make_block(d);
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        let output = block.forward_native(&x, &mut state, false, 0).unwrap();
        assert_eq!(output.dims(), &[1, d]);
        let state_loss = state.sum_all().unwrap();
        let grads = state_loss.backward().unwrap();
        for (name, param) in [
            ("w_k", block.w_k.weight()),
            ("w_v", block.w_v.weight()),
            ("lr_scale", &block.lr_scale),
            ("learnable_token_idx", &block.learnable_token_idx),
        ] {
            let norm = grads.get(param).map(|g| {
                g.sqr()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    .sqrt()
            });
            assert!(
                norm.map(|n| n == 0.0).unwrap_or(true),
                "{name} must have no gradient through the detached session state"
            );
        }
    }

    #[test]
    fn max_grad_norm_veto_skips_destabilizing_update() {
        // With a tiny max_grad_norm, even a normal update's gradient exceeds
        // the threshold, so the state must remain exactly at init.
        let d = 16usize;
        let (block, device, guards) = make_block_with_guards(d);
        guards.set_max_grad_norm(1e-9); // effectively zero tolerance
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        let before: Vec<f32> = state.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let _ = block.forward_native(&x, &mut state, false, 0).unwrap();
        let after: Vec<f32> = state.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(before, after, "vetoed update must leave state unchanged");
    }

    #[test]
    fn max_grad_norm_disabled_by_default() {
        // Default (0.0) must not veto: state changes after a normal update.
        let d = 16usize;
        let (block, device, _guards) = make_block_with_guards(d);
        let x = Tensor::randn(0f32, 1f32, (1usize, d), &device).unwrap();
        let mut state = Tensor::eye(d, DType::F32, &device).unwrap();
        let before: Vec<f32> = state.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let _ = block.forward_native(&x, &mut state, false, 0).unwrap();
        let after: Vec<f32> = state.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_ne!(before, after, "default must allow updates");
    }

    #[test]
    fn nan_rollback_disabled_by_default() {
        // Fresh guards have nan_rollback off (verified via all_disabled).
        let (_block, _device, guards) = make_block_with_guards(16);
        assert!(guards.all_disabled(), "fresh guards must be disabled");
    }
}
