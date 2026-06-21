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
    fn drift(&self) -> f32 {
        f32::from_bits(self.drift_reset_norm.load(Ordering::Relaxed))
    }
    fn min_error(&self) -> f32 {
        f32::from_bits(self.update_min_error.load(Ordering::Relaxed))
    }
    fn anchor(&self) -> f32 {
        f32::from_bits(self.anchor_strength.load(Ordering::Relaxed))
    }
    /// True when every guard is at its disabled default — lets `forward_native`
    /// skip the guard block (and its device syncs) entirely on the hot path.
    fn all_disabled(&self) -> bool {
        self.drift() == 0.0 && self.min_error() == 0.0 && self.anchor() == 0.0
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
            forget_gate,
            learned_gate,
            guards,
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

        // η is read live from the shared atomic so meta-training can decay it.
        let eta = f32::from_bits(self.inner_lr.load(Ordering::Relaxed));
        let lr = Tensor::new(eta, session_state.device())?;
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
            let pred_outer = pred.unsqueeze(1)?.matmul(&k_eff)?; // [d, d]
            let v_outer = v_vec.unsqueeze(1)?.matmul(&k_eff)?; // [d, d]
            let memory = session_state.sub(&pred_outer.broadcast_mul(&lr)?)?;
            let write = v_outer.broadcast_mul(&lr)?;
            // alpha_t is [1,1] → broadcasts across the [d,d] memory term.
            memory.broadcast_mul(&alpha_t)?.add(&write)?
        } else if alpha >= 1.0 {
            // Ungated delta rule (default, byte-identical to the original path):
            //   W̃ ← W̃ − η·(pred − v)⊗k = W̃(I − ηkkᵀ) + ηvkᵀ
            let grad = error.unsqueeze(1)?.matmul(&k_eff)?;
            session_state.sub(&grad.broadcast_mul(&lr)?)?
        } else {
            // Scalar (parameter-free) gated delta rule: decay only the
            // retained-memory term by α, leaving the fresh write ηvkᵀ at full
            // strength. With normalized keys ‖I − ηkkᵀ‖ ≤ 1 ⇒ spectral radius ≤ α.
            let pred_outer = pred.unsqueeze(1)?.matmul(&k_eff)?; // [d, d]
            let v_outer = v_vec.unsqueeze(1)?.matmul(&k_eff)?; // [d, d]
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
        let out = block.forward_native(&x, &mut state).unwrap();
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
        let _ = learned.forward_native(&x, &mut state).unwrap();
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
        let _ = block.forward_native(&x, &mut s_default).unwrap();
        // Explicitly set 1.0 and run a fresh state from the same input.
        block.forget_gate.store(1.0f32.to_bits(), Ordering::Relaxed);
        let _ = block.forward_native(&x, &mut s_gate1).unwrap();
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
                let _ = block.forward_native(&x, &mut state).unwrap();
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
                let _ = block.forward_native(&x, &mut state).unwrap();
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
            let _ = block.forward_native(&x, &mut state).unwrap();
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
                let _ = block.forward_native(&x, &mut state).unwrap();
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
            let _ = block.forward_native(&x, &mut state).unwrap();
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
}
