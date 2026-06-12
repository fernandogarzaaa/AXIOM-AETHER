//! Confidence-gated adaptive compression.
//!
//! Compression saves tokens by skeletonizing heavy context; but the *more*
//! novel that context is, the *less* safe it is to drop detail — the model
//! can't reconstruct what it has never seen, and Claude needs that detail
//! verbatim to ground its answer. So we let the drift signal gate the
//! compression budget on the request path:
//!
//!   * **Predictable** context (surprisal ≤ the drift gate) keeps the aggressive
//!     base threshold — compress hard, maximal token savings.
//!   * **Surprising / novel** context (surprisal above the gate) raises the
//!     threshold so fewer messages qualify as "heavy" and more is forwarded
//!     verbatim — trading a little savings for grounding fidelity exactly where
//!     hallucination risk is highest.
//!
//! This is the request-path counterpart to grounding verification on the
//! response path: both use the same drift signal to spend tokens where they
//! matter. Honest: the signal is only as sharp as the trained model — on the
//! bootstrap model surprisal sits near `ln(vocab)`, so the gate rarely trips
//! and behaviour ≈ fixed-threshold compression.

/// The effective per-message token threshold given the base threshold, the
/// heavy context's mean surprisal (cross-entropy under the model), and the
/// drift `gate`. Surprising context scales the threshold up toward 2× as it
/// exceeds the gate; predictable context is left at `base`.
pub fn adaptive_threshold(base: usize, surprisal: f32, gate: f32) -> usize {
    if !surprisal.is_finite() || gate <= 0.0 || surprisal <= gate {
        return base;
    }
    // 0 at the gate → 1 at 2× the gate (clamped): preserve progressively more.
    let over = ((surprisal - gate) / gate).clamp(0.0, 1.0);
    let factor = 1.0 + over; // 1.0 ..= 2.0
    ((base as f32) * factor).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictable_context_keeps_base_threshold() {
        // surprisal at/below the gate → unchanged (aggressive compression).
        assert_eq!(adaptive_threshold(200, 4.0, 7.0), 200);
        assert_eq!(adaptive_threshold(200, 7.0, 7.0), 200);
    }

    #[test]
    fn surprising_context_raises_threshold_to_preserve_detail() {
        // Above the gate → higher threshold (fewer msgs compressed, more verbatim).
        let t = adaptive_threshold(200, 10.5, 7.0); // 50% over → ~1.5x
        assert!(t > 200 && t <= 400, "raised but capped: {t}");
        assert!((t as i64 - 300).abs() <= 1, "≈1.5× at 50% over the gate: {t}");
    }

    #[test]
    fn scaling_is_capped_at_2x() {
        // Far above the gate saturates at 2× — never unbounded.
        assert_eq!(adaptive_threshold(200, 100.0, 7.0), 400);
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        assert_eq!(adaptive_threshold(200, f32::NAN, 7.0), 200);
        assert_eq!(adaptive_threshold(200, 9.0, 0.0), 200);
    }
}
