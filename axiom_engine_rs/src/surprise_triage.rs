//! Surprise-weighted context triage.
//!
//! AXIOM's TTT inner loop computes a per-chunk gradient on the fast weights
//! during ingestion. The norm of that update, `||ΔW̃||`, is a direct,
//! model-relative measure of how much *new information* a chunk carried — the
//! model's own learning *is* the relevance metric.
//!
//! This module turns that signal into an adaptive compression-budget
//! allocator:
//!
//! - [`SurpriseTriage`] tracks a running distribution of per-chunk update
//!   norms with Welford's online algorithm (with a short warmup to discount
//!   cold-start inflation, when fast weights are unadapted and every chunk
//!   looks surprising).
//! - [`SurpriseTriage::score_chunk`] maps a chunk's norm to a percentile
//!   in `[0, 1]` via a normal CDF of the z-score.
//! - [`SurpriseTriage::budget_for`] maps the percentile to a
//!   [`CompressionLevel`]: high-surprise chunks are preserved verbatim or
//!   lightly skeletonized, low-surprise chunks are compressed aggressively.
//!   The total token budget stays fixed; only the *allocation* is adaptive.
//!
//! Integration points:
//! - [`SurpriseTriage::observe_tensors`] computes `||Δ||` from pre/post
//!   fast-weight snapshots around a TTT adaptation window (see
//!   `context_compressor::adapt_session_with_triage`).
//! - [`SurpriseTriage::triage_compress`] is the "before compressing a chunk,
//!   compute its surprise score and select compression level" entry point.
//! - `digest::SkeletonDigestor::digest_with_triage` exposes the same hook at
//!   the digest layer.

use candle_core::{Result as CResult, Tensor};

/// How aggressively a chunk should be compressed, chosen by its surprise
/// percentile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// `p >= 0.90`: the chunk taught the model a lot — keep it verbatim.
    Verbatim,
    /// `0.50 <= p < 0.90`: normal structural skeleton within budget.
    Skeleton,
    /// `p < 0.50`: the chunk carried little new signal — aggressive digest.
    Digest,
}

/// Percentile cutoffs for [`CompressionLevel`] selection.
pub const VERBATIM_PERCENTILE: f32 = 0.90;
pub const SKELETON_PERCENTILE: f32 = 0.50;

/// Default number of warmup observations before percentiles are trusted.
/// Cold-start fast weights produce inflated norms; scoring is neutral (0.5)
/// until the running distribution has seen this many chunks.
pub const DEFAULT_WARMUP: u64 = 8;

/// Minimum token budget for the aggressive [`CompressionLevel::Digest`] tier,
/// so tiny base budgets still produce a non-empty digest.
pub const MIN_DIGEST_TOKENS: usize = 32;

/// Adaptive compression-budget allocator driven by TTT update norms.
///
/// Tracks the running mean/variance of per-chunk `||ΔW̃||` with Welford's
/// algorithm and converts each new norm into a session-relative percentile.
#[derive(Debug, Clone)]
pub struct SurpriseTriage {
    count: u64,
    mean: f64,
    m2: f64,
    warmup: u64,
}

impl SurpriseTriage {
    /// Create a triage with a custom warmup length.
    pub fn new(warmup: u64) -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            warmup,
        }
    }

    /// Number of norms observed so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Current running mean of observed norms (`None` before any observation).
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    /// Current running variance of observed norms (`None` with < 2 samples).
    pub fn variance(&self) -> Option<f64> {
        (self.count > 1).then_some(self.m2 / (self.count - 1) as f64)
    }

    /// Feed one per-chunk update norm into the running distribution.
    /// Non-finite or negative norms are ignored (they carry no signal).
    pub fn observe(&mut self, norm: f32) {
        if !norm.is_finite() || norm < 0.0 {
            return;
        }
        let x = norm as f64;
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Compute the total fast-weight update norm `||Δ||` across layers from
    /// pre/post adaptation snapshots, feed it to the running distribution,
    /// and return it.
    ///
    /// `before`/`after` are parallel slices of per-layer `W̃` tensors;
    /// the norm is `sqrt(Σ_layers ||after - before||_F²)`.
    pub fn observe_tensors(&mut self, before: &[Tensor], after: &[Tensor]) -> CResult<f32> {
        let mut sum_sq = 0.0f32;
        for (b, a) in before.iter().zip(after.iter()) {
            let diff = a.sub(b)?;
            let sq: f32 = diff.sqr()?.sum_all()?.to_scalar::<f32>()?;
            sum_sq += sq;
        }
        let norm = sum_sq.sqrt();
        self.observe(norm);
        Ok(norm)
    }

    /// Map a chunk's update norm to a percentile in `[0, 1]`.
    ///
    /// Uses the normal CDF of the z-score against the running distribution,
    /// so the score is session-relative: the same chunk scores differently
    /// depending on what the session already learned. Returns `0.5`
    /// (neutral) during warmup or when the variance is degenerate.
    pub fn score_chunk(&self, norm: f32) -> f32 {
        if self.count < self.warmup {
            return 0.5;
        }
        let var = match self.variance() {
            Some(v) if v > 0.0 => v,
            _ => return 0.5,
        };
        let std = var.sqrt();
        let z = (norm as f64 - self.mean) / std;
        normal_cdf(z as f32).clamp(0.0, 1.0)
    }

    /// Select a [`CompressionLevel`] and token budget for a chunk, given its
    /// update norm and the session's base budget.
    ///
    /// The percentile comes from [`SurpriseTriage::score_chunk`]; the total
    /// budget is unchanged in expectation — only its allocation adapts.
    pub fn budget_for(&self, norm: f32, base_budget_tokens: usize) -> (CompressionLevel, usize) {
        let p = self.score_chunk(norm);
        if p >= VERBATIM_PERCENTILE {
            (CompressionLevel::Verbatim, usize::MAX)
        } else if p >= SKELETON_PERCENTILE {
            (CompressionLevel::Skeleton, base_budget_tokens)
        } else {
            (
                CompressionLevel::Digest,
                (base_budget_tokens / 4).max(MIN_DIGEST_TOKENS),
            )
        }
    }

    /// End-to-end triage entry point: observe `update_norm`, score the chunk,
    /// pick a compression level, and compress `text` accordingly.
    ///
    /// - [`CompressionLevel::Verbatim`]: text returned unchanged.
    /// - [`CompressionLevel::Skeleton`]: PageRank-ranked skeleton within
    ///   `base_budget_tokens`.
    /// - [`CompressionLevel::Digest`]: aggressive short digest (quarter
    ///   budget, floored at [`MIN_DIGEST_TOKENS`]).
    ///
    /// Returns the compressed text and the level that was chosen.
    pub fn triage_compress(
        &mut self,
        text: &str,
        base_budget_tokens: usize,
        update_norm: f32,
        lang: &str,
    ) -> (String, CompressionLevel) {
        self.observe(update_norm);
        let (level, budget) = self.budget_for(update_norm, base_budget_tokens);
        let out = match level {
            CompressionLevel::Verbatim => text.to_string(),
            CompressionLevel::Skeleton => {
                let ranked =
                    crate::skeleton::skeletonize_ranked(text, lang, Some(budget));
                if ranked.trim().is_empty() {
                    aggressive_truncate(text, budget)
                } else {
                    ranked
                }
            }
            CompressionLevel::Digest => aggressive_truncate(text, budget),
        };
        (out, level)
    }
}

impl Default for SurpriseTriage {
    fn default() -> Self {
        Self::new(DEFAULT_WARMUP)
    }
}

/// Standard normal CDF via the Abramowitz–Stegun erf approximation
/// (max error ~1.5e-7 — plenty for a percentile heuristic).
fn normal_cdf(z: f32) -> f32 {
    // Φ(z) = 0.5 * (1 + erf(z / √2))
    let x = z / std::f32::consts::SQRT_2;
    0.5 * (1.0 + erf_approx(x))
}

fn erf_approx(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26
    let a1 = 0.2548296f32;
    let a2 = -0.2844967f32;
    let a3 = 1.4214137f32;
    let a4 = -1.453152f32;
    let a5 = 1.0614054f32;
    let p = 0.3275911f32;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Hard word-count truncation used by the aggressive digest tier.
fn aggressive_truncate(text: &str, budget_tokens: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= budget_tokens {
        return text.to_string();
    }
    let mut out = words[..budget_tokens].join(" ");
    out.push_str(" …");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_converges_to_known_mean_and_variance() {
        // Population: 2, 4, 4, 4, 5, 5, 7, 9 → mean 5,
        // sample variance Σ(x−mean)²/(n−1) = 32/7.
        let mut t = SurpriseTriage::new(0);
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            t.observe(x);
        }
        assert_eq!(t.count(), 8);
        assert!((t.mean().unwrap() - 5.0).abs() < 1e-9);
        assert!((t.variance().unwrap() - 32.0 / 7.0).abs() < 1e-9);
    }

    #[test]
    fn observe_ignores_non_finite_and_negative_norms() {
        let mut t = SurpriseTriage::new(0);
        t.observe(f32::NAN);
        t.observe(f32::INFINITY);
        t.observe(-1.0);
        assert_eq!(t.count(), 0);
        t.observe(1.0);
        assert_eq!(t.count(), 1);
    }

    #[test]
    fn score_is_neutral_during_warmup() {
        let mut t = SurpriseTriage::new(8);
        for _ in 0..7 {
            t.observe(1.0);
        }
        assert_eq!(t.score_chunk(1000.0), 0.5);
        assert_eq!(t.score_chunk(0.0), 0.5);
    }

    #[test]
    fn score_is_neutral_with_degenerate_variance() {
        // All identical norms → zero variance → neutral score.
        let mut t = SurpriseTriage::new(0);
        for _ in 0..10 {
            t.observe(3.0);
        }
        assert_eq!(t.score_chunk(3.0), 0.5);
        assert_eq!(t.score_chunk(99.0), 0.5);
    }

    #[test]
    fn high_norm_scores_above_low_norm() {
        let mut t = SurpriseTriage::new(0);
        // Bimodal-ish spread so z-scores separate clearly.
        for _ in 0..20 {
            t.observe(1.0);
        }
        for _ in 0..20 {
            t.observe(10.0);
        }
        let low = t.score_chunk(1.0);
        let high = t.score_chunk(10.0);
        assert!(high > low, "high={high} low={low}");
        assert!(high > 0.5);
        assert!(low < 0.5);
    }

    /// Feed a tight cluster around 5.0 so extreme norms map to extreme
    /// percentiles (|z| >> 1.28 for the 0.90 cutoff).
    fn tight_triage() -> SurpriseTriage {
        let mut t = SurpriseTriage::new(0);
        for i in 0..100 {
            t.observe(if i % 2 == 0 { 4.0 } else { 6.0 });
        }
        t
    }

    #[test]
    fn budget_for_maps_percentiles_to_levels() {
        let t = tight_triage();
        // Very high norm → z ≈ +94 → percentile ≈ 1.0 → Verbatim.
        let (level, budget) = t.budget_for(100.0, 400);
        assert_eq!(level, CompressionLevel::Verbatim);
        assert_eq!(budget, usize::MAX);
        // Very low norm → z ≈ −5 → percentile ≈ 0 → aggressive digest.
        let (level, budget) = t.budget_for(0.001, 400);
        assert_eq!(level, CompressionLevel::Digest);
        assert_eq!(budget, 100);
    }

    #[test]
    fn budget_for_skeleton_tier_uses_base_budget() {
        let mut t = SurpriseTriage::new(0);
        for _ in 0..100 {
            t.observe(5.0);
        }
        // A norm near the mean scores ~0.5 → Skeleton tier.
        let (level, budget) = t.budget_for(5.0, 400);
        assert_eq!(level, CompressionLevel::Skeleton);
        assert_eq!(budget, 400);
    }

    #[test]
    fn digest_budget_floored_at_minimum() {
        let t = tight_triage();
        let (level, budget) = t.budget_for(0.001, 40);
        assert_eq!(level, CompressionLevel::Digest);
        assert_eq!(budget, MIN_DIGEST_TOKENS);
    }

    #[test]
    fn triage_compress_verbatim_returns_text_unchanged() {
        let mut t = SurpriseTriage::new(0);
        for _ in 0..50 {
            t.observe(1.0);
        }
        let text = "pub fn hello() -> i32 { 42 }";
        let (out, level) = t.triage_compress(text, 100, 100.0, "rust");
        assert_eq!(level, CompressionLevel::Verbatim);
        assert_eq!(out, text);
    }

    #[test]
    fn triage_compress_digest_is_shorter_than_skeleton() {
        let mut t = SurpriseTriage::new(0);
        for _ in 0..50 {
            t.observe(100.0);
        }
        // Long prose: low-surprise norm → Digest tier (quarter budget).
        let text = "lorem ipsum dolor sit amet ".repeat(200);
        let (digest_out, digest_level) = t.triage_compress(&text, 400, 0.001, "");
        assert_eq!(digest_level, CompressionLevel::Digest);
        // Digest budget = 100 tokens → output well under the 400-token budget.
        assert!(digest_out.split_whitespace().count() <= 101);
        assert!(digest_out.len() < text.len());
    }

    #[test]
    fn triage_compress_observes_norm_into_distribution() {
        let mut t = SurpriseTriage::new(0);
        assert_eq!(t.count(), 0);
        let _ = t.triage_compress("hello world", 100, 2.5, "");
        assert_eq!(t.count(), 1);
        assert!((t.mean().unwrap() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn normal_cdf_sanity() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-5);
        assert!(normal_cdf(3.0) > 0.998);
        assert!(normal_cdf(-3.0) < 0.002);
        assert!(normal_cdf(10.0) <= 1.0);
        assert!(normal_cdf(-10.0) >= 0.0);
    }
}
