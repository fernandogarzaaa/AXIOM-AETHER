//! Beta-distribution beliefs — epistemic confidence with uncertainty.
//!
//! Ported and adapted from ChimeraLang's `cir/nodes.py::BetaDist`. A scalar
//! confidence cannot distinguish "succeeded 1/1" from "succeeded 50/50" — both
//! read as 1.0. A Beta belief carries the *estimate* (mean) **and** the
//! *uncertainty* (variance): little evidence ⇒ high variance, even at a high
//! mean. This is what lets immunity confidence mature honestly (an established
//! fix needs both a high mean and low variance) and wane into *uncertainty*
//! rather than a lowered estimate.
//!
//! Three operations matter for Axiom's swarm immunity:
//!  * [`BetaBelief::reinforce`] / [`BetaBelief::penalize`] — accumulate evidence.
//!  * [`BetaBelief::combine_ds`] — Dempster-Shafer evidence combination with
//!    *conflict detection*: irreconcilable peers raise [`DsConflict`] instead of
//!    silently averaging to mush.
//!  * [`BetaBelief::decayed`] — regress toward the uniform prior `Beta(1,1)` as a
//!    belief ages (staleness becomes uncertainty).

use serde::{Deserialize, Serialize};

/// Mean threshold above which a belief is "confident" (matches ChimeraLang's
/// guard: `mean >= 1 - max_risk` with max_risk ≈ 0.4 → 0.6).
pub const ESTABLISHED_MEAN: f32 = 0.6;
/// Variance ceiling for an *established* belief (ChimeraLang's guard variance).
pub const ESTABLISHED_VARIANCE: f32 = 0.05;
/// DS conflict mass above which two sources are deemed irreconcilable.
pub const DS_CONFLICT_THRESHOLD: f32 = 0.8;

/// Raised when two beliefs are too contradictory to combine.
#[derive(Debug, Clone, PartialEq)]
pub struct DsConflict {
    pub conflict_mass: f32,
    pub mean_a: f32,
    pub mean_b: f32,
}

/// A belief as `Beta(alpha, beta)`: alpha ~ successes, beta ~ failures.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BetaBelief {
    pub alpha: f32,
    pub beta: f32,
}

impl Default for BetaBelief {
    fn default() -> Self {
        Self::uniform()
    }
}

impl BetaBelief {
    /// The uniform prior `Beta(1,1)` — maximum uncertainty, mean 0.5.
    pub fn uniform() -> Self {
        Self { alpha: 1.0, beta: 1.0 }
    }

    /// Build from a scalar confidence in [0,1] with a pseudocount `strength`
    /// (higher strength ⇒ more evidence ⇒ lower variance). Used to migrate
    /// legacy scalar-confidence records.
    pub fn from_confidence(conf: f32, strength: f32) -> Self {
        let conf = conf.clamp(1e-6, 1.0 - 1e-6);
        let strength = strength.max(2.0);
        Self {
            alpha: conf * strength,
            beta: (1.0 - conf) * strength,
        }
    }

    pub fn mean(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
    }

    pub fn variance(&self) -> f32 {
        let s = self.alpha + self.beta;
        (self.alpha * self.beta) / (s * s * (s + 1.0))
    }

    /// Total evidence (pseudocount mass).
    pub fn evidence(&self) -> f32 {
        self.alpha + self.beta
    }

    /// A belief is *established* only with both a high mean and low variance —
    /// so a single 1/1 success (high mean, high variance) is not yet trusted.
    pub fn is_established(&self) -> bool {
        self.mean() >= ESTABLISHED_MEAN && self.variance() <= ESTABLISHED_VARIANCE
    }

    /// Human label spanning the evidence/uncertainty space.
    pub fn label(&self) -> &'static str {
        if self.is_established() {
            "established"
        } else if self.mean() >= ESTABLISHED_MEAN {
            "proven" // high mean but still uncertain (needs more evidence)
        } else if self.mean() >= 0.3 {
            "tentative"
        } else {
            "faded"
        }
    }

    /// Add one success pseudocount.
    pub fn reinforce(&mut self) {
        self.alpha += 1.0;
    }

    /// Add one failure pseudocount.
    pub fn penalize(&mut self) {
        self.beta += 1.0;
    }

    /// Regress toward the uniform prior by `factor` ∈ [0,1] (0 = unchanged,
    /// 1 = fully uniform). Increases variance — staleness as uncertainty.
    pub fn decayed(&self, factor: f32) -> Self {
        let f = factor.clamp(0.0, 1.0);
        Self {
            alpha: self.alpha * (1.0 - f) + 1.0 * f,
            beta: self.beta * (1.0 - f) + 1.0 * f,
        }
    }

    /// Dempster-Shafer-inspired evidence combination. Detects conflict (one
    /// source strongly yes, the other strongly no) and returns [`DsConflict`]
    /// rather than fabricating a midpoint. This is the swarm-merge trust rule:
    /// agreeing peers compound evidence; irreconcilable peers are flagged.
    pub fn combine_ds(&self, other: &BetaBelief) -> Result<BetaBelief, DsConflict> {
        let t1 = self.alpha + self.beta;
        let t2 = other.alpha + other.beta;
        let (m1_yes, m1_no) = (self.alpha / t1, self.beta / t1);
        let (m2_yes, m2_no) = (other.alpha / t2, other.beta / t2);
        let conflict = m1_yes * m2_no + m1_no * m2_yes;
        if conflict > DS_CONFLICT_THRESHOLD {
            return Err(DsConflict {
                conflict_mass: conflict,
                mean_a: self.mean(),
                mean_b: other.mean(),
            });
        }
        Ok(BetaBelief {
            alpha: (self.alpha + other.alpha - 1.0).max(1e-6),
            beta: (self.beta + other.beta - 1.0).max(1e-6),
        })
    }

    /// True when the belief's parameters are well-formed and within sane bounds.
    /// Byzantine defence: a peer cannot inject NaN/∞/negative or absurdly large
    /// pseudocounts (fabricated certainty) into a swarm merge.
    pub fn is_plausible(&self, max_evidence: f32) -> bool {
        self.alpha.is_finite()
            && self.beta.is_finite()
            && self.alpha > 0.0
            && self.beta > 0.0
            && self.evidence() <= max_evidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_confidence_clamps_extremes_and_stays_finite() {
        for c in [0.0f32, 1.0, -5.0, 9.0] {
            let b = BetaBelief::from_confidence(c, 10.0);
            assert!(b.alpha > 0.0 && b.beta > 0.0);
            assert!(b.mean().is_finite() && b.variance().is_finite());
        }
    }

    #[test]
    fn uncertainty_separates_one_success_from_many() {
        let mut few = BetaBelief::uniform();
        few.reinforce(); // 1/1 → Beta(2,1): high mean, high variance
        let mut many = BetaBelief::uniform();
        for _ in 0..10 {
            many.reinforce();
        }
        assert!(few.mean() > 0.6 && !few.is_established(), "1 success is not established");
        assert!(many.is_established(), "many successes → established");
        assert!(many.variance() < few.variance(), "more evidence → less uncertainty");
    }

    #[test]
    fn decay_regresses_toward_uniform() {
        let mut b = BetaBelief::uniform();
        for _ in 0..20 {
            b.reinforce();
        }
        assert!(b.is_established());
        let stale = b.decayed(1.0);
        assert!((stale.mean() - 0.5).abs() < 1e-5, "full decay → uniform mean");
        assert!(!stale.is_established(), "fully decayed belief is no longer trusted");
        assert!(stale.variance() > b.variance(), "decay raises uncertainty");
    }

    #[test]
    fn ds_combination_compounds_agreement() {
        let a = BetaBelief { alpha: 4.0, beta: 1.0 };
        let b = BetaBelief { alpha: 5.0, beta: 1.0 };
        let c = a.combine_ds(&b).unwrap();
        assert!(c.evidence() > a.evidence(), "agreeing evidence compounds");
        assert!(c.mean() > 0.8);
    }

    #[test]
    fn ds_combination_flags_conflict() {
        let yes = BetaBelief { alpha: 9.0, beta: 1.0 }; // strongly yes
        let no = BetaBelief { alpha: 1.0, beta: 9.0 }; // strongly no
        let err = yes.combine_ds(&no).unwrap_err();
        assert!(err.conflict_mass > DS_CONFLICT_THRESHOLD);
    }

    #[test]
    fn plausibility_rejects_byzantine_values() {
        assert!(BetaBelief { alpha: 5.0, beta: 2.0 }.is_plausible(1000.0));
        assert!(!BetaBelief { alpha: f32::NAN, beta: 1.0 }.is_plausible(1000.0));
        assert!(!BetaBelief { alpha: -1.0, beta: 1.0 }.is_plausible(1000.0));
        assert!(!BetaBelief { alpha: 1e9, beta: 1.0 }.is_plausible(1000.0), "fabricated certainty rejected");
    }
}
