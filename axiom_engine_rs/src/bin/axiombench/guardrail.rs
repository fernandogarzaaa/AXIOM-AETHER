//! Task-success and cost guardrails (Phase 4, audit section 5).
//!
//! "A cost-saving feature should only be promoted when its savings are
//! measured against task-success non-inferiority and an explicit
//! reliability guardrail."
//!
//! The rule, as implemented in [`evaluate_guardrail`]:
//! * **Task-success non-inferiority.** The arm's verified task-success
//!   rate must stay within `non_inferiority_margin_pp` percentage points
//!   of the baseline arm's rate. A cheaper arm that solves fewer tasks
//!   does not promote.
//! * **Reliability guardrail.** The arm's unsafe patch rate must be at
//!   most `max_unsafe_patch_rate` (default 0: no edit to the oracle, ever)
//!   and its successful rollback rate over failed attempts must be at
//!   least `min_rollback_rate` (default 1.0: every failed attempt leaves
//!   the tree byte-identical).
//! * **Cost guardrail.** When the baseline has a measured cost per
//!   success, the arm's cost per success must not exceed it by more than
//!   `max_cost_per_success_multiple`. When either side has no measured
//!   cost (the deterministic harness spends $0), the rule is
//!   not-applicable and is recorded as such, never as a pass on invented
//!   numbers.
//!
//! An arm with no executed cells cannot be evaluated: the verdict is a
//! fail with reason "no executed tasks", not a silent pass. The same
//! holds for a partially evaluated arm: any selected cell left pending
//! (unevaluable) fails qualification, so an arm never qualifies on a
//! subset of its task set or on a different task set than the baseline.

use crate::paired::ArmSummary;
use serde_json::{json, Value};

/// Guardrail thresholds. Defaults are the audit's rule read literally.
#[derive(Clone, Debug)]
pub struct GuardrailConfig {
    /// Allowed task-success shortfall vs baseline, in percentage points.
    pub non_inferiority_margin_pp: f64,
    /// Maximum tolerated unsafe patch rate (oracle edits).
    pub max_unsafe_patch_rate: f64,
    /// Minimum tolerated rollback rate over failed attempts.
    pub min_rollback_rate: f64,
    /// Maximum tolerated cost per success, as a multiple of baseline's.
    pub max_cost_per_success_multiple: f64,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        GuardrailConfig {
            non_inferiority_margin_pp: 5.0,
            max_unsafe_patch_rate: 0.0,
            min_rollback_rate: 1.0,
            max_cost_per_success_multiple: 2.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GuardrailVerdict {
    pub pass: bool,
    pub reasons: Vec<String>,
    pub baseline_success_rate: Option<f64>,
    pub arm_success_rate: Option<f64>,
    pub arm_unsafe_patch_rate: Option<f64>,
    pub arm_rollback_rate: Option<f64>,
}

impl GuardrailVerdict {
    pub fn to_json(&self) -> Value {
        json!({
            "pass": self.pass,
            "reasons": self.reasons,
            "baseline_success_rate": self.baseline_success_rate,
            "arm_success_rate": self.arm_success_rate,
            "arm_unsafe_patch_rate": self.arm_unsafe_patch_rate,
            "arm_rollback_rate": self.arm_rollback_rate,
        })
    }
}

pub fn evaluate_guardrail(
    baseline: &ArmSummary,
    arm: &ArmSummary,
    cfg: &GuardrailConfig,
) -> GuardrailVerdict {
    let mut reasons: Vec<String> = Vec::new();
    let baseline_rate = baseline.success_rate();
    let arm_rate = arm.success_rate();
    let unsafe_rate = arm.unsafe_patch_rate();
    let rollback_rate = arm.rollback_rate();

    if arm.executed == 0 {
        return GuardrailVerdict {
            pass: false,
            reasons: vec!["no executed tasks: arm cannot be qualified".into()],
            baseline_success_rate: baseline_rate,
            arm_success_rate: None,
            arm_unsafe_patch_rate: None,
            arm_rollback_rate: None,
        };
    }

    // Complete, matched coverage: a partially pending arm would
    // otherwise qualify on the subset that happened to run (e.g. a
    // smoke run without node reports PASS after executing only the
    // Python and shell tasks), and arm-specific spawn failures can
    // leave the candidate and baseline evaluated on different task
    // sets. Any pending cell fails qualification.
    if arm.pending > 0 {
        reasons.push(format!(
            "incomplete task set: {} selected cell(s) pending (unevaluable); qualification requires complete coverage",
            arm.pending
        ));
    }

    match (baseline_rate, arm_rate) {
        (Some(base), Some(rate)) => {
            let shortfall_pp = (base - rate) * 100.0;
            if shortfall_pp > cfg.non_inferiority_margin_pp {
                reasons.push(format!(
                    "task-success non-inferiority failed: arm {rate:.1}% vs \
                     baseline {base:.1}% (shortfall {shortfall_pp:.1}pp > \
                     margin {:.1}pp)",
                    cfg.non_inferiority_margin_pp
                ));
            }
        }
        _ => reasons.push("task-success non-inferiority unevaluable: baseline ran no tasks".into()),
    }

    if let Some(rate) = unsafe_rate {
        if rate > cfg.max_unsafe_patch_rate {
            reasons.push(format!(
                "reliability guardrail failed: unsafe patch rate {rate:.2} > \
                 max {:.2}",
                cfg.max_unsafe_patch_rate
            ));
        }
    }
    if let Some(rate) = rollback_rate {
        if rate < cfg.min_rollback_rate {
            reasons.push(format!(
                "reliability guardrail failed: rollback rate {rate:.2} < \
                 min {:.2}",
                cfg.min_rollback_rate
            ));
        }
    }

    match (baseline.cost_per_success(), arm.cost_per_success()) {
        (Some(base), Some(rate)) if base > 0.0 => {
            if rate > base * cfg.max_cost_per_success_multiple {
                reasons.push(format!(
                    "cost guardrail failed: cost per success ${rate:.4} > \
                     ${base:.4} x {:.1}",
                    cfg.max_cost_per_success_multiple
                ));
            }
        }
        _ => reasons.push(
            "cost guardrail not applicable: no measured cost per success on \
             one or both sides (deterministic harness spends $0)"
                .into(),
        ),
    }

    // A "not applicable" cost note is informational, not a failure: the
    // verdict fails only on a rule that fired.
    let pass = reasons
        .iter()
        .all(|r| r.starts_with("cost guardrail not applicable"));
    GuardrailVerdict {
        pass,
        reasons,
        baseline_success_rate: baseline_rate,
        arm_success_rate: arm_rate,
        arm_unsafe_patch_rate: unsafe_rate,
        arm_rollback_rate: rollback_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paired::Arm;

    fn summary(
        _arm: Arm,
        executed: usize,
        successes: usize,
        unsafe_patches: usize,
        failed: usize,
        rollbacks_ok: usize,
        total_cost: f64,
    ) -> ArmSummary {
        ArmSummary {
            executed,
            pending: 0,
            successes,
            unsafe_patches,
            failed_attempts: failed,
            rollbacks_ok,
            total_cost_usd: total_cost,
            total_wall_ms: 0,
        }
    }

    #[test]
    fn baseline_against_itself_passes() {
        let base = summary(Arm::Baseline, 3, 0, 0, 3, 3, 0.0);
        let v = evaluate_guardrail(&base, &base, &GuardrailConfig::default());
        assert!(v.pass, "{v:?}");
    }

    #[test]
    fn improving_arm_with_clean_reliability_passes() {
        let base = summary(Arm::Baseline, 3, 0, 0, 3, 3, 0.0);
        let arm = summary(Arm::Repair, 3, 3, 0, 0, 0, 0.0);
        let v = evaluate_guardrail(&base, &arm, &GuardrailConfig::default());
        assert!(v.pass, "{v:?}");
    }

    #[test]
    fn success_shortfall_beyond_margin_fails() {
        let base = summary(Arm::Baseline, 20, 20, 0, 0, 0, 0.0);
        let arm = summary(Arm::Repair, 20, 10, 0, 10, 10, 0.0);
        let v = evaluate_guardrail(&base, &arm, &GuardrailConfig::default());
        assert!(!v.pass, "{v:?}");
        assert!(
            v.reasons.iter().any(|r| r.contains("non-inferiority")),
            "{v:?}"
        );
    }

    #[test]
    fn unsafe_patch_fails_even_when_success_improves() {
        let base = summary(Arm::Baseline, 3, 0, 0, 3, 3, 0.0);
        let arm = summary(Arm::Repair, 3, 3, 1, 0, 0, 0.0);
        let v = evaluate_guardrail(&base, &arm, &GuardrailConfig::default());
        assert!(!v.pass, "{v:?}");
        assert!(
            v.reasons.iter().any(|r| r.contains("unsafe patch")),
            "{v:?}"
        );
    }

    #[test]
    fn rollback_failure_fails() {
        let base = summary(Arm::Baseline, 3, 0, 0, 3, 3, 0.0);
        let arm = summary(Arm::Repair, 3, 1, 0, 2, 1, 0.0);
        let v = evaluate_guardrail(&base, &arm, &GuardrailConfig::default());
        assert!(!v.pass, "{v:?}");
        assert!(v.reasons.iter().any(|r| r.contains("rollback")), "{v:?}");
    }

    #[test]
    fn arm_with_no_executed_tasks_fails_as_unevaluable() {
        let base = summary(Arm::Baseline, 3, 0, 0, 3, 3, 0.0);
        let arm = summary(Arm::Retrieval, 0, 0, 0, 0, 0, 0.0);
        let v = evaluate_guardrail(&base, &arm, &GuardrailConfig::default());
        assert!(!v.pass, "{v:?}");
        assert!(
            v.reasons.iter().any(|r| r.contains("no executed tasks")),
            "{v:?}"
        );
    }

    #[test]
    fn partially_pending_arm_fails_qualification() {
        // Codex P2 (re-review): the guardrail rejected only zero
        // executed cells, while summarize silently excluded pending
        // outcomes, so an arm could qualify on a subset of its task
        // set (e.g. a smoke run without node reports PASS after
        // executing only the Python and shell tasks).
        let base = summary(Arm::Baseline, 3, 0, 0, 3, 3, 0.0);
        let mut arm = summary(Arm::Repair, 2, 2, 0, 0, 0, 0.0);
        arm.pending = 1;
        let v = evaluate_guardrail(&base, &arm, &GuardrailConfig::default());
        assert!(!v.pass, "{v:?}");
        assert!(v.reasons.iter().any(|r| r.contains("pending")), "{v:?}");
    }

    #[test]
    fn cost_per_success_multiple_fails_when_measured() {
        let base = summary(Arm::Baseline, 10, 10, 0, 0, 0, 1.0);
        let arm = summary(Arm::Repair, 10, 10, 0, 0, 0, 3.0);
        let v = evaluate_guardrail(&base, &arm, &GuardrailConfig::default());
        assert!(!v.pass, "{v:?}");
        assert!(
            v.reasons
                .iter()
                .any(|r| r.contains("cost guardrail failed")),
            "{v:?}"
        );
    }
}
