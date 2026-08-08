//! Context economics: how much context to supply, decided independently of
//! which model will receive it.
//!
//! This module deliberately knows nothing about model names. Its inputs are a
//! context size and the set of reduction methods that were applied; its output
//! is a report. The separation is the point: [`crate::model_router`] answers
//! "which model has the required capability", this module answers "how much
//! context does that model need", and neither may read the other's answer.
//!
//! Two guards encode the project's stated principle that a shorter context
//! which drops information required for correctness is a *failed* optimization:
//!
//!   * [`should_compress`] — do not pay compression cost on a context that is
//!     already small enough to forward as-is (a no-op reduction is waste, and
//!     every rewrite risks a cache break).
//!   * [`preserves_evidence`] — a candidate reduction that would drop a class of
//!     information the downstream turn needs is rejected, and the caller
//!     forwards the larger, correct context instead.
//!
//! Token counts here are **estimates**, not measurements. The upstream provider
//! tokenizes with its own vocabulary, which this process does not have. Every
//! field carrying an estimate is suffixed `_est` and [`ContextReport::estimated`]
//! is always true, so a reader can never mistake these for billed counts.

use serde::{Deserialize, Serialize};

/// Contexts at or below this many estimated tokens are forwarded untouched.
/// Below this size the reduction cannot repay its own risk: rewriting bytes can
/// break an upstream prompt cache, which costs far more than the few tokens a
/// small context could shed.
pub const NOOP_THRESHOLD_TOKENS: usize = 200;

/// Classes of information a reduction must not silently drop. A reduction that
/// fails to carry one of these forward is rejected by [`preserves_evidence`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    /// Where the task currently stands.
    TaskState,
    /// Observations the conclusion rests on.
    Evidence,
    /// Properties that must hold.
    Invariants,
    /// What this work depends on.
    Dependencies,
    /// Limits the solution must respect.
    Constraints,
    /// Material the grounding check will verify against.
    Grounding,
}

impl EvidenceClass {
    /// Every class a reduction is accountable for.
    pub const ALL: [EvidenceClass; 6] = [
        EvidenceClass::TaskState,
        EvidenceClass::Evidence,
        EvidenceClass::Invariants,
        EvidenceClass::Dependencies,
        EvidenceClass::Constraints,
        EvidenceClass::Grounding,
    ];
}

/// A reduction method that was applied to the context. Names match the shipped
/// pipeline stages so a log line can be traced back to the code that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReductionMethod {
    /// Structural skeletonization (signatures kept, bodies recoverable).
    Skeleton,
    /// Heavy `tool_result` replaced by a digest plus a recoverable stub.
    Digest,
    /// Lossless dedup of repeated system-prefix blocks.
    PrefixDiet,
    /// Tool schemas deferred outside the working set.
    ToolDefer,
    /// Content restored from the L2 content-addressed store.
    L2Reuse,
}

impl ReductionMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            ReductionMethod::Skeleton => "skeleton",
            ReductionMethod::Digest => "digest",
            ReductionMethod::PrefixDiet => "prefix_diet",
            ReductionMethod::ToolDefer => "tool_defer",
            ReductionMethod::L2Reuse => "l2_reuse",
        }
    }
}

/// What the context pipeline did, independent of which model was selected.
///
/// Pair this with a [`crate::model_router::RoutingDecision`] to describe a turn
/// completely: the decision says *who*, this report says *how much*.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextReport {
    /// Estimated tokens before any reduction.
    pub tokens_in_est: usize,
    /// Estimated tokens actually forwarded.
    pub tokens_out_est: usize,
    /// Reduction methods applied, in the order the pipeline applied them.
    pub methods: Vec<ReductionMethod>,
    /// Whether any content was served from the L2 store rather than re-sent.
    pub cache_used: bool,
    /// Whether the evidence guard admitted this reduction.
    pub evidence_preserved: bool,
    /// Always true. Token figures are local estimates, never billed counts.
    /// `skip_deserializing` pins the invariant: an inbound payload cannot set
    /// it false and claim these are billed figures.
    #[serde(default = "always_true", skip_deserializing)]
    pub estimated: bool,
}

fn always_true() -> bool {
    true
}

impl ContextReport {
    /// A report for a context forwarded untouched.
    pub fn untouched(tokens_est: usize) -> Self {
        Self {
            tokens_in_est: tokens_est,
            tokens_out_est: tokens_est,
            methods: Vec::new(),
            cache_used: false,
            evidence_preserved: true,
            estimated: true,
        }
    }

    /// Estimated tokens removed. Saturating: a reduction that grew the context
    /// reports zero saved rather than underflowing.
    pub fn tokens_saved_est(&self) -> usize {
        self.tokens_in_est.saturating_sub(self.tokens_out_est)
    }

    /// Estimated fraction removed, in `0.0..=1.0`. Zero when nothing came in.
    pub fn reduction_ratio_est(&self) -> f64 {
        if self.tokens_in_est == 0 {
            return 0.0;
        }
        self.tokens_saved_est() as f64 / self.tokens_in_est as f64
    }

    /// True when no reduction method ran.
    pub fn is_noop(&self) -> bool {
        self.methods.is_empty()
    }

    /// One-line telemetry rendering. Deliberately carries no model name -- a
    /// reader must not be able to infer model selection from a context line.
    pub fn telemetry_line(&self) -> String {
        let methods: Vec<&str> = self.methods.iter().map(|m| m.as_str()).collect();
        format!(
            "tokens_in_est={} tokens_out_est={} saved_est={} ratio_est={:.3} methods=[{}] cache_used={} evidence_preserved={} estimated={}",
            self.tokens_in_est,
            self.tokens_out_est,
            self.tokens_saved_est(),
            self.reduction_ratio_est(),
            methods.join(","),
            self.cache_used,
            self.evidence_preserved,
            self.estimated
        )
    }
}

/// Should the pipeline spend effort reducing this context?
///
/// `false` for a context already at or below [`NOOP_THRESHOLD_TOKENS`]. This is
/// a pure function of size -- it never consults the model, so the answer is the
/// same for an Opus turn and a Haiku turn with identical context.
pub fn should_compress(tokens_in_est: usize) -> bool {
    tokens_in_est > NOOP_THRESHOLD_TOKENS
}

/// Does a candidate reduction carry every required evidence class forward?
///
/// `required` is what the downstream turn needs; `carried` is what the reduction
/// actually preserved. Any missing class rejects the reduction, and the caller
/// must forward the larger, correct context. Correctness outranks token count.
pub fn preserves_evidence(required: &[EvidenceClass], carried: &[EvidenceClass]) -> bool {
    required.iter().all(|r| carried.contains(r))
}

/// The evidence classes a reduction failed to carry. Empty when it is admissible.
pub fn missing_evidence(
    required: &[EvidenceClass],
    carried: &[EvidenceClass],
) -> Vec<EvidenceClass> {
    required
        .iter()
        .copied()
        .filter(|r| !carried.contains(r))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_context_is_not_compressed() {
        assert!(!should_compress(0));
        assert!(!should_compress(NOOP_THRESHOLD_TOKENS));
        assert!(should_compress(NOOP_THRESHOLD_TOKENS + 1));
    }

    #[test]
    fn untouched_report_is_a_noop_with_zero_savings() {
        let r = ContextReport::untouched(120);
        assert!(r.is_noop());
        assert_eq!(r.tokens_saved_est(), 0);
        assert_eq!(r.reduction_ratio_est(), 0.0);
        assert!(r.evidence_preserved);
        assert!(r.estimated, "token figures must always be labeled estimates");
    }

    #[test]
    fn reduction_ratio_is_computed_from_estimates() {
        let r = ContextReport {
            tokens_in_est: 100_000,
            tokens_out_est: 20_000,
            methods: vec![ReductionMethod::Skeleton, ReductionMethod::Digest],
            cache_used: true,
            evidence_preserved: true,
            estimated: true,
        };
        assert_eq!(r.tokens_saved_est(), 80_000);
        assert!((r.reduction_ratio_est() - 0.8).abs() < 1e-9);
        assert!(!r.is_noop());
    }

    #[test]
    fn a_reduction_that_grew_the_context_reports_zero_not_underflow() {
        let r = ContextReport {
            tokens_in_est: 10,
            tokens_out_est: 25,
            methods: vec![ReductionMethod::Skeleton],
            cache_used: false,
            evidence_preserved: true,
            estimated: true,
        };
        assert_eq!(r.tokens_saved_est(), 0);
        assert_eq!(r.reduction_ratio_est(), 0.0);
    }

    #[test]
    fn evidence_guard_admits_a_complete_reduction() {
        let required = [EvidenceClass::TaskState, EvidenceClass::Grounding];
        let carried = [
            EvidenceClass::TaskState,
            EvidenceClass::Grounding,
            EvidenceClass::Constraints,
        ];
        assert!(preserves_evidence(&required, &carried));
        assert!(missing_evidence(&required, &carried).is_empty());
    }

    #[test]
    fn evidence_guard_rejects_a_reduction_that_drops_required_information() {
        let required = [
            EvidenceClass::TaskState,
            EvidenceClass::Evidence,
            EvidenceClass::Grounding,
        ];
        let carried = [EvidenceClass::TaskState];
        assert!(
            !preserves_evidence(&required, &carried),
            "dropping required evidence must fail the guard, however many tokens it saves"
        );
        let missing = missing_evidence(&required, &carried);
        assert_eq!(missing.len(), 2);
        assert!(missing.contains(&EvidenceClass::Evidence));
        assert!(missing.contains(&EvidenceClass::Grounding));
    }

    #[test]
    fn telemetry_line_never_mentions_a_model() {
        let r = ContextReport {
            tokens_in_est: 100_000,
            tokens_out_est: 12_000,
            methods: vec![ReductionMethod::Skeleton],
            cache_used: false,
            evidence_preserved: true,
            estimated: true,
        };
        let line = r.telemetry_line();
        for token in ["opus", "sonnet", "haiku", "claude", "model"] {
            assert!(
                !line.to_ascii_lowercase().contains(token),
                "context telemetry must stay model-agnostic, found {token:?} in {line:?}"
            );
        }
        assert!(line.contains("estimated=true"));
    }
}
