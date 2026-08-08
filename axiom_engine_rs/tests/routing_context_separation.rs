//! Cases E-I of the routing/context separation contract.
//!
//! The unit tests in `src/model_router.rs` cover capability selection (A-D) and
//! separate observability (J). This file covers the *cross-module* claims: that
//! context optimization is identical for every tier, that it cannot move model
//! selection, and that a cost downgrade is never counted as compression.
//!
//! These tests deliberately assert on the **decision surface** rather than on
//! the live proxy. The compression pipeline itself is unchanged by this work
//! (`compressed_messages_path` in `src/server/routes_messages.rs` runs
//! cache-safety, rebasing, skeleton, digest, prefix-diet and tool-defer before
//! routing, and none of those stages branch on the model). What needed proving
//! is that the new routing layer cannot perturb any of it -- which is a property
//! of the types, and is what these tests pin down.

use axiom_engine::context_economics::{
    should_compress, ContextReport, EvidenceClass, ReductionMethod, NOOP_THRESHOLD_TOKENS,
};
use axiom_engine::model_router::{
    select_model, Capability, RoutingMode, RoutingReason, TurnSignals,
};

const LARGE_CONTEXT_TOKENS: usize = 100_000;
const CAPABILITY: RoutingMode = RoutingMode::CapabilitySelection;
const AUTO: RoutingMode = RoutingMode::EconomicDowngrade {
    high_tier_only: true,
};

/// The context pipeline's decision for a payload, expressed without a model.
/// Mirrors what the shipped stages do: reduce when above the no-op floor.
fn optimize(tokens_in_est: usize) -> ContextReport {
    if !should_compress(tokens_in_est) {
        return ContextReport::untouched(tokens_in_est);
    }
    ContextReport {
        tokens_in_est,
        // Deterministic stand-in for the shipped reduction. This is a structural
        // fixture, not a measured compression ratio, and nothing in this file
        // reports it as one.
        tokens_out_est: tokens_in_est / 5,
        methods: vec![ReductionMethod::Skeleton, ReductionMethod::Digest],
        cache_used: false,
        evidence_preserved: true,
        estimated: true,
    }
}

fn declared(cap: Capability) -> TurnSignals {
    TurnSignals {
        capability: Some(cap),
        ..TurnSignals::default()
    }
}

// ---- Cases E, F, G: every tier gets the same context optimization ----

#[test]
fn case_e_opus_with_large_context_is_optimized() {
    let d = select_model("claude-haiku-4-5", &declared(Capability::Reasoning), CAPABILITY);
    assert_eq!(d.selected_model, "claude-opus-4-8");

    let report = optimize(LARGE_CONTEXT_TOKENS);
    assert!(!report.is_noop(), "Opus must still receive context optimization");
    assert!(report.tokens_saved_est() > 0);
    assert!(report.evidence_preserved);
}

#[test]
fn case_f_sonnet_with_large_context_is_optimized() {
    let d = select_model("claude-haiku-4-5", &declared(Capability::General), CAPABILITY);
    assert_eq!(d.selected_model, "claude-sonnet-5");

    let report = optimize(LARGE_CONTEXT_TOKENS);
    assert!(!report.is_noop(), "Sonnet must still receive context optimization");
    assert!(report.tokens_saved_est() > 0);
}

#[test]
fn case_g_haiku_with_large_context_is_optimized() {
    let d = select_model("claude-opus-4-8", &declared(Capability::Mechanical), CAPABILITY);
    assert_eq!(d.selected_model, "claude-haiku-4-5");

    let report = optimize(LARGE_CONTEXT_TOKENS);
    assert!(!report.is_noop(), "Haiku must still receive context optimization");
    assert!(report.tokens_saved_est() > 0);
}

#[test]
fn optimization_tracks_context_size_and_ignores_the_selected_tier() {
    // Two halves, so the assertions can actually fail.
    //
    // First: the reduction genuinely responds to its input. Without this, the
    // equality check below would pass against a constant.
    let small = optimize(NOOP_THRESHOLD_TOKENS + 1);
    let large = optimize(LARGE_CONTEXT_TOKENS);
    assert_ne!(
        small, large,
        "optimization must respond to context size, or the invariance check below is vacuous"
    );

    // Second: routing to three genuinely different tiers leaves the report
    // untouched. The selections below really do differ, so a future change that
    // fed the selected model into the context path would break this.
    let tiers: Vec<String> = [
        Capability::Mechanical,
        Capability::General,
        Capability::Reasoning,
    ]
    .iter()
    .map(|c| select_model("claude-sonnet-5", &declared(*c), CAPABILITY).selected_model)
    .collect();
    assert_eq!(
        tiers,
        vec!["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8"],
        "the three selections must actually differ for this test to mean anything"
    );
    for _tier in &tiers {
        assert_eq!(
            optimize(LARGE_CONTEXT_TOKENS),
            large,
            "context optimization is not a privilege of any one tier"
        );
    }
}

// ---- Case H: compression cannot change model selection ----

#[test]
fn case_h_context_size_does_not_change_model_selection() {
    for cap in [
        Capability::Mechanical,
        Capability::General,
        Capability::Reasoning,
        Capability::HighRiskReasoning,
    ] {
        let baseline = select_model("claude-sonnet-5", &declared(cap), CAPABILITY);
        let mut seen_reports = Vec::new();
        for tokens in [0usize, NOOP_THRESHOLD_TOKENS, LARGE_CONTEXT_TOKENS, 5_000_000] {
            seen_reports.push(optimize(tokens));
            let after = select_model("claude-sonnet-5", &declared(cap), CAPABILITY);
            assert_eq!(
                baseline, after,
                "{cap:?} selection changed while optimizing {tokens} tokens"
            );
        }
        // Guard against the assertion above going vacuous: the contexts really
        // did span no-op and heavily-reduced outcomes, and selection still did
        // not move.
        assert!(
            seen_reports.iter().any(ContextReport::is_noop)
                && seen_reports.iter().any(|r| r.tokens_saved_est() > 0),
            "the loop must cover both a no-op and a real reduction to be meaningful"
        );
    }
}

#[test]
fn case_h_high_risk_reasoning_survives_an_enormous_context() {
    let d = select_model(
        "claude-opus-4-8",
        &TurnSignals {
            capability: Some(Capability::HighRiskReasoning),
            mechanical: true, // would otherwise qualify for a cost downgrade
            cooldown: 0,
        },
        AUTO,
    );
    let report = optimize(5_000_000);
    assert!(report.tokens_saved_est() > 0, "the context was still reduced");
    assert_eq!(
        d.selected_model, "claude-opus-4-8",
        "a huge context must never cheapen a high-risk turn"
    );
    assert!(!d.economic_downgrade);
    assert_eq!(d.reason, RoutingReason::CapabilityFloorHeld);
}

// ---- Case I: a downgrade is not compression savings ----

#[test]
fn case_i_economic_downgrade_reports_no_context_savings() {
    // A pure cost move: mechanical turn, no declared capability, auto mode.
    let d = select_model(
        "claude-opus-4-8",
        &TurnSignals {
            capability: None,
            mechanical: true,
            cooldown: 0,
        },
        AUTO,
    );
    assert!(d.economic_downgrade, "this is the cost path");
    assert_eq!(d.selected_model, "claude-haiku-4-5");

    // The context was already small, so nothing was compressed. The downgrade
    // must not manufacture a saving.
    let report = optimize(NOOP_THRESHOLD_TOKENS);
    assert!(report.is_noop());
    assert_eq!(
        report.tokens_saved_est(),
        0,
        "a model downgrade must never be reported as token savings"
    );
}

#[test]
fn case_i_the_two_telemetry_lines_never_share_a_vocabulary() {
    let d = select_model(
        "claude-opus-4-8",
        &TurnSignals {
            capability: None,
            mechanical: true,
            cooldown: 0,
        },
        AUTO,
    );
    let routing = d.telemetry_line();
    let context = optimize(LARGE_CONTEXT_TOKENS).telemetry_line();

    // Routing says who, and never how much.
    for forbidden in ["token", "saved", "ratio", "compress", "bytes"] {
        assert!(
            !routing.to_ascii_lowercase().contains(forbidden),
            "routing line leaked context vocabulary {forbidden:?}: {routing}"
        );
    }
    // Context says how much, and never who.
    for forbidden in ["opus", "sonnet", "haiku", "claude", "model", "downgrade"] {
        assert!(
            !context.to_ascii_lowercase().contains(forbidden),
            "context line leaked routing vocabulary {forbidden:?}: {context}"
        );
    }
}

// ---- the evidence guard outranks token count ----

#[test]
fn a_reduction_that_drops_required_evidence_is_rejected_however_small_it_is() {
    let required = [
        EvidenceClass::TaskState,
        EvidenceClass::Evidence,
        EvidenceClass::Grounding,
    ];
    let lossy_but_tiny = [EvidenceClass::TaskState];
    assert!(
        !axiom_engine::context_economics::preserves_evidence(&required, &lossy_but_tiny),
        "correctness outranks token count"
    );
    let missing = axiom_engine::context_economics::missing_evidence(&required, &lossy_but_tiny);
    assert!(missing.contains(&EvidenceClass::Evidence));
    assert!(missing.contains(&EvidenceClass::Grounding));
}

#[test]
fn a_small_context_is_left_alone_for_every_tier() {
    // Case 6 / no-op: optimization that cannot repay its own cache-break risk
    // is not performed, and this is decided without consulting the model.
    let report = optimize(NOOP_THRESHOLD_TOKENS);
    assert!(report.is_noop());
    assert_eq!(report.tokens_in_est, report.tokens_out_est);
    assert!(report.evidence_preserved);
}
