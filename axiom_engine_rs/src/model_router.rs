//! Model routing: which model has the capability a turn requires.
//!
//! This module answers *who* runs a turn. It must not answer *how much context*
//! that turn receives -- that is [`crate::context_economics`]. The two are kept
//! apart deliberately: the failure mode this prevents is treating "the context
//! is large" as a reason to use a weaker model, which trades correctness for
//! tokens without ever saying so.
//!
//! # Two concerns, explicitly separated
//!
//! **Capability selection** picks the tier a task needs. The capability is
//! **supplied by the caller / orchestrator** -- this module contains no
//! complexity classifier and never infers a capability from prompt length,
//! keywords, or context size. When no capability is supplied, capability
//! selection does nothing.
//!
//! **Economic downgrade** is the shipped PSS R1 behavior, preserved verbatim
//! under its own named mode ([`RoutingMode::EconomicDowngrade`]). Subscription
//! metering is per-bucket and the top tiers have their own scarce weekly buckets
//! (e.g. `seven_day_opus`, far tighter than Sonnet's), so moving a *mechanical*
//! high-tier turn to Haiku relieves the tightest bucket -- worth more than the
//! raw token-weight saving. Caches are model-scoped, so a Haiku turn cannot
//! destroy the top-tier cache; it simply cannot read it.
//!
//! A downgrade is a **cost** decision, never a context one.
//! [`RoutingDecision::economic_downgrade`] marks it as such so telemetry can
//! never present it as compression savings.
//!
//! Routing is confined to Anthropic Claude models via [`is_claude`]: an
//! arbitrary identifier like `openai-fable-5` is never rewritten, in any mode.
//!
//! # Modes (`AXIOM_MODEL_ROUTE`)
//!
//! | value | mode | behavior |
//! |---|---|---|
//! | `off`, empty | [`RoutingMode::Off`] | never rewrite the model |
//! | `auto` (default) | `EconomicDowngrade { high_tier_only: true }` | shipped behavior |
//! | `on` | `EconomicDowngrade { high_tier_only: false }` | shipped behavior, any Claude tier |
//! | `capability` | [`RoutingMode::CapabilitySelection`] | opt-in: select the tier the caller asked for |
//!
//! **`capability` replaces the economic path rather than layering on it.** It
//! ignores `mechanical` and `cooldown`, never sets `economic_downgrade`, and
//! keeps the requested model when no capability is declared. An operator who
//! switches from `auto` to `capability` therefore gives up the per-turn quota
//! relief `auto` provides on undeclared turns; that is the trade, and it is
//! deliberate -- mixing the two would reintroduce exactly the conflation this
//! module exists to remove.
//!
//! `off` / `auto` / `on` are behaviorally identical to the pre-existing
//! implementation when no capability is supplied, which is the backward-compat
//! contract every existing caller relies on.
//!
//! See docs/superpowers/plans/2026-07-11-prolonged-session-stack.md, step P4.

use serde::{Deserialize, Serialize};

/// The downgrade target: the cheapest current Claude tier.
pub const ROUTE_TARGET: &str = "claude-haiku-4-5";

/// Tier for mechanical, low-complexity, high-volume execution.
pub const MECHANICAL_TIER: &str = ROUTE_TARGET;
/// Tier for general reasoning, implementation, and analysis.
pub const GENERAL_TIER: &str = "claude-sonnet-5";
/// Tier for architecture, ambiguity, and high failure cost.
pub const REASONING_TIER: &str = "claude-opus-4-8";

/// Is `model` an Anthropic Claude model (the only routable set)? Matches the
/// `claude-` family plus the bare tier-prefixed aliases the API also accepts.
pub fn is_claude(model: &str) -> bool {
    model.contains("claude")
        || model.starts_with("opus-")
        || model.starts_with("sonnet-")
        || model.starts_with("haiku-")
        || model.starts_with("fable-")
        || model.starts_with("mythos-")
}

/// Is `model` a scarce high tier (Opus / Fable / Mythos) worth routing in
/// `auto` mode? Non-Claude ids are never high-tier.
pub fn is_high_tier(model: &str) -> bool {
    is_claude(model)
        && (model.contains("opus-4") || model.contains("fable-5") || model.contains("mythos-5"))
}

/// The reasoning capability a turn requires, **as declared by the caller**.
///
/// Ordered `Mechanical < General < Reasoning < HighRiskReasoning`, so a caller
/// can ask "does the selected tier clear the floor?" without naming a model.
///
/// `Reasoning` and `HighRiskReasoning` map to the same tier but are *not* the
/// same value: `HighRiskReasoning` additionally forbids economic downgrade in
/// every mode and is reported distinctly, which is the point of keeping it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Mechanical execution: no fresh reasoning, nothing to diagnose.
    Mechanical,
    /// General reasoning: implementation, analysis, bounded research.
    General,
    /// Architecture, decomposition, ambiguity.
    Reasoning,
    /// Reasoning where being wrong is expensive or hard to reverse.
    HighRiskReasoning,
}

impl Capability {
    /// The tier that satisfies this capability.
    pub fn tier(self) -> &'static str {
        match self {
            Capability::Mechanical => MECHANICAL_TIER,
            Capability::General => GENERAL_TIER,
            Capability::Reasoning | Capability::HighRiskReasoning => REASONING_TIER,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Capability::Mechanical => "mechanical",
            Capability::General => "general",
            Capability::Reasoning => "reasoning",
            Capability::HighRiskReasoning => "high_risk_reasoning",
        }
    }

    /// Parse a caller-supplied capability (header value, body field, MCP arg).
    /// Accepts snake_case, kebab-case, and any casing. Unknown values return
    /// `None` -- an unrecognized label is treated as "not supplied" rather than
    /// silently guessing a tier.
    pub fn parse(value: &str) -> Option<Capability> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "mechanical" => Some(Capability::Mechanical),
            "general" => Some(Capability::General),
            "reasoning" => Some(Capability::Reasoning),
            "high_risk_reasoning" => Some(Capability::HighRiskReasoning),
            _ => None,
        }
    }
}

/// How the router is allowed to act this turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingMode {
    /// Never rewrite the model.
    Off,
    /// Shipped PSS R1 cost behavior. `high_tier_only` distinguishes `auto`
    /// (scarce tiers only) from `on` (any Claude tier).
    EconomicDowngrade { high_tier_only: bool },
    /// Select the tier matching the caller-supplied capability, up or down.
    CapabilitySelection,
}

impl RoutingMode {
    /// Parse the `AXIOM_MODEL_ROUTE` value. Unknown values fall back to `Off`,
    /// the safe direction: an unrecognized mode never rewrites a model.
    pub fn parse(value: &str) -> RoutingMode {
        match value.trim() {
            "auto" => RoutingMode::EconomicDowngrade {
                high_tier_only: true,
            },
            "on" => RoutingMode::EconomicDowngrade {
                high_tier_only: false,
            },
            "capability" => RoutingMode::CapabilitySelection,
            _ => RoutingMode::Off,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RoutingMode::Off => "off",
            RoutingMode::EconomicDowngrade {
                high_tier_only: true,
            } => "economic_downgrade_high_tier",
            RoutingMode::EconomicDowngrade {
                high_tier_only: false,
            } => "economic_downgrade_any",
            RoutingMode::CapabilitySelection => "capability_selection",
        }
    }
}

/// Per-turn routing inputs.
///
/// `capability` is the caller's declaration. `mechanical` and `cooldown` are the
/// pre-existing PSS R1 signals, kept so existing callers behave unchanged.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TurnSignals {
    /// Caller-supplied capability. `None` means the orchestrator did not
    /// declare one; this module will not guess.
    pub capability: Option<Capability>,
    /// The newest turn is composed entirely of `tool_result` blocks and carries
    /// no error signature.
    pub mechanical: bool,
    /// Sticky post-error cooldown, in turns remaining.
    pub cooldown: u32,
}

/// Why the router decided as it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingReason {
    /// Mode is `off`.
    Disabled,
    /// Not an Anthropic Claude id; never rewritten.
    NotClaude,
    /// Capability mode, but the caller supplied no capability. No guess is made.
    CapabilityNotSupplied,
    /// The requested model already satisfies the requirement.
    CapabilitySatisfied,
    /// The declared capability outranks a downgrade target; the model is held.
    CapabilityFloorHeld,
    /// Capability mode selected the tier for the declared capability.
    CapabilityMatched,
    /// Mode admits only scarce high tiers and this is not one.
    ModeExcludesTier,
    /// Already on the cheapest tier; there is nothing to downgrade to. Distinct
    /// from `CapabilitySatisfied`, which means a *declared* capability matched.
    AlreadyCheapestTier,
    /// Not a mechanical turn (fresh reasoning, or an error cooldown is active).
    NotMechanical,
    /// Mechanical turn moved to the cheap tier to relieve a scarce quota bucket.
    MechanicalDowngrade,
}

impl RoutingReason {
    pub fn as_str(self) -> &'static str {
        match self {
            RoutingReason::Disabled => "disabled",
            RoutingReason::NotClaude => "not_claude",
            RoutingReason::CapabilityNotSupplied => "capability_not_supplied",
            RoutingReason::CapabilitySatisfied => "capability_satisfied",
            RoutingReason::CapabilityFloorHeld => "capability_floor_held",
            RoutingReason::CapabilityMatched => "capability_matched",
            RoutingReason::ModeExcludesTier => "mode_excludes_tier",
            RoutingReason::AlreadyCheapestTier => "already_cheapest_tier",
            RoutingReason::NotMechanical => "not_mechanical",
            RoutingReason::MechanicalDowngrade => "mechanical_downgrade",
        }
    }
}

/// A routing outcome: which model, and why.
///
/// Carries **no** context or token figures. Pair it with a
/// [`crate::context_economics::ContextReport`] to describe a turn completely.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub original_model: String,
    pub selected_model: String,
    /// What the caller declared, if anything. Never inferred.
    pub capability: Option<Capability>,
    pub mode: RoutingMode,
    pub reason: RoutingReason,
    /// True only for a cost-motivated move to the cheap tier. This is the field
    /// that keeps a downgrade from ever being counted as context compression.
    pub economic_downgrade: bool,
    pub changed: bool,
}

impl RoutingDecision {
    /// One-line telemetry. Deliberately carries no token figures, so a reader
    /// cannot infer context reduction from a routing line.
    ///
    /// The model fields are rendered with `{:?}`: `original_model` is
    /// client-controlled, and a value containing a newline or space would
    /// otherwise break the `key=value` framing and let a caller forge a second
    /// `[axiom-routing]` log line. Debug formatting escapes control characters.
    pub fn telemetry_line(&self) -> String {
        format!(
            "original_model={:?} selected_model={:?} capability={} mode={} reason={} economic_downgrade={} changed={}",
            self.original_model,
            self.selected_model,
            self.capability.map(Capability::as_str).unwrap_or("none"),
            self.mode.as_str(),
            self.reason.as_str(),
            self.economic_downgrade,
            self.changed
        )
    }
}

/// Select the model for a turn.
///
/// With `signals.capability == None` the `off` / `auto` / `on` modes reproduce
/// the shipped economic-downgrade behavior exactly.
pub fn select_model(requested: &str, signals: &TurnSignals, mode: RoutingMode) -> RoutingDecision {
    let keep = |reason: RoutingReason| RoutingDecision {
        original_model: requested.to_string(),
        selected_model: requested.to_string(),
        capability: signals.capability,
        mode,
        reason,
        economic_downgrade: false,
        changed: false,
    };

    if mode == RoutingMode::Off {
        return keep(RoutingReason::Disabled);
    }
    if !is_claude(requested) {
        return keep(RoutingReason::NotClaude);
    }

    match mode {
        RoutingMode::Off => keep(RoutingReason::Disabled),

        RoutingMode::CapabilitySelection => {
            // No classifier: without a declared capability there is nothing to
            // select on, so the model is left exactly as the caller sent it.
            let Some(cap) = signals.capability else {
                return keep(RoutingReason::CapabilityNotSupplied);
            };
            let target = cap.tier();
            if target == requested {
                return keep(RoutingReason::CapabilitySatisfied);
            }
            RoutingDecision {
                original_model: requested.to_string(),
                selected_model: target.to_string(),
                capability: Some(cap),
                mode,
                reason: RoutingReason::CapabilityMatched,
                economic_downgrade: false,
                changed: true,
            }
        }

        RoutingMode::EconomicDowngrade { high_tier_only } => {
            // A declared capability above Mechanical vetoes the cost move. This
            // is what stops a large-context reasoning turn from being cheapened.
            if let Some(cap) = signals.capability {
                if cap > Capability::Mechanical {
                    return keep(RoutingReason::CapabilityFloorHeld);
                }
            }
            if !signals.mechanical || signals.cooldown != 0 {
                return keep(RoutingReason::NotMechanical);
            }
            if requested.contains("haiku") {
                return keep(RoutingReason::AlreadyCheapestTier);
            }
            if high_tier_only && !is_high_tier(requested) {
                return keep(RoutingReason::ModeExcludesTier);
            }
            RoutingDecision {
                original_model: requested.to_string(),
                selected_model: ROUTE_TARGET.to_string(),
                capability: signals.capability,
                mode,
                reason: RoutingReason::MechanicalDowngrade,
                economic_downgrade: true,
                changed: true,
            }
        }
    }
}

/// Decide the routed model for a turn, or `None` to leave it untouched.
///
/// The pre-existing R1 interface, preserved. Now a thin wrapper over
/// [`select_model`] with no declared capability, so the legacy path and the new
/// one cannot drift apart.
pub fn route(model: &str, mechanical: bool, cooldown: u32, mode: &str) -> Option<&'static str> {
    let signals = TurnSignals {
        capability: None,
        mechanical,
        cooldown,
    };
    let decision = select_model(model, &signals, RoutingMode::parse(mode));
    if decision.economic_downgrade {
        Some(ROUTE_TARGET)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAPABILITY: RoutingMode = RoutingMode::CapabilitySelection;
    const AUTO: RoutingMode = RoutingMode::EconomicDowngrade {
        high_tier_only: true,
    };
    const ON: RoutingMode = RoutingMode::EconomicDowngrade {
        high_tier_only: false,
    };

    fn declared(cap: Capability) -> TurnSignals {
        TurnSignals {
            capability: Some(cap),
            ..TurnSignals::default()
        }
    }

    fn mechanical_turn() -> TurnSignals {
        TurnSignals {
            mechanical: true,
            ..TurnSignals::default()
        }
    }

    // ---- backward compatibility: the shipped `route` interface is unchanged ----

    #[test]
    fn auto_mode_routes_only_high_tier_mechanical_turns() {
        assert_eq!(route("claude-opus-4-8", true, 0, "auto"), Some("claude-haiku-4-5"));
        assert_eq!(route("claude-sonnet-5", true, 0, "auto"), None);
        assert_eq!(route("claude-fable-5", true, 0, "auto"), Some("claude-haiku-4-5"));
        assert_eq!(route("claude-opus-4-8", false, 0, "auto"), None); // hard turn
        assert_eq!(route("claude-opus-4-8", true, 2, "auto"), None); // cooldown
        assert_eq!(route("claude-haiku-4-5", true, 0, "auto"), None); // never touch haiku
    }

    #[test]
    fn on_mode_routes_any_claude_but_never_non_claude() {
        assert_eq!(route("claude-sonnet-5", true, 0, "on"), Some("claude-haiku-4-5"));
        assert_eq!(route("openai-fable-5", true, 0, "on"), None);
        assert_eq!(route("gpt-5", true, 0, "on"), None);
    }

    #[test]
    fn off_mode_never_routes() {
        assert_eq!(route("claude-opus-4-8", true, 0, "off"), None);
        assert_eq!(route("claude-opus-4-8", true, 0, ""), None);
    }

    #[test]
    fn is_high_tier_matches_opus_and_fable_not_sonnet() {
        assert!(is_high_tier("claude-opus-4-8"));
        assert!(is_high_tier("claude-fable-5"));
        assert!(!is_high_tier("claude-sonnet-5"));
        assert!(!is_high_tier("claude-haiku-4-5"));
        assert!(!is_high_tier("openai-fable-5"), "non-Claude fable is not high-tier");
    }

    #[test]
    fn is_claude_rejects_foreign_providers() {
        assert!(is_claude("claude-opus-4-8"));
        assert!(is_claude("opus-4-8"));
        assert!(is_claude("fable-5"));
        assert!(!is_claude("openai-fable-5"));
        assert!(!is_claude("gpt-5"));
        assert!(!is_claude("gemini-2"));
    }

    // ---- Cases A-D: declared capability selects the tier ----

    #[test]
    fn case_a_mechanical_selects_haiku() {
        let d = select_model("claude-opus-4-8", &declared(Capability::Mechanical), CAPABILITY);
        assert_eq!(d.selected_model, "claude-haiku-4-5");
        assert_eq!(d.reason, RoutingReason::CapabilityMatched);
        assert!(!d.economic_downgrade, "a capability match is not a cost move");
    }

    #[test]
    fn case_b_general_selects_sonnet() {
        let d = select_model("claude-haiku-4-5", &declared(Capability::General), CAPABILITY);
        assert_eq!(d.selected_model, "claude-sonnet-5");
        assert_eq!(d.reason, RoutingReason::CapabilityMatched);
    }

    #[test]
    fn case_c_reasoning_selects_opus() {
        let d = select_model("claude-haiku-4-5", &declared(Capability::Reasoning), CAPABILITY);
        assert_eq!(d.selected_model, "claude-opus-4-8");
    }

    #[test]
    fn case_d_high_risk_reasoning_selects_opus() {
        let d = select_model(
            "claude-sonnet-5",
            &declared(Capability::HighRiskReasoning),
            CAPABILITY,
        );
        assert_eq!(d.selected_model, "claude-opus-4-8");
        assert_eq!(d.capability, Some(Capability::HighRiskReasoning));
    }

    #[test]
    fn reasoning_and_high_risk_share_a_tier_but_remain_distinct_values() {
        assert_eq!(Capability::Reasoning.tier(), Capability::HighRiskReasoning.tier());
        assert_ne!(Capability::Reasoning, Capability::HighRiskReasoning);
        assert!(Capability::HighRiskReasoning > Capability::Reasoning);
    }

    // ---- the veto: declared reasoning is never downgraded for cost ----

    #[test]
    fn declared_reasoning_vetoes_economic_downgrade_in_every_mode() {
        for cap in [
            Capability::General,
            Capability::Reasoning,
            Capability::HighRiskReasoning,
        ] {
            for mode in [AUTO, ON] {
                let signals = TurnSignals {
                    capability: Some(cap),
                    mechanical: true, // would otherwise qualify for downgrade
                    cooldown: 0,
                };
                let d = select_model("claude-opus-4-8", &signals, mode);
                assert!(
                    !d.changed && !d.economic_downgrade,
                    "{cap:?} in {mode:?} must not be downgraded"
                );
                assert_eq!(d.reason, RoutingReason::CapabilityFloorHeld);
            }
        }
    }

    #[test]
    fn declared_mechanical_still_permits_the_economic_downgrade() {
        // The boundary of the veto: it is `> Mechanical`, not `>= Mechanical`.
        // Without this, changing the comparison to `>=` would silently disable
        // the shipped cost path and every other test would still pass.
        let signals = TurnSignals {
            capability: Some(Capability::Mechanical),
            mechanical: true,
            cooldown: 0,
        };
        let d = select_model("claude-opus-4-8", &signals, AUTO);
        assert!(d.economic_downgrade);
        assert_eq!(d.reason, RoutingReason::MechanicalDowngrade);
        assert_eq!(d.selected_model, ROUTE_TARGET);
    }

    #[test]
    fn already_haiku_is_distinct_from_a_declared_capability_match() {
        // Both keep the model, but a telemetry reader must be able to tell "no
        // cheaper tier exists" from "the caller's declared capability is met".
        let economic = select_model("claude-haiku-4-5", &mechanical_turn(), AUTO);
        assert_eq!(economic.reason, RoutingReason::AlreadyCheapestTier);
        assert_eq!(economic.capability, None);

        let declared_match =
            select_model("claude-haiku-4-5", &declared(Capability::Mechanical), CAPABILITY);
        assert_eq!(declared_match.reason, RoutingReason::CapabilitySatisfied);
        assert_ne!(economic.reason, declared_match.reason);
    }

    #[test]
    fn telemetry_line_escapes_a_forged_model_id() {
        // `original_model` is client-controlled and the line goes to stderr, so
        // a newline must not be able to fabricate a second log record.
        let signals = mechanical_turn();
        let d = select_model("claude-opus-4-8\n[axiom-routing] forged=true", &signals, AUTO);
        let line = d.telemetry_line();
        assert!(
            !line.contains('\n'),
            "a newline in a model id must not break the line framing: {line:?}"
        );
        assert!(line.contains("\\n"), "the newline should be escaped, not dropped");
    }

    #[test]
    fn no_declared_capability_means_no_guess_in_capability_mode() {
        let d = select_model("claude-opus-4-8", &mechanical_turn(), CAPABILITY);
        assert!(!d.changed, "the router must not infer a capability");
        assert_eq!(d.reason, RoutingReason::CapabilityNotSupplied);
        assert_eq!(d.capability, None);
    }

    // ---- Case J: the two concerns are separately observable ----

    #[test]
    fn case_j_capability_selection_and_economic_downgrade_are_distinguishable() {
        let cost_move = select_model("claude-opus-4-8", &mechanical_turn(), AUTO);
        assert!(cost_move.economic_downgrade);
        assert_eq!(cost_move.reason, RoutingReason::MechanicalDowngrade);
        assert_eq!(cost_move.capability, None);

        let capability_move =
            select_model("claude-haiku-4-5", &declared(Capability::Reasoning), CAPABILITY);
        assert!(
            !capability_move.economic_downgrade,
            "a capability selection must never be flagged as a cost move"
        );
        assert_eq!(capability_move.reason, RoutingReason::CapabilityMatched);

        // Both changed the model, but for reasons a reader can tell apart.
        assert!(cost_move.changed && capability_move.changed);
        assert_ne!(cost_move.reason, capability_move.reason);
    }

    #[test]
    fn routing_telemetry_carries_no_token_or_context_figures() {
        let d = select_model("claude-opus-4-8", &mechanical_turn(), AUTO);
        let line = d.telemetry_line();
        for token in ["token", "bytes", "saved", "ratio", "compress"] {
            assert!(
                !line.to_ascii_lowercase().contains(token),
                "routing telemetry must stay context-agnostic, found {token:?} in {line:?}"
            );
        }
        assert!(line.contains("economic_downgrade=true"));
        assert!(line.contains("capability=none"));
    }

    // ---- parsing ----

    #[test]
    fn capability_parses_declared_labels_and_rejects_unknown_ones() {
        assert_eq!(Capability::parse("mechanical"), Some(Capability::Mechanical));
        assert_eq!(Capability::parse("General"), Some(Capability::General));
        assert_eq!(Capability::parse("reasoning"), Some(Capability::Reasoning));
        assert_eq!(
            Capability::parse("high-risk-reasoning"),
            Some(Capability::HighRiskReasoning)
        );
        assert_eq!(
            Capability::parse("  HIGH_RISK_REASONING  "),
            Some(Capability::HighRiskReasoning)
        );
        assert_eq!(
            Capability::parse("extremely-hard"),
            None,
            "an unknown label must not be guessed into a tier"
        );
    }

    #[test]
    fn routing_mode_parses_and_falls_back_to_off() {
        assert_eq!(RoutingMode::parse("auto"), AUTO);
        assert_eq!(RoutingMode::parse("on"), ON);
        assert_eq!(RoutingMode::parse("capability"), CAPABILITY);
        assert_eq!(RoutingMode::parse("off"), RoutingMode::Off);
        assert_eq!(RoutingMode::parse(""), RoutingMode::Off);
        assert_eq!(
            RoutingMode::parse("nonsense"),
            RoutingMode::Off,
            "an unknown mode must never rewrite a model"
        );
    }
}
