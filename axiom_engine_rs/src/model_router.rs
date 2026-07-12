//! P4 (Prolonged-Session Stack): R1 high-tier-gated model routing.
//!
//! Subscription metering is per-bucket, and the top tiers have their own scarce
//! weekly buckets (e.g. `seven_day_opus`, far tighter than Sonnet's). Routing a
//! *mechanical* Opus/Fable turn -- a turn that carries no error and needs no
//! deep reasoning -- down to Haiku relieves the tightest bucket, worth more than
//! the raw token-weight saving. It is therefore gated to the high tiers and left
//! OFF for Sonnet (whose dual-cache economics make routing near-useless, per the
//! sim). Caches are model-scoped, so a Haiku turn cannot destroy the top-tier
//! cache -- it simply can't read it -- and Anthropic's own guidance endorses
//! spawning a cheaper call rather than switching the main loop's model.
//!
//! Routing is confined to Anthropic Claude models via [`is_claude`]: an
//! arbitrary identifier like `openai-fable-5` is never rewritten, even in the
//! forced `on` mode.
//!
//! See docs/superpowers/plans/2026-07-11-prolonged-session-stack.md, step P4.

/// The downgrade target: the cheapest current Claude tier.
pub const ROUTE_TARGET: &str = "claude-haiku-4-5";

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

/// Decide the routed model for a turn, or `None` to leave it untouched.
///
/// Returns `Some(ROUTE_TARGET)` only when ALL hold: `model` is a Claude model,
/// the turn is `mechanical` (no error, no fresh reasoning), the escalation
/// `cooldown` has elapsed (`== 0`), `model` is not already Haiku, and the mode
/// admits it -- `"on"` routes any Claude tier, `"auto"` routes high tiers only.
/// Any other case (`"off"`, Sonnet in `auto`, a hard turn, an active cooldown,
/// a non-Claude id) returns `None`.
pub fn route(model: &str, mechanical: bool, cooldown: u32, mode: &str) -> Option<&'static str> {
    let admitted_by_mode = mode == "on" || (mode == "auto" && is_high_tier(model));
    if is_claude(model)
        && mechanical
        && cooldown == 0
        && !model.contains("haiku")
        && admitted_by_mode
    {
        Some(ROUTE_TARGET)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // `on` downgrades even Sonnet...
        assert_eq!(route("claude-sonnet-5", true, 0, "on"), Some("claude-haiku-4-5"));
        // ...but a non-Claude id is never rewritten, even in `on`.
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
}
