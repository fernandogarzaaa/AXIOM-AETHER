//! Cache-safety hardening for the `/v1/messages` compression path.
//!
//! Anthropic prompt caching is a byte-exact prefix match rendered in
//! `tools -> system -> messages` order (see docs/superpowers/plans/
//! 2026-07-10-cvm-cost-stack.md, step S1). Axiom's TTT compression rewrites
//! heavy message content into a fingerprint that can look different between
//! turns, which invalidates the client's cache prefix the moment it touches
//! anything at or before an Anthropic `cache_control` breakpoint. Simulation
//! showed this costs more (a 1.0x/1.25x rewrite) than it saves (a would-have-
//! been 0.1x cache read) whenever the client actually caches -- and Claude
//! Code always does.
//!
//! This module identifies which leading messages are "frozen" (at or before
//! the last `cache_control` breakpoint) so the compression path can leave
//! them completely untouched, only ever compressing the mutable tail.

use serde_json::Value;

/// True if `v` (or anything nested inside it) carries an Anthropic
/// `cache_control` key, or if `v` itself is a request body with a top-level
/// automatic-caching `cache_control` field.
pub fn request_uses_cache(body: &Value) -> bool {
    has_cache_control(body)
}

fn has_cache_control(v: &Value) -> bool {
    match v {
        Value::Object(map) => {
            map.contains_key("cache_control") || map.values().any(has_cache_control)
        }
        Value::Array(arr) => arr.iter().any(has_cache_control),
        _ => false,
    }
}

/// Number of leading entries in `messages` that must be treated as
/// byte-frozen: every message at or before the last one carrying an
/// explicit `cache_control` marker (on any nested content block). `0` when
/// no message carries a marker.
///
/// If no per-message marker exists but the request opts into caching via a
/// top-level automatic `cache_control` field (`body["cache_control"]`),
/// conservatively freezes every message except the newest one -- this
/// matches Anthropic's own automatic-breakpoint semantics ("the system
/// automatically applies the cache breakpoint to the last cacheable block
/// and moves it forward as conversations grow"): the newest turn is what's
/// actually new, everything before it was already part of a prior cached
/// prefix.
pub fn frozen_prefix_len(body: &Value, messages: &[Value]) -> usize {
    let mut last_marked: Option<usize> = None;
    for (i, m) in messages.iter().enumerate() {
        if has_cache_control(m) {
            last_marked = Some(i);
        }
    }
    if let Some(i) = last_marked {
        return i + 1;
    }
    if body.get("cache_control").is_some() && messages.len() > 1 {
        return messages.len() - 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_uses_cache_detects_explicit_marker_on_a_message_block() {
        let body = json!({
            "model": "claude-sonnet-5",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ]}
            ]
        });
        assert!(request_uses_cache(&body));
    }

    #[test]
    fn request_uses_cache_detects_top_level_automatic_marker() {
        let body = json!({
            "model": "claude-sonnet-5",
            "cache_control": {"type": "ephemeral"},
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert!(request_uses_cache(&body));
    }

    #[test]
    fn request_uses_cache_false_when_no_marker_anywhere() {
        let body = json!({
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert!(!request_uses_cache(&body));
    }

    #[test]
    fn frozen_prefix_len_covers_up_to_and_including_the_last_marked_message() {
        let body = json!({});
        let messages: Vec<Value> = (0..6)
            .map(|i| {
                if i == 3 {
                    json!({"role": "user", "content": [
                        {"type": "text", "text": format!("msg{i}"),
                         "cache_control": {"type": "ephemeral"}}
                    ]})
                } else {
                    json!({"role": "user", "content": format!("msg{i}")})
                }
            })
            .collect();
        // cache_control on index 3 -> indices 0..=3 frozen (len 4).
        assert_eq!(frozen_prefix_len(&body, &messages), 4);
    }

    #[test]
    fn frozen_prefix_len_zero_when_no_marker_and_no_top_level_cache_control() {
        let body = json!({});
        let messages = vec![
            json!({"role": "user", "content": "a"}),
            json!({"role": "assistant", "content": "b"}),
        ];
        assert_eq!(frozen_prefix_len(&body, &messages), 0);
    }

    #[test]
    fn frozen_prefix_len_falls_back_to_all_but_newest_under_top_level_auto_cache() {
        let body = json!({"cache_control": {"type": "ephemeral"}});
        let messages = vec![
            json!({"role": "user", "content": "a"}),
            json!({"role": "assistant", "content": "b"}),
            json!({"role": "user", "content": "c"}),
        ];
        assert_eq!(frozen_prefix_len(&body, &messages), 2);
    }

    #[test]
    fn frozen_prefix_len_single_message_top_level_auto_cache_freezes_nothing() {
        // Nothing "prior" exists yet on a genuine first turn -- the lone
        // message is the mutable tail, not a frozen prefix of itself.
        let body = json!({"cache_control": {"type": "ephemeral"}});
        let messages = vec![json!({"role": "user", "content": "a"})];
        assert_eq!(frozen_prefix_len(&body, &messages), 0);
    }
}
