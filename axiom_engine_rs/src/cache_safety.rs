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

/// Minimum token estimate for a prefix to be worth caching.
/// Anthropic requires 1024 tokens minimum (Sonnet/Opus) or 2048 (Haiku);
/// below that, cache writes cost more than they save. We use 1024 as the
/// conservative floor.
pub const MIN_CACHEABLE_TOKENS: usize = 1024;

/// If `body` has no cache_control markers but the request has a long stable
/// prefix, inject an Anthropic `cache_control: {"type": "ephemeral"}` breakpoint
/// at the optimal position so the provider can cache the prefix.
///
/// Injection strategy:
/// - Only injects when NO existing cache_control is present (never overrides client intent)
/// - Places breakpoint after the system message + all but the last user turn
///   (the stable prefix; the newest turn is the mutable tail)
/// - Skips injection if the prefix is below MIN_CACHEABLE_TOKENS (rough char/4 estimate)
/// - Honors `AXIOM_CACHE_INJECT=0` to disable
///
/// Returns true if a breakpoint was injected.
pub fn maybe_inject_cache_breakpoint(body: &mut Value) -> bool {
    if std::env::var("AXIOM_CACHE_INJECT").as_deref() == Ok("0") {
        return false;
    }
    // Never override explicit client breakpoints.
    if request_uses_cache(body) {
        return false;
    }
    let msg_len = match body.get("messages").and_then(Value::as_array) {
        Some(m) if m.len() > 1 => m.len(),
        _ => return false,
    };
    // Estimate prefix tokens (all but last message). Rough: chars / 4.
    // Scope the immutable borrow so it drops before the mutable borrow below.
    let prefix_chars: usize = {
        let messages = body.get("messages").and_then(Value::as_array).unwrap();
        messages[..messages.len() - 1]
            .iter()
            .map(|m| {
                m.get("content")
                    .map(|c| match c {
                        Value::String(s) => s.len(),
                        Value::Array(arr) => arr
                            .iter()
                            .filter_map(|b| b.get("text").and_then(Value::as_str))
                            .map(str::len)
                            .sum(),
                        _ => 0,
                    })
                    .unwrap_or(0)
            })
            .sum()
    };
    if prefix_chars / 4 < MIN_CACHEABLE_TOKENS {
        return false;
    }
    // Inject breakpoint on the last message of the prefix (index len-2).
    // For string content, convert to content-block array form.
    let idx = msg_len - 2;
    let msgs = match body.get_mut("messages").and_then(Value::as_array_mut) {
        Some(m) => m,
        None => return false,
    };
    let target = &mut msgs[idx];
    let content_val = target.get("content").cloned();
    match content_val {
        Some(Value::String(s)) => {
            target["content"] = Value::Array(vec![serde_json::json!({
                "type": "text",
                "text": s,
                "cache_control": {"type": "ephemeral"}
            })]);
        }
        Some(Value::Array(mut arr)) => {
            if let Some(last_block) = arr.last_mut() {
                if let Some(obj) = last_block.as_object_mut() {
                    obj.insert(
                        "cache_control".to_string(),
                        serde_json::json!({"type": "ephemeral"}),
                    );
                }
            }
            target["content"] = Value::Array(arr);
        }
        _ => return false,
    }
    true
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

    #[test]
    fn maybe_inject_skips_when_client_already_caches() {
        let mut body = json!({
            "model": "claude-sonnet-5",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": "second"}
            ]
        });
        assert!(!maybe_inject_cache_breakpoint(&mut body));
    }

    #[test]
    fn maybe_inject_skips_short_prefix() {
        let mut body = json!({
            "model": "claude-sonnet-5",
            "messages": [
                {"role": "user", "content": "short"},
                {"role": "user", "content": "second"}
            ]
        });
        assert!(!maybe_inject_cache_breakpoint(&mut body));
    }

    #[test]
    fn maybe_inject_adds_breakpoint_to_long_prefix() {
        let long_text = "x".repeat(5000); // ~1250 tokens
        let mut body = json!({
            "model": "claude-sonnet-5",
            "messages": [
                {"role": "user", "content": long_text},
                {"role": "user", "content": "follow-up question"}
            ]
        });
        assert!(maybe_inject_cache_breakpoint(&mut body));
        // Breakpoint should be on the first message (index 0 = len-2).
        let msgs = body["messages"].as_array().unwrap();
        let first_content = &msgs[0]["content"];
        assert!(first_content.as_array().is_some());
        let block = &first_content.as_array().unwrap()[0];
        assert!(block.get("cache_control").is_some());
    }

    #[test]
    fn maybe_inject_handles_array_content_blocks() {
        let long_text = "y".repeat(5000);
        let mut body = json!({
            "model": "claude-sonnet-5",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": long_text},
                    {"type": "text", "text": "more context"}
                ]},
                {"role": "user", "content": "question"}
            ]
        });
        assert!(maybe_inject_cache_breakpoint(&mut body));
        let msgs = body["messages"].as_array().unwrap();
        let blocks = msgs[0]["content"].as_array().unwrap();
        // Breakpoint goes on the last block of the prefix message.
        assert!(blocks.last().unwrap().get("cache_control").is_some());
    }
}
