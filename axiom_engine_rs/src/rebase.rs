//! P2 (Prolonged-Session Stack): free-window rebasing + adaptive cache TTL.
//!
//! When the client's prompt cache is *already* broken (compaction, TTL expiry,
//! or session start -- detected via the S1 cache-safety memo), the whole
//! prefix is re-written at the premium rate regardless. In exactly that window
//! we restructure the transcript deeply at zero *marginal* cache cost:
//! digest every heavy `tool_result` block OLDER than the newest turn into a
//! stub + L2 page (S3 only ever touched the newest turn). This shrinks the
//! transcript re-read on every FUTURE turn. It is never proxy-initiated -- it
//! only ever piggybacks on a break the client already caused, so it respects
//! the CVM anti-pattern catalog (no scheduled eviction / deliberate breaks).
//!
//! Adaptive TTL: sessions with long thinking gaps repeatedly lose the 5-minute
//! cache. Anthropic offers a 1-hour TTL at a 2x write premium (vs 1.25x). Once
//! a session's long-gap count crosses a threshold, [`choose_ttl`] elects the
//! 1-hour TTL -- a one-time premium beats repeated full re-writes.
//!
//! See docs/superpowers/plans/2026-07-11-prolonged-session-stack.md, step P2.

use serde_json::Value;

use crate::cvm_store::{build_stub, CvmStore};
use crate::digest::{Digestor, SkeletonDigestor, DEFAULT_DIGEST_THRESHOLD_TOKENS};

/// Elect Anthropic's 1-hour cache TTL for a session once its observed
/// long-inter-turn-gap count reaches `threshold`; otherwise the default
/// 5-minute TTL (returned as `None`, i.e. no explicit `ttl` annotation).
pub fn choose_ttl(long_gap_count: u32, threshold: u32) -> Option<&'static str> {
    (long_gap_count >= threshold).then_some("1h")
}

/// Given the previous turn's unix timestamp and running long-gap count, return
/// the updated count after a turn at `now_unix`: a gap longer than
/// `gap_threshold_secs` (the 5-minute cache window) increments it. A `last_ts`
/// of `0` (an unseeded session) never counts as a gap.
pub fn next_long_gap_count(
    last_ts: u64,
    now_unix: u64,
    count: u32,
    gap_threshold_secs: u64,
) -> u32 {
    if last_ts > 0 && now_unix.saturating_sub(last_ts) > gap_threshold_secs {
        count + 1
    } else {
        count
    }
}

/// Annotate the newest cache breakpoint with an explicit `ttl`. Walks
/// `messages` from the end and, in the first message whose content array holds
/// a block carrying a `cache_control` object, sets `cache_control.ttl = ttl` on
/// the last such block. Returns `true` if a breakpoint was found and updated.
/// Only ever ADDS a field -- content, order, and count are untouched -- so the
/// cached prefix stays byte-stable.
pub fn set_newest_cache_ttl(messages: &mut [Value], ttl: &str) -> bool {
    for msg in messages.iter_mut().rev() {
        let Some(content) = msg.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in content.iter_mut().rev() {
            if let Some(cc) = block.get_mut("cache_control").and_then(Value::as_object_mut) {
                cc.insert("ttl".to_string(), Value::String(ttl.to_string()));
                return true;
            }
        }
    }
    false
}

/// Flatten a `tool_result` block's own `content` (which is independently a
/// string or an array of text blocks) to plain text.
fn tool_result_text(block: &Value) -> Option<String> {
    if block.get("type").and_then(Value::as_str) != Some("tool_result") {
        return None;
    }
    match block.get("content") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Array(parts)) => {
            let text: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("");
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn token_estimate(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Rebase the transcript: digest every heavy `tool_result` block in every
/// message EXCEPT the newest turn (the last message), storing the full
/// original in `store` and replacing the block's content with a
/// `[AXIOM-PAGE ...]` stub + a code-aware digest. The newest turn is returned
/// untouched. Cache-safety is the caller's responsibility -- this is only
/// invoked at a break window (see the module docs).
pub fn rebase_transcript(messages: &[Value], store: &CvmStore, session_id: &str) -> Vec<Value> {
    if messages.len() <= 1 {
        return messages.to_vec();
    }
    let newest_idx = messages.len() - 1;
    let digestor = SkeletonDigestor;

    messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            if i == newest_idx {
                return msg.clone(); // never touch the newest turn
            }
            let Some(content) = msg.get("content").and_then(Value::as_array) else {
                return msg.clone();
            };
            let mut changed = false;
            let new_content: Vec<Value> = content
                .iter()
                .map(|block| {
                    match tool_result_text(block) {
                        Some(text) if token_estimate(&text) >= DEFAULT_DIGEST_THRESHOLD_TOKENS => {
                            let orig_tokens = token_estimate(&text);
                            match store.put(session_id, "tool_result", &text) {
                                Ok(page_id) => {
                                    let budget = ((orig_tokens as f64) * 0.15).round() as usize;
                                    let stub =
                                        build_stub(&page_id, orig_tokens, "tool_result", &text);
                                    let digest = digestor.digest(&text, budget);
                                    let replacement = format!(
                                        "{stub}\n{digest}\n[AXIOM-PAGE-END expand with axiom_expand(\"{page_id}\")]"
                                    );
                                    let mut b = block.clone();
                                    if let Some(obj) = b.as_object_mut() {
                                        obj.insert(
                                            "content".to_string(),
                                            Value::String(replacement),
                                        );
                                    }
                                    changed = true;
                                    b
                                }
                                Err(e) => {
                                    eprintln!("[axiom-pss] rebase store.put failed: {e}");
                                    block.clone()
                                }
                            }
                        }
                        _ => block.clone(),
                    }
                })
                .collect();
            if changed {
                let mut m = msg.clone();
                if let Some(obj) = m.as_object_mut() {
                    obj.insert("content".to_string(), Value::Array(new_content));
                }
                m
            } else {
                msg.clone()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tempdir(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "pss-rebase-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn choose_ttl_picks_1h_only_after_repeated_long_gaps() {
        assert_eq!(choose_ttl(0, 3), None);
        assert_eq!(choose_ttl(2, 3), None);
        assert_eq!(choose_ttl(3, 3), Some("1h"));
        assert_eq!(choose_ttl(9, 3), Some("1h"));
    }

    #[test]
    fn set_newest_cache_ttl_annotates_only_the_last_breakpoint() {
        let mut messages = vec![
            json!({"role":"user","content":[
                {"type":"text","text":"old","cache_control":{"type":"ephemeral"}}]}),
            json!({"role":"user","content":[
                {"type":"text","text":"a"},
                {"type":"text","text":"newest","cache_control":{"type":"ephemeral"}}]}),
        ];
        assert!(set_newest_cache_ttl(&mut messages, "1h"));
        // newest breakpoint gets the ttl...
        assert_eq!(messages[1]["content"][1]["cache_control"]["ttl"], json!("1h"));
        // ...and the older breakpoint is left untouched.
        assert!(messages[0]["content"][0]["cache_control"].get("ttl").is_none());
    }

    #[test]
    fn next_long_gap_count_increments_only_past_the_window() {
        // Unseeded session never counts, even with a huge apparent gap.
        assert_eq!(next_long_gap_count(0, 10_000, 0, 240), 0);
        // A short gap (< 240s) does not increment.
        assert_eq!(next_long_gap_count(1_000, 1_100, 2, 240), 2);
        // A gap over the window increments once.
        assert_eq!(next_long_gap_count(1_000, 1_500, 2, 240), 3);
        // Exactly at the boundary does not increment (strictly greater).
        assert_eq!(next_long_gap_count(1_000, 1_240, 0, 240), 0);
    }

    #[test]
    fn set_newest_cache_ttl_is_false_without_a_breakpoint() {
        let mut messages = vec![json!({"role":"user","content":[{"type":"text","text":"no cc"}]})];
        assert!(!set_newest_cache_ttl(&mut messages, "1h"));
    }

    #[test]
    fn rebase_digests_all_old_heavy_but_never_the_newest_turn() {
        let dir = tempdir("basic");
        let store = CvmStore::open(&dir).unwrap();
        let big = "x ".repeat(9000); // 9000 tokens, well over the 4000 threshold
        let messages = vec![
            json!({"role":"user","content":[
                {"type":"tool_result","tool_use_id":"a","content": big}]}),
            json!({"role":"user","content":"newest turn stays whole"}),
        ];
        let out = rebase_transcript(&messages, &store, "s1");
        assert!(out[0].to_string().contains("AXIOM-PAGE"), "old heavy digested");
        assert!(!out[0].to_string().contains(&"x ".repeat(9000)), "raw text removed");
        assert_eq!(out[1]["content"], json!("newest turn stays whole"), "newest untouched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rebase_leaves_light_tool_results_alone() {
        let dir = tempdir("light");
        let store = CvmStore::open(&dir).unwrap();
        let messages = vec![
            json!({"role":"user","content":[
                {"type":"tool_result","tool_use_id":"a","content":"exit 0"}]}),
            json!({"role":"user","content":"newest"}),
        ];
        let out = rebase_transcript(&messages, &store, "s1");
        assert_eq!(out[0], messages[0], "sub-threshold tool_result untouched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rebase_original_is_recoverable_from_the_store() {
        let dir = tempdir("recover");
        let store = CvmStore::open(&dir).unwrap();
        let big = "y ".repeat(9000);
        let messages = vec![
            json!({"role":"user","content":[
                {"type":"tool_result","tool_use_id":"a","content": big.clone()}]}),
            json!({"role":"user","content":"newest"}),
        ];
        let _ = rebase_transcript(&messages, &store, "s1");
        let page_id = CvmStore::page_id_for(&big);
        assert_eq!(store.get("s1", &page_id), Some(big));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rebase_single_message_is_a_noop() {
        let dir = tempdir("single");
        let store = CvmStore::open(&dir).unwrap();
        let big = "z ".repeat(9000);
        let messages = vec![json!({"role":"user","content":[
            {"type":"tool_result","tool_use_id":"a","content": big}]})];
        let out = rebase_transcript(&messages, &store, "s1");
        assert_eq!(out, messages, "a lone newest turn is never rebased");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
