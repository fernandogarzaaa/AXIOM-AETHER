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

/// Fingerprint a frozen prefix: `(message count, SHA-256 hex of the serialized
/// slice)`. Stored per session so the next turn can tell append-only growth
/// from a genuine break.
pub fn frozen_fingerprint(frozen: &[Value]) -> (usize, String) {
    use sha2::{Digest, Sha256};
    let bytes = serde_json::to_vec(frozen).unwrap_or_default();
    (frozen.len(), format!("{:x}", Sha256::digest(&bytes)))
}

/// Is this turn's frozen prefix a GENUINE cache break relative to the previous
/// turn's fingerprint? Anthropic's automatic breakpoint moves forward as a
/// session grows, so the frozen prefix EXTENDING turn-over-turn is normal
/// cached operation -- calling that a break (as a whole-prefix hash compare
/// does) fires the rebase every turn and destroys the model's working history
/// (the 2026-07-12 live-eval FAIL). A genuine break is a NON-APPEND change:
/// the prefix shrank (compaction / session restructure) or its leading
/// `prev_len` messages are no longer byte-identical.
pub fn is_genuine_break(prev_len: usize, prev_hash: &str, frozen: &[Value]) -> bool {
    if frozen.len() < prev_len {
        return true; // prefix shrank: the old cached prefix cannot survive
    }
    let (_, head_hash) = frozen_fingerprint(&frozen[..prev_len]);
    head_hash != prev_hash
}

/// What [`elect_newest_cache_ttl`] did to a request body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtlElection {
    /// The newest breakpoint now carries the elected ttl.
    Applied,
    /// The newest breakpoint already carried it; nothing changed.
    AlreadySet,
    /// An earlier breakpoint is 5m (or has no ttl), and the API rejects a 1h
    /// breakpoint after a 5m one; nothing changed.
    EarlierShorterTtl,
    /// The body has a top-level `cache_control` (automatic caching), whose
    /// placement the proxy does not model; nothing changed.
    AutomaticCaching,
    /// No breakpoint to annotate.
    NoBreakpoint,
}

/// Every `cache_control` object in the order the API processes breakpoints
/// (`tools`, then `system`, then `messages`; blocks nested in a `tool_result`
/// before the block itself), as `(JSON pointer, is_1h)`. A breakpoint without
/// `"ttl":"1h"` is the 5-minute default.
fn cache_breakpoints(body: &Value) -> Vec<(String, bool)> {
    fn push(out: &mut Vec<(String, bool)>, block: &Value, path: String) {
        if let Some(cc) = block.get("cache_control").filter(|cc| cc.is_object()) {
            let is_1h = cc.get("ttl").and_then(Value::as_str) == Some("1h");
            out.push((format!("{path}/cache_control"), is_1h));
        }
    }
    fn items<'a>(v: &'a Value, key: &str) -> impl Iterator<Item = (usize, &'a Value)> {
        v.get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
    }
    let mut out = Vec::new();
    for (i, tool) in items(body, "tools") {
        push(&mut out, tool, format!("/tools/{i}"));
    }
    for (i, block) in items(body, "system") {
        push(&mut out, block, format!("/system/{i}"));
    }
    for (m, msg) in items(body, "messages") {
        for (b, block) in items(msg, "content") {
            let path = format!("/messages/{m}/content/{b}");
            for (n, inner) in items(block, "content") {
                push(&mut out, inner, format!("{path}/content/{n}"));
            }
            push(&mut out, block, path);
        }
    }
    out
}

/// Annotate the newest cache breakpoint with `ttl`, but only when the result
/// is a request the API accepts: a 1h breakpoint must never come after a 5m
/// one, in the order tools, system, messages. Claude Code subagents send 5m
/// breakpoints on tools and system under the parent's session id, so
/// upgrading their newest breakpoint alone was an API 400 on every call.
///
/// Only ever ADDS a field to the final breakpoint. Earlier breakpoints are
/// never rewritten, so the cached prefix the client already paid for keeps
/// its bytes and its ttl.
pub fn elect_newest_cache_ttl(body: &mut Value, ttl: &str) -> TtlElection {
    if body.get("cache_control").is_some() {
        return TtlElection::AutomaticCaching;
    }
    let points = cache_breakpoints(body);
    let Some(((newest, newest_is_1h), earlier)) = points.split_last() else {
        return TtlElection::NoBreakpoint;
    };
    let electing_1h = ttl == "1h";
    if electing_1h && *newest_is_1h {
        return TtlElection::AlreadySet;
    }
    if electing_1h && earlier.iter().any(|(_, is_1h)| !is_1h) {
        return TtlElection::EarlierShorterTtl;
    }
    match body.pointer_mut(newest).and_then(Value::as_object_mut) {
        Some(cc) => {
            cc.insert("ttl".to_string(), Value::String(ttl.to_string()));
            TtlElection::Applied
        }
        None => TtlElection::NoBreakpoint,
    }
}

/// Flatten a `tool_result` block's own `content` (which is independently a
/// string or an array of text blocks) to plain text. Shared with
/// [`crate::anthropic_forwarder`]'s S0 heavy-content partitioning, which
/// classifies `tool_result` blocks the same way this module's `tool_result`
/// digestion does.
///
/// Returns `None` for a mixed-content array (any block whose `type` isn't
/// `"text"` — e.g. `image`/`document`): every caller replaces the entire
/// `content` field with the extracted string, which would silently drop
/// non-text entries. Paging is skipped rather than losing data; only
/// pure-text `tool_result`s are eligible for digestion/extraction.
pub(crate) fn tool_result_text(block: &Value) -> Option<String> {
    if block.get("type").and_then(Value::as_str) != Some("tool_result") {
        return None;
    }
    match block.get("content") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Array(parts)) => {
            if parts
                .iter()
                .any(|p| p.get("type").and_then(Value::as_str) != Some("text"))
            {
                return None;
            }
            let text: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                // Newline-separate parts so a multi-block content array (stdout
                // + stderr, several file excerpts) keeps its boundaries: a bare
                // join would run words together, corrupting both the token
                // estimate and the "original" text persisted to the L2 store.
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn token_estimate(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Build the deterministic replacement string for a paged `tool_result`.
/// MUST be a pure function of `(page_id, session_id, text)`:
/// [`reapply_stubs`] regenerates it on every later turn and the bytes have
/// to match what [`rebase_transcript`] forwarded (and Anthropic cached) at
/// the break turn exactly. Both callers pass the request's session id, which
/// is stable for the life of the session, so determinism holds.
fn stub_replacement(page_id: &str, session_id: &str, text: &str) -> String {
    let orig_tokens = token_estimate(text);
    let budget = ((orig_tokens as f64) * 0.15).round() as usize;
    let stub = build_stub(page_id, session_id, orig_tokens, "tool_result", text);
    let digest = SkeletonDigestor.digest(text, budget);
    format!("{stub}\n{digest}\n[AXIOM-PAGE-END expand with axiom_expand(symbol=\"{page_id}\", session_id=\"{session_id}\")]")
}

/// A block whose text is itself a stub must never be paged again (a
/// stub-of-stub would orphan the real page id).
fn is_already_stubbed(text: &str) -> bool {
    text.trim_start().starts_with("[AXIOM-PAGE ")
}

/// Rewrite the heavy `tool_result` blocks of one message via `replace`,
/// which maps block text to its replacement (or `None` to keep the block).
fn rewrite_heavy_blocks(msg: &Value, replace: &mut dyn FnMut(&str) -> Option<String>) -> Value {
    let Some(content) = msg.get("content").and_then(Value::as_array) else {
        return msg.clone();
    };
    let mut changed = false;
    let new_content: Vec<Value> = content
        .iter()
        .map(|block| match tool_result_text(block) {
            Some(text)
                if token_estimate(&text) >= DEFAULT_DIGEST_THRESHOLD_TOKENS
                    && !is_already_stubbed(&text) =>
            {
                match replace(&text) {
                    Some(replacement) => {
                        let mut b = block.clone();
                        if let Some(obj) = b.as_object_mut() {
                            obj.insert("content".to_string(), Value::String(replacement));
                        }
                        changed = true;
                        b
                    }
                    None => block.clone(),
                }
            }
            _ => block.clone(),
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

    messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            if i == newest_idx {
                return msg.clone(); // never touch the newest turn
            }
            rewrite_heavy_blocks(
                msg,
                &mut |text| match store.put(session_id, "tool_result", text) {
                    Ok(page_id) => Some(stub_replacement(&page_id, session_id, text)),
                    Err(e) => {
                        eprintln!("[axiom-pss] rebase store.put failed: {e}");
                        None
                    }
                },
            )
        })
        .collect()
}

/// Deterministically re-apply existing stubs to a transcript slice WITHOUT
/// ever writing to the store. A heavy `tool_result` block is replaced only
/// when its content hash already has a page in `store` (i.e. an earlier
/// break-window [`rebase_transcript`] paged this exact text). Because both
/// the page id and the replacement string are pure functions of the block's
/// bytes, applying this every turn keeps the forwarded frozen prefix
/// byte-identical to what was cached after the break -- the mechanism that
/// makes rebasing the frozen prefix cache-safe on subsequent turns.
///
/// `protect_last` must be true when the slice's last message is the newest
/// turn of the whole request (i.e. the mutable tail is empty), preserving
/// rebase's never-touch-the-newest-turn contract.
pub fn reapply_stubs(
    messages: &[Value],
    store: &CvmStore,
    session_id: &str,
    protect_last: bool,
) -> Vec<Value> {
    let protected_idx = if protect_last && !messages.is_empty() {
        Some(messages.len() - 1)
    } else {
        None
    };
    messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            if Some(i) == protected_idx {
                return msg.clone();
            }
            rewrite_heavy_blocks(msg, &mut |text| {
                let page_id = CvmStore::page_id_for(text);
                store
                    .get(session_id, &page_id)
                    .map(|_| stub_replacement(&page_id, session_id, text))
            })
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
    fn elect_newest_cache_ttl_annotates_only_the_last_breakpoint() {
        let mut body = json!({"messages":[
            {"role":"user","content":[
                {"type":"text","text":"old","cache_control":{"type":"ephemeral","ttl":"1h"}}]},
            {"role":"user","content":[
                {"type":"text","text":"a"},
                {"type":"text","text":"newest","cache_control":{"type":"ephemeral"}}]},
        ]});
        assert_eq!(
            elect_newest_cache_ttl(&mut body, "1h"),
            TtlElection::Applied
        );
        assert_eq!(
            body["messages"][1]["content"][1]["cache_control"],
            json!({"type":"ephemeral","ttl":"1h"})
        );
        assert!(body["messages"][1]["content"][0]
            .get("cache_control")
            .is_none());
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            json!({"type":"ephemeral","ttl":"1h"})
        );
    }

    #[test]
    fn elect_newest_cache_ttl_refuses_1h_after_any_earlier_5m_breakpoint() {
        let default_cc = json!({"type":"ephemeral"});
        let five_min = json!({"type":"ephemeral","ttl":"5m"});
        let cases = [
            (
                "tool",
                json!({
                "tools":[{"name":"t","cache_control": default_cc}],
                "messages":[{"role":"user","content":[
                    {"type":"text","text":"n","cache_control":{"type":"ephemeral"}}]}]}),
            ),
            (
                "system",
                json!({
                "system":[{"type":"text","text":"s","cache_control": five_min}],
                "messages":[{"role":"user","content":[
                    {"type":"text","text":"n","cache_control":{"type":"ephemeral"}}]}]}),
            ),
            (
                "earlier message",
                json!({"messages":[
                {"role":"user","content":[
                    {"type":"text","text":"o","cache_control": default_cc}]},
                {"role":"user","content":[
                    {"type":"text","text":"n","cache_control":{"type":"ephemeral"}}]}]}),
            ),
            (
                "nested in tool_result",
                json!({"messages":[
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"x","content":[
                        {"type":"text","text":"o","cache_control": default_cc}]},
                    {"type":"text","text":"n","cache_control":{"type":"ephemeral"}}]}]}),
            ),
        ];
        for (name, body) in cases {
            let mut out = body.clone();
            assert_eq!(
                elect_newest_cache_ttl(&mut out, "1h"),
                TtlElection::EarlierShorterTtl,
                "{name}"
            );
            assert_eq!(out, body, "{name}: body untouched");
        }
    }

    #[test]
    fn elect_newest_cache_ttl_leaves_automatic_caching_and_set_ttls_alone() {
        let auto = json!({"cache_control":{"type":"ephemeral"},"messages":[
            {"role":"user","content":[
                {"type":"text","text":"n","cache_control":{"type":"ephemeral"}}]}]});
        let mut out = auto.clone();
        assert_eq!(
            elect_newest_cache_ttl(&mut out, "1h"),
            TtlElection::AutomaticCaching
        );
        assert_eq!(out, auto);

        let set = json!({"system":"plain string, no breakpoints","messages":[
            {"role":"user","content":[
                {"type":"text","text":"n","cache_control":{"type":"ephemeral","ttl":"1h"}}]}]});
        let mut out = set.clone();
        assert_eq!(
            elect_newest_cache_ttl(&mut out, "1h"),
            TtlElection::AlreadySet
        );
        assert_eq!(out, set);
    }

    #[test]
    fn appending_to_the_frozen_prefix_is_not_a_break() {
        // The 2026-07-12 live-eval failure mode: Anthropic's automatic
        // breakpoint moves forward every turn, so the frozen prefix GROWS
        // turn-over-turn. That is normal cached operation, never a break.
        let turn1 = vec![json!({"role":"user","content":"a"})];
        let (len1, hash1) = frozen_fingerprint(&turn1);
        let turn2 = vec![
            json!({"role":"user","content":"a"}),
            json!({"role":"assistant","content":"b"}),
        ];
        assert!(
            !is_genuine_break(len1, &hash1, &turn2),
            "append-only growth is not a break"
        );
        // And an identical prefix is not a break either.
        assert!(
            !is_genuine_break(len1, &hash1, &turn1),
            "unchanged prefix is not a break"
        );
    }

    #[test]
    fn shrinking_or_mutating_the_frozen_prefix_is_a_break() {
        let prev = vec![
            json!({"role":"user","content":"a"}),
            json!({"role":"assistant","content":"b"}),
        ];
        let (len, hash) = frozen_fingerprint(&prev);
        // Compaction: the prefix shrank.
        let shrunk = vec![json!({"role":"user","content":"summary of a+b"})];
        assert!(
            is_genuine_break(len, &hash, &shrunk),
            "a shrunken prefix is a break"
        );
        // Restructure: same length, different leading content.
        let mutated = vec![
            json!({"role":"user","content":"a CHANGED"}),
            json!({"role":"assistant","content":"b"}),
        ];
        assert!(
            is_genuine_break(len, &hash, &mutated),
            "a mutated prefix is a break"
        );
    }

    #[test]
    fn multipart_tool_result_is_joined_with_a_separator() {
        // Two text parts whose words would run together under a bare join.
        let block = json!({
            "type":"tool_result","tool_use_id":"a",
            "content":[{"type":"text","text":"alpha"},{"type":"text","text":"beta"}]
        });
        let text = tool_result_text(&block).unwrap();
        assert_eq!(text, "alpha\nbeta", "parts keep a boundary");
        // ...so the whitespace token estimate counts two tokens, not one.
        assert_eq!(token_estimate(&text), 2);
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
    fn elect_newest_cache_ttl_reports_no_breakpoint() {
        let mut body =
            json!({"messages":[{"role":"user","content":[{"type":"text","text":"no cc"}]}]});
        assert_eq!(
            elect_newest_cache_ttl(&mut body, "1h"),
            TtlElection::NoBreakpoint
        );
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
        assert!(
            out[0].to_string().contains("AXIOM-PAGE"),
            "old heavy digested"
        );
        assert!(
            !out[0].to_string().contains(&"x ".repeat(9000)),
            "raw text removed"
        );
        assert_eq!(
            out[1]["content"],
            json!("newest turn stays whole"),
            "newest untouched"
        );
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

    #[test]
    fn reapply_stubs_reproduces_rebase_transcript_byte_for_byte() {
        // The whole point of reapply_stubs: re-sending the ORIGINAL (unstubbed)
        // transcript on a later turn must transform into the EXACT bytes the
        // break-window rebase forwarded (and Anthropic cached), or the "frozen"
        // prefix silently diverges from the cached one.
        let dir = tempdir("reapply-parity");
        let store = CvmStore::open(&dir).unwrap();
        let big = "p ".repeat(9000);
        let original = vec![
            json!({"role":"user","content":[
                {"type":"tool_result","tool_use_id":"a","content": big.clone()}]}),
            json!({"role":"assistant","content":"ack"}),
        ];
        let rebased = rebase_transcript(&original, &store, "s1");
        // A later turn resends the client's ORIGINAL (unstubbed) bytes for the
        // now-old messages; protect_last=false since these aren't the newest
        // turn of the later request.
        let replayed = reapply_stubs(&original, &store, "s1", false);
        assert_eq!(
            replayed, rebased,
            "reapply must match the original rebase exactly"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reapply_stubs_never_touches_an_unpaged_block() {
        // No prior rebase_transcript call happened for this text, so no page
        // exists -- reapply_stubs must leave it byte-identical rather than
        // writing a new page (that's rebase_transcript's job, gated on a
        // genuine break, not reapply's).
        let dir = tempdir("reapply-unpaged");
        let store = CvmStore::open(&dir).unwrap();
        let big = "q ".repeat(9000);
        let messages = vec![json!({"role":"user","content":[
            {"type":"tool_result","tool_use_id":"a","content": big.clone()}]})];
        let out = reapply_stubs(&messages, &store, "s1", false);
        assert_eq!(out, messages, "unpaged heavy block is left untouched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reapply_stubs_protects_the_last_message_when_requested() {
        let dir = tempdir("reapply-protect-last");
        let store = CvmStore::open(&dir).unwrap();
        let big = "r ".repeat(9000);
        let original = vec![json!({"role":"user","content":[
            {"type":"tool_result","tool_use_id":"a","content": big.clone()}]})];
        // Page it via a two-message rebase first so a page exists for `big`.
        let mut two = original.clone();
        two.push(json!({"role":"assistant","content":"ack"}));
        let _ = rebase_transcript(&two, &store, "s1");
        // Now `original` (the paged block IS the last message) with
        // protect_last=true must stay untouched even though a page exists.
        let out = reapply_stubs(&original, &store, "s1", true);
        assert_eq!(out, original, "protect_last shields the final message");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reapply_stubs_is_idempotent_on_an_already_stubbed_transcript() {
        // Feeding the OUTPUT of a prior reapply back in must not double-stub
        // (is_already_stubbed guards this) or attempt a second page write.
        let dir = tempdir("reapply-idempotent");
        let store = CvmStore::open(&dir).unwrap();
        let big = "s ".repeat(9000);
        let original = vec![
            json!({"role":"user","content":[
                {"type":"tool_result","tool_use_id":"a","content": big.clone()}]}),
            json!({"role":"assistant","content":"ack"}),
        ];
        let once = reapply_stubs(
            &rebase_transcript(&original, &store, "s1"),
            &store,
            "s1",
            false,
        );
        let twice = reapply_stubs(&once, &store, "s1", false);
        assert_eq!(
            once, twice,
            "reapplying to an already-stubbed transcript is a no-op"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reapply_with_different_session_leaves_stubbed_blocks_byte_identical() {
        // Cache-safety with session ids embedded in stubs: a stub minted
        // under session "s1" must NOT be regenerated with "s2" on a later
        // turn (that would break Anthropic's byte-exact prefix cache). The
        // session-scoped store lookup fails closed, and is_already_stubbed
        // skips the block, so bytes are preserved verbatim.
        let dir = tempdir("reapply-cross-session");
        let store = CvmStore::open(&dir).unwrap();
        let big = "t ".repeat(9000);
        let original = vec![
            json!({"role":"user","content":[
                {"type":"tool_result","tool_use_id":"a","content": big.clone()}]}),
            json!({"role":"assistant","content":"ack"}),
        ];
        let rebased = rebase_transcript(&original, &store, "s1");
        let stub_text = rebased[0]["content"][0]["content"].as_str().unwrap();
        assert!(
            stub_text.contains("session=\"s1\""),
            "rebased stub must carry the storing session"
        );
        // Later turn arrives under a different (e.g. transient) session.
        let reapplied = reapply_stubs(&rebased, &store, "s2", false);
        assert_eq!(
            reapplied, rebased,
            "cross-session reapply must not rewrite stub bytes"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rebase_transcript_paging_the_same_text_twice_does_not_duplicate_store_rows() {
        // A break-window rebase can legitimately re-run over content it
        // already paged (e.g. two genuine breaks before the old turn ages
        // out). CvmStore::put must treat the identical content-hash page as
        // a no-op rather than growing the session file unboundedly.
        let dir = tempdir("no-dup-rows");
        let store = CvmStore::open(&dir).unwrap();
        let big = "t ".repeat(9000);
        let messages = vec![
            json!({"role":"user","content":[
                {"type":"tool_result","tool_use_id":"a","content": big.clone()}]}),
            json!({"role":"assistant","content":"ack"}),
        ];
        let first = rebase_transcript(&messages, &store, "s1");
        let second = rebase_transcript(&messages, &store, "s1");
        assert_eq!(
            first, second,
            "re-paging identical content is deterministic"
        );
        let page_id = CvmStore::page_id_for(&big);
        assert_eq!(store.get("s1", &page_id), Some(big));
        // Assert the underlying JSONL file directly: `first == second` and
        // `get` returning the right text would both still pass even if
        // `put` silently appended a duplicate row on the second call --
        // this is what actually catches that regression.
        let row_count = std::fs::read_to_string(store.session_path_for_test("s1"))
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count();
        assert_eq!(
            row_count, 1,
            "identical content paged twice must add exactly one row"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
