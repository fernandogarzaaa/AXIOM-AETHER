//! P3 (Prolonged-Session Stack): L-B local trivial-turn short-circuit.
//!
//! Long agent sessions are punctuated by mechanically trivial turns: a
//! `tool_result` reporting `exit 0`, a file written, a short clean stdout --
//! turns the model would answer with a one-line acknowledgement. Forwarding
//! them upstream re-reads (and re-pays for) the entire cached prefix just to
//! produce "ok". When a turn is provably trivial -- purely a small, clean
//! `tool_result` with no error signature and low surprisal (the model is not
//! surprised by it, so it carries little new information) -- we answer it
//! locally and never touch the network.
//!
//! [`is_trivial`] is deliberately *fail-closed*: it returns `false` unless
//! every condition holds, so anything ambiguous is forwarded upstream. An
//! error signature, fresh user prose, heavy content, or missing/high surprisal
//! all force a real round-trip.
//!
//! See docs/superpowers/plans/2026-07-11-prolonged-session-stack.md, step P3.

use serde_json::{json, Value};

/// A turn larger than this (whitespace tokens across its `tool_result` text)
/// is never "trivial" -- a big result plausibly carries content the model
/// must actually reason about, so it is forwarded upstream.
const TRIVIAL_MAX_TOKENS: usize = 200;

/// A hard byte cap on a trivial turn, independent of the token count. A large
/// minified JSON / base64 blob has almost no whitespace, so it would pass the
/// token cap as "one token" -- the byte cap catches it. Callers also apply this
/// bound *before* computing surprisal so a huge result never reaches the
/// pipeline (the heavy-content fail-closed contract).
pub const TRIVIAL_MAX_BYTES: usize = 2_000;

/// The fixed acknowledgement text for a locally-answered trivial turn.
const ACK_TEXT: &str =
    "Acknowledged (answered locally by Axiom -- no upstream call was made for this mechanically trivial turn).";

/// Substrings (matched case-insensitively) that mark a `tool_result` as
/// carrying a failure the model must actually see -- never short-circuited.
/// Shared with P4's routing gate.
const ERROR_SIGNATURES: [&str; 5] = ["error", "panic", "failed", "exception", "traceback"];

/// Case-insensitive test for any known failure signature in `text`. A turn
/// bearing one is never trivial (L-B) and never eligible for downgrade (P4).
pub fn has_error_signature(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    ERROR_SIGNATURES.iter().any(|sig| lower.contains(sig))
}

/// Flatten a `tool_result` block's `content` (a string or an array of text
/// blocks) to plain text, newline-separating parts. Returns `None` -- meaning
/// "not a plain-text tool_result, so never trivial" -- if the block is not a
/// `tool_result` at all, or if any array part is non-text (an image, document,
/// or unknown block the model may need to inspect). We must NOT silently drop
/// non-text parts: a mixed image + "ok" result would otherwise be admitted on
/// the "ok" alone.
fn tool_result_text(block: &Value) -> Option<String> {
    if block.get("type").and_then(Value::as_str) != Some("tool_result") {
        return None;
    }
    match block.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => {
            let mut texts = Vec::with_capacity(parts.len());
            for p in parts {
                // Any non-text part disqualifies the whole result.
                let text = p.get("text").and_then(Value::as_str)?;
                texts.push(text);
            }
            Some(texts.join("\n"))
        }
        _ => None,
    }
}

fn token_estimate(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Is the newest turn provably trivial -- safe to answer locally without a
/// round-trip? Fail-closed: `true` only when the newest message is composed
/// *entirely* of `tool_result` blocks (no fresh user prose), none of them bear
/// an error signature, their combined text is small, and `surprisal` is present
/// and below `gate` (the model finds the turn unsurprising). Any other shape --
/// a text block, an empty transcript, absent surprisal -- returns `false`.
pub fn is_trivial(messages: &[Value], surprisal: Option<f32>, gate: f32) -> bool {
    // Surprisal must be known and low: an unknown surprisal is not a licence
    // to skip the model.
    if !surprisal.is_some_and(|s| s < gate) {
        return false;
    }
    let Some(newest) = messages.last() else {
        return false;
    };
    // The newest turn must be a genuine user-role tool-result turn. A client
    // could otherwise submit an assistant-role `tool_result` and receive a
    // synthetic success instead of the upstream API's normal validation.
    if newest.get("role").and_then(Value::as_str) != Some("user") {
        return false;
    }
    let Some(blocks) = newest.get("content").and_then(Value::as_array) else {
        // A bare-string content is fresh user prose, not a mechanical result.
        return false;
    };
    if blocks.is_empty() {
        return false;
    }
    let mut total_tokens = 0usize;
    let mut total_bytes = 0usize;
    for block in blocks {
        // Every block must be a plain-text tool_result -- a single text block
        // means the user actually said something and deserves a real answer,
        // and a non-text (image/document) part means content to inspect.
        let Some(text) = tool_result_text(block) else {
            return false;
        };
        if has_error_signature(&text) {
            return false;
        }
        total_tokens += token_estimate(&text);
        total_bytes += text.len();
        if total_tokens > TRIVIAL_MAX_TOKENS || total_bytes > TRIVIAL_MAX_BYTES {
            return false;
        }
    }
    true
}

/// The Anthropic streaming (`text/event-stream`) form of [`local_ack`], for a
/// client that requested `"stream": true`. Emits the standard event sequence
/// (`message_start` → `content_block_start` → one `content_block_delta` →
/// `content_block_stop` → `message_delta` → `message_stop`) so a streaming
/// client sees a normal, complete turn. Marked `model: "axiom-local"`.
pub fn local_ack_sse() -> String {
    let id = format!("msg_axiomlocal_{}", uuid::Uuid::new_v4().simple());
    let start = json!({
        "type": "message_start",
        "message": {
            "id": id, "type": "message", "role": "assistant", "model": "axiom-local",
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }
    });
    let block_start = json!({
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}
    });
    let delta = json!({
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": ACK_TEXT}
    });
    let block_stop = json!({"type": "content_block_stop", "index": 0});
    let msg_delta = json!({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": null},
        "usage": {"output_tokens": 0}
    });
    let stop = json!({"type": "message_stop"});
    // Each SSE event: a named `event:` line + a `data:` line + a blank line.
    [
        ("message_start", start),
        ("content_block_start", block_start),
        ("content_block_delta", delta),
        ("content_block_stop", block_stop),
        ("message_delta", msg_delta),
        ("message_stop", stop),
    ]
    .iter()
    .map(|(name, payload)| format!("event: {name}\ndata: {payload}\n\n"))
    .collect()
}

/// A locally-generated Anthropic-shaped acknowledgement for a trivial turn.
/// Marked `model: "axiom-local"` so the client (and any telemetry) can tell it
/// never went upstream.
pub fn local_ack() -> Value {
    json!({
        "id": format!("msg_axiomlocal_{}", uuid::Uuid::new_v4().simple()),
        "type": "message",
        "role": "assistant",
        "model": "axiom-local",
        "content": [{
            "type": "text",
            "text": ACK_TEXT
        }],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_trivial_true_for_clean_mechanical_low_surprisal_turn() {
        let m = vec![json!({"role":"user","content":[
            {"type":"tool_result","tool_use_id":"x","content":"ok, exit 0"}]})];
        assert!(is_trivial(&m, Some(1.0), 7.03));
    }

    #[test]
    fn is_trivial_false_when_surprisal_above_gate() {
        let m = vec![json!({"role":"user","content":[
            {"type":"tool_result","tool_use_id":"x","content":"ok"}]})];
        assert!(!is_trivial(&m, Some(9.0), 7.03));
    }

    #[test]
    fn is_trivial_false_when_surprisal_absent() {
        let m = vec![json!({"role":"user","content":[
            {"type":"tool_result","tool_use_id":"x","content":"ok"}]})];
        assert!(!is_trivial(&m, None, 7.03), "unknown surprisal is fail-closed");
    }

    #[test]
    fn is_trivial_false_on_error_signature() {
        let m = vec![json!({"role":"user","content":[
            {"type":"tool_result","tool_use_id":"x","content":"Error: panicked"}]})];
        assert!(!is_trivial(&m, Some(1.0), 7.03));
    }

    #[test]
    fn is_trivial_false_for_fresh_user_prose() {
        let m = vec![json!({"role":"user","content":"please refactor the parser"})];
        assert!(!is_trivial(&m, Some(1.0), 7.03), "a text turn is never trivial");
        let m2 = vec![json!({"role":"user","content":[
            {"type":"text","text":"and also add tests"}]})];
        assert!(!is_trivial(&m2, Some(1.0), 7.03), "a text block is fresh prose");
    }

    #[test]
    fn is_trivial_false_for_heavy_tool_result() {
        let big = "word ".repeat(TRIVIAL_MAX_TOKENS + 50);
        let m = vec![json!({"role":"user","content":[
            {"type":"tool_result","tool_use_id":"x","content": big}]})];
        assert!(!is_trivial(&m, Some(1.0), 7.03), "a large result is forwarded");
    }

    #[test]
    fn is_trivial_false_for_whitespace_free_blob_over_byte_cap() {
        // A base64/minified blob: ~one whitespace token, but well over the byte
        // cap -- must not slip through as "one small token".
        let blob = "a".repeat(TRIVIAL_MAX_BYTES + 10);
        let m = vec![json!({"role":"user","content":[
            {"type":"tool_result","tool_use_id":"x","content": blob}]})];
        assert!(!is_trivial(&m, Some(1.0), 7.03), "byte cap catches token-light blobs");
    }

    #[test]
    fn is_trivial_false_for_non_user_role() {
        let m = vec![json!({"role":"assistant","content":[
            {"type":"tool_result","tool_use_id":"x","content":"ok"}]})];
        assert!(!is_trivial(&m, Some(1.0), 7.03), "only a user-role turn may short-circuit");
    }

    #[test]
    fn is_trivial_false_for_mixed_non_text_content() {
        // An image part alongside "ok" -- the model may need to see the image.
        let m = vec![json!({"role":"user","content":[
            {"type":"tool_result","tool_use_id":"x","content":[
                {"type":"text","text":"ok"},
                {"type":"image","source":{"type":"base64","data":"..."}}]}]})];
        assert!(!is_trivial(&m, Some(1.0), 7.03), "non-text part disqualifies the result");
    }

    #[test]
    fn local_ack_sse_emits_the_anthropic_event_sequence() {
        let sse = local_ack_sse();
        for ev in [
            "event: message_start",
            "event: content_block_start",
            "event: content_block_delta",
            "event: content_block_stop",
            "event: message_delta",
            "event: message_stop",
        ] {
            assert!(sse.contains(ev), "missing {ev}");
        }
        assert!(sse.contains("axiom-local"), "marked as a local response");
        assert!(sse.contains("text_delta"), "carries the ack text delta");
    }

    #[test]
    fn is_trivial_false_for_empty_transcript() {
        assert!(!is_trivial(&[], Some(1.0), 7.03));
    }

    #[test]
    fn has_error_signature_is_case_insensitive() {
        assert!(has_error_signature("Traceback (most recent call last)"));
        assert!(has_error_signature("FAILED"));
        assert!(has_error_signature("Uncaught Exception"));
        assert!(!has_error_signature("all green"));
    }

    #[test]
    fn local_ack_is_anthropic_shaped_and_marked_local() {
        let ack = local_ack();
        assert_eq!(ack["type"], json!("message"));
        assert_eq!(ack["role"], json!("assistant"));
        assert_eq!(ack["model"], json!("axiom-local"));
        assert_eq!(ack["stop_reason"], json!("end_turn"));
        assert_eq!(ack["content"][0]["type"], json!("text"));
        assert!(ack["id"].as_str().unwrap().starts_with("msg_axiomlocal_"));
    }
}
