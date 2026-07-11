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
/// blocks) to plain text, newline-separating parts.
fn tool_result_text(block: &Value) -> Option<String> {
    if block.get("type").and_then(Value::as_str) != Some("tool_result") {
        return None;
    }
    match block.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => Some(
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
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
    let Some(blocks) = newest.get("content").and_then(Value::as_array) else {
        // A bare-string content is fresh user prose, not a mechanical result.
        return false;
    };
    if blocks.is_empty() {
        return false;
    }
    let mut total_tokens = 0usize;
    for block in blocks {
        // Every block must be a tool_result -- a single text block means the
        // user actually said something and deserves a real answer.
        let Some(text) = tool_result_text(block) else {
            return false;
        };
        if has_error_signature(&text) {
            return false;
        }
        total_tokens += token_estimate(&text);
        if total_tokens > TRIVIAL_MAX_TOKENS {
            return false;
        }
    }
    true
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
            "text": "Acknowledged (answered locally by Axiom -- no upstream call was made for this mechanically trivial turn)."
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
