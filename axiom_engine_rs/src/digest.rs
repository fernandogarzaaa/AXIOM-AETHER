//! S3 (CVM cost stack): digest admission control -- the big lever.
//!
//! Prevention beats compression: heavy tool results should never enter the
//! (expensive, cached-forever) transcript at full size. On arrival, in the
//! newest turn only (by definition after every cache breakpoint and not yet
//! cached), replace heavy content with a digest + stub; the full text goes
//! to the L2 store (`cvm_store`). Cache-safe by construction (S1 froze the
//! prefix) and needs no determinism trick: the digest is created exactly
//! once and then it IS the history. Default off
//! (`AXIOM_CVM_DIGEST=off`) until S5 passes. See
//! docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S3.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use serde::Serialize;
use serde_json::Value;

/// A digest backend: takes heavy text and a hard token budget, returns a
/// shorter (never longer than the budget) representation.
pub trait Digestor {
    fn digest(&self, text: &str, budget_tokens: usize) -> String;
    fn name(&self) -> &'static str;
}

/// Default digestor: zero API cost, deterministic, no auth concerns. Reuses
/// `skeleton::build_digest`'s code-aware, signature-preserving,
/// prose-fallback logic -- stripping its `<axiom_context_digest>` wrapper
/// (designed for a different insertion point, the neural-fingerprint
/// digest) rather than duplicating or refactoring that module's internals,
/// so this ships with zero behavioral risk to `skeleton.rs`'s existing
/// callers/tests.
pub struct SkeletonDigestor;

/// Sentinel `state_hash` used only to locate the wrapper boundary in
/// `build_digest`'s output; never meaningful outside this function.
const WRAPPER_MARKER: &str = "axiom-cvm-s3-digest-marker";

impl Digestor for SkeletonDigestor {
    fn digest(&self, text: &str, budget_tokens: usize) -> String {
        // Heuristic ceiling on how many doc-comment lines the structural
        // skeleton keeps before falling back to pure truncation -- roughly
        // one doc line per 8 tokens of budget, floor 4 so tiny budgets
        // still get a little doc context.
        let max_doc_lines = (budget_tokens / 8).max(4);
        let wrapped = crate::skeleton::build_digest(
            text,
            "cvm-digest",
            0,
            0.0,
            WRAPPER_MARKER,
            max_doc_lines,
        );
        let body = extract_wrapped_body(&wrapped).unwrap_or(wrapped);
        truncate_to_token_budget(&body, budget_tokens)
    }

    fn name(&self) -> &'static str {
        "skeleton"
    }
}

/// Pull the body out of `skeleton::build_digest`'s
/// `<axiom_context_digest ...>...state_hash=X\n{body}\n</axiom_context_digest>`
/// wrapper, using [`WRAPPER_MARKER`] as the unambiguous anchor.
fn extract_wrapped_body(wrapped: &str) -> Option<String> {
    let start_marker = format!("state_hash={WRAPPER_MARKER}\n");
    let start = wrapped.find(&start_marker)? + start_marker.len();
    let end = wrapped.rfind("\n</axiom_context_digest>")?;
    (end >= start).then(|| wrapped[start..end].to_string())
}

/// Hard word-count truncation to `budget_tokens` (matches this crate's
/// whitespace-token convention, `anthropic_forwarder::whitespace_token_count`).
/// Cuts on a word boundary; appends an ellipsis only when something was
/// actually removed, so an already-short digest is untouched.
fn truncate_to_token_budget(text: &str, budget_tokens: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= budget_tokens {
        return text.to_string();
    }
    let mut out = words[..budget_tokens].join(" ");
    out.push_str(" …");
    out
}

/// Errors from the opt-in Haiku digestor -- always recoverable by falling
/// back to [`SkeletonDigestor`], never fatal to the request.
#[derive(Debug)]
pub enum HaikuDigestError {
    Network(String),
    Timeout,
    Upstream(String),
}

impl std::fmt::Display for HaikuDigestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HaikuDigestError::Network(m) => write!(f, "network error: {m}"),
            HaikuDigestError::Timeout => write!(f, "timed out"),
            HaikuDigestError::Upstream(m) => write!(f, "upstream error: {m}"),
        }
    }
}

/// Opt-in digestor (`AXIOM_CVM_DIGEST=haiku`): asks Claude Haiku 4.5 to
/// summarize, re-using the *current request's own* auth headers rather than
/// holding any credential of its own. Bills the user's account (~$0.02 per
/// heavy event) -- that is why it is opt-in, not the default.
///
/// Implemented as a standalone async function rather than a
/// [`Digestor`] impl: the trait is deliberately synchronous (matching
/// `SkeletonDigestor`, which does no I/O), but a real API call cannot be
/// synchronous inside a Tokio server. Callers gate on
/// `AXIOM_CVM_DIGEST=haiku`, await this directly, and fall back to
/// `SkeletonDigestor` on any `Err` -- headers are borrowed for the single
/// call and never stored.
pub async fn haiku_digest(
    forwarder: &crate::anthropic_forwarder::AnthropicForwarder,
    auth: &crate::anthropic_forwarder::ClientAuth,
    text: &str,
    budget_tokens: usize,
) -> Result<String, HaikuDigestError> {
    let prompt = format!(
        "Summarize the following content in at most {budget_tokens} words. Preserve concrete \
         facts, names, numbers, and code signatures; drop prose padding. Output only the \
         summary, no preamble.\n\n{text}"
    );
    let body = serde_json::json!({
        "model": "claude-haiku-4-5",
        "max_tokens": (budget_tokens * 2).clamp(64, 4096),
        "messages": [{"role": "user", "content": prompt}],
    });
    let call = forwarder.forward_messages_json(&body, auth);
    let resp = match tokio::time::timeout(std::time::Duration::from_secs(10), call).await {
        Ok(Ok(resp)) => resp,
        Ok(Err(e)) => return Err(HaikuDigestError::Upstream(e.to_string())),
        Err(_) => return Err(HaikuDigestError::Timeout),
    };
    let text = resp
        .get("content")
        .and_then(Value::as_array)
        .and_then(|blocks| blocks.first())
        .and_then(|b| b.get("text"))
        .and_then(Value::as_str)
        .ok_or_else(|| HaikuDigestError::Network("no text content in Haiku response".into()))?;
    Ok(truncate_to_token_budget(text, budget_tokens))
}

/// Default token threshold (`AXIOM_CVM_DIGEST_THRESHOLD_TOKENS`) above
/// which a `tool_result` block in the newest turn is digested.
pub const DEFAULT_DIGEST_THRESHOLD_TOKENS: usize = 4000;

/// One row appended to `checkpoints/cvm/faults.jsonl` every time
/// `POST /v1/expand` resolves a page id -- S7's training signal for
/// speculative prefetch. Field names/order match the blueprint's spec
/// verbatim.
#[derive(Serialize)]
struct FaultRow<'a> {
    session: &'a str,
    page_id: &'a str,
    turns_since_digest: u64,
}

fn faults_path() -> PathBuf {
    let root = std::env::var("AXIOM_CVM_DIR").unwrap_or_else(|_| "checkpoints/cvm".to_string());
    PathBuf::from(root).join("faults.jsonl")
}

/// Append one fault row. Best-effort: an I/O failure here must never fail
/// the `/v1/expand` request it's attached to, so this swallows errors after
/// logging once rather than propagating a `Result`.
pub fn append_fault(session: &str, page_id: &str, turns_since_digest: u64) {
    append_fault_to(&faults_path(), session, page_id, turns_since_digest);
}

/// Path-parameterized so tests can target an isolated temp file instead of
/// mutating the process-global `AXIOM_CVM_DIR` (which would race against
/// concurrent tests in this same binary that construct `AppState` and rely
/// on its default -- the same hazard class as the S1 `AXIOM_CACHE_SAFE`
/// race).
fn append_fault_to(path: &std::path::Path, session: &str, page_id: &str, turns_since_digest: u64) {
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("[axiom-cvm] failed to create faults.jsonl parent dir: {e}");
            return;
        }
    }
    let row = FaultRow {
        session,
        page_id,
        turns_since_digest,
    };
    let Ok(line) = serde_json::to_string(&row) else {
        return;
    };
    match OpenOptions::new().create(true).append(true).open(path) {
        Ok(mut f) => {
            let _ = writeln!(f, "{line}");
        }
        Err(e) => eprintln!("[axiom-cvm] failed to append fault row to {}: {e}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_fault_writes_a_well_formed_jsonl_row() {
        let dir = std::env::temp_dir().join(format!(
            "axiom-digest-fault-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("faults.jsonl");
        append_fault_to(&path, "sess-1", "a1b2c3d4e5f60718", 3);
        append_fault_to(&path, "sess-1", "0011223344556677", 7);

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        let row0: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(row0["session"], Value::from("sess-1"));
        assert_eq!(row0["page_id"], Value::from("a1b2c3d4e5f60718"));
        assert_eq!(row0["turns_since_digest"], Value::from(3));
        let row1: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(row1["turns_since_digest"], Value::from(7));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skeleton_digestor_respects_the_token_budget() {
        let code = (0..200)
            .map(|i| format!("pub fn function_{i}() {{\n    let x = {i};\n    x + 1\n}}\n"))
            .collect::<Vec<_>>()
            .join("\n");
        let digestor = SkeletonDigestor;
        let digest = digestor.digest(&code, 50);
        // +1 tolerance: the truncation marker ("…") is itself one
        // whitespace-split token, appended only when truncation actually
        // occurred (see truncate_to_token_budget_cuts_and_marks_when_over).
        assert!(
            digest.split_whitespace().count() <= 51,
            "digest must never exceed the requested token budget (+1 for the truncation marker)"
        );
    }

    #[test]
    fn skeleton_digestor_is_well_under_20pct_of_a_realistic_original() {
        let code = (0..300)
            .map(|i| format!("pub fn function_{i}(a: i32, b: i32) -> i32 {{\n    let sum = a + b + {i};\n    sum * 2\n}}\n"))
            .collect::<Vec<_>>()
            .join("\n");
        let original_tokens = code.split_whitespace().count();
        let budget = (original_tokens as f64 * 0.15) as usize;
        let digestor = SkeletonDigestor;
        let digest = digestor.digest(&code, budget);
        let digest_tokens = digest.split_whitespace().count();
        assert!(
            digest_tokens as f64 <= original_tokens as f64 * 0.20,
            "digest ({digest_tokens} tok) must stay <= 20% of original ({original_tokens} tok)"
        );
    }

    #[test]
    fn skeleton_digestor_keeps_code_signatures() {
        let code = "pub fn important_function(x: i32) -> i32 {\n    // a long implementation\n    let mut acc = 0;\n    for i in 0..x {\n        acc += i;\n    }\n    acc\n}\n";
        let digestor = SkeletonDigestor;
        let digest = digestor.digest(code, 100);
        assert!(digest.contains("important_function"));
    }

    #[test]
    fn skeleton_digestor_never_leaks_the_wrapper_or_marker() {
        let digestor = SkeletonDigestor;
        let digest = digestor.digest("plain prose with no code at all, just words.", 50);
        assert!(!digest.contains("axiom_context_digest"));
        assert!(!digest.contains(WRAPPER_MARKER));
    }

    #[test]
    fn skeleton_digestor_name_is_stable() {
        assert_eq!(SkeletonDigestor.name(), "skeleton");
    }

    #[test]
    fn truncate_to_token_budget_is_a_noop_under_budget() {
        let text = "short text under budget";
        assert_eq!(truncate_to_token_budget(text, 100), text);
    }

    #[test]
    fn truncate_to_token_budget_cuts_and_marks_when_over() {
        let text = "one two three four five six seven eight";
        let out = truncate_to_token_budget(text, 3);
        assert_eq!(out, "one two three …");
    }
}
