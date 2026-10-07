//! S3 (CVM cost stack): digest admission control -- the big lever.
//!
//! Prevention beats compression: heavy tool results should never enter the
//! (expensive, cached-forever) transcript at full size. On arrival, in the
//! newest turn only (by definition after every cache breakpoint and not yet
//! cached), replace heavy content with a digest + stub; the full text goes
//! to the L2 store (`cvm_store`). Cache-safe by construction (S1 froze the
//! prefix) and needs no determinism trick: the digest is created exactly
//! once and then it IS the history. Defaults to `skeleton` since S5's live
//! eval passed on 2026-07-11 (12/12 -> 11/12 correctness, 0% fault rate,
//! cost strictly lower); `AXIOM_CVM_DIGEST=off` opts back out. See
//! docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S3, and
//! bench/cvm/RESULTS-2026-07-11.md.

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
        // PageRank-ranked skeletonization: orders symbols by structural
        // importance and keeps the highest-ranked ones within budget,
        // instead of the old arbitrary word-truncation. Falls back to the
        // legacy build_digest path if the ranked output is empty.
        let lang = detect_code_language(text);
        let ranked = crate::skeleton::skeletonize_ranked(text, lang, Some(budget_tokens));
        if !ranked.trim().is_empty() {
            // Safety net: the ranked budget uses chars/4 approximation, but
            // this trait's contract is whitespace tokens. Truncate if needed.
            return truncate_to_token_budget(&ranked, budget_tokens);
        }
        // Fallback: legacy build_digest + truncation path.
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

impl SkeletonDigestor {
    /// Surprise-weighted digest: before compressing the chunk, feed its TTT
    /// update norm into `triage`, score it against the session's running
    /// distribution, and select the compression level adaptively.
    ///
    /// Returns the compressed text and the [`CompressionLevel`] that was
    /// chosen, so callers can log/meter the triage decision.
    ///
    /// Note: the [`CompressionLevel::Verbatim`] tier returns the chunk
    /// unchanged — the chunk earned its tokens by being highly surprising.
    /// Callers budgeting across many chunks should account for verbatim
    /// chunks when sizing the total budget.
    pub fn digest_with_triage(
        &self,
        text: &str,
        budget_tokens: usize,
        triage: &mut crate::surprise_triage::SurpriseTriage,
        update_norm: f32,
    ) -> (String, crate::surprise_triage::CompressionLevel) {
        let lang = detect_code_language(text);
        triage.triage_compress(text, budget_tokens, update_norm, lang)
    }
}

/// Heuristic language detection for ranked skeletonization.
/// Returns "rust" for Rust-like code, "" (generic path) otherwise.
/// The ranked skeletonizer uses tree-sitter for Rust and a
/// language-agnostic heuristic for everything else.
fn detect_code_language(text: &str) -> &'static str {
    // Strong Rust signals: `fn ` definitions plus either `struct`/`impl`
    // blocks or `use ...::` imports. Standalone `fn` (e.g. `fn main()`)
    // is valid Rust; don't require struct/impl/use as well.
    let has_fn = text.contains("fn ");
    if has_fn {
        "rust"
    } else {
        ""
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

// ---------------------------------------------------------------------------
// KV-Merge: merge-don't-evict session digests
// ---------------------------------------------------------------------------
// When a bounded digest store must shed old chunks, the naive policy drops
// them outright (similarity with the originals collapses to 0.0). KV-Merge
// instead preserves a lossy trace: evicted chunks are folded into a running
// summary via attention-weighted averaging of their embeddings, while a
// recent window stays exact (progressive resolution: far past coarse,
// recent detailed).
//
// There was no pre-existing chunk-eviction call site in the digest/skeleton
// modules to retrofit; `MergeDigestStore` is the bounded store whose
// eviction path merges instead of dropping, and `merge_chunks` is the
// merge primitive it (and future callers) use.

use std::collections::VecDeque;

/// Maximum characters retained in a merged summary's text. The embedding
/// carries the semantic trace; the text is a human-readable tail kept
/// bounded so summaries cannot grow without limit.
pub const MERGED_TEXT_BUDGET_CHARS: usize = 2000;

/// A digest chunk: human-readable text plus its embedding and an
/// attention/importance weight used by [`merge_chunks`].
#[derive(Debug, Clone, PartialEq)]
pub struct DigestChunk {
    /// Human-readable content (or summary) of the chunk.
    pub text: String,
    /// Embedding/fingerprint vector. All chunks merged together must share
    /// the same dimensionality; mismatched chunks are skipped defensively.
    pub embedding: Vec<f32>,
    /// Attention weight for the merge. Defaults to `1.0`; callers may set
    /// higher values for chunks the attention mechanism scored as important.
    pub weight: f32,
}

impl DigestChunk {
    /// Convenience constructor with unit weight.
    pub fn new(text: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            text: text.into(),
            embedding,
            weight: 1.0,
        }
    }
}

/// Cosine similarity between two embedding vectors, in [-1, 1].
/// Returns `0.0` when either vector is empty, all-zero, or the lengths
/// differ (no meaningful comparison possible).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

/// Attention-weighted merge of `old` (evicted) chunks with `new` (the
/// incoming chunk, or the running summary accumulator).
///
/// The merged embedding is the attention-weighted average over every
/// participating chunk: `Σ w_j * e_j / Σ w_j`, using each chunk's `weight`
/// (non-positive weights are treated as zero). Chunks whose embedding
/// dimensionality differs from `new`'s are skipped defensively. The merged
/// text concatenates participant texts (oldest first, `new` last), keeping
/// the most recent [`MERGED_TEXT_BUDGET_CHARS`] characters.
///
/// When `old` is empty the result is a clone of `new` — merging must be a
/// no-op, never a lossy rewrite, when nothing is being evicted.
pub fn merge_chunks(old: &[DigestChunk], new: &DigestChunk) -> DigestChunk {
    let dim = new.embedding.len();
    let mut total_weight = new.weight.max(0.0);
    let mut acc = vec![0.0f32; dim];
    for (i, v) in new.embedding.iter().enumerate() {
        acc[i] = v * total_weight;
    }
    let mut texts: Vec<&str> = Vec::with_capacity(old.len() + 1);
    for chunk in old {
        if chunk.embedding.len() != dim {
            continue; // defensive: never average across mismatched spaces
        }
        let w = chunk.weight.max(0.0);
        if w <= 0.0 && !chunk.embedding.is_empty() {
            // Zero-weight chunks contribute nothing to the embedding but
            // their text is still folded into the summary below.
        }
        total_weight += w;
        for (i, v) in chunk.embedding.iter().enumerate() {
            acc[i] += v * w;
        }
        texts.push(chunk.text.as_str());
    }
    texts.push(new.text.as_str());
    if total_weight > 0.0 {
        for v in acc.iter_mut() {
            *v /= total_weight;
        }
    }
    let mut text = texts.join("\n---\n");
    if text.len() > MERGED_TEXT_BUDGET_CHARS {
        let cut = text.len() - MERGED_TEXT_BUDGET_CHARS;
        // Cut on a char boundary; keep the most recent tail.
        let mut idx = cut;
        while idx < text.len() && !text.is_char_boundary(idx) {
            idx += 1;
        }
        text = format!("…{}", &text[idx..]);
    }
    DigestChunk {
        text,
        embedding: acc,
        weight: total_weight,
    }
}

/// Bounded digest store with merge-on-evict.
///
/// The `recent_window` newest chunks are kept exact (verbatim text and
/// embedding). When capacity is exceeded, the oldest chunk is evicted — but
/// instead of being dropped, it is folded into a running `summary` via
/// [`merge_chunks`]. Progressive resolution: far past coarse (one merged
/// summary), recent detailed (exact chunks).
///
/// `capacity` must exceed `recent_window`; both are at least 1.
pub struct MergeDigestStore {
    capacity: usize,
    recent_window: usize,
    recent: VecDeque<DigestChunk>,
    summary: Option<DigestChunk>,
}

impl MergeDigestStore {
    /// Create a store holding at most `capacity` chunks, of which the
    /// newest `recent_window` stay exact. Panics on `capacity == 0`,
    /// `recent_window == 0`, or `recent_window > capacity`.
    pub fn new(capacity: usize, recent_window: usize) -> Self {
        assert!(capacity >= 1, "capacity must be >= 1");
        assert!(recent_window >= 1, "recent_window must be >= 1");
        assert!(
            recent_window <= capacity,
            "recent_window must not exceed capacity"
        );
        Self {
            capacity,
            recent_window,
            recent: VecDeque::new(),
            summary: None,
        }
    }

    /// Push a chunk. If the store is full, the oldest chunk is evicted and
    /// merged into the running summary instead of being dropped. The
    /// `recent_window` newest chunks are always exact.
    pub fn push(&mut self, chunk: DigestChunk) {
        // Merge-on-evict: fold the outgoing chunk into the summary BEFORE
        // it leaves, so no information is silently discarded.
        if self.recent.len() >= self.capacity {
            if let Some(evicted) = self.recent.pop_front() {
                self.summary = Some(match self.summary.take() {
                    Some(acc) => merge_chunks(&[evicted], &acc),
                    None => evicted,
                });
            }
        }
        self.recent.push_back(chunk);
    }

    /// The merged summary of all evicted chunks, or `None` if nothing has
    /// been evicted yet.
    pub fn summary(&self) -> Option<&DigestChunk> {
        self.summary.as_ref()
    }

    /// The exact-recent window size: the newest this many chunks are always
    /// kept exact (eviction only ever removes the oldest).
    pub fn recent_window(&self) -> usize {
        self.recent_window
    }

    /// The exact recent window, oldest first.
    pub fn recent(&self) -> &VecDeque<DigestChunk> {
        &self.recent
    }

    /// Total chunks retained (exact recent + 1 if a summary exists).
    pub fn len(&self) -> usize {
        self.recent.len() + usize::from(self.summary.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.recent.is_empty() && self.summary.is_none()
    }
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

    // ------------------------------------------------------------------
    // KV-Merge tests
    // ------------------------------------------------------------------

    fn kv_chunk(text: &str, embedding: Vec<f32>, weight: f32) -> DigestChunk {
        DigestChunk {
            text: text.to_string(),
            embedding,
            weight,
        }
    }

    #[test]
    fn cosine_similarity_identical_is_one() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_mismatched_or_empty_is_zero() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn merge_empty_old_returns_new_unchanged() {
        let new = kv_chunk("incoming", vec![0.5, 0.5], 2.0);
        let merged = merge_chunks(&[], &new);
        assert_eq!(merged.text, "incoming");
        assert_eq!(merged.embedding, vec![0.5, 0.5]);
        assert_eq!(merged.weight, 2.0);
    }

    #[test]
    fn merge_preserves_cosine_similarity_above_08() {
        // Three chunks about the same topic: embeddings clustered together.
        // A pure-drop eviction retains similarity 0.0 with the evicted
        // chunks; the merged summary must stay above 0.8 with each.
        let old = vec![
            kv_chunk("auth module handles login", vec![1.0, 0.1, 0.05], 1.0),
            kv_chunk("auth module handles logout", vec![0.95, 0.12, 0.08], 1.0),
        ];
        let new = kv_chunk("auth module session refresh", vec![0.98, 0.09, 0.06], 1.0);
        let merged = merge_chunks(&old, &new);
        for chunk in old.iter().chain(std::iter::once(&new)) {
            let sim = cosine_similarity(&merged.embedding, &chunk.embedding);
            assert!(
                sim > 0.8,
                "merged summary must preserve similarity > 0.8 with '{}', got {sim:.4}",
                chunk.text
            );
        }
        // Deletion baseline: a dropped chunk leaves no vector behind.
        let dropped_similarity = 0.0f32;
        assert!(dropped_similarity < 0.8);
    }

    #[test]
    fn merge_weights_bias_toward_high_attention() {
        // Chunk A has 3x the attention weight of chunk B; the merged
        // embedding must sit closer to A than to B.
        let a = kv_chunk("important", vec![1.0, 0.0], 3.0);
        let b = kv_chunk("background", vec![0.0, 1.0], 1.0);
        let merged = merge_chunks(std::slice::from_ref(&b), &a);
        let sim_a = cosine_similarity(&merged.embedding, &a.embedding);
        let sim_b = cosine_similarity(&merged.embedding, &b.embedding);
        assert!(
            sim_a > sim_b,
            "merged must be closer to high-weight chunk (sim_a={sim_a:.3}, sim_b={sim_b:.3})"
        );
    }

    #[test]
    fn merge_skips_mismatched_dimensions() {
        let bad = kv_chunk("wrong dim", vec![1.0, 2.0, 3.0], 1.0);
        let new = kv_chunk("good", vec![1.0, 0.0], 1.0);
        let merged = merge_chunks(&[bad], &new);
        assert_eq!(merged.embedding.len(), 2);
        assert!((cosine_similarity(&merged.embedding, &new.embedding) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn store_keeps_recent_window_exact() {
        // capacity 4, recent_window 2, push 6 chunks: the two newest must be
        // byte-exact; older ones fold into the summary instead of vanishing.
        let mut store = MergeDigestStore::new(4, 2);
        for i in 0..6 {
            store.push(kv_chunk(
                &format!("chunk-{i}"),
                vec![i as f32, 1.0],
                1.0,
            ));
        }
        let recent: Vec<&str> = store
            .recent()
            .iter()
            .map(|c| c.text.as_str())
            .collect();
        assert_eq!(recent, vec!["chunk-2", "chunk-3", "chunk-4", "chunk-5"]);
        // The two newest are exactly as pushed (text AND embedding).
        let last = store.recent().back().unwrap();
        assert_eq!(last.text, "chunk-5");
        assert_eq!(last.embedding, vec![5.0, 1.0]);
        let prev = &store.recent()[store.recent().len() - 2];
        assert_eq!(prev.text, "chunk-4");
        assert_eq!(prev.embedding, vec![4.0, 1.0]);
    }

    #[test]
    fn store_summary_covers_evicted_chunks() {
        let mut store = MergeDigestStore::new(2, 1);
        store.push(kv_chunk("first", vec![1.0, 0.0], 1.0));
        store.push(kv_chunk("second", vec![0.0, 1.0], 1.0));
        assert!(store.summary().is_none(), "nothing evicted yet");
        store.push(kv_chunk("third", vec![1.0, 1.0], 1.0));
        let summary = store.summary().expect("evicted chunk must merge into summary");
        assert!(
            summary.text.contains("first"),
            "summary must retain a trace of the evicted chunk, got: {}",
            summary.text
        );
        // Merged embedding is the average of [1,0] (evicted "first").
        assert_eq!(summary.embedding, vec![1.0, 0.0]);
    }

    #[test]
    fn store_summary_accumulates_across_evictions() {
        let mut store = MergeDigestStore::new(2, 1);
        for (i, text) in ["a", "b", "c", "d"].iter().enumerate() {
            store.push(kv_chunk(text, vec![i as f32, 0.0], 1.0));
        }
        // Evicted: a, b. Summary = merge([b], a) -> texts "b\n---\na",
        // embedding = average of [1,0] and [0,0] = [0.5, 0].
        let summary = store.summary().unwrap();
        assert!(summary.text.contains('a') && summary.text.contains('b'));
        assert!((summary.embedding[0] - 0.5).abs() < 1e-6);
        assert_eq!(store.len(), 3); // 2 exact + 1 summary
    }
}
