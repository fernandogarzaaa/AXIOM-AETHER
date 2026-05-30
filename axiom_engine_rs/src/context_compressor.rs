//! Active neural context compression via online TTT adaptation.
//!
//! This module is the heart of the "Active Neural Context Compressor"
//! pipeline: it owns the multi-tenant session map of fast-weight
//! tensors and exposes two operations that the server intercepts on
//! every incoming `/v1/messages` call when compression mode is active:
//!
//! 1. [`TttSessionStore::adapt_session`] — streams a token sequence
//!    through the native TTT layer stack, mutating per-layer W̃ in
//!    place via the online MSE-on-reconstruction gradient step.
//! 2. [`TttSessionStore::extract_memory_vector`] — runs the user's
//!    query through the adapted state (the associative recall pass)
//!    and projects the result into a dense [`MemoryFingerprint`].
//!
//! The fingerprint is then serialised into a structured text block
//! and prepended to the outbound Anthropic payload in place of the
//! raw context that was just absorbed locally.
//!
//! ## Honesty footnote (read this before benchmarking)
//!
//! The fingerprint encodes deterministic statistics of the adapted
//! W̃ tensors plus the top-k logits produced by an associative recall
//! pass. With a *trained* LM head, those top-k indices decode through
//! the tokenizer to genuine next-token predictions — a real semantic
//! compression. With the random-init checkpoint shipped in this repo,
//! the indices are essentially noise. The pipeline is wired correctly
//! either way; the value of the compression depends entirely on
//! whether the projection weights have been meta-trained (see
//! `src/meta_train.rs`).

use std::sync::Arc;
use std::time::Instant;

use candle_core::{Result as CResult, Tensor, D};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex as AsyncMutex;

use crate::inference::InferencePipeline;

/// One session's per-layer fast-weight state, behind an async mutex so
/// the same session can be queued for sequential updates without
/// blocking the whole server.
pub type SessionStates = Arc<AsyncMutex<Vec<Tensor>>>;

/// Multi-tenant in-memory store of adapted W̃ states keyed by session id.
///
/// Uses [`DashMap`] for lock-free reads on the common path; per-session
/// mutation is serialised by the inner [`AsyncMutex`].
pub struct TttSessionStore {
    sessions: DashMap<String, SessionStates>,
}

impl TttSessionStore {
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
        }
    }

    /// Resolve an existing session or create a fresh one with
    /// identity-initialised W̃ matrices.
    pub fn get_or_create(
        &self,
        session_id: &str,
        pipeline: &InferencePipeline,
    ) -> CResult<SessionStates> {
        if let Some(s) = self.sessions.get(session_id) {
            return Ok(s.clone());
        }
        // Build the fresh state outside the map, then insert-if-absent
        // atomically via the entry API so two concurrent first-touch requests
        // for the same session can't both insert and clobber each other's W̃.
        let states = pipeline.init_session_states()?;
        let arc = Arc::new(AsyncMutex::new(states));
        let entry = self
            .sessions
            .entry(session_id.to_string())
            .or_insert(arc);
        Ok(entry.value().clone())
    }

    /// Number of live sessions; used by the metrics + stats endpoints.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Remove a single session and return its state handle, so the caller can
    /// (e.g.) EMA-merge the adapted W̃ into a master before it is freed.
    pub fn take_session(&self, session_id: &str) -> Option<SessionStates> {
        self.sessions.remove(session_id).map(|(_, states)| states)
    }

    /// Snapshot handles to every live session without removing them. Cheap —
    /// clones `Arc`s only. Used to flush all sessions into the master vibe on
    /// graceful shutdown.
    pub fn snapshot_handles(&self) -> Vec<(String, SessionStates)> {
        self.sessions
            .iter()
            .map(|kv| (kv.key().clone(), kv.value().clone()))
            .collect()
    }

    /// Drop every session.
    pub fn clear(&self) {
        self.sessions.clear();
    }
}

impl Default for TttSessionStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MemoryFingerprint
// ---------------------------------------------------------------------------

/// Dense semantic-layout fingerprint extracted from the adapted state.
///
/// Includes both deterministic state-statistics (so the recipient can
/// verify the W̃ snapshot) and the result of the associative recall
/// pass (top-k tokens predicted by the model after the context was
/// absorbed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFingerprint {
    pub schema: String,
    pub session_id: String,
    pub context_tokens_processed: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub state_hash: String,
    /// Frobenius norm per layer of W̃ after adaptation.
    pub layer_frobenius_norms: Vec<f32>,
    /// L2 norm of the recall-pass output vector.
    pub recall_norm: f32,
    /// L1 norm of the recall-pass output vector.
    pub recall_l1: f32,
    /// Top-k vocabulary indices ranked by absolute recall-pass logit.
    pub recall_top_k_indices: Vec<u32>,
    /// Tokenizer decode of `recall_top_k_indices` when a real tokenizer
    /// is loaded. Empty string when the hash fallback is active.
    pub recall_top_k_decoded: String,
    /// Wall-clock milliseconds spent in adapt + recall.
    pub elapsed_ms: u128,
}

impl MemoryFingerprint {
    /// Format the fingerprint as a strict XML/Markdown block suitable for
    /// prepending to a downstream frontier-model prompt.
    ///
    /// The block carries (a) machine-readable recall-layout fields and
    /// (b) explicit decode instructions telling the downstream model how
    /// to interpret the compressed semantic horizon. The outer element is
    /// a single, well-formed `<axiom_context_fingerprint>` tag with
    /// `session_id` / `tokens_compressed` attributes so the model can
    /// reliably locate the boundaries.
    pub fn to_prompt_block(&self) -> String {
        let layer_norm_summary = if self.layer_frobenius_norms.len() <= 8 {
            format!("{:?}", self.layer_frobenius_norms)
        } else {
            let head: Vec<f32> = self.layer_frobenius_norms.iter().take(4).copied().collect();
            let tail: Vec<f32> = self
                .layer_frobenius_norms
                .iter()
                .rev()
                .take(4)
                .copied()
                .collect();
            format!("first4={head:?} last4={tail:?}")
        };
        let decoded_line = if self.recall_top_k_decoded.is_empty() {
            String::new()
        } else {
            format!("recall_top_k_decoded={:?}\n", self.recall_top_k_decoded)
        };
        // NOTE: `state_hash=` is intentionally emitted at the start of its
        // own line so external tooling can grep the snapshot id verbatim.
        format!(
            "<axiom_context_fingerprint session_id=\"{session}\" tokens_compressed=\"{tokens}\" schema=\"{schema}\">\n\
             <recall_layout>\n\
             associative_recall_norm={norm:.6}\n\
             associative_recall_l1={l1:.6}\n\
             recall_top_k_indices={top:?}\n\
             {decoded}layers={layers} d_model={d}\n\
             layer_frobenius_norms={norms}\n\
             state_hash={hash}\n\
             compression_ms={ms}\n\
             </recall_layout>\n\
             <decode_instructions>\n\
             The block above is a lossy neural compression of prior heavy context that was\n\
             ingested locally through online test-time training (TTT). It is NOT raw text:\n\
             the original tokens were streamed through per-session fast-weight matrices (W̃),\n\
             and `recall_top_k_indices` are the vocabulary ids the adapted state most strongly\n\
             predicts when queried — i.e. a distilled semantic pointer into that context.\n\
             To decode: (1) treat `recall_top_k_decoded` (when present) and the top-k ids as\n\
             the salient topics/entities of the compressed material; (2) use\n\
             `associative_recall_norm` as a confidence signal — values near 0 mean weak recall,\n\
             higher magnitudes mean a sharp, well-conditioned memory; (3) `state_hash` uniquely\n\
             identifies this W̃ snapshot for this session, so identical hashes imply identical\n\
             absorbed context. Infer the user's intent over this compressed horizon and answer\n\
             the query that follows this block. If recall confidence is low, ask a brief\n\
             clarifying question rather than hallucinating the compressed content.\n\
             </decode_instructions>\n\
             </axiom_context_fingerprint>",
            session = self.session_id,
            tokens = self.context_tokens_processed,
            schema = self.schema,
            layers = self.n_layers,
            d = self.d_model,
            norms = layer_norm_summary,
            norm = self.recall_norm,
            l1 = self.recall_l1,
            top = self.recall_top_k_indices,
            decoded = decoded_line,
            hash = self.state_hash,
            ms = self.elapsed_ms,
        )
    }
}

// ---------------------------------------------------------------------------
// Adaptation + extraction
// ---------------------------------------------------------------------------

/// Stream a token sequence through the model's TTT layer stack,
/// mutating `states` in place with one online gradient step per token.
///
/// **Must be called from inside `tokio::task::spawn_blocking`** — the
/// inner loop is CPU-bound and would otherwise stall the async runtime.
pub fn adapt_session_blocking(
    pipeline: &InferencePipeline,
    states: &mut [Tensor],
    token_ids: &[u32],
) -> CResult<()> {
    if token_ids.is_empty() {
        return Ok(());
    }
    let device = pipeline.device();
    let input = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), device)?;
    let _logits = pipeline.model().forward_lm(&input, states)?;

    // Autograd truncation: detach each updated W̃ from its op-graph so history
    // does not accumulate across windows/calls. Without this, streaming a large
    // corpus builds a graph as deep as the token count; dropping that chain
    // recurses past the stack limit and crashes (observed on a 24k-token
    // saturation run). The TTT update is closed-form (see `forward_native` —
    // no `.backward()`), so the state VALUE is unchanged and nothing that is
    // ever backpropagated is lost. `detach` is infallible in candle 0.8.
    for state in states.iter_mut() {
        *state = state.detach();
    }
    Ok(())
}

/// Run the associative recall pass and extract the fingerprint.
///
/// The query tokens are streamed through the adapted state — same
/// `forward_lm` path used during adaptation, so we continue applying
/// the TTT update, but on a short query rather than the bulk context.
/// The final-token logits are then summarised into the fingerprint.
///
/// **Must also be called from inside `tokio::task::spawn_blocking`.**
pub fn extract_memory_vector_blocking(
    pipeline: &InferencePipeline,
    states: &mut [Tensor],
    query_token_ids: &[u32],
    session_id: &str,
    context_tokens_processed: usize,
    started: Instant,
    top_k: usize,
) -> CResult<MemoryFingerprint> {
    let n_layers = states.len();
    let d_model = pipeline.model().config.d_model;
    let device = pipeline.device();

    // Associative recall: project query through the adapted state. This
    // `forward_lm` mutates `states` in place (one more TTT step per query
    // token), so the hash/norms below are computed AFTER it to stay
    // consistent with the state actually left stored for the next request.
    let recall_query: Vec<u32> = if query_token_ids.is_empty() {
        vec![0]
    } else {
        query_token_ids.to_vec()
    };
    let q_len = recall_query.len();
    let q_input = Tensor::from_vec(recall_query.clone(), (1, q_len), device)?;
    let logits = pipeline.model().forward_lm(&q_input, states)?; // [1, T, vocab]
    // Take the final-token logit vector.
    let final_logits = logits.narrow(1, q_len - 1, 1)?.squeeze(1)?.squeeze(0)?;
    let final_vec: Vec<f32> = final_logits.to_vec1::<f32>()?;

    // Deterministic state hash + Frobenius norms over the post-recall W̃.
    let mut hasher = Sha256::new();
    let mut layer_frobenius_norms: Vec<f32> = Vec::with_capacity(n_layers);
    for layer in states.iter() {
        let flat: Vec<f32> = layer.flatten_all()?.to_vec1::<f32>()?;
        let mut sq_sum: f32 = 0.0;
        for v in &flat {
            sq_sum += v * v;
            hasher.update(v.to_le_bytes());
        }
        layer_frobenius_norms.push(sq_sum.sqrt());
    }
    let state_hash = format!("sha256:{:x}", hasher.finalize());

    let recall_norm: f32 = final_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let recall_l1: f32 = final_vec.iter().map(|x| x.abs()).sum();

    let mut ranked: Vec<(usize, f32)> = final_vec
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.abs()))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let recall_top_k_indices: Vec<u32> = ranked
        .iter()
        .take(top_k)
        .map(|(i, _)| *i as u32)
        .collect();

    let recall_top_k_decoded = if pipeline.has_real_tokenizer() {
        pipeline.decode_tokens(&recall_top_k_indices)
    } else {
        String::new()
    };

    // Argmax-based associative recall is intentionally driven through the
    // existing forward path (not a separate code path) so that meta-trained
    // projection matrices apply identically here and during inference.
    let _ = final_logits.argmax(D::Minus1).ok();

    Ok(MemoryFingerprint {
        schema: "axiom-ttt-context-fingerprint/v2".to_string(),
        session_id: session_id.to_string(),
        context_tokens_processed,
        n_layers,
        d_model,
        state_hash,
        layer_frobenius_norms,
        recall_norm,
        recall_l1,
        recall_top_k_indices,
        recall_top_k_decoded,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Static knobs for the compression pipeline. Read from environment
/// once at server startup.
#[derive(Debug, Clone)]
pub struct CompressorConfig {
    /// Token threshold above which a message is treated as "heavy" and
    /// routed through the local TTT compressor.
    pub heavy_message_threshold_tokens: usize,
    /// Number of top-k indices to record in the fingerprint.
    pub recall_top_k: usize,
    /// Whether compression mode is enabled.
    pub enabled: bool,
}

impl CompressorConfig {
    pub fn from_env() -> Self {
        let enabled = std::env::var("AXIOM_TTT_COMPRESS")
            .map(|v| {
                matches!(
                    v.to_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        let threshold = std::env::var("AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512);
        let top_k = std::env::var("AXIOM_TTT_COMPRESS_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(32);
        Self {
            enabled,
            heavy_message_threshold_tokens: threshold,
            recall_top_k: top_k,
        }
    }
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            heavy_message_threshold_tokens: 512,
            recall_top_k: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AxiomConfig;
    use crate::inference::InferencePipeline;
    use candle_core::Device;

    fn tiny_pipeline() -> InferencePipeline {
        let cfg = AxiomConfig {
            d_model: 16,
            n_layers: 2,
            vocab_size: 64,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        };
        InferencePipeline::new(cfg, Device::Cpu).expect("tiny pipeline must build")
    }

    #[test]
    fn store_creates_and_caches_sessions() {
        let store = TttSessionStore::new();
        let pipeline = tiny_pipeline();
        let s1 = store.get_or_create("sess-a", &pipeline).unwrap();
        let s2 = store.get_or_create("sess-a", &pipeline).unwrap();
        assert!(Arc::ptr_eq(&s1, &s2));
        assert_eq!(store.len(), 1);

        let _ = store.get_or_create("sess-b", &pipeline).unwrap();
        assert_eq!(store.len(), 2);
        assert!(store.take_session("sess-a").is_some());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn adapt_session_mutates_states() {
        let pipeline = tiny_pipeline();
        let mut states = pipeline.init_session_states().unwrap();
        let snapshot: Vec<Vec<f32>> = states
            .iter()
            .map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .collect();
        let token_ids: Vec<u32> = (0..16).collect();
        adapt_session_blocking(&pipeline, &mut states, &token_ids).unwrap();
        let after: Vec<Vec<f32>> = states
            .iter()
            .map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .collect();
        assert_ne!(snapshot, after, "W_tilde must move after adapt_session");
    }

    #[test]
    fn extract_memory_vector_emits_fingerprint() {
        let pipeline = tiny_pipeline();
        let mut states = pipeline.init_session_states().unwrap();
        let context_tokens: Vec<u32> = (0..32).collect();
        adapt_session_blocking(&pipeline, &mut states, &context_tokens).unwrap();

        let query_tokens = vec![1u32, 2, 3, 4];
        let started = Instant::now();
        let fp = extract_memory_vector_blocking(
            &pipeline,
            &mut states,
            &query_tokens,
            "sess-x",
            context_tokens.len(),
            started,
            8,
        )
        .unwrap();

        assert_eq!(fp.session_id, "sess-x");
        assert_eq!(fp.context_tokens_processed, 32);
        assert_eq!(fp.n_layers, 2);
        assert_eq!(fp.d_model, 16);
        assert_eq!(fp.layer_frobenius_norms.len(), 2);
        assert_eq!(fp.recall_top_k_indices.len(), 8);
        assert!(fp.state_hash.starts_with("sha256:"));
        assert!(fp.recall_norm.is_finite());
        assert!(fp.recall_l1.is_finite());
    }

    #[test]
    fn fingerprint_to_prompt_block_is_well_formed() {
        let fp = MemoryFingerprint {
            schema: "axiom-ttt-context-fingerprint/v2".into(),
            session_id: "sess-y".into(),
            context_tokens_processed: 1234,
            n_layers: 4,
            d_model: 16,
            state_hash: "sha256:deadbeef".into(),
            layer_frobenius_norms: vec![1.0, 2.0, 3.0, 4.0],
            recall_norm: 0.5,
            recall_l1: 1.25,
            recall_top_k_indices: vec![1, 7, 13],
            recall_top_k_decoded: "alpha beta gamma".into(),
            elapsed_ms: 42,
        };
        let block = fp.to_prompt_block();
        // Well-formed XML envelope with locating attributes.
        assert!(block.starts_with("<axiom_context_fingerprint "));
        assert!(block.contains("session_id=\"sess-y\""));
        assert!(block.contains("tokens_compressed=\"1234\""));
        assert!(block.contains("schema=\"axiom-ttt-context-fingerprint/v2\""));
        // Recall layout + decode-instruction sections present.
        assert!(block.contains("<recall_layout>"));
        assert!(block.contains("</recall_layout>"));
        assert!(block.contains("<decode_instructions>"));
        assert!(block.contains("state_hash=sha256:deadbeef"));
        assert!(block.contains("alpha beta gamma"));
        assert!(block.trim_end().ends_with("</axiom_context_fingerprint>"));
    }

    #[test]
    fn compressor_config_env_off_by_default() {
        std::env::remove_var("AXIOM_TTT_COMPRESS");
        let cfg = CompressorConfig::from_env();
        assert!(!cfg.enabled);
        assert_eq!(cfg.heavy_message_threshold_tokens, 512);
    }
}
