//! search_ingest.rs — Just-In-Time (JIT) search-reasoning ingestion.
//!
//! Transitions Axiom from a static code compressor into a live search node.
//! Scraped web text (from scripts/lib/axiom-scrape.js) is streamed into the
//! local BPE tokenizer, absorbed by an online Test-Time Training pass over
//! detached <=512-token chunks (protecting the RTX 2060's VRAM), and distilled
//! into an `<axiom_search_fingerprint>` — a dense, zero-token-waste semantic
//! pointer a lightweight local router can use WITHOUT calling a large external
//! LLM.

use std::time::Instant;

use candle_core::Result;

use crate::context_compressor::{adapt_session_blocking, extract_memory_vector_blocking};
use crate::inference::InferencePipeline;

/// Hard cap on tokens per TTT adaptation window (VRAM safety on a 6 GB 2060).
pub const SEARCH_CHUNK: usize = 512;
/// Floor below which we stop halving the chunk on memory pressure.
const MIN_CHUNK: usize = 32;

/// Distilled semantic pointer over a body of scraped web text.
pub struct SearchFingerprint {
    pub query: String,
    pub tokens_ingested: usize,
    pub chunks: usize,
    pub recall_norm: f32,
    pub recall_l1: f32,
    pub recall_top_k_indices: Vec<u32>,
    pub recall_top_k_decoded: String,
    pub state_hash: String,
    pub elapsed_ms: u128,
}

impl SearchFingerprint {
    /// Serialise the dense wire payload. The decoded top-k topics are the
    /// engine's high-confidence read of the scraped material; a local router
    /// can act on them without re-sending the raw text to a frontier model.
    pub fn to_wire(&self) -> String {
        let topics = if self.recall_top_k_decoded.trim().is_empty() {
            format!("(indices) {:?}", self.recall_top_k_indices)
        } else {
            self.recall_top_k_decoded.clone()
        };
        format!(
            "<axiom_search_fingerprint query=\"{query}\" tokens_ingested=\"{tokens}\" chunks=\"{chunks}\" schema=\"axiom-search/v1\">\n\
             recall_norm={norm:.4} recall_l1={l1:.4}\n\
             state_hash={hash}\n\
             recall_top_k_topics={topics}\n\
             <decode_instructions>\n\
             The engine performed online test-time training over live scraped web\n\
             results for the query above. `recall_top_k_topics` are the highest-\n\
             confidence tokens the adapted fast-weights predict — a distilled\n\
             semantic pointer into the scraped material. Treat them as the salient\n\
             entities/answers. Use `recall_norm` as a confidence signal (higher =\n\
             sharper memory). Answer the user's query from these topics; only\n\
             escalate to a larger model if recall confidence is low.\n\
             </decode_instructions>\n\
             </axiom_search_fingerprint>",
            query = self.query.replace('"', "'"),
            tokens = self.tokens_ingested,
            chunks = self.chunks,
            norm = self.recall_norm,
            l1 = self.recall_l1,
            hash = self.state_hash,
            topics = topics,
        )
    }
}

/// Adapt over a chunk, halving the window on memory pressure (OOM defense).
fn adapt_resilient(
    pipeline: &InferencePipeline,
    states: &mut [candle_core::Tensor],
    chunk: &[u32],
) -> Result<usize> {
    if chunk.is_empty() {
        return Ok(0);
    }
    match adapt_session_blocking(pipeline, states, chunk) {
        Ok(()) => Ok(1),
        Err(e) => {
            if chunk.len() <= MIN_CHUNK {
                return Err(e);
            }
            // Likely VRAM/RAM pressure: split and retry with smaller windows.
            eprintln!(
                "[search-ingest] adapt failed on {}-token chunk ({e}); halving window",
                chunk.len()
            );
            let mid = chunk.len() / 2;
            let a = adapt_resilient(pipeline, states, &chunk[..mid])?;
            let b = adapt_resilient(pipeline, states, &chunk[mid..])?;
            Ok(a + b)
        }
    }
}

/// Ingest scraped `text` for `query`: BPE-tokenize → chunked online TTT →
/// recall pass → `SearchFingerprint`.
pub fn ingest_search_text(
    pipeline: &InferencePipeline,
    query: &str,
    text: &str,
    top_k: usize,
) -> Result<SearchFingerprint> {
    let started = Instant::now();
    let ids = pipeline.encode_text(text);
    let tokens_ingested = ids.len();

    let mut states = pipeline.init_session_states()?;
    let mut chunks = 0usize;
    for chunk in ids.chunks(SEARCH_CHUNK) {
        chunks += adapt_resilient(pipeline, &mut states, chunk)?;
    }

    // Recall pass: project the query through the freshly-expert state.
    let query_ids = pipeline.encode_text(query);
    let fp = extract_memory_vector_blocking(
        pipeline,
        &mut states,
        &query_ids,
        &format!("search:{}", short_id(query)),
        tokens_ingested,
        started,
        top_k,
    )?;

    Ok(SearchFingerprint {
        query: query.to_string(),
        tokens_ingested,
        chunks,
        recall_norm: fp.recall_norm,
        recall_l1: fp.recall_l1,
        recall_top_k_indices: fp.recall_top_k_indices,
        recall_top_k_decoded: fp.recall_top_k_decoded,
        state_hash: fp.state_hash,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn short_id(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(s.as_bytes());
    format!("{:x}{:x}{:x}{:x}", d[0], d[1], d[2], d[3])
}
