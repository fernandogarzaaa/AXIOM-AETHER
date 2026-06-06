//! Two-stage hybrid recall. Stage 1: a cheap gate decides whether retrieval is
//! worth running at all (zero-cost skip on an empty/irrelevant query). Stage 2:
//! brute-force cosine top-k over the union of the requested scopes, then a
//! recency/supersession rerank. Returns real stored records — never neural
//! reconstructions.

use crate::memory_store::{top_k, MemoryRecord, MemoryStore};

/// A recall hit: the matched record plus its cosine score.
#[derive(Clone, Debug)]
pub struct RecallHit {
    pub score: f32,
    pub record: MemoryRecord,
}

/// Tunable recall parameters.
#[derive(Clone, Debug)]
pub struct RecallParams {
    /// Minimum cosine score for a hit to be returned.
    pub min_score: f32,
    /// Max hits returned.
    pub k: usize,
}

impl Default for RecallParams {
    fn default() -> Self {
        Self { min_score: 0.2, k: 5 }
    }
}

/// Stage-1 gate: should we run retrieval? Skips empty queries (zero cost).
pub fn should_recall(query_embedding: &[f32]) -> bool {
    // A non-degenerate (non-zero) embedding means there's something to match.
    query_embedding.iter().any(|x| *x != 0.0)
}

/// Stage-2 retrieval over the union of `scopes`. Higher cosine wins; ties break
/// toward newer records; a record that has been superseded by any present
/// record is dropped before ranking.
pub fn recall(
    store: &MemoryStore,
    scopes: &[String],
    query_embedding: &[f32],
    params: &RecallParams,
) -> Vec<RecallHit> {
    if !should_recall(query_embedding) {
        return Vec::new();
    }

    // Gather the union of all requested scopes.
    let mut pool: Vec<MemoryRecord> = Vec::new();
    for scope in scopes {
        pool.extend(store.load_scope(scope));
    }

    // Drop any record that a present record supersedes.
    let superseded: std::collections::HashSet<String> =
        pool.iter().filter_map(|r| r.supersedes.clone()).collect();
    pool.retain(|r| !superseded.contains(&r.id));

    // Rank by cosine, apply the score floor, cap at k.
    let ranked = top_k(query_embedding, &pool, params.k.max(1) * 2);
    let mut hits: Vec<RecallHit> = ranked
        .into_iter()
        .filter(|(score, _)| *score >= params.min_score)
        .map(|(score, r)| RecallHit { score, record: r.clone() })
        .collect();

    // Score order, recency tie-break.
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.record.ts.cmp(&a.record.ts))
    });
    hits.truncate(params.k);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_store::{MemoryKind, MemoryRecord};
    use std::path::PathBuf;

    fn now() -> u64 {
        crate::memory_store::now_secs()
    }

    fn rec(id: &str, scope: &str, body: &str, emb: Vec<f32>, ts: u64) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            scope: scope.to_string(),
            ts,
            kind: MemoryKind::Decision,
            body: body.to_string(),
            embedding: emb,
            drift_at_ingest: 5.0,
            supersedes: None,
            tombstone: false,
        }
    }

    fn temp_store(tag: &str) -> (MemoryStore, PathBuf) {
        let mut p = std::env::temp_dir();
        p.push(format!("axiom_recall_test_{tag}_{}_{}", now(), std::process::id()));
        (MemoryStore::open(&p).unwrap(), p)
    }

    #[test]
    fn gate_skips_zero_embedding() {
        assert!(!should_recall(&[0.0, 0.0, 0.0]));
        assert!(should_recall(&[0.0, 1.0]));
    }

    #[test]
    fn recall_returns_real_bodies_ranked() {
        let (store, root) = temp_store("ranked");
        store.append(&rec("near", "personal", "BODY-NEAR", vec![1.0, 0.0], now())).unwrap();
        store.append(&rec("far", "personal", "BODY-FAR", vec![0.0, 1.0], now())).unwrap();
        let hits = recall(
            &store,
            &["personal".to_string()],
            &[1.0, 0.0],
            &RecallParams { min_score: 0.1, k: 5 },
        );
        assert_eq!(hits[0].record.body, "BODY-NEAR");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn recall_searches_union_of_scopes() {
        let (store, root) = temp_store("union");
        store.append(&rec("a", "personal", "P", vec![1.0, 0.0], now())).unwrap();
        store.append(&rec("b", "project:zzz", "Q", vec![0.9, 0.1], now())).unwrap();
        let hits = recall(
            &store,
            &["personal".to_string(), "project:zzz".to_string()],
            &[1.0, 0.0],
            &RecallParams { min_score: 0.1, k: 5 },
        );
        assert_eq!(hits.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn recall_respects_min_score() {
        let (store, root) = temp_store("minscore");
        store.append(&rec("orth", "personal", "ORTH", vec![0.0, 1.0], now())).unwrap();
        let hits = recall(
            &store,
            &["personal".to_string()],
            &[1.0, 0.0],
            &RecallParams { min_score: 0.5, k: 5 },
        );
        assert!(hits.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn superseded_record_is_dropped_when_newer_returned() {
        let (store, root) = temp_store("supersede");
        let old = rec("old", "personal", "OLD-DECISION", vec![1.0, 0.0], 100);
        let mut new = rec("new", "personal", "NEW-DECISION", vec![1.0, 0.0], 200);
        new.supersedes = Some("old".to_string());
        store.append(&old).unwrap();
        store.append(&new).unwrap();
        let hits = recall(
            &store,
            &["personal".to_string()],
            &[1.0, 0.0],
            &RecallParams { min_score: 0.1, k: 5 },
        );
        let bodies: Vec<&str> = hits.iter().map(|h| h.record.body.as_str()).collect();
        assert!(bodies.contains(&"NEW-DECISION"));
        assert!(!bodies.contains(&"OLD-DECISION"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
