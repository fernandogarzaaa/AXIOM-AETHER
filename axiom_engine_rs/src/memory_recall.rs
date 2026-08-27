//! Two-stage hybrid recall. Stage 1: a cheap gate decides whether retrieval is
//! worth running at all (zero-cost skip on an empty/irrelevant query). Stage 2:
//! brute-force cosine top-k over the union of the requested scopes, then a
//! recency/supersession rerank. Returns real stored records — never neural
//! reconstructions.
//!
//! An optional third stage ([`recall_with_graph`]) widens the cosine hits with
//! bounded spreading activation over [`crate::graph_memory`]'s edge graph —
//! e.g. a record `CausedBy` a direct hit, or one that has co-occurred with it
//! in past recalls, even if its own embedding scores below `min_score` on this
//! query. Graph-expanded hits are never a substitute for a direct match: they
//! are returned separately, capped, and marked `via_graph = true` on
//! [`RecallHit`] so a caller (or an evaluation harness measuring wrong-memory
//! rate) can always tell a cosine match from an associative one.

use std::collections::{HashMap, HashSet};

use crate::graph_memory::{Adjacency, EdgeKind, EdgeStore, MemoryEdge, SpreadParams};
use crate::memory_store::{top_k, MemoryRecord, MemoryStore};

/// A recall hit: the matched record plus its cosine score.
#[derive(Clone, Debug)]
pub struct RecallHit {
    pub score: f32,
    pub record: MemoryRecord,
    /// `true` when this hit was not itself a direct cosine match on the
    /// query but was surfaced by spreading activation from one — see the
    /// module doc. Always `false` for [`recall`]'s plain output.
    pub via_graph: bool,
}

/// Tunable recall parameters.
#[derive(Clone, Debug)]
pub struct RecallParams {
    /// Minimum cosine score for a hit to be returned.
    pub min_score: f32,
    /// Max hits returned.
    pub k: usize,
    /// Max additional graph-expanded hits [`recall_with_graph`] may add on
    /// top of the direct cosine hits. `0` disables graph expansion even when
    /// an [`EdgeStore`] is supplied. Irrelevant to plain [`recall`].
    pub graph_k: usize,
}

impl Default for RecallParams {
    fn default() -> Self {
        Self {
            min_score: 0.2,
            k: 5,
            graph_k: 3,
        }
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
    let pool = load_scope_pool(store, scopes);
    direct_hits(&pool, query_embedding, params)
}

/// Load the deduplicated, supersession-filtered pool for `scopes` — the same
/// candidate set [`recall`] ranks. Exposed so [`recall_with_graph`] can reuse
/// it as the id→record lookup for resolving graph-expanded hits without a
/// second scan of the store.
fn load_scope_pool(store: &MemoryStore, scopes: &[String]) -> Vec<MemoryRecord> {
    let mut pool: Vec<MemoryRecord> = Vec::new();
    for scope in scopes {
        pool.extend(store.load_scope(scope));
    }
    // Drop any record that a present record supersedes.
    let superseded: HashSet<String> = pool.iter().filter_map(|r| r.supersedes.clone()).collect();
    pool.retain(|r| !superseded.contains(&r.id));
    pool
}

/// Cosine top-k over an already-loaded pool, score-floored and recency-tied.
/// The ranking half of [`recall`], factored out so [`recall_with_graph`] can
/// run it once against a pool it also needs for graph resolution.
fn direct_hits(pool: &[MemoryRecord], query_embedding: &[f32], params: &RecallParams) -> Vec<RecallHit> {
    let ranked = top_k(query_embedding, pool, params.k.max(1) * 2);
    let mut hits: Vec<RecallHit> = ranked
        .into_iter()
        .filter(|(score, _)| *score >= params.min_score)
        .map(|(score, r)| RecallHit {
            score,
            record: r.clone(),
            via_graph: false,
        })
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

/// [`recall`], widened with bounded spreading activation over `edges` (see the
/// module doc). Graph-expanded hits are drawn **only** from records already in
/// the requested `scopes`' pool — traversal can surface a record that scored
/// below `min_score` on this query, but never one outside the scopes the
/// caller was actually authorized to see, regardless of what a stored edge
/// points to. Direct hits are always returned first, in the same order
/// plain [`recall`] would; up to `params.graph_k` graph-expanded hits follow,
/// ranked by spread activation, each marked `via_graph = true`.
pub fn recall_with_graph(
    store: &MemoryStore,
    edges: &EdgeStore,
    scopes: &[String],
    query_embedding: &[f32],
    params: &RecallParams,
) -> Vec<RecallHit> {
    if !should_recall(query_embedding) {
        return Vec::new();
    }
    let pool = load_scope_pool(store, scopes);
    let mut hits = direct_hits(&pool, query_embedding, params);
    if params.graph_k == 0 || hits.is_empty() {
        return hits;
    }

    let by_id: HashMap<&str, &MemoryRecord> = pool.iter().map(|r| (r.id.as_str(), r)).collect();
    let already_hit: HashSet<String> = hits.iter().map(|h| h.record.id.clone()).collect();

    let all_edges = edges.load_all();
    let adjacency = Adjacency::build(&all_edges);
    let seeds: Vec<(String, f32)> = hits.iter().map(|h| (h.record.id.clone(), h.score)).collect();
    let spread_params = SpreadParams::default();
    let activated = crate::graph_memory::spread(&adjacency, &seeds, &spread_params);

    for (id, activation) in activated {
        let added_so_far = hits.iter().filter(|h| h.via_graph).count();
        if added_so_far >= params.graph_k {
            break;
        }
        if already_hit.contains(id.as_str()) {
            continue;
        }
        let Some(record) = by_id.get(id.as_str()) else {
            // Resolves to a record outside the requested scopes (or already
            // pruned as superseded) -- never surfaced, per the scope
            // confinement documented above.
            continue;
        };
        hits.push(RecallHit {
            score: activation,
            record: (*record).clone(),
            via_graph: true,
        });
    }
    hits
}

/// Record that `hit_ids` were retrieved together for the same query --
/// evidence they are associatively related even when their embeddings alone
/// wouldn't say so. Called from the recall path (see `mcp_stdio.rs`), not
/// from [`recall`]/[`recall_with_graph`] themselves, so a pure read never has
/// a side effect a caller didn't ask for. Bounded to avoid an O(k^2) edge
/// explosion: only the strongest `max_pairs` hits are connected, pairwise.
pub fn record_co_occurrence(edges: &EdgeStore, hit_ids: &[String], max_pairs: usize) -> std::io::Result<()> {
    let ts = crate::memory_store::now_secs();
    let bounded = &hit_ids[..hit_ids.len().min(max_pairs)];
    for i in 0..bounded.len() {
        for j in (i + 1)..bounded.len() {
            edges.append(&MemoryEdge {
                from: bounded[i].clone(),
                to: bounded[j].clone(),
                kind: EdgeKind::CoOccurred,
                weight: 0.5,
                ts,
                tombstone: false,
            })?;
        }
    }
    Ok(())
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
        p.push(format!(
            "axiom_recall_test_{tag}_{}_{}",
            now(),
            std::process::id()
        ));
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
        store
            .append(&rec("near", "personal", "BODY-NEAR", vec![1.0, 0.0], now()))
            .unwrap();
        store
            .append(&rec("far", "personal", "BODY-FAR", vec![0.0, 1.0], now()))
            .unwrap();
        let hits = recall(
            &store,
            &["personal".to_string()],
            &[1.0, 0.0],
            &RecallParams {
                min_score: 0.1,
                k: 5,
                ..Default::default()
            },
        );
        assert_eq!(hits[0].record.body, "BODY-NEAR");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn recall_searches_union_of_scopes() {
        let (store, root) = temp_store("union");
        store
            .append(&rec("a", "personal", "P", vec![1.0, 0.0], now()))
            .unwrap();
        store
            .append(&rec("b", "project:zzz", "Q", vec![0.9, 0.1], now()))
            .unwrap();
        let hits = recall(
            &store,
            &["personal".to_string(), "project:zzz".to_string()],
            &[1.0, 0.0],
            &RecallParams {
                min_score: 0.1,
                k: 5,
                ..Default::default()
            },
        );
        assert_eq!(hits.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn recall_respects_min_score() {
        let (store, root) = temp_store("minscore");
        store
            .append(&rec("orth", "personal", "ORTH", vec![0.0, 1.0], now()))
            .unwrap();
        let hits = recall(
            &store,
            &["personal".to_string()],
            &[1.0, 0.0],
            &RecallParams {
                min_score: 0.5,
                k: 5,
                ..Default::default()
            },
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
            &RecallParams {
                min_score: 0.1,
                k: 5,
                ..Default::default()
            },
        );
        let bodies: Vec<&str> = hits.iter().map(|h| h.record.body.as_str()).collect();
        assert!(bodies.contains(&"NEW-DECISION"));
        assert!(!bodies.contains(&"OLD-DECISION"));
        let _ = std::fs::remove_dir_all(&root);
    }

    fn temp_edges(tag: &str) -> (EdgeStore, PathBuf) {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "axiom_recall_edges_test_{tag}_{}_{}",
            now(),
            std::process::id()
        ));
        (EdgeStore::open(&p).unwrap(), p)
    }

    #[test]
    fn recall_with_graph_matches_plain_recall_when_no_edges_exist() {
        let (store, root) = temp_store("graph_no_edges");
        let (edges, edges_root) = temp_edges("graph_no_edges");
        store
            .append(&rec("near", "personal", "BODY-NEAR", vec![1.0, 0.0], now()))
            .unwrap();
        let params = RecallParams {
            min_score: 0.1,
            k: 5,
            ..Default::default()
        };
        let plain = recall(&store, &["personal".to_string()], &[1.0, 0.0], &params);
        let graph = recall_with_graph(&store, &edges, &["personal".to_string()], &[1.0, 0.0], &params);
        assert_eq!(plain.len(), graph.len());
        assert!(graph.iter().all(|h| !h.via_graph));
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&edges_root);
    }

    #[test]
    fn recall_with_graph_surfaces_a_linked_record_below_the_score_floor() {
        let (store, root) = temp_store("graph_expand");
        let (edges, edges_root) = temp_edges("graph_expand");
        // "hub" is the only direct cosine hit; "linked" is orthogonal to the
        // query (score 0.0, well under any floor) but CausedBy hub.
        store
            .append(&rec("hub", "personal", "HUB", vec![1.0, 0.0], now()))
            .unwrap();
        store
            .append(&rec("linked", "personal", "LINKED", vec![0.0, 1.0], now()))
            .unwrap();
        edges
            .append(&MemoryEdge {
                from: "linked".to_string(),
                to: "hub".to_string(),
                kind: EdgeKind::CausedBy,
                weight: 1.0,
                ts: now(),
                tombstone: false,
            })
            .unwrap();

        let params = RecallParams {
            min_score: 0.5,
            k: 5,
            graph_k: 3,
        };
        let direct = recall(&store, &["personal".to_string()], &[1.0, 0.0], &params);
        assert_eq!(direct.len(), 1, "only hub should pass the score floor directly");

        let widened = recall_with_graph(&store, &edges, &["personal".to_string()], &[1.0, 0.0], &params);
        assert_eq!(widened.len(), 2, "graph expansion should add the linked record");
        let hub_hit = widened.iter().find(|h| h.record.id == "hub").unwrap();
        assert!(!hub_hit.via_graph, "the direct hit must not be relabeled as graph-expanded");
        let linked_hit = widened.iter().find(|h| h.record.id == "linked").unwrap();
        assert!(linked_hit.via_graph, "the expanded hit must be marked via_graph");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&edges_root);
    }

    #[test]
    fn recall_with_graph_never_surfaces_a_record_outside_the_requested_scopes() {
        let (store, root) = temp_store("graph_scope_confine");
        let (edges, edges_root) = temp_edges("graph_scope_confine");
        store
            .append(&rec("hub", "personal", "HUB", vec![1.0, 0.0], now()))
            .unwrap();
        // "other" lives in a scope the caller does NOT request.
        store
            .append(&rec("other", "project:zzz", "OTHER", vec![0.0, 1.0], now()))
            .unwrap();
        edges
            .append(&MemoryEdge {
                from: "other".to_string(),
                to: "hub".to_string(),
                kind: EdgeKind::CausedBy,
                weight: 1.0,
                ts: now(),
                tombstone: false,
            })
            .unwrap();

        let params = RecallParams {
            min_score: 0.5,
            k: 5,
            graph_k: 3,
        };
        let widened = recall_with_graph(&store, &edges, &["personal".to_string()], &[1.0, 0.0], &params);
        assert!(
            widened.iter().all(|h| h.record.id != "other"),
            "a record outside the requested scopes must never be surfaced via graph expansion: {widened:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&edges_root);
    }

    #[test]
    fn recall_with_graph_respects_graph_k_cap() {
        let (store, root) = temp_store("graph_cap");
        let (edges, edges_root) = temp_edges("graph_cap");
        store
            .append(&rec("hub", "personal", "HUB", vec![1.0, 0.0], now()))
            .unwrap();
        for i in 0..5 {
            let id = format!("leaf{i}");
            store
                .append(&rec(&id, "personal", "LEAF", vec![0.0, 1.0], now()))
                .unwrap();
            edges
                .append(&MemoryEdge {
                    from: id,
                    to: "hub".to_string(),
                    kind: EdgeKind::CoOccurred,
                    weight: 1.0,
                    ts: now(),
                    tombstone: false,
                })
                .unwrap();
        }
        let params = RecallParams {
            min_score: 0.5,
            k: 5,
            graph_k: 2,
        };
        let widened = recall_with_graph(&store, &edges, &["personal".to_string()], &[1.0, 0.0], &params);
        let graph_hits = widened.iter().filter(|h| h.via_graph).count();
        assert_eq!(graph_hits, 2, "must not exceed graph_k regardless of how many leaves are linked");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&edges_root);
    }

    #[test]
    fn record_co_occurrence_links_every_pair_up_to_max_pairs() {
        let (edges, root) = temp_edges("co_occurrence");
        let ids = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        record_co_occurrence(&edges, &ids, 3).unwrap();
        let all = edges.load_all();
        assert_eq!(all.len(), 3, "3 ids -> 3 pairwise edges");
        assert!(all.iter().all(|e| e.kind == EdgeKind::CoOccurred));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn record_co_occurrence_respects_max_pairs_bound() {
        let (edges, root) = temp_edges("co_occurrence_bound");
        let ids: Vec<String> = (0..10).map(|i| format!("id{i}")).collect();
        record_co_occurrence(&edges, &ids, 3).unwrap();
        let all = edges.load_all();
        // Only the first 3 ids are connected pairwise: C(3,2) = 3 edges,
        // regardless of how many ids were passed in.
        assert_eq!(all.len(), 3);
        let _ = std::fs::remove_dir_all(&root);
    }
}
