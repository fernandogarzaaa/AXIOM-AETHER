//! Three-stage hybrid recall. Stage 1: a cheap gate decides whether retrieval
//! is worth running at all (zero-cost skip on an empty/irrelevant query).
//! Stage 2: brute-force cosine top-k over the union of the requested scopes,
//! then a recency/supersession rerank. Stage 3 (optional, on by default):
//! bounded spreading activation over the graph working memory
//! (`graph_memory`), seeded by the direct hits, which appends up to
//! `max_extra` live graph-neighbour records after the direct hits (flagged
//! `via_graph`). Graph hits never displace or reorder a direct hit, and edges
//! may cross scopes. Returns real stored records — never neural
//! reconstructions.

use std::collections::HashSet;

use crate::graph_memory::{edges_from_records, spread, Adjacency, EdgeStore, SpreadParams};
use crate::memory_store::{top_k, MemoryRecord, MemoryStore};

/// A recall hit: the matched record plus its score. For direct hits the score
/// is the cosine similarity; for graph hits it is the accumulated spreading
/// activation.
#[derive(Clone, Debug)]
pub struct RecallHit {
    pub score: f32,
    pub record: MemoryRecord,
    /// `true` when the record was surfaced by graph spreading activation
    /// rather than by its own cosine score.
    pub via_graph: bool,
}

/// Tuning for the graph stage of recall.
#[derive(Clone, Debug)]
pub struct GraphRecallParams {
    /// Max graph-surfaced hits appended after the direct hits.
    pub max_extra: usize,
    /// Spreading-activation parameters.
    pub spread: SpreadParams,
}

impl Default for GraphRecallParams {
    fn default() -> Self {
        Self {
            max_extra: 2,
            spread: SpreadParams::default(),
        }
    }
}

/// Tunable recall parameters.
#[derive(Clone, Debug)]
pub struct RecallParams {
    /// Minimum cosine score for a hit to be returned.
    pub min_score: f32,
    /// Max direct (cosine) hits returned.
    pub k: usize,
    /// Graph stage; `None` disables it (pure cosine recall).
    pub graph: Option<GraphRecallParams>,
}

impl Default for RecallParams {
    fn default() -> Self {
        Self {
            min_score: 0.2,
            k: 5,
            graph: Some(GraphRecallParams::default()),
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
/// record is dropped before ranking. Stage 3 then appends graph neighbours of
/// the direct hits (see the module docs); the result has at most
/// `k + graph.max_extra` entries, direct hits first.
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

    // Supersedes edges are derived from the full pool, before the filter below.
    let mut pool_edges = if params.graph.is_some() {
        edges_from_records(&pool)
    } else {
        Vec::new()
    };

    // Drop any record that a present record supersedes.
    let superseded: HashSet<String> = pool.iter().filter_map(|r| r.supersedes.clone()).collect();
    pool.retain(|r| !superseded.contains(&r.id));

    // Rank by cosine, apply the score floor, cap at k.
    let ranked = top_k(query_embedding, &pool, params.k.max(1) * 2);
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

    // Stage 3: graph neighbours of the direct hits.
    if let Some(graph) = &params.graph {
        if !hits.is_empty() {
            if let Ok(edge_store) = EdgeStore::open(store.root()) {
                let mut edges = edge_store.load_all();
                edges.append(&mut pool_edges);
                let adj = Adjacency::build(&edges);
                let seeds: Vec<(String, f32)> = hits
                    .iter()
                    .map(|h| (h.record.id.clone(), h.score))
                    .collect();
                let direct: HashSet<String> = seeds.iter().map(|(id, _)| id.clone()).collect();
                let mut extra: Vec<RecallHit> = Vec::new();
                for (id, activation) in spread(&adj, &seeds, &graph.spread) {
                    if extra.len() >= graph.max_extra {
                        break;
                    }
                    if direct.contains(&id) {
                        continue;
                    }
                    // `load_scope`/`get` already exclude tombstoned records.
                    let record = pool
                        .iter()
                        .find(|r| r.id == id)
                        .cloned()
                        .or_else(|| store.get(&id));
                    let Some(record) = record else {
                        continue;
                    };
                    if record.tombstone || superseded.contains(&record.id) {
                        continue;
                    }
                    extra.push(RecallHit {
                        score: activation,
                        record,
                        via_graph: true,
                    });
                }
                hits.extend(extra);
            }
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_memory::{EdgeKind, MemoryEdge};
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
                graph: None,
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
                graph: None,
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
                graph: None,
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
                graph: None,
            },
        );
        let bodies: Vec<&str> = hits.iter().map(|h| h.record.body.as_str()).collect();
        assert!(bodies.contains(&"NEW-DECISION"));
        assert!(!bodies.contains(&"OLD-DECISION"));
        let _ = std::fs::remove_dir_all(&root);
    }

    fn edge(from: &str, to: &str, kind: EdgeKind, weight: f32) -> MemoryEdge {
        MemoryEdge {
            from: from.to_string(),
            to: to.to_string(),
            kind,
            weight,
            ts: now(),
            tombstone: false,
        }
    }

    fn personal() -> Vec<String> {
        vec!["personal".to_string()]
    }

    fn graph_params(k: usize, max_extra: usize) -> RecallParams {
        RecallParams {
            min_score: 0.5,
            k,
            graph: Some(GraphRecallParams {
                max_extra,
                spread: SpreadParams::default(),
            }),
        }
    }

    #[test]
    fn graph_stage_surfaces_neighbour_of_direct_hit() {
        let (store, root) = temp_store("graph_neighbour");
        store
            .append(&rec("a", "personal", "A", vec![1.0, 0.0], now()))
            .unwrap();
        store
            .append(&rec("b", "personal", "B", vec![0.0, 1.0], now()))
            .unwrap();
        EdgeStore::open(store.root())
            .unwrap()
            .append(&edge("a", "b", EdgeKind::CausedBy, 0.8))
            .unwrap();
        let hits = recall(&store, &personal(), &[1.0, 0.0], &graph_params(5, 2));
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].record.id, "a");
        assert!(!hits[0].via_graph);
        assert_eq!(hits[1].record.id, "b");
        assert!(hits[1].via_graph);
        assert!(hits[1].score > 0.0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn graph_hits_never_displace_direct_hits() {
        let (store, root) = temp_store("graph_no_displace");
        store
            .append(&rec("top", "personal", "TOP", vec![1.0, 0.0], now()))
            .unwrap();
        store
            .append(&rec("second", "personal", "SECOND", vec![0.9, 0.1], now()))
            .unwrap();
        store
            .append(&rec("far", "personal", "FAR", vec![0.0, 1.0], now()))
            .unwrap();
        // Strong edge from the runner-up to `far`; also an edge into `top`.
        let edges = EdgeStore::open(store.root()).unwrap();
        edges
            .append(&edge("second", "far", EdgeKind::CoOccurred, 1.0))
            .unwrap();
        edges
            .append(&edge("top", "second", EdgeKind::CoOccurred, 1.0))
            .unwrap();
        let hits = recall(&store, &personal(), &[1.0, 0.0], &graph_params(1, 2));
        assert_eq!(hits[0].record.id, "top");
        assert!(!hits[0].via_graph);
        assert_eq!(hits.iter().filter(|h| !h.via_graph).count(), 1);
        assert!(hits.len() <= 3);
        let first_graph = hits.iter().position(|h| h.via_graph).unwrap();
        assert!(hits[first_graph..].iter().all(|h| h.via_graph));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn graph_stage_respects_max_extra_and_skips_superseded_and_tombstoned() {
        let (store, root) = temp_store("graph_filters");
        store
            .append(&rec("a", "personal", "A", vec![1.0, 0.0], now()))
            .unwrap();
        store
            .append(&rec("b", "personal", "B", vec![0.0, 1.0], now()))
            .unwrap();
        store
            .append(&rec("c", "personal", "C", vec![0.0, 1.0], 100))
            .unwrap();
        store
            .append(&rec("d", "personal", "D", vec![0.0, 1.0], now()))
            .unwrap();
        let mut c2 = rec("c2", "personal", "C2", vec![0.0, -1.0], now());
        c2.supersedes = Some("c".to_string());
        store.append(&c2).unwrap();
        store.tombstone("personal", "b").unwrap();
        let edges = EdgeStore::open(store.root()).unwrap();
        // B and C carry the strongest edges, so they would win if not filtered.
        edges
            .append(&edge("a", "b", EdgeKind::CausedBy, 1.0))
            .unwrap();
        edges
            .append(&edge("a", "c", EdgeKind::CausedBy, 0.9))
            .unwrap();
        edges
            .append(&edge("a", "d", EdgeKind::CausedBy, 0.5))
            .unwrap();
        let hits = recall(&store, &personal(), &[1.0, 0.0], &graph_params(5, 1));
        let graph_ids: Vec<&str> = hits
            .iter()
            .filter(|h| h.via_graph)
            .map(|h| h.record.id.as_str())
            .collect();
        assert_eq!(graph_ids, vec!["d"]);
        let all_ids: Vec<&str> = hits.iter().map(|h| h.record.id.as_str()).collect();
        assert!(!all_ids.contains(&"b"));
        assert!(!all_ids.contains(&"c"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn graph_stage_disabled_when_params_graph_is_none() {
        let (store, root) = temp_store("graph_off");
        store
            .append(&rec("a", "personal", "A", vec![1.0, 0.0], now()))
            .unwrap();
        store
            .append(&rec("b", "personal", "B", vec![0.0, 1.0], now()))
            .unwrap();
        EdgeStore::open(store.root())
            .unwrap()
            .append(&edge("a", "b", EdgeKind::CausedBy, 0.8))
            .unwrap();
        let params = RecallParams {
            min_score: 0.5,
            k: 5,
            graph: None,
        };
        let hits = recall(&store, &personal(), &[1.0, 0.0], &params);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, "a");
        assert!(!hits[0].via_graph);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn graph_stage_follows_cross_scope_edges() {
        let (store, root) = temp_store("graph_cross_scope");
        store
            .append(&rec("a", "personal", "A", vec![1.0, 0.0], now()))
            .unwrap();
        store
            .append(&rec("b", "project:zzz", "B", vec![0.0, 1.0], now()))
            .unwrap();
        EdgeStore::open(store.root())
            .unwrap()
            .append(&edge("a", "b", EdgeKind::DerivedFrom, 0.8))
            .unwrap();
        let hits = recall(&store, &personal(), &[1.0, 0.0], &graph_params(5, 2));
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[1].record.id, "b");
        assert_eq!(hits[1].record.scope, "project:zzz");
        assert!(hits[1].via_graph);
        let _ = std::fs::remove_dir_all(&root);
    }
}
