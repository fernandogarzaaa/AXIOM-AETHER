//! Graph working memory: directed, weighted edges between `MemoryRecord`s.
//! Storage mirrors `MemoryStore`'s append-only JSONL durability model, but
//! edges live in a single `edges.jsonl` at the memory root rather than one
//! file per scope, since edges legitimately cross scopes.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::memory_store::{now_secs, MemoryRecord};

/// The relationship a `MemoryEdge` records between two `MemoryRecord`s.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// Already implicit in `MemoryRecord.supersedes`.
    Supersedes,
    /// Fault trace -> heal.
    CausedBy,
    /// Specific heal -> heal class.
    GeneralizesTo,
    /// Synthesized artifact -> source fragment (Phase C).
    DerivedFrom,
    /// Tool -> tool composition (Phase C).
    DependsOn,
    /// Falls out of hallucination.rs verification.
    Contradicts,
    /// Same-session coactivation.
    CoOccurred,
}

/// One directed, weighted edge between two `MemoryRecord` ids.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemoryEdge {
    pub from: String,
    pub to: String,
    pub kind: EdgeKind,
    pub weight: f32,
    pub ts: u64,
    #[serde(default)]
    pub tombstone: bool,
}

/// Append-only JSONL store of `MemoryEdge`s, rooted at a directory. A single
/// `edges.jsonl` file, not one per scope: edges legitimately cross scopes and
/// a per-scope split would make traversal require opening every file.
pub struct EdgeStore {
    root: PathBuf,
}

impl EdgeStore {
    /// Open (creating the root dir if needed).
    pub fn open(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Path to the single edges file. Public for tests that append raw lines.
    pub fn edges_path(&self) -> PathBuf {
        self.root.join("edges.jsonl")
    }

    /// Append one edge, clamping `weight` to `0.0..=1.0`.
    pub fn append(&self, edge: &MemoryEdge) -> std::io::Result<()> {
        let mut clamped = edge.clone();
        clamped.weight = clamped.weight.clamp(0.0, 1.0);
        let path = self.edges_path();
        let mut f = OpenOptions::new().create(true).append(true).open(path)?;
        let line = serde_json::to_string(&clamped)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }

    /// Load every live (non-tombstoned) edge. A corrupt line is skipped
    /// (fail-open), never fatal, matching `MemoryStore::load_scope`.
    pub fn load_all(&self) -> Vec<MemoryEdge> {
        let path = self.edges_path();
        let Ok(file) = fs::File::open(&path) else {
            return Vec::new();
        };
        let mut live: Vec<MemoryEdge> = Vec::new();
        let mut tombstoned: std::collections::HashSet<(String, String, EdgeKind)> =
            std::collections::HashSet::new();
        for line in BufReader::new(file).lines().map_while(std::io::Result::ok) {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(edge) = serde_json::from_str::<MemoryEdge>(line) {
                if edge.tombstone {
                    tombstoned.insert((edge.from.clone(), edge.to.clone(), edge.kind));
                }
                live.push(edge);
            }
        }
        live.into_iter()
            .filter(|e| !e.tombstone && !tombstoned.contains(&(e.from.clone(), e.to.clone(), e.kind)))
            .collect()
    }

    /// Tombstone an edge by appending a tombstone record for the same
    /// `(from, to, kind)` triple.
    pub fn tombstone(&self, from: &str, to: &str, kind: EdgeKind) -> std::io::Result<()> {
        let edge = MemoryEdge {
            from: from.to_string(),
            to: to.to_string(),
            kind,
            weight: 0.0,
            ts: now_secs(),
            tombstone: true,
        };
        self.append(&edge)
    }
}

/// Derive `Supersedes` edges from the `supersedes` field already present on
/// existing records, so the graph is populated on day one with no migration.
pub fn edges_from_records(records: &[MemoryRecord]) -> Vec<MemoryEdge> {
    records
        .iter()
        .filter_map(|r| {
            r.supersedes.as_ref().map(|prev| MemoryEdge {
                from: r.id.clone(),
                to: prev.clone(),
                kind: EdgeKind::Supersedes,
                weight: 1.0,
                ts: r.ts,
                tombstone: false,
            })
        })
        .collect()
}

/// Bidirectional adjacency index built from a set of edges. Traversal follows
/// edges in both directions (a `CausedBy` edge is informative read from
/// either end), but out-edges and in-edges may carry different weights, so
/// the two directions are indexed separately.
pub struct Adjacency {
    out_edges: HashMap<String, Vec<(EdgeKind, String, f32)>>,
    in_edges: HashMap<String, Vec<(EdgeKind, String, f32)>>,
}

impl Adjacency {
    /// Build from a set of edges, skipping tombstoned ones.
    pub fn build(edges: &[MemoryEdge]) -> Self {
        let mut out_edges: HashMap<String, Vec<(EdgeKind, String, f32)>> = HashMap::new();
        let mut in_edges: HashMap<String, Vec<(EdgeKind, String, f32)>> = HashMap::new();
        for e in edges {
            if e.tombstone {
                continue;
            }
            out_edges
                .entry(e.from.clone())
                .or_default()
                .push((e.kind, e.to.clone(), e.weight));
            in_edges
                .entry(e.to.clone())
                .or_default()
                .push((e.kind, e.from.clone(), e.weight));
        }
        Self { out_edges, in_edges }
    }

    /// Every neighbor reachable from `id` in either direction, as
    /// `(edge kind, neighbor id, edge weight)`.
    fn neighbors(&self, id: &str) -> Vec<(EdgeKind, String, f32)> {
        let mut out = Vec::new();
        if let Some(v) = self.out_edges.get(id) {
            out.extend(v.iter().cloned());
        }
        if let Some(v) = self.in_edges.get(id) {
            out.extend(v.iter().cloned());
        }
        out
    }
}

/// Tuning parameters for `spread`.
pub struct SpreadParams {
    pub max_hops: usize,
    pub max_visited: usize,
    pub decay: f32,
    pub min_activation: f32,
    pub kind_weights: BTreeMap<EdgeKind, f32>,
}

impl Default for SpreadParams {
    fn default() -> Self {
        Self {
            max_hops: 2,
            max_visited: 256,
            decay: 0.5,
            min_activation: 1e-3,
            kind_weights: BTreeMap::new(),
        }
    }
}

/// Bounded spreading activation from `seeds` (id, initial activation) over
/// `adj`. Returns every reached node with accumulated activation, seeds
/// included, sorted by `(activation descending, id ascending)` for
/// deterministic output.
///
/// `max_visited` is a hard bound on *distinct nodes visited*, not on hop
/// depth: one hop into a dense region can visit thousands of nodes, so the
/// bound is enforced while expanding a level, not only between levels.
/// Already-visited nodes are never re-added to the expansion frontier, so
/// cycles (including self-loops) cannot cause the traversal to loop forever
/// -- termination is additionally guaranteed structurally, since the outer
/// loop runs at most `max_hops` times.
pub fn spread(adj: &Adjacency, seeds: &[(String, f32)], params: &SpreadParams) -> Vec<(String, f32)> {
    let mut activation: HashMap<String, f32> = HashMap::new();
    let mut visited: HashSet<String> = HashSet::new();

    for (id, act) in seeds {
        *activation.entry(id.clone()).or_insert(0.0) += act;
        visited.insert(id.clone());
    }

    let mut frontier: Vec<(String, f32)> = seeds.to_vec();

    for _ in 0..params.max_hops {
        if frontier.is_empty() || visited.len() >= params.max_visited {
            break;
        }
        frontier.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut next_frontier: Vec<(String, f32)> = Vec::new();
        for (node, act) in &frontier {
            if visited.len() >= params.max_visited {
                break;
            }
            for (kind, neighbor, weight) in adj.neighbors(node) {
                let kind_weight = params.kind_weights.get(&kind).copied().unwrap_or(1.0);
                let contribution = act * params.decay * weight * kind_weight;
                if contribution < params.min_activation {
                    continue;
                }
                if visited.contains(&neighbor) {
                    if let Some(a) = activation.get_mut(&neighbor) {
                        *a += contribution;
                    }
                    continue;
                }
                if visited.len() >= params.max_visited {
                    continue;
                }
                visited.insert(neighbor.clone());
                activation.insert(neighbor.clone(), contribution);
                next_frontier.push((neighbor, contribution));
            }
        }
        frontier = next_frontier;
    }

    let mut result: Vec<(String, f32)> = activation.into_iter().collect();
    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_store::MemoryKind;

    fn edge(from: &str, to: &str, kind: EdgeKind, weight: f32) -> MemoryEdge {
        MemoryEdge {
            from: from.to_string(),
            to: to.to_string(),
            kind,
            weight,
            ts: now_secs(),
            tombstone: false,
        }
    }

    fn rec_with_supersedes(id: &str, supersedes: Option<&str>) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            scope: "personal".to_string(),
            ts: now_secs(),
            kind: MemoryKind::Decision,
            body: format!("body of {id}"),
            embedding: vec![1.0, 0.0],
            drift_at_ingest: 5.0,
            supersedes: supersedes.map(|s| s.to_string()),
            tombstone: false,
        }
    }

    fn temp_root(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "axiom_graph_test_{tag}_{}_{}",
            now_secs(),
            std::process::id()
        ));
        p
    }

    #[test]
    fn edge_roundtrip_jsonl() {
        let root = temp_root("roundtrip");
        let store = EdgeStore::open(&root).unwrap();
        store
            .append(&edge("a", "b", EdgeKind::CausedBy, 0.7))
            .unwrap();
        store
            .append(&edge("b", "c", EdgeKind::GeneralizesTo, 0.9))
            .unwrap();
        let loaded = store.load_all();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].from, "a");
        assert_eq!(loaded[0].to, "b");
        assert_eq!(loaded[0].kind, EdgeKind::CausedBy);
        assert_eq!(loaded[1].from, "b");
        assert_eq!(loaded[1].to, "c");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn append_clamps_weight() {
        let root = temp_root("clamp");
        let store = EdgeStore::open(&root).unwrap();
        store
            .append(&edge("a", "b", EdgeKind::CoOccurred, 2.5))
            .unwrap();
        store
            .append(&edge("c", "d", EdgeKind::CoOccurred, -1.0))
            .unwrap();
        let loaded = store.load_all();
        assert_eq!(loaded[0].weight, 1.0);
        assert_eq!(loaded[1].weight, 0.0);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn malformed_line_is_skipped_not_fatal() {
        let root = temp_root("malformed");
        let store = EdgeStore::open(&root).unwrap();
        store
            .append(&edge("a", "b", EdgeKind::CausedBy, 0.5))
            .unwrap();
        let path = store.edges_path();
        let mut f = OpenOptions::new().append(true).open(path).unwrap();
        f.write_all(b"{ this is not json\n").unwrap();
        store
            .append(&edge("c", "d", EdgeKind::CausedBy, 0.6))
            .unwrap();
        let loaded = store.load_all();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].from, "a");
        assert_eq!(loaded[1].from, "c");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn supersedes_backfill_produces_edges() {
        let records = vec![
            rec_with_supersedes("a", None),
            rec_with_supersedes("b", Some("a")),
        ];
        let edges = edges_from_records(&records);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from, "b");
        assert_eq!(edges[0].to, "a");
        assert_eq!(edges[0].kind, EdgeKind::Supersedes);
    }

    #[test]
    fn tombstoned_edge_is_excluded() {
        let root = temp_root("tombstone");
        let store = EdgeStore::open(&root).unwrap();
        store
            .append(&edge("a", "b", EdgeKind::CausedBy, 0.5))
            .unwrap();
        store.tombstone("a", "b", EdgeKind::CausedBy).unwrap();
        let loaded = store.load_all();
        assert!(loaded.is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    fn seed(id: &str, act: f32) -> (String, f32) {
        (id.to_string(), act)
    }

    fn find<'a>(result: &'a [(String, f32)], id: &str) -> Option<&'a (String, f32)> {
        result.iter().find(|(i, _)| i == id)
    }

    #[test]
    fn spread_reaches_two_hops() {
        let edges = vec![
            edge("a", "b", EdgeKind::CausedBy, 1.0),
            edge("b", "c", EdgeKind::CausedBy, 1.0),
        ];
        let adj = Adjacency::build(&edges);
        let params = SpreadParams::default();
        let result = spread(&adj, &[seed("a", 1.0)], &params);
        assert!(find(&result, "a").is_some());
        assert!(find(&result, "b").is_some());
        assert!(find(&result, "c").is_some());
    }

    #[test]
    fn spread_decays_with_distance() {
        let edges = vec![
            edge("a", "b", EdgeKind::CausedBy, 1.0),
            edge("b", "c", EdgeKind::CausedBy, 1.0),
        ];
        let adj = Adjacency::build(&edges);
        let params = SpreadParams::default();
        let result = spread(&adj, &[seed("a", 1.0)], &params);
        let a = find(&result, "a").unwrap().1;
        let b = find(&result, "b").unwrap().1;
        let c = find(&result, "c").unwrap().1;
        assert!(c < b, "expected activation(c) < activation(b): {c} vs {b}");
        assert!(b < a, "expected activation(b) < activation(a): {b} vs {a}");
    }

    #[test]
    fn spread_respects_max_hops() {
        let edges = vec![
            edge("a", "b", EdgeKind::CausedBy, 1.0),
            edge("b", "c", EdgeKind::CausedBy, 1.0),
        ];
        let adj = Adjacency::build(&edges);
        let params = SpreadParams {
            max_hops: 1,
            ..SpreadParams::default()
        };
        let result = spread(&adj, &[seed("a", 1.0)], &params);
        assert!(find(&result, "c").is_none());
    }

    #[test]
    fn spread_respects_max_visited() {
        let mut edges = Vec::new();
        for i in 0..1000 {
            edges.push(edge("center", &format!("leaf{i}"), EdgeKind::CoOccurred, 1.0));
        }
        let adj = Adjacency::build(&edges);
        let params = SpreadParams {
            max_visited: 10,
            ..SpreadParams::default()
        };
        let result = spread(&adj, &[seed("center", 1.0)], &params);
        assert!(result.len() <= 10);
    }

    #[test]
    fn spread_is_deterministic() {
        let mut edges = Vec::new();
        for i in 0..50 {
            edges.push(edge("center", &format!("leaf{i}"), EdgeKind::CoOccurred, 1.0));
        }
        let adj = Adjacency::build(&edges);
        let params = SpreadParams {
            max_visited: 20,
            ..SpreadParams::default()
        };
        let r1 = spread(&adj, &[seed("center", 1.0)], &params);
        let r2 = spread(&adj, &[seed("center", 1.0)], &params);
        assert_eq!(r1, r2);
    }

    #[test]
    fn cycle_does_not_hang() {
        let edges = vec![
            edge("a", "b", EdgeKind::CausedBy, 1.0),
            edge("b", "a", EdgeKind::CausedBy, 1.0),
            edge("a", "a", EdgeKind::CausedBy, 1.0),
        ];
        let adj = Adjacency::build(&edges);
        let params = SpreadParams::default();
        // Bounded structurally by max_hops; if this returns at all, it has
        // not spun on the cycle.
        let result = spread(&adj, &[seed("a", 1.0)], &params);
        assert!(find(&result, "a").is_some());
        assert!(find(&result, "b").is_some());
    }

    #[test]
    fn tombstoned_edge_is_not_traversed() {
        let edges = vec![{
            let mut e = edge("a", "b", EdgeKind::CausedBy, 1.0);
            e.tombstone = true;
            e
        }];
        let adj = Adjacency::build(&edges);
        let params = SpreadParams::default();
        let result = spread(&adj, &[seed("a", 1.0)], &params);
        assert!(find(&result, "b").is_none());
    }

    #[test]
    fn kind_weights_change_ranking() {
        let edges = vec![edge("a", "b", EdgeKind::Contradicts, 1.0)];
        let adj = Adjacency::build(&edges);
        let mut params = SpreadParams::default();
        params.kind_weights.insert(EdgeKind::Contradicts, 0.0);
        let result = spread(&adj, &[seed("a", 1.0)], &params);
        assert!(find(&result, "b").is_none());
    }

    #[test]
    fn empty_seeds_returns_empty() {
        let edges = vec![edge("a", "b", EdgeKind::CausedBy, 1.0)];
        let adj = Adjacency::build(&edges);
        let params = SpreadParams::default();
        let result = spread(&adj, &[], &params);
        assert!(result.is_empty());
    }
}
