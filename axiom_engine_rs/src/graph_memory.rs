//! Graph working memory: directed, weighted edges between `MemoryRecord`s.
//! Storage mirrors `MemoryStore`'s append-only JSONL durability model, but
//! edges live in a single `edges.jsonl` at the memory root rather than one
//! file per scope, since edges legitimately cross scopes.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::memory_store::{now_secs, MemoryRecord};

/// The relationship a `MemoryEdge` records between two `MemoryRecord`s.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
}
