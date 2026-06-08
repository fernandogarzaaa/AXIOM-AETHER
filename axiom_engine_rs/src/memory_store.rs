//! Tier-2 lossless memory store: append-only JSONL, one file per scope.
//! Records carry the exact original text plus an L2-normalized embedding, so
//! recall returns real content (never a neural hallucination).

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What a memory is about. Extensible; serialized lower-case.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    Decision,
    Code,
    Conversation,
    Fix,
}

/// One stored memory.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemoryRecord {
    pub id: String,
    pub scope: String,
    pub ts: u64,
    pub kind: MemoryKind,
    pub body: String,
    pub embedding: Vec<f32>,
    pub drift_at_ingest: f32,
    #[serde(default)]
    pub supersedes: Option<String>,
    #[serde(default)]
    pub tombstone: bool,
}

/// Append-only JSONL store rooted at a directory. One `<scope>.jsonl` per scope.
pub struct MemoryStore {
    root: PathBuf,
}

/// Sanitize a scope into a filesystem-safe filename stem.
fn scope_file_stem(scope: &str) -> String {
    scope
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl MemoryStore {
    /// Open (creating the root dir if needed).
    pub fn open(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Path to a scope's JSONL file. Public for tests that append raw lines.
    pub fn scope_path(&self, scope: &str) -> PathBuf {
        self.root.join(format!("{}.jsonl", scope_file_stem(scope)))
    }

    /// Append one record to its scope file.
    pub fn append(&self, rec: &MemoryRecord) -> std::io::Result<()> {
        let path = self.scope_path(&rec.scope);
        let mut f = OpenOptions::new().create(true).append(true).open(path)?;
        let line = serde_json::to_string(rec)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }

    /// Load the live records for a scope: latest-write-wins per id, with any id
    /// whose latest record is a tombstone excluded. A corrupt line is skipped
    /// (fail-open), never fatal.
    pub fn load_scope(&self, scope: &str) -> Vec<MemoryRecord> {
        let path = self.scope_path(scope);
        let Ok(file) = fs::File::open(&path) else {
            return Vec::new();
        };
        let mut latest: std::collections::HashMap<String, MemoryRecord> =
            std::collections::HashMap::new();
        for line in BufReader::new(file).lines().map_while(std::io::Result::ok) {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(rec) = serde_json::from_str::<MemoryRecord>(line) {
                latest.insert(rec.id.clone(), rec);
            }
        }
        latest.into_values().filter(|r| !r.tombstone).collect()
    }

    /// Tombstone an id within a scope by appending a tombstone record.
    pub fn tombstone(&self, scope: &str, id: &str) -> std::io::Result<()> {
        let rec = MemoryRecord {
            id: id.to_string(),
            scope: scope.to_string(),
            ts: now_secs(),
            kind: MemoryKind::Conversation,
            body: String::new(),
            embedding: Vec::new(),
            drift_at_ingest: 0.0,
            supersedes: None,
            tombstone: true,
        };
        self.append(&rec)
    }
}

/// Current unix time in seconds (0 on clock error — never panics).
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Cosine similarity between two equal-length vectors. Returns 0.0 on length
/// mismatch or empty/zero input (defensive — recall must never panic).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Rank `records` by cosine similarity of their embedding to `query`, returning
/// up to `k` `(similarity, record)` pairs, highest first. Records with empty
/// embeddings are skipped.
pub fn top_k<'a>(
    query: &[f32],
    records: &'a [MemoryRecord],
    k: usize,
) -> Vec<(f32, &'a MemoryRecord)> {
    let mut scored: Vec<(f32, &MemoryRecord)> = records
        .iter()
        .filter(|r| !r.embedding.is_empty())
        .map(|r| (cosine(query, &r.embedding), r))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: &str, scope: &str, body: &str, emb: Vec<f32>) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            scope: scope.to_string(),
            ts: now_secs(),
            kind: MemoryKind::Decision,
            body: body.to_string(),
            embedding: emb,
            drift_at_ingest: 5.0,
            supersedes: None,
            tombstone: false,
        }
    }

    fn temp_root(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "axiom_mem_test_{tag}_{}_{}",
            now_secs(),
            std::process::id()
        ));
        p
    }

    #[test]
    fn append_then_load_roundtrip() {
        let root = temp_root("roundtrip");
        let store = MemoryStore::open(&root).unwrap();
        store
            .append(&rec("a", "personal", "use 4-space indent", vec![1.0, 0.0]))
            .unwrap();
        let loaded = store.load_scope("personal");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].body, "use 4-space indent");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn latest_write_wins_per_id() {
        let root = temp_root("latest");
        let store = MemoryStore::open(&root).unwrap();
        store
            .append(&rec("x", "personal", "old", vec![1.0]))
            .unwrap();
        store
            .append(&rec("x", "personal", "new", vec![1.0]))
            .unwrap();
        let loaded = store.load_scope("personal");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].body, "new");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn tombstone_excludes_record() {
        let root = temp_root("tomb");
        let store = MemoryStore::open(&root).unwrap();
        store
            .append(&rec("k", "personal", "secret-ish", vec![1.0]))
            .unwrap();
        store.tombstone("personal", "k").unwrap();
        let loaded = store.load_scope("personal");
        assert!(loaded.is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn scopes_are_isolated() {
        let root = temp_root("iso");
        let store = MemoryStore::open(&root).unwrap();
        store
            .append(&rec("p", "project:aaa", "repo A secret", vec![1.0]))
            .unwrap();
        store
            .append(&rec("q", "project:bbb", "repo B secret", vec![1.0]))
            .unwrap();
        let a = store.load_scope("project:aaa");
        let b = store.load_scope("project:bbb");
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(a[0].body, "repo A secret");
        assert_eq!(b[0].body, "repo B secret");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn corrupt_line_is_skipped() {
        let root = temp_root("corrupt");
        let store = MemoryStore::open(&root).unwrap();
        store
            .append(&rec("good", "personal", "ok", vec![1.0]))
            .unwrap();
        let path = store.scope_path("personal");
        let mut f = OpenOptions::new().append(true).open(path).unwrap();
        f.write_all(b"{ this is not json\n").unwrap();
        let loaded = store.load_scope("personal");
        assert_eq!(loaded.len(), 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![0.6, 0.8];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_length_mismatch_is_zero() {
        assert_eq!(cosine(&[1.0, 0.0], &[1.0]), 0.0);
    }

    #[test]
    fn top_k_ranks_by_similarity() {
        let recs = vec![
            rec("near", "personal", "near", vec![1.0, 0.0]),
            rec("far", "personal", "far", vec![0.0, 1.0]),
            rec("mid", "personal", "mid", vec![0.7071, 0.7071]),
        ];
        let ranked = top_k(&[1.0, 0.0], &recs, 2);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].1.id, "near");
        assert_eq!(ranked[1].1.id, "mid");
    }

    #[test]
    fn top_k_skips_empty_embeddings() {
        let recs = vec![rec("e", "personal", "empty", vec![])];
        let ranked = top_k(&[1.0, 0.0], &recs, 5);
        assert!(ranked.is_empty());
    }
}
