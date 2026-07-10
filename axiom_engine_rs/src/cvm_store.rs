//! S2 (CVM cost stack): content-addressed L2 store.
//!
//! Digest admission control (S3) will replace heavy transcript content with a
//! digest + one-line stub, but the full-fidelity original must stay
//! recoverable on demand (`POST /v1/expand`, the `axiom_expand` MCP tool).
//! This module is that store: `put` persists text under a session,
//! content-addressed by a short id; `get` retrieves it back. See
//! docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S2.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub type PageId = String;

/// One stored page: the full original text plus enough metadata to build a
/// stub and to age it out under the per-session byte cap.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct CvmPage {
    pub page_id: String,
    pub created_unix_s: u64,
    pub kind: String,
    pub bytes: usize,
    pub text: String,
}

/// Per-session cap: oldest rows are dropped (and the eviction logged) once a
/// session's total stored bytes would exceed this on a write.
const MAX_SESSION_BYTES: usize = 64 * 1024 * 1024;

/// Content-addressed, session-scoped store backed by append-only JSONL files
/// at `<root>/<session_id>.jsonl`, one file per session.
pub struct CvmStore {
    root: PathBuf,
    /// Per-session in-memory index, lazily populated from disk the first
    /// time a session is touched (`put` or `get`) and kept current
    /// thereafter -- avoids re-scanning the JSONL file on every call.
    index: Arc<RwLock<HashMap<String, Vec<CvmPage>>>>,
}

/// Sanitize a session id into a filesystem-safe filename stem.
fn session_file_stem(session_id: &str) -> String {
    session_id
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

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl CvmStore {
    /// Open (creating the root dir if needed). Does not eagerly scan disk;
    /// each session's rows are loaded lazily on first access.
    pub fn open(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            index: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.root
            .join(format!("{}.jsonl", session_file_stem(session_id)))
    }

    /// The content-addressed id for `text`: first 16 hex chars of its
    /// SHA-256 digest. Deterministic -- identical text always maps to the
    /// same `PageId`, so re-`put`ting identical content is idempotent.
    pub fn page_id_for(text: &str) -> PageId {
        let digest = Sha256::digest(text.as_bytes());
        format!("{digest:x}")[..16].to_string()
    }

    fn load_session_rows(&self, session_id: &str) -> Vec<CvmPage> {
        let path = self.session_path(session_id);
        let Ok(file) = fs::File::open(&path) else {
            return Vec::new();
        };
        BufReader::new(file)
            .lines()
            .map_while(std::io::Result::ok)
            .filter_map(|line| {
                let line = line.trim().to_string();
                if line.is_empty() {
                    None
                } else {
                    serde_json::from_str::<CvmPage>(&line).ok()
                }
            })
            .collect()
    }

    /// Ensure `session_id`'s rows are loaded into the in-memory index,
    /// loading from disk on first touch.
    fn ensure_loaded(&self, session_id: &str) {
        let already_loaded = self
            .index
            .read()
            .map(|idx| idx.contains_key(session_id))
            .unwrap_or(false);
        if already_loaded {
            return;
        }
        let rows = self.load_session_rows(session_id);
        if let Ok(mut idx) = self.index.write() {
            idx.entry(session_id.to_string()).or_insert(rows);
        }
    }

    fn rewrite_session_file(&self, session_id: &str, rows: &[CvmPage]) -> std::io::Result<()> {
        let path = self.session_path(session_id);
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        for row in rows {
            let line = serde_json::to_string(row)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            f.write_all(line.as_bytes())?;
            f.write_all(b"\n")?;
        }
        Ok(())
    }

    /// Store `original_text` for `session_id` under `kind`
    /// (`tool_result` / `file` / `output`), returning its content-addressed
    /// `PageId`. If the session's total stored bytes would exceed the
    /// 64 MiB cap, the oldest rows are dropped first (and the eviction
    /// logged) to make room.
    pub fn put(&self, session_id: &str, kind: &str, original_text: &str) -> std::io::Result<PageId> {
        self.ensure_loaded(session_id);
        let page_id = Self::page_id_for(original_text);
        let row = CvmPage {
            page_id: page_id.clone(),
            created_unix_s: now_secs(),
            kind: kind.to_string(),
            bytes: original_text.len(),
            text: original_text.to_string(),
        };

        let mut idx = self
            .index
            .write()
            .map_err(|_| std::io::Error::other("cvm store index lock poisoned"))?;
        let rows = idx.entry(session_id.to_string()).or_default();
        rows.push(row);

        let mut total: usize = rows.iter().map(|r| r.bytes).sum();
        let mut evicted = 0usize;
        while total > MAX_SESSION_BYTES && rows.len() > 1 {
            let removed = rows.remove(0);
            total -= removed.bytes;
            evicted += 1;
        }
        if evicted > 0 {
            eprintln!(
                "[axiom-cvm] evicted {evicted} page(s) from session={session_id} (over {MAX_SESSION_BYTES}-byte cap)"
            );
        }
        self.rewrite_session_file(session_id, rows)?;
        Ok(page_id)
    }

    /// Retrieve the full original text for `page_id` within `session_id`,
    /// or `None` if the session or page is unknown.
    pub fn get(&self, session_id: &str, page_id: &str) -> Option<String> {
        self.ensure_loaded(session_id);
        let idx = self.index.read().ok()?;
        idx.get(session_id)?
            .iter()
            .rev()
            .find(|r| r.page_id == page_id)
            .map(|r| r.text.clone())
    }

    /// Delete a session's CVM file and drop it from the in-memory index.
    /// Called on session drop unless `AXIOM_CVM_RETAIN=1`.
    pub fn delete_session(&self, session_id: &str) -> std::io::Result<()> {
        if let Ok(mut idx) = self.index.write() {
            idx.remove(session_id);
        }
        let path = self.session_path(session_id);
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    #[cfg(test)]
    fn session_path_for_test(&self, session_id: &str) -> PathBuf {
        self.session_path(session_id)
    }
}

/// Build the canonical single-line stub for a digested page. Must survive
/// model round-trips verbatim: no embedded newlines.
/// Format: `[AXIOM-PAGE <page_id> <orig_tokens>tok <kind>] <first 120 chars>...`
pub fn build_stub(page_id: &str, orig_tokens: usize, kind: &str, original_text: &str) -> String {
    let snippet: String = original_text.chars().take(120).collect();
    let snippet = snippet.replace(['\n', '\r'], " ");
    format!("[AXIOM-PAGE {page_id} {orig_tokens}tok {kind}] {snippet}...")
}

/// The fields recovered by parsing a stub line built by [`build_stub`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StubInfo {
    pub page_id: String,
    pub orig_tokens: usize,
    pub kind: String,
    pub snippet: String,
}

/// Parse a stub line built by [`build_stub`]. `None` if `line` isn't a
/// well-formed `AXIOM-PAGE` stub.
pub fn parse_stub(line: &str) -> Option<StubInfo> {
    let line = line.trim();
    let rest = line.strip_prefix("[AXIOM-PAGE ")?;
    let (header, tail) = rest.split_once(']')?;
    let mut parts = header.split_whitespace();
    let page_id = parts.next()?.to_string();
    let tok_part = parts.next()?;
    let orig_tokens: usize = tok_part.strip_suffix("tok")?.parse().ok()?;
    let kind = parts.next()?.to_string();
    let snippet = tail
        .strip_prefix(' ')
        .unwrap_or(tail)
        .strip_suffix("...")
        .unwrap_or(tail)
        .to_string();
    Some(StubInfo {
        page_id,
        orig_tokens,
        kind,
        snippet,
    })
}

/// Is `symbol` shaped like a `CvmStore` page id (16 lowercase hex chars)?
/// Used by the `/v1/expand` route to decide whether to try the CVM store
/// before falling back to skeleton symbol expansion.
pub fn looks_like_page_id(symbol: &str) -> bool {
    symbol.len() == 16 && symbol.chars().all(|c| c.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_round_trip() {
        let dir = tempdir();
        let store = CvmStore::open(&dir).unwrap();
        let page_id = store.put("s1", "tool_result", "hello world").unwrap();
        assert_eq!(store.get("s1", &page_id), Some("hello world".to_string()));
    }

    #[test]
    fn get_unknown_page_or_session_is_none() {
        let dir = tempdir();
        let store = CvmStore::open(&dir).unwrap();
        assert_eq!(store.get("nope", "0000000000000000"), None);
        let page_id = store.put("s1", "file", "content").unwrap();
        assert!(store.get("s1", &page_id).is_some());
        assert_eq!(store.get("s1", "0000000000000000"), None);
    }

    #[test]
    fn put_is_content_addressed_and_deterministic() {
        let dir = tempdir();
        let store = CvmStore::open(&dir).unwrap();
        let a = store.put("s1", "output", "same text").unwrap();
        let b = store.put("s1", "output", "same text").unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn round_trip_survives_a_process_restart() {
        let dir = tempdir();
        let page_id = {
            let store = CvmStore::open(&dir).unwrap();
            store.put("s1", "tool_result", "persisted content").unwrap()
        };
        // Fresh store instance over the same root -- simulates a restart,
        // proving the on-disk JSONL (not just the in-memory index) is the
        // source of truth.
        let store2 = CvmStore::open(&dir).unwrap();
        assert_eq!(
            store2.get("s1", &page_id),
            Some("persisted content".to_string())
        );
    }

    #[test]
    fn cap_eviction_drops_oldest_rows_and_keeps_newest_retrievable() {
        let dir = tempdir();
        let store = CvmStore::open(&dir).unwrap();
        // Each `put` rewrites the whole session file, so keep this to a
        // small number of large pages (not many small ones) to cross the
        // 64 MiB cap without an O(n^2) blowup in test runtime.
        let big = "x".repeat(10 * 1024 * 1024);
        let mut ids = Vec::new();
        for i in 0..8 {
            let text = format!("{big}-{i}");
            ids.push(store.put("s1", "tool_result", &text).unwrap());
        }
        // The earliest page must have been evicted...
        assert_eq!(store.get("s1", &ids[0]), None);
        // ...while the most recent one must still be retrievable.
        let last = ids.last().unwrap();
        assert!(store.get("s1", last).is_some());
    }

    #[test]
    fn delete_session_removes_file_and_index_entry() {
        let dir = tempdir();
        let store = CvmStore::open(&dir).unwrap();
        let page_id = store.put("s1", "file", "gone soon").unwrap();
        assert!(store.get("s1", &page_id).is_some());
        store.delete_session("s1").unwrap();
        assert_eq!(store.get("s1", &page_id), None);
        assert!(!store.session_path_for_test("s1").exists());
    }

    #[test]
    fn delete_session_on_missing_file_is_not_an_error() {
        let dir = tempdir();
        let store = CvmStore::open(&dir).unwrap();
        assert!(store.delete_session("never-existed").is_ok());
    }

    #[test]
    fn stub_build_parse_round_trip() {
        let stub = build_stub("a1b2c3d4e5f60718", 4231, "tool_result", "line one of output");
        assert_eq!(
            stub,
            "[AXIOM-PAGE a1b2c3d4e5f60718 4231tok tool_result] line one of output..."
        );
        let parsed = parse_stub(&stub).unwrap();
        assert_eq!(parsed.page_id, "a1b2c3d4e5f60718");
        assert_eq!(parsed.orig_tokens, 4231);
        assert_eq!(parsed.kind, "tool_result");
        assert_eq!(parsed.snippet, "line one of output");
    }

    #[test]
    fn stub_build_parse_round_trip_with_unicode_and_truncation() {
        let original = "caf\u{e9} \u{2192} \u{5317}\u{4eac} ".repeat(20);
        let stub = build_stub("00112233445566aa", 999, "output", &original);
        let parsed = parse_stub(&stub).unwrap();
        assert_eq!(parsed.page_id, "00112233445566aa");
        assert_eq!(parsed.orig_tokens, 999);
        assert_eq!(parsed.kind, "output");
        let expected_snippet: String = original.chars().take(120).collect();
        assert_eq!(parsed.snippet, expected_snippet);
        assert!(!stub.contains('\n'));
    }

    #[test]
    fn parse_stub_rejects_malformed_lines() {
        assert!(parse_stub("not a stub at all").is_none());
        assert!(parse_stub("[AXIOM-PAGE onlyid]").is_none());
        assert!(parse_stub("[AXIOM-PAGE id notatoken kind] snippet").is_none());
    }

    #[test]
    fn looks_like_page_id_matches_only_16_hex_chars() {
        assert!(looks_like_page_id("a1b2c3d4e5f60718"));
        assert!(!looks_like_page_id("a1b2c3d4e5f6071")); // 15 chars
        assert!(!looks_like_page_id("a1b2c3d4e5f60718a")); // 17 chars
        assert!(!looks_like_page_id("g1b2c3d4e5f60718")); // non-hex 'g'
        assert!(!looks_like_page_id("MySymbolName"));
    }

    /// Minimal temp-dir helper (crate has no `tempfile` dependency): a
    /// unique subdirectory under `std::env::temp_dir()`, cleaned up on drop.
    struct TempDir(PathBuf);
    impl AsRef<Path> for TempDir {
        fn as_ref(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir() -> TempDir {
        let path = std::env::temp_dir().join(format!(
            "axiom-cvm-store-test-{}-{}",
            std::process::id(),
            nanos_jitter()
        ));
        fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }
    fn nanos_jitter() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        n.wrapping_add(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}
