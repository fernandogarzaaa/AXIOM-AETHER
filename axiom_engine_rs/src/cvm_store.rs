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

/// Default for `AXIOM_CVM_MAX_INDEX_BYTES`: how much page text the in-memory
/// index may hold across *all* sessions before least-recently-used sessions
/// are dropped from memory (disk stays the source of truth; a dropped session
/// is reloaded lazily on its next touch).
pub const DEFAULT_MAX_INDEX_BYTES: usize = 256 * 1024 * 1024;

/// Default for `AXIOM_CVM_MAX_TOTAL_BYTES`: total on-disk size of the store
/// across all session files. Past it, the least-recently-used session files
/// are deleted outright (their pages become unrecoverable), so this is the
/// one lossy bound; size it to the volume, not to RAM.
pub const DEFAULT_MAX_TOTAL_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Bounds for a [`CvmStore`]. Session ids are client-chosen, so without a
/// cross-session bound a client rotating `X-Axiom-Session-Id` could grow the
/// index and the disk without limit (security audit addendum A9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CvmLimits {
    pub max_index_bytes: usize,
    pub max_total_bytes: u64,
}

impl Default for CvmLimits {
    fn default() -> Self {
        Self {
            max_index_bytes: DEFAULT_MAX_INDEX_BYTES,
            max_total_bytes: DEFAULT_MAX_TOTAL_BYTES,
        }
    }
}

impl CvmLimits {
    /// `AXIOM_CVM_MAX_INDEX_BYTES` / `AXIOM_CVM_MAX_TOTAL_BYTES`, each falling
    /// back to its default when unset, empty, zero or unparsable.
    pub fn from_env() -> Self {
        Self::parse(
            std::env::var("AXIOM_CVM_MAX_INDEX_BYTES").ok().as_deref(),
            std::env::var("AXIOM_CVM_MAX_TOTAL_BYTES").ok().as_deref(),
        )
    }

    /// Pure parser behind [`CvmLimits::from_env`].
    pub fn parse(index_bytes: Option<&str>, total_bytes: Option<&str>) -> Self {
        let d = Self::default();
        Self {
            max_index_bytes: index_bytes
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(d.max_index_bytes),
            max_total_bytes: total_bytes
                .and_then(|v| v.trim().parse::<u64>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(d.max_total_bytes),
        }
    }
}

/// One session's rows held in memory, with what the LRU needs.
struct SessionEntry {
    rows: Vec<CvmPage>,
    /// Sum of `rows[*].bytes`; kept current so the index budget check is O(1).
    bytes: usize,
    /// Monotonic touch stamp from [`Index::clock`]; larger = more recent.
    last_touch: u64,
}

/// Everything behind the store's single lock.
struct Index {
    sessions: HashMap<String, SessionEntry>,
    /// Sum of `sessions[*].bytes`.
    index_bytes: usize,
    /// On-disk size of every session file, keyed by file stem (the
    /// sanitised session id), including files written by earlier processes
    /// that this one has never touched.
    disk_bytes: HashMap<String, u64>,
    /// Recency stamp per file stem for the disk LRU: `(wall-clock seconds,
    /// in-process clock)`. Files found at `open` carry `(mtime, 0)`; a `put`
    /// refreshes to `(now, clock)`, so touches within the same second still
    /// order correctly.
    disk_touch: HashMap<String, (u64, u64)>,
    clock: u64,
}

impl Index {
    fn touch(&mut self, session_id: &str) {
        self.clock += 1;
        if let Some(entry) = self.sessions.get_mut(session_id) {
            entry.last_touch = self.clock;
        }
    }

    fn insert(&mut self, session_id: &str, rows: Vec<CvmPage>) {
        let bytes = rows.iter().map(|r| r.bytes).sum();
        self.clock += 1;
        let prev = self.sessions.insert(
            session_id.to_string(),
            SessionEntry {
                rows,
                bytes,
                last_touch: self.clock,
            },
        );
        self.index_bytes = self.index_bytes - prev.map_or(0, |p| p.bytes) + bytes;
    }

    fn remove(&mut self, session_id: &str) -> Option<SessionEntry> {
        let entry = self.sessions.remove(session_id)?;
        self.index_bytes -= entry.bytes;
        Some(entry)
    }

    fn disk_total(&self) -> u64 {
        self.disk_bytes.values().sum()
    }
}

/// Content-addressed, session-scoped store backed by append-only JSONL files
/// at `<root>/<session_id>.jsonl`, one file per session.
///
/// Three bounds apply, from cheapest to most drastic:
/// 1. per session, 64 MiB of page text (oldest pages dropped, file rewritten);
/// 2. in memory, [`CvmLimits::max_index_bytes`] across sessions (least
///    recently used sessions dropped from the index only -- lossless, they
///    reload from disk on the next touch);
/// 3. on disk, [`CvmLimits::max_total_bytes`] across session files (least
///    recently used session *files* deleted -- lossy, logged).
pub struct CvmStore {
    root: PathBuf,
    limits: CvmLimits,
    index: Arc<RwLock<Index>>,
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

/// A snapshot of the store's bookkeeping, for `/metrics` and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CvmStoreStats {
    pub sessions_in_memory: usize,
    pub index_bytes: usize,
    pub session_files: usize,
    pub disk_bytes: u64,
}

impl CvmStore {
    /// Open with limits from the environment (creating the root dir if
    /// needed). Scans the root once for existing session files so the disk
    /// bound counts what earlier processes left behind; rows are still
    /// loaded lazily per session.
    pub fn open(root: impl AsRef<Path>) -> std::io::Result<Self> {
        Self::open_with_limits(root, CvmLimits::from_env())
    }

    /// [`CvmStore::open`] with explicit bounds.
    pub fn open_with_limits(root: impl AsRef<Path>, limits: CvmLimits) -> std::io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        let mut disk_bytes = HashMap::new();
        let mut disk_touch = HashMap::new();
        for entry in fs::read_dir(&root)?.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() {
                continue;
            }
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            disk_bytes.insert(stem.to_string(), meta.len());
            disk_touch.insert(stem.to_string(), (mtime, 0));
        }
        Ok(Self {
            root,
            limits,
            index: Arc::new(RwLock::new(Index {
                sessions: HashMap::new(),
                index_bytes: 0,
                disk_bytes,
                disk_touch,
                clock: 0,
            })),
        })
    }

    /// The bounds this store enforces.
    pub fn limits(&self) -> CvmLimits {
        self.limits
    }

    /// Current bookkeeping snapshot.
    pub fn stats(&self) -> CvmStoreStats {
        let idx = self.index.read().unwrap_or_else(|e| e.into_inner());
        CvmStoreStats {
            sessions_in_memory: idx.sessions.len(),
            index_bytes: idx.index_bytes,
            session_files: idx.disk_bytes.len(),
            disk_bytes: idx.disk_total(),
        }
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

    /// Ensure `session_id`'s rows are in the index (loading from disk on
    /// first touch), stamp it as most recently used, and shed other
    /// sessions from memory if the index budget is exceeded. Returns the
    /// write guard so the caller can continue under the same lock.
    fn load_and_touch(
        &self,
        session_id: &str,
    ) -> std::io::Result<std::sync::RwLockWriteGuard<'_, Index>> {
        let loaded = self
            .index
            .read()
            .map(|idx| idx.sessions.contains_key(session_id))
            .unwrap_or(false);
        // Read the file outside the lock; another thread may race us to
        // insert the same session, in which case its copy wins and ours is
        // dropped (both were read from the same on-disk truth).
        let rows = if loaded {
            None
        } else {
            Some(self.load_session_rows(session_id))
        };
        let mut idx = self
            .index
            .write()
            .map_err(|_| std::io::Error::other("cvm store index lock poisoned"))?;
        match rows {
            Some(rows) if !idx.sessions.contains_key(session_id) => idx.insert(session_id, rows),
            _ => idx.touch(session_id),
        }
        self.shed_index(&mut idx, session_id);
        Ok(idx)
    }

    /// Drop least-recently-used sessions (never `keep`) from memory until
    /// the index is within budget. Disk is untouched, so this is lossless.
    fn shed_index(&self, idx: &mut Index, keep: &str) {
        let mut shed = 0usize;
        while idx.index_bytes > self.limits.max_index_bytes {
            let victim = idx
                .sessions
                .iter()
                .filter(|(id, _)| id.as_str() != keep)
                .min_by_key(|(_, e)| e.last_touch)
                .map(|(id, _)| id.clone());
            let Some(victim) = victim else { break };
            idx.remove(&victim);
            shed += 1;
        }
        if shed > 0 {
            eprintln!(
                "[axiom-cvm] index over {} bytes; dropped {shed} least-recently-used session(s) from memory (disk retained)",
                self.limits.max_index_bytes
            );
        }
    }

    /// Delete least-recently-used session *files* (never `keep`'s) until the
    /// store's on-disk total is within budget. Lossy: their pages can no
    /// longer be expanded, so every deletion is logged.
    fn shed_disk(&self, idx: &mut Index, keep: &str) {
        let keep_stem = session_file_stem(keep);
        let mut deleted = Vec::new();
        while idx.disk_total() > self.limits.max_total_bytes {
            let victim = idx
                .disk_bytes
                .keys()
                .filter(|stem| **stem != keep_stem)
                .min_by_key(|stem| idx.disk_touch.get(*stem).copied().unwrap_or((0, 0)))
                .cloned();
            let Some(stem) = victim else { break };
            let path = self.root.join(format!("{stem}.jsonl"));
            if let Err(e) = fs::remove_file(&path) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    eprintln!("[axiom-cvm] could not delete {}: {e}", path.display());
                    break;
                }
            }
            idx.disk_bytes.remove(&stem);
            idx.disk_touch.remove(&stem);
            let in_memory: Vec<String> = idx
                .sessions
                .keys()
                .filter(|id| session_file_stem(id) == stem)
                .cloned()
                .collect();
            for id in in_memory {
                idx.remove(&id);
            }
            deleted.push(stem);
        }
        if !deleted.is_empty() {
            eprintln!(
                "[axiom-cvm] store over {} bytes on disk; deleted {} least-recently-used session file(s): {}",
                self.limits.max_total_bytes,
                deleted.len(),
                deleted.join(", ")
            );
        }
    }

    fn encode_row(row: &CvmPage) -> std::io::Result<Vec<u8>> {
        let mut line = serde_json::to_vec(row)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        line.push(b'\n');
        Ok(line)
    }

    /// Append one row. On failure the file is truncated back to its
    /// previous length (best effort) so a half-written line cannot corrupt
    /// the next append. Returns the new file length.
    fn append_row(&self, session_id: &str, row: &CvmPage) -> std::io::Result<u64> {
        let path = self.session_path(session_id);
        let line = Self::encode_row(row)?;
        let mut f = OpenOptions::new().create(true).append(true).open(&path)?;
        let before = f.metadata()?.len();
        if let Err(e) = f.write_all(&line) {
            let _ = f.set_len(before);
            return Err(e);
        }
        Ok(before + line.len() as u64)
    }

    /// Rewrite the whole session file (only needed after a per-session
    /// eviction). Returns the new file length.
    fn rewrite_session_file(&self, session_id: &str, rows: &[CvmPage]) -> std::io::Result<u64> {
        let path = self.session_path(session_id);
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let mut written = 0u64;
        for row in rows {
            let line = Self::encode_row(row)?;
            f.write_all(&line)?;
            written += line.len() as u64;
        }
        Ok(written)
    }

    /// Store `original_text` for `session_id` under `kind`
    /// (`tool_result` / `file` / `output`), returning its content-addressed
    /// `PageId`. If the session's total stored bytes would exceed the
    /// 64 MiB cap, the oldest rows are dropped first (and the eviction
    /// logged) to make room. Then the cross-session bounds are applied
    /// (see the type docs).
    pub fn put(
        &self,
        session_id: &str,
        kind: &str,
        original_text: &str,
    ) -> std::io::Result<PageId> {
        let page_id = Self::page_id_for(original_text);
        let row = CvmPage {
            page_id: page_id.clone(),
            created_unix_s: now_secs(),
            kind: kind.to_string(),
            bytes: original_text.len(),
            text: original_text.to_string(),
        };

        let mut idx = self.load_and_touch(session_id)?;
        let stem = session_file_stem(session_id);
        let entry = idx
            .sessions
            .get_mut(session_id)
            .expect("load_and_touch inserted the session");
        // Page ids are content hashes: the same text paged again (e.g. a
        // repeated break-window rebase over an unchanged transcript) must be
        // a no-op, not a duplicate row + file write.
        if entry.rows.iter().any(|r| r.page_id == page_id) {
            return Ok(page_id);
        }

        // Mutate in place and roll back on a failed write, rather than
        // cloning the row set per call (O(session size) for a session near
        // the cap). After rollback the index still matches disk exactly.
        let added = row.bytes;
        entry.rows.push(row);
        entry.bytes += added;
        let mut evicted: Vec<CvmPage> = Vec::new();
        while entry.bytes > MAX_SESSION_BYTES && entry.rows.len() > 1 {
            let removed = entry.rows.remove(0);
            entry.bytes -= removed.bytes;
            evicted.push(removed);
        }
        if !evicted.is_empty() {
            eprintln!(
                "[axiom-cvm] evicted {} page(s) from session={session_id} (over {MAX_SESSION_BYTES}-byte cap)",
                evicted.len()
            );
        }

        // Append-only on the common path; a full rewrite only when rows
        // were evicted (the file must shrink).
        let write = if evicted.is_empty() {
            self.append_row(session_id, entry.rows.last().expect("just pushed"))
        } else {
            self.rewrite_session_file(session_id, &entry.rows)
        };
        let file_len = match write {
            Ok(len) => len,
            Err(e) => {
                let popped = entry.rows.pop();
                entry.bytes -= popped.map_or(0, |r| r.bytes);
                for r in evicted.into_iter().rev() {
                    entry.bytes += r.bytes;
                    entry.rows.insert(0, r);
                }
                return Err(e);
            }
        };
        idx.index_bytes = idx.index_bytes + added - evicted_bytes(&evicted);
        idx.disk_bytes.insert(stem.clone(), file_len);
        idx.clock += 1;
        let stamp = (now_secs(), idx.clock);
        idx.disk_touch.insert(stem, stamp);
        self.shed_index(&mut idx, session_id);
        self.shed_disk(&mut idx, session_id);
        Ok(page_id)
    }

    /// Retrieve the full original text for `page_id` within `session_id`,
    /// or `None` if the session or page is unknown.
    pub fn get(&self, session_id: &str, page_id: &str) -> Option<String> {
        let idx = self.load_and_touch(session_id).ok()?;
        idx.sessions
            .get(session_id)?
            .rows
            .iter()
            .rev()
            .find(|r| r.page_id == page_id)
            .map(|r| r.text.clone())
    }

    /// Delete a session's CVM file and drop it from the in-memory index.
    /// Called on session drop unless `AXIOM_CVM_RETAIN=1`.
    pub fn delete_session(&self, session_id: &str) -> std::io::Result<()> {
        let stem = session_file_stem(session_id);
        if let Ok(mut idx) = self.index.write() {
            idx.remove(session_id);
            idx.disk_bytes.remove(&stem);
            idx.disk_touch.remove(&stem);
        }
        let path = self.session_path(session_id);
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    #[cfg(test)]
    pub(crate) fn session_path_for_test(&self, session_id: &str) -> PathBuf {
        self.session_path(session_id)
    }
}

fn evicted_bytes(evicted: &[CvmPage]) -> usize {
    evicted.iter().map(|r| r.bytes).sum()
}

/// Build the canonical single-line stub for a digested page. Must survive
/// model round-trips verbatim: no embedded newlines.
/// Format: `[AXIOM-PAGE <page_id> session="<session_id>" <orig_tokens>tok <kind>] <first 120 chars>...`
///
/// The `session="..."` attribute is what `axiom_expand`'s `session_id`
/// parameter reads (see its tool schema in `mcp_stdio.rs`): without it the
/// agent cannot recover the full text from the stub alone.
pub fn build_stub(
    page_id: &str,
    session_id: &str,
    orig_tokens: usize,
    kind: &str,
    original_text: &str,
) -> String {
    let snippet: String = original_text.chars().take(120).collect();
    let snippet = snippet.replace(['\n', '\r'], " ");
    // JSON-encode the session id so quotes, backslashes, and brackets in a
    // hostile value round-trip instead of breaking the stub grammar.
    let session_attr = serde_json::to_string(session_id).expect("session IDs are serializable");
    format!("[AXIOM-PAGE {page_id} session={session_attr} {orig_tokens}tok {kind}] {snippet}...")
}

/// The fields recovered by parsing a stub line built by [`build_stub`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StubInfo {
    pub page_id: String,
    /// `None` for stubs built before the `session="..."` attribute existed.
    pub session_id: Option<String>,
    pub orig_tokens: usize,
    pub kind: String,
    pub snippet: String,
}

/// Split an optional `session=<json-string>` attribute out of a stub header,
/// returning `(header_without_attribute, session_value)`. Old-format stubs
/// without the attribute round-trip as `(header, None)`. The value is a
/// JSON string (see [`build_stub`]); it is decoded so escaped quotes and
/// backslashes resolve to the original session id.
fn split_session_attr(header: &str) -> (String, Option<String>) {
    const MARKER: &str = "session=\"";
    let Some(start) = header.find(MARKER) else {
        return (header.to_string(), None);
    };
    let val_start = start + MARKER.len();
    // Scan for the closing quote, skipping JSON escapes (\" and \\).
    let bytes = header.as_bytes();
    let mut i = val_start;
    let mut end = None;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2, // skip the escaped char
            b'"' => {
                end = Some(i);
                break;
            }
            _ => i += 1,
        }
    }
    let Some(end) = end else {
        return (header.to_string(), None);
    };
    // The raw JSON string including quotes, e.g. `"a\"b"`.
    let raw = &header[val_start - 1..=end];
    let session: String = match serde_json::from_str(raw) {
        Ok(s) => s,
        Err(_) => return (header.to_string(), None),
    };
    let mut cleaned = String::with_capacity(header.len());
    cleaned.push_str(header[..start].trim_end());
    cleaned.push(' ');
    cleaned.push_str(header[end + 1..].trim_start());
    (cleaned, Some(session))
}

/// Parse a stub line built by [`build_stub`]. `None` if `line` isn't a
/// well-formed `AXIOM-PAGE` stub.
pub fn parse_stub(line: &str) -> Option<StubInfo> {
    let line = line.trim();
    let rest = line.strip_prefix("[AXIOM-PAGE ")?;
    let (header, tail) = rest.split_once(']')?;
    let (header, session_id) = split_session_attr(header);
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
        session_id,
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
        let stub = build_stub(
            "a1b2c3d4e5f60718",
            "sess-abc-123",
            4231,
            "tool_result",
            "line one of output",
        );
        assert_eq!(
            stub,
            "[AXIOM-PAGE a1b2c3d4e5f60718 session=\"sess-abc-123\" 4231tok tool_result] line one of output..."
        );
        let parsed = parse_stub(&stub).unwrap();
        assert_eq!(parsed.page_id, "a1b2c3d4e5f60718");
        assert_eq!(parsed.session_id, Some("sess-abc-123".to_string()));
        assert_eq!(parsed.orig_tokens, 4231);
        assert_eq!(parsed.kind, "tool_result");
        assert_eq!(parsed.snippet, "line one of output");
    }

    #[test]
    fn stub_build_parse_round_trip_with_unicode_and_truncation() {
        let original = "caf\u{e9} \u{2192} \u{5317}\u{4eac} ".repeat(20);
        let stub = build_stub("00112233445566aa", "s1", 999, "output", &original);
        let parsed = parse_stub(&stub).unwrap();
        assert_eq!(parsed.page_id, "00112233445566aa");
        assert_eq!(parsed.session_id, Some("s1".to_string()));
        assert_eq!(parsed.orig_tokens, 999);
        assert_eq!(parsed.kind, "output");
        let expected_snippet: String = original.chars().take(120).collect();
        assert_eq!(parsed.snippet, expected_snippet);
        assert!(!stub.contains('\n'));
    }

    #[test]
    fn parse_stub_accepts_legacy_stubs_without_session() {
        // Stubs built before the session="..." attribute existed must still
        // parse, with session_id == None.
        let parsed =
            parse_stub("[AXIOM-PAGE a1b2c3d4e5f60718 4231tok tool_result] line one...").unwrap();
        assert_eq!(parsed.page_id, "a1b2c3d4e5f60718");
        assert_eq!(parsed.session_id, None);
        assert_eq!(parsed.orig_tokens, 4231);
    }

    #[test]
    fn parse_stub_session_value_may_contain_spaces() {
        let stub = build_stub("a1b2c3d4e5f60718", "my session 42", 10, "tool_result", "x");
        let parsed = parse_stub(&stub).unwrap();
        assert_eq!(parsed.session_id, Some("my session 42".to_string()));
        assert_eq!(parsed.orig_tokens, 10);
        assert_eq!(parsed.kind, "tool_result");
    }

    #[test]
    fn build_stub_json_encodes_session_id_with_quotes() {
        // A hostile session id must round-trip through JSON encoding.
        let stub = build_stub("a1b2c3d4e5f60718", "se\"ss", 10, "tool_result", "x");
        assert!(stub.contains("session=\"se\\\"ss\""));
        let parsed = parse_stub(&stub).unwrap();
        assert_eq!(parsed.session_id, Some("se\"ss".to_string()));
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

    fn line_count(path: &Path) -> usize {
        fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count()
    }

    #[test]
    fn put_appends_a_line_instead_of_rewriting() {
        let dir = tempdir();
        let store = CvmStore::open_with_limits(&dir, CvmLimits::default()).unwrap();
        store.put("s1", "tool_result", "first").unwrap();
        let path = store.session_path_for_test("s1");
        let after_one = fs::metadata(&path).unwrap().len();
        store.put("s1", "tool_result", "second").unwrap();
        let after_two = fs::metadata(&path).unwrap().len();
        assert_eq!(line_count(&path), 2);
        // The first line is byte-identical after the second put: it was
        // appended to, not rewritten.
        let text = fs::read_to_string(&path).unwrap();
        let first_line = text.lines().next().unwrap();
        assert!(first_line.contains("\"first\""), "{first_line}");
        assert!(after_two > after_one);
        // Bookkeeping tracks the real file size.
        assert_eq!(store.stats().disk_bytes, after_two);
        assert_eq!(store.stats().session_files, 1);
    }

    #[test]
    fn index_budget_sheds_least_recently_used_session_but_keeps_disk() {
        let dir = tempdir();
        let limits = CvmLimits {
            max_index_bytes: 40,
            max_total_bytes: DEFAULT_MAX_TOTAL_BYTES,
        };
        let store = CvmStore::open_with_limits(&dir, limits).unwrap();
        let a = store.put("a", "file", "a".repeat(30).as_str()).unwrap();
        assert_eq!(store.stats().sessions_in_memory, 1);
        // Session b's 30 bytes push the index past 40: a is shed from memory.
        let b = store.put("b", "file", "b".repeat(30).as_str()).unwrap();
        let stats = store.stats();
        assert_eq!(stats.sessions_in_memory, 1, "{stats:?}");
        assert!(stats.index_bytes <= 40, "{stats:?}");
        assert_eq!(stats.session_files, 2, "{stats:?}");
        // ...but a is still on disk and reloads transparently.
        assert_eq!(store.get("a", &a), Some("a".repeat(30)));
        // Now b is the least recently used and gets shed in turn; still readable.
        assert_eq!(store.get("b", &b), Some("b".repeat(30)));
        assert_eq!(store.stats().session_files, 2);
    }

    #[test]
    fn disk_budget_deletes_least_recently_used_session_file_never_the_current_one() {
        let dir = tempdir();
        // Each 150-char page serialises to roughly 246 bytes of JSONL
        // (field names, id, timestamp), so two files fit a 600-byte budget
        // and a third does not.
        let limits = CvmLimits {
            max_index_bytes: DEFAULT_MAX_INDEX_BYTES,
            max_total_bytes: 600,
        };
        let store = CvmStore::open_with_limits(&dir, limits).unwrap();
        let old = store.put("old", "file", "o".repeat(150).as_str()).unwrap();
        let mid = store.put("mid", "file", "m".repeat(150).as_str()).unwrap();
        assert_eq!(store.stats().session_files, 2);
        // A third session pushes the on-disk total past 600 bytes: the least
        // recently used file ("old") is deleted; the writer's own survives.
        let new = store.put("new", "file", "n".repeat(150).as_str()).unwrap();
        let stats = store.stats();
        assert!(stats.disk_bytes <= 600, "{stats:?}");
        assert!(!store.session_path_for_test("old").exists());
        assert_eq!(store.get("old", &old), None);
        assert_eq!(store.get("new", &new), Some("n".repeat(150)));
        // "mid" was more recent than "old"; whether it survived depends only
        // on the budget, and if it did it must still read back.
        if store.session_path_for_test("mid").exists() {
            assert_eq!(store.get("mid", &mid), Some("m".repeat(150)));
        }
    }

    #[test]
    fn open_counts_session_files_left_by_an_earlier_process() {
        let dir = tempdir();
        {
            let store = CvmStore::open_with_limits(&dir, CvmLimits::default()).unwrap();
            store.put("s1", "file", "persisted").unwrap();
            store.put("s2", "file", "also persisted").unwrap();
        }
        let store2 = CvmStore::open_with_limits(&dir, CvmLimits::default()).unwrap();
        let stats = store2.stats();
        assert_eq!(stats.session_files, 2, "{stats:?}");
        assert!(stats.disk_bytes > 0, "{stats:?}");
        assert_eq!(stats.sessions_in_memory, 0, "rows load lazily: {stats:?}");
    }

    #[test]
    fn delete_session_updates_disk_bookkeeping() {
        let dir = tempdir();
        let store = CvmStore::open_with_limits(&dir, CvmLimits::default()).unwrap();
        store.put("s1", "file", "bytes").unwrap();
        assert_eq!(store.stats().session_files, 1);
        store.delete_session("s1").unwrap();
        let stats = store.stats();
        assert_eq!(stats.session_files, 0, "{stats:?}");
        assert_eq!(stats.disk_bytes, 0, "{stats:?}");
        assert_eq!(stats.index_bytes, 0, "{stats:?}");
    }

    #[test]
    fn limits_parse_falls_back_on_bad_input() {
        let d = CvmLimits::default();
        assert_eq!(CvmLimits::parse(None, None), d);
        assert_eq!(CvmLimits::parse(Some("0"), Some("")), d);
        assert_eq!(CvmLimits::parse(Some("lots"), Some("-1")), d);
        let custom = CvmLimits::parse(Some(" 1024 "), Some("2048"));
        assert_eq!(custom.max_index_bytes, 1024);
        assert_eq!(custom.max_total_bytes, 2048);
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
