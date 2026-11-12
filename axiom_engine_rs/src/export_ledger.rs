//! Which fleet exports a store has already counted.
//!
//! A signed export ([`crate::provenance::SignedExport`]) says who produced it,
//! not how many times it was delivered. Anyone who has seen one valid export
//! can deliver it again, and a store that counted every delivery would let
//! them raise a patch candidate's confirmation count, or compound a heal
//! record's evidence, without a single new verification. The shortest version
//! needs no peer: read a node's own export and post it straight back.
//!
//! So each store has a ledger beside it, `<store>.exports.json`, holding the
//! SHA-256 digests of the export payloads the store has merged, and of the
//! ones the node has handed out itself. An export whose digest is in the
//! ledger is not merged again.
//!
//! The ledger is a file of its own, not a field of the store, so that
//! recording an export never rewrites the store. The export routes are polled,
//! and a store rewritten on a read could drop a fix that a solve or a merge
//! saved in between.
//!
//! * It is written by rename, so a reader never sees half a file.
//! * [`ExportLedger::with`] runs one load, check and save at a time within
//!   this process. Two processes updating the same ledger at the same instant
//!   can still lose one of the two digests; that export would then count once
//!   more if it were delivered again.
//! * A missing or unreadable ledger is an empty one. Refusing to merge
//!   because the record of past merges is damaged would stop a fleet for the
//!   sake of a counter.
//! * It is bounded ([`MAX_SEEN_EXPORTS`]). A digest is added only when a new
//!   export is merged or the node's own export changes, so the bound covers
//!   that many distinct exports.
//!
//! A *different* export from a key-holder is new information and counts. How
//! often one holder's changing store may count is not something a digest can
//! bound; that needs the merge to know which peer is speaking.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::provenance::sha256_hex;

/// How many export digests a ledger remembers, oldest dropped first.
pub const MAX_SEEN_EXPORTS: usize = 1024;

/// One ledger transaction at a time in this process.
static TRANSACTION: Mutex<()> = Mutex::new(());

#[derive(Debug, Default, Serialize, Deserialize)]
struct LedgerFile {
    #[serde(default)]
    seen: Vec<String>,
}

/// The digests of the exports a store has merged or handed out.
#[derive(Debug)]
pub struct ExportLedger {
    path: PathBuf,
    seen: Vec<String>,
    changed: bool,
}

impl ExportLedger {
    /// Where the ledger for the store at `store` lives:
    /// `heal_memory.json` → `heal_memory.exports.json`.
    pub fn path_for(store: &Path) -> PathBuf {
        store.with_extension("exports.json")
    }

    /// Load the ledger of the store at `store`, run `f` on it, and return
    /// what `f` returns. Nothing else in this process reads or writes a
    /// ledger meanwhile, so "is it new? then merge and record it" is one
    /// step. `f` calls [`Self::save`] when it wants the ledger kept; it must
    /// not call `with` itself.
    pub fn with<T>(store: &Path, f: impl FnOnce(&mut ExportLedger) -> T) -> T {
        let _one_at_a_time = TRANSACTION.lock().unwrap_or_else(|e| e.into_inner());
        let path = Self::path_for(store);
        let seen = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<LedgerFile>(&text).ok())
            .map(|file| file.seen)
            .unwrap_or_default();
        f(&mut ExportLedger {
            path,
            seen,
            changed: false,
        })
    }

    /// Has this exact export payload been merged, or handed out, already?
    pub fn contains(&self, payload: &str) -> bool {
        let digest = sha256_hex(payload.as_bytes());
        self.seen.contains(&digest)
    }

    /// Record `payload`. Returns `false`, changing nothing, if it was
    /// recorded before. The record is in memory until [`Self::save`].
    pub fn remember(&mut self, payload: &str) -> bool {
        let digest = sha256_hex(payload.as_bytes());
        if self.seen.contains(&digest) {
            return false;
        }
        self.seen.push(digest);
        if self.seen.len() > MAX_SEEN_EXPORTS {
            let excess = self.seen.len() - MAX_SEEN_EXPORTS;
            self.seen.drain(..excess);
        }
        self.changed = true;
        true
    }

    /// Write the ledger, if [`Self::remember`] added anything, by writing a
    /// file beside it and renaming that over it.
    pub fn save(&mut self) -> io::Result<()> {
        if !self.changed {
            return Ok(());
        }
        if let Some(parent) = self.path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string(&LedgerFile {
            seen: self.seen.clone(),
        })
        .map_err(|e| io::Error::other(e.to_string()))?;
        let scratch = self
            .path
            .with_extension(format!("json.tmp-{}", std::process::id()));
        std::fs::write(&scratch, json)?;
        if let Err(e) = std::fs::rename(&scratch, &self.path) {
            let _ = std::fs::remove_file(&scratch);
            return Err(e);
        }
        self.changed = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_path(tag: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join(format!("{tag}.json"));
        (dir, store)
    }

    #[test]
    fn an_export_is_new_once_and_the_record_survives_a_reload() {
        let (_dir, store) = store_path("heal_memory");
        ExportLedger::with(&store, |ledger| {
            assert!(!ledger.contains("export one"));
            assert!(ledger.remember("export one"));
            assert!(!ledger.remember("export one"), "second time: nothing new");
            assert!(ledger.contains("export one"));
            ledger.save().unwrap();
        });
        assert_eq!(
            ExportLedger::path_for(&store),
            store.with_file_name("heal_memory.exports.json")
        );
        ExportLedger::with(&store, |ledger| {
            assert!(ledger.contains("export one"));
            assert!(!ledger.contains("export two"));
        });
        // The store itself is never created or touched by the ledger.
        assert!(!store.exists());
    }

    #[test]
    fn a_record_that_was_not_saved_is_not_kept() {
        let (_dir, store) = store_path("unsaved");
        ExportLedger::with(&store, |ledger| {
            assert!(ledger.remember("export"));
        });
        ExportLedger::with(&store, |ledger| assert!(!ledger.contains("export")));
        assert!(!ExportLedger::path_for(&store).exists());
    }

    #[test]
    fn the_ledger_is_bounded_and_drops_the_oldest_first() {
        let (_dir, store) = store_path("bounded");
        ExportLedger::with(&store, |ledger| {
            for i in 0..(MAX_SEEN_EXPORTS + 10) {
                assert!(ledger.remember(&format!("export {i}")));
            }
            ledger.save().unwrap();
        });
        ExportLedger::with(&store, |ledger| {
            assert_eq!(ledger.seen.len(), MAX_SEEN_EXPORTS);
            assert!(!ledger.contains("export 9"), "the oldest ten are gone");
            assert!(ledger.contains("export 10"));
            assert!(ledger.contains(&format!("export {}", MAX_SEEN_EXPORTS + 9)));
        });
    }

    #[test]
    fn a_damaged_ledger_reads_as_empty_and_is_replaced_whole() {
        let (dir, store) = store_path("damaged");
        std::fs::write(ExportLedger::path_for(&store), "{ not json").unwrap();
        ExportLedger::with(&store, |ledger| {
            assert!(!ledger.contains("export"));
            assert!(ledger.remember("export"));
            ledger.save().unwrap();
        });
        ExportLedger::with(&store, |ledger| assert!(ledger.contains("export")));
        // Written by rename: no scratch file is left beside it.
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["damaged.exports.json".to_string()]);
    }

    #[test]
    fn concurrent_transactions_lose_no_record() {
        let (_dir, store) = store_path("concurrent");
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let store = store.clone();
                std::thread::spawn(move || {
                    for i in 0..20 {
                        ExportLedger::with(&store, |ledger| {
                            assert!(ledger.remember(&format!("export {t}-{i}")));
                            ledger.save().unwrap();
                        });
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        ExportLedger::with(&store, |ledger| {
            assert_eq!(ledger.seen.len(), 160);
            for t in 0..8 {
                for i in 0..20 {
                    assert!(ledger.contains(&format!("export {t}-{i}")));
                }
            }
        });
    }
}
