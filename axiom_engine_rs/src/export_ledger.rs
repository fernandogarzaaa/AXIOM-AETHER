//! Counting a fleet export once.
//!
//! A signed export ([`crate::provenance::SignedExport`]) says who produced it,
//! not how many times it was delivered. Anyone who has seen one valid export
//! can deliver it again, and a store that counted every delivery would let
//! them raise a patch candidate's confirmation count, or compound a heal
//! record's evidence, without a single new verification. The shortest version
//! needs no peer: read a node's own export and post it straight back.
//!
//! So a store does not merge an export it has merged before, or one the node
//! handed out itself. The two kinds are remembered in two places, because
//! they are written at two different moments:
//!
//! * **Merged exports** are remembered in the store file itself
//!   (`merged_exports`), written by the same save as the merge. One file,
//!   one write: either both the merge and its record are on disk or neither
//!   is, so a failed or interrupted save can neither count an export twice
//!   on retry nor mark as merged an export that was not.
//! * **Handed-out exports** are remembered in a ledger beside the store,
//!   `<store file name>.exports.json` ([`ExportLedger`]). The export routes
//!   are polled, and they must not rewrite the store on a read: a fix saved
//!   by a solve or a merge in between would be overwritten by the older
//!   snapshot. The ledger is written by rename, so a reader never sees half
//!   a file. A missing or unreadable ledger is an empty one.
//!
//! Both are SHA-256 digests of the export payload, bounded to the last
//! [`MAX_SEEN_EXPORTS`], oldest dropped first. A digest is added only when a
//! new export is merged or the node's own export changes, so the bound covers
//! that many distinct exports; a repeat does not move a digest back to the
//! front.
//!
//! [`ExportLedger::with`] runs one transaction at a time within this process,
//! and the merge routes do their whole load, check, merge and save inside
//! one, so the same export arriving twice at once is merged once. Another
//! process doing the same at the same instant is not excluded.
//!
//! A *different* export from a key-holder is new information and counts. How
//! often one holder's changing store may count is not something a digest can
//! bound; that needs the merge to know which peer is speaking.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::provenance::sha256_hex;

/// How many export digests are remembered, per store and per ledger, oldest
/// dropped first.
pub const MAX_SEEN_EXPORTS: usize = 1024;

/// One ledger transaction at a time in this process.
static TRANSACTION: Mutex<()> = Mutex::new(());

#[derive(Debug, Default, Serialize, Deserialize)]
struct LedgerFile {
    #[serde(default)]
    seen: Vec<String>,
}

/// The digests of the exports this node has handed out from one store.
#[derive(Debug)]
pub struct ExportLedger {
    path: PathBuf,
    seen: Vec<String>,
    changed: bool,
}

impl ExportLedger {
    /// Where the ledger for the store at `store` lives: the store's whole
    /// file name with `.exports.json` after it, so that two stores never
    /// share a ledger (`heal.json` and `heal.bak` would, if the extension
    /// were replaced). `heal_memory.json` → `heal_memory.json.exports.json`.
    pub fn path_for(store: &Path) -> PathBuf {
        with_suffix(store, ".exports.json")
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

    /// Has this exact export payload been handed out by this node?
    pub fn contains(&self, payload: &str) -> bool {
        has_digest(&self.seen, payload)
    }

    /// Record `payload` as handed out. Returns `false`, changing nothing, if
    /// it was recorded before. The record is in memory until [`Self::save`].
    pub fn remember(&mut self, payload: &str) -> bool {
        let added = remember_digest(&mut self.seen, payload);
        self.changed |= added;
        added
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
        let scratch = with_suffix(&self.path, &format!(".tmp-{}", std::process::id()));
        std::fs::write(&scratch, json)?;
        if let Err(e) = std::fs::rename(&scratch, &self.path) {
            let _ = std::fs::remove_file(&scratch);
            return Err(e);
        }
        self.changed = false;
        Ok(())
    }
}

/// Is the digest of `payload` in `digests`?
pub(crate) fn has_digest(digests: &[String], payload: &str) -> bool {
    let digest = sha256_hex(payload.as_bytes());
    digests.contains(&digest)
}

/// Add the digest of `payload` to `digests`, dropping the oldest beyond
/// [`MAX_SEEN_EXPORTS`]. Returns `false`, changing nothing, if it was there.
pub(crate) fn remember_digest(digests: &mut Vec<String>, payload: &str) -> bool {
    let digest = sha256_hex(payload.as_bytes());
    if digests.contains(&digest) {
        return false;
    }
    digests.push(digest);
    if digests.len() > MAX_SEEN_EXPORTS {
        let excess = digests.len() - MAX_SEEN_EXPORTS;
        digests.drain(..excess);
    }
    true
}

/// `path` with `suffix` appended to its last component.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
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
            store.with_file_name("heal_memory.json.exports.json")
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
        assert_eq!(names, vec!["damaged.json.exports.json".to_string()]);
    }

    #[test]
    fn stores_that_share_a_stem_do_not_share_a_ledger() {
        let dir = tempfile::tempdir().unwrap();
        let stores = ["heal.json", "heal.bak", "heal"].map(|name| dir.path().join(name));
        ExportLedger::with(&stores[0], |ledger| {
            ledger.remember("export");
            ledger.save().unwrap();
        });
        for other in &stores[1..] {
            ExportLedger::with(other, |ledger| {
                assert!(
                    !ledger.contains("export"),
                    "{} read the ledger of heal.json",
                    other.display()
                );
            });
        }
        let ledgers: std::collections::BTreeSet<PathBuf> =
            stores.iter().map(|s| ExportLedger::path_for(s)).collect();
        assert_eq!(ledgers.len(), 3, "{ledgers:?}");
    }
}
