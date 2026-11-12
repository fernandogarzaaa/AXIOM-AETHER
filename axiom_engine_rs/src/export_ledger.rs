//! Which fleet exports a store has already counted.
//!
//! A signed export ([`crate::provenance::SignedExport`]) says who produced it,
//! not how many times it was delivered. Anyone who has seen one valid export
//! can deliver it again, and a store that counted every delivery would let
//! them raise a patch candidate's confirmation count, or compound a heal
//! record's evidence, without a single new verification. The shortest version
//! needs no peer: read a node's own export and post it straight back.
//!
//! So each store has a ledger beside it, `<store file name>.exports.json`
//! (`heal_memory.json` → `heal_memory.json.exports.json`), holding the
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
//! * A merge records the export in the ledger *before* it saves the store,
//!   and takes the record back if the store cannot be saved
//!   ([`ExportLedger::save_around`]). The two files cannot be written as one,
//!   so one order has to be chosen, and this one fails the safe way: if the
//!   process dies between the two writes, the export reads as merged when it
//!   was not, and its content arrives with the peer's next changed export (an
//!   export is the peer's whole store). The other order would let a retry
//!   count the same export twice, which is what the ledger exists to stop.
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

    /// Take back a record made by [`Self::remember`]. Returns `false` if
    /// there was none.
    pub fn forget(&mut self, payload: &str) -> bool {
        let digest = sha256_hex(payload.as_bytes());
        let before = self.seen.len();
        self.seen.retain(|seen| seen != &digest);
        let forgotten = self.seen.len() != before;
        self.changed |= forgotten;
        forgotten
    }

    /// Persist a merge of `payload`, which the caller has already
    /// [`Self::remember`]ed: write the ledger, then run `save_store`. If the
    /// store cannot be saved, the record is taken back and the ledger written
    /// again, so that a retry merges the export instead of finding it "already
    /// merged".
    ///
    /// Nothing is saved, and `save_store` is not run, if the ledger cannot be
    /// written: a store that holds a merge the ledger does not know about
    /// would count the same export again on every retry.
    pub fn save_around(
        &mut self,
        payload: &str,
        save_store: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        self.save()?;
        let Err(store_error) = save_store() else {
            return Ok(());
        };
        self.forget(payload);
        match self.save() {
            Ok(()) => Err(store_error),
            Err(undo_error) => Err(io::Error::other(format!(
                "{store_error}; and the export could not be taken off the record \
                 ({undo_error}), so it will read as already merged until the peer's \
                 export changes"
            ))),
        }
    }

    /// Write the ledger, if [`Self::remember`] or [`Self::forget`] changed
    /// anything, by writing a file beside it and renaming that over it.
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

    #[test]
    fn a_merge_is_on_record_before_the_store_is_saved_and_off_it_if_that_fails() {
        let (_dir, store) = store_path("around");

        // The store save fails: the record is taken back, on disk too, so a
        // retry merges the export.
        ExportLedger::with(&store, |ledger| {
            ledger.remember("export");
            let on_disk_during_the_store_save = std::cell::Cell::new(false);
            let err = ledger
                .save_around("export", || {
                    let text = std::fs::read_to_string(ExportLedger::path_for(&store)).unwrap();
                    on_disk_during_the_store_save.set(text.contains(&sha256_hex(b"export")));
                    Err(io::Error::other("disk full"))
                })
                .unwrap_err();
            assert_eq!(err.to_string(), "disk full");
            assert!(
                on_disk_during_the_store_save.get(),
                "the ledger must be written before the store"
            );
        });
        ExportLedger::with(&store, |ledger| assert!(!ledger.contains("export")));

        // The store save succeeds: the record stays.
        ExportLedger::with(&store, |ledger| {
            ledger.remember("export");
            ledger.save_around("export", || Ok(())).unwrap();
        });
        ExportLedger::with(&store, |ledger| assert!(ledger.contains("export")));
    }

    #[test]
    fn the_store_is_not_saved_when_the_ledger_cannot_be_written() {
        let (_dir, store) = store_path("unwritable");
        // A directory sits where the ledger file goes: it cannot be replaced.
        std::fs::create_dir(ExportLedger::path_for(&store)).unwrap();
        ExportLedger::with(&store, |ledger| {
            ledger.remember("export");
            let store_saved = std::cell::Cell::new(false);
            let result = ledger.save_around("export", || {
                store_saved.set(true);
                Ok(())
            });
            assert!(result.is_err());
            assert!(
                !store_saved.get(),
                "a store holding a merge the ledger does not know would count it again"
            );
        });
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
