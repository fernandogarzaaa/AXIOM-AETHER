//! On-disk journal for source files the repair loop rewrites.
//!
//! `axiom solve` tries a candidate fix by writing it over the source file and
//! running the verify command. The original used to live only in memory, so a
//! solve that died with a candidate on disk (Ctrl-C, `kill`, a crash, power
//! loss) left the user's file holding an unverified edit.
//!
//! The journal closes that window. Before the first write to a file,
//! [`protect`] saves the original bytes to disk and syncs them. The original
//! is put back:
//!
//! * when a candidate is rejected, and when the [`Protected`] guard is dropped
//!   without [`Protected::commit`] (a failed repair, an early return, a panic);
//! * on SIGINT / SIGTERM / SIGHUP, by the thread [`restore_on_interrupt`]
//!   starts, before the process exits;
//! * at the start of the next solve, by [`recover`], when the previous one was
//!   killed outright and none of the above could run.
//!
//! A journal entry is three files in the journal directory, named after a hash
//! of the source path plus this process's id:
//!
//! * `<id>.lock` — held under an exclusive lock for as long as the owning
//!   process lives. [`recover`] treats an entry as abandoned only if it can
//!   take that lock, so it never touches the entry of a solve that is still
//!   running, in this process or another.
//! * `<id>.orig` — the original bytes.
//! * `<id>.json` — the source path, the hash of the original, and the hash of
//!   the last candidate written over the file.
//!
//! The last one is what lets [`recover`] tell "still holding our candidate"
//! from "somebody edited the file since". Only the first is restored. In the
//! second case the file is left alone and the saved original is kept, so
//! recovery never overwrites work done after the crash.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// What an entry's `.json` file records.
#[derive(Debug, Serialize, Deserialize)]
struct Meta {
    /// Canonical path of the protected source file.
    path: String,
    /// SHA-256 of the original bytes (the `.orig` file).
    original_sha256: String,
    /// SHA-256 of the last content written over the file through [`write`],
    /// or `None` if nothing has been written yet.
    written_sha256: Option<String>,
}

/// One protected file of this process.
struct Entry {
    target: PathBuf,
    original: Vec<u8>,
    /// `<journal dir>/<id>`; the entry's files are this plus an extension.
    stem: PathBuf,
    /// Holds the exclusive lock on `<id>.lock` until the entry is dropped.
    lock: File,
}

impl Entry {
    fn file(&self, ext: &str) -> PathBuf {
        self.stem.with_extension(ext)
    }

    fn meta(&self, written: Option<&[u8]>) -> Meta {
        Meta {
            path: self.target.to_string_lossy().into_owned(),
            original_sha256: sha256_hex(&self.original),
            written_sha256: written.map(sha256_hex),
        }
    }

    /// Put the original back if the file holds anything else.
    fn restore(&self) -> io::Result<()> {
        if fs::read(&self.target).ok().as_deref() == Some(self.original.as_slice()) {
            return Ok(());
        }
        fs::write(&self.target, &self.original)
    }

    fn remove_files(&self) {
        for ext in ["orig", "json", "lock"] {
            let _ = fs::remove_file(self.file(ext));
        }
    }
}

/// Entries of this process. The mutex also serializes every journaled write
/// with [`restore_all`], so an interrupt cannot interleave with a candidate
/// being written.
static ACTIVE: Mutex<Vec<Arc<Entry>>> = Mutex::new(Vec::new());
/// Set by [`restore_all`]: the process is on its way out and nothing may be
/// written over a source file any more.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static SEQ: AtomicU64 = AtomicU64::new(0);

fn active() -> MutexGuard<'static, Vec<Arc<Entry>>> {
    ACTIVE.lock().unwrap_or_else(|e| e.into_inner())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn interrupted_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::Interrupted,
        "interrupted: source files were restored and are no longer written to",
    )
}

/// Where journal entries live: `AXIOM_JOURNAL_DIR`, else `~/.axiom/journal`,
/// else a directory under the system temp dir.
pub fn default_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("AXIOM_JOURNAL_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    match dirs::home_dir() {
        Some(home) => home.join(".axiom").join("journal"),
        None => std::env::temp_dir().join("axiom-journal"),
    }
}

/// Write `bytes` to `path` and sync them to disk before returning.
fn write_durable(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Replace an entry's `.json` atomically (write a sibling, sync, rename).
fn write_meta(stem: &Path, meta: &Meta) -> io::Result<()> {
    let json = serde_json::to_vec(meta).map_err(io::Error::other)?;
    let tmp = stem.with_extension("json.tmp");
    write_durable(&tmp, &json)?;
    fs::rename(&tmp, stem.with_extension("json"))
}

/// Sync a directory so a freshly created entry survives power loss. Best
/// effort: not every platform can open a directory for syncing.
fn sync_dir(dir: &Path) {
    if let Ok(handle) = File::open(dir) {
        let _ = handle.sync_all();
    }
}

/// A source file whose original is saved in the journal. Dropping the guard
/// restores the original unless [`Protected::commit`] was called.
pub struct Protected {
    entry: Arc<Entry>,
    finished: bool,
}

impl Protected {
    /// The original bytes, as saved.
    pub fn original(&self) -> &[u8] {
        &self.entry.original
    }

    /// Put the original back now (a no-op if the file already holds it). The
    /// file stays protected.
    pub fn restore(&self) -> io::Result<()> {
        let _serial = active();
        self.entry.restore()
    }

    /// The file's current content is the one to keep: drop the journal entry
    /// without restoring.
    pub fn commit(mut self) {
        self.finish(false);
    }

    fn finish(&mut self, restore: bool) {
        if self.finished {
            return;
        }
        self.finished = true;
        let mut entries = active();
        // Absent means `restore_all` already restored and removed it.
        let Some(at) = entries.iter().position(|e| Arc::ptr_eq(e, &self.entry)) else {
            return;
        };
        entries.remove(at);
        if restore {
            if let Err(e) = self.entry.restore() {
                // Keep the saved original: it is the only good copy left.
                eprintln!(
                    "[axiom-journal] could not restore {} ({e}); its original is saved at {}",
                    self.entry.target.display(),
                    self.entry.file("orig").display()
                );
                let _ = fs::remove_file(self.entry.file("json"));
                let _ = fs::remove_file(self.entry.file("lock"));
                return;
            }
        }
        self.entry.remove_files();
    }

    /// Test hook: forget the entry as a killed process would, leaving its
    /// files on disk and releasing the lock.
    #[cfg(test)]
    fn abandon(mut self) {
        self.finished = true;
        let mut entries = active();
        if let Some(at) = entries.iter().position(|e| Arc::ptr_eq(e, &self.entry)) {
            entries.remove(at);
        }
    }
}

impl Drop for Protected {
    fn drop(&mut self) {
        self.finish(true);
    }
}

/// Save `path`'s current content in the default journal directory before it
/// is written to. See [`protect_in`].
pub fn protect(path: &Path) -> io::Result<Protected> {
    protect_in(&default_dir(), path)
}

/// Save `path`'s current content under `dir`, synced to disk, and return the
/// guard that restores it. Fails, without touching `path`, if the original
/// cannot be journaled; callers must then not write to the file.
pub fn protect_in(dir: &Path, path: &Path) -> io::Result<Protected> {
    if INTERRUPTED.load(Ordering::SeqCst) {
        return Err(interrupted_error());
    }
    let target = fs::canonicalize(path)?;
    let original = fs::read(&target)?;
    fs::create_dir_all(dir)?;
    let id = format!(
        "{}-{}-{}",
        &sha256_hex(target.to_string_lossy().as_bytes())[..16],
        std::process::id(),
        SEQ.fetch_add(1, Ordering::SeqCst)
    );
    let stem = dir.join(id);
    // The lock comes first: an entry without one would look abandoned to a
    // concurrent `recover`.
    let lock = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(stem.with_extension("lock"))?;
    let entry = Entry {
        target,
        original,
        stem,
        lock,
    };
    let journaled = (|| {
        match entry.lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(io::Error::other(
                    "journal entry is locked by another process",
                ))
            }
            Err(TryLockError::Error(e)) => return Err(e),
        }
        write_durable(&entry.file("orig"), &entry.original)?;
        write_meta(&entry.stem, &entry.meta(None))
    })();
    if let Err(e) = journaled {
        entry.remove_files();
        return Err(e);
    }
    sync_dir(dir);
    let entry = Arc::new(entry);
    let mut entries = active();
    // Re-checked under the lock `restore_all` takes: an entry registered
    // after it ran would never be restored.
    if INTERRUPTED.load(Ordering::SeqCst) {
        entry.remove_files();
        return Err(interrupted_error());
    }
    entries.push(Arc::clone(&entry));
    Ok(Protected {
        entry,
        finished: false,
    })
}

/// Write `contents` over the source file `path`. If the file is protected,
/// the journal first records what is about to be written, so that a later
/// [`recover`] can recognise the file as still holding this candidate. An
/// unprotected file is written as by [`std::fs::write`].
///
/// Refused once [`restore_all`] has run.
pub fn write(path: &Path, contents: &[u8]) -> io::Result<()> {
    let entries = active();
    if INTERRUPTED.load(Ordering::SeqCst) {
        return Err(interrupted_error());
    }
    if let Ok(target) = fs::canonicalize(path) {
        for entry in entries.iter().filter(|e| e.target == target) {
            write_meta(&entry.stem, &entry.meta(Some(contents)))?;
        }
    }
    fs::write(path, contents)
}

/// Restore every file this process has protected and drop the entries.
/// After this, [`write`] and [`protect`] refuse. For the interrupt path: the
/// process is about to exit. Returns how many files were restored.
pub fn restore_all() -> usize {
    let mut entries = active();
    INTERRUPTED.store(true, Ordering::SeqCst);
    let mut restored = 0;
    for entry in entries.drain(..) {
        match entry.restore() {
            Ok(()) => {
                restored += 1;
                entry.remove_files();
            }
            Err(e) => eprintln!(
                "[axiom-journal] could not restore {} ({e}); its original is saved at {}",
                entry.target.display(),
                entry.file("orig").display()
            ),
        }
    }
    restored
}

/// What [`recover`] did with an entry left behind by a process that died.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryOutcome {
    /// The file still held the dead process's last candidate; the original
    /// was put back.
    Restored,
    /// The file already held the original; nothing to do.
    AlreadyOriginal,
    /// The file holds neither the original nor the last candidate: it was
    /// edited after the crash. It was left alone, and the original is kept at
    /// `saved_original`.
    LeftChanged { saved_original: PathBuf },
    /// The file no longer exists; nothing was recreated.
    TargetMissing,
}

/// One entry handled by [`recover`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovery {
    pub target: PathBuf,
    pub outcome: RecoveryOutcome,
}

/// Handle entries abandoned in the default journal directory. See
/// [`recover_in`].
pub fn recover() -> Vec<Recovery> {
    recover_in(&default_dir())
}

/// Find entries under `dir` whose owning process is gone and restore their
/// files where that is safe (see [`RecoveryOutcome`]). Entries whose owner is
/// alive are skipped.
pub fn recover_in(dir: &Path) -> Vec<Recovery> {
    let Ok(listing) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut done = Vec::new();
    for lock_path in listing
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "lock"))
    {
        let Ok(lock) = OpenOptions::new().write(true).open(&lock_path) else {
            continue;
        };
        // Locked: the owner is alive (possibly this very process).
        if lock.try_lock().is_err() {
            continue;
        }
        let stem = lock_path.with_extension("");
        let orig_path = stem.with_extension("orig");
        let json_path = stem.with_extension("json");
        let meta = fs::read(&json_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Meta>(&bytes).ok());
        let original = fs::read(&orig_path).ok();
        let mut keep_original = false;
        // An entry cut short before both files were written protected a file
        // that had not been written to yet: nothing to restore.
        if let (Some(meta), Some(original)) = (meta, original) {
            if sha256_hex(&original) == meta.original_sha256 {
                let target = PathBuf::from(&meta.path);
                let outcome = match fs::read(&target) {
                    Err(_) => RecoveryOutcome::TargetMissing,
                    Ok(current) if current == original => RecoveryOutcome::AlreadyOriginal,
                    Ok(current) if Some(sha256_hex(&current)) == meta.written_sha256 => {
                        match fs::write(&target, &original) {
                            Ok(()) => RecoveryOutcome::Restored,
                            Err(_) => {
                                keep_original = true;
                                RecoveryOutcome::LeftChanged {
                                    saved_original: orig_path.clone(),
                                }
                            }
                        }
                    }
                    Ok(_) => {
                        keep_original = true;
                        RecoveryOutcome::LeftChanged {
                            saved_original: orig_path.clone(),
                        }
                    }
                };
                done.push(Recovery { target, outcome });
            }
        }
        if !keep_original {
            let _ = fs::remove_file(&orig_path);
        }
        let _ = fs::remove_file(&json_path);
        let _ = fs::remove_file(stem.with_extension("json.tmp"));
        drop(lock);
        let _ = fs::remove_file(&lock_path);
    }
    done
}

/// Start a thread that, on SIGINT, SIGTERM or SIGHUP (Ctrl-C on Windows),
/// restores every protected source file and exits with the conventional
/// status (130, 143, 129). Returns once the signals are being watched, so a
/// solve started right after cannot be interrupted unprotected. Idempotent.
///
/// For CLI entry points only: a library caller keeps its own signal handling.
pub fn restore_on_interrupt() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
        let spawned = std::thread::Builder::new()
            .name("axiom-interrupt".into())
            .spawn(move || {
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                let status = runtime.block_on(wait_for_interrupt(ready_tx));
                let restored = restore_all();
                eprintln!(
                    "[axiom-solve] interrupted: {restored} source file(s) restored to their original content"
                );
                std::process::exit(status);
            });
        if spawned.is_ok() {
            let _ = ready_rx.recv_timeout(std::time::Duration::from_secs(5));
        }
    });
}

/// Resolve to the exit status for the first termination signal received.
/// Never resolves if the signals cannot be watched.
async fn wait_for_interrupt(ready: std::sync::mpsc::Sender<()>) -> i32 {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let (Ok(mut interrupt), Ok(mut terminate), Ok(mut hangup)) = (
            signal(SignalKind::interrupt()),
            signal(SignalKind::terminate()),
            signal(SignalKind::hangup()),
        ) else {
            let _ = ready.send(());
            return std::future::pending().await;
        };
        let _ = ready.send(());
        tokio::select! {
            _ = interrupt.recv() => 130,
            _ = terminate.recv() => 143,
            _ = hangup.recv() => 129,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = ready.send(());
        match tokio::signal::ctrl_c().await {
            Ok(()) => 130,
            Err(_) => std::future::pending().await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A journal directory and a source file under one temp dir.
    fn fixture(content: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let journal = tmp.path().join("journal");
        let source = tmp.path().join("lib.sh");
        fs::write(&source, content).unwrap();
        (tmp, journal, source)
    }

    fn entry_files(journal: &Path) -> usize {
        fs::read_dir(journal).map(|d| d.count()).unwrap_or(0)
    }

    #[test]
    fn the_original_is_on_disk_before_the_first_write() {
        let (_tmp, journal, source) = fixture("original\n");
        let guard = protect_in(&journal, &source).unwrap();
        // Nothing has been written to the source yet, and the original is
        // already in the journal directory.
        let saved: Vec<PathBuf> = fs::read_dir(&journal)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "orig"))
            .collect();
        assert_eq!(saved.len(), 1);
        assert_eq!(fs::read(&saved[0]).unwrap(), b"original\n");
        assert_eq!(guard.original(), b"original\n");
        drop(guard);
        assert_eq!(entry_files(&journal), 0, "a finished entry leaves nothing");
    }

    #[test]
    fn dropping_the_guard_restores_and_commit_keeps() {
        let (_tmp, journal, source) = fixture("original\n");

        // A repair that fails: the candidate is undone when the guard goes.
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"candidate\n").unwrap();
        assert_eq!(fs::read(&source).unwrap(), b"candidate\n");
        drop(guard);
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
        assert_eq!(entry_files(&journal), 0);

        // A repair that verified: the candidate stays.
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"fixed\n").unwrap();
        guard.commit();
        assert_eq!(fs::read(&source).unwrap(), b"fixed\n");
        assert_eq!(entry_files(&journal), 0);
    }

    #[test]
    fn a_panic_while_a_candidate_is_on_disk_restores_the_original() {
        let (_tmp, journal, source) = fixture("original\n");
        let result = std::panic::catch_unwind(|| {
            let _guard = protect_in(&journal, &source).unwrap();
            write(&source, b"candidate\n").unwrap();
            panic!("the repair loop blew up");
        });
        assert!(result.is_err());
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
    }

    #[test]
    fn restore_puts_the_original_back_and_keeps_protecting() {
        let (_tmp, journal, source) = fixture("original\n");
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"first candidate\n").unwrap();
        guard.restore().unwrap();
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
        write(&source, b"second candidate\n").unwrap();
        drop(guard);
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
    }

    #[test]
    fn an_unprotected_file_is_written_plainly() {
        let (_tmp, journal, source) = fixture("original\n");
        write(&source, b"changed\n").unwrap();
        assert_eq!(fs::read(&source).unwrap(), b"changed\n");
        assert_eq!(entry_files(&journal), 0);
    }

    #[test]
    fn protect_fails_without_touching_a_file_it_cannot_journal() {
        let (tmp, _journal, source) = fixture("original\n");
        // The journal "directory" is a file: nothing can be saved in it.
        let blocked = tmp.path().join("not-a-directory");
        fs::write(&blocked, "x").unwrap();
        assert!(protect_in(&blocked, &source).is_err());
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
        // And a missing source has nothing to protect.
        assert!(protect_in(&tmp.path().join("journal"), &tmp.path().join("nope")).is_err());
    }

    #[test]
    fn recovery_restores_a_file_left_holding_a_dead_process_candidate() {
        let (_tmp, journal, source) = fixture("original\n");
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"unverified candidate\n").unwrap();
        // The process dies here: no restore, the entry stays on disk.
        guard.abandon();
        assert_eq!(fs::read(&source).unwrap(), b"unverified candidate\n");

        let recovered = recover_in(&journal);
        assert_eq!(
            recovered,
            vec![Recovery {
                target: fs::canonicalize(&source).unwrap(),
                outcome: RecoveryOutcome::Restored,
            }]
        );
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
        assert_eq!(entry_files(&journal), 0);
        // Nothing left to recover a second time.
        assert!(recover_in(&journal).is_empty());
    }

    #[test]
    fn recovery_leaves_a_file_edited_after_the_crash_and_keeps_the_original() {
        let (_tmp, journal, source) = fixture("original\n");
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"unverified candidate\n").unwrap();
        guard.abandon();
        // The user fixes the file by hand before the next solve.
        fs::write(&source, "hand-written fix\n").unwrap();

        let recovered = recover_in(&journal);
        assert_eq!(recovered.len(), 1);
        let RecoveryOutcome::LeftChanged { saved_original } = &recovered[0].outcome else {
            panic!("expected LeftChanged, got {:?}", recovered[0].outcome);
        };
        assert_eq!(
            fs::read(&source).unwrap(),
            b"hand-written fix\n",
            "work done after the crash is not overwritten"
        );
        assert_eq!(fs::read(saved_original).unwrap(), b"original\n");
    }

    #[test]
    fn recovery_is_quiet_about_files_that_need_nothing() {
        let (_tmp, journal, source) = fixture("original\n");
        // Died after a rollback: the file already holds the original.
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"candidate\n").unwrap();
        write(&source, b"original\n").unwrap();
        guard.abandon();
        assert_eq!(
            recover_in(&journal)[0].outcome,
            RecoveryOutcome::AlreadyOriginal
        );

        // Died, and the file was deleted since: it is not recreated.
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"candidate\n").unwrap();
        guard.abandon();
        fs::remove_file(&source).unwrap();
        assert_eq!(
            recover_in(&journal)[0].outcome,
            RecoveryOutcome::TargetMissing
        );
        assert!(!source.exists());
        assert_eq!(entry_files(&journal), 0);
    }

    #[test]
    fn recovery_skips_an_entry_whose_owner_is_alive() {
        let (_tmp, journal, source) = fixture("original\n");
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"candidate being verified\n").unwrap();
        // Another solve starts while this one is mid-verify.
        assert!(recover_in(&journal).is_empty());
        assert_eq!(fs::read(&source).unwrap(), b"candidate being verified\n");
        drop(guard);
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
    }
}
