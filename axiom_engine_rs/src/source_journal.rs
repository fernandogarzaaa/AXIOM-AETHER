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
//! A journal entry is a few owner-only files in the journal directory, named
//! after a hash of the source path plus this process's id:
//!
//! * `<id>.lock` — held under an exclusive lock for as long as the owning
//!   process lives. [`recover`] treats an entry as abandoned only if it can
//!   take that lock, so it never touches the entry of a solve that is still
//!   running, in this process or another.
//! * `<id>.orig` — the original bytes.
//! * `<id>.json` — the source path and the hash of the original.
//! * `<id>.cand` — the last content written over the file through [`write`].
//!
//! The last one is what lets [`recover`] tell "still holding what we wrote"
//! from "somebody edited the file since". A file is restored only if it holds
//! the recorded content, or the beginning of it or of the original (a write
//! that was cut short). Anything else was edited after the crash: the file is
//! left alone and the saved original is kept. So recovery does not overwrite
//! newer work, with one exception it cannot avoid: an edit that leaves the
//! file empty, or as exactly the beginning of the candidate or of the
//! original, looks the same as a cut-short write and is restored.
//!
//! Two solves must not repair the same file at once: the second would journal
//! the first one's candidate as "the original". While a file is protected its
//! solve holds an exclusive lock on it, and so does a recovery while it puts
//! a file back. On Unix the lock is an advisory lock on the source file
//! itself, so it is shared by every solve on the machine whatever journal
//! directory each one uses, and the kernel drops it if the holder dies.
//! Elsewhere, where a lock on the file would stop the verify command reading
//! it, the lock is on a `<hash of the source path>.target` file in one
//! per-user directory under the system temp dir, which does not depend on
//! the journal directory either. It does depend on the temp dir: there, two
//! solves started with different `TMP`/`TEMP` values do not exclude each
//! other. That is a known limit of the non-Unix fallback.
//!
//! Entries are only ever recovered from the journal directory in use. A file
//! left by a solve that used a different directory is not put right until a
//! solve runs with that directory again.

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
}

/// One protected file of this process.
struct Entry {
    target: PathBuf,
    original: Vec<u8>,
    /// `<journal dir>/<id>`; the entry's files are this plus an extension.
    stem: PathBuf,
    /// Holds the exclusive lock on `<id>.lock` until the entry is dropped.
    lock: File,
    /// The exclusive hold on the source file, released when the entry drops.
    target_lock: TargetLock,
}

impl Entry {
    fn file(&self, ext: &str) -> PathBuf {
        self.stem.with_extension(ext)
    }

    /// Put the original back if the file holds anything else.
    fn restore(&self) -> io::Result<()> {
        if fs::read(&self.target).ok().as_deref() == Some(self.original.as_slice()) {
            return Ok(());
        }
        fs::write(&self.target, &self.original)
    }

    /// Remove the entry's files and, where the target lock is a guard file,
    /// that file too (while its lock is still held, so nobody waits on it).
    fn remove_files(&self) {
        for ext in ENTRY_EXTENSIONS {
            let _ = fs::remove_file(self.file(ext));
        }
        self.target_lock.remove_guard_file();
    }
}

/// Extensions of the files one entry may own. `lock` goes last: an entry
/// without its lock file must have nothing else left to find.
const ENTRY_EXTENSIONS: [&str; 6] = ["orig", "json", "json.tmp", "cand", "cand.tmp", "lock"];

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

/// The part of a journal file name that identifies the source file.
fn target_hash(target: &Path) -> String {
    sha256_hex(target.to_string_lossy().as_bytes())[..16].to_string()
}

fn interrupted_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::Interrupted,
        "interrupted: source files were restored and are no longer written to",
    )
}

/// Where journal entries live: `AXIOM_JOURNAL_DIR` if set; otherwise a
/// `journal` directory beside the heal memory (so `AXIOM_HEAL_MEMORY` moves
/// it along with the rest of the solve state); otherwise `~/.axiom/journal`;
/// otherwise a directory under the system temp dir.
pub fn default_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("AXIOM_JOURNAL_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    let beside_heal_memory = crate::heal_memory::HealMemory::default_path()
        .and_then(|path| path.parent().map(|dir| dir.join("journal")));
    if let Some(dir) = beside_heal_memory {
        return dir;
    }
    match dirs::home_dir() {
        Some(home) => home.join(".axiom").join("journal"),
        None => std::env::temp_dir().join("axiom-journal"),
    }
}

/// How far a journal directory is open to users other than its owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Exposure {
    /// Owned by this user, closed to everyone else.
    Private,
    /// Owned by this user; others could read it but not put files in it.
    Readable,
    /// Owned by this user; others could create or replace files in it.
    Writable,
}

/// Who owns `dir` and how open it is. `Err` if another user owns it. Always
/// `Private` off Unix, where there are no mode bits to check.
fn exposure(dir: &Path) -> io::Result<Exposure> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = fs::metadata(dir)?;
        // SAFETY: geteuid(2) has no preconditions and cannot fail.
        if meta.uid() != unsafe { libc::geteuid() } {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "journal directory {} belongs to another user; point AXIOM_JOURNAL_DIR at \
                     a directory of your own",
                    dir.display()
                ),
            ));
        }
        Ok(match meta.mode() {
            mode if mode & 0o022 != 0 => Exposure::Writable,
            mode if mode & 0o077 != 0 => Exposure::Readable,
            _ => Exposure::Private,
        })
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        Ok(Exposure::Private)
    }
}

/// Create a directory that only its owner can enter.
fn create_private_dir(dir: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir)
}

/// Make `dir` owner-only.
fn close_to_others(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// What is appended to the name of each file of an entry that recovery
/// refused because others could have changed it. With it, the files are no
/// longer an entry.
const ENTRY_SET_ASIDE: &str = ".untrusted";

/// Could somebody other than this user have changed the journal file at
/// `path`? True for one that belongs to another user, that group or others
/// may write to, or that is not a regular file (a link could point anywhere).
/// A file that does not exist was not changed. Always false off Unix, where
/// there are no mode bits to check.
///
/// The journal creates its files owner-only, so this is never true for a
/// file it wrote. A directory that others could only read is closed and kept
/// ([`ensure_private_dir`]); a file in it that was open to writing is the one
/// thing there they could have altered.
fn others_could_have_changed(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match fs::symlink_metadata(path) {
            Err(e) if e.kind() == io::ErrorKind::NotFound => false,
            Err(_) => true,
            // SAFETY: geteuid(2) has no preconditions and cannot fail.
            Ok(meta) => {
                !meta.is_file()
                    || meta.uid() != unsafe { libc::geteuid() }
                    || meta.mode() & 0o022 != 0
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        false
    }
}

/// Make sure `dir` exists and belongs to this user alone. It holds copies of
/// source files, and [`recover`] writes what it finds there back over source
/// files, so nobody else may read it or put files in it.
///
/// * A missing directory is created owner-only.
/// * One owned by somebody else is refused.
/// * One this user owns that others could read is closed to them.
/// * One this user owns that others could write to is never used again.
///   Anybody could have planted an entry there, and recovery would write its
///   "original" over the file it names. The whole directory is renamed aside
///   ([`set_aside`]) and a fresh one is created in its place.
///
/// `dir` must be a real path ([`real_dir`]): a link to a directory is not
/// what gets checked or moved.
fn ensure_private_dir(dir: &Path) -> io::Result<()> {
    if was_set_aside(dir) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "{} was set aside as untrusted and is not used as a journal",
                dir.display()
            ),
        ));
    }
    if dir.is_dir() {
        match exposure(dir)? {
            Exposure::Private => return Ok(()),
            Exposure::Readable => return close_to_others(dir),
            Exposure::Writable => set_aside(dir)?,
        }
    }
    if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    match create_private_dir(dir) {
        Err(e) if e.kind() != io::ErrorKind::AlreadyExists => return Err(e),
        _ => {}
    }
    if !dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            format!("journal path {} is not a directory", dir.display()),
        ));
    }
    // Somebody may have created it between the two steps above: only a
    // directory that is ours and closed is used.
    match exposure(dir)? {
        Exposure::Private => Ok(()),
        _ => Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "journal directory {} can be reached by other users; point AXIOM_JOURNAL_DIR \
                 at a directory of your own",
                dir.display()
            ),
        )),
    }
}

/// What a directory moved aside by [`set_aside`] has in its name. Such a
/// directory is never used as a journal, even though it is closed to others
/// afterwards: what is in it was put there while it was open.
const SET_ASIDE_MARK: &str = ".untrusted-";

fn was_set_aside(dir: &Path) -> bool {
    dir.file_name()
        .is_some_and(|name| name.to_string_lossy().contains(SET_ASIDE_MARK))
}

/// The directory `dir` names once symbolic links are followed, so that it is
/// the directory itself that is checked, moved aside or created, never a link
/// to it. A path that does not exist yet is returned as given.
fn real_dir(dir: &Path) -> PathBuf {
    fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf())
}

/// Rename `dir`, with everything in it, to a sibling name. One atomic step:
/// afterwards nothing that was in the directory can be found at its path, so
/// stopping right after cannot leave an untrusted entry where recovery looks.
/// Nothing is deleted: a genuine backup in there can still be restored by
/// hand.
fn set_aside(dir: &Path) -> io::Result<()> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "journal".to_string());
    for n in 0..64 {
        let aside = dir.with_file_name(format!(
            "{name}{SET_ASIDE_MARK}{stamp}-{}-{n}",
            std::process::id()
        ));
        if aside.exists() {
            continue;
        }
        fs::rename(dir, &aside)?;
        // Best effort: it is ours, and nobody else needs to get in any more.
        let _ = close_to_others(&aside);
        eprintln!(
            "[axiom-journal] {} could be written to by other users, so nothing in it can be \
             trusted. It was moved to {} and a new journal directory is used; nothing from \
             the old one is restored automatically.",
            dir.display(),
            aside.display()
        );
        return Ok(());
    }
    Err(io::Error::other(format!(
        "could not set aside the untrusted journal directory {}",
        dir.display()
    )))
}

/// Options for creating a journal file: owner-only on Unix, whatever the
/// umask. A journal file can hold the full content of a private source file.
fn private_file() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

/// Write `bytes` to `path` and sync them to disk before returning.
fn write_durable(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = private_file().create(true).truncate(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Replace `<stem>.<ext>` atomically: write a sibling, sync it, rename it into
/// place. The file is always either its old content or the complete new one.
fn replace_durable(stem: &Path, ext: &str, bytes: &[u8]) -> io::Result<()> {
    let tmp = stem.with_extension(format!("{ext}.tmp"));
    write_durable(&tmp, bytes)?;
    fs::rename(&tmp, stem.with_extension(ext))
}

/// Sync a directory so a freshly created entry survives power loss. Best
/// effort: not every platform can open a directory for syncing.
fn sync_dir(dir: &Path) {
    if let Ok(handle) = File::open(dir) {
        let _ = handle.sync_all();
    }
}

/// The exclusive hold one solve (or one recovery) has on a source file.
struct TargetLock {
    /// The locked handle: the source file itself on Unix, a guard file in
    /// the journal directory elsewhere. Dropping it releases the lock.
    _handle: File,
    /// The guard file to delete on release, where one is used.
    guard_file: Option<PathBuf>,
}

impl TargetLock {
    fn remove_guard_file(&self) {
        if let Some(path) = &self.guard_file {
            let _ = fs::remove_file(path);
        }
    }
}

fn target_busy(target: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::WouldBlock,
        format!(
            "{} is being repaired by another axiom solve",
            target.display()
        ),
    )
}

/// Take the exclusive lock on `target`, or fail with `WouldBlock` if another
/// solve or recovery (in this process or any other) holds it. The lock does
/// not depend on the journal directory, so solves configured with different
/// ones still exclude each other.
fn lock_target(target: &Path) -> io::Result<TargetLock> {
    if cfg!(unix) {
        // An advisory lock on the file itself. Every solve that touches the
        // file meets on it; reading and writing the file are unaffected.
        let handle = File::open(target)?;
        return match handle.try_lock() {
            Ok(()) => Ok(TargetLock {
                _handle: handle,
                guard_file: None,
            }),
            Err(TryLockError::WouldBlock) => Err(target_busy(target)),
            Err(TryLockError::Error(e)) => Err(e),
        };
    }
    // A lock on the file itself would be mandatory here and stop the verify
    // command reading it, so lock a guard file named after the file, in one
    // place all of this user's solves share.
    lock_guard_file(&std::env::temp_dir().join("axiom-solve-locks"), target)
}

/// Lock `<guard_dir>/<hash of target>.target`.
///
/// The guard file is deleted when its holder is done. A deleted file can
/// still be open elsewhere, so after locking, check that the locked handle is
/// the file the path names now; if it is not, somebody removed it in between
/// and the lock is on a file nobody else will look at. Try again.
fn lock_guard_file(guard_dir: &Path, target: &Path) -> io::Result<TargetLock> {
    fs::create_dir_all(guard_dir)?;
    let path = guard_dir.join(format!("{}.target", target_hash(target)));
    for _ in 0..8 {
        let guard = private_file().create(true).truncate(false).open(&path)?;
        match guard.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(target_busy(target)),
            Err(TryLockError::Error(e)) => return Err(e),
        }
        if same_file(&guard, &path) {
            return Ok(TargetLock {
                _handle: guard,
                guard_file: Some(path),
            });
        }
    }
    Err(io::Error::other(format!(
        "could not take the repair lock for {}",
        target.display()
    )))
}

/// Does the open `handle` refer to the file `path` names right now?
fn same_file(handle: &File, path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (handle.metadata(), fs::metadata(path)) {
            (Ok(held), Ok(named)) => held.dev() == named.dev() && held.ino() == named.ino(),
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        // A locked, open file cannot be deleted from under its handle here.
        let _ = handle;
        path.exists()
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
                // Keep the whole entry: the saved original is the only good
                // copy left, and the next `recover` will try again.
                eprintln!(
                    "[axiom-journal] could not restore {} ({e}); its original is saved at {}",
                    self.entry.target.display(),
                    self.entry.file("orig").display()
                );
                self.entry.target_lock.remove_guard_file();
                return;
            }
        }
        self.entry.remove_files();
    }

    /// Test hook: forget the entry as a killed process would, leaving its
    /// files on disk and releasing its locks.
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

/// How long [`protect_in`] waits for a recovery that is holding an entry for
/// the same file: this many pauses of [`RECOVERY_WAIT`].
const RECOVERY_WAITS: u32 = 50;
const RECOVERY_WAIT: std::time::Duration = std::time::Duration::from_millis(20);

/// Save `path`'s current content in the default journal directory before it
/// is written to. See [`protect_in`].
pub fn protect(path: &Path) -> io::Result<Protected> {
    protect_in(&default_dir(), path)
}

/// Save `path`'s current content under `dir`, synced to disk, and return the
/// guard that restores it. Fails, without touching `path`, if the original
/// cannot be journaled or another solve is repairing the same file; callers
/// must then not write to the file.
pub fn protect_in(dir: &Path, path: &Path) -> io::Result<Protected> {
    if INTERRUPTED.load(Ordering::SeqCst) {
        return Err(interrupted_error());
    }
    let target = fs::canonicalize(path)?;
    let dir = &real_dir(dir);
    ensure_private_dir(dir)?;
    // From here until the guard is dropped, no other solve may protect this
    // file: it would read whatever candidate is on disk as "the original".
    let target_lock = lock_target(&target)?;
    // A solve that died while repairing this file may have left its candidate
    // in it. Put that right before reading the "original". If a recovery
    // elsewhere is holding that entry right now, it will let go as soon as it
    // finds the file locked by us: wait for it, then recover the entry here.
    let mut waits = 0;
    loop {
        let (recoveries, entry_in_use) = recover_entries(dir, Some(&target));
        for recovery in recoveries {
            if let Some(message) = recovery.describe() {
                println!("[axiom-journal] {message}");
            }
        }
        if !entry_in_use {
            break;
        }
        waits += 1;
        if waits > RECOVERY_WAITS {
            target_lock.remove_guard_file();
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "{} is being recovered by another axiom solve",
                    target.display()
                ),
            ));
        }
        std::thread::sleep(RECOVERY_WAIT);
    }
    let original = match fs::read(&target) {
        Ok(original) => original,
        Err(e) => {
            target_lock.remove_guard_file();
            return Err(e);
        }
    };
    let id = format!(
        "{}-{}-{}",
        target_hash(&target),
        std::process::id(),
        SEQ.fetch_add(1, Ordering::SeqCst)
    );
    let stem = dir.join(id);
    // The lock comes first: an entry without one would look abandoned to a
    // concurrent `recover`.
    let lock = match private_file()
        .create_new(true)
        .open(stem.with_extension("lock"))
    {
        Ok(lock) => lock,
        Err(e) => {
            target_lock.remove_guard_file();
            return Err(e);
        }
    };
    let entry = Entry {
        target,
        original,
        stem,
        lock,
        target_lock,
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
        let meta = Meta {
            path: entry.target.to_string_lossy().into_owned(),
            original_sha256: sha256_hex(&entry.original),
        };
        let json = serde_json::to_vec(&meta).map_err(io::Error::other)?;
        replace_durable(&entry.stem, "json", &json)
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
/// [`recover`] can recognise the file as still holding it, in full or cut
/// short. An unprotected file is written as by [`std::fs::write`].
///
/// Refused once [`restore_all`] has run.
pub fn write(path: &Path, contents: &[u8]) -> io::Result<()> {
    let entries = active();
    if INTERRUPTED.load(Ordering::SeqCst) {
        return Err(interrupted_error());
    }
    if let Ok(target) = fs::canonicalize(path) {
        for entry in entries.iter().filter(|e| e.target == target) {
            replace_durable(&entry.stem, "cand", contents)?;
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
    /// The file still held what the dead process last wrote (in full, or cut
    /// short by the crash); the original was put back.
    Restored,
    /// The file already held the original; nothing to do.
    AlreadyOriginal,
    /// The file holds something else: it was edited after the crash. It was
    /// left alone, and the original is kept at `saved_original`.
    LeftChanged { saved_original: PathBuf },
    /// The file no longer exists; nothing was recreated.
    TargetMissing,
    /// The file could not be read, or the original could not be written back
    /// (a permission change, say). Nothing was touched: the entry and the
    /// original at `saved_original` are kept, and the next recovery tries
    /// again.
    Deferred { saved_original: PathBuf },
}

/// One entry handled by [`recover`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovery {
    pub target: PathBuf,
    pub outcome: RecoveryOutcome,
}

impl Recovery {
    /// What to tell the user, if anything.
    pub fn describe(&self) -> Option<String> {
        let target = self.target.display();
        match &self.outcome {
            RecoveryOutcome::Restored => Some(format!(
                "an earlier solve was killed mid-repair: restored {target} to its original content"
            )),
            RecoveryOutcome::LeftChanged { saved_original } => Some(format!(
                "an earlier solve was killed mid-repair, and {target} has changed since; it was \
                 left as it is, and its content from before that solve is saved at {}",
                saved_original.display()
            )),
            RecoveryOutcome::Deferred { saved_original } => Some(format!(
                "an earlier solve was killed mid-repair, and {target} cannot be read or written \
                 right now; nothing was changed, and its content from before that solve is saved \
                 at {}",
                saved_original.display()
            )),
            RecoveryOutcome::AlreadyOriginal | RecoveryOutcome::TargetMissing => None,
        }
    }
}

/// Handle entries abandoned in the default journal directory. See
/// [`recover_in`].
pub fn recover() -> Vec<Recovery> {
    recover_in(&default_dir())
}

/// Was `current` left by the dead process? Yes if it is exactly what the
/// process last wrote, or a write that the crash cut short: `fs::write`
/// truncates and then writes from the start, so a cut-short write leaves the
/// beginning of what was being written (a candidate, or the original during
/// a rollback).
fn left_by_the_dead_process(current: &[u8], original: &[u8], last_written: Option<&[u8]>) -> bool {
    last_written.is_some_and(|written| written.starts_with(current))
        || original.starts_with(current)
}

/// Find entries under `dir` whose owning process is gone and restore their
/// files where that is safe (see [`RecoveryOutcome`]). Entries whose owner is
/// alive are skipped, and so is one whose file another solve holds right now
/// (that solve deals with it). A directory that other users can reach is not
/// trusted and nothing in it is used.
pub fn recover_in(dir: &Path) -> Vec<Recovery> {
    recover_entries(dir, None).0
}

/// [`recover_in`], for a caller that may already hold the lock on one file
/// (`held`, a canonical path). Entries for that file are recovered under the
/// caller's lock; for every other file the lock is taken here.
///
/// The second value is true if an entry for `held` exists but is locked by
/// somebody else. The caller holds that file's lock, so it cannot be a live
/// solve's entry: another process is in the middle of recovering it.
fn recover_entries(dir: &Path, held: Option<&Path>) -> (Vec<Recovery>, bool) {
    let dir = &real_dir(dir);
    if !dir.is_dir() {
        return (Vec::new(), false);
    }
    if was_set_aside(dir) {
        eprintln!(
            "[axiom-journal] not recovering anything: {} was set aside as untrusted",
            dir.display()
        );
        return (Vec::new(), false);
    }
    match exposure(dir) {
        Ok(Exposure::Private) => {}
        Ok(_) => {
            eprintln!(
                "[axiom-journal] not recovering anything: journal directory {} can be read or \
                 written by other users",
                dir.display()
            );
            return (Vec::new(), false);
        }
        Err(e) => {
            eprintln!("[axiom-journal] not recovering anything: {e}");
            return (Vec::new(), false);
        }
    }
    let Ok(listing) = fs::read_dir(dir) else {
        return (Vec::new(), false);
    };
    let held_prefix = held.map(|target| format!("{}-", target_hash(target)));
    let mut done = Vec::new();
    let mut held_entry_in_use = false;
    for lock_path in listing
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "lock"))
    {
        let Ok(lock) = OpenOptions::new().write(true).open(&lock_path) else {
            continue;
        };
        // Locked: the owner is alive (possibly this very process), or another
        // recovery has it.
        if lock.try_lock().is_err() {
            let name = lock_path.file_name().map(|n| n.to_string_lossy());
            if let (Some(prefix), Some(name)) = (&held_prefix, name) {
                held_entry_in_use |= name.starts_with(prefix.as_str());
            }
            continue;
        }
        let stem = lock_path.with_extension("");
        let orig_path = stem.with_extension("orig");
        // What an entry says is written over a source file, so an entry
        // somebody else could have edited is never acted on, now or later.
        // Its files are renamed so that they stop being an entry: closing
        // them afterwards does not undo a change already made. The lock goes
        // first, since an entry is found by its lock. Nothing is deleted: the
        // saved original may still be wanted.
        if ["orig", "json", "cand"]
            .iter()
            .any(|ext| others_could_have_changed(&stem.with_extension(ext)))
        {
            for ext in ["lock", "json", "cand", "orig"] {
                let file = stem.with_extension(ext);
                let aside = stem.with_extension(format!("{ext}{ENTRY_SET_ASIDE}"));
                if let Err(e) = fs::rename(&file, &aside) {
                    if e.kind() != io::ErrorKind::NotFound {
                        eprintln!(
                            "[axiom-journal] could not set {} aside: {e}",
                            file.display()
                        );
                    }
                }
            }
            eprintln!(
                "[axiom-journal] not restoring from {}: its files could have been changed by \
                 other users. No source file was touched. The entry was renamed to *{} and \
                 will not be used; check the saved original by hand.",
                orig_path.display(),
                ENTRY_SET_ASIDE
            );
            continue;
        }
        let meta = fs::read(stem.with_extension("json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Meta>(&bytes).ok());
        let original = fs::read(&orig_path).ok();
        let last_written = fs::read(stem.with_extension("cand")).ok();
        let mut keep_original = false;
        // An entry cut short before both files were written protected a file
        // that had not been written to yet: nothing to restore.
        if let (Some(meta), Some(original)) = (meta, original) {
            if sha256_hex(&original) == meta.original_sha256 {
                let target = PathBuf::from(&meta.path);
                let deferred = || RecoveryOutcome::Deferred {
                    saved_original: orig_path.clone(),
                };
                // Nobody else may be repairing the file while it is put back,
                // so it is only read or written with its lock held: by the
                // caller for `held`, taken here for any other file.
                let lock_taken = if held == Some(target.as_path()) {
                    Ok(None)
                } else {
                    lock_target(&target).map(Some)
                };
                let outcome = match lock_taken {
                    // A live solve has the file. It recovers this entry
                    // itself, under its lock; leave everything for it.
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => RecoveryOutcome::TargetMissing,
                    // The lock could not be taken for some other reason (the
                    // file is unreadable, say). Without it nothing is touched.
                    Err(_) => deferred(),
                    Ok(target_lock) => {
                        let outcome = match fs::read(&target) {
                            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                                RecoveryOutcome::TargetMissing
                            }
                            // Unreadable is not gone: the file may still hold
                            // the candidate, so the original is kept.
                            Err(_) => deferred(),
                            Ok(current) if current == original => RecoveryOutcome::AlreadyOriginal,
                            Ok(current)
                                if left_by_the_dead_process(
                                    &current,
                                    &original,
                                    last_written.as_deref(),
                                ) =>
                            {
                                match fs::write(&target, &original) {
                                    Ok(()) => RecoveryOutcome::Restored,
                                    Err(_) => deferred(),
                                }
                            }
                            Ok(_) => {
                                keep_original = true;
                                RecoveryOutcome::LeftChanged {
                                    saved_original: orig_path.clone(),
                                }
                            }
                        };
                        if let Some(lock) = &target_lock {
                            lock.remove_guard_file();
                        }
                        outcome
                    }
                };
                let retry_later = matches!(outcome, RecoveryOutcome::Deferred { .. });
                done.push(Recovery { target, outcome });
                if retry_later {
                    // Leave every file of the entry; dropping `lock` releases it.
                    continue;
                }
            }
        }
        for ext in ENTRY_EXTENSIONS {
            if ext == "lock" || (ext == "orig" && keep_original) {
                continue;
            }
            let _ = fs::remove_file(stem.with_extension(ext));
        }
        drop(lock);
        let _ = fs::remove_file(&lock_path);
    }
    (done, held_entry_in_use)
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
    fn recovery_keeps_the_original_when_the_file_cannot_be_read() {
        let (_tmp, journal, source) = fixture("original\n");
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"unverified candidate\n").unwrap();
        guard.abandon();
        // The file cannot be read right now (here: a directory sits at its
        // path; in the field, a permission change). That is not "gone".
        let aside = source.with_extension("aside");
        fs::rename(&source, &aside).unwrap();
        fs::create_dir(&source).unwrap();

        let recovered = recover_in(&journal);
        assert_eq!(recovered.len(), 1);
        let RecoveryOutcome::Deferred { saved_original } = &recovered[0].outcome else {
            panic!("expected Deferred, got {:?}", recovered[0].outcome);
        };
        assert_eq!(fs::read(saved_original).unwrap(), b"original\n");

        // Readable again, still holding the candidate: the next recovery
        // restores it from the entry that was kept.
        fs::remove_dir(&source).unwrap();
        fs::rename(&aside, &source).unwrap();
        assert_eq!(recover_in(&journal)[0].outcome, RecoveryOutcome::Restored);
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
        assert_eq!(entry_files(&journal), 0);
    }

    #[cfg(unix)]
    #[test]
    fn journal_files_are_readable_by_their_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let (_tmp, journal, source) = fixture("a private source file\n");
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"candidate\n").unwrap();
        let mode = |path: &Path| fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&journal), 0o700, "the journal directory");
        let files: Vec<PathBuf> = fs::read_dir(&journal)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .collect();
        // The lock, the original, the metadata and the last write (plus a
        // guard file where the target lock is not on the source itself).
        assert!(files.len() >= 4, "{files:?}");
        for file in &files {
            assert_eq!(mode(file), 0o600, "{}", file.display());
        }
        drop(guard);
    }

    #[cfg(unix)]
    #[test]
    fn a_journal_directory_open_to_others_is_closed_or_refused() {
        use std::os::unix::fs::PermissionsExt;
        let (_tmp, journal, source) = fixture("original\n");
        let mode = |path: &Path| fs::metadata(path).unwrap().permissions().mode() & 0o777;

        // An entry is abandoned, and then the directory is opened up.
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"unverified candidate\n").unwrap();
        guard.abandon();
        fs::set_permissions(&journal, fs::Permissions::from_mode(0o755)).unwrap();

        // Recovery does not act on files in a directory others can reach.
        assert!(recover_in(&journal).is_empty());
        assert_eq!(fs::read(&source).unwrap(), b"unverified candidate\n");

        // Protecting a file closes a directory this user owns, after which
        // the abandoned entry is recovered before the original is read.
        let guard = protect_in(&journal, &source).unwrap();
        assert_eq!(mode(&journal), 0o700);
        assert_eq!(guard.original(), b"original\n");
        drop(guard);

        // A directory that belongs to somebody else is never used. Only root
        // can give a directory away, so this part runs only as root.
        if std::os::unix::fs::chown(&journal, Some(54321), None).is_ok() {
            let err = protect_in(&journal, &source).err().expect("refused");
            assert!(err.to_string().contains("belongs to another user"), "{err}");
            assert_eq!(fs::read(&source).unwrap(), b"original\n");
        }
    }

    /// Make `journal` a world-writable directory holding a complete,
    /// self-consistent entry somebody planted: its "original" is their
    /// content, and its last write is what `victim` holds now, so recovery
    /// would take the file for an abandoned candidate. Returns their content.
    #[cfg(unix)]
    fn plant_forged_entry(journal: &Path, victim: &Path) -> &'static [u8] {
        use std::os::unix::fs::PermissionsExt;
        fs::create_dir(journal).unwrap();
        fs::set_permissions(journal, fs::Permissions::from_mode(0o777)).unwrap();
        let forged = b"attacker content\n";
        let stem = journal.join("0123456789abcdef-1-0");
        fs::write(stem.with_extension("lock"), b"").unwrap();
        fs::write(stem.with_extension("orig"), forged).unwrap();
        fs::write(stem.with_extension("cand"), fs::read(victim).unwrap()).unwrap();
        fs::write(
            stem.with_extension("json"),
            serde_json::to_vec(&Meta {
                path: victim.to_string_lossy().into_owned(),
                original_sha256: sha256_hex(forged),
            })
            .unwrap(),
        )
        .unwrap();
        forged
    }

    /// The directories [`set_aside`] left next to `<parent>/<name>`.
    #[cfg(unix)]
    fn set_aside_dirs(parent: &Path, name: &str) -> Vec<PathBuf> {
        fs::read_dir(parent)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(&format!("{name}{SET_ASIDE_MARK}"))
            })
            .collect()
    }

    #[cfg(unix)]
    #[test]
    fn a_forged_entry_in_a_directory_others_could_write_to_is_not_restored() {
        use std::os::unix::fs::PermissionsExt;
        let (tmp, journal, source) = fixture("original\n");
        let victim = tmp.path().join("victim.sh");
        fs::write(&victim, "the user's real file\n").unwrap();
        let victim = fs::canonicalize(&victim).unwrap();

        // The journal directory is world-writable, and somebody has planted
        // an entry in it.
        let forged = plant_forged_entry(&journal, &victim);

        // While it is open, recovery uses nothing from it.
        assert!(recover_in(&journal).is_empty());
        assert_eq!(fs::read(&victim).unwrap(), b"the user's real file\n");

        // Protecting a file never uses that directory again: it is moved
        // aside whole, and a fresh private one takes its place.
        let guard = protect_in(&journal, &source).unwrap();
        assert_eq!(
            fs::metadata(&journal).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(fs::read(&victim).unwrap(), b"the user's real file\n");
        drop(guard);
        assert_eq!(entry_files(&journal), 0, "the new directory starts empty");
        assert!(recover_in(&journal).is_empty());
        assert_eq!(fs::read(&victim).unwrap(), b"the user's real file\n");

        // Nothing was deleted: the old directory, planted files included, is
        // beside the new one and closed to others, where a genuine backup
        // could still be restored by hand.
        let aside = set_aside_dirs(tmp.path(), "journal");
        assert_eq!(aside.len(), 1, "{aside:?}");
        assert_eq!(
            fs::metadata(&aside[0]).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::read(aside[0].join("0123456789abcdef-1-0.orig")).unwrap(),
            forged
        );

        // Closed or not, the set-aside directory is never a journal: pointing
        // recovery or a solve at it restores nothing and is refused.
        assert!(recover_in(&aside[0]).is_empty());
        let err = protect_in(&aside[0], &source).err().expect("refused");
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied, "{err}");
        assert_eq!(fs::read(&victim).unwrap(), b"the user's real file\n");
    }

    #[cfg(unix)]
    #[test]
    fn an_entry_whose_files_others_could_write_to_is_not_restored() {
        use std::os::unix::fs::PermissionsExt;
        let set_modes = |journal: &Path, mode: u32| {
            for ext in ["orig", "json", "cand", "lock"] {
                let file = journal.join(format!("0123456789abcdef-1-0.{ext}"));
                fs::set_permissions(file, fs::Permissions::from_mode(mode)).unwrap();
            }
        };
        let (tmp, journal, source) = fixture("original\n");
        let victim = tmp.path().join("victim.sh");
        fs::write(&victim, "the user's real file\n").unwrap();
        let victim = fs::canonicalize(&victim).unwrap();

        // The directory only ever let others read it, so nobody else could
        // add a file. But the entry's files are open to writing: anybody
        // could have replaced the saved "original" and its recorded hash.
        let forged = plant_forged_entry(&journal, &victim);
        fs::set_permissions(&journal, fs::Permissions::from_mode(0o755)).unwrap();
        set_modes(&journal, 0o666);

        // A solve closes the directory and keeps it, and uses nothing from
        // that entry.
        let guard = protect_in(&journal, &source).unwrap();
        drop(guard);
        assert_eq!(
            fs::metadata(&journal).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert!(recover_in(&journal).is_empty());
        assert_eq!(fs::read(&victim).unwrap(), b"the user's real file\n");
        assert!(set_aside_dirs(tmp.path(), "journal").is_empty());

        // The entry's files are kept, for a look by hand, under names that
        // are not an entry's.
        let mut names: Vec<String> = fs::read_dir(&journal)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "0123456789abcdef-1-0.cand.untrusted",
                "0123456789abcdef-1-0.json.untrusted",
                "0123456789abcdef-1-0.lock.untrusted",
                "0123456789abcdef-1-0.orig.untrusted",
            ]
        );
        assert_eq!(
            fs::read(journal.join("0123456789abcdef-1-0.orig.untrusted")).unwrap(),
            forged
        );

        // Closing the files afterwards does not undo a change already made,
        // and does not make the entry usable.
        for name in &names {
            fs::set_permissions(journal.join(name), fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert!(recover_in(&journal).is_empty());
        let guard = protect_in(&journal, &source).unwrap();
        drop(guard);
        assert_eq!(fs::read(&victim).unwrap(), b"the user's real file\n");
    }

    #[cfg(unix)]
    #[test]
    fn an_entry_whose_files_only_its_owner_could_write_is_still_recovered() {
        // The counterpart of the test above: the same directory and the same
        // kind of entry, but its files were never open to writing by others,
        // so nobody else could have altered them. It is recovered.
        use std::os::unix::fs::PermissionsExt;
        let (tmp, journal, _source) = fixture("original\n");
        let target = tmp.path().join("target.sh");
        fs::write(&target, "abandoned candidate\n").unwrap();
        let target = fs::canonicalize(&target).unwrap();
        let saved = plant_forged_entry(&journal, &target);
        fs::set_permissions(&journal, fs::Permissions::from_mode(0o755)).unwrap();
        for ext in ["orig", "json", "cand", "lock"] {
            let file = journal.join(format!("0123456789abcdef-1-0.{ext}"));
            fs::set_permissions(file, fs::Permissions::from_mode(0o644)).unwrap();
        }
        // A directory others can read is not used for recovery as it stands.
        assert!(recover_in(&journal).is_empty());
        // Once a solve has closed it, the entry is recovered.
        ensure_private_dir(&journal).unwrap();
        let recovered = recover_in(&journal);
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].outcome, RecoveryOutcome::Restored);
        assert_eq!(fs::read(&target).unwrap(), saved);
    }

    #[cfg(unix)]
    #[test]
    fn stopping_right_after_an_untrusted_directory_is_moved_restores_nothing() {
        let (tmp, journal, source) = fixture("original\n");
        let victim = tmp.path().join("victim.sh");
        fs::write(&victim, "the user's real file\n").unwrap();
        let victim = fs::canonicalize(&victim).unwrap();
        plant_forged_entry(&journal, &victim);

        // The process gets as far as moving the directory and then dies:
        // no fresh directory was created, nothing else was done.
        set_aside(&journal).unwrap();
        assert!(!journal.exists());

        // The next solve finds no journal where it looks, and the planted
        // entry is nowhere it would look.
        assert!(recover_in(&journal).is_empty());
        assert_eq!(fs::read(&victim).unwrap(), b"the user's real file\n");
        let guard = protect_in(&journal, &source).unwrap();
        drop(guard);
        assert_eq!(entry_files(&journal), 0);
        assert_eq!(fs::read(&victim).unwrap(), b"the user's real file\n");
        assert_eq!(set_aside_dirs(tmp.path(), "journal").len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn a_journal_path_that_links_to_a_directory_others_could_write_to_is_not_trusted() {
        use std::os::unix::fs::PermissionsExt;
        let (tmp, link, source) = fixture("original\n");
        let victim = tmp.path().join("victim.sh");
        fs::write(&victim, "the user's real file\n").unwrap();
        let victim = fs::canonicalize(&victim).unwrap();

        // The configured journal path is a symbolic link. The directory it
        // points at is world-writable and holds a planted entry.
        let real = fs::canonicalize(tmp.path()).unwrap().join("real");
        let forged = plant_forged_entry(&real, &victim);
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(recover_in(&link).is_empty());

        // It is the directory, not the link, that is moved aside and
        // replaced: the planted entry is no longer at either path.
        let guard = protect_in(&link, &source).unwrap();
        drop(guard);
        assert!(fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read_link(&link).unwrap(), real);
        assert_eq!(
            fs::metadata(&real).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(entry_files(&real), 0, "a fresh directory, nothing planted");
        for path in [&link, &real] {
            assert!(recover_in(path).is_empty());
        }
        assert_eq!(fs::read(&victim).unwrap(), b"the user's real file\n");

        // The planted files are beside the real directory, for inspection.
        let aside = set_aside_dirs(real.parent().unwrap(), "real");
        assert_eq!(aside.len(), 1, "{aside:?}");
        assert_eq!(
            fs::read(aside[0].join("0123456789abcdef-1-0.orig")).unwrap(),
            forged
        );
        assert!(recover_in(&aside[0]).is_empty());
        assert_eq!(fs::read(&victim).unwrap(), b"the user's real file\n");

        // A link that points nowhere is refused, and nothing is created
        // through it.
        let dangling = tmp.path().join("dangling");
        std::os::unix::fs::symlink(tmp.path().join("missing"), &dangling).unwrap();
        assert!(protect_in(&dangling, &source).is_err());
        assert!(!tmp.path().join("missing").exists());
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
    }

    #[test]
    fn recovery_touches_nothing_when_the_files_lock_cannot_be_taken() {
        let tmp = tempfile::tempdir().unwrap();
        let journal = tmp.path().join("journal");
        let project = tmp.path().join("project");
        fs::create_dir(&project).unwrap();
        let source = project.join("lib.sh");
        fs::write(&source, "original\n").unwrap();
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, b"unverified candidate\n").unwrap();
        guard.abandon();

        // The file cannot even be opened now, and not because it is gone: a
        // regular file sits where its directory was.
        let moved = tmp.path().join("project.moved");
        fs::rename(&project, &moved).unwrap();
        fs::write(&project, "not a directory").unwrap();
        let recovered = recover_in(&journal);
        assert_eq!(recovered.len(), 1);
        assert!(
            matches!(recovered[0].outcome, RecoveryOutcome::Deferred { .. }),
            "{:?}",
            recovered[0].outcome
        );
        assert_eq!(
            fs::read(moved.join("lib.sh")).unwrap(),
            b"unverified candidate\n"
        );

        // Back in place: the entry was kept, and the file is restored.
        fs::remove_file(&project).unwrap();
        fs::rename(&moved, &project).unwrap();
        assert_eq!(recover_in(&journal)[0].outcome, RecoveryOutcome::Restored);
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
    }

    #[test]
    fn the_guard_file_lock_does_not_depend_on_the_journal_directory() {
        // The lock used where the source file itself cannot be locked.
        let (tmp, _journal, source) = fixture("original\n");
        let guards = tmp.path().join("locks");
        let first = lock_guard_file(&guards, &source).unwrap();
        let err = lock_guard_file(&guards, &source).err().expect("held");
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock, "{err}");
        // Another file is not affected.
        let other = tmp.path().join("other.sh");
        fs::write(&other, "x\n").unwrap();
        let second = lock_guard_file(&guards, &other).unwrap();
        second.remove_guard_file();
        drop(second);
        // Released and removed: free again, and nothing is left behind.
        first.remove_guard_file();
        drop(first);
        let again = lock_guard_file(&guards, &source).unwrap();
        again.remove_guard_file();
        drop(again);
        assert_eq!(fs::read_dir(&guards).unwrap().count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn a_second_protect_through_another_journal_directory_is_refused() {
        let (tmp, journal, source) = fixture("original\n");
        let other_journal = tmp.path().join("another-journal");
        let first = protect_in(&journal, &source).unwrap();
        write(&source, b"first solve's candidate\n").unwrap();

        // A solve configured with a different journal directory still meets
        // the first one on the file itself.
        let err = protect_in(&other_journal, &source).err().expect("refused");
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock, "{err}");
        assert_eq!(fs::read(&source).unwrap(), b"first solve's candidate\n");
        drop(first);
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
    }

    #[cfg(unix)]
    #[test]
    fn recovery_leaves_a_file_that_a_live_solve_holds() {
        let (_tmp, journal, source) = fixture("original\n");
        let dead = protect_in(&journal, &source).unwrap();
        write(&source, b"dead solve's candidate\n").unwrap();
        dead.abandon();

        // Another solve has the file right now (it holds the lock on it and
        // is about to deal with the abandoned entry itself).
        let held = File::open(&source).unwrap();
        held.try_lock().unwrap();
        assert!(recover_in(&journal).is_empty());
        assert_eq!(fs::read(&source).unwrap(), b"dead solve's candidate\n");
        assert!(entry_files(&journal) > 0, "the entry is kept for later");

        // Once the file is free, recovery restores it.
        drop(held);
        assert_eq!(recover_in(&journal)[0].outcome, RecoveryOutcome::Restored);
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
    }

    /// The `.lock` file of the one entry in `journal`.
    fn only_entry_lock(journal: &Path) -> PathBuf {
        let locks: Vec<PathBuf> = fs::read_dir(journal)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "lock"))
            .collect();
        assert_eq!(locks.len(), 1, "{locks:?}");
        locks.into_iter().next().unwrap()
    }

    #[test]
    fn protect_waits_for_a_recovery_that_has_the_files_entry() {
        let (_tmp, journal, source) = fixture("original\n");
        let dead = protect_in(&journal, &source).unwrap();
        write(&source, b"dead solve's candidate\n").unwrap();
        dead.abandon();

        // Another process's recovery has taken the entry and not yet put the
        // file back. It lets go of the entry shortly afterwards (as a real
        // one does when it finds the file locked by the protecting solve).
        let recoverer = OpenOptions::new()
            .write(true)
            .open(only_entry_lock(&journal))
            .unwrap();
        recoverer.try_lock().unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(150));
            drop(recoverer);
        });

        // Protect must not read the candidate as "the original" meanwhile.
        let live = protect_in(&journal, &source).unwrap();
        assert_eq!(live.original(), b"original\n");
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
        release.join().unwrap();
        drop(live);
        assert_eq!(entry_files(&journal), 0);
    }

    #[test]
    fn protect_gives_up_on_a_file_whose_recovery_never_finishes() {
        let (_tmp, journal, source) = fixture("original\n");
        let dead = protect_in(&journal, &source).unwrap();
        write(&source, b"dead solve's candidate\n").unwrap();
        dead.abandon();
        let stuck = OpenOptions::new()
            .write(true)
            .open(only_entry_lock(&journal))
            .unwrap();
        stuck.try_lock().unwrap();

        let err = protect_in(&journal, &source).err().expect("refused");
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock, "{err}");
        assert!(err.to_string().contains("being recovered"), "{err}");
        // Nothing was journaled or written.
        assert_eq!(fs::read(&source).unwrap(), b"dead solve's candidate\n");
        drop(stuck);
        assert_eq!(recover_in(&journal)[0].outcome, RecoveryOutcome::Restored);
    }

    #[test]
    fn a_second_protect_of_the_same_file_is_refused() {
        let (_tmp, journal, source) = fixture("original\n");
        let first = protect_in(&journal, &source).unwrap();
        write(&source, b"first solve's candidate\n").unwrap();

        // A second solve reaches the same file. If it were let in, it would
        // journal the first one's candidate as "the original".
        let err = protect_in(&journal, &source).err().expect("refused");
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock, "{err}");
        assert!(err.to_string().contains("another axiom solve"), "{err}");
        assert_eq!(fs::read(&source).unwrap(), b"first solve's candidate\n");

        drop(first);
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
        // Free again once the first solve is done with it.
        let again = protect_in(&journal, &source).unwrap();
        assert_eq!(again.original(), b"original\n");
        drop(again);
        assert_eq!(entry_files(&journal), 0);
    }

    #[test]
    fn protect_first_puts_right_a_file_left_by_a_dead_solve() {
        let (_tmp, journal, source) = fixture("original\n");
        let dead = protect_in(&journal, &source).unwrap();
        write(&source, b"dead solve's candidate\n").unwrap();
        dead.abandon();

        // A solve that was already running reaches the file without having
        // run recovery since. The original it journals must be the real one.
        let live = protect_in(&journal, &source).unwrap();
        assert_eq!(live.original(), b"original\n");
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
        drop(live);
        assert_eq!(entry_files(&journal), 0);
    }

    #[test]
    fn recovery_restores_a_write_that_the_crash_cut_short() {
        let candidate = b"a candidate that is long enough to be cut in the middle\n";
        // The crash came while the candidate was being written: the file
        // holds its first half, or nothing at all.
        for torn in [&candidate[..candidate.len() / 2], &candidate[..0]] {
            let (_tmp, journal, source) = fixture("original\n");
            let guard = protect_in(&journal, &source).unwrap();
            write(&source, candidate).unwrap();
            fs::write(&source, torn).unwrap();
            guard.abandon();
            assert_eq!(recover_in(&journal)[0].outcome, RecoveryOutcome::Restored);
            assert_eq!(fs::read(&source).unwrap(), b"original\n");
        }
        // The crash came during a rollback: the file holds the beginning of
        // the original.
        let (_tmp, journal, source) = fixture("original\n");
        let guard = protect_in(&journal, &source).unwrap();
        write(&source, candidate).unwrap();
        write(&source, b"original\n").unwrap();
        fs::write(&source, b"orig").unwrap();
        guard.abandon();
        assert_eq!(recover_in(&journal)[0].outcome, RecoveryOutcome::Restored);
        assert_eq!(fs::read(&source).unwrap(), b"original\n");
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
