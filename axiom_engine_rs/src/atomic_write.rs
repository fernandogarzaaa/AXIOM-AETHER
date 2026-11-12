//! Replace a file's contents so that neither a reader nor a crash ever finds
//! half of them.
//!
//! `std::fs::write` opens the file, empties it, and then writes. Until the
//! last byte is down, the only copy of the file is incomplete: a process
//! reading it meanwhile gets a fragment, and a process killed meanwhile leaves
//! one. For a store that loads an unparseable file as empty, that is the whole
//! store gone.
//!
//! [`write_replacing`] writes the new contents to a file of its own beside the
//! target, syncs it, and renames it over the target. A rename within one
//! directory is one step: at every moment the path names either the complete
//! old file or the complete new one.
//!
//! A replacement is a different file from the one it replaces, so what the
//! old file had besides its contents has to be carried over or the difference
//! accounted for. The rules are on [`write_replacing`].

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes the scratch files of concurrent writers in one process.
static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

/// How many scratch names that are already taken to step past before giving
/// up.
const SCRATCH_TRIES: u32 = 64;

/// Give `path` the contents `contents`, by rename.
///
/// * The directory of `path` is created if it is missing.
/// * If `path` is a symbolic link, the file it points at is replaced and the
///   link is left as it is. (A link that points nowhere is itself replaced
///   by the file.)
/// * **Who may save.** A rename needs the right to change the directory, not
///   the right to write the file. Writing the file in place needed the
///   latter, so that is still asked for: a file this process could not have
///   written is refused, and left as it was.
/// * **Who may read the result.** The new file has the old file's permission
///   bits, owner and group. The scratch file is created readable by its owner
///   alone and is given the old file's permissions before any contents are
///   written to it, so the contents are never readable by anyone who could
///   not read the old file. A file that did not exist before gets the process
///   default, as it always did.
/// * On any failure the old file is untouched and the scratch file is
///   removed.
///
/// Not carried over: access control lists and other extended attributes of
/// the old file, and any other names (hard links) it had.
///
/// One exception, so that a save which worked before this function existed
/// still works and leaves the file with the access it had: where the file
/// cannot be replaced (it is a single file mounted into a container, or it
/// sits in a directory this process may not create files in), or could only
/// be replaced by a file with a different owner or group, it is written in
/// place, as `std::fs::write` would. That is decided before any of the
/// contents are written, or by the kind of error the rename gives, and never
/// after a failed write of the contents: a full disk does not lead to the old
/// file being emptied.
pub fn write_replacing(path: &Path, contents: &[u8]) -> io::Result<()> {
    replace(path, contents, Otherwise::WriteInPlace, &Steps::REAL)
}

/// [`write_replacing`] without its exception: the file is replaced by rename
/// or the save fails, and the file is never written in place.
///
/// For a file this process creates and nothing else writes, where a
/// half-written file is worse than a failed save. If the old file's owner or
/// group cannot be given to the new one, the new file belongs to this
/// process.
pub fn write_replacing_or_fail(path: &Path, contents: &[u8]) -> io::Result<()> {
    replace(path, contents, Otherwise::Fail, &Steps::REAL)
}

/// What to do about a file that cannot be replaced by rename.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Otherwise {
    WriteInPlace,
    Fail,
}

/// The steps the platform can refuse, supplied so that a test can refuse
/// them.
struct Steps {
    /// Move the scratch file over the target.
    rename: fn(&Path, &Path) -> io::Result<()>,
    /// Give the scratch file to this owner and group.
    #[cfg(unix)]
    give: fn(&fs::File, u32, u32) -> io::Result<()>,
}

impl Steps {
    const REAL: Steps = Steps {
        rename: |from, to| fs::rename(from, to),
        #[cfg(unix)]
        give: |file, owner, group| std::os::unix::fs::fchown(file, Some(owner), Some(group)),
    };
}

fn replace(path: &Path, contents: &[u8], otherwise: Otherwise, steps: &Steps) -> io::Result<()> {
    // Follow links, so that it is the file that is replaced and not a link to
    // it. A path that does not exist yet is used as given.
    let target = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if let Some(parent) = target.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let existing = match fs::metadata(&target) {
        Ok(meta) => Some(meta),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => return Err(e),
    };
    if existing.is_some() {
        // The right an in-place write needed. Opening for writing asks for
        // exactly that and changes nothing: the file is not emptied, and
        // nothing is written to it.
        OpenOptions::new().write(true).open(&target)?;
    }
    // The file as `std::fs::write` would have left it. Only for a file that
    // is already there, and only for a caller that allows it.
    let in_place = |why: io::Error| match otherwise {
        Otherwise::WriteInPlace if existing.is_some() => fs::write(&target, contents),
        _ => Err(why),
    };

    let (scratch, mut file) = match create_scratch(&target, existing.is_some(), &SCRATCH_SEQ) {
        Ok(created) => created,
        // No file may be created beside the target.
        Err(e) if e.kind() == io::ErrorKind::PermissionDenied => return in_place(e),
        Err(e) => return Err(e),
    };
    // `Ok(false)`: nothing was written to the scratch file, because the
    // replacement would not belong to whom the old file belongs.
    let written = (|| -> io::Result<bool> {
        if let Some(old) = &existing {
            if !keep_owner(&file, old, steps) && otherwise == Otherwise::WriteInPlace {
                return Ok(false);
            }
            // After the owner: giving a file away clears some of its bits.
            // Before the contents: see "who may read the result".
            file.set_permissions(old.permissions())?;
        }
        file.write_all(contents)?;
        // The contents must be on disk before the name points at them.
        file.sync_all()?;
        Ok(true)
    })();
    drop(file);
    match written {
        Ok(true) => {}
        Ok(false) => {
            let _ = fs::remove_file(&scratch);
            return fs::write(&target, contents);
        }
        Err(e) => {
            let _ = fs::remove_file(&scratch);
            return Err(e);
        }
    }
    match (steps.rename)(&scratch, &target) {
        Ok(()) => {
            sync_directory(&target);
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_file(&scratch);
            // The complete new contents were written and synced, so the disk
            // has room for them; the target itself cannot be replaced.
            if matches!(
                e.kind(),
                io::ErrorKind::ResourceBusy | io::ErrorKind::PermissionDenied
            ) {
                in_place(e)
            } else {
                Err(e)
            }
        }
    }
}

/// Create the scratch file: `<target>.tmp-<pid>-<n>`, beside the target so
/// the rename stays within one directory.
///
/// `private`: readable and writable by its owner alone, whatever the process
/// default is. For the replacement of an existing file, which may hold what
/// only that file's readers may see.
///
/// A name that is taken is stepped over, not reused and not removed. It is
/// the leftover of a process that was killed in the middle of a save and
/// whose process id this one now has (a restarted container starts again at
/// the same id and the same `<n>`), or it belongs to a live writer with the
/// same process id in another container that shares the directory.
fn create_scratch(
    target: &Path,
    private: bool,
    seq: &AtomicU64,
) -> io::Result<(PathBuf, fs::File)> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    if private {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    #[cfg(not(unix))]
    let _ = private;

    let mut taken = 0;
    loop {
        let mut name = target.as_os_str().to_os_string();
        name.push(format!(
            ".tmp-{}-{}",
            std::process::id(),
            seq.fetch_add(1, Ordering::Relaxed)
        ));
        let scratch = PathBuf::from(name);
        match options.open(&scratch) {
            Ok(file) => return Ok((scratch, file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists && taken < SCRATCH_TRIES => {
                taken += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Make the scratch file belong to the owner and group of the file it will
/// replace. False if it does not and cannot: only a privileged process can
/// give a file to another user, and any other only to a group it is in.
#[cfg(unix)]
fn keep_owner(scratch: &fs::File, old: &fs::Metadata, steps: &Steps) -> bool {
    use std::os::unix::fs::MetadataExt;
    let Ok(new) = scratch.metadata() else {
        return false;
    };
    (new.uid() == old.uid() && new.gid() == old.gid())
        || (steps.give)(scratch, old.uid(), old.gid()).is_ok()
}

/// Ownership is not a property of a file that a rename changes here.
#[cfg(not(unix))]
fn keep_owner(_scratch: &fs::File, _old: &fs::Metadata, _steps: &Steps) -> bool {
    true
}

/// The directory that holds `target`: for a bare file name, the current one.
fn directory_of(target: &Path) -> Option<&Path> {
    target.parent().map(|parent| {
        if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        }
    })
}

/// Ask for the rename itself to reach the disk. Best effort: the file is
/// already complete under one of its two names, and not every platform can
/// sync a directory.
fn sync_directory(target: &Path) {
    if let Some(dir) = directory_of(target).and_then(|dir| fs::File::open(dir).ok()) {
        let _ = dir.sync_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn writes_a_new_file_and_replaces_an_old_one_leaving_nothing_else() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("store.json");
        write_replacing(&path, b"one").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"one");
        write_replacing(&path, b"two, longer").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"two, longer");
        write_replacing(&path, b"3").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"3");
        assert_eq!(names(path.parent().unwrap()), ["store.json"]);
    }

    #[cfg(unix)]
    #[test]
    fn the_old_file_is_replaced_not_written_into() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.json");
        write_replacing(&path, b"the old contents").unwrap();
        // A second name for the same file on disk shows whether the next
        // write goes into that file.
        let kept = dir.path().join("as-it-was");
        fs::hard_link(&path, &kept).unwrap();
        write_replacing(&path, b"new").unwrap();
        assert_eq!(fs::read(&kept).unwrap(), b"the old contents");
        assert_eq!(fs::read(&path).unwrap(), b"new");
    }

    #[test]
    fn a_failed_write_leaves_the_old_file_and_no_scratch_file() {
        let dir = tempfile::tempdir().unwrap();
        // The target is a directory: it cannot be written, or replaced.
        let path = dir.path().join("store.json");
        fs::create_dir(&path).unwrap();
        fs::write(path.join("inside"), b"untouched").unwrap();
        assert!(write_replacing(&path, b"new").is_err());
        assert_eq!(fs::read(path.join("inside")).unwrap(), b"untouched");
        assert_eq!(names(dir.path()), ["store.json"]);
    }

    /// The steps as they really are, but with a rename that is refused.
    fn rename_refused_as(rename: fn(&Path, &Path) -> io::Result<()>) -> Steps {
        Steps {
            rename,
            ..Steps::REAL
        }
    }

    #[test]
    fn a_file_that_cannot_be_replaced_is_written_in_place() {
        // A single file mounted into a container answers a rename over it
        // with "resource busy". Saving such a file worked before saves went
        // by rename, and must still work.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.json");
        for refused in [
            rename_refused_as(|_, _| Err(io::Error::from(io::ErrorKind::ResourceBusy))),
            rename_refused_as(|_, _| Err(io::Error::from(io::ErrorKind::PermissionDenied))),
        ] {
            fs::write(&path, b"old").unwrap();
            replace(&path, b"new contents", Otherwise::WriteInPlace, &refused).unwrap();
            assert_eq!(fs::read(&path).unwrap(), b"new contents");
            assert_eq!(names(dir.path()), ["store.json"], "scratch removed");
        }

        // Any other failure is reported, and the old file is left alone.
        fs::write(&path, b"old").unwrap();
        let full = rename_refused_as(|_, _| Err(io::Error::from(io::ErrorKind::StorageFull)));
        let err = replace(&path, b"new contents", Otherwise::WriteInPlace, &full).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::StorageFull);
        assert_eq!(fs::read(&path).unwrap(), b"old");
        assert_eq!(names(dir.path()), ["store.json"]);

        // A file that does not exist yet is not created by the fallback: the
        // refusal is reported.
        let missing = dir.path().join("missing.json");
        let busy = rename_refused_as(|_, _| Err(io::Error::from(io::ErrorKind::ResourceBusy)));
        assert!(replace(&missing, b"x", Otherwise::WriteInPlace, &busy).is_err());
        assert!(!missing.exists());
    }

    #[test]
    fn the_strict_form_never_writes_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ledger.json");
        fs::write(&path, b"old").unwrap();
        let busy = rename_refused_as(|_, _| Err(io::Error::from(io::ErrorKind::ResourceBusy)));
        let err = replace(&path, b"new contents", Otherwise::Fail, &busy).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ResourceBusy);
        assert_eq!(fs::read(&path).unwrap(), b"old");
        assert_eq!(names(dir.path()), ["ledger.json"]);

        // And where nothing is refused, it replaces the file like the other.
        write_replacing_or_fail(&path, b"new contents").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new contents");
        assert_eq!(names(dir.path()), ["ledger.json"]);
    }

    /// Whether this process can write the file where it is: what an in-place
    /// save needed. Permission bits do not bind a privileged process.
    fn can_write(path: &Path) -> bool {
        OpenOptions::new().write(true).open(path).is_ok()
    }

    /// Saving over `path`, which holds `frozen`, must do what an in-place
    /// save would have done: fail and leave everything as it is if this
    /// process may not write the file, and save it otherwise.
    fn assert_saves_only_if_writable(path: &Path) {
        let dir = path.parent().unwrap();
        let before = fs::metadata(path).unwrap().permissions();
        if can_write(path) {
            write_replacing(path, b"new").unwrap();
            assert_eq!(fs::read(path).unwrap(), b"new");
        } else {
            let err = write_replacing(path, b"new").unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::PermissionDenied, "{err}");
            assert_eq!(fs::read(path).unwrap(), b"frozen");
        }
        assert_eq!(fs::metadata(path).unwrap().permissions(), before);
        assert_eq!(names(dir), ["store.json"]);
    }

    #[test]
    fn a_read_only_file_is_refused_and_left_as_it_was() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.json");
        fs::write(&path, b"frozen").unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&path, permissions).unwrap();
        assert_saves_only_if_writable(&path);
    }

    #[cfg(unix)]
    #[test]
    fn a_write_bit_that_is_someone_elses_does_not_allow_the_save() {
        use std::os::unix::fs::PermissionsExt;
        // Writable by its group, not by its owner, which this process is:
        // the file is not "read-only", and this process still may not write
        // it. It may create files beside it, so a rename alone would go
        // through.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.json");
        fs::write(&path, b"frozen").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o464)).unwrap();
        assert!(!fs::metadata(&path).unwrap().permissions().readonly());
        assert_saves_only_if_writable(&path);
    }

    #[test]
    fn a_scratch_name_that_is_taken_is_stepped_over_and_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("store.json");
        let leftover = |n: u32| {
            dir.path()
                .join(format!("store.json.tmp-{}-{n}", std::process::id()))
        };
        fs::write(leftover(0), b"left by a save that was cut short").unwrap();
        fs::write(leftover(1), b"and another").unwrap();

        let seq = AtomicU64::new(0);
        let (scratch, _file) = create_scratch(&target, true, &seq).unwrap();
        assert_eq!(scratch, leftover(2));
        assert_eq!(
            fs::read(leftover(0)).unwrap(),
            b"left by a save that was cut short"
        );
        assert_eq!(fs::read(leftover(1)).unwrap(), b"and another");

        // It gives up in the end, and says why.
        let crowded = dir.path().join("crowded.json");
        for n in 0..=SCRATCH_TRIES {
            let name = format!("crowded.json.tmp-{}-{n}", std::process::id());
            fs::write(dir.path().join(name), b"").unwrap();
        }
        let err = create_scratch(&crowded, true, &AtomicU64::new(0)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
    }

    #[test]
    fn a_bare_file_name_is_in_the_current_directory() {
        assert_eq!(directory_of(Path::new("store.json")), Some(Path::new(".")));
        assert_eq!(
            directory_of(Path::new("data/store.json")),
            Some(Path::new("data"))
        );
        assert_eq!(
            directory_of(Path::new("/data/store.json")),
            Some(Path::new("/data"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_scratch_file_is_never_more_readable_than_the_old_file() {
        use std::os::unix::fs::PermissionsExt;
        let mode = |path: &Path| fs::metadata(path).unwrap().permissions().mode() & 0o777;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.json");

        // As created, before anything is in it: its owner's alone, whatever
        // the process default.
        let (scratch, file) = create_scratch(&path, true, &AtomicU64::new(0)).unwrap();
        assert_eq!(mode(&scratch), 0o600);
        drop(file);
        fs::remove_file(&scratch).unwrap();

        // With the contents in it, at the moment it is renamed: the old
        // file's permissions exactly.
        fs::write(&path, b"private").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let looking = rename_refused_as(|from, to| {
            let mode = fs::metadata(from).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "the scratch file, holding the contents");
            assert_eq!(fs::read(from).unwrap(), b"still private");
            fs::rename(from, to)
        });
        replace(&path, b"still private", Otherwise::WriteInPlace, &looking).unwrap();
        assert_eq!(mode(&path), 0o600);
        assert_eq!(fs::read(&path).unwrap(), b"still private");
    }

    /// Give the file to a group it is not in now, and say which: one of this
    /// process's own groups, or for a privileged process any group. `None`
    /// if this process has no such group to give it to.
    #[cfg(unix)]
    fn give_to_another_group(path: &Path) -> Option<u32> {
        use std::os::unix::fs::MetadataExt;
        let now = fs::metadata(path).unwrap().gid();
        let groups = std::process::Command::new("id").arg("-G").output().ok()?;
        String::from_utf8_lossy(&groups.stdout)
            .split_whitespace()
            .filter_map(|group| group.parse::<u32>().ok())
            .chain([54321])
            .find(|group| {
                *group != now && std::os::unix::fs::chown(path, None, Some(*group)).is_ok()
            })
    }

    #[cfg(unix)]
    #[test]
    fn the_new_file_keeps_the_old_files_group() {
        use std::os::unix::fs::MetadataExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.json");
        fs::write(&path, b"old").unwrap();
        let Some(group) = give_to_another_group(&path) else {
            eprintln!("not checked: this process has only one group to give a file to");
            return;
        };
        let kept = dir.path().join("as-it-was");
        fs::hard_link(&path, &kept).unwrap();

        write_replacing(&path, b"new").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        assert_eq!(fs::metadata(&path).unwrap().gid(), group);
        assert_eq!(
            fs::read(&kept).unwrap(),
            b"old",
            "replaced, not written into"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_file_whose_group_cannot_be_kept_is_written_in_place() {
        use std::os::unix::fs::MetadataExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.json");
        fs::write(&path, b"old").unwrap();
        let Some(group) = give_to_another_group(&path) else {
            eprintln!("not checked: this process has only one group to give a file to");
            return;
        };
        let kept = dir.path().join("as-it-was");
        fs::hard_link(&path, &kept).unwrap();
        // As for a process that is not in the file's group.
        let not_allowed = Steps {
            give: |_, _, _| Err(io::Error::from(io::ErrorKind::PermissionDenied)),
            ..Steps::REAL
        };

        replace(&path, b"new", Otherwise::WriteInPlace, &not_allowed).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        assert_eq!(fs::metadata(&path).unwrap().gid(), group);
        assert_eq!(
            fs::read(&kept).unwrap(),
            b"new",
            "written into, not replaced"
        );
        assert_eq!(names(dir.path()), ["as-it-was", "store.json"]);

        // The strict form replaces it all the same, and the new file is this
        // process's own.
        replace(&path, b"newer", Otherwise::Fail, &not_allowed).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"newer");
        assert_ne!(fs::metadata(&path).unwrap().gid(), group);
        assert_eq!(
            fs::read(&kept).unwrap(),
            b"new",
            "replaced, not written into"
        );
        assert_eq!(names(dir.path()), ["as-it-was", "store.json"]);
    }

    #[cfg(unix)]
    #[test]
    fn the_new_file_keeps_the_old_files_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.json");
        fs::write(&path, b"private").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        write_replacing(&path, b"still private").unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(fs::read(&path).unwrap(), b"still private");
    }

    #[cfg(unix)]
    #[test]
    fn a_link_to_the_file_stays_a_link_and_the_file_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real").join("store.json");
        fs::create_dir_all(real.parent().unwrap()).unwrap();
        fs::write(&real, b"old").unwrap();
        let link = dir.path().join("store.json");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        write_replacing(&link, b"new").unwrap();
        assert!(fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(&real).unwrap(), b"new");
        assert_eq!(fs::read(&link).unwrap(), b"new");
        assert_eq!(names(real.parent().unwrap()), ["store.json"]);
    }

    #[test]
    fn concurrent_writers_each_leave_a_complete_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.json");
        let payloads: Vec<Vec<u8>> = (0..8u8)
            .map(|t| vec![b'a' + t; 64 * 1024 + usize::from(t)])
            .collect();
        let writers: Vec<_> = payloads
            .iter()
            .cloned()
            .map(|payload| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        write_replacing(&path, &payload).unwrap();
                    }
                })
            })
            .collect();
        // A reader sees one writer's whole payload every time, never a mix
        // and never a fragment.
        let reader = {
            let path = path.clone();
            let payloads = payloads.clone();
            std::thread::spawn(move || {
                for _ in 0..200 {
                    if let Ok(seen) = fs::read(&path) {
                        assert!(
                            payloads.contains(&seen),
                            "read {} bytes that no writer wrote",
                            seen.len()
                        );
                    }
                }
            })
        };
        for writer in writers {
            writer.join().unwrap();
        }
        reader.join().unwrap();
        assert!(payloads.contains(&fs::read(&path).unwrap()));
        assert_eq!(names(dir.path()), ["store.json"]);
    }
}
