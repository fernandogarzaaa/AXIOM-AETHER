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

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes the scratch files of concurrent writers in one process.
static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

/// Give `path` the contents `contents`, by rename.
///
/// * The directory of `path` is created if it is missing.
/// * If `path` is a symbolic link, the file it points at is replaced and the
///   link is left as it is. (A link that points nowhere is itself replaced
///   by the file.)
/// * The new file takes the old file's permissions. A new file gets the
///   process default.
/// * A file marked read-only is refused, as writing to it in place would have
///   been; a rename would otherwise replace it without complaint.
/// * On any failure the old file is untouched and the scratch file is
///   removed.
///
/// One exception, so that a save which worked before this function existed
/// still works: where the file cannot be replaced at all (it is a single
/// file mounted into a container, or it sits in a directory this process may
/// not create files in), it is written in place, as `std::fs::write` would.
/// That is decided by the kind of error alone, and never after a failed
/// write of the contents: a full disk does not lead to the old file being
/// emptied.
pub fn write_replacing(path: &Path, contents: &[u8]) -> io::Result<()> {
    write_replacing_with(path, contents, |from, to| fs::rename(from, to))
}

/// [`write_replacing`], with the rename step supplied, so that a rename the
/// platform refuses can be tested.
fn write_replacing_with(
    path: &Path,
    contents: &[u8],
    rename: impl FnOnce(&Path, &Path) -> io::Result<()>,
) -> io::Result<()> {
    // Follow links, so that it is the file that is replaced and not a link to
    // it. A path that does not exist yet is used as given.
    let target = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if let Some(parent) = target.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let existing = match fs::metadata(&target) {
        Ok(meta) => Some(meta.permissions()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => return Err(e),
    };
    if existing.as_ref().is_some_and(|p| p.readonly()) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("{} is read-only", target.display()),
        ));
    }

    let scratch = scratch_path(&target);
    let mut file = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&scratch)
    {
        Ok(file) => file,
        // No file may be created beside the target. Nothing has been
        // written anywhere yet.
        Err(e) if e.kind() == io::ErrorKind::PermissionDenied && existing.is_some() => {
            return fs::write(&target, contents);
        }
        Err(e) => return Err(e),
    };
    let written = (|| {
        file.write_all(contents)?;
        if let Some(permissions) = existing.clone() {
            file.set_permissions(permissions)?;
        }
        // The contents must be on disk before the name points at them.
        file.sync_all()
    })();
    drop(file);
    if let Err(e) = written {
        let _ = fs::remove_file(&scratch);
        return Err(e);
    }
    match rename(&scratch, &target) {
        Ok(()) => {
            sync_directory(&target);
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_file(&scratch);
            // The complete new contents were written and synced, so the disk
            // has room for them; the target itself cannot be replaced.
            let cannot_be_replaced = matches!(
                e.kind(),
                io::ErrorKind::ResourceBusy | io::ErrorKind::PermissionDenied
            );
            if cannot_be_replaced && existing.is_some() {
                fs::write(&target, contents)
            } else {
                Err(e)
            }
        }
    }
}

/// `<target>.tmp-<pid>-<n>`, beside the target so the rename stays within one
/// directory.
fn scratch_path(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_os_string();
    name.push(format!(
        ".tmp-{}-{}",
        std::process::id(),
        SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    PathBuf::from(name)
}

/// Ask for the rename itself to reach the disk. Best effort: the file is
/// already complete under one of its two names, and not every platform can
/// sync a directory.
fn sync_directory(target: &Path) {
    #[cfg(unix)]
    if let Some(parent) = target.parent().filter(|p| !p.as_os_str().is_empty()) {
        if let Ok(dir) = fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    #[cfg(not(unix))]
    let _ = target;
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
        // The target is a directory: the rename over it cannot succeed.
        let path = dir.path().join("store.json");
        fs::create_dir(&path).unwrap();
        fs::write(path.join("inside"), b"untouched").unwrap();
        assert!(write_replacing(&path, b"new").is_err());
        assert_eq!(fs::read(path.join("inside")).unwrap(), b"untouched");
        assert_eq!(names(dir.path()), ["store.json"]);
    }

    #[test]
    fn a_file_that_cannot_be_replaced_is_written_in_place() {
        // A single file mounted into a container answers a rename over it
        // with "resource busy". Saving such a file worked before saves went
        // by rename, and must still work.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.json");
        fs::write(&path, b"old").unwrap();
        for kind in [io::ErrorKind::ResourceBusy, io::ErrorKind::PermissionDenied] {
            write_replacing_with(&path, b"new contents", |_, _| Err(io::Error::from(kind)))
                .unwrap();
            assert_eq!(fs::read(&path).unwrap(), b"new contents");
            assert_eq!(names(dir.path()), ["store.json"], "scratch removed");
            fs::write(&path, b"old").unwrap();
        }

        // Any other failure is reported, and the old file is left alone.
        let err = write_replacing_with(&path, b"new contents", |_, _| {
            Err(io::Error::from(io::ErrorKind::StorageFull))
        })
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::StorageFull);
        assert_eq!(fs::read(&path).unwrap(), b"old");
        assert_eq!(names(dir.path()), ["store.json"]);

        // A file that does not exist yet is not created by the fallback: the
        // refusal is reported.
        let missing = dir.path().join("missing.json");
        assert!(write_replacing_with(&missing, b"x", |_, _| {
            Err(io::Error::from(io::ErrorKind::ResourceBusy))
        })
        .is_err());
        assert!(!missing.exists());
    }

    #[test]
    fn a_read_only_file_is_refused_and_left_as_it_was() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.json");
        fs::write(&path, b"frozen").unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&path, permissions).unwrap();

        let err = write_replacing(&path, b"new").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied, "{err}");
        assert_eq!(fs::read(&path).unwrap(), b"frozen");
        assert_eq!(names(dir.path()), ["store.json"]);
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
