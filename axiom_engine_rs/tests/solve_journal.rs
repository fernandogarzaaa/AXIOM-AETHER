//! `axiom solve` must not leave a candidate fix in the user's file when it is
//! interrupted or killed.
//!
//! The repair loop tries a candidate by writing it over the source file and
//! running the verify command. These tests drive the real `axiom` binary,
//! stop it while a candidate is on disk, and check what the file holds
//! afterwards. Unix only: they send signals.
#![cfg(unix)]

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const ORIGINAL: &str = "is_adult() {\n  [ \"$1\" -gt 18 ]\n}\n";

/// The verify script: while `lib.sh` differs from the saved original (a
/// candidate is on disk) it says so and takes a long time, as a slow test
/// suite would.
const SLOW_TEST: &str = "#!/bin/sh\n\
    if ! cmp -s lib.sh lib.sh.orig; then touch candidate-on-disk; sleep 20; fi\n\
    . ./lib.sh\n\
    is_adult 18 || { echo \"test.sh:4: is_adult 18 should succeed\"; exit 1; }\n\
    is_adult 40 || { echo \"test.sh:5: is_adult 40 should succeed\"; exit 1; }\n\
    if is_adult 17; then echo \"test.sh:6: is_adult 17 should fail\"; exit 1; fi\n";

struct Project {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    home: PathBuf,
}

fn project() -> Project {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("project");
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(dir.join("lib.sh"), ORIGINAL).unwrap();
    std::fs::write(dir.join("lib.sh.orig"), ORIGINAL).unwrap();
    std::fs::write(dir.join("test.sh"), SLOW_TEST).unwrap();
    Project {
        _tmp: tmp,
        dir,
        home,
    }
}

/// `axiom solve -- <verify...>` in `dir`, with its own home directory (so its
/// journal, heal memory and patch store are private to the test) and no
/// inherited configuration.
fn axiom_solve(project: &Project, verify: &[&str]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_axiom"));
    cmd.arg("solve")
        .arg("--")
        .args(verify)
        .current_dir(&project.dir)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", &project.home)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

fn journal_files(project: &Project) -> Vec<PathBuf> {
    std::fs::read_dir(project.home.join(".axiom").join("journal"))
        .map(|d| d.flatten().map(|e| e.path()).collect())
        .unwrap_or_default()
}

/// Wait until the verify script reports a candidate on disk.
fn wait_for_candidate(project: &Project, child: &mut Child) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while !project.dir.join("candidate-on-disk").exists() {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("axiom exited ({status}) before writing any candidate");
        }
        assert!(Instant::now() < deadline, "no candidate was ever written");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_ne!(
        std::fs::read_to_string(project.dir.join("lib.sh")).unwrap(),
        ORIGINAL,
        "setup: a candidate is on disk"
    );
}

fn signal(child: &Child, signal: libc::c_int) {
    // SAFETY: kill(2) on the pid of a child this test spawned and has not
    // reaped; it reads no memory.
    let rc = unsafe { libc::kill(child.id() as libc::pid_t, signal) };
    assert_eq!(rc, 0, "kill failed");
}

fn wait_with_timeout(mut child: Child) -> std::process::ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("axiom did not exit after the signal");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn lib(project: &Project) -> String {
    std::fs::read_to_string(project.dir.join("lib.sh")).unwrap()
}

fn interrupted_solve_restores(sig: libc::c_int, expected_status: i32) {
    let project = project();
    let mut child = axiom_solve(&project, &["sh", "test.sh"]).spawn().unwrap();
    wait_for_candidate(&project, &mut child);

    signal(&child, sig);
    let status = wait_with_timeout(child);

    assert_eq!(
        lib(&project),
        ORIGINAL,
        "the file must hold its original content after the interrupt"
    );
    assert_eq!(
        status.code(),
        Some(expected_status),
        "axiom handles the signal and exits with the conventional status"
    );
    assert_eq!(
        journal_files(&project),
        Vec::<PathBuf>::new(),
        "a restored file leaves no journal entry behind"
    );
}

#[test]
fn ctrl_c_while_a_candidate_is_on_disk_restores_the_file() {
    interrupted_solve_restores(libc::SIGINT, 130);
}

#[test]
fn sigterm_while_a_candidate_is_on_disk_restores_the_file() {
    interrupted_solve_restores(libc::SIGTERM, 143);
}

#[test]
fn the_next_solve_restores_a_file_left_by_a_killed_one() {
    let project = project();
    let mut child = axiom_solve(&project, &["sh", "test.sh"]).spawn().unwrap();
    wait_for_candidate(&project, &mut child);

    // SIGKILL cannot be handled: the candidate stays in the file, and the
    // original stays in the journal.
    signal(&child, libc::SIGKILL);
    wait_with_timeout(child);
    assert_ne!(
        lib(&project),
        ORIGINAL,
        "the kill left the candidate on disk"
    );
    assert!(
        journal_files(&project)
            .iter()
            .any(|p| p.extension().is_some_and(|e| e == "orig")),
        "the original is still journaled: {:?}",
        journal_files(&project)
    );

    // Any later solve by the same user puts the file right first. This one
    // has nothing it can repair.
    let out = axiom_solve(&project, &["sh", "-c", "exit 3"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(lib(&project), ORIGINAL, "{stdout}");
    assert!(
        stdout.contains("restored") && stdout.contains("lib.sh"),
        "the user is told: {stdout}"
    );
    assert_eq!(journal_files(&project), Vec::<PathBuf>::new());
}

#[test]
fn recovery_does_not_overwrite_a_file_edited_after_the_kill() {
    let project = project();
    let mut child = axiom_solve(&project, &["sh", "test.sh"]).spawn().unwrap();
    wait_for_candidate(&project, &mut child);
    signal(&child, libc::SIGKILL);
    wait_with_timeout(child);

    // The user notices and fixes the file by hand before running axiom again.
    let by_hand = "is_adult() {\n  [ \"$1\" -ge 18 ]  # fixed by hand\n}\n";
    std::fs::write(project.dir.join("lib.sh"), by_hand).unwrap();

    let out = axiom_solve(&project, &["sh", "-c", "exit 3"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        lib(&project),
        by_hand,
        "work done after the kill is kept: {stdout}"
    );
    // The pre-solve content is kept, and the user is told where.
    let kept: Vec<PathBuf> = journal_files(&project);
    assert_eq!(kept.len(), 1, "{kept:?}");
    assert_eq!(std::fs::read_to_string(&kept[0]).unwrap(), ORIGINAL);
    assert!(
        stdout.contains(&kept[0].display().to_string()),
        "the saved original's path is reported: {stdout}"
    );
}
