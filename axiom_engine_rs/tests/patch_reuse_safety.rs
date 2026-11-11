//! A stored fix must never overwrite work it was not made against.
//!
//! `axiom solve` records each verified fix in the patch store and, on a later
//! failure, tries stored fixes before asking an LLM. A stored fix used to be
//! the whole file, keyed only by the verify command and the file's name, so a
//! later solve could replace a file that had since gained uncommitted work, or
//! a different project's file that merely shared a name and a test command,
//! as long as the tests then passed.
//!
//! The contract these tests pin: a stored fix is applied only to the exact
//! file bytes it was recorded against, in the repository it was recorded in.

use std::path::{Path, PathBuf};

use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::solve::{solve, SolveOptions, SolveReport};
use candle_core::Device;

fn tiny_pipeline() -> InferencePipeline {
    let cfg = AxiomConfig {
        d_model: 16,
        n_layers: 2,
        vocab_size: 64,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    InferencePipeline::new(cfg, Device::Cpu).expect("tiny pipeline")
}

fn unique_tmp(tag: &str) -> PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("axiom_patch_reuse_{tag}_{n}"))
}

/// The verify command every project below uses: `sh test.sh`, run in the
/// project directory. It sources `calc.sh` and checks one average. The last
/// command's status is the script's status, so there is no `exit` to rewrite.
const TEST_SH: &str = ". ./calc.sh\n[ \"$(average 2 4 6)\" = \"4\" ]\n";

/// A bug the deterministic mutation phase can repair (drop the `+ 1`).
const BUGGY_FIXABLE: &str = "average() {\n  echo $(( ($1 + $2 + $3) / (3 + 1) ))\n}\n";

/// The same file some time later: a different bug that no single-token
/// mutation repairs (the divisor is nowhere near 3), plus a new function the
/// test does not cover yet. This is the user's uncommitted work.
const NEWER_WORK: &str = "average() {\n  echo $(( ($1 + $2 + $3) / 30 ))\n}\n\n\
# new, uncommitted, not covered by the test yet\nmedian() {\n  echo \"$2\"\n}\n";

/// A different project that happens to have a `calc.sh` and the same test
/// command. Also not repairable by a single mutation.
const OTHER_PROJECT: &str = "# a different project entirely\nTAX=20\n\n\
average() {\n  echo $(( ($1 + $2 + $3) / 30 ))\n}\n\n\
with_tax() {\n  echo $(( $1 + $1 * TAX / 100 ))\n}\n";

fn write_project(dir: &Path, calc: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("calc.sh"), calc).unwrap();
    std::fs::write(dir.join("test.sh"), TEST_SH).unwrap();
}

/// Keeps a solve away from any LLM backend the developer has configured.
///
/// `solve` asks `AXIOM_BACKEND` for a repair backend once its deterministic
/// phases and the patch store have failed. These tests are about the patch
/// store alone: with a live backend selected, the cases that expect a file to
/// stay untouched would send it to a real API and fail whenever the model
/// repaired it. The variable is process-wide, so the guard also serializes the
/// solves in this file; it puts the variable back when dropped.
struct OfflineSolve {
    saved: Option<std::ffi::OsString>,
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl OfflineSolve {
    fn enter() -> Self {
        static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let saved = std::env::var_os("AXIOM_BACKEND");
        std::env::remove_var("AXIOM_BACKEND");
        Self {
            saved,
            _serial: serial,
        }
    }
}

impl Drop for OfflineSolve {
    fn drop(&mut self) {
        if let Some(value) = self.saved.take() {
            std::env::set_var("AXIOM_BACKEND", value);
        }
    }
}

/// Run `axiom solve -- sh test.sh` in `project`, with an isolated patch store
/// and heal memory under `state`, and no LLM backend.
fn solve_in(project: &Path, state: &Path) -> SolveReport {
    let _offline = OfflineSolve::enter();
    let opts = SolveOptions {
        max_rounds: 1,
        max_restarts: 1,
        heal_memory_path: Some(state.join("heal_memory.json")),
        anchor: Some(project.to_path_buf()),
        source_path: None,
        patch_memory_path: Some(state.join("axiom_patch_memory.json")),
    };
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || solve(&tiny_pipeline(), "sh", &["test.sh".to_string()], &opts).unwrap())
        .unwrap()
        .join()
        .unwrap()
}

/// First solve: repair [`BUGGY_FIXABLE`] and record the fix in the store.
fn solve_and_record_first_fix(project: &Path, state: &Path) {
    write_project(project, BUGGY_FIXABLE);
    let first = solve_in(project, state);
    assert!(first.solved, "setup: the first bug must be repaired");
    assert!(
        state.join("axiom_patch_memory.json").exists(),
        "setup: the verified fix must be recorded in the patch store"
    );
}

#[test]
fn a_stored_fix_does_not_overwrite_newer_uncommitted_work() {
    let base = unique_tmp("newer");
    let project = base.join("project");
    let state = base.join("state");
    std::fs::create_dir_all(&state).unwrap();
    solve_and_record_first_fix(&project, &state);

    // The user keeps working: a new function, and a new bug.
    std::fs::write(project.join("calc.sh"), NEWER_WORK).unwrap();
    let second = solve_in(&project, &state);

    let after = std::fs::read_to_string(project.join("calc.sh")).unwrap();
    assert_eq!(
        after, NEWER_WORK,
        "the stored fix was recorded against an older version of this file; \
         it must not be written over the newer one"
    );
    assert!(
        !second.fleet_patched,
        "no stored fix applies to a file that changed since it was recorded"
    );
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_stored_fix_does_not_apply_to_another_projects_same_named_file() {
    let base = unique_tmp("other");
    let project = base.join("project");
    let other = base.join("other_project");
    let state = base.join("state");
    std::fs::create_dir_all(&state).unwrap();
    solve_and_record_first_fix(&project, &state);

    // A different project: same file name, same verify command, same store.
    write_project(&other, OTHER_PROJECT);
    let report = solve_in(&other, &state);

    let after = std::fs::read_to_string(other.join("calc.sh")).unwrap();
    assert_eq!(
        after, OTHER_PROJECT,
        "a fix recorded in one project must not replace another project's file"
    );
    assert!(!report.fleet_patched);
    let _ = std::fs::remove_dir_all(&base);
}

/// The single-token mutation phase cannot repair `/ 30`; this is the fix.
const NEWER_WORK_FIXED: &str = "average() {\n  echo $(( ($1 + $2 + $3) / 3 ))\n}\n\n\
# new, uncommitted, not covered by the test yet\nmedian() {\n  echo \"$2\"\n}\n";

#[test]
fn a_stored_fix_is_reused_on_the_exact_file_it_was_recorded_against() {
    use axiom_engine::heal_memory::fingerprint;
    use axiom_engine::patch_memory::{repo_scope, PatchMemory};

    let base = unique_tmp("reuse");
    let project = base.join("project");
    let state = base.join("state");
    std::fs::create_dir_all(&state).unwrap();
    write_project(&project, NEWER_WORK);

    // A fix for exactly this file state is already in the store (as one
    // verified earlier by an LLM round, or merged from a fleet peer).
    let calc = project.join("calc.sh");
    let scope = repo_scope(&calc, Some(&project)).expect("the project directory is the scope");
    let store_path = state.join("axiom_patch_memory.json");
    let mut store = PatchMemory::new();
    store.record_verified(
        &fingerprint("sh", &["test.sh".to_string()]),
        &scope,
        NEWER_WORK,
        NEWER_WORK_FIXED,
    );
    store.save(&store_path).unwrap();
    // Only the changed line is stored, not the file.
    let stored = std::fs::read_to_string(&store_path).unwrap();
    assert!(stored.contains("/ 3 ))"), "{stored}");
    assert!(
        !stored.contains("median"),
        "unchanged lines are not stored: {stored}"
    );

    let report = solve_in(&project, &state);
    assert!(report.solved);
    assert!(report.fleet_patched, "the fix came from the store");
    assert!(
        report.source_patched,
        "a solve that changed a source file must report source_patched"
    );
    assert_eq!(
        std::fs::read_to_string(&calc).unwrap(),
        NEWER_WORK_FIXED,
        "the stored diff reproduces the recorded post-image, new function included"
    );

    // The same store, pointed at the same content in a different directory,
    // does nothing: it is a different repository.
    let elsewhere = base.join("elsewhere");
    write_project(&elsewhere, NEWER_WORK);
    let report = solve_in(&elsewhere, &state);
    assert!(!report.fleet_patched);
    assert_eq!(
        std::fs::read_to_string(elsewhere.join("calc.sh")).unwrap(),
        NEWER_WORK
    );
    let _ = std::fs::remove_dir_all(&base);
}
