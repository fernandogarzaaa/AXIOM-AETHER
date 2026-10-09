//! Component tests for the agent-driven task API. Each test owns its workspace;
//! verifiers use only the local shell, with no network or model dependencies.

use std::fs;
use std::path::{Path, PathBuf};

use axiom_engine::agent_task::{AgentTask, ProposeOutcome, TaskRegistry};
use axiom_engine::agentic::FileEdit;

fn task(files: &[PathBuf], verifier: &str) -> AgentTask {
    AgentTask::start(
        "test-task".into(),
        "repair the fixture".into(),
        verifier.into(),
        files.to_vec(),
        4,
        None,
    )
    .unwrap()
}

fn edit(path: &Path, content: &str) -> FileEdit {
    FileEdit {
        path: path.to_owned(),
        content: content.into(),
    }
}

fn assert_history(task: &AgentTask, outcomes: &[ProposeOutcome]) {
    assert_eq!(task.history().len(), outcomes.len());
    for (record, outcome) in task.history().iter().zip(outcomes) {
        assert_eq!(record.attempt, outcome.attempt);
        assert_eq!(record.passed, outcome.passed);
        assert_eq!(record.fingerprint, outcome.fingerprint);
        assert_eq!(record.output, outcome.output);
    }
}

#[test]
fn starts_with_empty_history_and_normalizes_zero_budget() {
    let task = AgentTask::start("id".into(), "goal".into(), "exit 0".into(), vec![], 0, None)
        .unwrap();
    assert_eq!(task.task_id, "id");
    assert_eq!(task.goal, "goal");
    assert_eq!(task.verify_cmd, "exit 0");
    assert!(task.files.is_empty());
    assert_eq!(task.max_attempts, 1);
    assert_eq!(task.attempt, 0);
    assert!(task.history().is_empty());
}

#[test]
fn empty_proposal_is_recorded_without_running_the_verifier() {
    let mut task = task(&[], "echo verifier-must-not-run");
    let outcome = task.propose(vec![]);
    assert!(!outcome.passed);
    assert_eq!(outcome.attempt, 1);
    assert!(outcome.fingerprint.is_empty());
    assert_eq!(outcome.output, "empty edit-set rejected");
    assert_history(&task, &[outcome]);
}

#[test]
fn passing_proposal_commits_all_files_and_records_the_outcome() {
    let dir = tempfile::tempdir().unwrap();
    let existing = dir.path().join("existing.txt");
    let created = dir.path().join("nested/new.txt");
    fs::write(&existing, "before").unwrap();
    let mut task = task(&[existing.clone(), created.clone()], "exit 0");

    let outcome = task.propose(vec![edit(&existing, "after"), edit(&created, "new")]);
    assert!(outcome.passed);
    assert_eq!(outcome.attempt, 1);
    assert!(!outcome.fingerprint.is_empty());
    assert_eq!(fs::read_to_string(&existing).unwrap(), "after");
    assert_eq!(fs::read_to_string(&created).unwrap(), "new");
    assert_history(&task, &[outcome]);
}

#[test]
fn rejection_restores_binary_and_empty_files_and_removes_created_files() {
    let dir = tempfile::tempdir().unwrap();
    let binary = dir.path().join("binary.dat");
    let empty = dir.path().join("empty.txt");
    let created = dir.path().join("new.txt");
    let original = [0xff, 0x00, 0x80, b'\n'];
    fs::write(&binary, original).unwrap();
    fs::write(&empty, []).unwrap();
    let mut task = task(&[binary.clone(), empty.clone(), created.clone()], "exit 7");

    let outcome = task.propose(vec![
        edit(&binary, "replacement"),
        edit(&empty, "nonempty"),
        edit(&created, "temporary"),
    ]);
    assert!(!outcome.passed);
    assert_eq!(fs::read(&binary).unwrap(), original);
    assert_eq!(fs::read(&empty).unwrap(), Vec::<u8>::new());
    assert!(!created.exists());
    assert_history(&task, &[outcome]);
}

#[test]
fn failed_proposal_preserves_the_last_successful_proposal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("file.txt");
    fs::write(&path, "initial").unwrap();
    let mut task = task(&[path.clone()], "exit 0");
    let accepted = task.propose(vec![edit(&path, "accepted")]);
    assert!(accepted.passed);
    task.verify_cmd = "exit 1".into();
    let rejected = task.propose(vec![edit(&path, "rejected")]);

    assert!(!rejected.passed);
    assert_eq!(rejected.attempt, 2);
    assert_ne!(accepted.fingerprint, rejected.fingerprint);
    assert_eq!(fs::read_to_string(&path).unwrap(), "accepted");
    assert_history(&task, &[accepted, rejected]);
}

#[test]
fn equivalent_rejected_edits_are_deduplicated_even_when_reordered() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.txt");
    let b = dir.path().join("b.txt");
    fs::write(&a, "a0").unwrap();
    fs::write(&b, "b0").unwrap();
    let mut task = task(&[a.clone(), b.clone()], "exit 1");
    let first = task.propose(vec![edit(&a, "a1"), edit(&b, "b1")]);
    assert!(!first.passed);

    // A verifier run would now pass and leave changed bytes behind.
    task.verify_cmd = "exit 0".into();
    let duplicate = task.propose(vec![edit(&b, "b1"), edit(&a, "a1")]);
    assert!(!duplicate.passed);
    assert_eq!(duplicate.attempt, 2);
    assert_eq!(duplicate.fingerprint, first.fingerprint);
    assert!(duplicate.output.contains("already rejected"));
    assert_eq!(fs::read_to_string(&a).unwrap(), "a0");
    assert_eq!(fs::read_to_string(&b).unwrap(), "b0");

    let changed = task.propose(vec![edit(&a, "a2"), edit(&b, "b1")]);
    assert!(changed.passed);
    assert_ne!(changed.fingerprint, first.fingerprint);
    assert_history(&task, &[first, duplicate, changed]);
}

#[test]
fn successful_proposals_are_not_added_to_rejection_memory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("file.txt");
    let mut task = task(&[path.clone()], "exit 0");
    let first = task.propose(vec![edit(&path, "accepted")]);
    let second = task.propose(vec![edit(&path, "accepted")]);
    assert!(first.passed && second.passed);
    assert_eq!(second.attempt, 2);
    assert_eq!(first.fingerprint, second.fingerprint);
    assert_history(&task, &[first, second]);
}

#[test]
fn apply_error_rolls_back_prior_writes_and_allows_retry() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("file.txt");
    let blocked = dir.path().join("directory");
    fs::write(&path, "original").unwrap();
    fs::create_dir(&blocked).unwrap();
    let mut task = task(&[path.clone(), blocked.clone()], "echo verifier-ran");
    let edits = vec![edit(&path, "changed"), edit(&blocked, "new file")];
    let failed = task.propose(edits.clone());
    assert!(!failed.passed);
    assert!(failed.output.starts_with("apply failed:"));
    assert!(!failed.output.contains("verifier-ran"));
    assert_eq!(fs::read_to_string(&path).unwrap(), "original");
    assert!(blocked.is_dir());

    fs::remove_dir(&blocked).unwrap();
    let retried = task.propose(edits);
    assert!(retried.passed);
    assert_eq!(retried.fingerprint, failed.fingerprint);
    assert_eq!(fs::read_to_string(&path).unwrap(), "changed");
    assert_eq!(fs::read_to_string(&blocked).unwrap(), "new file");
    assert_history(&task, &[failed, retried]);
}

#[test]
fn abort_restores_initial_snapshot_after_multiple_successes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("binary.dat");
    let created = dir.path().join("created.txt");
    let original = [0xff, 0x00, b'a'];
    fs::write(&path, original).unwrap();
    let mut task = task(&[path.clone(), created.clone()], "exit 0");
    assert!(task.propose(vec![edit(&path, "v1")]).passed);
    assert!(task
        .propose(vec![edit(&path, "v2"), edit(&created, "new")])
        .passed);

    task.finish(false).unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
    assert!(!created.exists());
    assert_eq!(task.history().len(), 2);
}

#[test]
fn finish_is_idempotent_and_rejects_further_proposals() {
    for commit in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        fs::write(&path, "initial").unwrap();
        let mut task = task(&[path.clone()], "exit 0");
        let accepted = task.propose(vec![edit(&path, "accepted")]);
        assert!(accepted.passed);
        task.finish(commit).unwrap();
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            if commit { "accepted" } else { "initial" }
        );

        // A second finish must not undo external work, even with the opposite flag.
        fs::write(&path, "external change").unwrap();
        task.finish(!commit).unwrap();
        let rejected = task.propose(vec![edit(&path, "too late")]);
        assert!(!rejected.passed);
        assert_eq!(rejected.output, "task already finished");
        assert_eq!(rejected.attempt, 1);
        assert!(rejected.fingerprint.is_empty());
        assert_eq!(fs::read_to_string(&path).unwrap(), "external change");
        assert_history(&task, &[accepted]);
    }
}

#[test]
fn attempt_budget_prevents_additional_file_changes() {
    for budget in [0, 1, 2] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        fs::write(&path, "initial").unwrap();
        let mut task = AgentTask::start(
            "limited".into(),
            "repair".into(),
            "exit 1".into(),
            vec![path.clone()],
            budget,
            None,
        )
        .unwrap();
        for attempt in 0..budget.max(1) {
            assert!(!task.propose(vec![edit(&path, &format!("bad-{attempt}"))]).passed);
        }
        task.verify_cmd = "exit 0".into();
        let over_budget = task.propose(vec![edit(&path, "must not apply")]);
        assert!(!over_budget.passed, "exhausted budget {budget} must reject edits");
        assert_eq!(fs::read_to_string(&path).unwrap(), "initial");
    }
}

#[test]
fn registry_keeps_tasks_independent_and_removal_returns_the_task() {
    let registry = TaskRegistry::new();
    let shared = registry.clone();
    let first = task(&[], "exit 0");
    let mut second = task(&[], "exit 0");
    second.task_id = "other".into();
    registry.insert(first);
    registry.insert(second);
    assert!(shared.with_task("missing", |_| panic!("must not run")).is_none());
    assert!(shared.remove("missing").is_none());
    shared.with_task("test-task", |t| t.propose(vec![])).unwrap();
    assert_eq!(registry.with_task("test-task", |t| t.history().len()), Some(1));
    assert_eq!(registry.with_task("other", |t| t.history().len()), Some(0));
    let removed = shared.remove("test-task").unwrap();
    assert_eq!(removed.task_id, "test-task");
    assert_eq!(removed.history().len(), 1);
    assert!(registry.with_task("test-task", |_| ()).is_none());
    assert!(registry.remove("test-task").is_none());
    assert!(registry.with_task("other", |_| ()).is_some());
}

// Exact output and file-dependent verifiers use POSIX shell builtins. The
// lifecycle tests above also run on Windows using the implementation's cmd path.
#[cfg(unix)]
mod verifier {
    use super::*;

    fn shell_quote(path: &Path) -> String {
        format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"))
    }

    #[test]
    fn verifier_observes_proposed_bytes_before_they_are_committed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file with 'quotes'.txt");
        fs::write(&path, "old").unwrap();
        let command = format!("test \"$(cat {})\" = expected", shell_quote(&path));
        let mut task = task(&[path.clone()], &command);
        assert!(!task.propose(vec![edit(&path, "incorrect")]).passed);
        assert_eq!(fs::read_to_string(&path).unwrap(), "old");
        assert!(task.propose(vec![edit(&path, "expected")]).passed);
        assert_eq!(fs::read_to_string(&path).unwrap(), "expected");
    }

    #[test]
    fn captures_stdout_stderr_and_exit_status() {
        for (command, expected, passed) in [
            ("printf stdout", "stdout", true),
            ("printf stderr >&2", "\n--- stderr ---\nstderr", true),
            (
                "printf stdout; printf stderr >&2; exit 3",
                "stdout\n--- stderr ---\nstderr",
                false,
            ),
            ("printf '\\377'", "\u{fffd}", true),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("file.txt");
            let mut task = task(&[path.clone()], command);
            let outcome = task.propose(vec![edit(&path, "new")]);
            assert_eq!(outcome.passed, passed, "{command}");
            assert_eq!(outcome.output, expected, "{command}");
            assert_eq!(path.exists(), passed);
            assert_history(&task, &[outcome]);
        }
    }

    #[test]
    fn truncates_only_output_exceeding_the_byte_limit() {
        for length in [7999, 8000, 8001] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("file.txt");
            let command = format!("printf '%s' '{}'", "x".repeat(length));
            let mut task = task(&[path.clone()], &command);
            let outcome = task.propose(vec![edit(&path, "new")]);
            assert!(outcome.passed);
            let mut expected = "x".repeat(length.min(8000));
            if length > 8000 {
                expected.push_str("\n...[truncated]");
            }
            assert_eq!(outcome.output, expected);
            assert_history(&task, &[outcome]);
        }
    }

    #[test]
    fn truncating_multibyte_output_does_not_panic_or_split_a_character() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        // The 8,000-byte limit falls in the middle of this three-byte character.
        let output = format!("{}\u{20ac}tail", "x".repeat(7999));
        let mut task = task(&[path.clone()], &format!("printf '%s' '{output}'"));
        let outcome = task.propose(vec![edit(&path, "accepted")]);
        assert!(outcome.passed);
        let prefix = outcome.output.strip_suffix("\n...[truncated]").unwrap();
        assert!(prefix.len() <= 8000);
        assert!(output.starts_with(prefix));
        assert!(prefix.len() >= 7999);
        assert_eq!(fs::read_to_string(&path).unwrap(), "accepted");
        assert_history(&task, &[outcome]);
    }
}
