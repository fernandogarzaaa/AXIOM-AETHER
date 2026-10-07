//! Agent-driven task loop — AXIOM's verifier-gated autonomy without an LLM API key.
//!
//! The [`agentic_loop`] in [`crate::agentic`] already decouples code generation
//! (the [`Proposer`] trait) from verification (the verifier closure) and rollback
//! ([`Transaction`]). This module inverts the driving relationship: instead of
//! the axiom binary calling out to an LLM API, an external agent (which *is* a
//! language model) drives the loop by proposing edit-sets through these primitives.
//!
//! The intended use is via the `axiom-agent-task` binary, which exposes these
//! as JSON-RPC tools over stdio:
//!
//! - `task_start {goal, verify_cmd, files[]}` → `{task_id}`
//! - `task_propose {task_id, edits[{path, content}]}` → `{passed, output, attempt}`
//! - `task_history {task_id}` → `{attempts[{attempt, fingerprint, passed, output}]}`  
//! - `task_finish {task_id, commit: bool}` → `{committed}`
//!
//! Safety properties (inherited from [`crate::agentic`]):
//! - Every proposal is applied as an all-or-nothing [`Transaction`].
//! - A rejected proposal is rolled back byte-for-byte.
//! - Identical failed edit-sets are never re-applied (deduplicated by fingerprint).
//! - The agent sees verifier output after each rejection to guide the next attempt.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::agentic::{AttemptMemory, EditSet, FileEdit, Transaction};

/// A single recorded attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptRecord {
    pub attempt: usize,
    pub fingerprint: String,
    pub passed: bool,
    /// Verifier stdout+stderr (truncated).
    pub output: String,
}

/// An active agent-driven task.
pub struct AgentTask {
    pub task_id: String,
    pub goal: String,
    pub verify_cmd: String,
    pub files: Vec<PathBuf>,
    pub max_attempts: usize,
    pub attempt: usize,
    memory: AttemptMemory,
    history: Vec<AttemptRecord>,
    /// Snapshot of original file contents for full abort.
    originals: HashMap<PathBuf, Option<Vec<u8>>>,
    finished: bool,
}

impl AgentTask {
    /// Start a task and snapshot `files` for a later `finish(false)`.
    ///
    /// `files` is the allowlist: only these paths can be edited via `propose`.
    /// They are also snapshotted for restoration on `finish(false)`.
    /// Snapshot read errors are treated as absent files, so this always returns
    /// `Ok`. `max_attempts` is stored with a minimum of one and enforced in `propose`.
    /// `verify_cmd` is a shell command run in the process's working directory.
    pub fn start(
        task_id: String,
        goal: String,
        verify_cmd: String,
        files: Vec<PathBuf>,
        max_attempts: usize,
    ) -> std::io::Result<Self> {
        let mut originals = HashMap::new();
        for f in &files {
            originals.insert(f.clone(), std::fs::read(f).ok());
        }
        Ok(Self {
            task_id,
            goal,
            verify_cmd,
            files,
            max_attempts: max_attempts.max(1),
            attempt: 0,
            memory: AttemptMemory::new(),
            history: Vec::new(),
            originals,
            finished: false,
        })
    }

    /// Apply full replacement file contents and keep them if the verifier passes.
    ///
    /// Returns the attempt number, edit fingerprint, pass status, and verifier
    /// output or rejection reason. Empty edits and edit-sets previously rejected
    /// by the verifier for this goal are recorded without applying them. Each
    /// call counts as an attempt unless the task is finished; finished tasks
    /// return a rejection without changing history.
    ///
    /// Apply and verifier execution errors become failed outcomes. Verifier
    /// failures trigger best-effort rollback of edited paths; restoration errors
    /// are ignored. Apply errors can leave partial changes if directory creation
    /// fails. Edits are restricted to the `files` allowlist from task start.
    ///
    /// # Panics
    ///
    /// Output is truncated at a UTF-8 character boundary to avoid panics.
    pub fn propose(&mut self, edits: Vec<FileEdit>) -> ProposeOutcome {
        if self.finished {
            return ProposeOutcome {
                passed: false,
                attempt: self.attempt,
                output: "task already finished".to_string(),
                fingerprint: String::new(),
            };
        }
        if self.attempt >= self.max_attempts {
            return ProposeOutcome {
                passed: false,
                attempt: self.attempt,
                output: format!(
                    "max attempts ({}) exceeded; task finished",
                    self.max_attempts
                ),
                fingerprint: String::new(),
            };
        }
        self.attempt += 1;
        let attempt_no = self.attempt;

        // Enforce the file allowlist: edits are restricted to paths
        // declared in `files` at task start. This prevents the agent
        // from modifying files outside the task scope.
        for e in &edits {
            if !self.originals.contains_key(&e.path) {
                return self.record(
                    attempt_no,
                    String::new(),
                    false,
                    format!(
                        "path '{}' not in task file allowlist;                          declare it in `files` at task_start to edit it",
                        e.path.display()
                    ),
                );
            }
        }

        let mut edit_set = EditSet::new();
        for e in edits {
            edit_set = edit_set.with(e.path, e.content);
        }

        if edit_set.is_empty() {
            return self.record(attempt_no, String::new(), false, "empty edit-set rejected".into());
        }

        let fingerprint = edit_set.fingerprint();
        if self.memory.was_rejected(&self.goal, &edit_set) {
            return self.record(
                attempt_no,
                fingerprint,
                false,
                "identical edit-set already rejected; not re-applied".into(),
            );
        }

        // Apply as all-or-nothing transaction.
        let mut tx = match Transaction::apply(&edit_set) {
            Ok(tx) => tx,
            Err(e) => {
                return self.record(attempt_no, fingerprint, false, format!("apply failed: {e}"));
            }
        };

        // Run the verifier.
        let (passed, output) = self.run_verifier();

        if passed {
            tx.commit();
            // Note: `originals` keeps the pre-task snapshot so that
            // `finish(commit=false)` restores the true initial state,
            // not the last-committed state.
            self.record(attempt_no, fingerprint, true, output)
        } else {
            tx.rollback();
            self.memory.record_rejected(&self.goal, &edit_set);
            self.record(attempt_no, fingerprint, false, output)
        }
    }

    /// Return recorded attempts in order, including empty and duplicate proposals.
    /// Calls made after the task is finished are not recorded.
    pub fn history(&self) -> &[AttemptRecord] {
        &self.history
    }

    /// Mark the task finished, keeping current edits when `commit` is true.
    ///
    /// When false, attempt to restore only the paths snapshotted at task start,
    /// removing those whose initial read failed. Write and removal errors are
    /// ignored. Repeated calls have no effect, even with a different `commit`.
    pub fn finish(&mut self, commit: bool) {
        if self.finished {
            return;
        }
        self.finished = true;
        if !commit {
            for (path, original) in &self.originals {
                match original {
                    Some(bytes) => {
                        let _ = std::fs::write(path, bytes);
                    }
                    None => {
                        let _ = std::fs::remove_file(path);
                    }
                }
            }
        }
    }

    /// Run `verify_cmd` through the platform shell in the current working directory.
    ///
    /// Return exit success and lossily decoded stdout followed by labeled stderr.
    /// Output over 8000 bytes is truncated to that length and given a truncation
    /// marker. Process execution errors return false with an error message.
    ///
    /// # Panics
    ///
    /// Panics if byte 8000 is inside a UTF-8 character when truncating output.
    fn run_verifier(&self) -> (bool, String) {
        let output = if cfg!(windows) {
            Command::new("cmd").args(["/C", &self.verify_cmd]).output()
        } else {
            Command::new("sh").args(["-c", &self.verify_cmd]).output()
        };
        match output {
            Ok(out) => {
                let mut combined = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr);
                if !stderr.is_empty() {
                    combined.push_str("\n--- stderr ---\n");
                    combined.push_str(&stderr);
                }
                // Truncate to keep responses bounded. Find a char boundary
                // to avoid panicking on multi-byte UTF-8 sequences.
                const MAX: usize = 8000;
                if combined.len() > MAX {
                    let mut end = MAX;
                    while !combined.is_char_boundary(end) {
                        end -= 1;
                    }
                    combined.truncate(end);
                    combined.push_str("\n...[truncated]");
                }
                (out.status.success(), combined)
            }
            Err(e) => (false, format!("failed to run verifier: {e}")),
        }
    }

    /// Append an attempt to history and return the matching proposal outcome.
    fn record(
        &mut self,
        attempt: usize,
        fingerprint: String,
        passed: bool,
        output: String,
    ) -> ProposeOutcome {
        self.history.push(AttemptRecord {
            attempt,
            fingerprint: fingerprint.clone(),
            passed,
            output: output.clone(),
        });
        ProposeOutcome {
            passed,
            attempt,
            output,
            fingerprint,
        }
    }
}

/// Outcome of a single `propose` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposeOutcome {
    pub passed: bool,
    pub attempt: usize,
    pub output: String,
    pub fingerprint: String,
}

/// Registry of live tasks, keyed by task_id.
#[derive(Default)]
pub struct TaskRegistry {
    tasks: Mutex<HashMap<String, AgentTask>>,
}

impl TaskRegistry {
    /// Create an empty registry shared through reference counting.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            tasks: Mutex::new(HashMap::new()),
        })
    }

    /// Store a task under its ID, replacing any existing task without finishing it.
    ///
    /// Recovers the lock if a previous panic poisoned the mutex.
    pub fn insert(&self, task: AgentTask) {
        self.tasks.lock().unwrap_or_else(|e| e.into_inner()).insert(task.task_id.clone(), task);
    }

    /// Call `f` with the matching task while holding the registry lock.
    /// Return `None` without calling `f` if the ID is unknown.
    /// The callback must not try to lock this registry again.
    ///
    /// Recovers the lock if a previous panic poisoned the mutex. A panic from `f` propagates
    /// and poisons the mutex.
    pub fn with_task<F, R>(&self, task_id: &str, f: F) -> Option<R>
    where
        F: FnOnce(&mut AgentTask) -> R,
    {
        let mut guard = self.tasks.lock().unwrap_or_else(|e| e.into_inner());
        guard.get_mut(task_id).map(f)
    }

    /// Remove and return a task without finishing it, or `None` for an unknown ID.
    ///
    /// Recovers the lock if a previous panic poisoned the mutex.
    pub fn remove(&self, task_id: &str) -> Option<AgentTask> {
        self.tasks.lock().unwrap_or_else(|e| e.into_inner()).remove(task_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_file(content: &str) -> (tempfile::NamedTempFile, PathBuf) {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        let path = f.path().to_path_buf();
        (f, path)
    }

    #[test]
    fn propose_commit_on_verify_pass() {
        let (_tmp, path) = tmp_file("v1");
        let mut task = AgentTask::start(
            "t1".into(),
            "test".into(),
            "exit 0".into(),
            vec![path.clone()],
            4,
        )
        .unwrap();
        let out = task.propose(vec![FileEdit {
            path: path.clone(),
            content: "v2".into(),
        }]);
        assert!(out.passed);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "v2");
    }

    #[test]
    fn propose_rollback_on_verify_fail() {
        let (_tmp, path) = tmp_file("v1");
        let mut task = AgentTask::start(
            "t1".into(),
            "test".into(),
            "exit 1".into(),
            vec![path.clone()],
            4,
        )
        .unwrap();
        let out = task.propose(vec![FileEdit {
            path: path.clone(),
            content: "v2".into(),
        }]);
        assert!(!out.passed);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "v1");
    }

    #[test]
    fn duplicate_rejected_not_reapplied() {
        let (_tmp, path) = tmp_file("v1");
        let mut task = AgentTask::start(
            "t1".into(),
            "test".into(),
            "exit 1".into(),
            vec![path.clone()],
            4,
        )
        .unwrap();
        let edit = FileEdit {
            path: path.clone(),
            content: "v2".into(),
        };
        let out1 = task.propose(vec![edit.clone()]);
        assert!(!out1.passed);
        let out2 = task.propose(vec![edit]);
        assert!(!out2.passed);
        assert!(out2.output.contains("already rejected"));
        // Only one real verifier run happened for the duplicate.
        assert_eq!(task.history().len(), 2);
    }

    #[test]
    fn propose_rejects_path_outside_allowlist() {
        let (_tmp, path) = tmp_file("v1");
        let (_tmp2, other) = tmp_file("other");
        let mut task = AgentTask::start(
            "t1".into(),
            "test".into(),
            "exit 0".into(),
            vec![path.clone()],
            4,
        )
        .unwrap();
        // Proposing an edit to a path not in the allowlist is rejected
        // without running the verifier or touching the file.
        let out = task.propose(vec![FileEdit {
            path: other.clone(),
            content: "hacked".into(),
        }]);
        assert!(!out.passed);
        assert!(out.output.contains("not in task file allowlist"));
        assert_eq!(std::fs::read_to_string(&other).unwrap(), "other");
        // The allowlisted file is untouched too.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "v1");
    }

    #[test]
    fn propose_rejects_after_max_attempts() {
        let (_tmp, path) = tmp_file("v1");
        let mut task = AgentTask::start(
            "t1".into(),
            "test".into(),
            "exit 1".into(),
            vec![path.clone()],
            2,
        )
        .unwrap();
        let edit = || FileEdit {
            path: path.clone(),
            content: "v2".into(),
        };
        // Two attempts allowed (both fail the verifier).
        assert!(!task.propose(vec![edit()]).passed);
        // Third attempt exceeds max_attempts=2.
        let out = task.propose(vec![edit()]);
        assert!(!out.passed);
        assert!(out.output.contains("max attempts"));
    }

    #[test]
    fn finish_abort_restores_originals() {
        let (_tmp, path) = tmp_file("v1");
        let mut task = AgentTask::start(
            "t1".into(),
            "test".into(),
            "exit 0".into(),
            vec![path.clone()],
            4,
        )
        .unwrap();
        let _ = task.propose(vec![FileEdit {
            path: path.clone(),
            content: "v2".into(),
        }]);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "v2");
        task.finish(false);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "v1");
    }
}
