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
    /// Start a new task. Snapshots the listed files so `finish(commit=false)`
    /// can restore the workspace to its pre-task state.
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

    /// Propose an edit-set. Applies it as a transaction, runs the verifier,
    /// commits on pass or rolls back on failure. Returns the outcome.
    pub fn propose(&mut self, edits: Vec<FileEdit>) -> ProposeOutcome {
        if self.finished {
            return ProposeOutcome {
                passed: false,
                attempt: self.attempt,
                output: "task already finished".to_string(),
                fingerprint: String::new(),
            };
        }
        self.attempt += 1;
        let attempt_no = self.attempt;

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
            // Update originals so a later abort doesn't clobber the good state.
            for edit in &edit_set.edits {
                self.originals
                    .insert(edit.path.clone(), std::fs::read(&edit.path).ok());
            }
            self.record(attempt_no, fingerprint, true, output)
        } else {
            tx.rollback();
            self.memory.record_rejected(&self.goal, &edit_set);
            self.record(attempt_no, fingerprint, false, output)
        }
    }

    /// History of all attempts this task.
    pub fn history(&self) -> &[AttemptRecord] {
        &self.history
    }

    /// Finish the task. If `commit` is false, restore all files to their
    /// pre-task state.
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
                // Truncate to keep responses bounded.
                const MAX: usize = 8000;
                if combined.len() > MAX {
                    combined.truncate(MAX);
                    combined.push_str("\n...[truncated]");
                }
                (out.status.success(), combined)
            }
            Err(e) => (false, format!("failed to run verifier: {e}")),
        }
    }

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
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            tasks: Mutex::new(HashMap::new()),
        })
    }

    pub fn insert(&self, task: AgentTask) {
        self.tasks.lock().unwrap().insert(task.task_id.clone(), task);
    }

    pub fn with_task<F, R>(&self, task_id: &str, f: F) -> Option<R>
    where
        F: FnOnce(&mut AgentTask) -> R,
    {
        let mut guard = self.tasks.lock().unwrap();
        guard.get_mut(task_id).map(f)
    }

    pub fn remove(&self, task_id: &str) -> Option<AgentTask> {
        self.tasks.lock().unwrap().remove(task_id)
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
