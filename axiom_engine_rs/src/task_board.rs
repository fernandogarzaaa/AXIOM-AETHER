//! Lightweight inter-agent task coordination board.
//!
//! Provides a persistent, channel-scoped task queue that multiple agents
//! (Claude, Codex, local scripts) can use to hand work to each other through
//! a shared Axiom instance. Tasks carry an optional Axiom compression digest
//! so the receiver can reconstruct the full context without re-reading it.
//!
//! Storage: one JSONL file per channel under `AXIOM_TASK_DIR`
//! (default: `checkpoints/tasks/`). Claiming is serialised with a per-channel
//! `Mutex` so concurrent MCP calls don't double-claim.
//!
//! # Lifecycle
//! ```text
//! axiom_post_task  →  Pending
//! axiom_claim_task →  Claimed   (assigned to one agent)
//! axiom_task_result→  Done | Failed
//! ```

use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Lifecycle state of a task.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Claimed,
    Done,
    Failed,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "pending"),
            TaskStatus::Claimed => write!(f, "claimed"),
            TaskStatus::Done => write!(f, "done"),
            TaskStatus::Failed => write!(f, "failed"),
        }
    }
}

/// A single task entry stored in the board.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TaskEntry {
    /// Unique task identifier (UUID v4).
    pub task_id: String,
    /// Channel name (namespaced work queue).
    pub channel: String,
    /// Current lifecycle status.
    pub status: TaskStatus,
    /// Human-readable description of the work to be done.
    pub description: String,
    /// Optional Axiom context fingerprint from `axiom_compress_path`.
    /// The claimer can call `axiom_expand` to retrieve dropped symbols.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_digest: Option<String>,
    /// Agent-reported remaining token budget when the task was posted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_snapshot: Option<usize>,
    /// 0 = normal, 1 = high priority.
    #[serde(default)]
    pub priority: u8,
    /// Identifier of the agent that posted the task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub posted_by: Option<String>,
    /// Identifier of the agent that claimed the task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_by: Option<String>,
    /// Result body written by the claimer on completion/failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// Unix seconds when the task was posted.
    pub posted_at: u64,
    /// Unix seconds when the task was claimed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_at: Option<u64>,
    /// Unix seconds when the task reached Done or Failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_at: Option<u64>,
}

/// Persistent inter-agent task board backed by per-channel JSONL files.
pub struct TaskBoard {
    root: PathBuf,
    /// Per-channel mutexes so concurrent claim calls serialise correctly.
    channel_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl TaskBoard {
    /// Open the board, creating the storage directory if needed.
    pub fn open(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            channel_locks: Mutex::new(HashMap::new()),
        })
    }

    /// Open from the `AXIOM_TASK_DIR` env var (default `checkpoints/tasks/`).
    pub fn from_env() -> std::io::Result<Self> {
        let dir = std::env::var("AXIOM_TASK_DIR")
            .unwrap_or_else(|_| "checkpoints/tasks".to_string());
        Self::open(dir)
    }

    fn channel_path(&self, channel: &str) -> PathBuf {
        // Percent-encode everything outside [A-Za-z0-9_-] so distinct channel
        // names always map to distinct filenames (no silent collision).
        let stem: String = channel
            .bytes()
            .flat_map(|b| {
                if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' {
                    vec![b as char]
                } else {
                    format!("%{b:02X}").chars().collect::<Vec<_>>()
                }
            })
            .collect();
        self.root.join(format!("{stem}.jsonl"))
    }

    fn channel_lock(&self, channel: &str) -> Arc<Mutex<()>> {
        let mut locks = self.channel_locks.lock().unwrap();
        locks
            .entry(channel.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    fn load_channel(&self, channel: &str) -> Vec<TaskEntry> {
        let path = self.channel_path(channel);
        let Ok(file) = fs::File::open(&path) else {
            return Vec::new();
        };
        BufReader::new(file)
            .lines()
            .map_while(Result::ok)
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(&l).ok())
            .collect()
    }

    fn rewrite_channel(&self, channel: &str, tasks: &[TaskEntry]) -> std::io::Result<()> {
        let path = self.channel_path(channel);
        let tmp = path.with_extension("jsonl.tmp");
        {
            let mut f = fs::File::create(&tmp)?;
            for task in tasks {
                let line = serde_json::to_string(task)
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                writeln!(f, "{line}")?;
            }
        }
        fs::rename(tmp, path)?;
        Ok(())
    }

    /// Post a new task to `channel`. Returns the created entry.
    pub fn post_task(
        &self,
        channel: &str,
        description: impl Into<String>,
        context_digest: Option<String>,
        budget_snapshot: Option<usize>,
        priority: u8,
        posted_by: Option<String>,
    ) -> std::io::Result<TaskEntry> {
        let lock = self.channel_lock(channel);
        let _guard = lock.lock().unwrap();

        let entry = TaskEntry {
            task_id: Uuid::new_v4().to_string(),
            channel: channel.to_string(),
            status: TaskStatus::Pending,
            description: description.into(),
            context_digest,
            budget_snapshot,
            priority,
            posted_by,
            claimed_by: None,
            result: None,
            posted_at: now_secs(),
            claimed_at: None,
            done_at: None,
        };

        let path = self.channel_path(channel);
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let line = serde_json::to_string(&entry)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        writeln!(file, "{line}")?;

        Ok(entry)
    }

    /// Claim the highest-priority oldest pending task in `channel`.
    /// Returns `None` when no pending tasks exist.
    pub fn claim_task(
        &self,
        channel: &str,
        agent_id: Option<String>,
    ) -> std::io::Result<Option<TaskEntry>> {
        let lock = self.channel_lock(channel);
        let _guard = lock.lock().unwrap();

        let mut tasks = self.load_channel(channel);

        // Find index of best candidate: highest priority first, then oldest
        // (earliest posted_at), then lowest insertion index to break ties.
        let idx = tasks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.status == TaskStatus::Pending)
            .min_by(|(ia, a), (ib, b)| {
                // Higher priority wins (descending), then earlier posted_at
                // (ascending), then lower index (ascending) to break ties.
                b.priority.cmp(&a.priority)
                    .then(a.posted_at.cmp(&b.posted_at))
                    .then(ia.cmp(ib))
            });

        let Some((i, _)) = idx else {
            return Ok(None);
        };

        tasks[i].status = TaskStatus::Claimed;
        tasks[i].claimed_by = agent_id;
        tasks[i].claimed_at = Some(now_secs());

        let claimed = tasks[i].clone();
        self.rewrite_channel(channel, &tasks)?;
        Ok(Some(claimed))
    }

    /// Record the result of a claimed task.
    /// `success` = true → Done, false → Failed.
    pub fn task_result(
        &self,
        task_id: &str,
        result: impl Into<String>,
        success: bool,
    ) -> std::io::Result<bool> {
        // Find which channel owns this task_id.
        let channel = self.find_channel_for_task(task_id)?;
        let Some(channel) = channel else {
            return Ok(false);
        };

        let lock = self.channel_lock(&channel);
        let _guard = lock.lock().unwrap();

        let mut tasks = self.load_channel(&channel);
        let Some(task) = tasks.iter_mut().find(|t| t.task_id == task_id) else {
            return Ok(false);
        };

        task.status = if success {
            TaskStatus::Done
        } else {
            TaskStatus::Failed
        };
        task.result = Some(result.into());
        task.done_at = Some(now_secs());

        self.rewrite_channel(&channel, &tasks)?;
        Ok(true)
    }

    /// List tasks in `channel`, optionally filtered by status string.
    pub fn list_tasks(
        &self,
        channel: &str,
        status_filter: Option<&str>,
    ) -> Vec<TaskEntry> {
        self.load_channel(channel)
            .into_iter()
            .filter(|t| match status_filter {
                None => true,
                Some(f) => t.status.to_string() == f,
            })
            .collect()
    }

    /// Return all channel names (JSONL stems in the root directory).
    pub fn channels(&self) -> Vec<String> {
        let Ok(entries) = fs::read_dir(&self.root) else {
            return Vec::new();
        };
        entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let p = e.path();
                if p.extension()?.to_str()? == "jsonl" {
                    Some(
                        p.file_stem()?
                            .to_str()?
                            .to_string(),
                    )
                } else {
                    None
                }
            })
            .collect()
    }

    fn find_channel_for_task(&self, task_id: &str) -> std::io::Result<Option<String>> {
        for channel in self.channels() {
            let tasks = self.load_channel(&channel);
            if tasks.iter().any(|t| t.task_id == task_id) {
                return Ok(Some(channel));
            }
        }
        Ok(None)
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_board_std() -> (TaskBoard, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "axiom_task_test_{}", uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let board = TaskBoard::open(&dir).unwrap();
        (board, dir)
    }

    #[test]
    fn post_creates_pending_task() {
        let (board, _dir) = tmp_board_std();
        let task = board
            .post_task("test-chan", "do something", None, None, 0, None)
            .unwrap();
        assert_eq!(task.status, TaskStatus::Pending);
        assert!(!task.task_id.is_empty());
    }

    #[test]
    fn claim_returns_oldest_pending() {
        let (board, _dir) = tmp_board_std();
        board.post_task("chan", "first", None, None, 0, None).unwrap();
        board.post_task("chan", "second", None, None, 0, None).unwrap();
        let claimed = board.claim_task("chan", Some("agent-1".into())).unwrap();
        assert!(claimed.is_some());
        let c = claimed.unwrap();
        assert_eq!(c.status, TaskStatus::Claimed);
        assert_eq!(c.claimed_by.as_deref(), Some("agent-1"));
        assert_eq!(c.description, "first");
    }

    #[test]
    fn claim_returns_none_when_empty() {
        let (board, _dir) = tmp_board_std();
        let r = board.claim_task("empty-chan", None).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn task_result_marks_done() {
        let (board, _dir) = tmp_board_std();
        let t = board.post_task("chan", "work", None, None, 0, None).unwrap();
        board.claim_task("chan", None).unwrap();
        let ok = board.task_result(&t.task_id, "finished!", true).unwrap();
        assert!(ok);
        let tasks = board.list_tasks("chan", Some("done"));
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].result.as_deref(), Some("finished!"));
    }

    #[test]
    fn high_priority_claimed_first() {
        let (board, _dir) = tmp_board_std();
        board.post_task("chan", "normal", None, None, 0, None).unwrap();
        board.post_task("chan", "urgent", None, None, 1, Some("codex".into())).unwrap();
        let claimed = board.claim_task("chan", None).unwrap().unwrap();
        assert_eq!(claimed.description, "urgent");
    }

    #[test]
    fn list_tasks_filters_by_status() {
        let (board, _dir) = tmp_board_std();
        board.post_task("ch", "a", None, None, 0, None).unwrap();
        board.post_task("ch", "b", None, None, 0, None).unwrap();
        board.claim_task("ch", None).unwrap();
        let pending = board.list_tasks("ch", Some("pending"));
        let claimed = board.list_tasks("ch", Some("claimed"));
        assert_eq!(pending.len(), 1);
        assert_eq!(claimed.len(), 1);
    }

    #[test]
    fn channels_lists_created_channels() {
        let (board, _dir) = tmp_board_std();
        board.post_task("alpha", "x", None, None, 0, None).unwrap();
        board.post_task("beta", "y", None, None, 0, None).unwrap();
        let mut chans = board.channels();
        chans.sort();
        assert!(chans.contains(&"alpha".to_string()));
        assert!(chans.contains(&"beta".to_string()));
    }
}
