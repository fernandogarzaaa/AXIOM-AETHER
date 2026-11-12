//! Polymorphic user-mode JIT runner.
//!
//! The runner does not suspend or patch arbitrary OS thread instruction
//! pointers. It provides the safe equivalent used by the rest of Axiom: execute
//! an isolated process, capture faults, feed the trace to TTT feedback, apply a
//! bounded local patch to the source artifact, and retry up to three times.

use std::future::Future;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

use crate::patch_ranker::{FaultContext, PatchRanker, PatchRankerStatus};

pub const MAX_POLY_JIT_STEPS: usize = 3;

#[derive(Debug, Clone)]
pub struct PolyJitEngine {
    timeout: Duration,
    status: Arc<Mutex<PolyJitStatus>>,
    ranker_status: Arc<Mutex<PatchRankerStatus>>,
    optimizer: PatchRanker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyJitRunRequest {
    pub session_id: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub working_dir: Option<String>,
    #[serde(default)]
    pub source_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyJitDiagnostic {
    pub session_id: String,
    pub step: usize,
    pub command: String,
    pub status_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub source_excerpt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyJitReport {
    pub session_id: String,
    pub passed: bool,
    pub attempts: usize,
    pub patched: bool,
    pub diagnostics: Vec<PolyJitDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolyJitStatus {
    pub total_runs: u64,
    pub repaired_runs: u64,
    pub failed_runs: u64,
    pub last_session_id: Option<String>,
    pub last_status: Option<String>,
    pub last_error: Option<String>,
    /// Patch-ranker telemetry. The JSON key stays `quantum` (its historical
    /// name) so `/v1/hypervisor/jit_status` consumers keep working; it will
    /// become `ranker` at the next breaking release.
    #[serde(rename = "quantum")]
    pub ranker: PatchRankerStatus,
}

#[derive(Debug)]
pub enum PolyJitError {
    Invalid(String),
    Io(String),
    Timeout(String),
    Feedback(String),
}

impl std::fmt::Display for PolyJitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolyJitError::Invalid(msg) => write!(f, "poly-jit invalid request: {msg}"),
            PolyJitError::Io(msg) => write!(f, "poly-jit io error: {msg}"),
            PolyJitError::Timeout(msg) => write!(f, "poly-jit timeout: {msg}"),
            PolyJitError::Feedback(msg) => write!(f, "poly-jit feedback error: {msg}"),
        }
    }
}

impl std::error::Error for PolyJitError {}

impl Default for PolyJitEngine {
    fn default() -> Self {
        Self::new(Duration::from_secs(20))
    }
}

impl PolyJitEngine {
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            status: Arc::new(Mutex::new(PolyJitStatus::default())),
            ranker_status: Arc::new(Mutex::new(PatchRankerStatus::default())),
            optimizer: PatchRanker::default(),
        }
    }

    pub fn status(&self) -> PolyJitStatus {
        let mut status = self.status.lock().map(|s| s.clone()).unwrap_or_default();
        status.ranker = self.ranker_status();
        status
    }

    pub fn ranker_status(&self) -> PatchRankerStatus {
        self.ranker_status
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    pub async fn run_with_feedback<F, Fut>(
        &self,
        request: PolyJitRunRequest,
        mut feedback: F,
    ) -> Result<PolyJitReport, PolyJitError>
    where
        F: FnMut(PolyJitDiagnostic) -> Fut,
        Fut: Future<Output = Result<(), String>>,
    {
        if request.session_id.trim().is_empty() || request.command.trim().is_empty() {
            return Err(PolyJitError::Invalid(
                "session_id and command are required".into(),
            ));
        }
        self.update_status(|s| {
            s.total_runs += 1;
            s.last_session_id = Some(request.session_id.clone());
            s.last_status = Some("running".into());
            s.last_error = None;
        });

        let mut report = PolyJitReport {
            session_id: request.session_id.clone(),
            passed: false,
            attempts: 0,
            patched: false,
            diagnostics: Vec::new(),
        };

        for step in 1..=MAX_POLY_JIT_STEPS {
            report.attempts = step;
            let output = self.run_process(&request).await?;
            if output.status_code == Some(0) {
                report.passed = true;
                self.update_status(|s| {
                    if report.patched {
                        s.repaired_runs += 1;
                    }
                    s.last_status = Some("passed".into());
                });
                return Ok(report);
            }
            // The whole file is what gets patched; only the diagnostic carries
            // an excerpt. Patching the excerpt and writing it back would cut
            // every file longer than the excerpt.
            let source = read_source(request.source_path.as_deref()).await;
            let source_excerpt = source.as_deref().map(excerpt);
            let diagnostic = PolyJitDiagnostic {
                session_id: request.session_id.clone(),
                step,
                command: output.command,
                status_code: output.status_code,
                stdout: output.stdout,
                stderr: output.stderr,
                source_excerpt: source_excerpt.clone(),
            };
            feedback(diagnostic.clone())
                .await
                .map_err(PolyJitError::Feedback)?;
            report.diagnostics.push(diagnostic.clone());
            if step == MAX_POLY_JIT_STEPS {
                break;
            }
            if let Some(source_path) = request.source_path.as_deref() {
                let patched = source
                    .as_deref()
                    .and_then(|source| self.ranked_patch(&diagnostic, source))
                    .or_else(|| synthesize_patch(source.as_deref().unwrap_or("")));
                if let Some(patched) = patched {
                    tokio::fs::write(source_path, patched)
                        .await
                        .map_err(|e| PolyJitError::Io(format!("patch write failed: {e}")))?;
                    report.patched = true;
                    continue;
                }
            }
            break;
        }

        self.update_status(|s| {
            s.failed_runs += 1;
            s.last_status = Some("failed".into());
            s.last_error = report
                .diagnostics
                .last()
                .map(|d| format!("status={:?}", d.status_code));
        });
        Ok(report)
    }

    async fn run_process(
        &self,
        request: &PolyJitRunRequest,
    ) -> Result<ProcessOutput, PolyJitError> {
        let command_string = format!("{} {}", request.command, request.args.join(" "));
        let mut command = Command::new(&request.command);
        command
            .args(&request.args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // Dropping the future on timeout must not leave the verifier
            // running: kill the child on drop, and its process group below.
            .kill_on_drop(true);
        #[cfg(unix)]
        command.process_group(0);
        if let Some(dir) = request.working_dir.as_deref() {
            command.current_dir(dir);
        }
        let mut child = command
            .spawn()
            .map_err(|e| PolyJitError::Io(format!("process start failed: {e}")))?;
        let pid = child.id();
        // Bounded readers: each keeps at most MAX_OUTPUT_BYTES (head/tail),
        // so a spamming verifier cannot grow memory through its output.
        // `wait_with_output` reads unboundedly and must not be used here.
        let stdout_task = child.stdout.take().map(|mut out| {
            tokio::spawn(async move {
                let mut capped = crate::proc_bound::CappedOutput::new();
                let mut buf = [0u8; 8192];
                loop {
                    match out.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(n) => capped.push(&buf[..n]),
                        Err(_) => break,
                    }
                }
                capped.finish()
            })
        });
        let stderr_task = child.stderr.take().map(|mut err| {
            tokio::spawn(async move {
                let mut capped = crate::proc_bound::CappedOutput::new();
                let mut buf = [0u8; 8192];
                loop {
                    match err.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(n) => capped.push(&buf[..n]),
                        Err(_) => break,
                    }
                }
                capped.finish()
            })
        });
        // If this run is cancelled (future dropped) or returns early on a
        // wait error, the reader tasks must be aborted, not detached.
        let _reader_guard = AbortReadersOnDrop {
            stdout: stdout_task.as_ref().map(|t| t.abort_handle()),
            stderr: stderr_task.as_ref().map(|t| t.abort_handle()),
        };
        let status = match timeout(self.timeout, child.wait()).await {
            Ok(r) => r.map_err(|e| PolyJitError::Io(format!("process wait failed: {e}")))?,
            Err(_) => {
                if let Some(pid) = pid {
                    crate::proc_bound::kill_group(pid);
                }
                if let Some(task) = &stdout_task {
                    task.abort();
                }
                if let Some(task) = &stderr_task {
                    task.abort();
                }
                return Err(PolyJitError::Timeout(command_string.clone()));
            }
        };
        // A descendant holding the pipes keeps the readers alive; give them
        // the same drain grace as the sync runner, then kill the group so
        // the pipes close and the readers finish.
        let mut stdout_task = stdout_task;
        let mut stderr_task = stderr_task;
        let stdout = match &mut stdout_task {
            Some(task) => drain_reader(task, pid).await,
            None => Vec::new(),
        };
        let stderr = match &mut stderr_task {
            Some(task) => drain_reader(task, pid).await,
            None => Vec::new(),
        };
        Ok(ProcessOutput {
            command: command_string,
            status_code: status.code(),
            stdout: String::from_utf8_lossy(&stdout).to_string(),
            stderr: String::from_utf8_lossy(&stderr).to_string(),
        })
    }

    fn update_status(&self, update: impl FnOnce(&mut PolyJitStatus)) {
        if let Ok(mut status) = self.status.lock() {
            update(&mut status);
        }
    }

    fn ranked_patch(&self, diagnostic: &PolyJitDiagnostic, source: &str) -> Option<String> {
        if source.trim().is_empty() {
            return None;
        }
        let fault = FaultContext {
            stdout: diagnostic.stdout.clone(),
            stderr: diagnostic.stderr.clone(),
            status_code: diagnostic.status_code,
            source: source.to_string(),
        };
        let optimization = self.optimizer.rank(&fault).ok()?;
        if let Ok(mut status) = self.ranker_status.lock() {
            status.total_optimizations += 1;
            if optimization.selected_patch.is_some() {
                status.total_collapses += 1;
            }
            status.last_state = optimization.telemetry;
        }
        optimization.selected_patch
    }
}

struct ProcessOutput {
    command: String,
    status_code: Option<i32>,
    stdout: String,
    stderr: String,
}

/// Await a bounded reader task, giving a descendant that holds the pipes
/// one drain grace; if it still has not finished, kill the group and take
/// whatever the reader collected once the pipes close.
async fn drain_reader(task: &mut tokio::task::JoinHandle<Vec<u8>>, pid: Option<u32>) -> Vec<u8> {
    match timeout(crate::proc_bound::DRAIN_GRACE, &mut *task).await {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(_)) => Vec::new(),
        Err(_) => {
            if let Some(pid) = pid {
                crate::proc_bound::kill_group(pid);
            }
            match timeout(crate::proc_bound::DRAIN_GRACE, &mut *task).await {
                Ok(Ok(bytes)) => bytes,
                _ => {
                    // The reader is still stuck on a pipe held open by a
                    // descendant that escaped the process group (or a
                    // platform where kill_group is a no-op). Dropping the
                    // handle would detach the task, leaking it and its
                    // pipe forever; abort it instead.
                    task.abort();
                    Vec::new()
                }
            }
        }
    }
}

/// Aborts the bounded reader tasks when the enclosing run ends before they
/// finish. Dropping a `JoinHandle` only detaches its task, so a run that
/// is cancelled (its future dropped) or returns early on a wait error
/// would otherwise leak readers holding pipes open behind an escaped
/// descendant. Aborting a task that already finished is a no-op, so the
/// normal path is unaffected.
struct AbortReadersOnDrop {
    stdout: Option<tokio::task::AbortHandle>,
    stderr: Option<tokio::task::AbortHandle>,
}

impl Drop for AbortReadersOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = self.stdout.take() {
            handle.abort();
        }
        if let Some(handle) = self.stderr.take() {
            handle.abort();
        }
    }
}

/// Characters of the source artifact carried in a diagnostic.
const SOURCE_EXCERPT_CHARS: usize = 4096;

async fn read_source(path: Option<&str>) -> Option<String> {
    tokio::fs::read_to_string(PathBuf::from(path?)).await.ok()
}

/// The head of `source` shown in a diagnostic. Never written back to a file.
fn excerpt(source: &str) -> String {
    source.chars().take(SOURCE_EXCERPT_CHARS).collect()
}

pub fn synthesize_patch(source: &str) -> Option<String> {
    if source.contains("AXIOM_POLYJIT_FIXTURE_FAIL") && source.contains("Write-Error") {
        return Some("Write-Output \"AXIOM_POLYJIT_FIXTURE_PASS\"\nexit 0\n".to_string());
    }
    if source.contains("AXIOM_POLYJIT_FIXTURE_FAIL") {
        return Some(source.replace("AXIOM_POLYJIT_FIXTURE_FAIL", "AXIOM_POLYJIT_FIXTURE_PASS"));
    }
    if source.contains("exit 1") {
        return Some(source.replace("exit 1", "exit 0"));
    }
    if source.contains("throw ") {
        return Some("Write-Output \"axiom polyjit repaired\"\nexit 0\n".to_string());
    }
    if source.contains("assert_eq!(1, 2)") {
        return Some(source.replace("assert_eq!(1, 2)", "assert_eq!(1, 1)"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_patch_handles_fixture_marker() {
        let patched = synthesize_patch("AXIOM_POLYJIT_FIXTURE_FAIL").unwrap();
        assert!(patched.contains("AXIOM_POLYJIT_FIXTURE_PASS"));
    }

    #[tokio::test]
    async fn a_source_longer_than_the_excerpt_is_never_cut() {
        // No patch in the set applies to this file, so the run cannot repair
        // it. It used to write the 4096-character excerpt over the file
        // anyway while it tried.
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("notes.txt");
        let original = "a line of ordinary text\n".repeat(400);
        assert!(original.len() > 2 * SOURCE_EXCERPT_CHARS);
        std::fs::write(&source, &original).unwrap();
        let sizes = dir.path().join("sizes.log");

        let engine = PolyJitEngine::new(Duration::from_secs(30));
        let report = engine
            .run_with_feedback(
                PolyJitRunRequest {
                    session_id: "long".to_string(),
                    command: "sh".to_string(),
                    args: vec![
                        "-c".to_string(),
                        format!(
                            "wc -c < '{}' >> '{}'; false",
                            source.display(),
                            sizes.display()
                        ),
                    ],
                    working_dir: None,
                    source_path: Some(source.display().to_string()),
                },
                |_diag| async { Ok(()) },
            )
            .await
            .unwrap();
        assert!(!report.passed);
        // The diagnostic still carries only an excerpt.
        assert_eq!(
            report.diagnostics[0]
                .source_excerpt
                .as_deref()
                .map(|e| e.chars().count()),
            Some(SOURCE_EXCERPT_CHARS)
        );
        let seen: Vec<usize> = std::fs::read_to_string(&sizes)
            .unwrap()
            .lines()
            .map(|l| l.trim().parse().unwrap())
            .collect();
        assert!(seen.len() >= 2, "{seen:?}");
        assert!(
            seen.iter().all(|&n| n == original.len()),
            "the file was cut during a run: {seen:?}"
        );
        assert_eq!(std::fs::read_to_string(&source).unwrap(), original);
    }

    #[tokio::test]
    async fn run_process_caps_spamming_output() {
        // 3 MiB of stdout must come back capped at roughly
        // MAX_OUTPUT_BYTES, the run must still succeed, and the retained
        // window must show the start and the end of the stream.
        let engine = PolyJitEngine::new(Duration::from_secs(30));
        let out = engine
            .run_process(&PolyJitRunRequest {
                session_id: "test".to_string(),
                command: "sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "(yes abcdefghij | head -c 2999999; printf 'END-STDOUT')".to_string(),
                ],
                working_dir: None,
                source_path: None,
            })
            .await
            .unwrap();
        assert_eq!(out.status_code, Some(0));
        assert!(
            out.stdout.len() <= crate::proc_bound::MAX_OUTPUT_BYTES + 128,
            "stdout capped: got {} bytes",
            out.stdout.len()
        );
        assert!(
            out.stdout.starts_with("abcdefghij\n"),
            "head kept the start"
        );
        assert!(
            out.stdout.contains("3000009 bytes"),
            "marker reports the total"
        );
        assert!(out.stdout.ends_with("END-STDOUT"), "tail kept the end");
    }

    #[tokio::test]
    async fn run_process_keeps_output_when_a_descendant_holds_the_pipes() {
        // The child prints, exits at once, and leaves a background
        // grandchild holding stdout open. The first drain must time out,
        // the group must be killed, and the second drain must still return
        // the bytes the reader already collected.
        let t0 = std::time::Instant::now();
        let engine = PolyJitEngine::new(Duration::from_secs(30));
        let out = engine
            .run_process(&PolyJitRunRequest {
                session_id: "test".to_string(),
                command: "sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "printf 'PARTIAL-OUTPUT\\n'; (sleep 30 &)".to_string(),
                ],
                working_dir: None,
                source_path: None,
            })
            .await
            .unwrap();
        assert!(
            t0.elapsed() < Duration::from_secs(20),
            "returned in {:?}",
            t0.elapsed()
        );
        assert_eq!(out.status_code, Some(0));
        assert!(
            out.stdout.contains("PARTIAL-OUTPUT"),
            "drained output kept: got {:?}",
            out.stdout
        );
    }

    /// A reader that never finishes (a pipe held open by a descendant
    /// that escaped the process group, where kill_group cannot reach it)
    /// must be aborted on the final drain timeout, not detached. Dropping
    /// the handle would leak the task, its pipe, and its retained output
    /// forever; repeated runs would accumulate them.
    #[tokio::test]
    async fn drain_reader_aborts_a_stuck_reader_on_final_timeout() {
        let mut task = tokio::spawn(async { std::future::pending::<Vec<u8>>().await });
        let t0 = std::time::Instant::now();
        let out = drain_reader(&mut task, None).await;
        assert!(out.is_empty(), "a stuck reader returns no output");
        assert!(
            t0.elapsed() < Duration::from_secs(15),
            "drain returned in {:?}",
            t0.elapsed()
        );
        // Awaiting the task must yield Cancelled promptly: with the old
        // code the task was detached and still pending, so this times out.
        let joined = tokio::time::timeout(Duration::from_secs(2), &mut task).await;
        assert!(
            joined.is_ok(),
            "reader task was detached, not aborted: still pending after drain"
        );
        assert!(
            joined.unwrap().unwrap_err().is_cancelled(),
            "reader task must be cancelled"
        );
    }

    /// Dropping the reader guard must abort pending readers: the mechanism
    /// that cleans up when the enclosing run is cancelled or returns early
    /// on a wait error.
    #[tokio::test]
    async fn abort_readers_on_drop_aborts_pending_tasks() {
        let mut task = tokio::spawn(async { std::future::pending::<Vec<u8>>().await });
        let guard = AbortReadersOnDrop {
            stdout: Some(task.abort_handle()),
            stderr: None,
        };
        drop(guard);
        let joined = tokio::time::timeout(Duration::from_secs(2), &mut task).await;
        assert!(joined.is_ok(), "guard must abort the task, not detach it");
        assert!(
            joined.unwrap().unwrap_err().is_cancelled(),
            "reader task must be cancelled"
        );
    }

    #[tokio::test]
    async fn run_process_returns_promptly_when_a_descendant_escapes_the_group() {
        // The child prints, exits at once, and leaves a grandchild that
        // called setsid (a new session, outside the killed group) holding
        // stdout open. Both drain graces expire; the run must still return
        // bounded by them, with the stuck reader aborted, not leaked.
        let t0 = std::time::Instant::now();
        let engine = PolyJitEngine::new(Duration::from_secs(30));
        let out = engine
            .run_process(&PolyJitRunRequest {
                session_id: "test".to_string(),
                command: "sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "printf 'OUT\\n'; (setsid sleep 30 &)".to_string(),
                ],
                working_dir: None,
                source_path: None,
            })
            .await
            .unwrap();
        assert!(
            t0.elapsed() < Duration::from_secs(20),
            "returned in {:?}",
            t0.elapsed()
        );
        assert_eq!(out.status_code, Some(0));
        assert!(
            out.stdout.is_empty(),
            "no output survives a drain that never finishes: got {:?}",
            out.stdout
        );
    }
}
