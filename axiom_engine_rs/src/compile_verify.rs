//! Closed-loop local compilation sandbox for synthesized code blocks.
//!
//! The sandbox never writes into the real repository. It creates an isolated
//! temp Cargo package, runs compiler checks through `tokio::process::Command`,
//! captures diagnostics, and lets the server feed those diagnostics back into
//! Axiom's TTT feedback path.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio::time::timeout;
use uuid::Uuid;

pub const DEFAULT_MAX_OPTIMIZATION_STEPS: usize = 3;
const DEFAULT_TIMEOUT_SECS: u64 = 20;

#[derive(Debug, Clone)]
pub struct SandboxController {
    root: PathBuf,
    max_steps: usize,
    timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeBlock {
    pub language: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxDiagnostic {
    pub session_id: String,
    pub step: usize,
    pub command: String,
    pub workspace: String,
    pub status_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

impl SandboxDiagnostic {
    pub fn feedback_trace(&self) -> String {
        format!(
            "command: {}\nworkspace: {}\nstdout:\n{}\nstderr:\n{}",
            self.command, self.workspace, self.stdout, self.stderr
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxRunReport {
    pub session_id: String,
    pub blocks_checked: usize,
    pub passed: bool,
    pub attempts: usize,
    pub diagnostics: Vec<SandboxDiagnostic>,
}

#[derive(Debug)]
pub enum SandboxError {
    Io(String),
    Timeout(String),
    Feedback(String),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::Io(msg) => write!(f, "sandbox io error: {msg}"),
            SandboxError::Timeout(msg) => write!(f, "sandbox timeout: {msg}"),
            SandboxError::Feedback(msg) => write!(f, "sandbox feedback error: {msg}"),
        }
    }
}

impl std::error::Error for SandboxError {}

impl SandboxController {
    pub fn from_env() -> Option<Self> {
        if std::env::var("AXIOM_SANDBOX_VERIFY")
            .map(|v| matches!(v.as_str(), "0" | "false" | "FALSE" | "off" | "OFF"))
            .unwrap_or(false)
        {
            return None;
        }
        let root = std::env::var("AXIOM_SANDBOX_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("axiom-sandbox"));
        let max_steps = std::env::var("AXIOM_SANDBOX_MAX_STEPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_MAX_OPTIMIZATION_STEPS)
            .min(DEFAULT_MAX_OPTIMIZATION_STEPS);
        let timeout_secs = std::env::var("AXIOM_SANDBOX_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_TIMEOUT_SECS);
        Some(Self::new(
            root,
            max_steps,
            Duration::from_secs(timeout_secs),
        ))
    }

    pub fn new(root: PathBuf, max_steps: usize, timeout: Duration) -> Self {
        Self {
            root,
            max_steps: max_steps.clamp(1, DEFAULT_MAX_OPTIMIZATION_STEPS),
            timeout,
        }
    }

    pub fn rust_code_blocks(payload: &str) -> Vec<CodeBlock> {
        extract_fenced_code_blocks(payload)
            .into_iter()
            .filter(|block| is_rust_language(&block.language))
            .collect()
    }

    pub async fn verify_rust_code_blocks_with_feedback<F, Fut>(
        &self,
        session_id: &str,
        payload: &str,
        mut feedback: F,
    ) -> Result<SandboxRunReport, SandboxError>
    where
        F: FnMut(SandboxDiagnostic) -> Fut,
        Fut: Future<Output = Result<(), String>>,
    {
        let blocks = Self::rust_code_blocks(payload);
        let mut report = SandboxRunReport {
            session_id: session_id.to_string(),
            blocks_checked: blocks.len(),
            passed: true,
            attempts: 0,
            diagnostics: Vec::new(),
        };
        for block in blocks {
            let block_report = self
                .verify_rust_code_with_feedback(session_id, &block.code, &mut feedback)
                .await?;
            report.attempts += block_report.attempts;
            report.passed &= block_report.passed;
            report.diagnostics.extend(block_report.diagnostics);
        }
        Ok(report)
    }

    pub async fn verify_rust_code_with_feedback<F, Fut>(
        &self,
        session_id: &str,
        code: &str,
        feedback: &mut F,
    ) -> Result<SandboxRunReport, SandboxError>
    where
        F: FnMut(SandboxDiagnostic) -> Fut,
        Fut: Future<Output = Result<(), String>>,
    {
        tokio::fs::create_dir_all(&self.root)
            .await
            .map_err(|e| SandboxError::Io(format!("create root failed: {e}")))?;
        let workspace = self.root.join(format!(
            "{}-{}",
            sanitize_session_id(session_id),
            Uuid::new_v4()
        ));
        create_rust_workspace(&workspace, code).await?;

        let mut report = SandboxRunReport {
            session_id: session_id.to_string(),
            blocks_checked: 1,
            passed: false,
            attempts: 0,
            diagnostics: Vec::new(),
        };
        for step in 1..=self.max_steps {
            report.attempts = step;
            match self.run_cargo_check(&workspace, session_id, step).await? {
                CheckOutcome::Pass => {
                    report.passed = true;
                    break;
                }
                CheckOutcome::Fail(diagnostic) => {
                    feedback(diagnostic.clone())
                        .await
                        .map_err(SandboxError::Feedback)?;
                    report.diagnostics.push(diagnostic);
                }
            }
        }
        let _ = tokio::fs::remove_dir_all(&workspace).await;
        Ok(report)
    }

    async fn run_cargo_check(
        &self,
        workspace: &Path,
        session_id: &str,
        step: usize,
    ) -> Result<CheckOutcome, SandboxError> {
        let command = "cargo check --message-format=json".to_string();
        let output = timeout(
            self.timeout,
            Command::new("cargo")
                .args(["check", "--message-format=json"])
                .current_dir(workspace)
                .output(),
        )
        .await
        .map_err(|_| SandboxError::Timeout(command.clone()))?
        .map_err(|e| SandboxError::Io(format!("cargo check failed to start: {e}")))?;

        if output.status.success() {
            return Ok(CheckOutcome::Pass);
        }
        Ok(CheckOutcome::Fail(SandboxDiagnostic {
            session_id: session_id.to_string(),
            step,
            command,
            workspace: workspace.display().to_string(),
            status_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        }))
    }
}

enum CheckOutcome {
    Pass,
    Fail(SandboxDiagnostic),
}

pub fn extract_fenced_code_blocks(payload: &str) -> Vec<CodeBlock> {
    let mut blocks = Vec::new();
    let mut lines = payload.lines();
    while let Some(line) = lines.next() {
        let Some(rest) = line.trim_start().strip_prefix("```") else {
            continue;
        };
        let language = rest
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let mut code = String::new();
        for body_line in lines.by_ref() {
            if body_line.trim_start().starts_with("```") {
                break;
            }
            code.push_str(body_line);
            code.push('\n');
        }
        blocks.push(CodeBlock { language, code });
    }
    blocks
}

fn is_rust_language(language: &str) -> bool {
    matches!(language.to_ascii_lowercase().as_str(), "rust" | "rs")
}

fn sanitize_session_id(session_id: &str) -> String {
    session_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
}

async fn create_rust_workspace(path: &Path, code: &str) -> Result<(), SandboxError> {
    tokio::fs::create_dir_all(path.join("src"))
        .await
        .map_err(|e| SandboxError::Io(format!("create workspace failed: {e}")))?;
    tokio::fs::write(
        path.join("Cargo.toml"),
        "[package]\nname = \"axiom_sandbox\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
    )
    .await
    .map_err(|e| SandboxError::Io(format!("write Cargo.toml failed: {e}")))?;
    tokio::fs::write(path.join("src/lib.rs"), code)
        .await
        .map_err(|e| SandboxError::Io(format!("write src/lib.rs failed: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_only_rust_fenced_blocks() {
        let payload = "text\n```rust\npub fn ok() {}\n```\n```python\nprint('x')\n```\n```rs\nfn two() {}\n```";
        let blocks = SandboxController::rust_code_blocks(payload);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].code.contains("pub fn ok"));
        assert!(blocks[1].code.contains("fn two"));
    }

    #[tokio::test]
    async fn invalid_rust_reports_diagnostics() {
        let root = std::env::temp_dir().join(format!("axiom-sandbox-test-{}", Uuid::new_v4()));
        let sandbox = SandboxController::new(root, 1, Duration::from_secs(30));
        let mut seen = Vec::new();
        let report = sandbox
            .verify_rust_code_with_feedback("s1", "pub fn broken( {", &mut |diag| {
                seen.push(diag.clone());
                async { Ok(()) }
            })
            .await
            .unwrap();
        assert!(!report.passed);
        assert_eq!(report.attempts, 1);
        assert_eq!(seen.len(), 1);
        assert!(seen[0].stderr.contains("error") || seen[0].stdout.contains("\"reason\""));
    }
}
