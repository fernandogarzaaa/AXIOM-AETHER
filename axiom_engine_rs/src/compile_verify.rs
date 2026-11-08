//! Compile-time verifier for synthesized Rust code blocks: it runs `cargo check`
//! in an isolated temp Cargo package (no dependencies, no `build.rs`), and is
//! not a process or OS isolation boundary (no seccomp, no container, no
//! resource limits beyond a wall-clock timeout).
//!
//! This module was previously named `sandbox`; that name overstated what it
//! does and is kept only as a deprecated alias for one release. Because
//! `cargo check` still runs the real toolchain as the current user, treat the
//! checked code as untrusted input to a compiler, not as contained code.
//!
//! The verifier never writes into the real repository. It creates the temp
//! package under a scratch root, runs `cargo check` through
//! `tokio::process::Command`, captures diagnostics, and lets the server feed
//! those diagnostics back into Axiom's TTT feedback path.
//!
//! Environment variables (the `AXIOM_SANDBOX_*` aliases were removed in 0.7.0):
//! `AXIOM_COMPILE_VERIFY`, `AXIOM_COMPILE_VERIFY_ROOT`,
//! `AXIOM_COMPILE_VERIFY_MAX_STEPS`, `AXIOM_COMPILE_VERIFY_TIMEOUT_SECS`.

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
pub struct CompileVerifier {
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
pub struct CompileDiagnostic {
    pub session_id: String,
    pub step: usize,
    pub command: String,
    pub workspace: String,
    pub status_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

impl CompileDiagnostic {
    pub fn feedback_trace(&self) -> String {
        format!(
            "command: {}\nworkspace: {}\nstdout:\n{}\nstderr:\n{}",
            self.command, self.workspace, self.stdout, self.stderr
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileRunReport {
    pub session_id: String,
    pub blocks_checked: usize,
    pub passed: bool,
    pub attempts: usize,
    pub diagnostics: Vec<CompileDiagnostic>,
}

#[derive(Debug)]
pub enum CompileVerifyError {
    Io(String),
    Timeout(String),
    Feedback(String),
}

impl std::fmt::Display for CompileVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileVerifyError::Io(msg) => write!(f, "compile-verify io error: {msg}"),
            CompileVerifyError::Timeout(msg) => write!(f, "compile-verify timeout: {msg}"),
            CompileVerifyError::Feedback(msg) => write!(f, "compile-verify feedback error: {msg}"),
        }
    }
}

impl std::error::Error for CompileVerifyError {}

impl CompileVerifier {
    pub fn from_env() -> Option<Self> {
        let verify = std::env::var("AXIOM_COMPILE_VERIFY").ok();
        if verify
            .as_deref()
            .map(|v| matches!(v, "0" | "false" | "FALSE" | "off" | "OFF"))
            .unwrap_or(false)
        {
            return None;
        }
        let root = std::env::var("AXIOM_COMPILE_VERIFY_ROOT")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("axiom-compile-verify"));
        let max_steps = std::env::var("AXIOM_COMPILE_VERIFY_MAX_STEPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_MAX_OPTIMIZATION_STEPS)
            .min(DEFAULT_MAX_OPTIMIZATION_STEPS);
        let timeout_secs = std::env::var("AXIOM_COMPILE_VERIFY_TIMEOUT_SECS")
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
    ) -> Result<CompileRunReport, CompileVerifyError>
    where
        F: FnMut(CompileDiagnostic) -> Fut,
        Fut: Future<Output = Result<(), String>>,
    {
        let blocks = Self::rust_code_blocks(payload);
        let mut report = CompileRunReport {
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
    ) -> Result<CompileRunReport, CompileVerifyError>
    where
        F: FnMut(CompileDiagnostic) -> Fut,
        Fut: Future<Output = Result<(), String>>,
    {
        tokio::fs::create_dir_all(&self.root)
            .await
            .map_err(|e| CompileVerifyError::Io(format!("create root failed: {e}")))?;
        let workspace = self.root.join(format!(
            "{}-{}",
            sanitize_session_id(session_id),
            Uuid::new_v4()
        ));
        create_rust_workspace(&workspace, code).await?;

        let mut report = CompileRunReport {
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
                        .map_err(CompileVerifyError::Feedback)?;
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
    ) -> Result<CheckOutcome, CompileVerifyError> {
        let command = "cargo check --message-format=json".to_string();
        let output = timeout(
            self.timeout,
            Command::new("cargo")
                .args(["check", "--message-format=json"])
                .current_dir(workspace)
                .output(),
        )
        .await
        .map_err(|_| CompileVerifyError::Timeout(command.clone()))?
        .map_err(|e| CompileVerifyError::Io(format!("cargo check failed to start: {e}")))?;

        if output.status.success() {
            return Ok(CheckOutcome::Pass);
        }
        Ok(CheckOutcome::Fail(CompileDiagnostic {
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
    Fail(CompileDiagnostic),
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

async fn create_rust_workspace(path: &Path, code: &str) -> Result<(), CompileVerifyError> {
    tokio::fs::create_dir_all(path.join("src"))
        .await
        .map_err(|e| CompileVerifyError::Io(format!("create workspace failed: {e}")))?;
    tokio::fs::write(
        path.join("Cargo.toml"),
        "[package]\nname = \"axiom_compile_verify\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
    )
    .await
    .map_err(|e| CompileVerifyError::Io(format!("write Cargo.toml failed: {e}")))?;
    tokio::fs::write(path.join("src/lib.rs"), code)
        .await
        .map_err(|e| CompileVerifyError::Io(format!("write src/lib.rs failed: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_only_rust_fenced_blocks() {
        let payload = "text\n```rust\npub fn ok() {}\n```\n```python\nprint('x')\n```\n```rs\nfn two() {}\n```";
        let blocks = CompileVerifier::rust_code_blocks(payload);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].code.contains("pub fn ok"));
        assert!(blocks[1].code.contains("fn two"));
    }

    #[tokio::test]
    async fn invalid_rust_reports_diagnostics() {
        let root =
            std::env::temp_dir().join(format!("axiom-compile-verify-test-{}", Uuid::new_v4()));
        let verifier = CompileVerifier::new(root, 1, Duration::from_secs(30));
        let mut seen = Vec::new();
        let report = verifier
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
