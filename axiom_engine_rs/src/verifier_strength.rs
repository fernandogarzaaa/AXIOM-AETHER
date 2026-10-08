//! Verifier strength warning for the agent-task loop.
//!
//! AXIOM's task loop trusts whatever `verify_cmd` the agent provides. Mutation
//! testing has shown that "passing" test suites can have low mutation scores
//! (e.g. 62%), meaning AXIOM would commit broken code as "verified."
//!
//! This module provides a best-effort warning: when `task_start` receives a
//! `verify_cmd` that looks like a test command, it tries to run an available
//! mutation testing tool (testteeth for Python, cargo-mutants for Rust). If the
//! score is below [`WEAK_VERIFIER_THRESHOLD`], a warning string is returned.
//!
//! This is a warning, not a gate. If the mutation tools are not installed, or
//! the check fails for any reason, it skips silently so existing workflows
//! never break.

use std::process::Command;
use std::time::Duration;

/// Mutation score below which a verifier is considered weak.
pub const WEAK_VERIFIER_THRESHOLD: f64 = 70.0;

/// How long to wait for a mutation tool before giving up (best-effort).
const MUTATION_TIMEOUT: Duration = Duration::from_secs(120);

/// The test runner detected in a verify command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestRunner {
    Pytest,
    CargoTest,
}

/// Detect whether a verify command looks like a test invocation.
///
/// Returns the detected runner, or `None` for non-test commands
/// (builds, linters, custom scripts, etc.).
fn detect_runner(verify_cmd: &str) -> Option<TestRunner> {
    let lower = verify_cmd.to_lowercase();
    // Check for pytest variants: `pytest`, `python -m pytest`, `pytest -q ...`
    if lower.contains("pytest") {
        return Some(TestRunner::Pytest);
    }
    // Check for cargo test: `cargo test`, `cargo test --locked`, etc.
    // Avoid matching `cargo build` or `cargo clippy`.
    if lower.contains("cargo test") {
        return Some(TestRunner::CargoTest);
    }
    None
}

/// Check whether a binary is available on PATH.
fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run a command with a timeout, returning stdout+stderr combined on success.
fn run_with_timeout(mut cmd: Command) -> Option<String> {
    use std::io::Read;

    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;

    let deadline = std::time::Instant::now() + MUTATION_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_string(&mut out);
                }
                if let Some(mut stderr) = child.stderr.take() {
                    let mut err = String::new();
                    let _ = stderr.read_to_string(&mut err);
                    out.push_str(&err);
                }
                return if status.success() { Some(out) } else { None };
            }
            Ok(None) => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(_) => return None,
        }
    }
}

/// Extract a mutation score percentage from tool output.
///
/// Looks for patterns like `62.2%`, `score: 62`, `28/45 killed`, etc.
/// Returns the percentage as 0.0-100.0, or `None` if no score found.
fn parse_score(output: &str) -> Option<f64> {
    // Pattern 1: explicit percentage like "62.2%" or "mutation score: 62%"
    for line in output.lines() {
        let lower = line.to_lowercase();
        if lower.contains("mutation") || lower.contains("score") || lower.contains("killed") {
            // Find a number followed by %
            let mut chars = line.chars().peekable();
            let mut num = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() || c == '.' {
                    num.push(c);
                    chars.next();
                } else if c == '%' && !num.is_empty() {
                    if let Ok(v) = num.parse::<f64>() {
                        if (0.0..=100.0).contains(&v) {
                            return Some(v);
                        }
                    }
                    num.clear();
                    chars.next();
                } else {
                    if !num.is_empty() && c != '.' {
                        num.clear();
                    }
                    chars.next();
                }
            }
        }
    }
    // Pattern 2: "X/Y killed" → compute percentage
    for line in output.lines() {
        if line.to_lowercase().contains("killed") {
            // Find "N/M" pattern
            for token in line.split_whitespace() {
                // Strip trailing punctuation like commas or parens
                let token = token.trim_matches(|c: char| c == ',' || c == '(' || c == ')');
                if let Some(slash) = token.find('/') {
                    let (a, b) = token.split_at(slash);
                    let b = &b[1..];
                    if let (Ok(killed), Ok(total)) = (a.parse::<f64>(), b.parse::<f64>()) {
                        if total > 0.0 {
                            return Some(killed / total * 100.0);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Try testteeth for a Python pytest command. Returns score 0-100 or None.
fn testteeth_score() -> Option<f64> {
    if !tool_available("testteeth") {
        return None;
    }
    let mut cmd = Command::new("testteeth");
    cmd.arg("run");
    // Use the current directory (task's working directory)
    let output = run_with_timeout(cmd)?;
    parse_score(&output)
}

/// Try cargo-mutants for a Rust cargo test command. Returns score 0-100 or None.
fn cargo_mutants_score() -> Option<f64> {
    if !tool_available("cargo-mutants") {
        return None;
    }
    let mut cmd = Command::new("cargo");
    cmd.args(["mutants", "--no-shuffle"]);
    let output = run_with_timeout(cmd)?;
    parse_score(&output)
}

/// Check verifier strength and return a warning if it looks weak.
///
/// Returns `None` when:
/// - the command doesn't look like a test command,
/// - no mutation tool is available,
/// - the mutation check fails or times out,
/// - the score is at or above the threshold.
///
/// Returns `Some(warning)` when the score is below the threshold.
pub fn check_verifier_strength(verify_cmd: &str) -> Option<String> {
    let runner = detect_runner(verify_cmd)?;
    let score = match runner {
        TestRunner::Pytest => testteeth_score()?,
        TestRunner::CargoTest => cargo_mutants_score()?,
    };
    if score < WEAK_VERIFIER_THRESHOLD {
        Some(format!(
            "Mutation score {score:.1}% is below {WEAK_VERIFIER_THRESHOLD:.0}% - \
             verifier may be weak. Consider strengthening tests before relying on this gate. \
             (Task started anyway; this is a warning, not a block.)"
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_pytest() {
        assert_eq!(detect_runner("pytest tests/ -q"), Some(TestRunner::Pytest));
        assert_eq!(
            detect_runner("python -m pytest tests/test_foo.py"),
            Some(TestRunner::Pytest)
        );
        assert_eq!(
            detect_runner("PYTEST_ADDOPTS='' pytest -x"),
            Some(TestRunner::Pytest)
        );
    }

    #[test]
    fn detects_cargo_test() {
        assert_eq!(
            detect_runner("cargo test --locked"),
            Some(TestRunner::CargoTest)
        );
        assert_eq!(
            detect_runner("cargo test --lib verify_suggest"),
            Some(TestRunner::CargoTest)
        );
    }

    #[test]
    fn rejects_non_test_commands() {
        assert_eq!(detect_runner("cargo build --release"), None);
        assert_eq!(detect_runner("cargo clippy -- -D warnings"), None);
        assert_eq!(detect_runner("cargo check --all-targets"), None);
        assert_eq!(detect_runner("./my_custom_verify.sh"), None);
        assert_eq!(detect_runner("npm test"), None); // not yet supported
        assert_eq!(detect_runner("go test ./..."), None); // not yet supported
        assert_eq!(detect_runner(""), None);
    }

    #[test]
    fn parses_percentage_scores() {
        assert_eq!(parse_score("mutation score: 62.2%"), Some(62.2));
        assert_eq!(parse_score("Score: 85%"), Some(85.0));
        assert_eq!(parse_score("28/45 killed"), Some(28.0 / 45.0 * 100.0));
    }

    #[test]
    fn parses_killed_ratio() {
        let score = parse_score("14 survived, 28/45 killed").unwrap();
        assert!((score - 62.222).abs() < 0.01);
    }

    #[test]
    fn returns_none_for_garbage() {
        assert_eq!(parse_score("all tests passed"), None);
        assert_eq!(parse_score(""), None);
        assert_eq!(parse_score("error: something broke"), None);
    }

    #[test]
    fn skips_when_tools_missing() {
        // If neither tool is installed, check returns None (no warning, no crash).
        // If tools ARE installed, it may return Some or None depending on the
        // repo state — either way it must not panic.
        let _ = check_verifier_strength("pytest tests/ -q");
        let _ = check_verifier_strength("cargo test");
    }

    #[test]
    fn non_test_command_never_warns() {
        assert_eq!(check_verifier_strength("cargo build"), None);
        assert_eq!(check_verifier_strength("./verify.sh"), None);
    }

    #[test]
    fn warning_format_mentions_score_and_threshold() {
        // Simulate via parse + format logic: construct the warning directly
        // to verify the message shape without needing real mutation tools.
        let score = 62.2;
        let warning = format!(
            "Mutation score {score:.1}% is below {WEAK_VERIFIER_THRESHOLD:.0}% - \
             verifier may be weak. Consider strengthening tests before relying on this gate. \
             (Task started anyway; this is a warning, not a block.)"
        );
        assert!(warning.contains("62.2%"));
        assert!(warning.contains("70%"));
        assert!(warning.contains("warning, not a block"));
    }
}
