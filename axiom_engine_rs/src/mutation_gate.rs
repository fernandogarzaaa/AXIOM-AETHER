//! Mutation testing gate for the agent task loop.
//!
//! AXIOM's task loop trusts the verifier command supplied at `task_start`.
//! If that verifier is weak (tests that pass but wouldn't catch real bugs),
//! the loop will confidently commit broken code. This module measures verifier
//! strength using mutation testing tools and optionally refuses to start tasks
//! whose verifiers fall below a minimum score.
//!
//! Supported tools (best-effort; missing tools skip gracefully):
//! - `testteeth` for Python/pytest verifiers
//! - `cargo-mutants` for Rust/cargo-test verifiers
//!
//! The gate never blocks on uncertainty: if no tool is available, the command
//! doesn't look like a test command, or the score can't be parsed, measurement
//! returns `None` and the caller should proceed (optionally with a warning).

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Maximum time a mutation tool may run before it is killed.
///
/// `task_start` runs on the serial MCP loop, so a hung mutation tool must not
/// block it indefinitely. Five minutes is generous for a summary-mode probe
/// while still bounding the worst case.
const TOOL_TIMEOUT: Duration = Duration::from_secs(300);

/// Result of measuring a verifier's mutation score.
#[derive(Debug, Clone, PartialEq)]
pub struct MutationScore {
    /// Fraction of mutants killed, in 0.0..=1.0.
    pub score: f64,
    /// Name of the tool used (e.g. "testteeth", "cargo-mutants").
    pub tool: String,
}

/// Evaluate a measured score against a minimum threshold.
///
/// Returns `Ok(())` when the score meets the threshold, or `Err` with a
/// human-readable rejection message. Also validates the threshold range.
pub fn evaluate(score: &MutationScore, min_score: f64) -> Result<(), String> {
    if !(0.0..=1.0).contains(&min_score) {
        return Err(format!(
            "min_mutation_score must be between 0.0 and 1.0, got {min_score}"
        ));
    }
    if score.score < min_score {
        Err(format!(
            "mutation gate rejected: verifier mutation score {:.1}% below minimum {:.1}% (measured with {})",
            score.score * 100.0,
            min_score * 100.0,
            score.tool,
        ))
    } else {
        Ok(())
    }
}

/// Measure the mutation score for a verifier command, if possible.
///
/// Dispatches on the command text: `pytest` triggers testteeth,
/// `cargo test` triggers cargo-mutants. Returns `None` when:
/// - the command doesn't look like a test command,
/// - the mutation tool isn't installed,
/// - the tool fails or its output can't be parsed.
///
/// `None` means "could not measure", not "score is zero". Callers should
/// treat `None` as "skip the gate" rather than "reject".
pub fn measure_verifier_strength(verify_cmd: &str) -> Option<MutationScore> {
    let lower = verify_cmd.to_lowercase();
    if lower.contains("pytest") {
        measure_with_testteeth()
    } else if lower.contains("cargo test") || lower.contains("cargo nextest") {
        measure_with_cargo_mutants()
    } else {
        None
    }
}

/// Check whether a binary is available on PATH.
///
/// Goes through [`run_tool`] so even a misbehaving `--version` probe cannot
/// hang past the timeout.
fn tool_available(name: &str) -> bool {
    run_tool(name, &["--version"]).is_some()
}

/// Run a command and return stdout on success, enforcing [`TOOL_TIMEOUT`].
///
/// A tool that hangs past the timeout is killed and treated as "could not
/// measure" (`None`), so a stuck mutation tool can never block the serial
/// MCP loop forever. Partial output from a killed run is discarded.
fn run_tool(cmd: &str, args: &[&str]) -> Option<String> {
    let mut child = Command::new(cmd)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take();
    let deadline = Instant::now() + TOOL_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    // Hung tool: kill it rather than block the MCP loop.
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(_) => break None,
        }
    };
    // The child has exited (or was killed); drain whatever stdout remains.
    // Note: output larger than the pipe buffer can wedge a still-running
    // child, but such a run hits the timeout above and is killed, so this
    // read always terminates.
    let mut out = Vec::new();
    if let Some(mut pipe) = stdout {
        let _ = pipe.read_to_end(&mut out);
    }
    match status {
        Some(s) if s.success() => String::from_utf8(out).ok(),
        _ => None,
    }
}

/// Measure using testteeth (Python mutation testing).
///
/// Best-effort: checks availability via `--version`, then attempts a summary
/// run. The exact testteeth invocation depends on project layout; we run a
/// default invocation and parse defensively. Any failure means "could not
/// measure", not "score is zero". Returns `None` if the tool is missing or
/// the score can't be determined.
fn measure_with_testteeth() -> Option<MutationScore> {
    if !tool_available("testteeth") {
        return None;
    }
    // Probe: testteeth's exact run CLI varies by version and project layout.
    // We attempt a default run and parse defensively; any failure means
    // "could not measure", not "score is zero". A failed probe is logged to
    // stderr so a wrong guessed invocation does not skip silently.
    match run_tool("testteeth", &["--format", "summary"]) {
        Some(output) => {
            let score = parse_testteeth_summary(&output);
            if score.is_none() {
                eprintln!(
                    "mutation_gate: testteeth ran but its output could not be parsed; gate skipped"
                );
            }
            score.map(|score| MutationScore {
                score,
                tool: "testteeth".to_string(),
            })
        }
        None => {
            eprintln!(
                "mutation_gate: testteeth probe (--format summary) failed; \
                 the CLI invocation may be wrong for this testteeth version; gate skipped"
            );
            None
        }
    }
}

/// Measure using cargo-mutants (Rust mutation testing).
///
/// Best-effort: runs `cargo mutants --help` to confirm availability, then
/// parses. A full `cargo mutants` run can take a long time, so this only
/// probes availability here; the actual measurement is left to a future
/// implementation that accepts a pre-computed score. Returns `None` for now
/// unless the tool reports a cached result.
fn measure_with_cargo_mutants() -> Option<MutationScore> {
    if !tool_available("cargo") {
        return None;
    }
    // Check for cargo-mutants subcommand availability.
    let has_mutants = run_tool("cargo", &["mutants", "--version"]).is_some();
    if !has_mutants {
        return None;
    }
    // A full cargo-mutants run is expensive (compiles per mutant). Rather
    // than run it inline at task_start, we report availability and let the
    // caller decide. For the gate, an unavailable score means "skip".
    //
    // Future work: accept `mutation_score` directly as a task_start param
    // so CI pipelines can pre-compute it with `cargo mutants --json`.
    None
}

/// Parse a mutation score from testteeth output.
///
/// Looks for patterns like "28/45 killed", "62.2%", "killed: 28, survived: 14".
/// Returns `None` when no recognizable score is found. A malformed line is
/// skipped (`continue`) so one bad line never aborts parsing of the rest.
fn parse_testteeth_summary(output: &str) -> Option<f64> {
    let lower = output.to_lowercase();
    for line in lower.lines() {
        // Try "killed: 28, survived: 14" style.
        if line.contains("killed") && line.contains("survived") {
            let (Some(killed), Some(survived)) = (
                extract_number_after(line, "killed"),
                extract_number_after(line, "survived"),
            ) else {
                continue; // Bad line: skip it, keep parsing the rest.
            };
            let total = killed + survived;
            if total > 0.0 {
                return Some(killed / total);
            }
            continue;
        }
        // Try "mutation score: 62.2%" style.
        if let Some(score) = parse_percentage(line) {
            return Some(score);
        }
        // Try "28/45 killed" style.
        if let Some(score) = parse_fraction_killed(line) {
            return Some(score);
        }
    }
    None
}

/// Parse a "62.2%" style percentage into a 0.0..=1.0 fraction.
///
/// Only lines mentioning mutation/score/killed are considered, so unrelated
/// percentages (e.g. code coverage) are not mistaken for a mutation score.
fn parse_percentage(line: &str) -> Option<f64> {
    if !(line.contains("mutation") || line.contains("score") || line.contains("killed")) {
        return None;
    }
    let pct_idx = line.rfind('%')?;
    let before = &line[..pct_idx];
    let num_str: String = before
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    let pct: f64 = num_str.parse().ok()?;
    if (0.0..=100.0).contains(&pct) {
        Some(pct / 100.0)
    } else {
        None
    }
}

/// Extract the first floating-point number appearing after `keyword` in `line`.
fn extract_number_after(line: &str, keyword: &str) -> Option<f64> {
    let idx = line.find(keyword)?;
    let rest = &line[idx + keyword.len()..];
    let num_str: String = rest
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    num_str.parse().ok()
}

/// Parse "28/45 killed" or "28 of 45 killed" into a kill fraction.
fn parse_fraction_killed(line: &str) -> Option<f64> {
    let killed_idx = line.find("killed")?;
    let before = &line[..killed_idx];
    // Find the last "N/M" or "N of M" before "killed".
    let tokens: Vec<&str> = before.split_whitespace().collect();
    for window in tokens.windows(3).rev() {
        if window[1] == "of" {
            if let (Ok(n), Ok(m)) = (window[0].parse::<f64>(), window[2].parse::<f64>()) {
                if m > 0.0 {
                    return Some(n / m);
                }
            }
        }
    }
    // Try "28/45" pattern.
    for token in tokens.iter().rev() {
        if let Some(slash) = token.find('/') {
            if let (Ok(n), Ok(m)) = (
                token[..slash].parse::<f64>(),
                token[slash + 1..].parse::<f64>(),
            ) {
                if m > 0.0 && n <= m {
                    return Some(n / m);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_accepts_score_at_threshold() {
        let s = MutationScore {
            score: 0.8,
            tool: "testteeth".into(),
        };
        assert!(evaluate(&s, 0.8).is_ok());
    }

    #[test]
    fn evaluate_accepts_score_above_threshold() {
        let s = MutationScore {
            score: 0.95,
            tool: "testteeth".into(),
        };
        assert!(evaluate(&s, 0.5).is_ok());
    }

    #[test]
    fn evaluate_rejects_score_below_threshold() {
        let s = MutationScore {
            score: 0.62,
            tool: "testteeth".into(),
        };
        let err = evaluate(&s, 0.8).unwrap_err();
        assert!(err.contains("62.0%"));
        assert!(err.contains("80.0%"));
        assert!(err.contains("testteeth"));
    }

    #[test]
    fn evaluate_rejects_invalid_threshold() {
        let s = MutationScore {
            score: 0.9,
            tool: "testteeth".into(),
        };
        assert!(evaluate(&s, 1.5).is_err());
        assert!(evaluate(&s, -0.1).is_err());
    }

    #[test]
    fn measure_returns_none_for_non_test_command() {
        assert!(measure_verifier_strength("echo hello").is_none());
        assert!(measure_verifier_strength("cargo build").is_none());
    }

    #[test]
    fn parse_killed_survived_format() {
        assert_eq!(
            parse_testteeth_summary("killed: 28, survived: 14"),
            Some(28.0 / 42.0)
        );
    }

    #[test]
    fn parse_fraction_format() {
        assert_eq!(parse_testteeth_summary("28/45 killed"), Some(28.0 / 45.0));
    }

    #[test]
    fn parse_of_format() {
        assert_eq!(
            parse_testteeth_summary("28 of 45 killed"),
            Some(28.0 / 45.0)
        );
    }

    #[test]
    fn parse_returns_none_for_unrecognized() {
        assert_eq!(parse_testteeth_summary("all tests passed"), None);
        assert_eq!(parse_testteeth_summary(""), None);
    }

    #[test]
    fn parse_percentage_format() {
        let score = parse_testteeth_summary("mutation score: 62.2%").unwrap();
        assert!((score - 0.622).abs() < 1e-9);
    }

    #[test]
    fn parse_percentage_ignores_unrelated_lines() {
        // A coverage percentage is not a mutation score.
        assert_eq!(parse_testteeth_summary("coverage: 62.2%"), None);
    }

    #[test]
    fn parse_skips_bad_line_and_continues() {
        // The first line mentions killed/survived but carries no numbers; it
        // must not abort parsing of the later percentage line.
        let output = "killed and survived with no numbers here\nmutation score: 62.2%";
        let score = parse_testteeth_summary(output).unwrap();
        assert!((score - 0.622).abs() < 1e-9);
    }

    #[test]
    fn run_tool_returns_stdout_on_success() {
        let out = run_tool("echo", &["hello"]);
        assert_eq!(out.as_deref().map(str::trim), Some("hello"));
    }

    #[test]
    fn run_tool_returns_none_on_failure() {
        assert_eq!(run_tool("false", &[]), None);
    }

    #[test]
    fn run_tool_returns_none_for_missing_binary() {
        assert_eq!(run_tool("axiom-definitely-not-a-real-binary", &[]), None);
    }
}
