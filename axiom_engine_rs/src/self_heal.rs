//! `axiom run -- <command>` — the self-healing runtime supervisor.
//!
//! This is the vision primitive: a program executed *inside* Axiom does not
//! simply crash on an environmental anomaly. The supervisor:
//!
//! 1. **Feels the fault as tension** — the failure trace is scored through the
//!    model (cross-entropy), the same signal the drift gate uses. An anomalous
//!    trace is a loss spike inside the network.
//! 2. **Absorbs it into W̃** — the trace is wrapped in the execution-feedback
//!    schema and streamed through the TTT stack, taking real gradient steps so
//!    the session's fast-weights physically move toward the failure. The CE is
//!    re-measured after adaptation: the tension drop is printed, not asserted.
//! 3. **Heals the environment** — deterministic, *safe* policies repair what a
//!    process cannot survive on its own. v1 ships the canonical one: a missing
//!    directory (`ENOENT` / "Directory nonexistent") is created with the
//!    equivalent of `mkdir -p`. Policies only ever create directories — they
//!    never delete, overwrite, or write file content.
//! 4. **Continues the thread** — the process is restarted (same TTT session, so
//!    the failure history keeps compounding in W̃) until it exits cleanly, no
//!    new heal applies, or the restart budget is exhausted.
//!
//! Honesty notes: source-artifact patching stays in `poly_jit`; this module
//! heals the *environment*. Restarting a process is not literally resuming a
//! suspended thread — v1 targets batch/idempotent programs. The TTT absorption
//! is real (the weights move; the CE numbers are measured), but with an unbaked
//! checkpoint the absolute CE values are only meaningful relative to each other.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use candle_core::{Result as CResult, Tensor};

use crate::context_compressor::{adapt_session_blocking, feedback_adaptation_text};
use crate::inference::InferencePipeline;

/// Restart budget when the caller does not override it.
pub const DEFAULT_MAX_RESTARTS: usize = 3;

/// Per-attempt cap on environmental heals, so a pathological trace cannot make
/// the supervisor create unbounded directories.
const MAX_HEALS_PER_ATTEMPT: usize = 4;

/// How much of each captured stream is kept for scoring/absorption.
const TRACE_TAIL_BYTES: usize = 8 * 1024;

/// CE scoring window (mirrors eval_model's chunking).
const CE_CHUNK: usize = 512;

/// One applied environmental repair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Heal {
    /// `mkdir -p <path>` equivalent.
    CreatedDirectory(PathBuf),
}

impl std::fmt::Display for Heal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Heal::CreatedDirectory(p) => write!(f, "created directory {}", p.display()),
        }
    }
}

/// One failed attempt: the tension measured on its trace, before and after the
/// trace was absorbed into the session fast-weights.
#[derive(Debug, Clone)]
pub struct TensionSample {
    pub attempt: usize,
    pub ce_before: f32,
    pub ce_after: f32,
}

/// Outcome of a supervised run.
#[derive(Debug, Clone)]
pub struct RunReport {
    pub success: bool,
    pub attempts: usize,
    pub exit_code: Option<i32>,
    pub heals: Vec<Heal>,
    pub tension: Vec<TensionSample>,
    pub tokens_absorbed: usize,
}

/// Mean next-token cross-entropy of `ids` through the model with the given
/// session states (read-only on the states: each chunk runs on a clone so the
/// measurement itself never mutates W̃).
fn sequence_ce(pipeline: &InferencePipeline, states: &[Tensor], ids: &[u32]) -> CResult<f32> {
    if ids.len() < 2 {
        return Ok(f32::NAN);
    }
    let dev = pipeline.device();
    let mut total = 0.0f32;
    let mut n = 0usize;
    let mut probe: Vec<Tensor> = states.to_vec();
    for w in ids.chunks(CE_CHUNK) {
        if w.len() < 2 {
            continue;
        }
        let m = w.len();
        let input = Tensor::from_vec(w[..m - 1].to_vec(), (1, m - 1), dev)?;
        let logits = pipeline.model().forward_lm(&input, &mut probe)?;
        let (_, t, v) = logits.dims3()?;
        let l2d = logits.squeeze(0)?.reshape((t, v))?;
        let tgt = Tensor::from_vec(w[1..].to_vec(), (m - 1,), dev)?;
        total += candle_nn::loss::cross_entropy(&l2d, &tgt)?.to_scalar::<f32>()? * t as f32;
        for s in probe.iter_mut() {
            *s = s.detach();
        }
        n += t;
    }
    if n == 0 {
        Ok(f32::NAN)
    } else {
        Ok(total / n as f32)
    }
}

/// Extract filesystem paths implicated in a missing-file/dir failure trace.
///
/// Recognised phrasings (Python, Rust, coreutils, POSIX shells):
///   * `FileNotFoundError: [Errno 2] No such file or directory: '/a/b.txt'`
///   * `cat: /a/b: No such file or directory`
///   * `/bin/sh: 1: cannot create /a/b/out.txt: Directory nonexistent`
///   * `bash: line 1: /a/b/out.txt: No such file or directory`
pub fn extract_missing_paths(trace: &str) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for line in trace.lines() {
        let lower = line.to_ascii_lowercase();
        if !(lower.contains("no such file or directory")
            || lower.contains("directory nonexistent")
            || lower.contains("kind: notfound"))
        {
            continue;
        }
        // Quoted candidates: '...' and "..." containing a path separator.
        for quote in ['\'', '"'] {
            let mut rest = line;
            while let Some(start) = rest.find(quote) {
                let after = &rest[start + 1..];
                if let Some(end) = after.find(quote) {
                    let candidate = &after[..end];
                    if candidate.contains('/') && seen.insert(candidate.to_string()) {
                        found.push(PathBuf::from(candidate));
                    }
                    rest = &after[end + 1..];
                } else {
                    break;
                }
            }
        }
        // Unquoted candidates: any whitespace- or colon-delimited word that
        // looks like a path. Covers `cat: /a/b: No such file or directory` and
        // `sh: 1: cannot create /a/b/out.txt: Directory nonexistent`, where the
        // path sits mid-token rather than at a colon boundary.
        for word in line.split(|c: char| c.is_whitespace() || c == ':') {
            let t = word.trim().trim_end_matches([',', ';']);
            if (t.starts_with('/') || t.starts_with("./")) && t.len() > 1 && seen.insert(t.into())
            {
                found.push(PathBuf::from(t));
            }
        }
    }
    found
}

/// Apply the missing-directory heal for one implicated path.
///
/// A component with an extension is treated as a file → its parent directory is
/// created; otherwise the path itself is created. Only `create_dir_all` is ever
/// performed — nothing is deleted or written.
fn heal_missing_path(path: &Path) -> Option<Heal> {
    let dir = if path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.contains('.'))
        .unwrap_or(false)
    {
        path.parent()?.to_path_buf()
    } else {
        path.to_path_buf()
    };
    if dir.as_os_str().is_empty() || dir.exists() {
        return None;
    }
    std::fs::create_dir_all(&dir).ok()?;
    Some(Heal::CreatedDirectory(dir))
}

fn tail(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    // Snap to a char boundary.
    let mut start = s.len() - max;
    while !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

/// Run `command args...` under self-healing supervision.
pub fn run_supervised(
    pipeline: &InferencePipeline,
    command: &str,
    args: &[String],
    max_restarts: usize,
) -> CResult<RunReport> {
    let mut report = RunReport {
        success: false,
        attempts: 0,
        exit_code: None,
        heals: Vec::new(),
        tension: Vec::new(),
        tokens_absorbed: 0,
    };
    // One session for the whole supervised lifetime: every failure compounds
    // into the same fast-weights.
    let mut states = pipeline.init_session_states()?;
    let mut healed_paths: HashSet<PathBuf> = HashSet::new();

    for attempt in 1..=max_restarts + 1 {
        report.attempts = attempt;
        let started = Instant::now();
        let output = Command::new(command).args(args).output().map_err(|e| {
            candle_core::Error::Msg(format!("failed to spawn '{command}': {e}"))
        })?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        // The supervised program's output still belongs to the user.
        print!("{stdout}");
        eprint!("{stderr}");

        report.exit_code = output.status.code();
        if output.status.success() {
            report.success = true;
            println!(
                "[axiom-run] attempt {attempt}: exited cleanly in {:.1}s",
                started.elapsed().as_secs_f32()
            );
            return Ok(report);
        }

        let trace = format!(
            "{}\n{}",
            tail(&stdout, TRACE_TAIL_BYTES),
            tail(&stderr, TRACE_TAIL_BYTES)
        );
        println!(
            "[axiom-run] attempt {attempt}: exit={:?} — absorbing failure into W̃",
            report.exit_code
        );

        // 1. Feel the tension: CE of the trace through the current state.
        let feedback = feedback_adaptation_text(
            "process_failure",
            &format!("{command} {} (exit {:?})", args.join(" "), report.exit_code),
            Some(&trace),
        );
        let ids = pipeline.encode_text(&feedback);
        let ce_before = sequence_ce(pipeline, &states, &ids)?;

        // 2. Absorb: real TTT gradient steps on the failure trace.
        adapt_session_blocking(pipeline, &mut states, &ids)?;
        report.tokens_absorbed += ids.len();
        let ce_after = sequence_ce(pipeline, &states, &ids)?;
        println!(
            "[axiom-run]   tension: CE {ce_before:.3} -> {ce_after:.3} after absorption ({} tokens)",
            ids.len()
        );
        report.tension.push(TensionSample {
            attempt,
            ce_before,
            ce_after,
        });

        // 3. Heal the environment (new heals only — never loop on the same fix).
        let mut applied_new_heal = false;
        for path in extract_missing_paths(&trace)
            .into_iter()
            .take(MAX_HEALS_PER_ATTEMPT)
        {
            if healed_paths.contains(&path) {
                continue;
            }
            if let Some(heal) = heal_missing_path(&path) {
                println!("[axiom-run]   heal: {heal}");
                healed_paths.insert(path);
                report.heals.push(heal);
                applied_new_heal = true;
            }
        }

        // 4. Continue only when something actually changed; a blind restart of
        //    an unhealed environment would just replay the same failure.
        if !applied_new_heal {
            println!("[axiom-run] no applicable heal for this failure — stopping");
            return Ok(report);
        }
        if attempt == max_restarts + 1 {
            println!("[axiom-run] restart budget exhausted");
            return Ok(report);
        }
        println!("[axiom-run] environment healed — restarting");
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_python_quoted_path() {
        let trace = "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/ax/data.csv'";
        let paths = extract_missing_paths(trace);
        assert!(paths.contains(&PathBuf::from("/tmp/ax/data.csv")));
    }

    #[test]
    fn extracts_coreutils_colon_path() {
        let trace = "cat: /tmp/ax/missing: No such file or directory";
        let paths = extract_missing_paths(trace);
        assert!(paths.contains(&PathBuf::from("/tmp/ax/missing")));
    }

    #[test]
    fn extracts_shell_directory_nonexistent() {
        let trace = "/bin/sh: 1: cannot create /tmp/ax/out/r.txt: Directory nonexistent";
        let paths = extract_missing_paths(trace);
        assert!(paths.contains(&PathBuf::from("/tmp/ax/out/r.txt")));
    }

    #[test]
    fn ignores_lines_without_enoent_phrases() {
        let trace = "error: something else entirely about /tmp/ax/file";
        assert!(extract_missing_paths(trace).is_empty());
    }

    #[test]
    fn heal_creates_parent_for_file_like_paths() {
        let base = std::env::temp_dir().join(format!(
            "axiom_heal_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let file_path = base.join("deep").join("out.txt");
        let heal = heal_missing_path(&file_path).expect("heal must apply");
        assert_eq!(heal, Heal::CreatedDirectory(base.join("deep")));
        assert!(base.join("deep").exists());
        assert!(!file_path.exists(), "heal must never create the file itself");
        let _ = std::fs::remove_dir_all(&base);
    }
}
