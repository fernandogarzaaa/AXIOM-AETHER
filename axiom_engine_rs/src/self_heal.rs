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

/// Cap on transient-fault retries per supervised run (waiting is the heal, but
/// only for recognised transient phrases and only this many times).
const MAX_TRANSIENT_RETRIES: usize = 2;

/// Transient-fault phrases where the correct heal is simply waiting and
/// retrying: the environment repairs itself with time.
const TRANSIENT_PHRASES: &[&str] = &[
    "connection refused",
    "connection reset",
    "connection timed out",
    "temporary failure in name resolution",
    "resource temporarily unavailable",
    "operation timed out",
];

/// One applied environmental repair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Heal {
    /// `mkdir -p <path>` equivalent.
    CreatedDirectory(PathBuf),
    /// A recognised transient fault: backed off and retried.
    TransientRetry(&'static str),
    /// Learned immunity: a directory this program needed in a past run,
    /// re-created prophylactically *before* the first attempt.
    Immunized(PathBuf),
    /// A file the program tried to execute lacked the execute bit; `chmod +x`
    /// was applied (the file's contents are never touched).
    MadeExecutable(PathBuf),
}

impl std::fmt::Display for Heal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Heal::CreatedDirectory(p) => write!(f, "created directory {}", p.display()),
            Heal::TransientRetry(phrase) => {
                write!(f, "transient fault (\"{phrase}\") — backing off and retrying")
            }
            Heal::Immunized(p) => {
                write!(f, "immunity: pre-created remembered directory {}", p.display())
            }
            Heal::MadeExecutable(p) => write!(f, "made executable (chmod +x) {}", p.display()),
        }
    }
}

/// Detect a recognised transient-fault phrase in a failure trace.
pub fn detect_transient(trace: &str) -> Option<&'static str> {
    let lower = trace.to_ascii_lowercase();
    TRANSIENT_PHRASES
        .iter()
        .find(|p| lower.contains(*p))
        .copied()
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
    /// Total TTT absorption passes run across all failures. Tension-gated: a
    /// novel/first-seen fault is absorbed harder than a familiar one.
    pub absorption_passes: usize,
    /// Non-corrective findings surfaced to the operator (e.g. a required
    /// environment variable Axiom cannot fabricate). Recorded and reported, but
    /// they do not by themselves justify a restart.
    pub diagnostics: Vec<String>,
}

/// Absorption passes for a familiar (KNOWN) failure: light reinforcement, the
/// engine has already moved its weights toward this fault before.
const LIGHT_ABSORPTION_PASSES: usize = 1;

/// Absorption passes for a FIRST/NOVEL fault (or when there is no history to
/// classify against): concentrate gradient effort on the surprising tension.
const DEEP_ABSORPTION_PASSES: usize = 3;

/// Tension-gated absorption depth. Without a classification (no heal memory) we
/// treat the fault as unfamiliar and absorb deeply.
fn absorption_passes(novelty: Option<crate::heal_memory::Novelty>) -> usize {
    use crate::heal_memory::Novelty;
    match novelty {
        Some(Novelty::Known) => LIGHT_ABSORPTION_PASSES,
        _ => DEEP_ABSORPTION_PASSES,
    }
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
        // looks like a path. Covers `cat: /a/b: No such file or directory`,
        // `sh: 1: cannot create /a/b/out.txt: Directory nonexistent`, and bare
        // *relative* paths like `build/out/app.bin` (which is what a shell
        // prints when the program used a cwd-relative path — the portable case).
        for word in line.split(|c: char| c.is_whitespace() || c == ':') {
            let t = word.trim().trim_end_matches([',', ';']);
            if t.len() <= 1 || t.contains("://") || seen.contains(t) {
                continue; // too short, a URL, or already captured
            }
            let absolute = t.starts_with('/') || t.starts_with("./");
            let relative_path = t.contains('/')
                && t
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_');
            if absolute || relative_path {
                seen.insert(t.to_string());
                found.push(PathBuf::from(t));
            }
        }
    }
    found
}

/// Apply the missing-directory heal for one implicated path.
///
/// A component with an extension is treated as a file → its parent directory is
/// created; otherwise the path itself is created. Relative paths (as they often
/// appear in error traces) resolve against `anchor` — the supervised process's
/// working directory — so the heal lands where the program actually looked.
/// Only `create_dir_all` is ever performed — nothing is deleted or written.
fn heal_missing_path(path: &Path, anchor: &Path) -> Option<Heal> {
    let resolved = if path.is_relative() {
        anchor.join(path)
    } else {
        path.to_path_buf()
    };
    let dir = if resolved
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.contains('.'))
        .unwrap_or(false)
    {
        resolved.parent()?.to_path_buf()
    } else {
        resolved
    };
    if dir.as_os_str().is_empty() || dir.exists() {
        return None;
    }
    std::fs::create_dir_all(&dir).ok()?;
    Some(Heal::CreatedDirectory(dir))
}

/// Extract paths implicated in a "Permission denied" failure trace — candidate
/// targets for the execute-bit heal. Same path-token heuristic as
/// [`extract_missing_paths`], gated on the permission-denied phrase.
pub fn extract_permission_denied_paths(trace: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for line in trace.lines() {
        if !line.to_ascii_lowercase().contains("permission denied") {
            continue;
        }
        for raw in line.split(|c: char| c.is_whitespace() || c == ':') {
            let t = raw.trim().trim_matches(|c| c == '\'' || c == '"' || c == ',' || c == ';');
            if t.len() <= 1 || t.contains("://") || seen.contains(t) {
                continue;
            }
            let looks_path = t.starts_with('/')
                || t.starts_with("./")
                || (t.contains('/')
                    && t.chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_'));
            if looks_path {
                seen.insert(t.to_string());
                found.push(PathBuf::from(t));
            }
        }
    }
    found
}

/// Extract environment-variable names a trace reports as required-but-unset.
/// Recognised phrasings (broad but anchored on "environment variable" / "env
/// var" / "must be set" / "is not set" near an UPPER_SNAKE identifier):
///   * `FOO environment variable not set`
///   * `missing required environment variable: API_TOKEN`
///   * `error: BAR must be set`
///   * `KeyError: 'DATABASE_URL'` on a line mentioning environ/env
pub fn extract_required_env_vars(trace: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for line in trace.lines() {
        let lower = line.to_ascii_lowercase();
        let env_context = lower.contains("environment variable")
            || lower.contains("env var")
            || lower.contains("environ")
            || lower.contains("must be set")
            || lower.contains("is not set")
            || lower.contains("not set");
        if !env_context {
            continue;
        }
        // UPPER_SNAKE identifiers, the near-universal convention for env var
        // names. Requiring an underscore keeps precision high: it admits
        // API_TOKEN / DATABASE_URL while rejecting all-caps log words like
        // FATAL / ERROR / WARNING that share a line with "must be set".
        for raw in line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
            let t = raw.trim();
            if t.len() >= 3
                && t.contains('_')
                && t.chars().any(|c| c.is_ascii_alphabetic())
                && t.chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
                && seen.insert(t.to_string())
            {
                found.push(t.to_string());
            }
        }
    }
    found
}

/// Surface non-correctable conditions as actionable diagnostics (Axiom cannot
/// safely auto-fix these — it makes the failure legible rather than fabricating
/// a fix). Currently: disk full (`ENOSPC`) and missing executables
/// (`command not found`). Returns one human-readable line per finding.
pub fn extract_diagnostics(trace: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut push = |s: String, out: &mut Vec<String>| {
        if seen.insert(s.clone()) {
            out.push(s);
        }
    };
    for line in trace.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("no space left on device") {
            push(
                "disk full (ENOSPC) — free space and re-run; Axiom cannot reclaim space safely"
                    .to_string(),
                &mut out,
            );
        }
        if lower.contains("command not found") || lower.contains(": not found") {
            // The command is the colon-segment immediately before the marker:
            // e.g. "bash: line 1: foo: command not found" → "foo".
            let segs: Vec<&str> = line.split(':').map(str::trim).collect();
            let marker = segs
                .iter()
                .position(|s| s.contains("command not found") || *s == "not found");
            if let Some(i) = marker {
                if let Some(cmd) = i.checked_sub(1).and_then(|j| segs.get(j)) {
                    let cmd = cmd.trim();
                    if !cmd.is_empty() && !cmd.contains(char::is_whitespace) {
                        push(
                            format!(
                                "missing executable `{cmd}` — install it or fix PATH; \
                                 Axiom cannot install software"
                            ),
                            &mut out,
                        );
                    }
                }
            }
        }
    }
    out
}

/// Apply the execute-bit heal: if `path` (resolved against `anchor`) is an
/// existing regular file lacking the execute bit, add it (`chmod +x`). Never
/// touches file contents; a no-op on non-Unix or when already executable.
fn heal_permission_denied(path: &Path, anchor: &Path) -> Option<Heal> {
    let resolved = crate::heal_memory::resolve_dir(path, anchor);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let meta = std::fs::metadata(&resolved).ok()?;
        if !meta.is_file() {
            return None;
        }
        let mode = meta.permissions().mode();
        if mode & 0o111 != 0 {
            return None; // already executable
        }
        let mut perms = meta.permissions();
        perms.set_mode(mode | 0o111);
        std::fs::set_permissions(&resolved, perms).ok()?;
        Some(Heal::MadeExecutable(resolved))
    }
    #[cfg(not(unix))]
    {
        let _ = resolved;
        None
    }
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

/// Optional persistence wiring for a supervised run. All paths default to
/// `None` (purely in-memory supervision).
#[derive(Debug, Clone, Default)]
pub struct SupervisorOptions {
    /// Restart budget after the first attempt.
    pub max_restarts: usize,
    /// EMA-merge the run's adapted W̃ into this master vibe on completion.
    pub vibe_path: Option<PathBuf>,
    /// Persistent heal memory (learned + swarm immunity, FIRST/KNOWN/NOVEL).
    pub heal_memory_path: Option<PathBuf>,
    /// MemoryStore root: novel healed failures are written here as `Fix`
    /// memories the proxy's recall layer can surface into Claude's context.
    pub remember_into: Option<PathBuf>,
    /// Working directory for the supervised process and the anchor for
    /// location-invariant immunity: heals under this dir are remembered
    /// *relative* to it and re-anchored here on immunize, so a fix learned in
    /// one checkout (or on another machine, via swarm immunity) applies in
    /// another. `None` → the current process working directory.
    pub anchor: Option<PathBuf>,
}

impl SupervisorOptions {
    /// In-memory supervision with the given restart budget and no persistence.
    pub fn new(max_restarts: usize) -> Self {
        Self {
            max_restarts,
            ..Self::default()
        }
    }
}

/// An anticipatory failure prediction made *before* a command runs, from
/// accumulated immunity (missing learned prerequisites).
#[derive(Debug, Clone)]
pub struct FailurePrediction {
    /// True when the command is predicted to fail as the environment stands.
    pub likely: bool,
    /// Learned prerequisite directories that are currently missing (resolved).
    pub missing_prerequisites: Vec<PathBuf>,
    /// Matured confidence of the underlying immunity (0 when unknown).
    pub confidence: f32,
    /// Human-readable explanation.
    pub rationale: String,
}

/// Predict, before running, whether `command args...` is likely to fail in the
/// current environment — purely from learned immunity (no execution, no model).
/// This is the anticipatory counterpart to the reactive supervisor.
pub fn predict_failure(command: &str, args: &[String], opts: &SupervisorOptions) -> FailurePrediction {
    let anchor = opts
        .anchor
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    let Some(path) = opts.heal_memory_path.as_deref() else {
        return FailurePrediction {
            likely: false,
            missing_prerequisites: Vec::new(),
            confidence: 0.0,
            rationale: "heal memory disabled — no basis to predict".into(),
        };
    };
    let fp = crate::heal_memory::fingerprint(command, args);
    let mem = crate::heal_memory::HealMemory::load(path);
    if mem.record(&fp).is_none() {
        return FailurePrediction {
            likely: false,
            missing_prerequisites: Vec::new(),
            confidence: 0.0,
            rationale: "no prior experience with this command".into(),
        };
    }
    let (missing, confidence) = mem.missing_prerequisites(&fp, &anchor);
    if missing.is_empty() {
        FailurePrediction {
            likely: false,
            missing_prerequisites: missing,
            confidence,
            rationale: "all learned prerequisites already satisfied".into(),
        }
    } else {
        let list = missing
            .iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        FailurePrediction {
            likely: true,
            rationale: format!(
                "predicting failure (confidence {confidence:.2}): {} missing learned prerequisite(s): {list}",
                missing.len()
            ),
            missing_prerequisites: missing,
            confidence,
        }
    }
}

/// Run `command args...` under self-healing supervision.
///
/// Persistence is opt-in via [`SupervisorOptions`]: the run's adapted W̃ can
/// EMA-merge into a master vibe, heals are remembered for prophylactic
/// immunity, and novel healed failures can be written into the recall memory
/// store so the program's lived failure history reaches the reasoning layer.
pub fn run_supervised(
    pipeline: &InferencePipeline,
    command: &str,
    args: &[String],
    opts: &SupervisorOptions,
) -> CResult<RunReport> {
    let max_restarts = opts.max_restarts;
    let vibe_path = opts.vibe_path.as_deref();
    let heal_memory_path = opts.heal_memory_path.as_deref();
    // Anchor = supervised process working directory = the frame heals are made
    // portable against. Defaults to the current process cwd.
    let anchor = opts
        .anchor
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    let mut report = RunReport {
        success: false,
        attempts: 0,
        exit_code: None,
        heals: Vec::new(),
        tension: Vec::new(),
        tokens_absorbed: 0,
        absorption_passes: 0,
        diagnostics: Vec::new(),
    };
    // One session for the whole supervised lifetime: every failure compounds
    // into the same fast-weights.
    let mut states = pipeline.init_session_states()?;
    let mut healed_paths: HashSet<PathBuf> = HashSet::new();
    let mut transient_retries = 0usize;

    // Anticipatory pre-flight: predict failure from learned immunity before the
    // command runs, so the proactive immunization below has a stated rationale.
    let prediction = predict_failure(command, args, opts);
    if prediction.likely {
        println!("[axiom-run] pre-flight: {}", prediction.rationale);
    }

    // Learned immunity: re-create what this program needed in past runs BEFORE
    // the first attempt, so remembered failure modes never recur.
    let command_line = format!("{command} {}", args.join(" "));
    let fp = crate::heal_memory::fingerprint(command, args);
    let mut memory = heal_memory_path.map(crate::heal_memory::HealMemory::load);
    if let Some(mem) = memory.as_ref() {
        for dir in mem.immunize_anchored(&fp, &anchor) {
            let heal = Heal::Immunized(dir.clone());
            println!("[axiom-run] {heal}");
            healed_paths.insert(dir);
            report.heals.push(heal);
        }
    }

    for attempt in 1..=max_restarts + 1 {
        report.attempts = attempt;
        let started = Instant::now();
        let output = match Command::new(command)
            .args(args)
            .current_dir(&anchor)
            .output()
        {
            Ok(o) => o,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                // The program itself isn't executable. Try the exec-bit heal on
                // its path, then retry; if it can't be healed, surface the error.
                if attempt <= max_restarts {
                    if let Some(heal) = heal_permission_denied(Path::new(command), &anchor) {
                        if !healed_paths.contains(Path::new(command)) {
                            println!("[axiom-run]   heal: {heal}");
                            healed_paths.insert(PathBuf::from(command));
                            report.heals.push(heal);
                            println!("[axiom-run] environment healed — restarting");
                            continue;
                        }
                    }
                }
                return Err(candle_core::Error::Msg(format!(
                    "failed to spawn '{command}': {e}"
                )));
            }
            Err(e) => {
                return Err(candle_core::Error::Msg(format!(
                    "failed to spawn '{command}': {e}"
                )))
            }
        };
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
            commit_to_vibe(pipeline, &states, &report, vibe_path);
            finalize_memory(memory.as_mut(), &fp, &command_line, &report, &anchor);
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

        // 2a. Classify the fault against this program's tension history BEFORE
        //     deciding how hard to absorb it.
        let novelty = memory
            .as_mut()
            .map(|mem| mem.observe_failure(&fp, &command_line, ce_before));
        if let Some(n) = novelty {
            println!("[axiom-run]   failure mode: {n} (vs this program's tension history)");
        }

        // 2b. Tension-gated absorption: concentrate gradient effort on
        //     new/surprising faults; lightly reinforce familiar ones.
        let passes = absorption_passes(novelty);
        for _ in 0..passes {
            adapt_session_blocking(pipeline, &mut states, &ids)?;
        }
        report.tokens_absorbed += ids.len();
        report.absorption_passes += passes;
        let ce_after = sequence_ce(pipeline, &states, &ids)?;
        println!(
            "[axiom-run]   tension: CE {ce_before:.3} -> {ce_after:.3} after {passes} absorption pass(es) ({} tokens)",
            ids.len()
        );
        report.tension.push(TensionSample {
            attempt,
            ce_before,
            ce_after,
        });

        // 3. Heal the environment (new heals only — never loop on the same fix).
        let mut applied_new_heal = false;
        let mut new_heal_descs: Vec<String> = Vec::new();
        for path in extract_missing_paths(&trace)
            .into_iter()
            .take(MAX_HEALS_PER_ATTEMPT)
        {
            if healed_paths.contains(&path) {
                continue;
            }
            if let Some(heal) = heal_missing_path(&path, &anchor) {
                println!("[axiom-run]   heal: {heal}");
                healed_paths.insert(path);
                new_heal_descs.push(heal.to_string());
                report.heals.push(heal);
                applied_new_heal = true;
            }
        }

        // 3-exec. A different pathogen class: a file the program tried to run
        // lacked the execute bit. `chmod +x` it (contents untouched).
        for path in extract_permission_denied_paths(&trace)
            .into_iter()
            .take(MAX_HEALS_PER_ATTEMPT)
        {
            if healed_paths.contains(&path) {
                continue;
            }
            if let Some(heal) = heal_permission_denied(&path, &anchor) {
                println!("[axiom-run]   heal: {heal}");
                healed_paths.insert(path);
                new_heal_descs.push(heal.to_string());
                report.heals.push(heal);
                applied_new_heal = true;
            }
        }

        // 3-env. A diagnostic pathogen class: a required environment variable
        // is unset. Axiom cannot fabricate a value, so it does NOT mark a heal
        // (no blind restart) — it surfaces an actionable diagnostic once and
        // remembers the requirement so advisories/`axiom immunity` can tell the
        // operator (and Claude) to export it.
        for var in extract_required_env_vars(&trace) {
            if std::env::var_os(&var).is_some() {
                continue; // already set in the environment — not the problem
            }
            let diag = format!(
                "requires environment variable {var} (unset) — set it and re-run; \
                 Axiom cannot safely fabricate its value"
            );
            if !report.diagnostics.contains(&diag) {
                println!("[axiom-run]   diagnosis: {diag}");
                report.diagnostics.push(diag);
                if let Some(mem) = memory.as_mut() {
                    mem.remember_env_requirement(&fp, &command_line, &[var]);
                }
            }
        }

        // 3-diag. Other non-correctable conditions (disk full, missing
        // executable): surface once as a legible diagnostic, no heal/restart.
        for diag in extract_diagnostics(&trace) {
            if !report.diagnostics.contains(&diag) {
                println!("[axiom-run]   diagnosis: {diag}");
                report.diagnostics.push(diag);
            }
        }

        // 3a. Bridge to the reasoning layer: a NOVEL/first-seen fault we just
        //     healed is exactly the kind of lived experience worth surfacing to
        //     Claude later. Write it into the recall memory store as a `Fix`,
        //     using the measured tension (CE) as its salience. KNOWN faults are
        //     already remembered; we don't re-log them.
        let is_novel = !matches!(novelty, Some(crate::heal_memory::Novelty::Known));
        if applied_new_heal && is_novel {
            if let Some(root) = opts.remember_into.as_deref() {
                remember_failure_fix(
                    pipeline,
                    root,
                    &command_line,
                    report.exit_code,
                    &new_heal_descs,
                    ce_before,
                );
            }
        }

        // 3b. Transient faults: when nothing in the filesystem could be fixed
        //     but the trace shows a recognised transient phrase, waiting IS the
        //     heal — back off briefly and retry (bounded, never blind).
        if !applied_new_heal && transient_retries < MAX_TRANSIENT_RETRIES {
            if let Some(phrase) = detect_transient(&trace) {
                transient_retries += 1;
                let heal = Heal::TransientRetry(phrase);
                println!("[axiom-run]   heal: {heal}");
                report.heals.push(heal);
                std::thread::sleep(std::time::Duration::from_millis(500 * attempt as u64));
                applied_new_heal = true;
            }
        }

        // 4. Continue only when something actually changed; a blind restart of
        //    an unhealed environment would just replay the same failure.
        if !applied_new_heal {
            println!("[axiom-run] no applicable heal for this failure — stopping");
            commit_to_vibe(pipeline, &states, &report, vibe_path);
            finalize_memory(memory.as_mut(), &fp, &command_line, &report, &anchor);
            return Ok(report);
        }
        if attempt == max_restarts + 1 {
            println!("[axiom-run] restart budget exhausted");
            commit_to_vibe(pipeline, &states, &report, vibe_path);
            finalize_memory(memory.as_mut(), &fp, &command_line, &report, &anchor);
            return Ok(report);
        }
        println!("[axiom-run] environment healed — restarting");
    }
    Ok(report)
}

/// Persist what this run taught the heal memory: the failure-tension history is
/// always saved; directory heals are remembered only when the run ultimately
/// SUCCEEDED (a dir created during a failing run is not proven medicine).
fn finalize_memory(
    memory: Option<&mut crate::heal_memory::HealMemory>,
    fp: &str,
    command_line: &str,
    report: &RunReport,
    anchor: &Path,
) {
    let Some(mem) = memory else { return };
    if report.success {
        // Remember directory heals *relative to the anchor* when they live
        // under it, so immunity is portable to other checkouts/machines; heals
        // outside the anchor stay absolute. Immunized dirs are included (the
        // run proved them) — remember_dirs dedups against what's already stored.
        let dirs: Vec<PathBuf> = report
            .heals
            .iter()
            .filter_map(|h| match h {
                Heal::CreatedDirectory(p) | Heal::Immunized(p) => {
                    Some(crate::heal_memory::relativize_dir(p, anchor))
                }
                _ => None,
            })
            .collect();
        mem.remember_dirs(fp, command_line, &dirs);
        // Affinity maturation: if immunity was applied *and* the run succeeded,
        // reinforce this program's confidence (and reset its waning clock).
        let immunity_applied = report
            .heals
            .iter()
            .any(|h| matches!(h, Heal::Immunized(_)));
        if immunity_applied {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            mem.reinforce_immunity(fp, now);
        }
    }
    if let Err(e) = mem.save() {
        eprintln!("[axiom-run] heal memory save skipped: {e}");
    }
}

/// Write a healed novel failure into the recall memory store as a `Fix`, so the
/// proxy's recall layer can later surface it into Claude's context. Best-effort:
/// any failure (embed/open/append) is logged and swallowed — supervision must
/// never break because the recall store is unavailable.
fn remember_failure_fix(
    pipeline: &InferencePipeline,
    root: &Path,
    command_line: &str,
    exit_code: Option<i32>,
    heal_descs: &[String],
    tension_ce: f32,
) {
    let heals = if heal_descs.is_empty() {
        "no environmental heal".to_string()
    } else {
        heal_descs.join("; ")
    };
    let body = format!(
        "Program `{command_line}` failed (exit {exit:?}) and Axiom self-healed it: {heals}. \
         If this command fails again in this environment, apply that fix.",
        exit = exit_code
    );
    let store = match crate::memory_store::MemoryStore::open(root) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[axiom-run] recall memory unavailable ({e}); skipping remember");
            return;
        }
    };
    let embedding = match crate::embedder::embed_text(pipeline, &body) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[axiom-run] embed failed ({e}); skipping remember");
            return;
        }
    };
    let rec = crate::memory_store::MemoryRecord {
        id: uuid::Uuid::new_v4().simple().to_string(),
        scope: "personal".to_string(),
        ts: crate::memory_store::now_secs(),
        kind: crate::memory_store::MemoryKind::Fix,
        body,
        embedding,
        // The failure's tension IS its salience.
        drift_at_ingest: tension_ce,
        supersedes: None,
        tombstone: false,
    };
    match store.append(&rec) {
        Ok(()) => println!(
            "[axiom-run]   remembered fix for the reasoning layer (recall id={})",
            rec.id
        ),
        Err(e) => eprintln!("[axiom-run] remember append failed: {e}"),
    }
}

/// Persist the supervised session's adapted W̃ into the master vibe, when a
/// vibe path was requested and there is anything to persist.
fn commit_to_vibe(
    pipeline: &InferencePipeline,
    states: &[Tensor],
    report: &RunReport,
    vibe_path: Option<&Path>,
) {
    let Some(path) = vibe_path else { return };
    if report.tokens_absorbed == 0 || states.is_empty() {
        return;
    }
    let d_model = match states[0].dim(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut vibe = crate::vibe_memory::MasterVibe::load_or_init(
        path,
        states.len(),
        d_model,
        pipeline.device(),
        crate::vibe_memory::DEFAULT_VIBE_DECAY,
    );
    match vibe.commit_and_save(states) {
        Ok(()) => println!(
            "[axiom-run] failure history persisted to master vibe: {}",
            path.display()
        ),
        Err(e) => eprintln!("[axiom-run] vibe persist skipped: {e}"),
    }
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
    fn extracts_diagnostics_for_disk_full_and_missing_command() {
        assert!(extract_diagnostics("cp: error writing '/out': No space left on device")
            .iter()
            .any(|d| d.contains("disk full")));
        let d = extract_diagnostics("bash: line 1: ghostscript: command not found");
        assert!(d.iter().any(|s| s.contains("ghostscript") && s.contains("missing executable")));
        // Clean trace → no diagnostics.
        assert!(extract_diagnostics("all good here").is_empty());
    }

    #[test]
    fn extracts_required_env_vars_from_varied_phrasings() {
        assert_eq!(
            extract_required_env_vars("error: API_TOKEN environment variable not set"),
            vec!["API_TOKEN"]
        );
        assert_eq!(
            extract_required_env_vars("missing required environment variable: DATABASE_URL"),
            vec!["DATABASE_URL"]
        );
        assert_eq!(
            extract_required_env_vars("FATAL: SECRET_KEY must be set"),
            vec!["SECRET_KEY"]
        );
        // No env context → nothing, even with an UPPER_SNAKE token present.
        assert!(extract_required_env_vars("compiled OK with FLAG_X enabled").is_empty());
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
        // Absolute path → anchor is ignored.
        let heal = heal_missing_path(&file_path, Path::new("/unused")).expect("heal must apply");
        assert_eq!(heal, Heal::CreatedDirectory(base.join("deep")));
        assert!(base.join("deep").exists());
        assert!(!file_path.exists(), "heal must never create the file itself");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn relative_trace_path_heals_under_anchor() {
        let anchor = std::env::temp_dir().join(format!(
            "axiom_anchor_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&anchor).unwrap();
        // A relative path from a trace resolves against the anchor (child cwd).
        let heal = heal_missing_path(Path::new("build/out.bin"), &anchor).expect("heal applies");
        assert_eq!(heal, Heal::CreatedDirectory(anchor.join("build")));
        assert!(anchor.join("build").exists());
        let _ = std::fs::remove_dir_all(&anchor);
    }
}
