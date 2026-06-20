//! Pillar 3 — Autonomy: the self-directed repair loop (`axiom solve`).
//!
//! Pillars 1 and 2 are mechanisms — compress/absorb context, and heal a running
//! program's environment. Pillar 3 is the *autonomous orchestrator* that chains
//! every subsystem into one closed loop that drives a failing target to green
//! and remembers how:
//!
//!   1. **Environment self-heal** (Pillar 2 / `self_heal`) — run the verify
//!      command under the supervisor: missing dirs / exec-bit / transient faults
//!      are repaired, the failure trace is absorbed into W̃ (tension), and
//!      learned immunity is applied. If it goes green, we are done.
//!   2. **Source repair** (hypervisor / `poly_jit`) — if a `source_path` is
//!      given and the environment heal did not suffice, drive the Poly JIT
//!      closed loop: bounded, Q-TTT-ranked, *reversible* source patches. The
//!      source is backed up and restored byte-for-byte if the repair fails.
//!   3. **Verify + remember** — success persists what was learned (heal memory
//!      via the supervisor); the round's provenance is reported.
//!
//! The loop runs up to `max_rounds`, stopping early when a round makes no
//! progress (no new environment heal and no successful source patch) so it can
//! never spin. Honest scope: environment heals are corrective; source repair is
//! the bounded deterministic patch set; Pillar-1 whole-project priming remains
//! the separate `axiom prime` path (the supervisor already absorbs each fault
//! trace into W̃ here).

use std::path::{Path, PathBuf};

use candle_core::Result as CResult;

use crate::claude_backend::ClaudeBackend;
use crate::inference::InferencePipeline;
use crate::poly_jit::{PolyJitEngine, PolyJitRunRequest};
use crate::self_heal::{run_supervised, SupervisorOptions};

/// Knobs for an autonomous solve run.
#[derive(Debug, Clone, Default)]
pub struct SolveOptions {
    /// Max environment-heal + source-repair rounds.
    pub max_rounds: usize,
    /// Restart budget for the environment supervisor within each round.
    pub max_restarts: usize,
    /// Persistent heal memory (learned/anticipatory immunity).
    pub heal_memory_path: Option<PathBuf>,
    /// Working directory for the verify command and the immunity anchor.
    pub anchor: Option<PathBuf>,
    /// When set, enables Poly JIT source repair against this artifact.
    pub source_path: Option<PathBuf>,
    /// Verified-patch store for fleet patch sharing. When set, the loop tries
    /// stored (incl. peer-merged) candidates — re-verified locally — before the
    /// LLM, and records successful fixes here for export.
    pub patch_memory_path: Option<PathBuf>,
}

/// Unified provenance of an autonomous solve.
#[derive(Debug, Clone, Default)]
pub struct SolveReport {
    pub solved: bool,
    pub rounds: usize,
    /// Environment heals applied across all rounds (human-readable).
    pub env_heals: Vec<String>,
    /// True if a Poly JIT source patch ultimately made the target pass.
    pub source_patched: bool,
    /// True if an LLM-proposed, *verifier-gated* source patch made the target
    /// pass (only kept when the verify command independently went green).
    pub llm_patched: bool,
    /// True if a patch from the verified-patch store (incl. peer-merged fleet
    /// patches) was re-verified green locally and applied.
    pub fleet_patched: bool,
    /// Non-correctable diagnostics surfaced (missing env var, disk full, …).
    pub diagnostics: Vec<String>,
    /// Exit code of the last verify attempt.
    pub final_exit: Option<i32>,
    /// Total failure tokens absorbed into W̃ across the loop.
    pub tokens_absorbed: usize,
}

/// Run the autonomous repair loop. Must be called on a large-stack thread (the
/// TTT backward graph recurses) — `axiom solve` does this for you.
pub fn solve(
    pipeline: &InferencePipeline,
    command: &str,
    args: &[String],
    opts: &SolveOptions,
) -> CResult<SolveReport> {
    let mut report = SolveReport::default();
    let max_rounds = opts.max_rounds.max(1);

    for round in 1..=max_rounds {
        report.rounds = round;

        // --- Phase 1: environment self-heal -------------------------------
        let sup = run_supervised(
            pipeline,
            command,
            args,
            &SupervisorOptions {
                max_restarts: opts.max_restarts,
                heal_memory_path: opts.heal_memory_path.clone(),
                anchor: opts.anchor.clone(),
                ..SupervisorOptions::default()
            },
        )?;
        report.final_exit = sup.exit_code;
        report.tokens_absorbed += sup.tokens_absorbed;
        let round_heals = sup.heals.len();
        for h in &sup.heals {
            report.env_heals.push(h.to_string());
        }
        for d in &sup.diagnostics {
            if !report.diagnostics.contains(d) {
                report.diagnostics.push(d.clone());
            }
        }
        if sup.success {
            report.solved = true;
            println!("[axiom-solve] round {round}: environment heal solved it");
            return Ok(report);
        }

        // --- Phase 2: source repair (Poly JIT), reversible ----------------
        // The repair phases need a target file. Use the caller-named source if
        // given; otherwise localize the faulty file directly from the verify
        // command's own output (compiler/test runner paths), which is what makes
        // the loop language-general rather than hand-driven and Rust-bound.
        let located: Option<PathBuf> = if opts.source_path.is_none() {
            let working_dir = opts.anchor.as_deref();
            let (_still_failing, trace) = run_verify_capture(command, args, working_dir);
            let root = opts
                .anchor
                .clone()
                .or_else(|| std::env::current_dir().ok())
                .unwrap_or_else(|| PathBuf::from("."));
            let found = crate::fault_locate::best_source(&trace, &root);
            if let Some(ref p) = found {
                println!("[axiom-solve] round {round}: localized fault to {}", p.display());
            }
            found
        } else {
            None
        };
        if let Some(src) = opts.source_path.as_deref().or(located.as_deref()) {
            // Single portable patch key, shared by fleet-reuse (try) and record,
            // so a patch is only ever re-applied to the file it was learned for.
            let rel = rel_path_for(src, opts.anchor.as_deref());
            let backup = std::fs::read_to_string(src).ok();
            let engine = PolyJitEngine::default();
            let run_req = PolyJitRunRequest {
                session_id: format!("solve-{round}"),
                command: command.to_string(),
                args: args.to_vec(),
                working_dir: opts
                    .anchor
                    .as_ref()
                    .map(|p| p.display().to_string()),
                source_path: Some(src.display().to_string()),
            };
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| candle_core::Error::Msg(format!("solve runtime: {e}")))?;
            let poly = rt
                .block_on(engine.run_with_feedback(run_req, |_diag| async { Ok(()) }))
                .map_err(|e| candle_core::Error::Msg(format!("poly-jit: {e}")))?;
            if poly.passed {
                report.source_patched = poly.patched;
                report.solved = true;
                report.final_exit = Some(0);
                println!("[axiom-solve] round {round}: source repair solved it (patched={})", poly.patched);
                return Ok(report);
            }
            // Failed repair → restore the artifact byte-for-byte.
            if let Some(b) = backup {
                let _ = std::fs::write(src, b);
            }

            // --- Phase 2.5: fleet patch reuse (re-verified locally) -----------
            // Before spending an LLM call, try patches the verified-patch store
            // already knows for this failure (including peer-merged fleet
            // patches). SAFETY: each candidate is applied through
            // PatchMemory::try_candidates, which re-runs THIS node's verify and
            // keeps it only if green — a peer's patch is never trusted blindly.
            if let Some(pm_path) = opts.patch_memory_path.as_deref() {
                let fp = crate::heal_memory::fingerprint(command, args);
                let pm = crate::patch_memory::PatchMemory::load(pm_path);
                let working_dir = opts.anchor.as_deref();
                if let Some(sha) =
                    pm.try_candidates(&fp, &rel, src, || run_verify(command, args, working_dir))
                {
                    report.fleet_patched = true;
                    report.solved = true;
                    report.final_exit = Some(0);
                    println!(
                        "[axiom-solve] round {round}: fleet patch {} verified green locally",
                        &sha[..sha.len().min(12)]
                    );
                    return Ok(report);
                }
            }

            // --- Phase 3: autonomous LLM repair (verifier-gated, reversible) ---
            // When the deterministic Poly-JIT patches don't fix it, ask the LLM
            // backend for a patch — but keep it ONLY if the verify command then
            // goes green. The model never gets to "fix" anything that isn't
            // independently verified; a rejected patch is rolled back byte-for-
            // byte. This is the autonomous code-repair step: general proposals,
            // grounded acceptance.
            if let Some(backend) = ClaudeBackend::from_env() {
                let working_dir = opts.anchor.as_deref();
                // Feed the model the REAL compile/test failure trace. The
                // supervisor's `sup.diagnostics` only holds non-correctable
                // *environment* diagnoses (missing env var, disk full) and is
                // usually empty for ordinary source failures, so capture the
                // verify command's stdout/stderr directly. Fall back to the env
                // diagnoses only if the command produced no output.
                let (_still_failing, trace) = run_verify_capture(command, args, working_dir);
                let failure_output = if trace.trim().is_empty() {
                    sup.diagnostics.join("\n")
                } else {
                    trace
                };
                let solved = llm_repair_round(
                    &backend,
                    command,
                    args,
                    working_dir,
                    src,
                    &failure_output,
                );
                if solved {
                    report.llm_patched = true;
                    report.solved = true;
                    report.final_exit = Some(0);
                    println!(
                        "[axiom-solve] round {round}: LLM-proposed patch verified green"
                    );
                    // Record the verified fix so this node — and, via signed
                    // gossip, the fleet — can reuse it next time this failure
                    // signature recurs (always re-verified before trust).
                    if let Some(pm_path) = opts.patch_memory_path.as_deref() {
                        if let Ok(content) = std::fs::read_to_string(src) {
                            let fp = crate::heal_memory::fingerprint(command, args);
                            let mut pm =
                                crate::patch_memory::PatchMemory::load(pm_path);
                            pm.record_verified(&fp, &rel, &content);
                            let _ = pm.save(pm_path);
                        }
                    }
                    return Ok(report);
                }
            }
        }

        // Source repair loops to its own step cap within one call, so another
        // round only helps if the *environment* advanced this round. If no env
        // heal was applied and we are still failing, stop rather than spin.
        if round_heals == 0 {
            println!("[axiom-solve] round {round}: no progress — stopping");
            break;
        }
        println!("[axiom-solve] round {round}: environment advanced, continuing");
    }

    Ok(report)
}

/// Bounded number of LLM repair attempts per round (override with
/// `AXIOM_SOLVE_LLM_ATTEMPTS`). Each attempt gets the previous attempt's verifier
/// output as feedback — multi-step debugging, not one shot.
fn llm_repair_attempts() -> usize {
    std::env::var("AXIOM_SOLVE_LLM_ATTEMPTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(3)
}

/// Build an advisory "focus the fix at line N" hint from a verifier failure for
/// the file under repair, or "" when the trace blames no line in it. Recomputed
/// per attempt so it always reflects the latest failure. Trailing blank line so
/// it drops cleanly out of the prompt when empty.
fn fault_hint_from(failure: &str, source_path: &Path) -> String {
    let lines = crate::fault_locate::line_hints_for(failure, source_path);
    if lines.is_empty() {
        return String::new();
    }
    let shown: Vec<String> = lines.iter().take(5).map(|l| l.to_string()).collect();
    format!(
        "The failure points at {} line(s): {}. Focus the fix there.\n\n",
        source_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        shown.join(", "),
    )
}

/// Ask the LLM backend for a corrected source and apply it under the
/// verifier-gated, reversible, *iterative* policy: up to `llm_repair_attempts()`
/// tries, each re-prompted with the latest verifier failure output. Returns true
/// iff some attempt made the verify command pass. Thin wrapper over
/// [`apply_verified_patch_iterative`] (the unit-tested core).
fn llm_repair_round(
    backend: &ClaudeBackend,
    command: &str,
    args: &[String],
    working_dir: Option<&Path>,
    source_path: &Path,
    failure_output: &str,
) -> bool {
    let max_attempts = llm_repair_attempts();
    apply_verified_patch_iterative(
        command,
        args,
        working_dir,
        source_path,
        max_attempts,
        failure_output,
        |failure, original| {
            // Compute the line hint from the CURRENT failure each attempt — the
            // iterative loop re-prompts with fresh verifier output, so a hint
            // baked once before the loop would point at a stale line on retries.
            let fault_hint = fault_hint_from(failure, source_path);
            let prompt = format!(
                "A verification command is failing and you must fix the source so it passes.\n\n\
                 Command: {command} {args}\n\n\
                 Failure output:\n{failure}\n\n\
                 {fault_hint}\
                 Current contents of {path}:\n```\n{original}\n```\n\n\
                 Return ONLY the complete corrected file contents — no explanation, no commentary, \
                 no markdown code fences.",
                args = args.join(" "),
                path = source_path.display(),
            );
            match backend.generate(&prompt, 8192) {
                Ok(text) => {
                    let candidate = strip_code_fences(&text);
                    // Guard against a truncated full-file rewrite (response hit
                    // the token cap): if the corrected file comes back
                    // dramatically shorter than a non-trivial original, treat it
                    // as untrustworthy and skip — never risk persisting a cut-off
                    // file that only partially verifies.
                    if original.len() > 1024 && candidate.len() < original.len() / 2 {
                        eprintln!(
                            "[axiom-solve] LLM repair skipped (candidate {} B vs original {} B — likely truncated)",
                            candidate.len(),
                            original.len()
                        );
                        None
                    } else {
                        Some(candidate)
                    }
                }
                Err(e) => {
                    eprintln!("[axiom-solve] LLM repair skipped (backend error: {e})");
                    None
                }
            }
        },
    )
}

/// Verifier-gated, reversible, iterative patch application (the testable core of
/// autonomous repair). For up to `max_attempts`, `propose(failure, original)` is
/// given the latest verifier failure output (seeded with `initial_failure`) plus
/// the original source and returns a candidate; the candidate is written, the
/// verify command re-run under a timeout, and **kept only if it goes green**. On
/// any rejection the original is restored byte-for-byte and the captured failure
/// trace feeds the next attempt. Returns true iff some attempt verified green.
fn apply_verified_patch_iterative<P>(
    command: &str,
    args: &[String],
    working_dir: Option<&Path>,
    source_path: &Path,
    max_attempts: usize,
    initial_failure: &str,
    mut propose: P,
) -> bool
where
    P: FnMut(&str, &str) -> Option<String>,
{
    let Ok(original) = std::fs::read_to_string(source_path) else {
        return false;
    };
    let mut failure = initial_failure.to_string();
    for attempt in 1..=max_attempts.max(1) {
        let Some(candidate) = propose(&failure, &original) else {
            break; // no usable candidate / backend error → stop
        };
        // No-op or empty candidate → nothing to verify.
        if candidate.trim().is_empty() || candidate == original {
            break;
        }
        if let Err(e) = std::fs::write(source_path, &candidate) {
            // A failed write may have already truncated the file — restore first.
            eprintln!("[axiom-solve] patch write failed; restoring original ({e})");
            let _ = std::fs::write(source_path, &original);
            break;
        }
        let (green, trace) = run_verify_capture(command, args, working_dir);
        if green {
            if attempt > 1 {
                eprintln!("[axiom-solve] LLM repair verified green on attempt {attempt}");
            }
            return true;
        }
        // Rejected: roll back byte-for-byte, feed this attempt's trace forward.
        if let Err(e) = std::fs::write(source_path, &original) {
            eprintln!("[axiom-solve] WARNING: failed to restore original after rejected patch: {e}");
        }
        if !trace.trim().is_empty() {
            failure = trace;
        }
    }
    false
}

/// Default wall-clock cap for a single verify run (override with
/// `AXIOM_SOLVE_VERIFY_TIMEOUT_SECS`). A hung verifier must never wedge the solve
/// loop with a candidate already on disk — on timeout we kill it and report
/// failure so the caller rolls back.
fn verify_timeout() -> std::time::Duration {
    let secs = std::env::var("AXIOM_SOLVE_VERIFY_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(120);
    std::time::Duration::from_secs(secs)
}

/// Portable key for a patched source file: anchor-relative when the source lives
/// under the working dir (so the same logical file matches across checkouts and
/// on different nodes), else the bare file name, else the full display path. This
/// MUST be computed identically at record time and at try time so a patch is only
/// ever re-applied to the file it was learned for.
fn rel_path_for(src: &Path, anchor: Option<&Path>) -> String {
    if let Some(a) = anchor {
        if let Ok(stripped) = src.strip_prefix(a) {
            let s = stripped.to_string_lossy();
            if !s.is_empty() {
                return s.into_owned();
            }
        }
    }
    src.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| src.display().to_string())
}

/// Run the verify command, returning whether it exited 0. Bounded by
/// [`verify_timeout`] (delegates to [`run_verify_capture`]).
fn run_verify(command: &str, args: &[String], working_dir: Option<&Path>) -> bool {
    run_verify_capture(command, args, working_dir).0
}

/// Run the verify command under a wall-clock timeout, returning
/// `(success, combined stdout+stderr trace)` capped to the last ~4000 chars.
/// stdout/stderr are drained on threads so a chatty verifier can't deadlock on a
/// full pipe; on timeout the child is killed and `success` is false.
fn run_verify_capture(
    command: &str,
    args: &[String],
    working_dir: Option<&Path>,
) -> (bool, String) {
    use std::io::Read;
    use std::process::Stdio;
    use std::time::Instant;

    let mut cmd = std::process::Command::new(command);
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(dir) = working_dir {
        cmd.current_dir(dir);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return (false, format!("failed to spawn verify command: {e}")),
    };

    // Drain both pipes concurrently to avoid a full-buffer deadlock.
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let out_h = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(ref mut o) = stdout {
            let _ = o.read_to_string(&mut s);
        }
        s
    });
    let err_h = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(ref mut e) = stderr {
            let _ = e.read_to_string(&mut s);
        }
        s
    });

    let deadline = Instant::now() + verify_timeout();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break Some(st),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break None;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(_) => break None,
        }
    };

    let mut trace = out_h.join().unwrap_or_default();
    trace.push_str(&err_h.join().unwrap_or_default());
    // Keep the tail (failures usually surface there); char-boundary safe.
    let trace: String = {
        let chars: Vec<char> = trace.chars().collect();
        if chars.len() > 4000 {
            chars[chars.len() - 4000..].iter().collect()
        } else {
            trace
        }
    };

    match status {
        Some(st) => (st.success(), trace),
        None if timed_out => (false, format!("{trace}\n[verify timed out]")),
        None => (false, trace),
    }
}

/// Strip a single leading/trailing markdown code fence if the model wrapped the
/// file in ``` despite being told not to. Leaves un-fenced content untouched.
fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return text.to_string();
    }
    let mut lines: Vec<&str> = trimmed.lines().collect();
    // Only unwrap a *paired* fence — stripping a lone opening fence would corrupt
    // legitimate content that happens to start with ```.
    let has_opening = matches!(lines.first(), Some(l) if l.starts_with("```"));
    let has_closing = lines.len() > 1 && matches!(lines.last(), Some(l) if l.trim() == "```");
    if !(has_opening && has_closing) {
        return text.to_string();
    }
    lines.remove(0); // opening fence (possibly ```rust)
    lines.pop(); // closing fence
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("axiom_solve_{}_{}", std::process::id(), name));
        p
    }

    // verify command: pass only if the file contains "FIXED".
    fn grep_fixed(path: &Path) -> (String, Vec<String>) {
        (
            "sh".to_string(),
            vec![
                "-c".to_string(),
                format!("grep -q FIXED '{}'", path.display()),
            ],
        )
    }

    #[test]
    fn verified_patch_is_kept_only_when_it_goes_green() {
        let path = tmp("green.txt");
        std::fs::write(&path, "BROKEN").unwrap();
        let (cmd, args) = grep_fixed(&path);
        let kept = apply_verified_patch_iterative(&cmd, &args, None, &path, 1, "", |_f, _o| {
            Some("FIXED".into())
        });
        assert!(kept, "a patch that makes verify pass must be kept");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "FIXED");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejected_patch_is_rolled_back_byte_for_byte() {
        let path = tmp("red.txt");
        std::fs::write(&path, "BROKEN").unwrap();
        let (cmd, args) = grep_fixed(&path);
        // Proposes a different-but-still-failing source → verify stays red.
        let kept = apply_verified_patch_iterative(&cmd, &args, None, &path, 1, "", |_f, _o| {
            Some("STILL WRONG".into())
        });
        assert!(!kept, "a patch that does not pass verify must be rejected");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "BROKEN",
            "rejected patch must restore the original byte-for-byte"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn noop_or_empty_candidate_is_not_applied() {
        let path = tmp("noop.txt");
        std::fs::write(&path, "SAME").unwrap();
        let (cmd, args) = grep_fixed(&path);
        assert!(!apply_verified_patch_iterative(&cmd, &args, None, &path, 1, "", |_f, o| Some(o.to_string())));
        assert!(!apply_verified_patch_iterative(&cmd, &args, None, &path, 1, "", |_f, _o| Some("   ".into())));
        assert!(!apply_verified_patch_iterative(&cmd, &args, None, &path, 1, "", |_f, _o| None));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "SAME");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn iterative_repair_retries_with_feedback_until_green() {
        // First attempt (no prior failure) proposes a still-wrong source; once it
        // receives a non-empty failure trace it proposes the fix. Verifies the
        // loop retries AND threads the verifier output back into propose().
        let path = tmp("iter.txt");
        std::fs::write(&path, "BROKEN").unwrap();
        // Verify emits a marker on stderr when failing, so the rejected attempt's
        // trace is non-empty and gets threaded back as feedback (grep -q alone is
        // silent, which would give no feedback to test with).
        let cmd = "sh".to_string();
        let args = vec![
            "-c".to_string(),
            format!("grep -q FIXED '{}' || {{ echo NEED_FIXED >&2; exit 1; }}", path.display()),
        ];
        let mut calls = 0usize;
        let mut saw_feedback = false;
        let kept = apply_verified_patch_iterative(&cmd, &args, None, &path, 3, "", |failure, _o| {
            calls += 1;
            if failure.trim().is_empty() {
                Some("STILL WRONG".into()) // attempt 1: will be rejected
            } else {
                saw_feedback = true; // attempt 2 saw the prior verifier trace
                Some("FIXED".into())
            }
        });
        assert!(kept, "iterative repair should succeed once it proposes FIXED");
        assert!(calls >= 2, "should have retried at least once (calls={calls})");
        assert!(saw_feedback, "later attempts must receive the failure trace");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "FIXED");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn run_verify_reflects_exit_code() {
        assert!(run_verify("sh", &["-c".into(), "exit 0".into()], None));
        assert!(!run_verify("sh", &["-c".into(), "exit 1".into()], None));
    }

    #[test]
    fn run_verify_capture_returns_stdout_stderr_and_status() {
        let (ok, trace) = run_verify_capture(
            "sh",
            &["-c".into(), "echo out_marker; echo err_marker >&2; exit 3".into()],
            None,
        );
        assert!(!ok, "exit 3 → not success");
        assert!(trace.contains("out_marker"), "stdout captured");
        assert!(trace.contains("err_marker"), "stderr captured");
    }

    #[test]
    fn strip_code_fences_unwraps_only_paired_fences() {
        assert_eq!(strip_code_fences("fn main() {}"), "fn main() {}");
        assert_eq!(strip_code_fences("```rust\nfn main() {}\n```"), "fn main() {}");
        assert_eq!(strip_code_fences("```\nabc\ndef\n```"), "abc\ndef");
        // A lone opening fence (no closing) must NOT be stripped — that would
        // corrupt content that legitimately begins with ```.
        assert_eq!(strip_code_fences("```rust\nfn x() {}"), "```rust\nfn x() {}");
    }
}
