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
        if let Some(src) = opts.source_path.as_deref() {
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

            // --- Phase 3: autonomous LLM repair (verifier-gated, reversible) ---
            // When the deterministic Poly-JIT patches don't fix it, ask the LLM
            // backend for a patch — but keep it ONLY if the verify command then
            // goes green. The model never gets to "fix" anything that isn't
            // independently verified; a rejected patch is rolled back byte-for-
            // byte. This is the autonomous code-repair step: general proposals,
            // grounded acceptance.
            if let Some(backend) = ClaudeBackend::from_env() {
                let failure_output = sup.diagnostics.join("\n");
                let working_dir = opts.anchor.as_deref();
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

/// Ask the LLM backend for a corrected source, then apply it under the
/// verifier-gated reversible policy. Thin wrapper: it builds the prompt and
/// delegates the apply/verify/rollback to [`apply_verified_patch`] (which is the
/// unit-tested core). Returns true iff a proposed patch made the verify pass.
fn llm_repair_round(
    backend: &ClaudeBackend,
    command: &str,
    args: &[String],
    working_dir: Option<&Path>,
    source_path: &Path,
    failure_output: &str,
) -> bool {
    apply_verified_patch(command, args, working_dir, source_path, |original| {
        let prompt = format!(
            "A verification command is failing and you must fix the source so it passes.\n\n\
             Command: {command} {args}\n\n\
             Failure output:\n{failure}\n\n\
             Current contents of {path}:\n```\n{original}\n```\n\n\
             Return ONLY the complete corrected file contents — no explanation, no commentary, \
             no markdown code fences.",
            args = args.join(" "),
            failure = failure_output,
            path = source_path.display(),
        );
        match backend.generate(&prompt, 4096) {
            Ok(text) => Some(strip_code_fences(&text)),
            Err(e) => {
                eprintln!("[axiom-solve] LLM repair skipped (backend error: {e})");
                None
            }
        }
    })
}

/// Verifier-gated, reversible patch application (the testable core of autonomous
/// repair). `propose` is given the current source and returns a candidate; the
/// candidate is written, the verify command re-run, and the patch **kept only if
/// it goes green**. On any failure the original is restored byte-for-byte.
fn apply_verified_patch<P>(
    command: &str,
    args: &[String],
    working_dir: Option<&Path>,
    source_path: &Path,
    propose: P,
) -> bool
where
    P: FnOnce(&str) -> Option<String>,
{
    let Ok(original) = std::fs::read_to_string(source_path) else {
        return false;
    };
    let Some(candidate) = propose(&original) else {
        return false;
    };
    // No-op or empty candidate → nothing to verify.
    if candidate.trim().is_empty() || candidate == original {
        return false;
    }
    if std::fs::write(source_path, &candidate).is_err() {
        return false;
    }
    if run_verify(command, args, working_dir) {
        true
    } else {
        // Rejected: roll back byte-for-byte so a bad patch never persists.
        let _ = std::fs::write(source_path, &original);
        false
    }
}

/// Run the verify command and report whether it exited 0. Output is inherited so
/// it surfaces in the solve log; callers that need the text capture it elsewhere.
fn run_verify(command: &str, args: &[String], working_dir: Option<&Path>) -> bool {
    let mut cmd = std::process::Command::new(command);
    cmd.args(args);
    if let Some(dir) = working_dir {
        cmd.current_dir(dir);
    }
    matches!(cmd.status(), Ok(status) if status.success())
}

/// Strip a single leading/trailing markdown code fence if the model wrapped the
/// file in ``` despite being told not to. Leaves un-fenced content untouched.
fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return text.to_string();
    }
    let mut lines: Vec<&str> = trimmed.lines().collect();
    if lines.first().map(|l| l.starts_with("```")).unwrap_or(false) {
        lines.remove(0); // opening fence (possibly ```rust)
    }
    if lines.last().map(|l| l.trim() == "```").unwrap_or(false) {
        lines.pop(); // closing fence
    }
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
        let kept = apply_verified_patch(&cmd, &args, None, &path, |_orig| Some("FIXED".into()));
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
        let kept = apply_verified_patch(&cmd, &args, None, &path, |_orig| Some("STILL WRONG".into()));
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
        assert!(!apply_verified_patch(&cmd, &args, None, &path, |o| Some(o.to_string())));
        assert!(!apply_verified_patch(&cmd, &args, None, &path, |_| Some("   ".into())));
        assert!(!apply_verified_patch(&cmd, &args, None, &path, |_| None));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "SAME");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn run_verify_reflects_exit_code() {
        assert!(run_verify("sh", &["-c".into(), "exit 0".into()], None));
        assert!(!run_verify("sh", &["-c".into(), "exit 1".into()], None));
    }

    #[test]
    fn strip_code_fences_unwraps_only_when_fenced() {
        assert_eq!(strip_code_fences("fn main() {}"), "fn main() {}");
        assert_eq!(strip_code_fences("```rust\nfn main() {}\n```"), "fn main() {}");
        assert_eq!(strip_code_fences("```\nabc\ndef\n```"), "abc\ndef");
    }
}
