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

use std::path::PathBuf;

use candle_core::Result as CResult;

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
