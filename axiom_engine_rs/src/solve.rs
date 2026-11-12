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
//!   2. **Source repair** — if the environment heal did not suffice, localize
//!      the blamed files (`fault_locate::repair_targets`: non-test files first,
//!      then the modules a blamed test imports, then the test itself) and, per
//!      file, run the deterministic mutation layer (`mutation_repair`:
//!      comparison / off-by-one / boolean / logic / arithmetic / swapped-args /
//!      missing-return / shell-test candidates, blamed lines first, never an
//!      assertion line), then stored fixes, then the LLM. Every candidate is
//!      *reversible*: the original is journaled to disk before the first write
//!      (`source_journal`) and restored byte-for-byte unless the verify command
//!      goes green — also when the solve is interrupted or killed.
//!   3. **Verify + remember** — success persists what was learned (heal memory
//!      via the supervisor); the round's provenance is reported.
//!
//! The loop runs up to `max_rounds`, stopping early when a round makes no
//! progress (no new environment heal and no successful source patch) so it can
//! never spin. Honest scope: environment heals are corrective; source repair is
//! the bounded deterministic mutation set; Pillar-1 whole-project priming remains
//! the separate `axiom prime` path (the supervisor already absorbs each fault
//! trace into W̃ here).

use std::path::{Path, PathBuf};

use candle_core::Result as CResult;

use crate::backend_router::ChatBackend;
use crate::inference::InferencePipeline;
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
    /// When set, source repair targets this file instead of localizing one
    /// from the verify command's output.
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
    /// True if the solve left a source file changed by a deterministic phase:
    /// a mutation repair, or a stored fix (in which case `fleet_patched` is
    /// set too).
    pub source_patched: bool,
    /// True if an LLM-proposed, *verifier-gated* source patch made the target
    /// pass (only kept when the verify command independently went green).
    pub llm_patched: bool,
    /// True if a fix from the verified-patch store (incl. peer-merged fleet
    /// patches) matched this file's exact content, was applied, and was
    /// re-verified green locally. Implies `source_patched`.
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

    // A previous solve that was killed with a candidate on disk left its
    // original in the journal: put those files right before doing anything.
    for recovery in crate::source_journal::recover() {
        use crate::source_journal::RecoveryOutcome;
        match recovery.outcome {
            RecoveryOutcome::Restored => println!(
                "[axiom-solve] an earlier solve was killed mid-repair: restored {} to its original content",
                recovery.target.display()
            ),
            RecoveryOutcome::LeftChanged { saved_original } => println!(
                "[axiom-solve] an earlier solve was killed mid-repair, and {} has changed since; \
                 it was left as it is, and its content from before that solve is saved at {}",
                recovery.target.display(),
                saved_original.display()
            ),
            RecoveryOutcome::Deferred { saved_original } => println!(
                "[axiom-solve] an earlier solve was killed mid-repair, and {} cannot be read or \
                 written right now; nothing was changed, and its content from before that solve \
                 is saved at {}",
                recovery.target.display(),
                saved_original.display()
            ),
            RecoveryOutcome::AlreadyOriginal | RecoveryOutcome::TargetMissing => {}
        }
    }

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
                // The command is a verifier: bound it like every other run.
                timeout: Some(verify_timeout()),
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

        // --- Phase 2/2.5/3: source repair, reversible ---------------------
        // The repair phases need a target file. Use the caller-named source if
        // given; otherwise localize candidate files directly from the verify
        // command's own output (compiler/test runner paths) — ranked
        // most-blamed first — and try each in turn. This is what makes the loop
        // language-general and able to handle a failure that points at more than
        // one file, rather than hand-driven and single-file.
        let candidates: Vec<PathBuf> = if let Some(s) = opts.source_path.as_deref() {
            vec![s.to_path_buf()]
        } else {
            let working_dir = opts.anchor.as_deref();
            let (_still_failing, trace) = run_verify_capture(command, args, working_dir);
            let root = opts
                .anchor
                .clone()
                .or_else(|| std::env::current_dir().ok())
                .unwrap_or_else(|| PathBuf::from("."));
            let mut cands = crate::fault_locate::repair_targets(&trace, &root);
            if cands.is_empty() {
                // Nothing in the output names a file (e.g. the run was killed
                // on a timeout before printing): use what the command names.
                cands = crate::fault_locate::targets_from_command(args, &root);
            }
            if let Some(first) = cands.first() {
                println!(
                    "[axiom-solve] round {round}: localized {} candidate file(s); first: {}",
                    cands.len(),
                    first.display()
                );
            }
            cands
        };
        for src in &candidates {
            if attempt_source_repair(
                command,
                args,
                opts,
                round,
                &sup.diagnostics,
                src,
                &mut report,
            )? {
                return Ok(report);
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

/// Run the reversible repair phases against ONE target file, with its
/// original journaled to disk first ([`crate::source_journal`]). Returns
/// `Ok(true)` and sets the matching `report` flags if some phase made the
/// verify command pass; the fix is then kept. In every other case — no phase
/// fixed it, a phase returned an error, the thread panicked, the process was
/// interrupted — the file is put back byte-for-byte, so the caller can safely
/// try the next candidate file.
#[allow(clippy::too_many_arguments)]
fn attempt_source_repair(
    command: &str,
    args: &[String],
    opts: &SolveOptions,
    round: usize,
    env_diagnostics: &[String],
    src: &Path,
    report: &mut SolveReport,
) -> CResult<bool> {
    // No journal, no writes: a file whose original cannot be saved to disk
    // first is not touched.
    let journal = match crate::source_journal::protect(src) {
        Ok(journal) => journal,
        Err(e) => {
            println!(
                "[axiom-solve] round {round}: not repairing {} (its original could not be journaled: {e})",
                src.display()
            );
            return Ok(false);
        }
    };
    let solved = run_repair_phases(command, args, opts, round, env_diagnostics, src, report)?;
    if solved {
        journal.commit();
    }
    // Otherwise dropping `journal` restores the original.
    Ok(solved)
}

/// The repair phases for one journaled file: Phase 2.1 (deterministic mutation
/// repair, `crate::mutation_repair`) → Phase 2.5 (stored-fix reuse, re-verified
/// locally) → Phase 3 (verifier-gated LLM repair). Each phase rolls its own
/// rejected candidates back; [`attempt_source_repair`] is the backstop.
///
/// There is no phase before 2.1 any more. The Poly-JIT patch set that used to
/// run first only knew fixture-shaped rewrites — `exit 1` → `exit 0`, a marker
/// rename, replacing the whole file with a script that exits 0 — and it wrote
/// a 4096-character excerpt over the file to try them. On real code that
/// turned a failing check into a passing one without fixing anything.
#[allow(clippy::too_many_arguments)]
fn run_repair_phases(
    command: &str,
    args: &[String],
    opts: &SolveOptions,
    round: usize,
    env_diagnostics: &[String],
    src: &Path,
    report: &mut SolveReport,
) -> CResult<bool> {
    // Where this file lives: repository identity + repository-relative path.
    // Shared by record and reuse below, so a stored fix is only ever tried on
    // the file it was learned for, in the repository it was learned in. `None`
    // (the file is under no identifiable root) disables both for this file.
    let scope = crate::patch_memory::repo_scope(src, opts.anchor.as_deref());
    // Name used in log lines only.
    let rel = scope
        .as_ref()
        .map(|s| s.rel_path.clone())
        .unwrap_or_else(|| src.display().to_string());

    // --- Phase 2.1: deterministic mutation repair (verifier-gated) ----------
    // A bounded, ranked set of single-edit mutations (comparison / off-by-one /
    // boolean / logic / arithmetic / swapped args / missing return / shell
    // test operators), focused on the lines the failure blames and never on an
    // assertion line. Each candidate is one verifier run under the same
    // reversible policy as the LLM path: kept only if the command goes green,
    // otherwise rolled back byte-for-byte. No model, no network.
    //
    // Never on a test file: a blind single-token edit of the oracle can only
    // weaken it (swapping two names in a test's setup made a QuixBugs test
    // pass without touching the buggy program). Phase 3 may still edit a test
    // it judges to be wrong.
    if !crate::fault_locate::is_test_path(src) {
        let working_dir = opts.anchor.as_deref();
        let (_still_failing, trace) = run_verify_capture(command, args, working_dir);
        if let Ok(original) = std::fs::read_to_string(src) {
            let plan = crate::mutation_repair::plan(
                &original,
                src,
                &trace,
                crate::mutation_repair::max_candidates(),
            );
            if !plan.is_empty() {
                let budget = plan.len();
                let mut queue = plan.into_iter().enumerate();
                let mut applied: Option<String> = None;
                // AXIOM_REPAIR_VERBOSE=1 lists every candidate as it is tried.
                let verbose = std::env::var("AXIOM_REPAIR_VERBOSE").is_ok_and(|v| v == "1");
                let solved = apply_verified_patch_iterative(
                    command,
                    args,
                    working_dir,
                    src,
                    budget,
                    &trace,
                    |_failure, _original| {
                        let (i, c) = queue.next()?;
                        if verbose {
                            println!("[axiom-solve]   candidate {}/{budget}: {}", i + 1, c.label);
                        }
                        applied = Some(c.label);
                        Some(c.source)
                    },
                );
                if solved {
                    report.source_patched = true;
                    report.solved = true;
                    report.final_exit = Some(0);
                    println!(
                        "[axiom-solve] round {round}: mutation repair verified green ({}) in {}",
                        applied.as_deref().unwrap_or("?"),
                        rel
                    );
                    record_verified_fix(opts, scope.as_ref(), command, args, &original, src);
                    return Ok(true);
                }
            }
        }
    }

    // --- Phase 2.5: stored-fix reuse (pre-image matched, re-verified) -------
    // Before spending an LLM call, try fixes the verified-patch store already
    // knows for this failure (including peer-merged fleet patches). SAFETY:
    // PatchMemory::try_candidates only considers a fix recorded against the
    // exact bytes this file holds now, in this repository — so a file that has
    // changed since (newer or uncommitted work, another project's same-named
    // file) is never written to — and then re-runs THIS node's verify, keeping
    // the fix only if green.
    if let (Some(pm_path), Some(scope)) = (opts.patch_memory_path.as_deref(), scope.as_ref()) {
        let fp = crate::heal_memory::fingerprint(command, args);
        let pm = crate::patch_memory::PatchMemory::load(pm_path);
        let working_dir = opts.anchor.as_deref();
        if let Some(sha) =
            pm.try_candidates(&fp, scope, src, || run_verify(command, args, working_dir))
        {
            // The source file was changed: say so, alongside where the fix
            // came from.
            report.source_patched = true;
            report.fleet_patched = true;
            report.solved = true;
            report.final_exit = Some(0);
            println!(
                "[axiom-solve] round {round}: stored fix {} applied to {rel} and verified green locally",
                &sha[..sha.len().min(12)]
            );
            return Ok(true);
        }
    }

    // --- Phase 3: autonomous LLM repair (verifier-gated, reversible) --------
    // When neither a mutation nor a stored fix repairs it, ask the LLM backend
    // for a patch — but keep it ONLY if the verify command then goes green. The
    // model never gets to "fix" anything that isn't independently verified; a
    // rejected patch is rolled back byte-for-byte.
    //
    // The backend is whatever `AXIOM_BACKEND` selects for the repair paths:
    // `claude` (API key) or, in an experimental build, `cli` (a logged-in
    // vendor CLI driven as a subprocess). A configuration error (e.g. a CLI
    // outside its version pin) disables LLM repair for this run rather than
    // guessing — the deterministic phases above have already had their turn.
    let backend = match crate::backend_live::repair_backend_from_env() {
        Ok(backend) => backend,
        Err(e) => {
            eprintln!("[axiom-solve] LLM repair unavailable: {e}");
            None
        }
    };
    if let Some(backend) = backend {
        let working_dir = opts.anchor.as_deref();
        // Feed the model the REAL compile/test failure trace. The supervisor's
        // diagnostics only hold non-correctable *environment* diagnoses and are
        // usually empty for ordinary source failures, so capture the verify
        // command's stdout/stderr directly; fall back to env diagnoses only if
        // the command produced no output.
        let (_still_failing, trace) = run_verify_capture(command, args, working_dir);
        let failure_output = if trace.trim().is_empty() {
            env_diagnostics.join("\n")
        } else {
            trace
        };
        // The pre-image a verified LLM fix is recorded against.
        let before_llm = std::fs::read_to_string(src).ok();
        let solved = llm_repair_round(
            backend.as_ref(),
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
            println!("[axiom-solve] round {round}: LLM-proposed patch verified green");
            // Record the verified fix so this node — and, via signed gossip, the
            // fleet — can reuse it next time this failure signature recurs
            // (always re-verified before trust).
            if let Some(original) = before_llm.as_deref() {
                record_verified_fix(opts, scope.as_ref(), command, args, original, src);
            }
            return Ok(true);
        }
    }
    Ok(false)
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
    backend: &dyn ChatBackend,
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
            match backend.complete(&prompt, 8192) {
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
        let before = std::fs::metadata(source_path)
            .and_then(|m| m.modified())
            .ok();
        if let Err(e) = crate::source_journal::write(source_path, candidate.as_bytes()) {
            // A failed write may have already truncated the file — restore first.
            eprintln!("[axiom-solve] patch write failed; restoring original ({e})");
            let _ = crate::source_journal::write(source_path, original.as_bytes());
            break;
        }
        // Same-size candidate in the same second: without this, Python's
        // __pycache__ (and other mtime-keyed caches) run the ORIGINAL and the
        // verifier rejects a correct fix. See `agentic::mark_fresh`.
        crate::agentic::mark_fresh(source_path, before);
        let (green, trace) = run_verify_capture(command, args, working_dir);
        if green {
            if attempt > 1 {
                eprintln!("[axiom-solve] repair verified green on attempt {attempt}");
            }
            return true;
        }
        // Rejected: roll back byte-for-byte, feed this attempt's trace forward.
        let candidate_mtime = std::fs::metadata(source_path)
            .and_then(|m| m.modified())
            .ok();
        if let Err(e) = crate::source_journal::write(source_path, original.as_bytes()) {
            eprintln!(
                "[axiom-solve] WARNING: failed to restore original after rejected patch: {e}"
            );
        }
        // Newer than the rejected candidate's build output, or the next
        // candidate's verify could be served the stale artifact.
        crate::agentic::mark_fresh(source_path, candidate_mtime);
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

/// Record a fix that just verified green in the patch store: `src` read
/// `original` before the fix and holds the fixed content now. Stored as a diff
/// keyed by `scope` (repository + path) and the hash of `original`, so it can
/// later be applied only to that exact file state. A no-op when there is no
/// patch store or no repository scope for this file.
fn record_verified_fix(
    opts: &SolveOptions,
    scope: Option<&crate::patch_memory::RepoScope>,
    command: &str,
    args: &[String],
    original: &str,
    src: &Path,
) {
    let (Some(pm_path), Some(scope)) = (opts.patch_memory_path.as_deref(), scope) else {
        return;
    };
    let Ok(fixed) = std::fs::read_to_string(src) else {
        return;
    };
    let fp = crate::heal_memory::fingerprint(command, args);
    let mut pm = crate::patch_memory::PatchMemory::load(pm_path);
    pm.record_verified(&fp, scope, original, &fixed);
    if let Err(e) = pm.save(pm_path) {
        eprintln!("[axiom-solve] could not save the patch store: {e}");
    }
}

/// Resolves a POSIX-compatible shell for running the `sh`/`sh -c "..."`
/// verify commands used throughout the deterministic eval fixtures
/// ([`crate::agentic_eval::builtin_cases`]) and this module's own test
/// suite. On Unix this is always `"sh"` -- unconditionally present, nothing
/// changes there. On Windows, a bare `"sh"` command often fails even on a
/// machine with Git for Windows installed, because `git.exe` being on PATH
/// does not put its bundled `sh.exe` on PATH too. This checks the common
/// Git-for-Windows install locations first; if none exist, it falls back to
/// the bare `"sh"` unchanged -- so a machine with no POSIX shell at all
/// fails exactly as it did before this function existed, rather than being
/// silently papered over.
pub fn posix_shell() -> String {
    if cfg!(windows) {
        let candidates = [
            r"C:\Program Files\Git\bin\sh.exe",
            r"C:\Program Files\Git\usr\bin\sh.exe",
            r"C:\Program Files (x86)\Git\bin\sh.exe",
            r"C:\Program Files (x86)\Git\usr\bin\sh.exe",
        ];
        for c in candidates {
            if std::path::Path::new(c).exists() {
                return c.to_string();
            }
        }
    }
    "sh".to_string()
}

/// Run the verify command, returning whether it exited 0. Bounded by
/// [`verify_timeout`] (delegates to [`run_verify_capture`]).
pub fn run_verify(command: &str, args: &[String], working_dir: Option<&Path>) -> bool {
    run_verify_capture(command, args, working_dir).0
}

/// Run the verify command under a wall-clock timeout, returning
/// `(success, combined stdout+stderr trace)` capped to the last ~4000 chars.
/// stdout/stderr are drained on threads so a chatty verifier can't deadlock on a
/// full pipe; on timeout the child is killed and `success` is false.
/// Characters of verifier output kept from the start and from the end.
const TRACE_HEAD_CHARS: usize = 2000;
const TRACE_TAIL_CHARS: usize = 4000;

/// Bound a verifier's output to what the repair loop keeps: the first
/// [`TRACE_HEAD_CHARS`] and the last [`TRACE_TAIL_CHARS`] characters, with an
/// elision marker between. The head matters as much as the tail: a test
/// runner prints the first failure (and, with `RUST_BACKTRACE=1`, its frames
/// citing the faulty line) near the top, and a long backtrace or a run of
/// property-test failures can push every `file:line` citation out of a
/// tail-only window. Char-boundary safe.
pub fn bound_trace(trace: &str) -> String {
    let n = trace.chars().count();
    if n <= TRACE_HEAD_CHARS + TRACE_TAIL_CHARS {
        return trace.to_string();
    }
    let head: String = trace.chars().take(TRACE_HEAD_CHARS).collect();
    let tail: String = trace.chars().skip(n - TRACE_TAIL_CHARS).collect();
    format!(
        "{head}\n[… {} chars elided …]\n{tail}",
        n - TRACE_HEAD_CHARS - TRACE_TAIL_CHARS
    )
}

pub fn run_verify_capture(
    command: &str,
    args: &[String],
    working_dir: Option<&Path>,
) -> (bool, String) {
    let mut cmd = std::process::Command::new(command);
    cmd.args(args);
    if let Some(dir) = working_dir {
        cmd.current_dir(dir);
    }
    // Bounded, and the verifier's whole process group is killed on timeout,
    // so a runaway test (or a subprocess it started) cannot hang the loop.
    match crate::proc_bound::output_bounded(&mut cmd, Some(verify_timeout())) {
        Ok(out) => {
            let mut trace = String::from_utf8_lossy(&out.stdout).into_owned();
            trace.push_str(&String::from_utf8_lossy(&out.stderr));
            (out.status.success(), bound_trace(&trace))
        }
        Err(e) => (false, format!("failed to spawn verify command: {e}")),
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
            posix_shell(),
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
        assert!(!apply_verified_patch_iterative(
            &cmd,
            &args,
            None,
            &path,
            1,
            "",
            |_f, o| Some(o.to_string())
        ));
        assert!(!apply_verified_patch_iterative(
            &cmd,
            &args,
            None,
            &path,
            1,
            "",
            |_f, _o| Some("   ".into())
        ));
        assert!(!apply_verified_patch_iterative(
            &cmd,
            &args,
            None,
            &path,
            1,
            "",
            |_f, _o| None
        ));
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
        let cmd = posix_shell();
        let args = vec![
            "-c".to_string(),
            format!(
                "grep -q FIXED '{}' || {{ echo NEED_FIXED >&2; exit 1; }}",
                path.display()
            ),
        ];
        let mut calls = 0usize;
        let mut saw_feedback = false;
        let kept =
            apply_verified_patch_iterative(&cmd, &args, None, &path, 3, "", |failure, _o| {
                calls += 1;
                if failure.trim().is_empty() {
                    Some("STILL WRONG".into()) // attempt 1: will be rejected
                } else {
                    saw_feedback = true; // attempt 2 saw the prior verifier trace
                    Some("FIXED".into())
                }
            });
        assert!(
            kept,
            "iterative repair should succeed once it proposes FIXED"
        );
        assert!(
            calls >= 2,
            "should have retried at least once (calls={calls})"
        );
        assert!(
            saw_feedback,
            "later attempts must receive the failure trace"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "FIXED");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn run_verify_reflects_exit_code() {
        assert!(run_verify(
            &posix_shell(),
            &["-c".into(), "exit 0".into()],
            None
        ));
        assert!(!run_verify(
            &posix_shell(),
            &["-c".into(), "exit 1".into()],
            None
        ));
    }

    #[test]
    fn bound_trace_keeps_head_and_tail_of_long_output() {
        let short = "panicked at src/lib.rs:3:5";
        assert_eq!(bound_trace(short), short);
        let first = "thread 't' panicked at src/x.rs:1:1\n   at ./src/mesh.rs:296:51\n";
        let noise = "   frame at /rustc/abc/library/core/src/panicking.rs:80:14\n".repeat(400);
        let last = "error: test failed, to rerun pass `--lib`";
        let b = bound_trace(&format!("{first}{noise}{last}"));
        assert!(b.contains("src/mesh.rs:296:51"), "head kept");
        assert!(b.ends_with(last), "tail kept");
        assert!(b.contains("chars elided"));
        assert!(b.chars().count() < TRACE_HEAD_CHARS + TRACE_TAIL_CHARS + 64);
        // Multi-byte characters never split.
        let wide = "é".repeat(10_000);
        assert!(bound_trace(&wide).starts_with('é'));
    }

    #[test]
    fn run_verify_capture_returns_stdout_stderr_and_status() {
        let (ok, trace) = run_verify_capture(
            &posix_shell(),
            &[
                "-c".into(),
                "echo out_marker; echo err_marker >&2; exit 3".into(),
            ],
            None,
        );
        assert!(!ok, "exit 3 → not success");
        assert!(trace.contains("out_marker"), "stdout captured");
        assert!(trace.contains("err_marker"), "stderr captured");
    }

    #[test]
    fn strip_code_fences_unwraps_only_paired_fences() {
        assert_eq!(strip_code_fences("fn main() {}"), "fn main() {}");
        assert_eq!(
            strip_code_fences("```rust\nfn main() {}\n```"),
            "fn main() {}"
        );
        assert_eq!(strip_code_fences("```\nabc\ndef\n```"), "abc\ndef");
        // A lone opening fence (no closing) must NOT be stripped — that would
        // corrupt content that legitimately begins with ```.
        assert_eq!(
            strip_code_fences("```rust\nfn x() {}"),
            "```rust\nfn x() {}"
        );
    }
}
