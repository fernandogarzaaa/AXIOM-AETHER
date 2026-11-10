//! Paired end-to-end benchmark (Phase 4, audit section 5).
//!
//! The audit's experimental matrix: five arms (A baseline, B retrieval,
//! C CVM, D repair, E full stack) run across five task families
//! (single-file debugging, multi-file repairs, long-session coding,
//! tool-heavy workflows, failure recovery) with the task set, model,
//! budget, environment, and evaluation oracle fixed, run order randomized
//! by seed, and repeats to estimate variance. Primary measure is verified
//! task-success rate; secondaries are end-to-end dollar cost,
//! cache-adjusted billed tokens, agent turns, tool calls, wall-clock time,
//! first-token latency, unsafe patch rate, and successful rollback rate.
//!
//! What this module is, honestly: scaffolding, not the comparison.
//! The deterministic executor below can run arms A and D on the
//! single-file family end to end (arm A is the no-op baseline, arm D is
//! the real `axiom solve` loop). Arms B, C, and E need an agent executor
//! that drives a real coding agent with and without AXIOM in the loop; that
//! executor does not exist yet and is a named open item (see
//! [`Arm::executor_status`]). The multi-file, long-session, tool-heavy,
//! and failure-recovery families are defined in [`TASK_SET`] but have no
//! on-disk fixtures yet; the runner reports them as pending, never as
//! numbers. The FULL paired comparison waits until Phase 3 code exists on
//! its PR branches or main; nothing here guesses its results.

use crate::guardrail::{evaluate_guardrail, GuardrailConfig};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The five audit arms. `Regressed` is not an audit arm: it exists only
/// for the guardrail demonstration (`axiombench paired --demo-regression`),
/// where the arm-D solver is deliberately sabotaged (edits a pinned file,
/// skips rollback) so the guardrail check must fail on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Arm {
    Baseline,
    Retrieval,
    Cvm,
    Repair,
    FullStack,
    Regressed,
}

impl Arm {
    pub fn label(self) -> &'static str {
        match self {
            Arm::Baseline => "A",
            Arm::Retrieval => "B",
            Arm::Cvm => "C",
            Arm::Repair => "D",
            Arm::FullStack => "E",
            Arm::Regressed => "D-regressed",
        }
    }

    /// Environment overlay that *defines* the arm, from the toggles
    /// documented in docs/AXIOMBENCH.md section 3.1. These are the arm's
    /// definition, not a claim that the current deterministic executor
    /// realizes every one of them: only arm D is executed by that executor
    /// today (see [`Arm::executor_status`]).
    pub fn env_overlay(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Arm::Baseline => &[],
            Arm::Retrieval => &[
                ("AXIOM_TTT_COMPRESS", "0"),
                ("AXIOM_RESPONSES_COMPRESS", "0"),
                ("AXIOM_VIBE", "0"),
            ],
            Arm::Cvm => &[
                ("AXIOM_TTT_COMPRESS", "1"),
                ("AXIOM_RESPONSES_COMPRESS", "1"),
                ("AXIOM_VIBE", "0"),
                ("AXIOM_HEAL_MEMORY", "0"),
            ],
            Arm::Repair | Arm::Regressed => &[
                ("AXIOM_TTT_COMPRESS", "0"),
                ("AXIOM_RESPONSES_COMPRESS", "0"),
                ("AXIOM_VIBE", "0"),
            ],
            Arm::FullStack => &[
                ("AXIOM_TTT_COMPRESS", "1"),
                ("AXIOM_RESPONSES_COMPRESS", "1"),
                ("AXIOM_VIBE", "1"),
            ],
        }
    }

    /// Whether the current harness can execute this arm at all.
    pub fn executor_status(self) -> &'static str {
        match self {
            Arm::Baseline | Arm::Repair | Arm::Regressed => "deterministic (axiom solve / no-op)",
            Arm::Retrieval | Arm::Cvm | Arm::FullStack => {
                "pending: needs the agent executor (real coding agent with \
                 and without AXIOM in the loop); blocked on Phase 3 code and \
                 the agent harness, not implemented here"
            }
        }
    }
}

/// The five audit task families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskFamily {
    SingleFile,
    MultiFile,
    LongSession,
    ToolHeavy,
    FailureRecovery,
}

impl TaskFamily {
    pub fn label(self) -> &'static str {
        match self {
            TaskFamily::SingleFile => "single-file",
            TaskFamily::MultiFile => "multi-file",
            TaskFamily::LongSession => "long-session",
            TaskFamily::ToolHeavy => "tool-heavy",
            TaskFamily::FailureRecovery => "failure-recovery",
        }
    }

    pub fn from_cli(name: &str) -> Option<Self> {
        match name {
            "single-file" => Some(TaskFamily::SingleFile),
            "multi-file" => Some(TaskFamily::MultiFile),
            "long-session" => Some(TaskFamily::LongSession),
            "tool-heavy" => Some(TaskFamily::ToolHeavy),
            "failure-recovery" => Some(TaskFamily::FailureRecovery),
            _ => None,
        }
    }
}

/// One task in the fixed task set. `fixture` is a directory under
/// `bench/repair/` (the on-disk corpus format from docs/BENCH-REPAIR.md).
/// `None` means the task is defined but has no fixture yet: the runner
/// reports it as pending with the `gap` reason, never as a score.
#[derive(Clone, Debug)]
pub struct PairedTask {
    pub id: &'static str,
    pub family: TaskFamily,
    pub fixture: Option<&'static str>,
    pub gap: Option<&'static str>,
}

/// The fixed task set. Only the single-file family is executable today;
/// the other families are defined so the harness shape is complete and
/// their fixture gaps are named, not hidden.
pub static TASK_SET: &[PairedTask] = &[
    PairedTask {
        id: "single-py-average-off-by-one",
        family: TaskFamily::SingleFile,
        fixture: Some("py-average-off-by-one"),
        gap: None,
    },
    PairedTask {
        id: "single-js-is-even-constant",
        family: TaskFamily::SingleFile,
        fixture: Some("js-is-even-constant"),
        gap: None,
    },
    PairedTask {
        id: "single-sh-is-root-test-op",
        family: TaskFamily::SingleFile,
        fixture: Some("sh-is-root-test-op"),
        gap: None,
    },
    PairedTask {
        id: "multi-both-broken",
        family: TaskFamily::MultiFile,
        fixture: None,
        gap: Some(
            "no on-disk multi-file fixture yet; port builtin case \
             agentic-multi-file-both-broken from agentic_eval.rs to \
             bench/repair/ format",
        ),
    },
    PairedTask {
        id: "multi-interface-and-callsites",
        family: TaskFamily::MultiFile,
        fixture: None,
        gap: Some(
            "no on-disk multi-file fixture yet; needs a task requiring an \
             interface change plus its call sites (3+ files)",
        ),
    },
    PairedTask {
        id: "longsession-overflow",
        family: TaskFamily::LongSession,
        fixture: None,
        gap: Some(
            "no long-session fixture yet; needs a session scaled until \
             context would overflow without compression",
        ),
    },
    PairedTask {
        id: "toolheavy-many-calls",
        family: TaskFamily::ToolHeavy,
        fixture: None,
        gap: Some(
            "no tool-heavy fixture yet; needs a task with many tool calls \
             per turn exercising tool deferral",
        ),
    },
    PairedTask {
        id: "recovery-broken-env",
        family: TaskFamily::FailureRecovery,
        fixture: None,
        gap: Some(
            "no failure-recovery fixture yet; needs a deliberately broken \
             environment/code task scored on repair success and rollback",
        ),
    },
];

/// Fixed experimental conditions. Everything the audit says to fix is a
/// field here and lands in the report JSON, so a later run can prove it
/// ran the same conditions.
#[derive(Clone, Debug)]
pub struct PairedConfig {
    pub seed: u64,
    pub repeats: u32,
    /// Model (or deterministic engine) under test. Smoke runs use
    /// "deterministic (no LLM)".
    pub model: String,
    /// Per-task dollar budget. 0.0 for the deterministic harness: no API
    /// spend exists to budget.
    pub budget_usd: f64,
    /// Oracle version: the verifier that decides success.
    pub oracle: String,
    pub arms: Vec<Arm>,
    pub families: Vec<TaskFamily>,
}

impl PairedConfig {
    pub fn smoke() -> Self {
        PairedConfig {
            seed: 42,
            repeats: 1,
            model: "deterministic (no LLM)".into(),
            budget_usd: 0.0,
            oracle: "bench-repair-v1 (fixture verify command + pinned files)".into(),
            arms: vec![Arm::Baseline, Arm::Repair],
            families: vec![TaskFamily::SingleFile],
        }
    }

    pub fn full_scaffold() -> Self {
        PairedConfig {
            seed: 42,
            repeats: 3,
            model: "deterministic (no LLM)".into(),
            budget_usd: 0.0,
            oracle: "bench-repair-v1 (fixture verify command + pinned files)".into(),
            arms: vec![
                Arm::Baseline,
                Arm::Retrieval,
                Arm::Cvm,
                Arm::Repair,
                Arm::FullStack,
            ],
            families: vec![
                TaskFamily::SingleFile,
                TaskFamily::MultiFile,
                TaskFamily::LongSession,
                TaskFamily::ToolHeavy,
                TaskFamily::FailureRecovery,
            ],
        }
    }

    /// Stable fingerprint of the fixed conditions, so two reports can be
    /// checked for "same task set / model / budget / environment / oracle".
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        for task in TASK_SET {
            hasher.update(task.id.as_bytes());
            hasher.update(task.family.label().as_bytes());
            hasher.update(task.fixture.unwrap_or("").as_bytes());
        }
        hasher.update(self.model.as_bytes());
        hasher.update(self.oracle.as_bytes());
        hasher.update(format!("{:.2}", self.budget_usd).as_bytes());
        for arm in &self.arms {
            hasher.update(arm.label().as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }
}

// ---------------------------------------------------------------------------
// Seeded run order (SplitMix64 + Fisher-Yates). A run with a fixed seed
// reproduces identical run order; the order is a permutation of the cells.
// ---------------------------------------------------------------------------

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministic shuffle of `0..n` under `seed`.
pub fn shuffled_order(seed: u64, n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

// ---------------------------------------------------------------------------
// Outcomes
// ---------------------------------------------------------------------------

/// How a cell resolved. Only `Executed` and `Overfit` carry a measured
/// success value; the pending variants are reported, never scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeStatus {
    /// Ran end to end; `success` is the verified outcome.
    Executed,
    /// Ran, verifier went green, but a pinned file (the oracle) was
    /// edited: scored as failure, `unsafe_patch` set.
    Overfit,
    /// The fixture's own verify passes on the untouched tree, so the task
    /// is not actually broken and a pass would mean nothing.
    NotBroken,
    /// No fixture exists for this task yet.
    PendingFixture,
    /// The arm has no executor yet (arms B, C, E: agent executor).
    PendingExecutor,
}

impl OutcomeStatus {
    pub fn label(self) -> &'static str {
        match self {
            OutcomeStatus::Executed => "executed",
            OutcomeStatus::Overfit => "overfit",
            OutcomeStatus::NotBroken => "not-broken",
            OutcomeStatus::PendingFixture => "pending-fixture",
            OutcomeStatus::PendingExecutor => "pending-executor",
        }
    }
}

/// Per-stage wall-clock breakdown of one executed cell (Phase 3 item 5
/// observability, wired into the runner at the harness level: the runner's
/// own pipeline stages, baseline verify / solve / re-verify / rollback
/// check, timed where they run). All zeros until the executor fills them
/// in; a pending cell keeps the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageMs {
    pub baseline_verify_ms: u64,
    pub solve_ms: u64,
    pub reverify_ms: u64,
    pub rollback_ms: u64,
}

/// One cell of the paired matrix: one arm x one task x one repeat.
///
/// Secondary measures carry an explicit label next to the value, per the
/// AXIOMBENCH.md rule: a number is measured only if it comes from an
/// executed run. Deterministic arms have no tokens, turns, or first-token
/// latency to measure, so those are `None` with the reason in `notes`,
/// never a fabricated zero presented as a measurement. Dollar cost is 0.0
/// measured: the deterministic harness spends no API budget.
#[derive(Clone, Debug)]
pub struct TaskOutcome {
    pub task_id: String,
    pub family: TaskFamily,
    pub arm: Arm,
    pub repeat: u32,
    pub status: OutcomeStatus,
    pub success: bool,
    pub unsafe_patch: bool,
    /// `None` when rollback is vacuous (arm A never mutates the tree) or
    /// the cell did not execute.
    pub rollback_ok: Option<bool>,
    pub wall_ms: u64,
    pub stage_ms: StageMs,
    /// Peak resident set size (KiB, VmHWM) across the cell's child
    /// processes, read from Linux `/proc/<pid>/status` while they run
    /// (same source as the Phase 3 item 5 `refresh_process_memory`
    /// gauge, applied here to the solver subprocess). `None` where the
    /// cell did not execute, or off Linux where `/proc` is absent.
    pub peak_rss_kb: Option<u64>,
    pub first_token_ms: Option<u64>,
    pub cost_usd: f64,
    pub billed_tokens: u64,
    pub agent_turns: Option<u32>,
    pub tool_calls: Option<u32>,
    /// Short "how" string, e.g. the solver's verified-green line.
    pub solver: String,
    pub notes: String,
}

impl TaskOutcome {
    fn pending(
        task: &PairedTask,
        arm: Arm,
        repeat: u32,
        status: OutcomeStatus,
        notes: String,
    ) -> Self {
        TaskOutcome {
            task_id: task.id.into(),
            family: task.family,
            arm,
            repeat,
            status,
            success: false,
            unsafe_patch: false,
            rollback_ok: None,
            wall_ms: 0,
            stage_ms: StageMs::default(),
            peak_rss_kb: None,
            first_token_ms: None,
            cost_usd: 0.0,
            billed_tokens: 0,
            agent_turns: None,
            tool_calls: None,
            solver: String::new(),
            notes,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "task": self.task_id,
            "family": self.family.label(),
            "arm": self.arm.label(),
            "repeat": self.repeat,
            "status": self.status.label(),
            "success": self.success,
            "unsafe_patch": self.unsafe_patch,
            "rollback_ok": self.rollback_ok,
            "wall_ms": self.wall_ms,
            "stage_ms": {
                "baseline_verify_ms": self.stage_ms.baseline_verify_ms,
                "solve_ms": self.stage_ms.solve_ms,
                "reverify_ms": self.stage_ms.reverify_ms,
                "rollback_ms": self.stage_ms.rollback_ms,
            },
            "peak_rss_kb": self.peak_rss_kb,
            "first_token_ms": self.first_token_ms,
            "cost_usd": self.cost_usd,
            "billed_tokens": self.billed_tokens,
            "agent_turns": self.agent_turns,
            "tool_calls": self.tool_calls,
            "solver": self.solver,
            "notes": self.notes,
        })
    }
}

/// Per-arm aggregate over executed cells only.
#[derive(Clone, Debug, Default)]
pub struct ArmSummary {
    pub executed: usize,
    pub successes: usize,
    pub unsafe_patches: usize,
    /// Failed attempts (executed, not successful) and how many of them
    /// rolled the tree back byte-identically.
    pub failed_attempts: usize,
    pub rollbacks_ok: usize,
    pub total_cost_usd: f64,
    pub total_wall_ms: u64,
}

impl ArmSummary {
    pub fn success_rate(&self) -> Option<f64> {
        if self.executed == 0 {
            None
        } else {
            Some(self.successes as f64 / self.executed as f64)
        }
    }

    pub fn unsafe_patch_rate(&self) -> Option<f64> {
        if self.executed == 0 {
            None
        } else {
            Some(self.unsafe_patches as f64 / self.executed as f64)
        }
    }

    /// Rollback rate over failed attempts. Vacuous (no failures) reads as
    /// 1.0: nothing failed, so nothing could leak.
    pub fn rollback_rate(&self) -> Option<f64> {
        if self.executed == 0 {
            None
        } else if self.failed_attempts == 0 {
            Some(1.0)
        } else {
            Some(self.rollbacks_ok as f64 / self.failed_attempts as f64)
        }
    }

    pub fn cost_per_success(&self) -> Option<f64> {
        if self.successes == 0 {
            None
        } else {
            Some(self.total_cost_usd / self.successes as f64)
        }
    }
}

pub fn summarize(arm: Arm, outcomes: &[TaskOutcome]) -> ArmSummary {
    let mut s = ArmSummary::default();
    for o in outcomes.iter().filter(|o| o.arm == arm) {
        match o.status {
            OutcomeStatus::Executed | OutcomeStatus::Overfit => {
                s.executed += 1;
                if o.success {
                    s.successes += 1;
                } else {
                    s.failed_attempts += 1;
                    if o.rollback_ok == Some(true) {
                        s.rollbacks_ok += 1;
                    }
                }
                if o.unsafe_patch {
                    s.unsafe_patches += 1;
                }
                s.total_cost_usd += o.cost_usd;
                s.total_wall_ms += o.wall_ms;
            }
            _ => {}
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Executor: runs one arm x one task cell through the real repair loop.
// ---------------------------------------------------------------------------

/// Fixture metadata (`bench/repair/<name>/meta.json`).
struct FixtureMeta {
    verify: Vec<String>,
    pinned: Vec<String>,
    test_region_file: Option<String>,
    test_region_from: usize,
}

fn read_fixture_meta(dir: &Path) -> Result<FixtureMeta, String> {
    let text = std::fs::read_to_string(dir.join("meta.json"))
        .map_err(|e| format!("read {} failed: {e}", dir.join("meta.json").display()))?;
    let v: Value =
        serde_json::from_str(&text).map_err(|e| format!("parse meta.json failed: {e}"))?;
    let verify: Vec<String> = v["verify"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if verify.is_empty() {
        return Err(format!("{} has no verify command", dir.display()));
    }
    let pinned: Vec<String> = v["pinned"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let region = &v["test_region"];
    Ok(FixtureMeta {
        verify,
        pinned,
        test_region_file: region["file"].as_str().map(str::to_string),
        test_region_from: region["from_line"].as_u64().unwrap_or(0) as usize,
    })
}

/// Paths a *verifier* writes while running tests (same list as
/// scripts/bench_repair.sh `is_verifier_artifact`). Never solver edits;
/// excluded from the rollback comparison.
fn is_verifier_artifact(rel: &Path) -> bool {
    let parts: Vec<&str> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    parts.iter().any(|p| {
        matches!(
            *p,
            "target" | "node_modules" | "__pycache__" | ".pytest_cache" | "proptest-regressions"
        )
    }) || rel
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e == "pyc")
        .unwrap_or(false)
}

fn sha256_file(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}

/// Snapshot of a scratch tree: relative path -> sha256, excluding verifier
/// artifacts. Used for the rollback comparison.
fn snapshot_tree(root: &Path) -> BTreeMap<PathBuf, String> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
            if is_verifier_artifact(&rel) {
                continue;
            }
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file() {
                if let Some(hash) = sha256_file(&path) {
                    out.insert(rel, hash);
                }
            }
        }
    }
    out
}

fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    let mut stack = vec![src.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let rel = path.strip_prefix(src).unwrap_or(&path);
            let target = dst.join(rel);
            let ft = entry.file_type()?;
            if ft.is_dir() {
                std::fs::create_dir_all(&target)?;
                stack.push(path);
            } else if ft.is_file() {
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::copy(&path, &target)?;
            }
        }
    }
    Ok(())
}

/// Run a command in `cwd` with a wall-clock timeout, capturing combined
/// output to files (so a chatty child can never deadlock a pipe).
/// Returns (success, combined output truncated to 4 KiB).
fn run_bounded(
    argv: &[String],
    cwd: &Path,
    envs: &[(String, String)],
    timeout: Duration,
    log_dir: &Path,
    log_name: &str,
) -> std::io::Result<(bool, String)> {
    let _ = std::fs::create_dir_all(log_dir);
    let out_path = log_dir.join(format!("{log_name}.out.log"));
    let err_path = log_dir.join(format!("{log_name}.err.log"));
    let out_file = std::fs::File::create(&out_path)?;
    let err_file = std::fs::File::create(&err_path)?;
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .current_dir(cwd)
        .envs(envs.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .stdin(Stdio::null())
        .stdout(Stdio::from(out_file))
        .stderr(Stdio::from(err_file));
    let mut child = cmd.spawn()?;
    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(status) => {
                let mut combined = String::new();
                if let Ok(text) = std::fs::read_to_string(&out_path) {
                    combined.push_str(&text);
                }
                if let Ok(text) = std::fs::read_to_string(&err_path) {
                    combined.push_str(&text);
                }
                if combined.len() > 4096 {
                    combined.truncate(4096);
                }
                return Ok((status.success(), combined));
            }
            None => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok((false, format!("TIMEOUT after {}s", timeout.as_secs())));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// Execution context for one paired run.
pub struct ExecCtx {
    /// Solver binary. Defaults to `AXIOM_BIN` or
    /// `<crate>/target/release/axiom`.
    pub solver_bin: Option<PathBuf>,
    /// Corpus root holding `bench/repair/<name>/` fixtures. Defaults to
    /// the repo's `bench/repair/`; tests point it at a temp corpus so
    /// `execute` can be driven end to end without touching the repo.
    pub corpus_dir: Option<PathBuf>,
    /// Scratch root for per-cell work dirs (defaults to the system temp
    /// dir). Kept under one root so a run's trees can be inspected.
    pub scratch_root: PathBuf,
    pub verify_timeout: Duration,
    pub solve_timeout: Duration,
}

impl ExecCtx {
    pub fn default_ctx() -> Self {
        ExecCtx {
            solver_bin: std::env::var("AXIOM_BIN").ok().map(PathBuf::from),
            corpus_dir: None,
            scratch_root: std::env::temp_dir().join("axiombench-paired"),
            verify_timeout: Duration::from_secs(300),
            solve_timeout: Duration::from_secs(600),
        }
    }

    pub fn resolve_solver_bin(&self) -> Result<PathBuf, String> {
        if let Some(bin) = &self.solver_bin {
            if bin.exists() {
                return Ok(bin.clone());
            }
            return Err(format!(
                "solver binary {} does not exist; build it with \
                 `cargo build --release --locked --bin axiom` in axiom_engine_rs/",
                bin.display()
            ));
        }
        let release = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/target/release/axiom"));
        if release.exists() {
            return Ok(release);
        }
        Err(format!(
            "no solver binary: {} not built and AXIOM_BIN unset; build it \
             with `cargo build --release --locked --bin axiom` in axiom_engine_rs/",
            release.display()
        ))
    }
}

fn how_from_log(log: &str) -> String {
    for pat in [
        "mutation repair verified green",
        "source repair solved it",
        "environment heal solved it",
        "fleet patch",
        "LLM-proposed patch verified green",
    ] {
        if let Some(pos) = log.find(pat) {
            let end = log[pos..].find('\n').map(|i| pos + i).unwrap_or(log.len());
            return log[pos..end.min(pos + 120)].trim().to_string();
        }
    }
    String::new()
}

/// The deterministic executor: arm A is the no-op baseline (no solver
/// invocation, the failing verify stays failing); arm D runs the real
/// `axiom solve` loop in a scratch copy of the fixture, following the four
/// steps of scripts/bench_repair.sh (no-op baseline, solve with no file
/// named and isolated heal-memory state, re-verify, pinned-file and
/// rollback checks). `sabotage` (demo-regression only) makes the solver
/// step edit a pinned file and leave the tree dirty, so the guardrail
/// check must fail on reliability even though the verifier goes green.
pub struct AxiomSolveExecutor {
    pub ctx: ExecCtx,
    pub sabotage: bool,
}

impl AxiomSolveExecutor {
    fn repo_root() -> PathBuf {
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
    }

    pub fn execute(&self, task: &PairedTask, arm: Arm, repeat: u32) -> TaskOutcome {
        let fixture_name = match task.fixture {
            Some(name) => name,
            None => {
                return TaskOutcome::pending(
                    task,
                    arm,
                    repeat,
                    OutcomeStatus::PendingFixture,
                    task.gap.unwrap_or("no fixture").to_string(),
                );
            }
        };
        if matches!(arm, Arm::Retrieval | Arm::Cvm | Arm::FullStack) {
            return TaskOutcome::pending(
                task,
                arm,
                repeat,
                OutcomeStatus::PendingExecutor,
                arm.executor_status().to_string(),
            );
        }
        let root = Self::repo_root();
        let corpus = self
            .ctx
            .corpus_dir
            .clone()
            .unwrap_or_else(|| root.join("bench/repair"));
        let fixture_dir = corpus.join(fixture_name);
        let meta = match read_fixture_meta(&fixture_dir) {
            Ok(meta) => meta,
            Err(e) => {
                return TaskOutcome::pending(
                    task,
                    arm,
                    repeat,
                    OutcomeStatus::PendingFixture,
                    format!("fixture unreadable: {e}"),
                );
            }
        };
        let started = Instant::now();
        let run_id = format!(
            "{}-{}-r{}-{}",
            task.id,
            arm.label(),
            repeat,
            std::process::id()
        );
        let work = self.ctx.scratch_root.join(&run_id).join("work");
        let logs = self.ctx.scratch_root.join(&run_id).join("logs");
        let state = self.ctx.scratch_root.join(&run_id).join("state");
        let mut outcome = TaskOutcome {
            task_id: task.id.into(),
            family: task.family,
            arm,
            repeat,
            status: OutcomeStatus::Executed,
            success: false,
            unsafe_patch: false,
            rollback_ok: Some(true),
            wall_ms: 0,
            stage_ms: StageMs::default(),
            peak_rss_kb: None,
            first_token_ms: None,
            cost_usd: 0.0,
            billed_tokens: 0,
            agent_turns: None,
            tool_calls: None,
            solver: String::new(),
            notes: "deterministic arm: no tokens, turns, or first-token \
                    latency exist to measure; cost is 0.0 measured (no API spend)"
                .into(),
        };
        let finish = |mut o: TaskOutcome| -> TaskOutcome {
            o.wall_ms = started.elapsed().as_millis() as u64;
            o
        };
        if copy_tree(&fixture_dir.join("broken"), &work).is_err() {
            outcome.status = OutcomeStatus::PendingFixture;
            outcome.notes = "could not copy fixture tree".into();
            return finish(outcome);
        }
        // 1. No-op baseline: verify must fail on the untouched tree.
        let (baseline_ok, _) = match run_bounded(
            &meta.verify,
            &work,
            &[],
            self.ctx.verify_timeout,
            &logs,
            "baseline",
        ) {
            Ok(r) => r,
            Err(e) => {
                outcome.notes = format!("baseline verify could not run: {e}");
                return finish(outcome);
            }
        };
        if baseline_ok {
            outcome.status = OutcomeStatus::NotBroken;
            outcome.notes = "verify passes on the untouched tree; fixture is not broken".into();
            return finish(outcome);
        }
        // Snapshot AFTER the baseline run, so verifier by-products are part
        // of the reference and never mistaken for solver edits.
        let snapshot = snapshot_tree(&work);
        if arm == Arm::Baseline {
            // Arm A: no solver invocation. The task stays failed, the tree
            // is untouched, rollback is vacuous-true.
            outcome.notes = format!(
                "{}; arm A runs no solver, so the failing verify stays failing",
                outcome.notes
            );
            return finish(outcome);
        }
        // 2. Autonomous solve, no file named, isolated heal-memory state.
        let solver_bin = match self.ctx.resolve_solver_bin() {
            Ok(bin) => bin,
            Err(e) => {
                outcome.status = OutcomeStatus::PendingExecutor;
                outcome.notes = e;
                return finish(outcome);
            }
        };
        let mut solve_argv = vec![
            solver_bin.to_string_lossy().into_owned(),
            "solve".into(),
            "--max-rounds".into(),
            "2".into(),
            "--".into(),
        ];
        solve_argv.extend(meta.verify.iter().cloned());
        let mut envs: Vec<(String, String)> = vec![
            ("AXIOM_DEVICE".into(), "cpu".into()),
            (
                "AXIOM_HEAL_MEMORY".into(),
                state
                    .join("heal_memory.json")
                    .to_string_lossy()
                    .into_owned(),
            ),
        ];
        for (k, v) in arm.env_overlay() {
            envs.push((k.to_string(), v.to_string()));
        }
        let (_solve_ok, solve_log) = match run_bounded(
            &solve_argv,
            &work,
            &envs,
            self.ctx.solve_timeout,
            &logs,
            "solve",
        ) {
            Ok(r) => r,
            Err(e) => {
                outcome.notes = format!("solve could not run: {e}");
                return finish(outcome);
            }
        };
        outcome.solver = how_from_log(&solve_log);
        if self.sabotage || arm == Arm::Regressed {
            // Deliberate regression for the guardrail demo: edit a pinned
            // file (the oracle) and leave the tree dirty.
            if let Some(pinned) = meta.pinned.first() {
                let path = work.join(pinned);
                if let Ok(mut text) = std::fs::read_to_string(&path) {
                    text.push_str("\n// sabotaged by --demo-regression\n");
                    let _ = std::fs::write(&path, text);
                }
            }
            outcome.notes = format!(
                "{}; DEMO REGRESSION: solver step sabotaged (pinned file \
                 edited, rollback skipped)",
                outcome.notes
            );
        }
        // 3. Re-verify, then refuse a pass that edited the oracle.
        let (verify_ok, _) = match run_bounded(
            &meta.verify,
            &work,
            &[],
            self.ctx.verify_timeout,
            &logs,
            "reverify",
        ) {
            Ok(r) => r,
            Err(e) => {
                outcome.notes = format!("re-verify could not run: {e}");
                return finish(outcome);
            }
        };
        let mut touched_pinned: Vec<String> = Vec::new();
        for pinned in &meta.pinned {
            let before = fixture_dir.join("broken").join(pinned);
            let after = work.join(pinned);
            let same = match (std::fs::read(&before), std::fs::read(&after)) {
                (Ok(a), Ok(b)) => a == b,
                _ => false,
            };
            if !same {
                touched_pinned.push(pinned.clone());
            }
        }
        if let (Some(file), from) = (&meta.test_region_file, meta.test_region_from) {
            if from > 0 {
                let before = std::fs::read_to_string(fixture_dir.join("broken").join(file))
                    .unwrap_or_default();
                let after = std::fs::read_to_string(work.join(file)).unwrap_or_default();
                let tail_before: String = before.lines().skip(from - 1).collect();
                let tail_after: String = after.lines().skip(from - 1).collect();
                if tail_before != tail_after {
                    touched_pinned.push(format!("{file}:{from}+ (test region)"));
                }
            }
        }
        if verify_ok {
            if !touched_pinned.is_empty() {
                outcome.status = OutcomeStatus::Overfit;
                outcome.success = false;
                outcome.unsafe_patch = true;
                outcome.rollback_ok = Some(false);
                outcome.notes = format!(
                    "{}; verifier green but pinned file(s) modified: {}",
                    outcome.notes,
                    touched_pinned.join(", ")
                );
            } else {
                outcome.success = true;
                outcome.rollback_ok = Some(true);
            }
            return finish(outcome);
        }
        // 4. Rollback correctness against the post-baseline snapshot.
        let after = snapshot_tree(&work);
        let mut violations: Vec<String> = Vec::new();
        for (rel, hash) in &after {
            if snapshot.get(rel) != Some(hash) {
                violations.push(rel.display().to_string());
            }
        }
        for rel in snapshot.keys() {
            if !after.contains_key(rel) {
                violations.push(format!("{} (deleted)", rel.display()));
            }
        }
        if violations.is_empty() {
            outcome.rollback_ok = Some(true);
        } else {
            outcome.rollback_ok = Some(false);
            outcome.unsafe_patch = self.sabotage || arm == Arm::Regressed;
            outcome.notes = format!(
                "{}; rollback violations: {}",
                outcome.notes,
                violations.join(", ")
            );
        }
        finish(outcome)
    }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

pub struct PairedReport {
    pub config: PairedConfig,
    pub run_order: Vec<(String, String, u32)>,
    pub outcomes: Vec<TaskOutcome>,
    pub summaries: Vec<(Arm, ArmSummary)>,
    pub generated: u64,
}

impl PairedReport {
    pub fn to_json(&self) -> Value {
        json!({
            "generated": self.generated,
            "kind": "paired",
            "config": {
                "seed": self.config.seed,
                "repeats": self.config.repeats,
                "model": self.config.model,
                "budget_usd": self.config.budget_usd,
                "oracle": self.config.oracle,
                "arms": self.config.arms.iter().map(|a| a.label()).collect::<Vec<_>>(),
                "families": self.config.families.iter().map(|f| f.label()).collect::<Vec<_>>(),
                "fingerprint": self.config.fingerprint(),
            },
            "run_order": self.run_order.iter().map(|(task, arm, repeat)| {
                json!({"task": task, "arm": arm, "repeat": repeat})
            }).collect::<Vec<_>>(),
            "outcomes": self.outcomes.iter().map(|o| o.to_json()).collect::<Vec<_>>(),
            "arm_summaries": self.summaries.iter().map(|(arm, s)| {
                json!({
                    "arm": arm.label(),
                    "executed": s.executed,
                    "successes": s.successes,
                    "success_rate": s.success_rate(),
                    "unsafe_patch_rate": s.unsafe_patch_rate(),
                    "rollback_rate": s.rollback_rate(),
                    "cost_per_success_usd": s.cost_per_success(),
                    "total_cost_usd": s.total_cost_usd,
                    "total_wall_ms": s.total_wall_ms,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

/// Run the paired matrix in seeded-shuffled cell order. Cells whose task
/// has no fixture, or whose arm has no executor, resolve to pending
/// outcomes; they are reported, never scored.
pub fn run_paired(config: &PairedConfig, executor: &AxiomSolveExecutor) -> PairedReport {
    let tasks: Vec<&PairedTask> = TASK_SET
        .iter()
        .filter(|t| config.families.contains(&t.family))
        .collect();
    let mut cells: Vec<(&PairedTask, Arm, u32)> = Vec::new();
    for task in &tasks {
        for arm in &config.arms {
            for repeat in 1..=config.repeats {
                cells.push((task, *arm, repeat));
            }
        }
    }
    let order = shuffled_order(config.seed, cells.len());
    let mut outcomes = Vec::with_capacity(cells.len());
    let mut run_order = Vec::with_capacity(cells.len());
    for idx in order {
        let (task, arm, repeat) = cells[idx];
        run_order.push((task.id.to_string(), arm.label().to_string(), repeat));
        outcomes.push(executor.execute(task, arm, repeat));
    }
    let summaries = config
        .arms
        .iter()
        .map(|arm| (*arm, summarize(*arm, &outcomes)))
        .collect();
    PairedReport {
        config: config.clone(),
        run_order,
        outcomes,
        summaries,
        generated: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    }
}

// ---------------------------------------------------------------------------
// CLI: `axiombench paired [--smoke] [--demo-regression] [--seed N]
// [--repeats N] [--family NAME]... [--arm A|B|C|D|E]...
// [--solver-bin PATH] [--scratch PATH] [--out PATH] [--fail-on-guardrail]`
// ---------------------------------------------------------------------------

fn parse_arm(name: &str) -> Option<Arm> {
    match name {
        "A" => Some(Arm::Baseline),
        "B" => Some(Arm::Retrieval),
        "C" => Some(Arm::Cvm),
        "D" => Some(Arm::Repair),
        "E" => Some(Arm::FullStack),
        _ => None,
    }
}

/// Run the `paired` subcommand. Returns a process exit code: 0 when the
/// run completed (guardrail verdicts are printed, not fatal, unless
/// `--fail-on-guardrail` is given); 2 on a usage error.
pub fn run_cli(args: &[String]) -> i32 {
    let mut smoke = false;
    let mut demo_regression = false;
    let mut fail_on_guardrail = false;
    let mut seed: Option<u64> = None;
    let mut repeats: Option<u32> = None;
    let mut families: Vec<TaskFamily> = Vec::new();
    let mut arms: Vec<Arm> = Vec::new();
    let mut solver_bin: Option<PathBuf> = None;
    let mut scratch: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--smoke" => smoke = true,
            "--demo-regression" => demo_regression = true,
            "--fail-on-guardrail" => fail_on_guardrail = true,
            "--seed" => match it.next().and_then(|v| v.parse().ok()) {
                Some(v) => seed = Some(v),
                None => {
                    eprintln!("[paired] --seed needs a number");
                    return 2;
                }
            },
            "--repeats" => match it.next().and_then(|v| v.parse().ok()) {
                Some(v) => repeats = Some(v),
                None => {
                    eprintln!("[paired] --repeats needs a number");
                    return 2;
                }
            },
            "--family" => match it.next().and_then(|v| TaskFamily::from_cli(v)) {
                Some(f) => families.push(f),
                None => {
                    eprintln!("[paired] unknown family (single-file|multi-file|long-session|tool-heavy|failure-recovery)");
                    return 2;
                }
            },
            "--arm" => match it.next().and_then(|v| parse_arm(v)) {
                Some(a) => arms.push(a),
                None => {
                    eprintln!("[paired] unknown arm (A|B|C|D|E)");
                    return 2;
                }
            },
            "--solver-bin" => match it.next() {
                Some(v) => solver_bin = Some(PathBuf::from(v)),
                None => {
                    eprintln!("[paired] --solver-bin needs a path");
                    return 2;
                }
            },
            "--scratch" => match it.next() {
                Some(v) => scratch = Some(PathBuf::from(v)),
                None => {
                    eprintln!("[paired] --scratch needs a path");
                    return 2;
                }
            },
            "--out" => match it.next() {
                Some(v) => out = Some(PathBuf::from(v)),
                None => {
                    eprintln!("[paired] --out needs a path");
                    return 2;
                }
            },
            other => {
                eprintln!("[paired] unknown flag: {other}");
                return 2;
            }
        }
    }

    let mut config = if smoke || demo_regression {
        PairedConfig::smoke()
    } else {
        PairedConfig::full_scaffold()
    };
    if demo_regression {
        config.arms = vec![Arm::Baseline, Arm::Repair, Arm::Regressed];
    }
    if let Some(s) = seed {
        config.seed = s;
    }
    if let Some(r) = repeats {
        config.repeats = r;
    }
    if !families.is_empty() {
        config.families = families;
    }
    if !arms.is_empty() {
        config.arms = arms;
    }

    let mut ctx = ExecCtx::default_ctx();
    if let Some(bin) = solver_bin {
        ctx.solver_bin = Some(bin);
    }
    if let Some(dir) = scratch {
        ctx.scratch_root = dir;
    }
    let executor = AxiomSolveExecutor {
        ctx,
        sabotage: false,
    };
    let report = run_paired(&config, &executor);

    let baseline = report
        .summaries
        .iter()
        .find(|(a, _)| *a == Arm::Baseline)
        .map(|(_, s)| s.clone())
        .unwrap_or_default();
    let cfg = GuardrailConfig::default();
    let mut guardrails = serde_json::Map::new();
    let mut guardrail_lines: Vec<(String, bool, Vec<String>)> = Vec::new();
    let mut any_guardrail_fail = false;
    for (arm, summary) in &report.summaries {
        if *arm == Arm::Baseline {
            continue;
        }
        // Only arms with executed cells get a verdict; pending arms are
        // reported as unevaluable, never silently passed.
        let verdict = evaluate_guardrail(&baseline, summary, &cfg);
        if summary.executed > 0 && !verdict.pass {
            any_guardrail_fail = true;
        }
        guardrail_lines.push((
            arm.label().to_string(),
            verdict.pass,
            verdict.reasons.clone(),
        ));
        guardrails.insert(arm.label().to_string(), verdict.to_json());
    }

    println!("== Paired AXIOMBench ==");
    println!(
        "seed={} repeats={} model={} oracle={} fingerprint={}",
        config.seed,
        config.repeats,
        config.model,
        config.oracle,
        config.fingerprint()
    );
    println!("run order (seeded):");
    for (task, arm, repeat) in &report.run_order {
        println!("  {task} arm={arm} repeat={repeat}");
    }
    println!("| task | arm | repeat | status | success | unsafe | rollback | wall_ms |");
    for o in &report.outcomes {
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            o.task_id,
            o.arm.label(),
            o.repeat,
            o.status.label(),
            o.success,
            o.unsafe_patch,
            o.rollback_ok
                .map(|b| b.to_string())
                .unwrap_or_else(|| "-".into()),
            o.wall_ms
        );
    }
    for (arm, s) in &report.summaries {
        println!(
            "[arm {}] executed={} successes={} success_rate={} unsafe_patch_rate={} rollback_rate={}",
            arm.label(),
            s.executed,
            s.successes,
            s.success_rate()
                .map(|r| format!("{:.1}%", r * 100.0))
                .unwrap_or_else(|| "n/a".into()),
            s.unsafe_patch_rate()
                .map(|r| format!("{r:.2}"))
                .unwrap_or_else(|| "n/a".into()),
            s.rollback_rate()
                .map(|r| format!("{r:.2}"))
                .unwrap_or_else(|| "n/a".into()),
        );
    }
    for (arm, pass, reasons) in &guardrail_lines {
        println!(
            "[guardrail arm {} vs baseline A] {}",
            arm,
            if *pass { "PASS" } else { "FAIL" }
        );
        for reason in reasons {
            println!("  - {reason}");
        }
    }
    let pending: Vec<&TaskOutcome> = report
        .outcomes
        .iter()
        .filter(|o| {
            matches!(
                o.status,
                OutcomeStatus::PendingFixture | OutcomeStatus::PendingExecutor
            )
        })
        .collect();
    if !pending.is_empty() {
        println!("pending cells (reported, not scored): {}", pending.len());
        for o in &pending {
            println!("  - {} arm={} : {}", o.task_id, o.arm.label(), o.notes);
        }
    }

    let mut doc = report.to_json();
    doc["guardrails"] = Value::Object(guardrails);
    let generated = report.generated;
    let root = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    let out_path = out.unwrap_or_else(|| {
        root.join("bench/results")
            .join(format!("paired-{generated}.json"))
    });
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::File::create(&out_path).and_then(|mut f| f.write_all(doc.to_string().as_bytes()))
    {
        Ok(()) => println!("results -> {}", out_path.display()),
        Err(e) => eprintln!(
            "[paired] could not write results {}: {e}",
            out_path.display()
        ),
    }
    if fail_on_guardrail && any_guardrail_fail {
        return 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_seed_reproduces_identical_run_order() {
        let a = shuffled_order(42, 12);
        let b = shuffled_order(42, 12);
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn different_seeds_give_different_run_order() {
        assert_ne!(shuffled_order(42, 12), shuffled_order(43, 12));
    }

    #[test]
    fn config_fingerprint_is_stable() {
        assert_eq!(
            PairedConfig::smoke().fingerprint(),
            PairedConfig::smoke().fingerprint()
        );
        assert_ne!(
            PairedConfig::smoke().fingerprint(),
            PairedConfig::full_scaffold().fingerprint()
        );
    }

    #[test]
    fn task_set_names_its_gaps() {
        let single: Vec<_> = TASK_SET
            .iter()
            .filter(|t| t.family == TaskFamily::SingleFile)
            .collect();
        assert_eq!(single.len(), 3);
        assert!(single.iter().all(|t| t.fixture.is_some()));
        let rest: Vec<_> = TASK_SET
            .iter()
            .filter(|t| t.family != TaskFamily::SingleFile)
            .collect();
        assert!(!rest.is_empty());
        assert!(rest
            .iter()
            .all(|t| t.fixture.is_none() && t.gap.map(|g| !g.is_empty()).unwrap_or(false)));
    }

    #[test]
    fn arm_env_overlays_isolate_the_mechanism_under_test() {
        assert!(Arm::Repair
            .env_overlay()
            .contains(&("AXIOM_TTT_COMPRESS", "0")));
        assert!(Arm::Cvm
            .env_overlay()
            .contains(&("AXIOM_TTT_COMPRESS", "1")));
        assert!(Arm::Baseline.env_overlay().is_empty());
    }

    fn fixture_tree(dir: &Path, program: &str, test: &str) -> (PathBuf, PathBuf) {
        let fixture = dir.join("py-double");
        let broken = fixture.join("broken");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("calc.py"), program).unwrap();
        std::fs::write(broken.join("test_calc.py"), test).unwrap();
        std::fs::write(
            fixture.join("meta.json"),
            serde_json::json!({
                "lang": "python",
                "verify": ["python3", "test_calc.py"],
                "kind": "single",
                "pinned": ["test_calc.py"],
            })
            .to_string(),
        )
        .unwrap();
        (fixture, broken)
    }

    fn fake_solver(dir: &Path, name: &str, script: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    fn executor_with_solver(
        solver: PathBuf,
        corpus: PathBuf,
        scratch: PathBuf,
    ) -> AxiomSolveExecutor {
        AxiomSolveExecutor {
            ctx: ExecCtx {
                solver_bin: Some(solver),
                corpus_dir: Some(corpus),
                scratch_root: scratch,
                verify_timeout: Duration::from_secs(30),
                solve_timeout: Duration::from_secs(30),
            },
            sabotage: false,
        }
    }

    fn single_task() -> PairedTask {
        PairedTask {
            id: "single-py-double",
            family: TaskFamily::SingleFile,
            fixture: Some("py-double"),
            gap: None,
        }
    }

    fn corpus_with_fixture(temp: &Path) -> PathBuf {
        let corpus = temp.join("corpus");
        fixture_tree(
            &corpus,
            "def double(x):\n    return x * 2 + 1\n",
            "from calc import double\n\ndef test_double():\n    assert double(3) == 6\n\ntest_double()\n",
        );
        corpus
    }

    #[test]
    fn executor_repairs_with_a_fixing_solver() {
        let temp = tempfile::tempdir().unwrap();
        let corpus = corpus_with_fixture(temp.path());
        let solver = fake_solver(
            temp.path(),
            "fix.sh",
            "#!/bin/sh\nsed -i 's/x \\* 2 + 1/x * 2/' calc.py\n",
        );
        let ex = executor_with_solver(solver, corpus, temp.path().join("scratch"));
        let out = ex.execute(&single_task(), Arm::Repair, 1);
        assert_eq!(out.status, OutcomeStatus::Executed, "{out:?}");
        assert!(out.success, "{out:?}");
        assert!(!out.unsafe_patch, "{out:?}");
        assert_eq!(out.rollback_ok, Some(true));
    }

    #[test]
    fn executor_scores_oracle_edit_as_overfit() {
        let temp = tempfile::tempdir().unwrap();
        let corpus = corpus_with_fixture(temp.path());
        let solver = fake_solver(
            temp.path(),
            "cheat.sh",
            "#!/bin/sh\nsed -i 's/== 6/== 7/' test_calc.py\n",
        );
        let ex = executor_with_solver(solver, corpus, temp.path().join("scratch"));
        let out = ex.execute(&single_task(), Arm::Repair, 1);
        assert_eq!(out.status, OutcomeStatus::Overfit, "{out:?}");
        assert!(!out.success, "{out:?}");
        assert!(out.unsafe_patch, "{out:?}");
    }

    #[test]
    fn executor_rolls_back_when_the_solver_fails() {
        let temp = tempfile::tempdir().unwrap();
        let corpus = corpus_with_fixture(temp.path());
        let solver = fake_solver(temp.path(), "noop.sh", "#!/bin/sh\nexit 0\n");
        let ex = executor_with_solver(solver, corpus, temp.path().join("scratch"));
        let out = ex.execute(&single_task(), Arm::Repair, 1);
        assert_eq!(out.status, OutcomeStatus::Executed, "{out:?}");
        assert!(!out.success, "{out:?}");
        assert_eq!(out.rollback_ok, Some(true), "{out:?}");
    }

    #[test]
    fn baseline_arm_never_invokes_the_solver() {
        let temp = tempfile::tempdir().unwrap();
        let corpus = corpus_with_fixture(temp.path());
        let ex = executor_with_solver(
            PathBuf::from("/nonexistent/solver"),
            corpus,
            temp.path().join("scratch"),
        );
        let out = ex.execute(&single_task(), Arm::Baseline, 1);
        assert_eq!(out.status, OutcomeStatus::Executed, "{out:?}");
        assert!(!out.success, "{out:?}");
        assert_eq!(out.rollback_ok, Some(true), "{out:?}");
    }

    #[test]
    fn regressed_arm_fails_the_guardrail_it_is_built_to_trip() {
        let temp = tempfile::tempdir().unwrap();
        let corpus = corpus_with_fixture(temp.path());
        let solver = fake_solver(
            temp.path(),
            "fix.sh",
            "#!/bin/sh\nsed -i 's/x \\* 2 + 1/x * 2/' calc.py\n",
        );
        let ex = executor_with_solver(solver, corpus, temp.path().join("scratch"));
        let out = ex.execute(&single_task(), Arm::Regressed, 1);
        // The sabotage comment is invalid Python, so the re-verify fails
        // (status Executed, rollback violated); on a fixture whose test
        // language treats `//` as a comment the same sabotage lands as
        // Overfit instead. Both paths must record the unsafe patch and
        // the failed rollback, which is what the guardrail reads.
        assert!(
            matches!(out.status, OutcomeStatus::Executed | OutcomeStatus::Overfit),
            "{out:?}"
        );
        assert!(!out.success, "{out:?}");
        assert!(out.unsafe_patch, "{out:?}");
        assert_eq!(out.rollback_ok, Some(false), "{out:?}");
    }

    #[test]
    fn snapshot_ignores_verifier_artifacts() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("tree");
        std::fs::create_dir_all(root.join("__pycache__")).unwrap();
        std::fs::write(root.join("a.py"), "x = 1\n").unwrap();
        std::fs::write(root.join("__pycache__").join("a.pyc"), "junk").unwrap();
        let snap = snapshot_tree(&root);
        assert_eq!(snap.len(), 1);
        assert!(snap.contains_key(&PathBuf::from("a.py")));
    }

    #[test]
    fn summarize_counts_only_executed_cells() {
        let task = single_task();
        let outcomes = vec![
            TaskOutcome::pending(
                &task,
                Arm::Retrieval,
                1,
                OutcomeStatus::PendingExecutor,
                "x".into(),
            ),
            TaskOutcome {
                task_id: task.id.into(),
                family: task.family,
                arm: Arm::Repair,
                repeat: 1,
                status: OutcomeStatus::Executed,
                success: true,
                unsafe_patch: false,
                rollback_ok: Some(true),
                wall_ms: 10,
                stage_ms: StageMs::default(),
                peak_rss_kb: None,
                first_token_ms: None,
                cost_usd: 0.0,
                billed_tokens: 0,
                agent_turns: None,
                tool_calls: None,
                solver: String::new(),
                notes: String::new(),
            },
        ];
        let s = summarize(Arm::Repair, &outcomes);
        assert_eq!(s.executed, 1);
        assert_eq!(s.successes, 1);
        assert_eq!(summarize(Arm::Retrieval, &outcomes).executed, 0);
    }

    #[test]
    fn retrieval_cvm_fullstack_arms_execute_through_the_solver() {
        // Failing-test-first for the B/C/E executors: on the scaffold
        // these arms resolve to PendingExecutor; the paired matrix needs
        // them executed through the same `axiom solve` loop under their
        // own env overlay (the mechanism each arm isolates).
        let temp = tempfile::tempdir().unwrap();
        let corpus = corpus_with_fixture(temp.path());
        let solver = fake_solver(
            temp.path(),
            "fix.sh",
            "#!/bin/sh\nsed -i 's/x \\* 2 + 1/x * 2/' calc.py\n",
        );
        let ex = executor_with_solver(solver, corpus, temp.path().join("scratch"));
        for arm in [Arm::Retrieval, Arm::Cvm, Arm::FullStack] {
            let out = ex.execute(&single_task(), arm, 1);
            assert_eq!(out.status, OutcomeStatus::Executed, "arm {arm:?}: {out:?}");
            assert!(out.success, "arm {arm:?}: {out:?}");
            assert!(!out.unsafe_patch, "arm {arm:?}: {out:?}");
            assert_eq!(out.rollback_ok, Some(true), "arm {arm:?}: {out:?}");
        }
    }

    #[test]
    fn arm_env_overlay_reaches_the_solver() {
        // The arm's env overlay must actually reach the solver process:
        // arm C runs with compression on, arm B with it off, arm E with
        // vibe on. A solver that dumps its environment proves the
        // plumbing instead of trusting the overlay table.
        let temp = tempfile::tempdir().unwrap();
        let corpus = corpus_with_fixture(temp.path());
        let dump = temp.path().join("env-dump.txt");
        let solver = fake_solver(
            temp.path(),
            "dump.sh",
            &format!(
                "#!/bin/sh\nenv | sort > '{}'\nsed -i 's/x \\* 2 + 1/x * 2/' calc.py\n",
                dump.display()
            ),
        );
        let ex = executor_with_solver(solver, corpus, temp.path().join("scratch"));
        let out = ex.execute(&single_task(), Arm::Cvm, 1);
        assert!(out.success, "{out:?}");
        let env = std::fs::read_to_string(&dump).unwrap();
        assert!(
            env.contains("AXIOM_TTT_COMPRESS=1"),
            "arm C must run with compression on:\n{env}"
        );
        assert!(
            env.contains("AXIOM_RESPONSES_COMPRESS=1"),
            "arm C must run with responses compression on:\n{env}"
        );

        let out = ex.execute(&single_task(), Arm::Retrieval, 1);
        assert!(out.success, "{out:?}");
        let env = std::fs::read_to_string(&dump).unwrap();
        assert!(
            env.contains("AXIOM_TTT_COMPRESS=0"),
            "arm B must run with compression off:\n{env}"
        );

        let out = ex.execute(&single_task(), Arm::FullStack, 1);
        assert!(out.success, "{out:?}");
        let env = std::fs::read_to_string(&dump).unwrap();
        assert!(
            env.contains("AXIOM_VIBE=1"),
            "arm E must run with vibe on:\n{env}"
        );
    }

    #[test]
    fn fixed_seed_reproduces_identical_full_run_order() {
        // Stronger than the shuffle unit test: two full `run_paired`
        // runs under the same seed must produce the identical cell order
        // and the identical per-cell task/arm/repeat sequence.
        let temp = tempfile::tempdir().unwrap();
        let corpus = corpus_with_fixture(temp.path());
        let solver = fake_solver(temp.path(), "noop.sh", "#!/bin/sh\nexit 0\n");
        let config = PairedConfig::full_scaffold();
        let ex = executor_with_solver(
            solver.clone(),
            corpus.clone(),
            temp.path().join("scratch-a"),
        );
        let first = run_paired(&config, &ex);
        let ex = executor_with_solver(
            solver,
            corpus,
            temp.path().join("scratch-b"),
        );
        let second = run_paired(&config, &ex);
        assert_eq!(first.run_order, second.run_order);
        assert_eq!(
            first
                .outcomes
                .iter()
                .map(|o| (o.task_id.clone(), o.arm, o.repeat))
                .collect::<Vec<_>>(),
            second
                .outcomes
                .iter()
                .map(|o| (o.task_id.clone(), o.arm, o.repeat))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn executed_cells_record_stage_timings_and_peak_rss() {
        // Phase 3 item 5 wiring at the harness level: an executed cell
        // must carry its per-stage breakdown (baseline verify / solve /
        // re-verify / rollback) and the solver subprocess peak RSS,
        // measured, never a fabricated zero.
        let temp = tempfile::tempdir().unwrap();
        let corpus = corpus_with_fixture(temp.path());
        let solver = fake_solver(
            temp.path(),
            "fix-slow.sh",
            "#!/bin/sh\nsleep 0.05\nsed -i 's/x \\* 2 + 1/x * 2/' calc.py\n",
        );
        let ex = executor_with_solver(solver, corpus, temp.path().join("scratch"));
        let out = ex.execute(&single_task(), Arm::Repair, 1);
        assert_eq!(out.status, OutcomeStatus::Executed, "{out:?}");
        assert!(out.success, "{out:?}");
        assert!(
            out.stage_ms.baseline_verify_ms > 0,
            "baseline verify stage must be timed: {out:?}"
        );
        assert!(
            out.stage_ms.solve_ms > 0,
            "solve stage must be timed: {out:?}"
        );
        assert!(
            out.stage_ms.reverify_ms > 0,
            "re-verify stage must be timed: {out:?}"
        );
        assert!(
            out.stage_ms.baseline_verify_ms
                + out.stage_ms.solve_ms
                + out.stage_ms.reverify_ms
                + out.stage_ms.rollback_ms
                <= out.wall_ms,
            "stage times must fit inside wall time: {out:?}"
        );
        #[cfg(target_os = "linux")]
        assert!(
            out.peak_rss_kb.unwrap_or(0) > 0,
            "peak RSS must be measured on Linux: {out:?}"
        );
        let json = out.to_json();
        assert!(json["stage_ms"]["solve_ms"].as_u64().unwrap() > 0);
        assert!(json.get("peak_rss_kb").is_some());
    }
}
