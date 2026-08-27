//! Self-healing ablation pillar: baseline (no repair) vs +AXIOM (the real
//! `solve` loop), measured over the same deterministic fixture suite
//! `eval-agentic` already uses.
//!
//! This is deliberately narrower than "baseline agent vs full AXIOM on real
//! coding tasks" -- building that credibly needs live-agent infrastructure
//! this binary doesn't have (see `docs/AXIOMBENCH.md` §3 for the full gap
//! analysis and why it isn't faked here). What this pillar *can* measure
//! honestly with what already exists: does AXIOM's self-healing repair loop
//! (env-heal -> Poly-JIT -> verify-gate, see `crate::solve`) actually improve
//! task pass rate over doing nothing, on the same 9 seeded broken-repo
//! fixtures `axiom eval-agentic` already runs. Both arms execute the real
//! code path -- nothing here is simulated or estimated.

use std::path::PathBuf;

use axiom_engine::agentic_eval::{builtin_cases, run_eval, EvalCase};
use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::solve::run_verify;
use candle_core::Device;
use serde_json::json;

use crate::cognition::PillarResult;

fn is_safe_rel(rel: &str) -> bool {
    !rel.starts_with('/') && !rel.split('/').any(|part| part == "..")
}

fn unique_root(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("axiombench-ablation-{tag}-{nanos}-{}", std::process::id()))
}

/// Materialize `case`'s starting files (never the fix) and run its verify
/// command exactly once, with no healing/repair attempted at all -- the
/// "baseline agent" arm. Returns `false` (not "unsolved") on any I/O error,
/// same fail-closed convention `agentic_eval::run_one` uses, so a fixture
/// that can't even be materialized never counts as a false pass.
fn run_baseline_once(case: &EvalCase) -> bool {
    let root = unique_root(&case.name);
    let result = (|| -> std::io::Result<bool> {
        std::fs::create_dir_all(&root)?;
        for (rel, content) in &case.files {
            if !is_safe_rel(rel) {
                return Ok(false);
            }
            let path = root.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, content)?;
        }
        Ok(run_verify(&case.command, &case.args, Some(root.as_path())))
    })()
    .unwrap_or(false);
    let _ = std::fs::remove_dir_all(&root);
    result
}

fn build_pipeline() -> candle_core::Result<InferencePipeline> {
    InferencePipeline::new(AxiomConfig::runtime_small(), Device::Cpu)
}

/// Run both arms over `agentic_eval::builtin_cases()` and report the
/// pass-rate delta. Must run on a large-stack thread -- same constraint
/// `solve`/`agentic_eval::run_eval` document (the native TTT backward graph
/// recurses) -- callers should follow the same `std::thread::Builder`
/// pattern `axiom eval-agentic` uses; `run_ablation` does this internally so
/// callers of this function don't have to know that.
pub fn run_ablation() -> PillarResult {
    let cases = builtin_cases();
    let n = cases.len();

    let spawned = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            let pipeline = build_pipeline();
            let baseline_pass: usize = cases.iter().filter(|c| run_baseline_once(c)).count();
            let axiom_report = pipeline.as_ref().ok().map(|p| run_eval(p, &cases));
            (baseline_pass, axiom_report)
        });
    let Ok(handle) = spawned else {
        return PillarResult {
            name: "ablation".into(),
            headline: "skipped - could not spawn the eval thread".into(),
            detail: json!({ "skipped": true, "reason": "thread spawn failed" }),
            ..Default::default()
        };
    };
    let Ok((baseline_pass, axiom_report)) = handle.join() else {
        return PillarResult {
            name: "ablation".into(),
            headline: "skipped - eval thread panicked".into(),
            detail: json!({ "skipped": true, "reason": "thread panic" }),
            ..Default::default()
        };
    };
    let Some(axiom_report) = axiom_report else {
        return PillarResult {
            name: "ablation".into(),
            headline: "skipped - could not build an inference pipeline".into(),
            detail: json!({ "skipped": true, "reason": "pipeline build failed", "baseline_pass": baseline_pass }),
            ..Default::default()
        };
    };
    let axiom_pass = axiom_report.solved();

    let headline = format!(
        "self-heal repair loop: {baseline_pass}/{n} pass with no repair attempted vs {axiom_pass}/{n} with AXIOM's solve loop"
    );
    PillarResult {
        name: "ablation".into(),
        headline,
        detail: json!({
            "task_suite": "agentic_eval::builtin_cases (the same 9 fixtures `axiom eval-agentic` runs)",
            "n": n,
            "baseline_pass": baseline_pass,
            "axiom_pass": axiom_pass,
            "baseline_arm": "materialize the fixture's starting (broken) files, run the verify command once, no repair attempted",
            "axiom_arm": "crate::solve's real loop: environment self-heal -> Poly-JIT source repair -> verify-gate, exactly what `axiom eval-agentic` runs",
            "per_case": axiom_report.results.iter().map(|r| json!({ "name": r.name, "axiom_solved": r.solved })).collect::<Vec<_>>(),
            "scope_note": "measures the self-healing repair capability specifically, over a fixed deterministic offline fixture suite -- not a general agent-task-success benchmark. See docs/AXIOMBENCH.md for what this is and is not evidence of.",
        }),
        sample_n: Some(n as u64),
        read_as: Some("measured, deterministic, offline -- narrow scope (self-heal only), see docs/AXIOMBENCH.md".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_fails_every_fixture_without_repair() {
        // Every builtin fixture is seeded broken specifically so it needs
        // Poly-JIT/agentic repair to pass -- the baseline arm (no repair at
        // all) must never accidentally pass one, or the ablation would be
        // measuring nothing. This is the regression test for that
        // precondition, independent of building a real pipeline.
        for case in builtin_cases() {
            assert!(
                !run_baseline_once(&case),
                "fixture '{}' passed verification with zero repair attempted -- \
                 it is not actually broken, or run_baseline_once has a bug",
                case.name
            );
        }
    }

    #[test]
    fn is_safe_rel_rejects_absolute_and_traversal_paths() {
        assert!(is_safe_rel("src/lib.rs"));
        assert!(!is_safe_rel("/etc/passwd"));
        assert!(!is_safe_rel("../../etc/passwd"));
        assert!(!is_safe_rel("a/../../b"));
    }

    #[test]
    fn unique_root_is_under_temp_dir_and_distinct_per_call() {
        let a = unique_root("tag");
        let b = unique_root("tag");
        assert_ne!(a, b);
        assert!(a.starts_with(std::env::temp_dir()));
    }
}
