//! Integration tests for `axiom run` — the self-healing runtime supervisor.

use std::path::PathBuf;

use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::self_heal::{run_supervised, Heal};
use candle_core::Device;

fn tiny_pipeline() -> InferencePipeline {
    let cfg = AxiomConfig {
        d_model: 16,
        n_layers: 2,
        vocab_size: 64,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    InferencePipeline::new(cfg, Device::Cpu).expect("tiny pipeline must build")
}

fn unique_tmp(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("axiom_run_{tag}_{nanos}"))
}

/// Runs the supervisor on a large stack (TTT adaptation recurses through
/// candle's graph teardown), mirroring what the CLI handler does.
fn supervise(
    pipeline: InferencePipeline,
    cmd: String,
    args: Vec<String>,
    max_restarts: usize,
) -> axiom_engine::self_heal::RunReport {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run_supervised(&pipeline, &cmd, &args, max_restarts).unwrap())
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn heals_missing_directory_and_completes() {
    let base = unique_tmp("heal");
    let target = base.join("out").join("result.txt");
    let script = format!("echo healed-run-output > {}", target.display());

    let report = supervise(
        tiny_pipeline(),
        "sh".into(),
        vec!["-c".into(), script],
        3,
    );

    assert!(report.success, "run must succeed after the heal");
    assert_eq!(report.attempts, 2, "fail once, heal, succeed on restart");
    assert_eq!(
        report.heals,
        vec![Heal::CreatedDirectory(base.join("out"))],
        "the missing directory must be the applied heal"
    );
    assert!(report.tokens_absorbed > 0, "the failure trace must be absorbed");
    assert_eq!(report.tension.len(), 1, "one failure → one tension sample");
    assert!(
        report.tension[0].ce_before.is_finite() && report.tension[0].ce_after.is_finite(),
        "tension must be measured, not NaN"
    );
    assert_eq!(
        std::fs::read_to_string(&target).unwrap().trim(),
        "healed-run-output"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn stops_without_applicable_heal_and_propagates_exit_code() {
    let report = supervise(
        tiny_pipeline(),
        "sh".into(),
        vec!["-c".into(), "exit 7".into()],
        3,
    );
    assert!(!report.success);
    assert_eq!(report.attempts, 1, "no heal → no blind restart");
    assert_eq!(report.exit_code, Some(7), "child exit code must be preserved");
    assert!(report.heals.is_empty());
}

#[test]
fn never_fabricates_missing_file_content() {
    // `cat` on a missing file: the parent dir heal applies once, but the file
    // itself must never be created — the supervisor stops after the re-failure.
    let base = unique_tmp("nofab");
    let target = base.join("cfg").join("settings.json");

    let report = supervise(
        tiny_pipeline(),
        "cat".into(),
        vec![target.display().to_string()],
        3,
    );

    assert!(!report.success, "cat of a missing file cannot be healed");
    assert_eq!(report.attempts, 2, "heal dir, retry, then stop");
    assert!(!target.exists(), "the supervisor must never create file content");

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn clean_run_is_untouched() {
    let report = supervise(
        tiny_pipeline(),
        "sh".into(),
        vec!["-c".into(), "true".into()],
        3,
    );
    assert!(report.success);
    assert_eq!(report.attempts, 1);
    assert!(report.heals.is_empty());
    assert_eq!(report.tokens_absorbed, 0, "no failure → nothing absorbed");
}
