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
    supervise_with_vibe(pipeline, cmd, args, max_restarts, None)
}

fn supervise_with_vibe(
    pipeline: InferencePipeline,
    cmd: String,
    args: Vec<String>,
    max_restarts: usize,
    vibe_path: Option<PathBuf>,
) -> axiom_engine::self_heal::RunReport {
    supervise_full(pipeline, cmd, args, max_restarts, vibe_path, None)
}

fn supervise_full(
    pipeline: InferencePipeline,
    cmd: String,
    args: Vec<String>,
    max_restarts: usize,
    vibe_path: Option<PathBuf>,
    heal_memory: Option<PathBuf>,
) -> axiom_engine::self_heal::RunReport {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            run_supervised(
                &pipeline,
                &cmd,
                &args,
                max_restarts,
                vibe_path.as_deref(),
                heal_memory.as_deref(),
            )
            .unwrap()
        })
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

#[test]
fn transient_fault_retries_then_succeeds() {
    // First run prints a recognised transient phrase and fails; the marker file
    // makes the second run succeed — modelling a service that came back up.
    let marker = unique_tmp("transient_marker");
    let script = format!(
        "if [ -f {m} ]; then echo recovered; else touch {m}; echo 'connect: Connection refused' >&2; exit 1; fi",
        m = marker.display()
    );
    let report = supervise(tiny_pipeline(), "sh".into(), vec!["-c".into(), script], 3);

    assert!(report.success, "must recover after the transient retry");
    assert_eq!(report.attempts, 2);
    assert_eq!(report.heals, vec![Heal::TransientRetry("connection refused")]);

    let _ = std::fs::remove_file(&marker);
}

#[test]
fn transient_retries_are_bounded() {
    // Always fails with a transient phrase: 1 attempt + 2 transient retries,
    // then the supervisor must stop rather than loop.
    let report = supervise(
        tiny_pipeline(),
        "sh".into(),
        vec![
            "-c".into(),
            "echo 'connect: Connection timed out' >&2; exit 1".into(),
        ],
        5,
    );
    assert!(!report.success);
    assert_eq!(report.attempts, 3, "1 attempt + MAX_TRANSIENT_RETRIES");
    assert_eq!(report.heals.len(), 2);
}

#[test]
fn learned_immunity_pre_heals_a_fresh_environment() {
    use axiom_engine::self_heal::Heal::{CreatedDirectory, Immunized};

    let base = unique_tmp("immunity");
    let out_dir = base.join("out");
    let target = out_dir.join("result.txt");
    let memory = unique_tmp("immunity_mem").with_extension("json");
    let script = format!("echo run-output > {}", target.display());
    let args = vec!["-c".to_string(), script];

    // Run 1: fails on the missing dir, heals reactively, succeeds, learns.
    let first = supervise_full(
        tiny_pipeline(),
        "sh".into(),
        args.clone(),
        3,
        None,
        Some(memory.clone()),
    );
    assert!(first.success);
    assert_eq!(first.attempts, 2);
    assert_eq!(first.heals, vec![CreatedDirectory(out_dir.clone())]);

    // Fresh environment: the directory is gone again.
    std::fs::remove_dir_all(&base).unwrap();
    assert!(!out_dir.exists());

    // Run 2: immunity pre-creates the remembered dir → zero failures.
    let second = supervise_full(
        tiny_pipeline(),
        "sh".into(),
        args,
        3,
        None,
        Some(memory.clone()),
    );
    assert!(second.success);
    assert_eq!(second.attempts, 1, "immunized run must succeed first try");
    assert_eq!(second.heals, vec![Immunized(out_dir.clone())]);
    assert_eq!(second.tokens_absorbed, 0, "no failure ever happened");
    assert_eq!(
        std::fs::read_to_string(&target).unwrap().trim(),
        "run-output"
    );

    let _ = std::fs::remove_dir_all(&base);
    let _ = std::fs::remove_file(&memory);
}

#[test]
fn immunity_is_per_command_fingerprint() {
    // A different command line must not inherit another program's immunity.
    let base = unique_tmp("immunity_other");
    let memory = unique_tmp("immunity_other_mem").with_extension("json");
    let script_a = format!("echo a > {}", base.join("a").join("f.txt").display());

    let first = supervise_full(
        tiny_pipeline(),
        "sh".into(),
        vec!["-c".into(), script_a],
        3,
        None,
        Some(memory.clone()),
    );
    assert!(first.success);

    // Different command: must start with no immunity heals applied.
    let other = supervise_full(
        tiny_pipeline(),
        "sh".into(),
        vec!["-c".into(), "true".into()],
        3,
        None,
        Some(memory.clone()),
    );
    assert!(other.success);
    assert!(other.heals.is_empty(), "no inherited immunity across commands");

    let _ = std::fs::remove_dir_all(&base);
    let _ = std::fs::remove_file(&memory);
}

#[test]
fn novel_faults_are_absorbed_harder_than_familiar_ones() {
    // A non-healable, deterministic failure (`exit 7`): no heal applies, so the
    // run absorbs once and stops. With a shared memory, the FIRST run sees a
    // novel fault (deep absorption); the SECOND run recognises it (light).
    let memory = unique_tmp("gate_mem").with_extension("json");
    let args = vec!["-c".to_string(), "exit 7".to_string()];

    let first = supervise_full(
        tiny_pipeline(),
        "sh".into(),
        args.clone(),
        3,
        None,
        Some(memory.clone()),
    );
    assert!(!first.success);
    assert_eq!(first.attempts, 1, "no heal → single attempt");
    assert_eq!(
        first.absorption_passes, 3,
        "a first-seen fault must be absorbed deeply"
    );

    let second = supervise_full(
        tiny_pipeline(),
        "sh".into(),
        args,
        3,
        None,
        Some(memory.clone()),
    );
    assert!(!second.success);
    assert_eq!(
        second.absorption_passes, 1,
        "a recognised (KNOWN) fault must be reinforced lightly"
    );

    let _ = std::fs::remove_file(&memory);
}

#[test]
fn absorption_is_deep_without_memory_history() {
    // No heal memory → no classification → treat the fault as unfamiliar.
    let report = supervise(
        tiny_pipeline(),
        "sh".into(),
        vec!["-c".into(), "exit 1".into()],
        3,
    );
    assert!(!report.success);
    assert_eq!(report.absorption_passes, 3);
}

#[test]
fn failure_history_persists_to_vibe_when_requested() {
    let vibe = unique_tmp("vibe").with_extension("bin");
    let report = supervise_with_vibe(
        tiny_pipeline(),
        "sh".into(),
        vec!["-c".into(), "exit 3".into()],
        1,
        Some(vibe.clone()),
    );
    assert!(!report.success);
    assert!(report.tokens_absorbed > 0);
    assert!(
        vibe.exists(),
        "absorbed failure history must be committed to the master vibe"
    );
    let _ = std::fs::remove_file(&vibe);
}
