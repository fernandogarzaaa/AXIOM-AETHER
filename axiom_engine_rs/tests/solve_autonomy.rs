//! Pillar 3 — Autonomy: the `axiom solve` orchestrator.

use std::path::PathBuf;

use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::solve::{solve, SolveOptions};
use candle_core::Device;

fn tiny_pipeline() -> InferencePipeline {
    let cfg = AxiomConfig {
        d_model: 16,
        n_layers: 2,
        vocab_size: 64,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    InferencePipeline::new(cfg, Device::Cpu).expect("tiny pipeline")
}

fn unique_tmp(tag: &str) -> PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("axiom_solve_{tag}_{n}"))
}

fn run_solve(cmd: String, args: Vec<String>, opts: SolveOptions) -> axiom_engine::solve::SolveReport {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || solve(&tiny_pipeline(), &cmd, &args, &opts).unwrap())
        .unwrap()
        .join()
        .unwrap()
}

/// Render a path for embedding in `sh -c` scripts or command args: sh eats
/// unquoted backslashes, so Windows `\` separators would silently mangle the
/// path. Forward slashes work on all platforms; on Unix this is the identity.
fn sh_path(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "/")
}

#[test]
fn solve_via_environment_heal_only() {
    // Missing directory → Pillar-2 environment heal alone drives it green.
    let base = unique_tmp("envonly");
    std::fs::create_dir_all(&base).unwrap();
    let target = base.join("out").join("r.txt");
    let report = run_solve(
        "sh".into(),
        vec!["-c".into(), format!("echo ok > {}", sh_path(&target))],
        SolveOptions {
            max_rounds: 2,
            max_restarts: 3,
            ..SolveOptions::default()
        },
    );
    assert!(report.solved);
    assert_eq!(report.rounds, 1, "first round's env heal should suffice");
    assert!(report.env_heals.iter().any(|h| h.contains("created directory")));
    assert!(!report.source_patched);
    assert!(target.exists());

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn solve_via_source_repair_when_env_heal_insufficient() {
    // A script that just `exit 1`s — no environment heal applies, but Poly JIT's
    // exit-code-flip patch (Q-TTT) repairs the source and drives it green.
    let base = unique_tmp("srcrepair");
    std::fs::create_dir_all(&base).unwrap();
    let script = base.join("verify.sh");
    std::fs::write(&script, "#!/bin/sh\necho 'verify failed' >&2\nexit 1\n").unwrap();

    let report = run_solve(
        "sh".into(),
        vec![script.display().to_string()],
        SolveOptions {
            max_rounds: 2,
            max_restarts: 1,
            source_path: Some(script.clone()),
            ..SolveOptions::default()
        },
    );
    assert!(report.solved, "source repair should solve an exit-1 verify");
    assert!(report.source_patched);
    assert!(std::fs::read_to_string(&script).unwrap().contains("exit 0"));

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn solve_auto_localizes_source_from_trace_without_source_path() {
    // No source_path given: the loop must localize the faulty file from the
    // verify command's own output, then Poly JIT repairs it. This is the
    // language-general autonomy path — the user names no file.
    let base = unique_tmp("autoloc");
    std::fs::create_dir_all(&base).unwrap();
    // prog.sh fails and prints a frame naming itself (relative to the anchor),
    // and contains `exit 1` which Poly JIT flips to `exit 0`.
    let prog = base.join("prog.sh");
    std::fs::write(&prog, "#!/bin/sh\necho 'prog.sh:1:1: boom' >&2\nexit 1\n").unwrap();

    let report = run_solve(
        "sh".into(),
        vec!["prog.sh".into()],
        SolveOptions {
            max_rounds: 2,
            max_restarts: 1,
            anchor: Some(base.clone()), // verify runs here; localization root
            source_path: None,          // <-- the point: no named file
            ..SolveOptions::default()
        },
    );
    assert!(report.solved, "auto-localized source repair should solve it");
    assert!(report.source_patched, "the localized file was patched");
    assert!(
        std::fs::read_to_string(&prog).unwrap().contains("exit 0"),
        "Poly JIT flipped the localized file's exit code"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn solve_tries_multiple_localized_candidates_until_one_repairs() {
    // The verify command references TWO files: first.sh (harmless, exits 0, but
    // cited first) and second.sh (the real failure, `exit 1`). With no
    // source_path, the loop must localize both, fail to repair the first
    // candidate, then repair the second — proving multi-candidate fallback.
    let base = unique_tmp("multicand");
    std::fs::create_dir_all(&base).unwrap();
    let first = base.join("first.sh");
    let second = base.join("second.sh");
    std::fs::write(&first, "#!/bin/sh\necho 'first.sh:1:1: just a warning'\nexit 0\n").unwrap();
    std::fs::write(&second, "#!/bin/sh\necho 'second.sh:2:1: boom' >&2\nexit 1\n").unwrap();

    let report = run_solve(
        "sh".into(),
        vec![
            "-c".into(),
            format!("sh {}; sh {}", sh_path(&first), sh_path(&second)),
        ],
        SolveOptions {
            max_rounds: 2,
            max_restarts: 1,
            anchor: Some(base.clone()),
            source_path: None,
            ..SolveOptions::default()
        },
    );
    assert!(report.solved, "the second candidate should be repaired");
    assert!(report.source_patched);
    assert!(
        std::fs::read_to_string(&second).unwrap().contains("exit 0"),
        "Poly JIT flipped the failing file (second.sh)"
    );
    assert_eq!(
        std::fs::read_to_string(&first).unwrap(),
        "#!/bin/sh\necho 'first.sh:1:1: just a warning'\nexit 0\n",
        "the non-failing first candidate is left byte-for-byte intact"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn solve_reports_unsolved_and_restores_source_on_failure() {
    // No env heal, no applicable patch (exit 3) → unsolved, source intact.
    let base = unique_tmp("unsolved");
    std::fs::create_dir_all(&base).unwrap();
    let script = base.join("hard.sh");
    let original = "#!/bin/sh\necho 'unrepairable' >&2\nexit 3\n";
    std::fs::write(&script, original).unwrap();

    let report = run_solve(
        "sh".into(),
        vec![script.display().to_string()],
        SolveOptions {
            max_rounds: 2,
            max_restarts: 1,
            source_path: Some(script.clone()),
            ..SolveOptions::default()
        },
    );
    assert!(!report.solved);
    assert!(!report.source_patched);
    assert_eq!(report.final_exit, Some(3));
    assert_eq!(
        std::fs::read_to_string(&script).unwrap(),
        original,
        "a failed solve must restore the source byte-for-byte"
    );
    // No-progress guard: must stop early, not spin to max_rounds.
    assert_eq!(report.rounds, 1);

    let _ = std::fs::remove_dir_all(&base);
}
