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

fn run_solve(
    cmd: String,
    args: Vec<String>,
    opts: SolveOptions,
) -> axiom_engine::solve::SolveReport {
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
    assert!(report
        .env_heals
        .iter()
        .any(|h| h.contains("created directory")));
    assert!(!report.source_patched);
    assert!(target.exists());

    let _ = std::fs::remove_dir_all(&base);
}

/// A shell library with one wrong token: `-gt` where `-ge` was meant.
const BUGGY_LIMITS: &str = "is_adult() {\n  [ \"$1\" -gt 18 ]\n}\n";

/// `sh -c` body that sources `lib` and checks `is_adult`, blaming `frame` when
/// the check fails. The check is in the command, so no file holds an oracle
/// that could be edited to pass.
fn is_adult_check(lib: &std::path::Path, frame: &str) -> String {
    format!(
        ". {}; {{ is_adult 18 && is_adult 40 && ! is_adult 17; }} || {{ echo '{frame}' >&2; exit 1; }}",
        sh_path(lib)
    )
}

#[test]
fn solve_via_source_repair_when_env_heal_insufficient() {
    // No environment heal applies to a wrong comparison: the named source file
    // has to be repaired, and only a real fix makes the check pass.
    let base = unique_tmp("srcrepair");
    std::fs::create_dir_all(&base).unwrap();
    let lib = base.join("limits.sh");
    std::fs::write(&lib, BUGGY_LIMITS).unwrap();

    let report = run_solve(
        "sh".into(),
        vec![
            "-c".into(),
            is_adult_check(&lib, "limits.sh:2:3: is_adult is wrong"),
        ],
        SolveOptions {
            max_rounds: 2,
            max_restarts: 1,
            anchor: Some(base.clone()),
            source_path: Some(lib.clone()),
            ..SolveOptions::default()
        },
    );
    assert!(report.solved, "source repair should fix the comparison");
    assert!(report.source_patched);
    let fixed = std::fs::read_to_string(&lib).unwrap();
    assert_ne!(fixed, BUGGY_LIMITS);
    assert_eq!(
        fixed.lines().count(),
        BUGGY_LIMITS.lines().count(),
        "a one-token fix, not a rewrite: {fixed}"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn solve_does_not_flip_an_exit_code_to_pass_a_check() {
    // A release gate that fails for a real reason. The only way to make it
    // pass without doing the work is to edit the gate: `exit 1` -> `exit 0`.
    // That is what the first repair phase used to do, and report SOLVED.
    let base = unique_tmp("exitflip");
    std::fs::create_dir_all(&base).unwrap();
    std::fs::write(base.join("CHANGELOG.md"), "# Changelog\n\n## 0.7.0\n").unwrap();
    let gate = base.join("check.sh");
    let original = "#!/bin/sh\n# Release gate: the changelog must mention the release.\n\
                    if ! grep -q '0.8.0' CHANGELOG.md; then\n  \
                    echo \"check.sh:3: CHANGELOG.md has no 0.8.0 entry\"\n  exit 1\nfi\n";
    std::fs::write(&gate, original).unwrap();

    let report = run_solve(
        "sh".into(),
        vec!["check.sh".into()],
        SolveOptions {
            max_rounds: 2,
            max_restarts: 1,
            anchor: Some(base.clone()),
            source_path: None,
            ..SolveOptions::default()
        },
    );
    assert_eq!(
        std::fs::read_to_string(&gate).unwrap(),
        original,
        "the gate must not be edited to make itself pass"
    );
    assert!(!report.solved, "nothing was fixed, so nothing is solved");
    assert!(!report.source_patched);
    assert_eq!(report.final_exit, Some(1));

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_file_longer_than_4096_characters_is_whole_during_every_verify_run() {
    // The first repair phase used to write the first 4096 characters of the
    // file over the whole file while it tried patches. The verify command
    // here records the file's size each time it runs.
    let base = unique_tmp("longfile");
    std::fs::create_dir_all(&base).unwrap();
    let mut source = String::from("# helpers\n");
    for i in 0..120 {
        source.push_str(&format!(
            "helper_{i}() {{ echo \"helper number {i} does nothing useful\"; }}\n"
        ));
    }
    source.push_str(BUGGY_LIMITS);
    assert!(source.len() > 4096 * 3 / 2);
    let lib = base.join("lib.sh");
    std::fs::write(&lib, &source).unwrap();
    std::fs::write(
        base.join("test.sh"),
        "#!/bin/sh\nwc -c < lib.sh >> sizes.log\n. ./lib.sh\n\
         is_adult 18 || { echo \"test.sh:4: is_adult 18 should succeed\"; exit 1; }\n\
         is_adult 40 || { echo \"test.sh:5: is_adult 40 should succeed\"; exit 1; }\n\
         if is_adult 17; then echo \"test.sh:6: is_adult 17 should fail\"; exit 1; fi\n",
    )
    .unwrap();

    let report = run_solve(
        "sh".into(),
        vec!["test.sh".into()],
        SolveOptions {
            max_rounds: 2,
            max_restarts: 1,
            anchor: Some(base.clone()),
            source_path: None,
            ..SolveOptions::default()
        },
    );
    let sizes: Vec<usize> = std::fs::read_to_string(base.join("sizes.log"))
        .unwrap()
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    assert!(sizes.len() >= 2, "the verify command ran: {sizes:?}");
    // A one-token candidate changes the size by a character or two at most.
    assert!(
        sizes.iter().all(|&n| n.abs_diff(source.len()) <= 4),
        "the file was cut during a verify run: sizes {sizes:?}, original {}",
        source.len()
    );
    assert!(report.solved, "the comparison is fixed");
    let fixed = std::fs::read_to_string(&lib).unwrap();
    assert!(
        fixed.contains("helper_119()") && fixed.len().abs_diff(source.len()) <= 4,
        "the whole file is still there after the fix"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn solve_auto_localizes_source_from_trace_without_source_path() {
    // No source_path given: the loop must localize the faulty file from the
    // verify command's own output and repair it. This is the language-general
    // autonomy path — the user names no file.
    let base = unique_tmp("autoloc");
    std::fs::create_dir_all(&base).unwrap();
    let lib = base.join("limits.sh");
    std::fs::write(&lib, BUGGY_LIMITS).unwrap();

    let report = run_solve(
        "sh".into(),
        vec![
            "-c".into(),
            // The frame names the file relative to the anchor.
            ". ./limits.sh; { is_adult 18 && is_adult 40 && ! is_adult 17; } || \
             { echo 'limits.sh:2:3: is_adult is wrong' >&2; exit 1; }"
                .into(),
        ],
        SolveOptions {
            max_rounds: 2,
            max_restarts: 1,
            anchor: Some(base.clone()), // verify runs here; localization root
            source_path: None,          // <-- the point: no named file
            ..SolveOptions::default()
        },
    );
    assert!(
        report.solved,
        "auto-localized source repair should solve it"
    );
    assert!(report.source_patched, "the localized file was patched");
    assert_ne!(std::fs::read_to_string(&lib).unwrap(), BUGGY_LIMITS);

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn solve_tries_multiple_localized_candidates_until_one_repairs() {
    // The verify command's output cites TWO files: first.sh (correct, but
    // cited first) and second.sh (the real bug). With no source_path, the loop
    // must localize both, find no fix in the first, then repair the second —
    // proving multi-candidate fallback.
    let base = unique_tmp("multicand");
    std::fs::create_dir_all(&base).unwrap();
    let first = base.join("first.sh");
    let second = base.join("second.sh");
    let first_src = "double() {\n  echo $(( $1 * 2 ))\n}\n";
    std::fs::write(&first, first_src).unwrap();
    std::fs::write(&second, BUGGY_LIMITS).unwrap();

    let report = run_solve(
        "sh".into(),
        vec![
            "-c".into(),
            ". ./first.sh; . ./second.sh; echo 'first.sh:1:1: note: loaded'; \
             { is_adult 18 && is_adult 40 && ! is_adult 17; } || \
             { echo 'second.sh:2:3: is_adult is wrong' >&2; exit 1; }"
                .into(),
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
    assert_ne!(
        std::fs::read_to_string(&second).unwrap(),
        BUGGY_LIMITS,
        "the failing file (second.sh) was repaired"
    );
    assert_eq!(
        std::fs::read_to_string(&first).unwrap(),
        first_src,
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
