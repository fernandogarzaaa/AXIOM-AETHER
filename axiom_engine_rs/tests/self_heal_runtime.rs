//! Integration tests for `axiom run` — the self-healing runtime supervisor.

use std::path::PathBuf;

use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::self_heal::{run_supervised, Heal, SupervisorOptions};
use axiom_engine::solve::posix_shell;
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
    supervise_opts(
        pipeline,
        cmd,
        args,
        SupervisorOptions {
            max_restarts,
            vibe_path,
            heal_memory_path: heal_memory,
            remember_into: None,
            anchor: None,
        },
    )
}

/// Render a path for embedding in `sh -c` scripts or command args: sh eats
/// unquoted backslashes, so Windows `\` separators would silently mangle the
/// path (the redirect succeeds to a wrong-but-writable filename and the
/// fail→heal→retry choreography never happens). Forward slashes work on all
/// platforms; on Unix this is the identity.
fn sh_path(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "/")
}

fn supervise_opts(
    pipeline: InferencePipeline,
    cmd: String,
    args: Vec<String>,
    opts: SupervisorOptions,
) -> axiom_engine::self_heal::RunReport {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run_supervised(&pipeline, &cmd, &args, &opts).unwrap())
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn heals_missing_directory_and_completes() {
    let base = unique_tmp("heal");
    let target = base.join("out").join("result.txt");
    let script = format!("echo healed-run-output > {}", sh_path(&target));

    let report = supervise(
        tiny_pipeline(),
        posix_shell(),
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
        posix_shell(),
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
        vec![sh_path(&target)],
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
        posix_shell(),
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
    let report = supervise(tiny_pipeline(), posix_shell(), vec!["-c".into(), script], 3);

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
        posix_shell(),
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
    let script = format!("echo run-output > {}", sh_path(&target));
    let args = vec!["-c".to_string(), script];

    // Run 1: fails on the missing dir, heals reactively, succeeds, learns.
    let first = supervise_full(
        tiny_pipeline(),
        posix_shell(),
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
        posix_shell(),
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
        posix_shell(),
        vec!["-c".into(), script_a],
        3,
        None,
        Some(memory.clone()),
    );
    assert!(first.success);

    // Different command: must start with no immunity heals applied.
    let other = supervise_full(
        tiny_pipeline(),
        posix_shell(),
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
        posix_shell(),
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
        posix_shell(),
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
        posix_shell(),
        vec!["-c".into(), "exit 1".into()],
        3,
    );
    assert!(!report.success);
    assert_eq!(report.absorption_passes, 3);
}

#[test]
fn immunity_is_location_invariant_across_working_directories() {
    use axiom_engine::self_heal::Heal::{CreatedDirectory, Immunized};

    // Same command, run in two different working directories. A relative-path
    // heal learned in dir A must transfer to dir B — a location the program has
    // never run in — because the heal is remembered relative to the anchor.
    let dir_a = unique_tmp("loc_a");
    let dir_b = unique_tmp("loc_b");
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();
    let memory = unique_tmp("loc_mem").with_extension("json");
    // cwd-relative output path — identical command string → identical fingerprint.
    let args = vec!["-c".to_string(), "echo built > build/out/app.bin".to_string()];

    let mk = |anchor: PathBuf| SupervisorOptions {
        max_restarts: 3,
        heal_memory_path: Some(memory.clone()),
        anchor: Some(anchor),
        ..SupervisorOptions::default()
    };

    // Run 1 in dir A: fails on missing build/out, heals it, succeeds, learns.
    let a = supervise_opts(tiny_pipeline(), posix_shell(), args.clone(), mk(dir_a.clone()));
    assert!(a.success);
    assert_eq!(a.heals, vec![CreatedDirectory(dir_a.join("build").join("out"))]);

    // Run 2 in dir B: never ran here, yet immunity pre-creates build/out under B
    // and the program succeeds on the first attempt.
    assert!(!dir_b.join("build").join("out").exists());
    let b = supervise_opts(tiny_pipeline(), posix_shell(), args, mk(dir_b.clone()));
    assert!(b.success);
    assert_eq!(b.attempts, 1, "portable immunity → first-try success in a new location");
    assert_eq!(b.heals, vec![Immunized(dir_b.join("build").join("out"))]);
    assert!(dir_b.join("build").join("out").join("app.bin").exists());

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
    let _ = std::fs::remove_file(&memory);
}

#[test]
fn predicts_failure_from_missing_prerequisites_before_running() {
    use axiom_engine::heal_memory::{fingerprint, HealMemory};
    use axiom_engine::self_heal::predict_failure;

    let anchor = unique_tmp("predict_anchor");
    std::fs::create_dir_all(&anchor).unwrap();
    let memory = unique_tmp("predict_mem").with_extension("json");

    // Teach Axiom that this command needs ./build/out (relative).
    let mut mem = HealMemory::load(&memory);
    mem.remember_dirs(
        &fingerprint(&posix_shell(), &["-c".into(), "echo x > build/out/r".into()]),
        "sh -c echo x > build/out/r",
        &[PathBuf::from("build/out")],
    );
    mem.save().unwrap();

    let opts = |anchor: PathBuf| SupervisorOptions {
        heal_memory_path: Some(memory.clone()),
        anchor: Some(anchor),
        ..SupervisorOptions::default()
    };
    let cmd = posix_shell();
    let args = vec!["-c".to_string(), "echo x > build/out/r".to_string()];

    // Prerequisite missing under the anchor → failure predicted before running.
    let p = predict_failure(&cmd, &args, &opts(anchor.clone()));
    assert!(p.likely, "missing prerequisite must predict failure");
    assert_eq!(p.missing_prerequisites, vec![anchor.join("build").join("out")]);

    // Create it → prediction flips to no-failure (prerequisites satisfied).
    std::fs::create_dir_all(anchor.join("build").join("out")).unwrap();
    let p2 = predict_failure(&cmd, &args, &opts(anchor.clone()));
    assert!(!p2.likely);
    assert!(p2.missing_prerequisites.is_empty());

    // An unknown command yields no prediction.
    let p3 = predict_failure("totally", &["unknown".into()], &opts(anchor.clone()));
    assert!(!p3.likely);
    assert!(p3.rationale.contains("no prior experience"));

    let _ = std::fs::remove_dir_all(&anchor);
    let _ = std::fs::remove_file(&memory);
}

#[cfg(unix)]
#[test]
fn heals_missing_execute_bit_on_a_script() {
    use axiom_engine::self_heal::Heal::MadeExecutable;
    use std::os::unix::fs::PermissionsExt;

    let dir = unique_tmp("execbit");
    std::fs::create_dir_all(&dir).unwrap();
    let script = dir.join("build.sh");
    std::fs::write(&script, "#!/bin/sh\necho ran\n").unwrap();
    // Non-executable (rw-r--r--).
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o644)).unwrap();

    // Supervise the script directly: spawn fails with PermissionDenied → chmod
    // +x heal → retry → success.
    let report = supervise(
        tiny_pipeline(),
        script.display().to_string(),
        vec![],
        3,
    );
    assert!(report.success, "must run after the execute-bit heal");
    assert_eq!(report.heals, vec![MadeExecutable(script.clone())]);
    let mode = std::fs::metadata(&script).unwrap().permissions().mode();
    assert!(mode & 0o111 != 0, "execute bit must be set");
    // Contents are never touched by the heal.
    assert_eq!(std::fs::read_to_string(&script).unwrap(), "#!/bin/sh\necho ran\n");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn diagnoses_and_remembers_missing_env_var() {
    use axiom_engine::heal_memory::{fingerprint, HealMemory};

    let memory = unique_tmp("envmem").with_extension("json");
    // Fails citing a missing env var (a name not set in this process).
    let script = "echo 'error: AXIOM_FAKE_TOKEN_XYZ must be set' >&2; exit 1";
    let args = vec!["-c".to_string(), script.to_string()];
    let report = supervise_opts(
        tiny_pipeline(),
        posix_shell(),
        args.clone(),
        SupervisorOptions {
            max_restarts: 3,
            heal_memory_path: Some(memory.clone()),
            ..SupervisorOptions::default()
        },
    );

    // Can't fabricate the value → run fails, but a diagnostic is surfaced…
    assert!(!report.success);
    assert!(
        report.diagnostics.iter().any(|d| d.contains("AXIOM_FAKE_TOKEN_XYZ")),
        "missing env var must be diagnosed: {:?}",
        report.diagnostics
    );
    // …and the requirement is remembered for advisories / `axiom immunity`.
    let fp = fingerprint(&posix_shell(), &args);
    let mem = HealMemory::load(&memory);
    assert_eq!(
        mem.record(&fp).map(|r| r.required_env.clone()),
        Some(vec!["AXIOM_FAKE_TOKEN_XYZ".to_string()])
    );

    let _ = std::fs::remove_file(&memory);
}

#[test]
fn novel_healed_failure_is_written_to_recall_memory() {
    use axiom_engine::memory_store::{MemoryKind, MemoryStore};

    let base = unique_tmp("remember_env");
    let target = base.join("out").join("r.txt");
    let mem_root = unique_tmp("recall_store");
    let script = format!("echo x > {}", sh_path(&target));

    let report = supervise_opts(
        tiny_pipeline(),
        posix_shell(),
        vec!["-c".into(), script],
        SupervisorOptions {
            max_restarts: 3,
            remember_into: Some(mem_root.clone()),
            ..SupervisorOptions::default()
        },
    );
    assert!(report.success);

    // The healed novel failure must be recorded as a Fix memory the proxy can recall.
    let store = MemoryStore::open(&mem_root).unwrap();
    let records = store.load_scope("personal");
    assert_eq!(records.len(), 1, "one novel healed failure → one memory");
    let rec = &records[0];
    assert!(matches!(rec.kind, MemoryKind::Fix));
    assert!(rec.body.contains("self-healed"));
    assert!(
        rec.body.contains(&sh_path(&base.join("out"))),
        "memory body must name the directory heal"
    );
    assert!(!rec.embedding.is_empty(), "memory must be embedded for recall");

    let _ = std::fs::remove_dir_all(&base);
    let _ = std::fs::remove_dir_all(&mem_root);
}

#[test]
fn structural_immunity_generalizes_across_unrelated_commands() {
    use axiom_engine::heal_memory::{fingerprint, HealMemory};
    use axiom_engine::self_heal::Heal::{CreatedDirectory, StructurallyImmunized};

    let anchor = unique_tmp("structural_anchor");
    std::fs::create_dir_all(&anchor).unwrap();
    let memory = unique_tmp("structural_mem").with_extension("json");
    let opts = SupervisorOptions {
        max_restarts: 3,
        heal_memory_path: Some(memory.clone()),
        anchor: Some(anchor.clone()),
        ..SupervisorOptions::default()
    };

    // Program A: fails on missing `dist`, heals reactively, learns it directly.
    let a_args = vec!["-c".to_string(), "echo a > dist/a.out".to_string()];
    let a = supervise_opts(tiny_pipeline(), posix_shell(), a_args, opts.clone());
    assert!(a.success);
    assert_eq!(a.heals, vec![CreatedDirectory(anchor.join("dist"))]);

    // Fresh environment again.
    std::fs::remove_dir_all(anchor.join("dist")).unwrap();

    // Program B: a *different* command, unrelated to A by fingerprint. Only
    // one program (A) has corroborated `dist` so far -- below the
    // corroboration threshold -- so B gets no structural immunity yet and
    // must heal reactively too, exactly like A did.
    let b_args = vec!["-c".to_string(), "echo b > dist/b.out".to_string()];
    let b = supervise_opts(tiny_pipeline(), posix_shell(), b_args, opts.clone());
    assert!(b.success);
    assert_eq!(
        b.heals,
        vec![CreatedDirectory(anchor.join("dist"))],
        "a single prior program must not yet generalize"
    );

    std::fs::remove_dir_all(anchor.join("dist")).unwrap();

    // Program C: brand new, never run before, zero history of its own. With
    // A and B both independently corroborating `dist`, it is now a structural
    // heal -- C gets it pre-created before its very first attempt.
    let c_args = vec!["-c".to_string(), "echo c > dist/c.out".to_string()];
    let fp_c = fingerprint(&posix_shell(), &c_args);
    assert!(HealMemory::load(&memory).record(&fp_c).is_none(), "C must be genuinely unseen");

    let c = supervise_opts(tiny_pipeline(), posix_shell(), c_args, opts);
    assert!(c.success);
    assert_eq!(c.attempts, 1, "structural immunity must prevent C's first failure entirely");
    assert_eq!(c.heals, vec![StructurallyImmunized(anchor.join("dist"), 2)]);
    assert_eq!(
        std::fs::read_to_string(anchor.join("dist").join("c.out")).unwrap().trim(),
        "c"
    );

    // The successful structural heal must become C's own earned experience,
    // not just a one-off borrowed from A and B.
    let mem = HealMemory::load(&memory);
    let record = mem.record(&fp_c).expect("C must now have its own record");
    assert!(record.dirs.contains(&PathBuf::from("dist")), "C must own the dir heal directly now");
    assert_eq!(record.immunizations, 1, "the successful structural run must reinforce C's own belief");

    let _ = std::fs::remove_dir_all(&anchor);
    let _ = std::fs::remove_file(&memory);
}

#[test]
fn failure_history_persists_to_vibe_when_requested() {
    let vibe = unique_tmp("vibe").with_extension("bin");
    let report = supervise_with_vibe(
        tiny_pipeline(),
        posix_shell(),
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