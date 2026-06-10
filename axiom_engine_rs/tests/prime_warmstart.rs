//! Integration test for `axiom prime` (warm-start vibe priming).
//!
//! Builds a tiny pipeline, writes a small source tree to a temp dir, primes it,
//! and asserts the crawl absorbed the expected files and persisted a master vibe.

use std::fs;
use std::path::PathBuf;

use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::prime::run_prime;
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
    std::env::temp_dir().join(format!("axiom_prime_{tag}_{nanos}"))
}

#[test]
fn prime_absorbs_sources_and_writes_vibe() {
    let root = unique_tmp("repo");
    let nested = root.join("src").join("inner");
    fs::create_dir_all(&nested).unwrap();
    // Two priming-eligible source files...
    fs::write(root.join("lib.rs"), "pub fn alpha() -> usize { 1 }\n").unwrap();
    fs::write(
        nested.join("mod.rs"),
        "pub struct Beta;\nimpl Beta { pub fn go(&self) {} }\n",
    )
    .unwrap();
    // ...one ignored extension, and one inside a skip-dir (must NOT be absorbed).
    fs::write(root.join("notes.bin"), "not source\n").unwrap();
    let skip = root.join("target");
    fs::create_dir_all(&skip).unwrap();
    fs::write(skip.join("artifact.rs"), "fn should_be_skipped() {}\n").unwrap();

    let vibe_path = unique_tmp("vibe").with_extension("bin");
    let pipeline = tiny_pipeline();

    let report = run_prime(&root, &pipeline, &vibe_path).expect("prime must succeed");

    assert_eq!(
        report.files_absorbed, 2,
        "only the two real source files (not .bin, not target/) should be absorbed"
    );
    assert!(report.tokens_absorbed > 0, "some tokens must be absorbed");
    assert_eq!(report.vibe_path, vibe_path);
    assert!(vibe_path.exists(), "master vibe must be persisted to disk");
    assert!(
        fs::metadata(&vibe_path).unwrap().len() > 0,
        "persisted vibe must be non-empty"
    );

    // Cleanup (best-effort).
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_file(&vibe_path);
}

#[test]
fn prime_on_empty_dir_writes_identity_vibe() {
    // Priming a directory with no source still succeeds and seeds an identity
    // master (commit of the untouched identity-initialised W̃).
    let root = unique_tmp("empty");
    fs::create_dir_all(&root).unwrap();
    let vibe_path = unique_tmp("vibe_empty").with_extension("bin");

    let pipeline = tiny_pipeline();
    let report = run_prime(&root, &pipeline, &vibe_path).expect("prime must succeed on empty dir");

    assert_eq!(report.files_absorbed, 0);
    assert_eq!(report.tokens_absorbed, 0);
    assert!(vibe_path.exists(), "an identity master vibe is still written");

    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_file(&vibe_path);
}
