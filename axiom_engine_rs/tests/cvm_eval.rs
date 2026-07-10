//! S5 (CVM cost stack) behavior eval gate -- a thin `cargo test` entry point
//! for `scripts/cvm_eval.sh`, the actual harness.
//!
//! Gated behind the `live-eval` feature (see `Cargo.toml`'s `[[test]]`
//! declaration for this file: `required-features = ["live-eval"]`), so
//! ordinary `cargo test` never discovers or compiles this file. Running it
//! bills the operator's own Anthropic account for ~24 real API calls
//! (`claude -p --model claude-haiku-4-5`). Never run this in CI or
//! unattended.
//!
//! ```text
//! cargo test --features live-eval --test cvm_eval -- --ignored --nocapture
//! ```
//!
//! (also `--ignored`: the test itself carries `#[ignore]` as a second,
//! redundant guard against accidental execution even if someone builds with
//! the feature enabled by habit.)

use std::process::Command;

#[test]
#[ignore = "spends real Anthropic API credits; run deliberately via scripts/cvm_eval.sh"]
fn cvm_eval_harness_passes() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("axiom_engine_rs has a parent directory")
        .to_path_buf();
    let script = repo_root.join("scripts").join("cvm_eval.sh");
    assert!(
        script.exists(),
        "scripts/cvm_eval.sh not found at {}",
        script.display()
    );

    let status = Command::new("bash")
        .arg(&script)
        .current_dir(&repo_root)
        .status()
        .expect("failed to spawn scripts/cvm_eval.sh");

    assert!(
        status.success(),
        "cvm_eval.sh reported FAIL (exit {status:?}) -- see the printed per-check \
         breakdown above and the written bench/cvm/RESULTS-<date>.md report"
    );
}
