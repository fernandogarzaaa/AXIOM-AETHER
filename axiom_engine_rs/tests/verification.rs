//! Passkey / needle verification for the active-compression pipeline.
//!
//! This is the convergence probe described in the project plan: it answers
//! the question "do our meta-trained projection matrices actually lift a
//! unique secret token out of the adapted hidden state, or does `recall_norm`
//! collapse to zero the way it does with random-init weights?"
//!
//! Flow:
//!   1. Meta-train a fresh checkpoint on the local repo files (the container
//!      is ephemeral, so we cannot rely on a checkpoint existing on disk).
//!   2. Load that checkpoint through the real `InferencePipeline`.
//!   3. Build a synthetic "passkey sequence": a long (>100 token) block of
//!      code context with a unique needle embedded in the middle.
//!   4. Ingest the context through the TTT compression path
//!      (`adapt_session_blocking`) to mutate the session's fast weights.
//!   5. Run the associative recall pass (`extract_memory_vector_blocking`)
//!      with a query that targets the secret token.
//!   6. Print the exact fingerprint block and assert `recall_norm > 0`.

use std::time::Instant;

use axiom_engine::config::AxiomConfig;
use axiom_engine::context_compressor::{adapt_session_blocking, extract_memory_vector_blocking};
use axiom_engine::inference::InferencePipeline;
use axiom_engine::meta_train::MetaTrainer;
use candle_core::Device;

/// The default config the engine runs with locally — must match what
/// `meta-train` writes so the checkpoint shapes load cleanly.
fn default_config() -> AxiomConfig {
    AxiomConfig {
        d_model: 64,
        n_layers: 2,
        vocab_size: 256,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    }
}

/// A >100-token block of code context with a unique needle in the middle.
fn passkey_context() -> String {
    let preamble: String = (0..60)
        .map(|i| format!("fn helper_{i}(arg_{i}: usize) -> usize {{ arg_{i} + {i} }}"))
        .collect::<Vec<_>>()
        .join("\n");
    let postamble: String = (0..60)
        .map(|i| format!("let result_{i} = helper_{i}({i});"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{preamble}\n\
         const AXIOM_SECRET_VALIDATION_TOKEN: &str = \"TTT_CONVERGED_SUCCESSFULLY\";\n\
         {postamble}\n"
    )
}

#[test]
fn passkey_recall_is_alive_after_meta_training() {
    let config = default_config();
    let device = Device::Cpu;

    // --- 1. Meta-train a checkpoint into a temp path -----------------------
    let tmp = std::env::temp_dir().join(format!(
        "axiom_passkey_ckpt_{}.safetensors",
        std::process::id()
    ));
    let checkpoint_path = tmp.to_string_lossy().to_string();

    // The crate manifest dir is the Rust sub-project; its parent is the repo
    // root, which has many more ingestible files for the dataset.
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")));

    let mut trainer = MetaTrainer::build(
        config.clone(),
        device.clone(),
        repo_root,
        checkpoint_path.clone(),
        /* batch_size   */ 8,
        /* seq_len      */ 32,
        /* max_files    */ 256,
        /* max_sequences*/ 1024,
        /* seed         */ 42,
    )
    .expect("meta-trainer must build from repo files");

    assert!(
        trainer.dataset_len() > 0,
        "meta-train dataset must contain at least one window"
    );

    // A short but real training run: enough optimiser steps to move the
    // projection matrices off their random init.
    let final_loss = trainer
        .run(/* epochs */ 1, /* steps_per_epoch */ 60, /* lr */ 1e-3)
        .expect("meta-training run must succeed");
    println!("[passkey] meta-train final loss = {final_loss:.4}");

    // --- 2. Load the trained checkpoint ------------------------------------
    let pipeline = InferencePipeline::with_checkpoint(config, device, checkpoint_path.clone())
        .expect("pipeline must load the freshly meta-trained checkpoint");

    // --- 3 + 4. Ingest the passkey context to mutate fast weights ----------
    let context = passkey_context();
    let ctx_tokens = pipeline.encode_text(&context);
    assert!(
        ctx_tokens.len() > 100,
        "passkey context must exceed 100 tokens (got {})",
        ctx_tokens.len()
    );

    let mut states = pipeline
        .init_session_states()
        .expect("session states must initialise");

    // Capture the pre-adaptation norms so we can show W̃ actually moved.
    let pre = extract_memory_vector_blocking(
        &pipeline,
        &mut states.clone(),
        &pipeline.encode_text("probe"),
        "passkey-pre",
        0,
        Instant::now(),
        8,
    )
    .expect("pre-adaptation probe must run");

    adapt_session_blocking(&pipeline, &mut states, &ctx_tokens)
        .expect("context ingestion (TTT adaptation) must succeed");

    // --- 5. Associative recall targeting the secret ------------------------
    let query = "What is the value of AXIOM_SECRET_VALIDATION_TOKEN?";
    let q_tokens = pipeline.encode_text(query);
    let fingerprint = extract_memory_vector_blocking(
        &pipeline,
        &mut states,
        &q_tokens,
        "passkey-verify",
        ctx_tokens.len(),
        Instant::now(),
        32,
    )
    .expect("associative recall pass must succeed");

    // --- 6. Print the exact fingerprint block for visual inspection --------
    println!("\n================ PASSKEY FINGERPRINT ================");
    println!("{}", fingerprint.to_prompt_block());
    println!("====================================================");
    println!(
        "[passkey] pre-adapt Frobenius norms : {:?}",
        pre.layer_frobenius_norms
    );
    println!(
        "[passkey] post-adapt Frobenius norms: {:?}",
        fingerprint.layer_frobenius_norms
    );
    println!("[passkey] recall_norm = {:.6}", fingerprint.recall_norm);
    println!("[passkey] recall_l1   = {:.6}", fingerprint.recall_l1);

    // --- Assertions: the recall vector must be alive -----------------------
    assert!(
        fingerprint.recall_norm > 0.0,
        "recall_norm collapsed to zero — trained projections are NOT lifting \
         the secret out of the hidden state (recall_norm={})",
        fingerprint.recall_norm
    );
    assert!(
        fingerprint.recall_norm.is_finite(),
        "recall_norm must be finite, got {}",
        fingerprint.recall_norm
    );
    assert_eq!(
        fingerprint.context_tokens_processed,
        ctx_tokens.len(),
        "fingerprint must report the full ingested-token count"
    );
    assert!(
        fingerprint
            .layer_frobenius_norms
            .iter()
            .all(|n| n.is_finite() && *n > 0.0),
        "every layer's adapted W̃ must have a finite, non-zero Frobenius norm"
    );

    // Best-effort cleanup of the temp checkpoint.
    let _ = std::fs::remove_file(&checkpoint_path);
}
