//! Passkey / needle convergence harness for the active-compression pipeline.
//!
//! This probe answers the question "do our meta-trained outer-loop projection
//! matrices (W_q / W_k / W_v) actually lift a unique secret token out of the
//! adapted fast-weight state, or does `recall_norm` collapse to zero the way it
//! does with random-init weights?"
//!
//! Phases:
//!   1. Checkpoint ingestion & pipeline warmup — load a meta-trained weights
//!      binary into the `InferencePipeline` (generated here if absent, since the
//!      execution container is ephemeral) and init a clean session slice.
//!   2. Passkey needle-in-a-haystack — build a >200-token bloated code context
//!      with a unique needle, then stream it through `forward_native` (via
//!      `adapt_session_blocking`) to trigger the test-time gradient updates
//!      W̃_t = W̃_{t-1} − η ∇L(W̃_{t-1}; x_t).
//!   3. Associative recall extraction & assertion — run the recall pass with a
//!      prompt querying the secret, print the dense fingerprint block, and
//!      assert recall_norm breaks past 0.000 into a healthy dynamic range.

use std::time::Instant;

use axiom_engine::config::AxiomConfig;
use axiom_engine::context_compressor::{adapt_session_blocking, extract_memory_vector_blocking};
use axiom_engine::inference::InferencePipeline;
use axiom_engine::meta_train::MetaTrainer;
use candle_core::Device;

/// Targeted meta-trained weights binary the harness loads from.
const CONVERGED_CHECKPOINT: &str = "./checkpoints/axiom_converged.bin";

/// The unique semantic needle. Must not appear in the training corpus.
const NEEDLE: &str =
    "const AXIOM_SECRET_VALIDATION_TOKEN: &str = \"TTT_ALIVE_AND_CONVERGING\";";

/// Healthy dynamic range for a live recall vector. Below `LO` the context is
/// collapsing through LayerNorm (untrained projections); above `HI` it has
/// blown up.
const HEALTHY_LO: f32 = 0.350;
const HEALTHY_HI: f32 = 1.500;

/// The engine's local default config — must match what `meta-train` writes so
/// the checkpoint tensor shapes load cleanly.
fn default_config() -> AxiomConfig {
    AxiomConfig {
        d_model: 64,
        n_layers: 2,
        vocab_size: 256,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    }
}

/// A >200-token block of structural source code with the needle injected
/// mid-way through the payload (not at the start or end).
fn passkey_haystack() -> String {
    let pre: String = (0..70)
        .map(|i| format!("fn module_{i}_handler(ctx: &Ctx, arg_{i}: usize) -> usize {{ ctx.scale(arg_{i}) + {i} }}"))
        .collect::<Vec<_>>()
        .join("\n");
    let post: String = (0..70)
        .map(|i| format!("let route_{i} = router.bind(\"/api/v{i}\", module_{i}_handler);"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{pre}\n{NEEDLE}\n{post}\n")
}

/// Resolve a loaded pipeline: load the converged checkpoint if it exists,
/// otherwise meta-train one into `CONVERGED_CHECKPOINT` first (the container is
/// ephemeral, so we cannot assume a binary is already on disk).
fn load_or_train_pipeline(config: &AxiomConfig, device: &Device) -> (InferencePipeline, String) {
    if !std::path::Path::new(CONVERGED_CHECKPOINT).exists() {
        println!("[passkey] no converged checkpoint at {CONVERGED_CHECKPOINT}; meta-training one");
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")));

        let mut trainer = MetaTrainer::build(
            config.clone(),
            device.clone(),
            repo_root,
            CONVERGED_CHECKPOINT.to_string(),
            /* batch_size    */ 8,
            /* seq_len       */ 32,
            /* max_files     */ 256,
            /* max_sequences */ 1024,
            /* seed          */ 42,
        )
        .expect("meta-trainer must build from repo files");
        assert!(trainer.dataset_len() > 0, "meta-train dataset must be non-empty");
        let final_loss = trainer
            .run(/* epochs */ 1, /* steps_per_epoch */ 60, /* lr */ 1e-3)
            .expect("meta-training run must succeed");
        println!("[passkey] meta-train final loss = {final_loss:.4}");
    }

    let pipeline = InferencePipeline::with_checkpoint(
        config.clone(),
        device.clone(),
        CONVERGED_CHECKPOINT.to_string(),
    )
    .expect("pipeline must load the converged checkpoint");
    (pipeline, CONVERGED_CHECKPOINT.to_string())
}

#[test]
fn passkey_recall_breaks_past_zero_after_convergence() {
    let config = default_config();
    let device = Device::Cpu;

    // --- PHASE 1: checkpoint ingestion & pipeline warmup -------------------
    let (pipeline, checkpoint) = load_or_train_pipeline(&config, &device);
    println!("[passkey] loaded checkpoint: {checkpoint}");

    // Clean, multi-tenant session slice (per-layer identity fast weights).
    let mut states = pipeline
        .init_session_states()
        .expect("session states must initialise");

    // --- PHASE 2: passkey needle-in-a-haystack -----------------------------
    let context = passkey_haystack();
    let ctx_tokens = pipeline.encode_text(&context);
    assert!(
        ctx_tokens.len() > 200,
        "context must exceed 200 tokens of bloat (got {})",
        ctx_tokens.len()
    );

    // Baseline probe before adaptation, to show the fast weights actually move.
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

    // Stream the haystack through forward_native to trigger TTT updates.
    adapt_session_blocking(&pipeline, &mut states, &ctx_tokens)
        .expect("context ingestion (TTT adaptation) must succeed");

    // --- PHASE 3: associative recall extraction & assertion ----------------
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

    println!("\n================ PASSKEY FINGERPRINT ================");
    println!("{}", fingerprint.to_prompt_block());
    println!("====================================================");
    println!("[telemetry] needle              : {NEEDLE}");
    println!("[telemetry] context_tokens      : {}", ctx_tokens.len());
    println!(
        "[telemetry] pre-adapt  W̃ norms  : {:?}",
        pre.layer_frobenius_norms
    );
    println!(
        "[telemetry] post-adapt W̃ norms  : {:?}",
        fingerprint.layer_frobenius_norms
    );
    println!("[telemetry] pre-adapt recall_norm  = {:.6}", pre.recall_norm);
    println!(
        "[telemetry] post-adapt recall_norm = {:.6}  (healthy {HEALTHY_LO}..{HEALTHY_HI})",
        fingerprint.recall_norm
    );
    println!("[telemetry] recall_l1              = {:.6}", fingerprint.recall_l1);

    // The recall vector must be alive and finite...
    assert!(
        fingerprint.recall_norm.is_finite(),
        "recall_norm must be finite, got {}",
        fingerprint.recall_norm
    );
    assert!(
        fingerprint.recall_norm > 0.0,
        "recall_norm collapsed to zero — trained projections are NOT lifting \
         the secret out of the hidden state (recall_norm={})",
        fingerprint.recall_norm
    );
    // ...and within the healthy dynamic range, confirming convergence.
    assert!(
        fingerprint.recall_norm >= HEALTHY_LO && fingerprint.recall_norm <= HEALTHY_HI,
        "recall_norm={} is outside the healthy convergence range {}..{}",
        fingerprint.recall_norm,
        HEALTHY_LO,
        HEALTHY_HI
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
        "every adapted layer W̃ must have a finite, non-zero Frobenius norm"
    );
}
