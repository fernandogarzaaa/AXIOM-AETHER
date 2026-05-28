//! Dynamic data harvester + production meta-training run.
//!
//! Aggressively crawls one or more local source trees (the repo, vendored
//! dependencies, sibling code directories, …), slices them into
//! next-token-prediction context windows, and runs an expanded multi-epoch
//! meta-training schedule with cosine decay on both the outer meta-learning
//! rate α and the inner test-time rate η. The converged projection matrices
//! are serialised to `./checkpoints/axiom_production.bin`.
//!
//! Usage:
//!
//! ```bash
//! # Crawl the current repo (default) and train with the default schedule:
//! cargo run --release --bin harvest
//!
//! # Crawl several trees and override the schedule via env:
//! AXIOM_HARVEST_EPOCHS=8 AXIOM_HARVEST_STEPS=400 \
//! AXIOM_HARVEST_ALPHA_START=3e-3 AXIOM_HARVEST_ALPHA_END=1e-5 \
//! AXIOM_HARVEST_ETA_START=2e-3 AXIOM_HARVEST_ETA_END=5e-4 \
//!   cargo run --release --bin harvest -- ./ ../ ~/.cargo/registry/src
//! ```

use std::path::PathBuf;

use axiom_engine::config::AxiomConfig;
use axiom_engine::meta_train::{LrSchedule, MetaTrainer};
use candle_core::{Device, Result};

const PRODUCTION_CHECKPOINT: &str = "./checkpoints/axiom_production.bin";

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    // ---- Resolve crawl roots ------------------------------------------------
    // Positional args are directories to crawl. With none, default to the
    // current dir and its parent (so a run from `axiom_engine_rs/` still
    // sweeps the wider repo).
    let mut roots: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if roots.is_empty() {
        let cwd = std::env::current_dir()
            .map_err(|e| candle_core::Error::Msg(format!("could not resolve cwd: {e}")))?;
        if let Some(parent) = cwd.parent() {
            roots.push(parent.to_path_buf());
        }
        roots.push(cwd);
    }

    // ---- Hyper-parameters (env-overridable) --------------------------------
    let epochs: usize = env_parse("AXIOM_HARVEST_EPOCHS", 6);
    let steps_per_epoch: usize = env_parse("AXIOM_HARVEST_STEPS", 300);
    let batch_size: usize = env_parse("AXIOM_HARVEST_BATCH", 16);
    let seq_len: usize = env_parse("AXIOM_HARVEST_SEQ_LEN", 64);
    let max_files: usize = env_parse("AXIOM_HARVEST_MAX_FILES", 4096);
    let max_sequences: usize = env_parse("AXIOM_HARVEST_MAX_SEQS", 32768);
    let seed: u64 = env_parse("AXIOM_HARVEST_SEED", 1337);

    let schedule = LrSchedule {
        alpha_start: env_parse("AXIOM_HARVEST_ALPHA_START", 3e-3),
        alpha_end: env_parse("AXIOM_HARVEST_ALPHA_END", 1e-5),
        eta_start: env_parse("AXIOM_HARVEST_ETA_START", 2e-3_f32),
        eta_end: env_parse("AXIOM_HARVEST_ETA_END", 5e-4_f32),
    };

    let checkpoint = std::env::var("AXIOM_HARVEST_CHECKPOINT")
        .unwrap_or_else(|_| PRODUCTION_CHECKPOINT.to_string());

    // Must match the runtime engine config so the checkpoint loads cleanly.
    let config = AxiomConfig {
        d_model: 64,
        n_layers: 2,
        vocab_size: 256,
        lr_inner: schedule.eta_start,
        norm_eps: 1e-6,
    };
    let device = Device::Cpu;

    println!("[harvest] crawling {} root(s):", roots.len());
    for r in &roots {
        println!("           - {}", r.display());
    }

    let mut trainer = MetaTrainer::build_multi(
        config.clone(),
        device,
        roots,
        checkpoint.clone(),
        batch_size,
        seq_len,
        max_files,
        max_sequences,
        seed,
    )?;

    println!(
        "[harvest] dataset: {} windows (seq_len={}, batch={})",
        trainer.dataset_len(),
        seq_len,
        batch_size
    );
    println!(
        "[harvest] schedule: epochs={} steps/epoch={} | α {:.3e}→{:.3e} | η {:.3e}→{:.3e}",
        epochs,
        steps_per_epoch,
        schedule.alpha_start,
        schedule.alpha_end,
        schedule.eta_start,
        schedule.eta_end,
    );

    let final_loss = trainer.run_with_schedule(epochs, steps_per_epoch, schedule)?;
    println!("[harvest] done. final meta-loss = {final_loss:.5} → {checkpoint}");

    Ok(())
}
