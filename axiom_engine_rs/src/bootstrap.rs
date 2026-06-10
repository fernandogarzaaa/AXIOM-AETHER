//! Local checkpoint bootstrap.
//!
//! A fresh clone ships no checkpoint, so the proxy boots on **random weights** —
//! the neural recall fingerprint is then pure noise (`recall_norm = 0`). That is
//! the single worst first-run experience. This module trains a small model
//! *locally and offline* (the procedural dataset needs no corpus and no network)
//! and persists it, so `axiom init` can guarantee the runtime always loads real,
//! converged-enough weights instead of noise.
//!
//! The bootstrap is intentionally tiny and bounded: it matches the CPU-friendly
//! runtime dims (d_model=64, n_layers=2) and a short epoch/step budget, so it
//! finishes in seconds. Users who want the full scaled BPE model still run the
//! `train_tokenizer` + `train_semantic` pipeline; this just removes the
//! random-weights cliff.

use candle_core::{Device, Result};

use crate::config::AxiomConfig;
use crate::train::AxiomTrainer;

/// CPU-friendly runtime dims — must match the legacy base the server/prime use.
pub fn bootstrap_config() -> AxiomConfig {
    AxiomConfig {
        d_model: 64,
        n_layers: 2,
        vocab_size: 256,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

/// Train a small model on the procedural dataset and persist it to
/// `checkpoint_path`. Returns the number of training steps run.
///
/// Budget is overridable via `AXIOM_INIT_EPOCHS` / `AXIOM_INIT_STEPS`.
pub fn train_bootstrap_checkpoint(checkpoint_path: &str, device: Device) -> Result<usize> {
    let epochs = env_usize("AXIOM_INIT_EPOCHS", 3);
    let steps = env_usize("AXIOM_INIT_STEPS", 60);
    let mut trainer =
        AxiomTrainer::with_settings(bootstrap_config(), device, checkpoint_path, 8, 32)?;
    trainer.run_training_epochs(epochs, steps)?;
    Ok(epochs * steps)
}

/// Bootstrap a checkpoint at `checkpoint_path` iff one does not already exist.
///
/// Returns `Ok(true)` when a checkpoint was trained, `Ok(false)` when one was
/// already present (no-op). Best-effort: callers should not treat a training
/// error as fatal to `init`.
pub fn ensure_checkpoint(checkpoint_path: &str, device: Device) -> Result<bool> {
    if std::path::Path::new(checkpoint_path).exists() {
        return Ok(false);
    }
    train_bootstrap_checkpoint(checkpoint_path, device)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{InferencePipeline, InferenceRuntimeOptions};

    fn tmp_ckpt(tag: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("axiom_bootstrap_{tag}_{nanos}.bin"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn ensure_checkpoint_trains_then_is_noop() {
        std::env::set_var("AXIOM_INIT_EPOCHS", "1");
        std::env::set_var("AXIOM_INIT_STEPS", "3");
        let path = tmp_ckpt("noop");

        // First call trains and writes a real checkpoint.
        let trained = ensure_checkpoint(&path, Device::Cpu).expect("bootstrap must succeed");
        assert!(trained, "first call should train");
        assert!(std::path::Path::new(&path).exists(), "checkpoint must be written");

        // The persisted checkpoint must load back into a pipeline (i.e. it is a
        // real, dimension-correct model, not random in-memory weights).
        let runtime = InferenceRuntimeOptions {
            tokenizer_path: None,
            context_api_url: None,
            context_api_key: None,
            max_context_tokens: 0,
        };
        let _pipeline = InferencePipeline::with_checkpoint_and_options(
            bootstrap_config(),
            Device::Cpu,
            path.clone(),
            runtime,
        )
        .expect("bootstrapped checkpoint must load");

        // Second call is a no-op because the file now exists.
        let trained_again = ensure_checkpoint(&path, Device::Cpu).expect("second call ok");
        assert!(!trained_again, "second call should not retrain");

        let _ = std::fs::remove_file(&path);
    }
}
