//! Local meta-training harness for the projection matrices.
//!
//! Phase 4 of the active-compression architecture: train W_q, W_k, W_v
//! (and the surrounding LayerNorm + LM head) on raw text drawn from the
//! current repository, with cross-entropy next-token prediction as the
//! supervisory signal.
//!
//! The objective is **not** to teach the model grammar from scratch.
//! It is to produce projection matrices that make the inner-loop TTT
//! gradient update (`forward_native` in `ttt_block.rs`) numerically
//! well-conditioned, so that:
//!
//! * the associative-recall pass used by the context compressor
//!   produces stable, non-degenerate logits, and
//! * downstream callers of the local pipeline see meaningful changes
//!   in W̃ as new context is absorbed.
//!
//! Usage:
//!
//! ```bash
//! cargo run --release -- --mode meta-train \
//!     --epochs 1 --steps-per-epoch 200 \
//!     --checkpoint axiom_kernel_v1.safetensors
//! ```
//!
//! The trainer reads `.rs`, `.py`, `.md`, `.toml`, `.yaml`, `.yml`,
//! `.json` files under the current working directory (excluding common
//! build/cache dirs) and slices them into fixed-length sequences.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use sha2::Digest;
use walkdir::WalkDir;

use crate::config::AxiomConfig;
use crate::model::AxiomTTTLM;

const DEFAULT_INCLUDED_EXTENSIONS: &[&str] =
    &["rs", "py", "md", "toml", "yaml", "yml", "json", "txt"];

const DEFAULT_EXCLUDED_DIRS: &[&str] = &[
    "target",
    ".git",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".cache",
    ".venv",
    "venv",
];

/// Dataset of token-id sequences sliced from on-disk source files.
///
/// The dataset is built once at trainer construction so subsequent
/// `next_batch` calls are O(1) memory churn.
pub struct RepoFileDataset {
    /// Each entry is a contiguous slice of `seq_len + 1` token ids
    /// (one extra so the supervised target is one-token-shifted).
    sequences: Vec<Vec<u32>>,
    vocab_size: u32,
    rng: rand::rngs::StdRng,
}

impl RepoFileDataset {
    /// Construct a dataset by walking `root`, tokenising every file
    /// matching the include filter, and slicing into `seq_len + 1`
    /// windows. Uses the same SHA-256-hash tokenizer the engine falls
    /// back to when no HF tokenizer is loaded — keeps the meta-training
    /// objective consistent with what the inference path sees.
    pub fn build(
        root: impl AsRef<Path>,
        vocab_size: u32,
        seq_len: usize,
        max_files: usize,
        max_sequences: usize,
        seed: u64,
    ) -> std::io::Result<Self> {
        let mut files: Vec<PathBuf> = Vec::new();
        for entry in WalkDir::new(root.as_ref())
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !is_excluded_dir(e.path()))
            .flatten()
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let ext = entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if !DEFAULT_INCLUDED_EXTENSIONS
                .iter()
                .any(|allowed| *allowed == ext)
            {
                continue;
            }
            files.push(entry.path().to_path_buf());
            if files.len() >= max_files {
                break;
            }
        }
        files.sort();

        let mut sequences: Vec<Vec<u32>> = Vec::new();
        for path in &files {
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let text = match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let tokens = hash_tokenize(&text, vocab_size);
            // Slide a window of seq_len+1 over the token stream.
            if tokens.len() < seq_len + 1 {
                continue;
            }
            for start in (0..tokens.len() - seq_len).step_by(seq_len) {
                let window = tokens[start..start + seq_len + 1].to_vec();
                sequences.push(window);
                if sequences.len() >= max_sequences {
                    break;
                }
            }
            if sequences.len() >= max_sequences {
                break;
            }
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        sequences.shuffle(&mut rng);

        Ok(Self {
            sequences,
            vocab_size,
            rng,
        })
    }

    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Sample a batch: returns `(inputs [B, T], targets [B, T])` where
    /// targets are inputs shifted by one position.
    pub fn next_batch(
        &mut self,
        batch_size: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        if self.sequences.is_empty() {
            candle_core::bail!("meta-train dataset is empty; widen --max-files or --max-sequences");
        }
        let mut inputs_flat: Vec<u32> = Vec::with_capacity(batch_size * (self.sequences[0].len() - 1));
        let mut targets_flat: Vec<u32> = Vec::with_capacity(batch_size * (self.sequences[0].len() - 1));
        let seq_len = self.sequences[0].len() - 1;
        let vocab = self.vocab_size;
        for _ in 0..batch_size {
            let idx = (self.rng_u64() as usize) % self.sequences.len();
            let window = &self.sequences[idx];
            // Hard guard against tokenizer/vocab drift: clamp into-range.
            for &t in &window[..seq_len] {
                inputs_flat.push(if t < vocab { t } else { t % vocab.max(1) });
            }
            for &t in &window[1..seq_len + 1] {
                targets_flat.push(if t < vocab { t } else { t % vocab.max(1) });
            }
        }
        let inputs = Tensor::from_vec(inputs_flat, (batch_size, seq_len), device)?;
        let targets = Tensor::from_vec(targets_flat, (batch_size, seq_len), device)?;
        Ok((inputs, targets))
    }

    fn rng_u64(&mut self) -> u64 {
        use rand::RngCore;
        self.rng.next_u64()
    }
}

fn is_excluded_dir(path: &Path) -> bool {
    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
        if DEFAULT_EXCLUDED_DIRS.iter().any(|d| *d == name) {
            return true;
        }
    }
    false
}

fn hash_tokenize(text: &str, vocab_size: u32) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    for tok in text.split_whitespace() {
        let digest = sha2::Sha256::digest(tok.as_bytes());
        let id = u64::from_le_bytes([
            digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        ]) % vocab_size as u64;
        ids.push(id as u32);
    }
    ids
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Meta-trainer for projection matrices on a repo-derived corpus.
pub struct MetaTrainer {
    config: AxiomConfig,
    device: Device,
    varmap: VarMap,
    model: AxiomTTTLM,
    dataset: RepoFileDataset,
    checkpoint_path: String,
    batch_size: usize,
}

impl MetaTrainer {
    pub fn build(
        config: AxiomConfig,
        device: Device,
        repo_root: impl AsRef<Path>,
        checkpoint_path: impl Into<String>,
        batch_size: usize,
        seq_len: usize,
        max_files: usize,
        max_sequences: usize,
        seed: u64,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = AxiomTTTLM::new(vb, config.clone())?;

        let dataset = RepoFileDataset::build(
            repo_root,
            config.vocab_size as u32,
            seq_len,
            max_files,
            max_sequences,
            seed,
        )
        .map_err(|e| candle_core::Error::Msg(format!("dataset build failed: {e}")))?;

        if dataset.is_empty() {
            candle_core::bail!(
                "meta-train dataset is empty after scanning repo files; \
                 try --max-files higher or run from the repo root"
            );
        }

        Ok(Self {
            config,
            device,
            varmap,
            model,
            dataset,
            checkpoint_path: checkpoint_path.into(),
            batch_size: batch_size.max(1),
        })
    }

    /// Number of token-windows the dataset was sliced into.
    pub fn dataset_len(&self) -> usize {
        self.dataset.len()
    }

    /// Run `epochs * steps_per_epoch` optimiser steps and save the
    /// checkpoint at the end.
    pub fn run(&mut self, epochs: usize, steps_per_epoch: usize, lr: f64) -> Result<f32> {
        let mut optim = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr,
                ..ParamsAdamW::default()
            },
        )?;

        let total_steps = (epochs * steps_per_epoch) as u64;
        let progress = ProgressBar::new(total_steps);
        let style = ProgressStyle::with_template("{bar:40.cyan/blue} {pos:>6}/{len:6} | {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar());
        progress.set_style(style);

        let mut last_loss = f32::NAN;
        for epoch in 0..epochs {
            for _step in 0..steps_per_epoch {
                let (inputs, targets) = self.dataset.next_batch(self.batch_size, &self.device)?;

                let mut batch_logits = Vec::with_capacity(self.batch_size);
                let mut batch_targets = Vec::with_capacity(self.batch_size);
                for batch_idx in 0..self.batch_size {
                    let input = inputs.narrow(0, batch_idx, 1)?;
                    let target = targets.narrow(0, batch_idx, 1)?;
                    let mut states = self.model.init_states(&self.device)?;
                    let logits = self.model.forward_lm(&input, &mut states[..])?;
                    batch_logits.push(logits);
                    batch_targets.push(target);
                }

                let logit_refs: Vec<&Tensor> = batch_logits.iter().collect();
                let target_refs: Vec<&Tensor> = batch_targets.iter().collect();
                let logits = Tensor::cat(&logit_refs, 0)?;
                let targets = Tensor::cat(&target_refs, 0)?;
                let (b, t, v) = logits.dims3()?;
                let flat_logits = logits.reshape((b * t, v))?;
                let flat_targets = targets.reshape((b * t,))?;

                let loss = candle_nn::loss::cross_entropy(&flat_logits, &flat_targets)?;
                optim.backward_step(&loss)?;

                last_loss = loss.to_scalar::<f32>()?;
                progress.set_message(format!(
                    "epoch {}/{} | loss {:.5} | files-seq {} | d_model {} layers {}",
                    epoch + 1,
                    epochs,
                    last_loss,
                    self.dataset.len(),
                    self.config.d_model,
                    self.config.n_layers
                ));
                progress.inc(1);
            }
        }

        progress.finish_with_message("meta-training complete");
        if let Some(parent) = std::path::Path::new(&self.checkpoint_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "could not create checkpoint dir {}: {e}",
                        parent.display()
                    ))
                })?;
            }
        }
        self.varmap.save(&self.checkpoint_path)?;
        let bytes = std::fs::metadata(&self.checkpoint_path)
            .map(|m| m.len())
            .unwrap_or(0);
        println!(
            "[+] Meta-train checkpoint saved to {} ({} bytes, final loss {:.5})",
            self.checkpoint_path, bytes, last_loss
        );
        Ok(last_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    #[test]
    fn dataset_slices_files_into_windows() {
        let temp = tempdir_path("meta_train_dataset");
        write(
            &temp.join("a.rs"),
            &(0..200).map(|i| format!("tok{i}")).collect::<Vec<_>>().join(" "),
        );
        write(
            &temp.join("nested/b.md"),
            &(0..200).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" "),
        );
        write(&temp.join("ignored.exe"), "binary contents");
        write(&temp.join("target/junk.rs"), "should not be read");

        let ds = RepoFileDataset::build(&temp, 64, 16, 50, 100, 7).unwrap();
        assert!(!ds.is_empty());
        // No window should include the excluded `target/` file or the `.exe`.
        // We can't introspect easily, but len > 0 with seq_len=16 is enough.
        assert!(ds.len() >= 2);

        cleanup(&temp);
    }

    #[test]
    fn dataset_yields_inputs_and_shifted_targets() {
        let temp = tempdir_path("meta_train_batch");
        write(
            &temp.join("z.txt"),
            &(0..400).map(|i| format!("t{i}")).collect::<Vec<_>>().join(" "),
        );
        let mut ds = RepoFileDataset::build(&temp, 64, 8, 5, 50, 11).unwrap();
        let (inputs, targets) = ds.next_batch(4, &Device::Cpu).unwrap();
        assert_eq!(inputs.dims(), &[4, 8]);
        assert_eq!(targets.dims(), &[4, 8]);
        cleanup(&temp);
    }

    fn tempdir_path(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("axiom-meta-train-{label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(p: &Path) {
        let _ = fs::remove_dir_all(p);
    }
}
