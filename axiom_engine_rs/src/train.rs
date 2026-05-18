use candle_core::{DType, Device, Result};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::AxiomConfig;
use crate::config::DEFAULT_CHECKPOINT_PATH;
use crate::data_gen::ProceduralDataset;
use crate::kernel::AxiomTTTEngine;

pub struct AxiomTrainer {
    config: AxiomConfig,
    device: Device,
    varmap: VarMap,
    engine: AxiomTTTEngine,
    dataset: ProceduralDataset,
    checkpoint_path: String,
    batch_size: usize,
    seq_len: usize,
}

impl AxiomTrainer {
    pub fn new(config: AxiomConfig, device: Device) -> Result<Self> {
        Self::with_settings(config, device, DEFAULT_CHECKPOINT_PATH, 8, 32)
    }

    pub fn with_settings(
        config: AxiomConfig,
        device: Device,
        checkpoint_path: impl Into<String>,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let engine = AxiomTTTEngine::new(vb, config.clone())?;
        let dataset = ProceduralDataset::new(config.vocab_size);

        Ok(Self {
            config,
            device,
            varmap,
            engine,
            dataset,
            checkpoint_path: checkpoint_path.into(),
            batch_size: batch_size.max(1),
            seq_len: seq_len.max(1),
        })
    }

    pub fn run_training_epochs(&mut self, epochs: usize, steps_per_epoch: usize) -> Result<()> {
        let mut optim = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr: 1e-4,
                ..ParamsAdamW::default()
            },
        )?;

        let total_steps = (epochs * steps_per_epoch) as u64;
        let progress = ProgressBar::new(total_steps);
        let style = ProgressStyle::with_template("{bar:40.cyan/blue} {pos:>6}/{len:6} | {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar());
        progress.set_style(style);

        for epoch in 0..epochs {
            for _step in 0..steps_per_epoch {
                let (inputs, targets) =
                    self.dataset
                        .generate_batch(self.batch_size, self.seq_len, &self.device)?;

                let (logits, _) = self.engine.forward(&inputs, None, false)?;
                let (b, t, v) = logits.dims3()?;
                let flat_logits = logits.reshape((b * t, v))?;
                let flat_targets = targets.reshape((b * t,))?;

                let loss = candle_nn::loss::cross_entropy(&flat_logits, &flat_targets)?;
                optim.backward_step(&loss)?;

                let loss_value = loss.to_scalar::<f32>()?;
                progress.set_message(format!(
                    "epoch {}/{} | loss {:.5} | d_model {} layers {}",
                    epoch + 1,
                    epochs,
                    loss_value,
                    self.config.d_model,
                    self.config.n_layers
                ));
                progress.inc(1);
            }
        }

        progress.finish_with_message("training complete");
        self.varmap.save(&self.checkpoint_path)?;
        println!("[+] Checkpoint saved to {}", self.checkpoint_path);

        Ok(())
    }
}
