use candle_core::{DType, Device, Result};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::AxiomConfig;
use crate::config::DEFAULT_CHECKPOINT_PATH;
use crate::data_gen::ProceduralDataset;
use crate::model::AxiomTTTLM;

pub struct AxiomTrainer {
    config: AxiomConfig,
    device: Device,
    varmap: VarMap,
    model: AxiomTTTLM,
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
        let model = AxiomTTTLM::new(vb, config.clone())?;
        let dataset = ProceduralDataset::new(config.vocab_size);

        Ok(Self {
            config,
            device,
            varmap,
            model,
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

        // Snapshot the outer-loop projection matrices (W_q / W_k / W_v) so we
        // can read their gradients out of the GradStore each step. The Var's
        // backing tensor id is stable across steps, so a one-time snapshot is
        // enough to look the gradient up later.
        let projection_groups = self.snapshot_projection_vars();

        let total_steps = (epochs * steps_per_epoch) as u64;
        let progress = ProgressBar::new(total_steps);
        let style = ProgressStyle::with_template("{bar:40.cyan/blue} {pos:>6}/{len:6} | {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar());
        progress.set_style(style);

        for epoch in 0..epochs {
            let mut epoch_loss_sum = 0.0f32;
            // Accumulated per-step gradient variance for each projection group.
            let mut grad_var_sums = [0.0f32; 3]; // [W_q, W_k, W_v]

            for _step in 0..steps_per_epoch {
                let (inputs, targets) =
                    self.dataset
                        .generate_batch(self.batch_size, self.seq_len, &self.device)?;

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

                let logit_refs: Vec<&candle_core::Tensor> = batch_logits.iter().collect();
                let target_refs: Vec<&candle_core::Tensor> = batch_targets.iter().collect();
                let logits = candle_core::Tensor::cat(&logit_refs, 0)?;
                let targets = candle_core::Tensor::cat(&target_refs, 0)?;
                let (b, t, v) = logits.dims3()?;
                let flat_logits = logits.reshape((b * t, v))?;
                let flat_targets = targets.reshape((b * t,))?;

                let loss = candle_nn::loss::cross_entropy(&flat_logits, &flat_targets)?;

                // Explicit backward so we can inspect projection-matrix
                // gradients before the optimiser consumes them.
                let grads = loss.backward()?;
                for (group_idx, group) in projection_groups.iter().enumerate() {
                    grad_var_sums[group_idx] += projection_grad_variance(&grads, group)?;
                }
                optim.step(&grads)?;

                let loss_value = loss.to_scalar::<f32>()?;
                epoch_loss_sum += loss_value;
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

            // Phase-4 telemetry: per-epoch meta-loss decay + the delta variance
            // of the three projection-matrix gradient groups.
            let steps_f = steps_per_epoch.max(1) as f32;
            progress.suspend(|| {
                println!(
                    "[meta] epoch {}/{} | L_meta={:.5} | grad-var ∇W_q={:.3e} ∇W_k={:.3e} ∇W_v={:.3e}",
                    epoch + 1,
                    epochs,
                    epoch_loss_sum / steps_f,
                    grad_var_sums[0] / steps_f,
                    grad_var_sums[1] / steps_f,
                    grad_var_sums[2] / steps_f,
                );
            });
        }

        progress.finish_with_message("training complete");
        self.save_checkpoint_verified()?;

        Ok(())
    }

    /// Group the projection-matrix variables by role (W_q / W_k / W_v) so we
    /// can aggregate their gradients across every TTT layer.
    fn snapshot_projection_vars(&self) -> [Vec<candle_core::Tensor>; 3] {
        let mut groups: [Vec<candle_core::Tensor>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let data = self
            .varmap
            .data()
            .lock()
            .expect("varmap mutex must not be poisoned");
        for (name, var) in data.iter() {
            let tensor = var.as_tensor().clone();
            if name.contains("w_q") {
                groups[0].push(tensor);
            } else if name.contains("w_k") {
                groups[1].push(tensor);
            } else if name.contains("w_v") {
                groups[2].push(tensor);
            }
        }
        groups
    }

    /// Persist the checkpoint and verify the bytes actually landed on disk.
    fn save_checkpoint_verified(&self) -> Result<()> {
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
        match std::fs::metadata(&self.checkpoint_path) {
            Ok(meta) => println!(
                "[+] Checkpoint written and verified: {} ({} bytes on disk)",
                self.checkpoint_path,
                meta.len()
            ),
            Err(e) => {
                return Err(candle_core::Error::Msg(format!(
                    "checkpoint save reported success but file is missing at {}: {e}",
                    self.checkpoint_path
                )))
            }
        }
        Ok(())
    }
}

/// Variance of the gradient values aggregated across one projection group
/// (e.g. every layer's `W_q`). Returns 0.0 when no gradient is present —
/// which itself is a useful "dead gradient" signal in the telemetry.
fn projection_grad_variance(
    grads: &candle_core::backprop::GradStore,
    group: &[candle_core::Tensor],
) -> Result<f32> {
    let mut values: Vec<f32> = Vec::new();
    for tensor in group {
        if let Some(grad) = grads.get(tensor) {
            values.extend(grad.flatten_all()?.to_vec1::<f32>()?);
        }
    }
    if values.is_empty() {
        return Ok(0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    Ok(var)
}
