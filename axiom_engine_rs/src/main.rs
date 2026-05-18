mod config;
mod data_gen;
mod inference;
mod kernel;
mod train;
mod ttt_layer;

use std::env;

use candle_core::{bail, Device, Result};
use config::{AxiomConfig, DEFAULT_CHECKPOINT_PATH};
use inference::InferencePipeline;
use train::AxiomTrainer;

#[derive(Debug)]
struct CliArgs {
    mode: String,
    prompt: Option<String>,
    checkpoint_path: String,
    epochs: usize,
    steps_per_epoch: usize,
    batch_size: usize,
    seq_len: usize,
    max_new_tokens: usize,
}

fn usage() -> &'static str {
    "Usage:\n  cargo run --release -- --mode train [--epochs N] [--steps-per-epoch N] [--batch-size N] [--seq-len N] [--checkpoint PATH]\n  cargo run --release -- --mode generate \"your prompt\" [--max-new-tokens N] [--checkpoint PATH]"
}

fn parse_cli() -> Result<CliArgs> {
    let argv: Vec<String> = env::args().collect();
    if argv.len() == 1 || argv.iter().any(|arg| arg == "--help" || arg == "-h") {
        bail!("{}", usage());
    }

    let mut mode = String::from("generate");
    let mut checkpoint_path = DEFAULT_CHECKPOINT_PATH.to_string();
    let mut epochs: usize = 1;
    let mut steps_per_epoch: usize = 100;
    let mut batch_size: usize = 8;
    let mut seq_len: usize = 32;
    let mut max_new_tokens: usize = 32;
    let mut prompt_parts: Vec<String> = Vec::new();

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--mode" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --mode");
                }
                mode = argv[i].clone();
            }
            "--epochs" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --epochs");
                }
                epochs = argv[i]
                    .parse::<usize>()
                    .map_err(|_| candle_core::Error::Msg("invalid --epochs value".into()))?;
            }
            "--steps-per-epoch" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --steps-per-epoch");
                }
                steps_per_epoch = argv[i].parse::<usize>().map_err(|_| {
                    candle_core::Error::Msg("invalid --steps-per-epoch value".into())
                })?;
            }
            "--batch-size" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --batch-size");
                }
                batch_size = argv[i]
                    .parse::<usize>()
                    .map_err(|_| candle_core::Error::Msg("invalid --batch-size value".into()))?;
            }
            "--seq-len" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --seq-len");
                }
                seq_len = argv[i]
                    .parse::<usize>()
                    .map_err(|_| candle_core::Error::Msg("invalid --seq-len value".into()))?;
            }
            "--max-new-tokens" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --max-new-tokens");
                }
                max_new_tokens = argv[i].parse::<usize>().map_err(|_| {
                    candle_core::Error::Msg("invalid --max-new-tokens value".into())
                })?;
            }
            "--checkpoint" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --checkpoint");
                }
                checkpoint_path = argv[i].clone();
            }
            value => prompt_parts.push(value.to_string()),
        }
        i += 1;
    }

    let prompt = if prompt_parts.is_empty() {
        None
    } else {
        Some(prompt_parts.join(" "))
    };

    Ok(CliArgs {
        mode,
        prompt,
        checkpoint_path,
        epochs,
        steps_per_epoch,
        batch_size,
        seq_len,
        max_new_tokens,
    })
}

fn main() -> Result<()> {
    let args = parse_cli()?;

    // Keep local defaults small enough for CPU experimentation.
    let config = AxiomConfig {
        d_model: 64,
        n_layers: 2,
        num_heads: 4,
        head_dim: 16,
        vocab_size: 256,
        lr_inner: 1e-3,
        rms_norm_eps: 1e-6,
    };

    match args.mode.as_str() {
        "train" => {
            let mut trainer = if args.checkpoint_path == DEFAULT_CHECKPOINT_PATH
                && args.batch_size == 8
                && args.seq_len == 32
            {
                AxiomTrainer::new(config, Device::Cpu)?
            } else {
                AxiomTrainer::with_settings(
                    config,
                    Device::Cpu,
                    args.checkpoint_path,
                    args.batch_size,
                    args.seq_len,
                )?
            };
            trainer.run_training_epochs(args.epochs, args.steps_per_epoch)?;
        }
        "generate" => {
            let prompt = args.prompt.ok_or_else(|| {
                candle_core::Error::Msg("missing prompt for --mode generate".into())
            })?;
            let pipeline = if args.checkpoint_path == DEFAULT_CHECKPOINT_PATH {
                InferencePipeline::new(config, Device::Cpu)?
            } else {
                InferencePipeline::with_checkpoint(config, Device::Cpu, args.checkpoint_path)?
            };
            let output = pipeline.generate(&prompt, args.max_new_tokens)?;
            println!("{output}");
        }
        other => bail!("unsupported mode '{other}'. Use --mode train or --mode generate"),
    }

    Ok(())
}
