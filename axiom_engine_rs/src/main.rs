mod config;
mod data_gen;
mod inference;
mod kernel;
mod train;
mod ttt_layer;

use std::env;

use candle_core::{bail, Device, Result};
use config::AxiomConfig;
use inference::InferencePipeline;
use train::AxiomTrainer;

#[derive(Debug)]
struct CliArgs {
    mode: String,
    prompt: Option<String>,
    epochs: usize,
    steps_per_epoch: usize,
    max_new_tokens: usize,
}

fn parse_cli() -> Result<CliArgs> {
    let argv: Vec<String> = env::args().collect();

    let mut mode = String::from("generate");
    let mut epochs: usize = 1;
    let mut steps_per_epoch: usize = 100;
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
            "--max-new-tokens" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --max-new-tokens");
                }
                max_new_tokens = argv[i].parse::<usize>().map_err(|_| {
                    candle_core::Error::Msg("invalid --max-new-tokens value".into())
                })?;
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
        epochs,
        steps_per_epoch,
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
            let mut trainer = AxiomTrainer::new(config, Device::Cpu)?;
            trainer.run_training_epochs(args.epochs, args.steps_per_epoch)?;
        }
        "generate" => {
            let prompt = args.prompt.ok_or_else(|| {
                candle_core::Error::Msg("missing prompt for --mode generate".into())
            })?;
            let pipeline = InferencePipeline::new(config, Device::Cpu)?;
            let output = pipeline.generate(&prompt, args.max_new_tokens)?;
            println!("{output}");
        }
        other => bail!("unsupported mode '{other}'. Use --mode train or --mode generate"),
    }

    Ok(())
}
