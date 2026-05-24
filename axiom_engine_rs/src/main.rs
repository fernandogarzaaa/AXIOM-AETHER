mod config;
mod data_gen;
mod inference;
mod jit_streamer;
mod kernel;
mod log_scan;
mod server;
mod train;
mod ttt_layer;

use std::env;

use candle_core::{bail, Device, Result};
use config::{AxiomConfig, DEFAULT_CHECKPOINT_PATH, DEFAULT_LOG_SCAN_AUTO_THRESHOLD};
use inference::{InferencePipeline, InferenceRuntimeOptions};
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
    tokenizer_path: Option<String>,
    context_api_url: Option<String>,
    context_api_key: Option<String>,
    max_context_tokens: usize,
    host: String,
    port: u16,
    use_log_scan: bool,
    /// Compute device: "cpu", "cuda", or "metal".
    device: String,
}

fn usage() -> &'static str {
    "Usage:\n  cargo run --release -- --mode train [--epochs N] [--steps-per-epoch N] [--batch-size N] [--seq-len N] [--checkpoint PATH] [--use-log-scan] [--device cpu|cuda|metal]\n  cargo run --release -- --mode generate \"your prompt\" [--max-new-tokens N] [--checkpoint PATH] [--tokenizer PATH] [--context-api-url URL] [--context-api-key KEY] [--max-context-tokens N] [--use-log-scan] [--device cpu|cuda|metal]\n  cargo run --release -- --mode server [--host HOST] [--port PORT] [--checkpoint PATH] [--use-log-scan] [--device cpu|cuda|metal]"
}

/// Resolve a `Device` from a string name.
///
/// CUDA and Metal support requires the crate to be compiled with the respective
/// feature flag (`--features cuda` or `--features metal`).
fn device_from_str(s: &str) -> Result<Device> {
    match s {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0),
        #[cfg(not(feature = "cuda"))]
        "cuda" => bail!(
            "CUDA device requested but the 'cuda' feature is not compiled in.\n\
             Rebuild with: cargo build --release --features cuda"
        ),
        #[cfg(feature = "metal")]
        "metal" => Device::new_metal(0),
        #[cfg(not(feature = "metal"))]
        "metal" => bail!(
            "Metal device requested but the 'metal' feature is not compiled in.\n\
             Rebuild with: cargo build --release --features metal"
        ),
        other => bail!("unsupported device '{other}'. Valid options: cpu, cuda, metal"),
    }
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
    let mut tokenizer_path: Option<String> = None;
    let mut context_api_url: Option<String> = env::var("AXIOM_CONTEXT_API_URL").ok();
    let mut context_api_key: Option<String> = env::var("AXIOM_CONTEXT_API_KEY").ok();
    let mut max_context_tokens: usize = env::var("AXIOM_MAX_CONTEXT_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256);
    let mut host = env::var("AXIOM_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let mut port = env::var("AXIOM_PORT")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(8080);
    let mut use_log_scan = env::var("AXIOM_USE_LOG_SCAN")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "True"))
        .unwrap_or(false);
    let mut device = env::var("AXIOM_DEVICE").unwrap_or_else(|_| "cpu".to_string());
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
            "--tokenizer" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --tokenizer");
                }
                tokenizer_path = Some(argv[i].clone());
            }
            "--context-api-url" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --context-api-url");
                }
                context_api_url = Some(argv[i].clone());
            }
            "--context-api-key" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --context-api-key");
                }
                context_api_key = Some(argv[i].clone());
            }
            "--max-context-tokens" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --max-context-tokens");
                }
                max_context_tokens = argv[i].parse::<usize>().map_err(|_| {
                    candle_core::Error::Msg("invalid --max-context-tokens value".into())
                })?;
            }
            "--host" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --host");
                }
                host = argv[i].clone();
            }
            "--port" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --port");
                }
                port = argv[i]
                    .parse::<u16>()
                    .map_err(|_| candle_core::Error::Msg("invalid --port value".into()))?;
            }
            "--use-log-scan" => {
                use_log_scan = true;
                i += 1;
                continue;
            }
            "--device" => {
                i += 1;
                if i >= argv.len() {
                    bail!("missing value for --device");
                }
                device = argv[i].clone();
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

    if max_context_tokens > DEFAULT_LOG_SCAN_AUTO_THRESHOLD {
        use_log_scan = true;
    }

    Ok(CliArgs {
        mode,
        prompt,
        checkpoint_path,
        epochs,
        steps_per_epoch,
        batch_size,
        seq_len,
        max_new_tokens,
        tokenizer_path,
        context_api_url,
        context_api_key,
        max_context_tokens,
        host,
        port,
        use_log_scan,
        device,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_cli()?;
    let device = device_from_str(&args.device)?;

    // Keep local defaults small enough for CPU experimentation.
    let config = AxiomConfig {
        d_model: 64,
        n_layers: 2,
        num_heads: 4,
        head_dim: 16,
        vocab_size: 256,
        lr_inner: 1e-3,
        rms_norm_eps: 1e-6,
        use_log_scan: args.use_log_scan,
        log_scan_auto_threshold: DEFAULT_LOG_SCAN_AUTO_THRESHOLD,
    };

    match args.mode.as_str() {
        "train" => {
            let mut trainer = if args.checkpoint_path == DEFAULT_CHECKPOINT_PATH
                && args.batch_size == 8
                && args.seq_len == 32
            {
                AxiomTrainer::new(config, device)?
            } else {
                AxiomTrainer::with_settings(
                    config,
                    device,
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
            let runtime = InferenceRuntimeOptions {
                tokenizer_path: args.tokenizer_path,
                context_api_url: args.context_api_url,
                context_api_key: args.context_api_key,
                max_context_tokens: args.max_context_tokens,
            };
            let pipeline = InferencePipeline::with_checkpoint_and_options(
                config,
                device,
                args.checkpoint_path,
                runtime,
            )?;
            let output = pipeline.generate(&prompt, args.max_new_tokens)?;
            println!("{output}");
        }
        "server" => {
            server::run_server(&args.host, args.port, config, &args.checkpoint_path, device)
                .await
                .map_err(|e| candle_core::Error::Msg(format!("server startup failed: {e}")))?;
        }
        other => {
            bail!("unsupported mode '{other}'. Use --mode train, --mode generate, or --mode server")
        }
    }

    Ok(())
}
