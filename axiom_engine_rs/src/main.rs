mod anthropic_forwarder;
mod claude_backend;
mod cluster;
mod config;
mod context_compressor;
mod contrastive;
mod corpus;
mod data_gen;
mod embedder;
mod encoder;
mod hardware;
mod inference;
mod jit_streamer;
mod kernel;
mod mcp_stdio;
mod memory_pool;
mod memory_recall;
mod memory_store;
mod meta_train;
mod metrics;
mod model;
mod model_meta;
mod openai_forwarder;
mod pairs;
mod quantization;
mod server;
mod skeleton;
mod train;
mod ttt_block;
mod vibe_memory;

use std::env;

use candle_core::{bail, Device, Result};
use config::{AxiomConfig, DEFAULT_CHECKPOINT_PATH};
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
    /// Compute device: "cpu", "cuda", or "metal".
    device: String,
}

fn usage() -> &'static str {
    "Usage:\n  cargo run --release -- --mode train [--epochs N] [--steps-per-epoch N] [--batch-size N] [--seq-len N] [--checkpoint PATH] [--device cpu|cuda|metal]\n  cargo run --release -- --mode generate \"your prompt\" [--max-new-tokens N] [--checkpoint PATH] [--tokenizer PATH] [--context-api-url URL] [--context-api-key KEY] [--max-context-tokens N] [--device cpu|cuda|metal]\n  cargo run --release -- --mode server [--host HOST] [--port PORT] [--checkpoint PATH] [--device cpu|cuda|metal]"
}

/// Resolve a `Device` from a string name.
///
/// CUDA and Metal support requires the crate to be compiled with the respective
/// feature flag (`--features cuda` or `--features metal`).
/// Objective 1.3: production model selection, SAFETY-GATED.
///
/// The live proxy only switches from the legacy 256-hash model to the scaled
/// BPE model when `AXIOM_PRODUCTION_BPE=1` AND the tokenizer artifact exists —
/// so it can never accidentally serve an unbaked model. On activation it sets
/// `AXIOM_TOKENIZER` (consumed by run_server / mcp_stdio) and returns the scaled
/// config (vocab read from the tokenizer) plus the BPE checkpoint path.
fn resolve_production_model(legacy: AxiomConfig, default_ckpt: &str) -> (AxiomConfig, String) {
    let enabled = std::env::var("AXIOM_PRODUCTION_BPE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !enabled {
        return (legacy, default_ckpt.to_string());
    }
    let bpe = std::env::var("AXIOM_TOKENIZER")
        .unwrap_or_else(|_| "checkpoints/axiom_bpe.json".to_string());
    let ckpt = std::env::var("AXIOM_BPE_CKPT")
        .unwrap_or_else(|_| "checkpoints/axiom_production_bpe.bin".to_string());
    if !std::path::Path::new(&bpe).exists() {
        eprintln!("[axiom] AXIOM_PRODUCTION_BPE=1 but tokenizer '{bpe}' missing — staying on legacy model");
        return (legacy, default_ckpt.to_string());
    }
    match tokenizers::Tokenizer::from_file(&bpe) {
        Ok(tok) => {
            let vocab = tok.get_vocab_size(true);
            std::env::set_var("AXIOM_TOKENIZER", &bpe);
            // Prefer dims from the checkpoint sidecar so any baked size loads;
            // fall back to the original 256/4 when no sidecar is present.
            let (d_model, n_layers, lr_inner, norm_eps) =
                match axiom_engine::model_meta::ModelMeta::load(&ckpt) {
                    Some(m) => (m.d_model, m.n_layers, m.lr_inner, m.norm_eps),
                    None => (256, 4, 1e-3, 1e-6),
                };
            eprintln!("[axiom] PRODUCTION MODEL = BPE (vocab {vocab}, d_model {d_model}, n_layers {n_layers}); checkpoint {ckpt}");
            let cfg = AxiomConfig {
                d_model,
                n_layers,
                vocab_size: vocab,
                lr_inner,
                norm_eps,
            };
            (cfg, ckpt)
        }
        Err(e) => {
            eprintln!("[axiom] failed to load tokenizer '{bpe}' ({e}) — staying on legacy model");
            (legacy, default_ckpt.to_string())
        }
    }
}

/// Resolve `--device auto` using the hardware co-tenancy guard rather than a
/// blind `cuda_if_available`. This is the crash fix: when a training job already
/// holds the GPU, the proxy is steered to CPU so the two cannot OOM each other.
/// Falls back to CPU when the `cuda` feature is absent or CUDA init fails.
fn resolve_auto_device() -> Device {
    let profile = hardware::detect();
    let rec = hardware::recommend(&profile);
    eprintln!("[axiom] auto-device: {} ({})", rec.proxy_device, rec.reason);
    match rec.proxy_device {
        hardware::DeviceChoice::Cuda => match try_new_cuda() {
            Some(dev) => dev,
            None => {
                eprintln!("[axiom] auto-device: CUDA unavailable at init — using CPU");
                Device::Cpu
            }
        },
        hardware::DeviceChoice::Cpu => Device::Cpu,
    }
}

#[cfg(feature = "cuda")]
fn try_new_cuda() -> Option<Device> {
    Device::new_cuda(0).ok()
}
#[cfg(not(feature = "cuda"))]
fn try_new_cuda() -> Option<Device> {
    None
}

fn device_from_str(s: &str) -> Result<Device> {
    match s {
        // Hardware-aware auto: honours the co-tenancy guard (see resolve_auto_device).
        "auto" => Ok(resolve_auto_device()),
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
        device,
    })
}

fn main() -> Result<()> {
    // candle's autograd backward pass recurses deeply. The default ~2 MB tokio
    // worker stack overflows the moment the /v1/messages compression path runs
    // the TTT model over heavy context — which silently killed every real
    // compression (the train_semantic binary already uses a 1 GiB stack for the
    // same model for exactly this reason). Build the runtime with a large worker
    // stack so compression is safe on real inputs.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(256 * 1024 * 1024) // 256 MB per worker
        .build()
        .map_err(|e| candle_core::Error::Msg(format!("tokio runtime build failed: {e}")))?;
    runtime.block_on(run_main())
}

async fn run_main() -> Result<()> {
    let args = parse_cli()?;

    // `doctor` reports the hardware profile + recommended config and exits. It
    // needs no model or device, so handle it before device resolution. This is
    // the open-source self-diagnosis entry point: a user runs it once to see
    // exactly what Axiom will do on their machine and why.
    if args.mode == "doctor" {
        let profile = hardware::detect();
        let rec = hardware::recommend(&profile);
        print!("{}", hardware::report(&profile, &rec));
        return Ok(());
    }

    let device = device_from_str(&args.device)?;

    // Keep local defaults small enough for CPU experimentation.
    let config = AxiomConfig {
        d_model: 64,
        n_layers: 2,
        vocab_size: 256,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
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
            let (cfg, ckpt) = resolve_production_model(config.clone(), &args.checkpoint_path);
            server::run_server(&args.host, args.port, cfg, &ckpt, device)
                .await
                .map_err(|e| candle_core::Error::Msg(format!("server startup failed: {e}")))?;
        }
        "mcp" => {
            // Native MCP server over JSON-RPC 2.0 stdio. Runs as a dedicated
            // process (separate from the HTTP proxy) so stdout stays a pure
            // protocol channel; all diagnostics go to stderr.
            let (cfg, ckpt) = resolve_production_model(config.clone(), &args.checkpoint_path);
            mcp_stdio::run_stdio_server(cfg, device, ckpt)
                .await
                .map_err(|e| candle_core::Error::Msg(format!("mcp server failed: {e}")))?;
        }
        "meta-train" => {
            // Phase 4: train projection matrices on raw repo files so
            // the online TTT updates produce well-conditioned, non-degenerate
            // hidden states that the context compressor can read out.
            let repo_root = std::env::current_dir().map_err(|e| {
                candle_core::Error::Msg(format!("could not resolve repo root: {e}"))
            })?;
            let max_files: usize = std::env::var("AXIOM_META_TRAIN_MAX_FILES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512);
            let max_sequences: usize = std::env::var("AXIOM_META_TRAIN_MAX_SEQS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096);
            let seed: u64 = std::env::var("AXIOM_META_TRAIN_SEED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(42);
            let lr: f64 = std::env::var("AXIOM_META_TRAIN_LR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1e-4);
            let mut trainer = meta_train::MetaTrainer::build(
                config,
                device,
                repo_root,
                args.checkpoint_path,
                args.batch_size,
                args.seq_len,
                max_files,
                max_sequences,
                seed,
            )?;
            println!(
                "[+] Meta-training dataset: {} windows from local repo files (seq_len={})",
                trainer.dataset_len(),
                args.seq_len
            );
            trainer.run(args.epochs, args.steps_per_epoch, lr)?;
        }
        other => {
            bail!(
                "unsupported mode '{other}'. Use --mode train | generate | server | mcp | meta-train | doctor"
            )
        }
    }

    Ok(())
}
