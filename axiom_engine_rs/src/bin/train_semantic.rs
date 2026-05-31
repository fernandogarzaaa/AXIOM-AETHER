//! train_semantic — resumable converged trainer for the scaled BPE TTT model.
//!
//! Trains AxiomTTTLM (d_model=256, n_layers=4, BPE vocab) on the clean repo
//! corpus and bakes a production checkpoint. RESUMABLE: loads the existing
//! checkpoint if present and continues, so convergence can be accumulated over
//! several bounded passes (CPU TTT training is ~seconds/step; a single
//! to-convergence run would peg this volatile box for hours).
//!
//! Build/run (separate binary — never touches the running proxy exe):
//!   cargo build --release --bin train_semantic
//!   AXIOM_EPOCHS=8 AXIOM_MAX_TOKENS=12000 ./target/release/train_semantic
//!
//! Env: AXIOM_DMODEL(256) AXIOM_NLAYERS(4) AXIOM_LR(3e-3) AXIOM_INNER_LR(1e-3)
//!      AXIOM_TRAIN_WIN(128) AXIOM_EPOCHS(8) AXIOM_STEP_CAP(900)
//!      AXIOM_MAX_TOKENS(12000)  cap corpus for faster convergence on CPU
//!      AXIOM_BPE(checkpoints/axiom_bpe.json)
//!      AXIOM_BPE_CKPT(checkpoints/axiom_production_bpe.bin)

use std::path::{Path, PathBuf};

use axiom_engine::config::AxiomConfig;
use axiom_engine::model::AxiomTTTLM;
use candle_core::{DType, Device, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use tokenizers::Tokenizer;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn main() {
    // Large stack: candle's backward() recurses through the op graph and a deep
    // training window overflows the default Windows stack.
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

fn run() {
    let root = repo_root();
    let bpe = std::env::var("AXIOM_BPE")
        .unwrap_or_else(|_| root.join("checkpoints/axiom_bpe.json").to_string_lossy().into());
    let ckpt = std::env::var("AXIOM_BPE_CKPT").unwrap_or_else(|_| {
        root.join("checkpoints/axiom_production_bpe.bin").to_string_lossy().into()
    });
    let tok = Tokenizer::from_file(&bpe).expect("load BPE tokenizer");
    let vocab = tok.get_vocab_size(true);

    let d_model = env_usize("AXIOM_DMODEL", 256);
    let n_layers = env_usize("AXIOM_NLAYERS", 4);
    let lr = env_f64("AXIOM_LR", 3e-3);
    let inner_lr = env_f64("AXIOM_INNER_LR", 1e-3) as f32;
    let epochs = env_usize("AXIOM_EPOCHS", 8);
    let step_cap = env_usize("AXIOM_STEP_CAP", 900);
    let train_win = env_usize("AXIOM_TRAIN_WIN", 128);
    let max_tokens = env_usize("AXIOM_MAX_TOKENS", 12000);

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    eprintln!(
        "[train] device={} d_model={d_model} n_layers={n_layers} vocab={vocab} lr={lr} epochs={epochs} win={train_win} max_tokens={max_tokens}",
        if device.is_cuda() { "CUDA:0" } else { "CPU" }
    );

    let config = AxiomConfig { d_model, n_layers, vocab_size: vocab, lr_inner: inner_lr, norm_eps: 1e-6 };
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AxiomTTTLM::new(vb, config.clone()).expect("build model");

    // RESUME: load prior weights if a checkpoint exists.
    if Path::new(&ckpt).exists() {
        match varmap.load(&ckpt) {
            Ok(()) => eprintln!("[train] resumed from {ckpt}"),
            Err(e) => eprintln!("[train] WARN: could not resume ({e}); starting fresh"),
        }
    } else {
        eprintln!("[train] no checkpoint at {ckpt}; starting fresh");
    }

    // Corpus: clean src/*.rs, concatenated, capped for tractable CPU convergence.
    let src = root.join("axiom_engine_rs/src");
    let mut ents: Vec<PathBuf> = std::fs::read_dir(&src).unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("rs"))
        .collect();
    ents.sort();
    let mut toks: Vec<u32> = Vec::new();
    for p in &ents {
        let t = std::fs::read_to_string(p).unwrap_or_default();
        toks.extend(tok.encode(t, false).map(|e| e.get_ids().to_vec()).unwrap_or_default());
        if toks.len() >= max_tokens { toks.truncate(max_tokens); break; }
    }
    eprintln!("[train] corpus tokens={}", toks.len());
    assert!(toks.len() > train_win);

    let windows: Vec<&[u32]> = toks.chunks(train_win).filter(|c| c.len() >= 2).collect();
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr, ..Default::default() }).unwrap();
    let t0 = std::time::Instant::now();
    let mut step = 0usize;
    let mut last_epoch_loss = f32::INFINITY;
    'train: for ep in 0..epochs {
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for w in &windows {
            let n = w.len();
            let mut states = model.init_states(&device).unwrap();
            let input = Tensor::from_vec(w[..n - 1].to_vec(), (1, n - 1), &device).unwrap();
            let logits = model.forward_lm(&input, &mut states).unwrap();
            let l2d = logits.squeeze(0).unwrap().reshape((n - 1, vocab)).unwrap();
            let tgt = Tensor::from_vec(w[1..].to_vec(), (n - 1,), &device).unwrap();
            let loss = candle_nn::loss::cross_entropy(&l2d, &tgt).unwrap();
            opt.backward_step(&loss).unwrap();
            sum += loss.to_scalar::<f32>().unwrap();
            cnt += 1;
            step += 1;
            if step >= step_cap {
                eprintln!("[train] step cap {step_cap} hit");
                last_epoch_loss = sum / cnt.max(1) as f32;
                break 'train;
            }
        }
        last_epoch_loss = sum / cnt.max(1) as f32;
        eprintln!("[train] epoch {} avg_loss={:.4} (step {}, {:.0}s)", ep + 1, last_epoch_loss, step, t0.elapsed().as_secs_f32());
    }

    if let Some(parent) = Path::new(&ckpt).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    varmap.save(&ckpt).expect("save checkpoint");
    eprintln!("[train] BAKED checkpoint -> {ckpt} ({} steps total this pass, final_avg_loss={:.4})", step, last_epoch_loss);
    println!("{last_epoch_loss:.4}");
}
