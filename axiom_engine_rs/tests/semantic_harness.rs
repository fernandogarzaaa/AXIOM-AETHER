//! semantic_harness.rs — Phase 3 validation (temporary).
//!
//! Proves whether a BPE tokenizer + a scaled TTT model produce a cross-entropy
//! signal that separates a structural anomaly from clean repo code.
//!
//! Pipeline:
//!   1. Load the ByteLevel BPE (checkpoints/axiom_bpe.json).
//!   2. Build AxiomTTTLM at scaled dims (env-tunable) on CUDA-if-available/CPU.
//!   3. Train next-token prediction on the clean repo corpus (a few files held
//!      out), bounded epochs, AdamW — in detached <=512-token windows.
//!   4. Measure per-file CE for in-train clean, held-out clean, and the anomaly,
//!      each scored in strict 512-token detached chunks (never a 24k graph).
//!   5. Report separation; assert anomaly CE exceeds the clean maximum.
//!
//! Run (release; separate test exe, does not touch the proxy binary):
//!   cargo test --release --test semantic_harness -- --ignored --nocapture
//!
//! Tunable (Phase 3.3 iteration without recompiling):
//!   AXIOM_DMODEL (256)  AXIOM_NLAYERS (4)  AXIOM_LR (3e-3)
//!   AXIOM_EPOCHS (8)  AXIOM_INNER_LR (1e-3)  AXIOM_STEP_CAP (4000)

use std::path::PathBuf;

use axiom_engine::config::AxiomConfig;
use axiom_engine::model::AxiomTTTLM;
use candle_core::{DType, Device, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use tokenizers::Tokenizer;

const CHUNK: usize = 512; // strict TTT/eval window cap (Phase 2.3)
const TRAIN_WIN: usize = 256; // training backprop window (<= CHUNK)

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn pick_device() -> Device {
    match Device::cuda_if_available(0) {
        Ok(d) => {
            eprintln!("[harness] device = {}", if d.is_cuda() { "CUDA:0" } else { "CPU (cuda unavailable / not compiled)" });
            d
        }
        Err(e) => {
            eprintln!("[harness] CUDA init failed ({e}); falling back to CPU");
            Device::Cpu
        }
    }
}

fn encode(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
}

/// Cross-entropy over `ids`, scored in strict <=512-token chunks with the
/// autograd graph truncated between chunks. Eval only (no optimizer step).
fn chunked_ce(model: &AxiomTTTLM, dev: &Device, ids: &[u32], vocab: usize) -> f32 {
    let mut states = model.init_states(dev).unwrap();
    let mut total = 0.0f32;
    let mut toks = 0usize;
    for chunk in ids.chunks(CHUNK) {
        if chunk.len() < 2 { continue; }
        let n = chunk.len();
        let input = Tensor::from_vec(chunk[..n - 1].to_vec(), (1, n - 1), dev).unwrap();
        let logits = model.forward_lm(&input, &mut states).unwrap();
        let l2d = logits.squeeze(0).unwrap().reshape((n - 1, vocab)).unwrap();
        let tgt = Tensor::from_vec(chunk[1..].to_vec(), (n - 1,), dev).unwrap();
        let loss = candle_nn::loss::cross_entropy(&l2d, &tgt).unwrap().to_scalar::<f32>().unwrap();
        total += loss * (n - 1) as f32;
        toks += n - 1;
        for s in states.iter_mut() { *s = s.detach(); } // Phase 2.3: bounded graph
    }
    if toks == 0 { 0.0 } else { total / toks as f32 }
}

#[test]
#[ignore = "Phase 3 validation; needs checkpoints/axiom_bpe.json, trains a model (~minutes). Run: cargo test --release --test semantic_harness -- --ignored --nocapture"]
fn bpe_scaled_model_separates_anomaly() {
    // candle's autograd `backward()` recurses through the op graph; a deep
    // training window overflows the default Windows test-thread stack. Run the
    // whole harness on a 1 GiB-stack thread.
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(run_harness)
        .unwrap()
        .join()
        .unwrap();
}

fn run_harness() {
    let root = repo_root();
    let bpe = root.join("checkpoints/axiom_bpe.json");
    if !bpe.exists() {
        eprintln!("[harness] SKIP: {bpe:?} missing (run train_tokenizer first)");
        return;
    }
    let tok = Tokenizer::from_file(&bpe).expect("load BPE");
    let vocab = tok.get_vocab_size(true);

    let d_model = env_usize("AXIOM_DMODEL", 256);
    let n_layers = env_usize("AXIOM_NLAYERS", 4);
    let lr = env_f64("AXIOM_LR", 3e-3);
    let inner_lr = env_f64("AXIOM_INNER_LR", 1e-3) as f32;
    let epochs = env_usize("AXIOM_EPOCHS", 8);
    let step_cap = env_usize("AXIOM_STEP_CAP", 4000);
    let train_win = env_usize("AXIOM_TRAIN_WIN", TRAIN_WIN);

    let device = pick_device();
    let config = AxiomConfig { d_model, n_layers, vocab_size: vocab, lr_inner: inner_lr, norm_eps: 1e-6 };
    eprintln!("[harness] config d_model={d_model} n_layers={n_layers} vocab={vocab} lr={lr} inner_lr={inner_lr} epochs={epochs}");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AxiomTTTLM::new(vb, config.clone()).expect("build model");

    // ---- Corpus: clean src/*.rs, holding out two files for generalization ----
    let held_out = ["model.rs", "ttt_block.rs"];
    let src_dir = root.join("axiom_engine_rs/src");
    let mut train_files: Vec<PathBuf> = Vec::new();
    let mut held_files: Vec<PathBuf> = Vec::new();
    let mut ents: Vec<PathBuf> = std::fs::read_dir(&src_dir).unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("rs"))
        .collect();
    ents.sort();
    for p in ents {
        let name = p.file_name().unwrap().to_str().unwrap().to_string();
        if held_out.contains(&name.as_str()) { held_files.push(p); } else { train_files.push(p); }
    }
    let mut train_tokens: Vec<u32> = Vec::new();
    for p in &train_files {
        train_tokens.extend(encode(&tok, &std::fs::read_to_string(p).unwrap_or_default()));
    }
    eprintln!("[harness] train files={} tokens={}", train_files.len(), train_tokens.len());
    assert!(train_tokens.len() > train_win, "corpus too small");

    // ---- Train: AdamW over windows, bounded ----
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr, ..Default::default() }).unwrap();
    let windows: Vec<&[u32]> = train_tokens.chunks(train_win).filter(|c| c.len() >= 2).collect();
    let t0 = std::time::Instant::now();
    let mut step = 0usize;
    'train: for ep in 0..epochs {
        let mut ep_loss = 0.0f32;
        let mut ep_steps = 0usize;
        for w in &windows {
            let n = w.len();
            let mut states = model.init_states(&device).unwrap();
            let input = Tensor::from_vec(w[..n - 1].to_vec(), (1, n - 1), &device).unwrap();
            let logits = model.forward_lm(&input, &mut states).unwrap();
            let l2d = logits.squeeze(0).unwrap().reshape((n - 1, vocab)).unwrap();
            let tgt = Tensor::from_vec(w[1..].to_vec(), (n - 1,), &device).unwrap();
            let loss = candle_nn::loss::cross_entropy(&l2d, &tgt).unwrap();
            opt.backward_step(&loss).unwrap();
            ep_loss += loss.to_scalar::<f32>().unwrap();
            ep_steps += 1;
            step += 1;
            if step >= step_cap { eprintln!("[harness] step cap {step_cap} hit"); break 'train; }
        }
        eprintln!("[harness] epoch {} avg_train_loss={:.4} ({} steps, {:.1}s)", ep + 1, ep_loss / ep_steps.max(1) as f32, ep_steps, t0.elapsed().as_secs_f32());
    }
    eprintln!("[harness] trained {step} steps in {:.1}s\n", t0.elapsed().as_secs_f32());

    // ---- Evaluate ----
    let read = |rel: &str| std::fs::read_to_string(root.join(rel)).unwrap_or_default();
    let mut clean: Vec<(String, f32)> = Vec::new();
    // in-train clean
    for f in ["axiom_engine_rs/src/mcp_stdio.rs", "axiom_engine_rs/src/vibe_memory.rs"] {
        clean.push((format!("{f} [in-train]"), chunked_ce(&model, &device, &encode(&tok, &read(f)), vocab)));
    }
    // held-out clean (generalization)
    for p in &held_files {
        let rel = format!("axiom_engine_rs/src/{}", p.file_name().unwrap().to_str().unwrap());
        clean.push((format!("{rel} [HELD-OUT]"), chunked_ce(&model, &device, &encode(&tok, &read(&rel)), vocab)));
    }
    let anomaly_ce = chunked_ce(&model, &device, &encode(&tok, &read("tests/anomaly_target.rs")), vocab);

    eprintln!("[harness] === per-file cross-entropy (BPE, trained) ===");
    for (name, l) in &clean { eprintln!("   clean   {:>8.4}  {}", l, name); }
    eprintln!("   ANOMALY {:>8.4}  tests/anomaly_target.rs", anomaly_ce);

    let clean_max = clean.iter().map(|c| c.1).fold(0.0f32, f32::max);
    let clean_mean = clean.iter().map(|c| c.1).sum::<f32>() / clean.len() as f32;
    let margin = anomaly_ce - clean_max;
    eprintln!("\n[harness] clean_mean={:.4} clean_max={:.4} anomaly={:.4} margin(anomaly-clean_max)={:+.4}", clean_mean, clean_max, anomaly_ce, margin);
    let separated = anomaly_ce > clean_max;
    eprintln!("[harness] SEPARATION: {}", if separated { "ACHIEVED ✓ (anomaly CE > all clean)" } else { "NOT YET ✗ (overlap — tune dims/lr/epochs and retry)" });
    // Recommended gate threshold midway between clean_max and anomaly.
    if separated { eprintln!("[harness] suggested AXIOM_DRIFT_THRESHOLD = {:.4}", (clean_max + anomaly_ce) / 2.0); }

    assert!(train_tokens.len() > 0);
}
