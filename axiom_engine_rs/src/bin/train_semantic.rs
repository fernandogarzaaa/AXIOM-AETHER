//! train_semantic — resumable, auto-sizing converged trainer for the scaled BPE
//! TTT model. Streams the on-disk corpus, splits train/val, auto-sizes the model
//! to free VRAM, trains with AdamW under early-stopping on held-out CE, and bakes
//! the BEST checkpoint plus a `*.meta.json` sidecar (dims/vocab/val_ce).
//!
//! RESUMABLE: loads an existing checkpoint and continues. Runs on a 1 GiB-stack
//! thread (candle backward recurses) in strict <=`win`-token detached windows.
//!
//! Build/run (separate binary — never touches the running proxy exe):
//!   cargo build --release --bin train_semantic
//!   AXIOM_EPOCHS=12 ./target/release/train_semantic
//!
//! Env: AXIOM_DMODEL / AXIOM_NLAYERS (override auto-size)
//!      AXIOM_VRAM_BUDGET_MB (override VRAM probe)
//!      AXIOM_LR(3e-3) AXIOM_INNER_LR(1e-3) AXIOM_EPOCHS(8) AXIOM_STEP_CAP(4000)
//!      AXIOM_TRAIN_WIN(128) AXIOM_MAX_TOKENS(2000000) AXIOM_PATIENCE(3)
//!      AXIOM_CORPUS_OUT(checkpoints/corpus)
//!      AXIOM_BPE(checkpoints/axiom_bpe.json)
//!      AXIOM_BPE_CKPT(checkpoints/axiom_production_bpe.bin)

use std::path::{Path, PathBuf};

use axiom_engine::config::AxiomConfig;
use axiom_engine::model::AxiomTTTLM;
use axiom_engine::model_meta::{default_ladder, pick_config, ModelMeta};
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

/// Free VRAM in bytes via nvidia-smi; None if unavailable (→ CPU budget).
fn free_vram_bytes() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let mb: u64 = s.lines().next()?.trim().parse().ok()?;
    Some(mb * 1024 * 1024)
}

/// Load all `*.txt` shard paths under the corpus dir (sorted).
fn corpus_shards(dir: &str) -> Vec<PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("txt"))
        .collect();
    v.sort();
    v
}

/// Held-out cross-entropy over `ids` in `win`-token chunks; no optimizer step.
fn val_ce(model: &AxiomTTTLM, dev: &Device, ids: &[u32], vocab: usize, win: usize) -> f32 {
    if ids.len() < 2 {
        return f32::INFINITY;
    }
    let mut total = 0.0f32;
    let mut n = 0usize;
    for w in ids.chunks(win) {
        if w.len() < 2 {
            continue;
        }
        let m = w.len();
        let mut states = model.init_states(dev).unwrap();
        let input = Tensor::from_vec(w[..m - 1].to_vec(), (1, m - 1), dev).unwrap();
        let logits = model.forward_lm(&input, &mut states).unwrap();
        let l2d = logits.squeeze(0).unwrap().reshape((m - 1, vocab)).unwrap();
        let tgt = Tensor::from_vec(w[1..].to_vec(), (m - 1,), dev).unwrap();
        total += candle_nn::loss::cross_entropy(&l2d, &tgt).unwrap().to_scalar::<f32>().unwrap()
            * (m - 1) as f32;
        n += m - 1;
    }
    if n == 0 {
        f32::INFINITY
    } else {
        total / n as f32
    }
}

fn main() {
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

fn run() {
    let repo = repo_root();
    let bpe = std::env::var("AXIOM_BPE")
        .unwrap_or_else(|_| repo.join("checkpoints/axiom_bpe.json").to_string_lossy().into());
    let ckpt = std::env::var("AXIOM_BPE_CKPT").unwrap_or_else(|_| {
        repo.join("checkpoints/axiom_production_bpe.bin").to_string_lossy().into()
    });
    let tok = Tokenizer::from_file(&bpe).expect("load BPE tokenizer");
    let vocab = tok.get_vocab_size(true);

    let lr = env_f64("AXIOM_LR", 3e-3);
    let inner_lr = env_f64("AXIOM_INNER_LR", 1e-3) as f32;
    let epochs = env_usize("AXIOM_EPOCHS", 8);
    let step_cap = env_usize("AXIOM_STEP_CAP", 4000);
    let win = env_usize("AXIOM_TRAIN_WIN", 128);
    let max_tokens = env_usize("AXIOM_MAX_TOKENS", 2_000_000);
    let patience = env_usize("AXIOM_PATIENCE", 3);
    // Stability controls (matter for deep/wide models like d512/8L that otherwise
    // diverge to NaN): global gradient-norm clip + linear LR warmup.
    let grad_clip = env_f64("AXIOM_GRAD_CLIP", 1.0);
    let warmup = env_usize("AXIOM_WARMUP_STEPS", 100);
    let log_every = env_usize("AXIOM_LOG_EVERY", 0);
    // Inner-loop stabilization (normalized keys + state clamp): required for
    // deep/wide models (d384, d512+) to stay finite. Recorded in the sidecar so
    // the proxy runs the checkpoint the same way it was trained.
    let stabilize = std::env::var("AXIOM_TTT_STABILIZE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    // --- Auto-size to VRAM (env override → nvidia-smi probe → CPU budget) ---
    let budget = std::env::var("AXIOM_VRAM_BUDGET_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024)
        // Only trust the VRAM probe when actually training on CUDA; on CPU use a
        // conservative RAM budget so we don't pick a GPU-sized model for the CPU.
        .or_else(|| {
            if device.is_cuda() {
                free_vram_bytes().map(|b| (b as f64 * 0.7) as u64) // 30% headroom
            } else {
                None
            }
        })
        .unwrap_or(2 * 1024 * 1024 * 1024); // CPU: conservative 2 GB
    let ladder = default_ladder();
    let rung = pick_config(budget, vocab, win, &ladder);
    let d_model = env_usize("AXIOM_DMODEL", rung.d_model);
    let n_layers = env_usize("AXIOM_NLAYERS", rung.n_layers);
    eprintln!(
        "[train] device={} auto-size budget={} MB → d_model={d_model} n_layers={n_layers} vocab={vocab}",
        if device.is_cuda() { "CUDA:0" } else { "CPU" },
        budget / (1024 * 1024)
    );

    let config = AxiomConfig {
        d_model,
        n_layers,
        vocab_size: vocab,
        lr_inner: inner_lr,
        norm_eps: 1e-6,
    };
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AxiomTTTLM::new(vb, config.clone()).expect("build model");
    model.set_stabilize(stabilize);
    eprintln!("[train] inner-loop stabilization: {stabilize}");

    // RESUME: continue from an existing checkpoint when dims match.
    if Path::new(&ckpt).exists() {
        match varmap.load(&ckpt) {
            Ok(()) => eprintln!("[train] resumed from {ckpt}"),
            Err(e) => eprintln!("[train] WARN: could not resume ({e}); starting fresh"),
        }
    } else {
        eprintln!("[train] no checkpoint at {ckpt}; starting fresh");
    }

    // --- Stream shards → tokens (capped); split 95/5 train/val ----------------
    let corpus_dir = std::env::var("AXIOM_CORPUS_OUT")
        .unwrap_or_else(|_| repo.join("checkpoints/corpus").to_string_lossy().into());
    let mut toks: Vec<u32> = Vec::new();
    let shards = corpus_shards(&corpus_dir);
    if shards.is_empty() {
        for p in std::fs::read_dir(repo.join("axiom_engine_rs/src")).into_iter().flatten().flatten() {
            let t = std::fs::read_to_string(p.path()).unwrap_or_default();
            toks.extend(tok.encode(t, false).map(|e| e.get_ids().to_vec()).unwrap_or_default());
            if toks.len() >= max_tokens {
                break;
            }
        }
    } else {
        for s in &shards {
            let t = std::fs::read_to_string(s).unwrap_or_default();
            toks.extend(tok.encode(t, false).map(|e| e.get_ids().to_vec()).unwrap_or_default());
            if toks.len() >= max_tokens {
                toks.truncate(max_tokens);
                break;
            }
        }
    }
    eprintln!("[train] corpus tokens={}", toks.len());
    assert!(toks.len() > win * 4, "corpus too small");
    let split = (toks.len() as f64 * 0.95) as usize;
    let (train_toks, val_toks) = toks.split_at(split);

    // --- Train: AdamW + early-stop on val CE; OOM-safe steps; bake best -------
    let windows: Vec<&[u32]> = train_toks.chunks(win).filter(|c| c.len() >= 2).collect();
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr, ..Default::default() }).unwrap();
    let t0 = std::time::Instant::now();
    let mut best_val = f32::INFINITY;
    let mut since_improve = 0usize;
    let mut step = 0usize;
    'train: for ep in 0..epochs {
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for w in &windows {
            let n = w.len();
            let mut states = model.init_states(&device).unwrap();
            let input = Tensor::from_vec(w[..n - 1].to_vec(), (1, n - 1), &device).unwrap();
            // OOM-resilient step: on a memory failure, skip this window gracefully.
            // Also NaN-guarded + grad-clipped + LR-warmed to keep deep models stable.
            let stepped = (|| -> candle_core::Result<f32> {
                // Linear LR warmup over the first `warmup` steps.
                if warmup > 0 && step < warmup {
                    opt.set_learning_rate(lr * (step + 1) as f64 / warmup as f64);
                } else if step == warmup {
                    opt.set_learning_rate(lr);
                }
                let logits = model.forward_lm(&input, &mut states)?;
                let l2d = logits.squeeze(0)?.reshape((n - 1, vocab))?;
                let tgt = Tensor::from_vec(w[1..].to_vec(), (n - 1,), &device)?;
                let loss = candle_nn::loss::cross_entropy(&l2d, &tgt)?;
                let lval = loss.to_scalar::<f32>()?;
                // Skip poisoned steps: never let a NaN/inf gradient touch the weights.
                if !lval.is_finite() {
                    return Ok(lval);
                }
                let mut grads = loss.backward()?;
                if grad_clip > 0.0 {
                    // Global L2 norm across all parameter grads, then scale if over budget.
                    let vars = varmap.all_vars();
                    let mut total = 0f64;
                    for v in &vars {
                        if let Some(g) = grads.get(v.as_tensor()) {
                            total += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                        }
                    }
                    let norm = total.sqrt();
                    if norm.is_finite() && norm > grad_clip {
                        let scale = grad_clip / (norm + 1e-6);
                        for v in &vars {
                            if let Some(g) = grads.get(v.as_tensor()) {
                                let clipped = (g * scale)?;
                                grads.insert(v.as_tensor(), clipped);
                            }
                        }
                    }
                }
                opt.step(&grads)?;
                Ok(lval)
            })();
            match stepped {
                Ok(l) if l.is_finite() => {
                    sum += l;
                    cnt += 1;
                }
                Ok(l) => eprintln!("[train]   WARN non-finite loss ({l}) at step {step} — skipped"),
                Err(e) => eprintln!("[train] step skipped (mem?): {e}"),
            }
            step += 1;
            // Step-level heartbeat so long GPU runs are observable before the
            // first per-epoch eval (set AXIOM_LOG_EVERY>0 to enable).
            if log_every > 0 && step % log_every == 0 {
                eprintln!(
                    "[train]   step {} loss~{:.4} ({:.0}s)",
                    step,
                    sum / cnt.max(1) as f32,
                    t0.elapsed().as_secs_f32()
                );
            }
            if step >= step_cap {
                eprintln!("[train] step cap {step_cap} hit");
                break 'train;
            }
        }
        let v = val_ce(&model, &device, val_toks, vocab, win);
        eprintln!(
            "[train] epoch {} train_loss={:.4} val_ce={:.4} (step {}, {:.0}s)",
            ep + 1,
            sum / cnt.max(1) as f32,
            v,
            step,
            t0.elapsed().as_secs_f32()
        );
        if v + 1e-3 < best_val {
            best_val = v;
            since_improve = 0;
            // Save the BEST checkpoint + sidecar.
            varmap.save(&ckpt).expect("save checkpoint");
            let _ = ModelMeta {
                d_model,
                n_layers,
                vocab_size: vocab,
                lr_inner: inner_lr,
                norm_eps: 1e-6,
                val_ce: best_val,
                tokenizer: bpe.clone(),
                stabilize,
            }
            .save(&ckpt);
        } else {
            since_improve += 1;
            if since_improve >= patience {
                eprintln!("[train] early stop (no val improvement for {patience} evals)");
                break;
            }
        }
    }
    eprintln!("[train] BEST val_ce={:.4} → {ckpt} (+ sidecar), {:.0}s", best_val, t0.elapsed().as_secs_f32());
    println!("{best_val:.4}");
}
