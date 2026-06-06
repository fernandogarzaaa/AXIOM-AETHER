//! train_embedder — contrastive trainer for the BiEncoder. Loads pairs.jsonl,
//! splits train/val, trains with symmetric InfoNCE + in-batch negatives under
//! AdamW (grad-clip + warmup), saves the BEST checkpoint (by val recall@1) to
//! axiom_embedder.bin + sidecar. GPU if available (move the proxy to CPU first).
//!
//!   AXIOM_EMB_EPOCHS=10 AXIOM_EMB_BS=32 cargo run --release --bin train_embedder
//!
//! Env: AXIOM_EMB_D(384) AXIOM_EMB_LAYERS(6) AXIOM_EMB_HEADS(6) AXIOM_EMB_FFN(1536)
//!      AXIOM_EMB_SEQ(256) AXIOM_EMB_LR(3e-4) AXIOM_EMB_TAU(0.05) AXIOM_EMB_BS(32)
//!      AXIOM_EMB_EPOCHS(10) AXIOM_EMB_GRAD_CLIP(1.0) AXIOM_EMB_WARMUP(100)
//!      AXIOM_PAIRS_OUT(checkpoints/pairs.jsonl)
//!      AXIOM_EMB_CKPT(checkpoints/axiom_embedder.bin)
//!      AXIOM_BPE(checkpoints/axiom_bpe.json)

use axiom_engine::contrastive::{batch_recall_at_1, info_nce};
use axiom_engine::encoder::{BiEncoder, EmbedderMeta, EncoderConfig};
use axiom_engine::pairs::read_pairs_jsonl;
use candle_core::{DType, Device, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use tokenizers::Tokenizer;

fn envu(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn envf(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn repo() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn main() {
    std::thread::Builder::new().stack_size(1024 * 1024 * 1024).spawn(run).unwrap().join().unwrap();
}

fn run() {
    let bpe = std::env::var("AXIOM_BPE")
        .unwrap_or_else(|_| repo().join("checkpoints/axiom_bpe.json").to_string_lossy().into());
    let ckpt = std::env::var("AXIOM_EMB_CKPT")
        .unwrap_or_else(|_| repo().join("checkpoints/axiom_embedder.bin").to_string_lossy().into());
    let pairs_path = std::env::var("AXIOM_PAIRS_OUT")
        .unwrap_or_else(|_| repo().join("checkpoints/pairs.jsonl").to_string_lossy().into());
    let tok = Tokenizer::from_file(&bpe).expect("tokenizer");
    let vocab = tok.get_vocab_size(true);
    // AXIOM_EMB_DEVICE=cpu forces CPU (avoids the GPU entirely — no contention
    // with any other GPU job); default "auto" = CUDA if available.
    let device = match std::env::var("AXIOM_EMB_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        _ => Device::cuda_if_available(0).unwrap_or(Device::Cpu),
    };
    eprintln!("[emb] device={}", if device.is_cuda() { "CUDA:0" } else { "CPU" });

    let cfg = EncoderConfig {
        vocab_size: vocab,
        d_model: envu("AXIOM_EMB_D", 384),
        n_layers: envu("AXIOM_EMB_LAYERS", 6),
        n_heads: envu("AXIOM_EMB_HEADS", 6),
        ffn_dim: envu("AXIOM_EMB_FFN", 1536),
        max_seq: envu("AXIOM_EMB_SEQ", 256),
        norm_eps: 1e-5,
    };
    let lr = envf("AXIOM_EMB_LR", 3e-4);
    let tau = envf("AXIOM_EMB_TAU", 0.05);
    let bs = envu("AXIOM_EMB_BS", 32);
    let epochs = envu("AXIOM_EMB_EPOCHS", 10);
    let grad_clip = envf("AXIOM_EMB_GRAD_CLIP", 1.0);
    let warmup = envu("AXIOM_EMB_WARMUP", 100);

    let mut pairs = read_pairs_jsonl(&pairs_path);
    assert!(pairs.len() >= bs * 2, "need >= {} pairs; got {}", bs * 2, pairs.len());
    // Deterministic shuffle so a batch spans many files (meaningful in-batch negatives).
    pairs.sort_by_key(|p| {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        p.anchor.hash(&mut h);
        p.positive.hash(&mut h);
        h.finish()
    });
    // Optional cap (AXIOM_EMB_MAX_PAIRS) — keeps a CPU run feasible; 0 = all.
    let cap = envu("AXIOM_EMB_MAX_PAIRS", 0);
    if cap > 0 && pairs.len() > cap {
        pairs.truncate(cap);
        eprintln!("[emb] capped to {cap} pairs (AXIOM_EMB_MAX_PAIRS)");
    }
    let split = (pairs.len() as f32 * 0.9) as usize;
    let (train, val) = pairs.split_at(split);
    eprintln!("[emb] pairs train={} val={}", train.len(), val.len());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let enc = BiEncoder::new(vb, cfg.clone()).unwrap();

    let encode = |text: &str| -> Tensor {
        let ids = tok.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default();
        enc.encode(&enc.ids_tensor(&ids, &device).unwrap()).unwrap()
    };
    let stack = |texts: &[String]| -> Tensor {
        let rows: Vec<Tensor> = texts.iter().map(|s| encode(s).unsqueeze(0).unwrap()).collect();
        Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0).unwrap()
    };
    let val_recall = || -> f32 {
        let mut tot = 0.0f32;
        let mut nb = 0usize;
        for chunk in val.chunks(bs).filter(|c| c.len() >= 2) {
            let a: Vec<String> = chunk.iter().map(|p| p.anchor.clone()).collect();
            let p: Vec<String> = chunk.iter().map(|p| p.positive.clone()).collect();
            tot += batch_recall_at_1(&stack(&a), &stack(&p)).unwrap();
            nb += 1;
        }
        tot / nb.max(1) as f32
    };

    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr, ..Default::default() }).unwrap();
    let t0 = std::time::Instant::now();
    let mut best = 0.0f32;
    let mut step = 0usize;
    for ep in 0..epochs {
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for chunk in train.chunks(bs).filter(|c| c.len() >= 2) {
            if warmup > 0 && step < warmup {
                opt.set_learning_rate(lr * (step + 1) as f64 / warmup as f64);
            } else if step == warmup {
                opt.set_learning_rate(lr);
            }
            let a: Vec<String> = chunk.iter().map(|p| p.anchor.clone()).collect();
            let p: Vec<String> = chunk.iter().map(|p| p.positive.clone()).collect();
            let res = (|| -> candle_core::Result<f32> {
                let loss = info_nce(&stack(&a), &stack(&p), tau)?;
                let lval = loss.to_scalar::<f32>()?;
                if !lval.is_finite() {
                    return Ok(lval);
                }
                let mut grads = loss.backward()?;
                if grad_clip > 0.0 {
                    let vars = varmap.all_vars();
                    let mut tot = 0f64;
                    for v in &vars {
                        if let Some(g) = grads.get(v.as_tensor()) {
                            tot += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                        }
                    }
                    let norm = tot.sqrt();
                    if norm.is_finite() && norm > grad_clip {
                        let s = grad_clip / (norm + 1e-6);
                        for v in &vars {
                            if let Some(g) = grads.get(v.as_tensor()) {
                                let c = (g * s)?;
                                grads.insert(v.as_tensor(), c);
                            }
                        }
                    }
                }
                opt.step(&grads)?;
                Ok(lval)
            })();
            if let Ok(l) = res {
                if l.is_finite() {
                    sum += l;
                    cnt += 1;
                }
            }
            step += 1;
        }
        let vr = val_recall();
        eprintln!(
            "[emb] epoch {} loss={:.4} val_recall@1={:.3} ({:.0}s)",
            ep + 1,
            sum / cnt.max(1) as f32,
            vr,
            t0.elapsed().as_secs_f32()
        );
        if vr > best {
            best = vr;
            varmap.save(&ckpt).expect("save");
            let _ = EmbedderMeta {
                d_model: cfg.d_model,
                n_layers: cfg.n_layers,
                n_heads: cfg.n_heads,
                ffn_dim: cfg.ffn_dim,
                max_seq: cfg.max_seq,
                vocab_size: vocab,
                tokenizer: bpe.clone(),
                val_recall_at_1: best,
            }
            .save(&ckpt);
        }
    }
    eprintln!("[emb] BEST val_recall@1={:.3} → {ckpt}", best);
    println!("{best:.3}");
}
