//! smoke_embed — Phase 2.0.1a de-risk. Mines doc<->body + markdown pairs from
//! THIS repo, trains a tiny BiEncoder contrastively on CPU for a short budget,
//! and reports held-out recall@1 + mean inter-anchor cosine (must rise / fall
//! vs an untrained baseline). Proves the architecture+loss work before any
//! GPU run. CPU only; never touches the proxy.
//!   cargo run --release --bin smoke_embed --manifest-path axiom_engine_rs/Cargo.toml

use axiom_engine::contrastive::{batch_recall_at_1, info_nce};
use axiom_engine::encoder::{BiEncoder, EncoderConfig};
use axiom_engine::pairs::{mine_doc_body, mine_markdown, Pair};
use candle_core::{DType, Device, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use tokenizers::Tokenizer;

fn main() {
    std::thread::Builder::new().stack_size(512 * 1024 * 1024).spawn(run).unwrap().join().unwrap();
}

fn run() {
    let repo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let tok = Tokenizer::from_file(repo.join("checkpoints/axiom_bpe.json")).expect("tokenizer");
    let vocab = tok.get_vocab_size(true);
    let device = Device::Cpu;

    // Mine pairs from this repo's source + docs.
    let mut pairs: Vec<Pair> = Vec::new();
    for entry in std::fs::read_dir(repo.join("axiom_engine_rs/src")).unwrap().flatten() {
        if let Ok(t) = std::fs::read_to_string(entry.path()) {
            pairs.extend(mine_doc_body(&t, 25));
        }
    }
    for entry in std::fs::read_dir(repo.join("axiom_engine_rs/src/bin")).into_iter().flatten().flatten() {
        if let Ok(t) = std::fs::read_to_string(entry.path()) {
            pairs.extend(mine_doc_body(&t, 25));
        }
    }
    for entry in std::fs::read_dir(repo.join("docs/superpowers/specs")).into_iter().flatten().flatten() {
        if let Ok(t) = std::fs::read_to_string(entry.path()) {
            pairs.extend(mine_markdown(&t, 40));
        }
    }
    eprintln!("[smoke] mined {} pairs", pairs.len());
    assert!(pairs.len() >= 16, "need >=16 pairs to train; got {}", pairs.len());

    // CRITICAL: shuffle so a batch spans many source files. Mined pairs are in
    // file order; without this, every in-batch negative is near-identical to the
    // positive (same file) and the contrastive task is degenerate. Deterministic
    // hash shuffle for reproducibility.
    pairs.sort_by_key(|p| {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        p.anchor.hash(&mut h);
        p.positive.hash(&mut h);
        h.finish()
    });

    let cfg = EncoderConfig {
        vocab_size: vocab,
        d_model: 128,
        n_layers: 3,
        n_heads: 4,
        ffn_dim: 512,
        max_seq: 128,
        norm_eps: 1e-5,
    };
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let enc = BiEncoder::new(vb, cfg.clone()).unwrap();

    let encode = |text: &str| -> Tensor {
        let ids = tok.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default();
        let t = enc.ids_tensor(&ids, &device).unwrap();
        enc.encode(&t).unwrap()
    };
    let stack = |texts: &[String]| -> Tensor {
        let rows: Vec<Tensor> = texts.iter().map(|s| encode(s).unsqueeze(0).unwrap()).collect();
        Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0).unwrap()
    };

    // held-out split
    let split = (pairs.len() as f32 * 0.85) as usize;
    let (train, val) = pairs.split_at(split.max(1));

    let va: Vec<String> = val.iter().map(|p| p.anchor.clone()).collect();
    let vp: Vec<String> = val.iter().map(|p| p.positive.clone()).collect();
    let base = batch_recall_at_1(&stack(&va), &stack(&vp)).unwrap();
    eprintln!("[smoke] baseline val recall@1 = {base:.3}");

    // --- Overfit sanity probe: can the model drive loss→0 on ONE fixed batch? --
    // If this fails, the architecture/loss/optimizer is broken (not a data issue).
    {
        let probe: Vec<&Pair> = train.iter().take(8).collect();
        let a: Vec<String> = probe.iter().map(|p| p.anchor.clone()).collect();
        let p: Vec<String> = probe.iter().map(|p| p.positive.clone()).collect();
        let mut probe_opt =
            AdamW::new(varmap.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() }).unwrap();
        eprintln!("[probe] overfitting one fixed 8-pair batch (loss should fall toward 0):");
        let vars = varmap.all_vars();
        eprintln!("[probe] n_vars={}", vars.len());
        for s in 0..200 {
            let loss = info_nce(&stack(&a), &stack(&p), 0.05).unwrap();
            let g = loss.backward().unwrap();
            if s == 0 {
                // How many of our Vars actually received a gradient, and total norm?
                let mut with_grad = 0usize;
                let mut total = 0f64;
                for v in &vars {
                    if let Some(gr) = g.get(v.as_tensor()) {
                        with_grad += 1;
                        total += gr.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap() as f64;
                    }
                }
                eprintln!("[probe] vars_with_grad={with_grad}/{} grad_norm={:.6}", vars.len(), total.sqrt());
            }
            probe_opt.step(&g).unwrap();
            if s % 40 == 0 {
                eprintln!("[probe]   step {s} loss={:.4}", loss.to_scalar::<f32>().unwrap());
            }
        }
        let final_r = batch_recall_at_1(&stack(&a), &stack(&p)).unwrap();
        eprintln!("[probe] final train-batch recall@1 = {final_r:.3} (want ~1.0)");
    }

    let lr = 3e-4f64;
    let warmup = 100usize;
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr, ..Default::default() }).unwrap();
    let bs: usize = std::env::var("SMOKE_BS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let steps: usize = std::env::var("SMOKE_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    for step in 0..steps {
        // linear LR warmup
        if step < warmup {
            opt.set_learning_rate(lr * (step + 1) as f64 / warmup as f64);
        } else if step == warmup {
            opt.set_learning_rate(lr);
        }
        let start = (step * bs) % train.len().max(1);
        let batch: Vec<&Pair> = (0..bs).map(|j| &train[(start + j) % train.len()]).collect();
        let a: Vec<String> = batch.iter().map(|p| p.anchor.clone()).collect();
        let p: Vec<String> = batch.iter().map(|p| p.positive.clone()).collect();
        let av = stack(&a);
        let pv = stack(&p);
        let loss = info_nce(&av, &pv, 0.05).unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
        if step % 100 == 0 {
            // val-recall curve: does it rise (generalization) before overfit?
            let vr = batch_recall_at_1(&stack(&va), &stack(&vp)).unwrap();
            eprintln!(
                "[smoke] step {step} loss={:.4} val_recall@1={:.3}",
                loss.to_scalar::<f32>().unwrap(),
                vr
            );
        }
    }

    let trained = batch_recall_at_1(&stack(&va), &stack(&vp)).unwrap();
    let av = stack(&va);
    let avv: Vec<Vec<f32>> = (0..va.len())
        .map(|i| av.narrow(0, i, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    let mut s = 0.0f32;
    let mut n = 0usize;
    for i in 0..avv.len() {
        for j in (i + 1)..avv.len() {
            s += avv[i].iter().zip(&avv[j]).map(|(x, y)| x * y).sum::<f32>();
            n += 1;
        }
    }
    let inter = s / n.max(1) as f32;
    println!("\n=== smoke 2.0.1a ===");
    println!("val recall@1: baseline {base:.3} -> trained {trained:.3}");
    println!("mean inter-anchor cosine (val): {inter:.3}");
    if trained > base && inter < 0.93 {
        println!("GATE 2.0.1a: PASS (recipe de-collapses + improves recall)");
    } else {
        println!("GATE 2.0.1a: FAIL — reconsider architecture/loss before the full run");
    }
}
