//! eval_model — acceptance suite for a baked checkpoint. Loads dims from the
//! sidecar, then reports: held-out perplexity (CE on a clean file), clean-vs-
//! anomaly drift margin, and a recalibrated AXIOM_DRIFT_THRESHOLD (written to
//! checkpoints/axiom_drift_gate.txt). The proxy should only swap on PASS.
//!
//! Run: cargo run --release --bin eval_model

use std::path::PathBuf;

use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::{InferencePipeline, InferenceRuntimeOptions};
use axiom_engine::model_meta::ModelMeta;
use candle_core::{Device, Tensor};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

const CHUNK: usize = 512;

fn chunked_ce(pipeline: &InferencePipeline, ids: &[u32], vocab: usize) -> f32 {
    if ids.len() < 2 {
        return f32::INFINITY;
    }
    let dev = pipeline.device();
    let mut total = 0.0f32;
    let mut n = 0usize;
    for w in ids.chunks(CHUNK) {
        if w.len() < 2 {
            continue;
        }
        let m = w.len();
        let mut states = pipeline.init_session_states().unwrap();
        let input = Tensor::from_vec(w[..m - 1].to_vec(), (1, m - 1), dev).unwrap();
        let logits = pipeline.model().forward_lm(&input, &mut states).unwrap();
        let l2d = logits.squeeze(0).unwrap().reshape((m - 1, vocab)).unwrap();
        let tgt = Tensor::from_vec(w[1..].to_vec(), (m - 1,), dev).unwrap();
        total += candle_nn::loss::cross_entropy(&l2d, &tgt).unwrap().to_scalar::<f32>().unwrap()
            * (m - 1) as f32;
        for s in states.iter_mut() {
            *s = s.detach();
        }
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
    let root = repo_root();
    let ckpt = std::env::var("AXIOM_BPE_CKPT").unwrap_or_else(|_| {
        root.join("checkpoints/axiom_production_bpe.bin").to_string_lossy().into()
    });
    let meta = ModelMeta::load(&ckpt).expect("sidecar .meta.json (run train_semantic first)");
    let bpe = meta.tokenizer.clone();
    let vocab = meta.vocab_size;
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let config = AxiomConfig {
        d_model: meta.d_model,
        n_layers: meta.n_layers,
        vocab_size: vocab,
        lr_inner: meta.lr_inner,
        norm_eps: meta.norm_eps,
    };
    let runtime = InferenceRuntimeOptions { tokenizer_path: Some(bpe.clone()), ..Default::default() };
    let pipeline = InferencePipeline::with_checkpoint_and_options(config, device, &ckpt, runtime)
        .expect("load pipeline");

    let read = |rel: &str| std::fs::read_to_string(root.join(rel)).unwrap_or_default();
    let enc = |t: &str| pipeline.encode_text(t);

    // Held-out perplexity proxy: a clean repo file.
    let held = chunked_ce(&pipeline, &enc(&read("axiom_engine_rs/src/server.rs")), vocab);
    // Drift separation: clean vs anomaly.
    let clean: Vec<f32> = ["axiom_engine_rs/src/model.rs", "axiom_engine_rs/src/inference.rs"]
        .iter()
        .map(|f| chunked_ce(&pipeline, &enc(&read(f)), vocab))
        .collect();
    let anomaly = chunked_ce(&pipeline, &enc(&read("tests/anomaly_target.rs")), vocab);
    let clean_max = clean.iter().cloned().fold(0.0f32, f32::max);
    let margin = anomaly - clean_max;
    let gate = (clean_max + anomaly) / 2.0;

    eprintln!("[eval] model d{}/{}L vocab{} val_ce(train)={:.3}", meta.d_model, meta.n_layers, vocab, meta.val_ce);
    eprintln!("[eval] held-out CE (server.rs)   = {held:.4}");
    eprintln!("[eval] clean CE                  = {clean:?} (max {clean_max:.4})");
    eprintln!("[eval] anomaly CE                = {anomaly:.4}");
    eprintln!("[eval] drift separation margin   = {margin:+.4}");
    let pass = margin > 0.0 && held.is_finite();
    eprintln!("[eval] ACCEPTANCE: {}", if pass { "PASS" } else { "FAIL" });
    if pass {
        let gate_file = root.join("checkpoints/axiom_drift_gate.txt");
        std::fs::write(&gate_file, format!("{gate:.4}")).ok();
        eprintln!("[eval] recalibrated AXIOM_DRIFT_THRESHOLD={gate:.4} → {}", gate_file.display());
    }
    println!("{}", if pass { "PASS" } else { "FAIL" });
}
