//! Recall-quality acceptance harness (Phase 2.0).
//!
//! Seeds labeled memories, queries with related phrasing, and reports
//! precision@k — the spec's headline metric. Run on CPU:
//!   cargo run --release --bin eval_recall --manifest-path axiom_engine_rs/Cargo.toml

use axiom_engine::config::AxiomConfig;
use axiom_engine::embedder::embed_text;
use axiom_engine::inference::{InferencePipeline, InferenceRuntimeOptions};
use axiom_engine::memory_recall::{recall, RecallParams};
use axiom_engine::memory_store::{now_secs, MemoryKind, MemoryRecord, MemoryStore};
use candle_core::Device;

/// Elementwise mean over a set of equal-length vectors.
fn mean_vec(vecs: &[Vec<f32>]) -> Vec<f32> {
    if vecs.is_empty() {
        return Vec::new();
    }
    let d = vecs[0].len();
    let mut m = vec![0.0f32; d];
    for v in vecs {
        for (i, x) in v.iter().enumerate() {
            m[i] += x;
        }
    }
    let n = vecs.len() as f32;
    for x in &mut m {
        *x /= n;
    }
    m
}

/// Subtract `mean` (anisotropy fix) then L2-renormalize.
fn center_renorm(v: &[f32], mean: &[f32]) -> Vec<f32> {
    let centered: Vec<f32> = v.iter().zip(mean).map(|(x, m)| x - m).collect();
    let norm: f32 = centered.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return centered;
    }
    centered.into_iter().map(|x| x / norm).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn main() {
    // Prefer the real production tokenizer + checkpoint if present, so the eval
    // reflects the deployed model; else fall back to a tiny random pipeline.
    let tok = std::env::var("AXIOM_TOKENIZER").ok().filter(|p| !p.trim().is_empty());
    let ckpt =
        std::env::var("AXIOM_BPE_CKPT").unwrap_or_else(|_| "____no_such_checkpoint____".to_string());

    let config =
        AxiomConfig { d_model: 256, n_layers: 4, vocab_size: 16000, lr_inner: 1e-3, norm_eps: 1e-6 };
    let runtime = InferenceRuntimeOptions { tokenizer_path: tok, ..Default::default() };
    let pipeline =
        match InferencePipeline::with_checkpoint_and_options(config, Device::Cpu, &ckpt, runtime) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[eval_recall] could not build production pipeline ({e}); using tiny random model");
                let tiny = AxiomConfig {
                    d_model: 64,
                    n_layers: 2,
                    vocab_size: 256,
                    lr_inner: 1e-3,
                    norm_eps: 1e-6,
                };
                InferencePipeline::with_checkpoint(tiny, Device::Cpu, "____none____").unwrap()
            }
        };

    // Prefer the trained contrastive embedder (Phase 2.0.1) when present.
    let emb_ckpt =
        std::env::var("AXIOM_EMB_CKPT").unwrap_or_else(|_| "checkpoints/axiom_embedder.bin".to_string());
    let trained = axiom_engine::embedder::EmbeddingModel::load(&emb_ckpt, Device::Cpu);
    eprintln!(
        "[eval_recall] embedder: {}",
        if trained.is_some() { "TRAINED contrastive" } else { "TTT pooling (no axiom_embedder.bin)" }
    );
    let embed = |text: &str| -> Vec<f32> {
        match &trained {
            Some(e) => e.embed(text).unwrap(),
            None => embed_text(&pipeline, text).unwrap(),
        }
    };

    let mut root = std::env::temp_dir();
    root.push(format!("axiom_eval_recall_{}", now_secs()));
    let store = MemoryStore::open(&root).unwrap();

    // (label, body) seed set. Each query should retrieve its labeled body first.
    let seeds = [
        ("auth", "We chose JWT access tokens with a 15-minute expiry and refresh-token rotation for authentication."),
        ("db", "The project uses PostgreSQL with sqlx; migrations live in db/migrations and run on boot."),
        ("style", "Rust code uses 4-space indentation and every public function gets a doc comment."),
        ("retry", "Network calls retry 3 times with exponential backoff starting at 200ms."),
        ("cache", "Hot config is cached in-process for 60 seconds; never cache user-specific data."),
    ];
    let mut seed_embs: Vec<(&str, Vec<f32>)> = Vec::new();
    for (label, body) in seeds {
        let emb = embed(body);
        seed_embs.push((label, emb.clone()));
        let rec = MemoryRecord {
            id: label.to_string(),
            scope: "personal".to_string(),
            ts: now_secs(),
            kind: MemoryKind::Decision,
            body: body.to_string(),
            embedding: emb,
            drift_at_ingest: 0.0,
            supersedes: None,
            tombstone: false,
        };
        store.append(&rec).unwrap();
    }

    // Paraphrased queries → expected label.
    let queries = [
        ("how do we handle login tokens?", "auth"),
        ("what database does this project use?", "db"),
        ("what's our indentation convention?", "style"),
        ("how should failed network requests behave?", "retry"),
        ("how long is config cached?", "cache"),
    ];

    let k = 3usize;
    let mut hits_at_1 = 0usize;
    let mut hits_at_k = 0usize;
    for (q, expected) in queries {
        let q_emb = embed(q);
        let results =
            recall(&store, &["personal".to_string()], &q_emb, &RecallParams { min_score: 0.0, k });
        let ids: Vec<&str> = results.iter().map(|h| h.record.id.as_str()).collect();
        if ids.first() == Some(&expected) {
            hits_at_1 += 1;
        }
        if ids.contains(&expected) {
            hits_at_k += 1;
        }
        println!("query={q:?} expected={expected} got={ids:?}");
    }

    let n = queries.len() as f32;
    let p1 = hits_at_1 as f32 / n;
    let pk = hits_at_k as f32 / n;
    println!("\n=== Recall eval (raw last-token) ===");
    println!("precision@1 = {p1:.2}");
    println!("precision@{k} = {pk:.2}");

    // --- 2.0.1a experiment: anisotropy fix (mean-center + renormalize) -------
    // Subtract the dominant common direction so cosine can discriminate.
    let mean = mean_vec(&seed_embs.iter().map(|(_, e)| e.clone()).collect::<Vec<_>>());
    let centered_seeds: Vec<(&str, Vec<f32>)> = seed_embs
        .iter()
        .map(|(l, e)| (*l, center_renorm(e, &mean)))
        .collect();

    let mut c_hits_at_1 = 0usize;
    let mut c_hits_at_k = 0usize;
    for (q, expected) in queries {
        let q_emb = embed(q);
        let q_c = center_renorm(&q_emb, &mean);
        let mut scored: Vec<(f32, &str)> = centered_seeds
            .iter()
            .map(|(l, e)| (cosine(&q_c, e), *l))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let ids: Vec<&str> = scored.iter().take(k).map(|(_, l)| *l).collect();
        if ids.first() == Some(&expected) {
            c_hits_at_1 += 1;
        }
        if ids.contains(&expected) {
            c_hits_at_k += 1;
        }
        println!("[centered] query={q:?} expected={expected} got={ids:?}");
    }
    let cp1 = c_hits_at_1 as f32 / n;
    let cpk = c_hits_at_k as f32 / n;
    println!("\n=== Recall eval (mean-centered / anisotropy fix) ===");
    println!("precision@1 = {cp1:.2}");
    println!("precision@{k} = {cpk:.2}");

    // --- Diagnostic: are the embeddings actually distinct per input? --------
    let q_embs: Vec<Vec<f32>> = queries.iter().map(|(q, _)| embed(q)).collect();
    let mut q_pair_sum = 0.0f32;
    let mut q_pairs = 0usize;
    for i in 0..q_embs.len() {
        for j in (i + 1)..q_embs.len() {
            q_pair_sum += cosine(&q_embs[i], &q_embs[j]);
            q_pairs += 1;
        }
    }
    let seed_vecs: Vec<Vec<f32>> = seed_embs.iter().map(|(_, e)| e.clone()).collect();
    let mut s_pair_sum = 0.0f32;
    let mut s_pairs = 0usize;
    for i in 0..seed_vecs.len() {
        for j in (i + 1)..seed_vecs.len() {
            s_pair_sum += cosine(&seed_vecs[i], &seed_vecs[j]);
            s_pairs += 1;
        }
    }
    println!("\n=== Diagnostic ===");
    println!("mean inter-QUERY cosine = {:.4} (1.0 = all queries identical → no signal)", q_pair_sum / q_pairs.max(1) as f32);
    println!("mean inter-SEED  cosine = {:.4}", s_pair_sum / s_pairs.max(1) as f32);
    // Score vector for the first query against each seed (do scores vary?).
    print!("query[0] cosine to each seed: ");
    for (l, e) in &seed_embs {
        print!("{l}={:.4} ", cosine(&q_embs[0], e));
    }
    println!();

    let _ = std::fs::remove_dir_all(&root);

    if pk < 0.6 && cpk < 0.6 {
        eprintln!("[eval_recall] WARNING: precision@{k} below 0.60 for both raw and centered — recall quality not yet acceptable");
    }
}
