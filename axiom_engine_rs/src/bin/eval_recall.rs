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
    for (label, body) in seeds {
        let emb = embed_text(&pipeline, body).unwrap();
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
        let q_emb = embed_text(&pipeline, q).unwrap();
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
    println!("\n=== Recall eval ===");
    println!("precision@1 = {p1:.2}");
    println!("precision@{k} = {pk:.2}");

    let _ = std::fs::remove_dir_all(&root);

    if pk < 0.6 {
        eprintln!("[eval_recall] WARNING: precision@{k} below 0.60 — recall quality not yet acceptable");
    }
}
