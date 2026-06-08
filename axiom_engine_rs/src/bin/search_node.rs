//! search_node - live scrape -> BPE -> online TTT -> <axiom_search_fingerprint>.
//!
//! End-to-end demonstration of the JIT search-reasoning node: scrapes the web
//! for a query with native Rust HTTP/HTML parsing, absorbs the results into
//! local fast-weights via online TTT, and emits a dense semantic pointer.
//!
//! Run:  cargo build --release --bin search_node
//!       ./target/release/search_node "your query"
//!
//! Env:  AXIOM_BPE (tokenizer.json)  AXIOM_BPE_CKPT (production checkpoint)
//!       AXIOM_SEARCH_TOPK (32)       AXIOM_SEARCH_MAX_PAGES (5)

use std::path::PathBuf;

use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::{InferencePipeline, InferenceRuntimeOptions};
use axiom_engine::search_ingest::ingest_search_text;
use axiom_engine::search_scrape::{default_max_pages, scrape_query_text};
use candle_core::Device;
use tokenizers::Tokenizer;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn main() {
    // TTT adaptation is stack-heavy on long streams; keep the search demo on a
    // large worker stack rather than the platform default.
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

fn run() {
    let query = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if query.trim().is_empty() {
        eprintln!("usage: search_node \"<query-or-url>\"");
        std::process::exit(2);
    }

    let root = repo_root();
    let bpe = std::env::var("AXIOM_BPE").unwrap_or_else(|_| {
        root.join("checkpoints/axiom_bpe.json")
            .to_string_lossy()
            .into()
    });
    let ckpt = std::env::var("AXIOM_BPE_CKPT").unwrap_or_else(|_| {
        root.join("checkpoints/axiom_production_bpe.bin")
            .to_string_lossy()
            .into()
    });
    let top_k: usize = std::env::var("AXIOM_SEARCH_TOPK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let max_pages: usize = std::env::var("AXIOM_SEARCH_MAX_PAGES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(default_max_pages);

    eprintln!("[search_node] scraping: {query:?}");
    let text = scrape_query_text(&query, max_pages).expect("native scrape");
    eprintln!("[search_node] scraped {} chars", text.len());

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let vocab = Tokenizer::from_file(&bpe)
        .expect("load BPE")
        .get_vocab_size(true);
    let config = AxiomConfig {
        d_model: 256,
        n_layers: 4,
        vocab_size: vocab,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    let runtime = InferenceRuntimeOptions {
        tokenizer_path: Some(bpe.clone()),
        ..Default::default()
    };
    let pipeline = InferencePipeline::with_checkpoint_and_options(config, device, &ckpt, runtime)
        .expect("build pipeline");

    let fp = ingest_search_text(&pipeline, &query, &text, top_k).expect("ingest");
    eprintln!(
        "[search_node] ingested {} tokens in {} chunks ({} ms), recall_norm={:.3}",
        fp.tokens_ingested, fp.chunks, fp.elapsed_ms, fp.recall_norm
    );
    println!("{}", fp.to_wire());
}
