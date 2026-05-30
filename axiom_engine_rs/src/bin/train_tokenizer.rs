//! train_tokenizer — offline ByteLevel-BPE trainer for the Axiom engine.
//!
//! Trains a Byte-Pair Encoding tokenizer on the repository's Rust corpus (fully
//! local; no network/HF Hub) and saves `tokenizer.json`. This replaces the
//! legacy SHA-256 hash-bucket "tokenizer" with a real semantic vocabulary.
//!
//! Build/run (separate binary — does not touch the running proxy exe):
//!   cargo build --release --bin train_tokenizer
//!   AXIOM_BPE_VOCAB=8000 ./target/release/train_tokenizer
//!
//! Env:
//!   AXIOM_BPE_VOCAB  target vocab size (default 8000 — hardware-aware: a 50k
//!                    lm_head is impractical to train on CPU; 8k still yields a
//!                    true semantic BPE, a vast upgrade over 256 hash buckets)
//!   AXIOM_BPE_OUT    output path (default <repo>/checkpoints/axiom_bpe.json)

use std::path::{Path, PathBuf};

use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Result, Tokenizer};

fn collect_rs(dir: &Path, out: &mut Vec<String>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_rs(&p, out);
            } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
                out.push(p.to_string_lossy().to_string());
            }
        }
    }
}

fn main() -> Result<()> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate parent")
        .to_path_buf();
    let vocab: usize = std::env::var("AXIOM_BPE_VOCAB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8000);
    let out = std::env::var("AXIOM_BPE_OUT").unwrap_or_else(|_| {
        repo.join("checkpoints/axiom_bpe.json")
            .to_string_lossy()
            .to_string()
    });
    if let Some(parent) = Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut files = Vec::new();
    collect_rs(&repo.join("axiom_engine_rs/src"), &mut files);
    collect_rs(&repo.join("axiom_engine_rs/tests"), &mut files);
    collect_rs(&repo.join("tests"), &mut files);
    eprintln!("[bpe] corpus: {} Rust files; target vocab={vocab}", files.len());
    if files.is_empty() {
        return Err("no .rs corpus files found".into());
    }

    // ByteLevel BPE: full-byte alphabet ⇒ no UNK; reversible.
    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(ByteLevel::new(false, true, true));
    tokenizer.with_decoder(ByteLevel::new(false, true, true));

    let bpe_trainer = BpeTrainerBuilder::new()
        .vocab_size(vocab)
        .min_frequency(2)
        .initial_alphabet(ByteLevel::alphabet())
        .special_tokens(vec![
            AddedToken::from("<unk>", true),
            AddedToken::from("<pad>", true),
            AddedToken::from("<eos>", true),
        ])
        .build();
    // The type-erased `Tokenizer` carries `Model = ModelWrapper`; train_from_files
    // requires a `Trainer<Model = ModelWrapper>`, so wrap the BPE trainer.
    let mut trainer: TrainerWrapper = bpe_trainer.into();

    tokenizer.train_from_files(&mut trainer, files)?;
    let real_vocab = tokenizer.get_vocab_size(true);
    tokenizer.save(&out, true)?;
    eprintln!("[bpe] DONE: trained vocab_size={real_vocab}; saved -> {out}");
    // Emit the real vocab on stdout so scripts can capture it.
    println!("{real_vocab}");
    Ok(())
}
