//! `axiom prime <path>` — warm-start the persistent vibe memory from a codebase.
//!
//! Online TTT's real differentiator is that the per-layer fast-weight matrix
//! `W̃` *is* a neural associative memory: streaming tokens through it bends the
//! weights toward the structure of what it just read. `vibe_memory` already
//! EMA-merges those adapted matrices into a long-lived "codebase DNA" tensor set
//! — but only as a side effect of live proxy traffic. This command makes priming
//! a first-class, offline step: crawl a repo, absorb its source through the TTT
//! stack, and commit the result so future sessions start *already adapted* to
//! the code (with `AXIOM_VIBE_PRIME=1`).
//!
//! It is intentionally bounded (file count / size / total tokens) so priming a
//! large monorepo stays a quick, predictable batch job rather than an open-ended
//! crawl.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::Result;
use walkdir::WalkDir;

use crate::context_compressor::adapt_session_blocking;
use crate::inference::InferencePipeline;
use crate::vibe_memory::{MasterVibe, DEFAULT_VIBE_DECAY};

/// Source extensions worth absorbing. Skews toward code; includes a few prose /
/// config formats that carry real project structure.
const PRIME_EXTENSIONS: &[&str] = &[
    "rs", "go", "py", "js", "ts", "tsx", "jsx", "java", "cs", "c", "cc", "cpp", "h", "hpp", "rb",
    "php", "swift", "kt", "scala", "sh", "toml", "yaml", "yml", "md",
];

/// Directory names never worth crawling (build output, vendored deps, VCS).
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    "vendor",
    "__pycache__",
    ".venv",
    "venv",
    ".axiom",
    "checkpoints",
];

const MAX_FILE_BYTES: u64 = 256 * 1024;
const DEFAULT_MAX_FILES: usize = 2000;
const DEFAULT_MAX_TOTAL_TOKENS: usize = 200_000;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

/// Summary of a priming run, returned for callers/tests that want to assert on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimeReport {
    pub files_absorbed: usize,
    pub tokens_absorbed: usize,
    pub vibe_path: PathBuf,
}

/// Walk `target` and return source files worth priming, deepest-first stable by
/// path, bounded by `max_files`.
fn collect_source_files(target: &Path, max_files: usize) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in WalkDir::new(target)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            // Prune skip-dirs by name at any depth.
            !(e.file_type().is_dir()
                && e.file_name()
                    .to_str()
                    .map(|n| SKIP_DIRS.contains(&n))
                    .unwrap_or(false))
        })
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let ext_ok = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| PRIME_EXTENSIONS.contains(&e))
            .unwrap_or(false);
        if !ext_ok {
            continue;
        }
        if entry.metadata().map(|m| m.len() > MAX_FILE_BYTES).unwrap_or(true) {
            continue;
        }
        files.push(path.to_path_buf());
        if files.len() >= max_files {
            break;
        }
    }
    // Stable order so a given repo primes deterministically.
    files.sort();
    files
}

/// Crawl `target`, absorb its source through the TTT stack into one session
/// state, then EMA-merge that state into the master vibe at `vibe_path`.
pub fn run_prime(
    target: &Path,
    pipeline: &InferencePipeline,
    vibe_path: &Path,
) -> Result<PrimeReport> {
    let max_files = env_usize("AXIOM_PRIME_MAX_FILES", DEFAULT_MAX_FILES);
    let max_total_tokens = env_usize("AXIOM_PRIME_MAX_TOKENS", DEFAULT_MAX_TOTAL_TOKENS);

    let files = collect_source_files(target, max_files);
    println!(
        "[prime] absorbing {} source file(s) from {} (cap {} files / {} tokens)",
        files.len(),
        target.display(),
        max_files,
        max_total_tokens
    );

    let mut states = pipeline.init_session_states()?;
    let n_layers = states.len();
    let d_model = if n_layers > 0 { states[0].dim(0)? } else { 0 };

    let started = Instant::now();
    let mut files_absorbed = 0usize;
    let mut tokens_absorbed = 0usize;

    for path in &files {
        if tokens_absorbed >= max_total_tokens {
            break;
        }
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => continue, // skip non-UTF8 / unreadable
        };
        if text.trim().is_empty() {
            continue;
        }
        let mut tokens = pipeline.encode_text(&text);
        if tokens.is_empty() {
            continue;
        }
        let budget = max_total_tokens - tokens_absorbed;
        if tokens.len() > budget {
            tokens.truncate(budget);
        }
        adapt_session_blocking(pipeline, &mut states, &tokens)?;
        files_absorbed += 1;
        tokens_absorbed += tokens.len();
    }

    let mut vibe =
        MasterVibe::load_or_init(vibe_path, n_layers, d_model, pipeline.device(), DEFAULT_VIBE_DECAY);
    vibe.commit_and_save(&states)?;

    let elapsed = started.elapsed();
    println!(
        "[prime] absorbed {files_absorbed} file(s) / {tokens_absorbed} tokens into {n_layers}×[{d_model}×{d_model}] W̃ in {:.1}s",
        elapsed.as_secs_f32()
    );
    println!("[prime] committed to master vibe: {}", vibe.path().display());
    println!("[prime] new sessions can start from it with AXIOM_VIBE_PRIME=1");

    Ok(PrimeReport {
        files_absorbed,
        tokens_absorbed,
        vibe_path: vibe.path().to_path_buf(),
    })
}
