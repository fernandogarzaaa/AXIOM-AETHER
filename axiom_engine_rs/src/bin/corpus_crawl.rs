//! corpus_crawl — build a deduped, size-capped, multi-language code corpus on
//! disk from local roots (+ optional extra roots passed as args). Streams files;
//! RAM holds only one file + the dedup hash set. Writes shards of <= SHARD_BYTES
//! to <out>/shard_NNNN.txt so the trainer can stream them.
//!
//! Run: cargo run --release --bin corpus_crawl
//! Env: AXIOM_CORPUS_OUT (default checkpoints/corpus)
//!      AXIOM_CORPUS_MAX_MB (default 200)        total corpus cap
//!      AXIOM_CORPUS_MAX_FILE_KB (default 256)   per-file cap
//!      AXIOM_CORPUS_SHARD_MB (default 8)
//! Extra roots: pass directories as CLI args to add to the defaults.

use std::io::Write;
use std::path::PathBuf;

use axiom_engine::corpus::{collect_files, Deduper};

fn env_u64(k: &str, d: u64) -> u64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn default_roots() -> Vec<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    let home = PathBuf::from(home);
    let mut roots = vec![
        home.join(".cargo/registry/src"), // Rust (bulk)
        PathBuf::from("/c/Program Files/Python314/Lib/site-packages"), // Python
    ];
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    roots.retain(|p| p.exists());
    roots
}

fn main() -> std::io::Result<()> {
    let out = std::env::var("AXIOM_CORPUS_OUT").unwrap_or_else(|_| "checkpoints/corpus".into());
    let max_total = env_u64("AXIOM_CORPUS_MAX_MB", 200) * 1024 * 1024;
    let max_file = env_u64("AXIOM_CORPUS_MAX_FILE_KB", 256) * 1024;
    let shard_bytes = env_u64("AXIOM_CORPUS_SHARD_MB", 8) * 1024 * 1024;
    std::fs::create_dir_all(&out)?;

    let mut roots = default_roots();
    for a in std::env::args().skip(1) {
        let p = PathBuf::from(a);
        if p.exists() {
            roots.push(p);
        }
    }
    eprintln!(
        "[crawl] roots: {}",
        roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", ")
    );

    let mut dedup = Deduper::new();
    let mut total: u64 = 0;
    let mut shard_idx = 0usize;
    let mut shard_len: u64 = 0;
    let mut shard = std::fs::File::create(format!("{out}/shard_{shard_idx:04}.txt"))?;
    let mut files_used = 0usize;

    'roots: for root in &roots {
        // 80k file cap per root keeps the walk bounded; dedup handles overlap.
        for path in collect_files(root, 80_000, max_file) {
            if total >= max_total {
                break 'roots;
            }
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            if std::str::from_utf8(&bytes).is_err() {
                continue; // skip binary
            }
            if !dedup.accept(&bytes) {
                continue; // skip duplicates
            }
            shard.write_all(&bytes)?;
            shard.write_all(b"\n")?;
            total += bytes.len() as u64 + 1;
            shard_len += bytes.len() as u64 + 1;
            files_used += 1;
            if shard_len >= shard_bytes {
                shard_idx += 1;
                shard = std::fs::File::create(format!("{out}/shard_{shard_idx:04}.txt"))?;
                shard_len = 0;
            }
        }
    }
    eprintln!(
        "[crawl] DONE: {files_used} unique files, {} MB across {} shard(s) -> {out}",
        total / (1024 * 1024),
        shard_idx + 1
    );
    println!("{total}");
    Ok(())
}
