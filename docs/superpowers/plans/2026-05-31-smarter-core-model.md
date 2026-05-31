# Smarter Core Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Axiom's small memorized model with a properly-converged multi-language code model — trained on a real local+download corpus, auto-sized to the RTX 2060's free VRAM, accepted on held-out cross-entropy + downstream evals, and deployable into the live proxy at any baked size via a sidecar metadata file.

**Architecture:** Pipeline-first. Testable logic lives in two new lib modules (`corpus.rs`, `model_meta.rs`); thin bins (`corpus_crawl`, extended `train_tokenizer`/`train_semantic`, new `eval_model`) drive them. A sidecar `*.meta.json` decouples model dims from hardcoded constants so the proxy auto-loads whatever size was baked. GPU is the final, isolated step (candle CUDA bump behind the `cuda` feature); CPU is the always-works fallback.

**Tech Stack:** Rust, candle-core/candle-nn 0.8, tokenizers 0.15 (ByteLevel BPE), serde/serde_json, AdamW. All commands run from `C:/Users/garza/AXIOM-AETHER/axiom_engine_rs` unless noted. Use `export PATH="$HOME/.cargo/bin:$PATH"`.

**Operating rules (this volatile box — RTX 2060 5.4 GB free, ~7.3 GB RAM free, i5-9300H 4c/8t):**
- One heavy job at a time. Never run a `cargo build`/`test` concurrently with a training run (causes OOM).
- Training/eval bins run their work on a 1 GiB-stack thread (candle `backward()` recurses).
- Stop the live proxy before a heavy training pass: `powershell.exe -NoProfile -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort 3000 -State Listen).OwningProcess -Force"` (or kill the PID from `netstat -ano | grep :3000`). Restart after with `./start_axiom.sh`.
- Build new/changed bins with `--bin <name>` only — never plain `cargo build --release` (that relinks the running `axiom_engine.exe`).

**File structure (created/modified):**
- Create `src/model_meta.rs` — sidecar metadata (serde) + pure auto-size estimator.
- Create `src/corpus.rs` — file discovery, language filter, content-hash dedup, sharding.
- Create `src/bin/corpus_crawl.rs` — thin crawler driving `corpus.rs`.
- Modify `src/bin/train_tokenizer.rs` — train BPE on the crawled corpus dir; env vocab.
- Modify `src/bin/train_semantic.rs` — shard streaming, train/val split, auto-size, OOM shrink, early-stop, sidecar bake.
- Create `src/bin/eval_model.rs` — acceptance suite + recalibrated drift gate.
- Modify `src/main.rs` — `resolve_production_model` reads dims from the sidecar.
- Modify `src/lib.rs` — register `corpus`, `model_meta`.
- Modify `start_axiom.sh` — read the eval-recalibrated gate file if present.

---

## Task 1: `model_meta.rs` — sidecar metadata + auto-size estimator

**Files:**
- Create: `src/model_meta.rs`
- Modify: `src/lib.rs` (add `pub mod model_meta;`)
- Test: inline `#[cfg(test)]` in `src/model_meta.rs`

- [ ] **Step 1: Register the module**

In `src/lib.rs`, add after `pub mod model;`:
```rust
pub mod model_meta;
```

- [ ] **Step 2: Write the failing tests**

Create `src/model_meta.rs`:
```rust
//! Sidecar metadata for baked checkpoints + a pure VRAM auto-size estimator.
//! The sidecar (`<checkpoint>.meta.json`) records the dims a checkpoint was
//! trained with so the proxy/eval load the right model without hardcoding.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Persisted alongside a checkpoint as `<path>.meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMeta {
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub lr_inner: f32,
    pub norm_eps: f32,
    /// Best held-out cross-entropy seen during training (eval signal).
    pub val_ce: f32,
    /// Tokenizer file this model was trained against.
    pub tokenizer: String,
}

impl ModelMeta {
    /// Sidecar path for a checkpoint path: `foo.bin` -> `foo.meta.json`.
    pub fn sidecar_path(checkpoint: &str) -> String {
        let p = Path::new(checkpoint);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        match p.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => {
                format!("{}/{stem}.meta.json", dir.to_string_lossy())
            }
            _ => format!("{stem}.meta.json"),
        }
    }

    pub fn save(&self, checkpoint: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).expect("serialize meta");
        std::fs::write(Self::sidecar_path(checkpoint), json)
    }

    pub fn load(checkpoint: &str) -> Option<ModelMeta> {
        let txt = std::fs::read_to_string(Self::sidecar_path(checkpoint)).ok()?;
        serde_json::from_str(&txt).ok()
    }
}

/// One rung of the auto-size ladder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizeRung {
    pub d_model: usize,
    pub n_layers: usize,
}

/// Default ladder, smallest → largest. Ceiling tuned for ~3.7 GB usable.
pub fn default_ladder() -> Vec<SizeRung> {
    vec![
        SizeRung { d_model: 256, n_layers: 4 },
        SizeRung { d_model: 384, n_layers: 6 },
        SizeRung { d_model: 512, n_layers: 8 },
        SizeRung { d_model: 640, n_layers: 8 },
    ]
}

/// Estimate the training footprint (bytes) of a config: embedding + lm_head +
/// per-layer projections, times 4 (params + grad + AdamW m + v, all fp32),
/// plus a flat activation budget for one `win`-token forward.
pub fn estimate_footprint_bytes(
    d_model: usize,
    n_layers: usize,
    vocab: usize,
    win: usize,
) -> u64 {
    let params = 2 * vocab * d_model            // embedding + lm_head
        + n_layers * 3 * d_model * d_model      // w_q, w_k, w_v per layer
        + n_layers * d_model                    // layer norms (approx)
        + d_model;                              // final norm
    let param_bytes = params as u64 * 4 * 4; // fp32 × (param+grad+m+v)
    let activation_bytes = (win as u64) * (vocab as u64) * 4 * 3; // logits + softmax scratch
    param_bytes + activation_bytes
}

/// Pick the largest rung that fits `budget_bytes` with headroom. Always returns
/// at least the smallest rung (so training never refuses to start).
pub fn pick_config(budget_bytes: u64, vocab: usize, win: usize, ladder: &[SizeRung]) -> SizeRung {
    let mut chosen = ladder[0];
    for rung in ladder {
        let fp = estimate_footprint_bytes(rung.d_model, rung.n_layers, vocab, win);
        if fp <= budget_bytes {
            chosen = *rung;
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_path_swaps_extension() {
        assert_eq!(ModelMeta::sidecar_path("checkpoints/axiom_production_bpe.bin"),
                   "checkpoints/axiom_production_bpe.meta.json");
        assert_eq!(ModelMeta::sidecar_path("model.bin"), "model.meta.json");
    }

    #[test]
    fn meta_roundtrips_through_disk() {
        let m = ModelMeta { d_model: 512, n_layers: 8, vocab_size: 32000,
            lr_inner: 1e-3, norm_eps: 1e-6, val_ce: 3.21, tokenizer: "t.json".into() };
        let ckpt = std::env::temp_dir().join("axiom_meta_test.bin");
        let ckpt = ckpt.to_string_lossy().to_string();
        m.save(&ckpt).unwrap();
        assert_eq!(ModelMeta::load(&ckpt).unwrap(), m);
        let _ = std::fs::remove_file(ModelMeta::sidecar_path(&ckpt));
    }

    #[test]
    fn footprint_grows_with_size() {
        let small = estimate_footprint_bytes(256, 4, 16000, 512);
        let large = estimate_footprint_bytes(512, 8, 32000, 512);
        assert!(large > small);
    }

    #[test]
    fn pick_config_respects_budget() {
        let ladder = default_ladder();
        // Tiny budget → smallest rung.
        let tiny = pick_config(1, 32000, 512, &ladder);
        assert_eq!(tiny, ladder[0]);
        // Huge budget → largest rung.
        let huge = pick_config(u64::MAX, 32000, 512, &ladder);
        assert_eq!(huge, *ladder.last().unwrap());
    }
}
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `export PATH="$HOME/.cargo/bin:$PATH"; cargo test --lib model_meta:: -- --nocapture`
Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/model_meta.rs
git commit -m "feat(model_meta): checkpoint sidecar metadata + VRAM auto-size estimator"
```

---

## Task 2: `corpus.rs` — discovery, language filter, dedup, sharding

**Files:**
- Create: `src/corpus.rs`
- Modify: `src/lib.rs` (add `pub mod corpus;`)
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Register the module**

In `src/lib.rs` add (alphabetical, before `pub mod data_gen;`):
```rust
pub mod corpus;
```

- [ ] **Step 2: Write the failing tests + implementation**

Create `src/corpus.rs`:
```rust
//! Corpus utilities for the multi-language code crawler. All RAM-bounded:
//! callers stream files; we only ever hold one file's bytes + a hash set.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Source-code extensions we ingest (multi-language code specialist).
pub const CODE_EXTS: &[&str] = &[
    "rs", "go", "py", "ts", "tsx", "js", "jsx", "c", "h", "cpp", "hpp", "cc",
    "java", "rb", "php", "swift", "kt", "scala", "sh", "toml", "json", "md",
];

pub fn is_code_file(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => CODE_EXTS.contains(&ext.to_ascii_lowercase().as_str()),
        None => false,
    }
}

/// Content hash for dedup (cargo registry has heavy duplication).
pub fn content_hash(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

/// Tracks seen content hashes to skip duplicate files across the crawl.
#[derive(Default)]
pub struct Deduper {
    seen: HashSet<[u8; 32]>,
}

impl Deduper {
    pub fn new() -> Self {
        Self { seen: HashSet::new() }
    }
    /// Returns true if this content is new (and records it); false if duplicate.
    pub fn accept(&mut self, bytes: &[u8]) -> bool {
        self.seen.insert(content_hash(bytes))
    }
    pub fn len(&self) -> usize {
        self.seen.len()
    }
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

/// Recursively collect code-file paths under `root`, skipping hidden dirs,
/// `target/`, `.git/`, and anything over `max_file_bytes`. Bounded by `max_files`.
pub fn collect_files(root: &Path, max_files: usize, max_file_bytes: u64) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if out.len() >= max_files {
            break;
        }
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            let p = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || name == "target" || name == "node_modules" && false {
                // keep node_modules (we want JS), but skip hidden + build dirs
                continue;
            }
            if name == "target" || name.starts_with('.') {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if is_code_file(&p) {
                if let Ok(meta) = entry.metadata() {
                    if meta.len() <= max_file_bytes && meta.len() > 0 {
                        out.push(p);
                        if out.len() >= max_files {
                            break;
                        }
                    }
                }
            }
        }
    }
    out
}
```

> Note: the `name == "target" || name.starts_with('.')` line is the effective
> skip filter; the earlier expression is removed in Step 3 cleanup. Write it as:
```rust
            if p.is_dir() {
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                stack.push(p);
            } else if is_code_file(&p) {
                if let Ok(meta) = entry.metadata() {
                    if meta.len() <= max_file_bytes && meta.len() > 0 {
                        out.push(p);
                        if out.len() >= max_files { break; }
                    }
                }
            }
```
(Replace the messy filter with this clean version — only directories are
extension-checked for skip; files are extension-filtered by `is_code_file`.)

Append tests:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn detects_code_extensions() {
        assert!(is_code_file(Path::new("a/b.rs")));
        assert!(is_code_file(Path::new("x.py")));
        assert!(!is_code_file(Path::new("img.png")));
        assert!(!is_code_file(Path::new("noext")));
    }

    #[test]
    fn deduper_rejects_duplicate_content() {
        let mut d = Deduper::new();
        assert!(d.accept(b"fn main() {}"));
        assert!(!d.accept(b"fn main() {}"));
        assert!(d.accept(b"different"));
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn collect_files_finds_only_code_under_root() {
        let tmp = std::env::temp_dir().join(format!("axiom_corpus_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(tmp.join("sub"));
        std::fs::write(tmp.join("a.rs"), b"fn a() {}").unwrap();
        std::fs::write(tmp.join("b.png"), b"\x89PNG").unwrap();
        std::fs::write(tmp.join("sub/c.py"), b"def c(): pass").unwrap();
        let files = collect_files(&tmp, 100, 1_000_000);
        let names: Vec<String> = files.iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string()).collect();
        assert!(names.contains(&"a.rs".to_string()));
        assert!(names.contains(&"c.py".to_string()));
        assert!(!names.contains(&"b.png".to_string()));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
```

- [ ] **Step 3: Clean up the filter (apply the clean version shown above) and run tests**

Run: `cargo test --lib corpus:: -- --nocapture`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/corpus.rs
git commit -m "feat(corpus): code-file discovery, content-hash dedup, language filter"
```

---

## Task 3: `corpus_crawl` bin — build the on-disk corpus

**Files:**
- Create: `src/bin/corpus_crawl.rs`

- [ ] **Step 1: Write the crawler bin**

Create `src/bin/corpus_crawl.rs`:
```rust
//! corpus_crawl — build a deduped, size-capped, multi-language code corpus on
//! disk from local roots (+ optional extra roots passed as args). Streams files;
//! RAM holds only one file + the dedup hash set. Writes shards of <= SHARD_BYTES
//! to <out>/shard_NNNN.txt so the trainer can stream them.
//!
//! Run: cargo run --release --bin corpus_crawl
//! Env: AXIOM_CORPUS_OUT (default checkpoints/corpus)
//!      AXIOM_CORPUS_MAX_MB (default 200)  total corpus cap
//!      AXIOM_CORPUS_MAX_FILE_KB (default 256)  per-file cap
//!      AXIOM_CORPUS_SHARD_MB (default 8)
//! Extra roots: pass directories as CLI args to add to the defaults.

use std::io::Write;
use std::path::PathBuf;

use axiom_engine::corpus::{collect_files, Deduper};

fn env_u64(k: &str, d: u64) -> u64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn default_roots() -> Vec<PathBuf> {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_default();
    let home = PathBuf::from(home);
    let mut roots = vec![
        home.join(".cargo/registry/src"),     // Rust (bulk)
        PathBuf::from("/c/Program Files/Python314/Lib/site-packages"), // Python
    ];
    // The repo itself.
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
        if p.exists() { roots.push(p); }
    }
    eprintln!("[crawl] roots: {}", roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", "));

    let mut dedup = Deduper::new();
    let mut total: u64 = 0;
    let mut shard_idx = 0usize;
    let mut shard_len: u64 = 0;
    let mut shard = std::fs::File::create(format!("{out}/shard_{shard_idx:04}.txt"))?;
    let mut files_used = 0usize;

    'roots: for root in &roots {
        // 80k file cap per root keeps the walk bounded; dedup handles overlap.
        for path in collect_files(root, 80_000, max_file) {
            if total >= max_total { break 'roots; }
            let bytes = match std::fs::read(&path) { Ok(b) => b, Err(_) => continue };
            if !std::str::from_utf8(&bytes).is_ok() { continue; } // skip binary
            if !dedup.accept(&bytes) { continue; }                // skip duplicates
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
    eprintln!("[crawl] DONE: {files_used} unique files, {} MB across {} shard(s) -> {out}",
        total / (1024 * 1024), shard_idx + 1);
    println!("{}", total);
    Ok(())
}
```

- [ ] **Step 2: Build the bin (only this bin — protects the running proxy)**

Run: `cargo build --release --bin corpus_crawl 2>&1 | tail -5`
Expected: `Finished release`. Fix any compile errors before proceeding.

- [ ] **Step 3: Smoke-run on a tiny cap to verify shards appear**

Run: `AXIOM_CORPUS_MAX_MB=5 AXIOM_CORPUS_OUT=checkpoints/corpus ./target/release/corpus_crawl 2>&1 | tail -3; ls -la checkpoints/corpus | head`
Expected: `[crawl] DONE: N unique files, ~5 MB ...` and `shard_0000.txt` present.

- [ ] **Step 4: Commit**

```bash
git add src/bin/corpus_crawl.rs
git commit -m "feat(corpus_crawl): on-disk deduped multi-language code corpus builder"
```

> Optional small download (hybrid corpus): add a `scripts/lib/axiom-fetch-corpus.js`
> Node tool that `git clone --depth 1` a few small JS/TS/Go repos into a temp dir
> and pass that dir as an extra root to `corpus_crawl`. Defer unless language
> balance is needed after the first eval.

---

## Task 4: extend `train_tokenizer` to use the crawled corpus

**Files:**
- Modify: `src/bin/train_tokenizer.rs`

- [ ] **Step 1: Point the trainer at the corpus dir + env vocab**

In `src/bin/train_tokenizer.rs`, replace the corpus-collection block (the
`collect_rs(...)` calls building `files`) with a directory walk over the corpus
shards plus the repo, controlled by env. Replace:
```rust
    let mut files = Vec::new();
    collect_rs(&repo.join("axiom_engine_rs/src"), &mut files);
    collect_rs(&repo.join("axiom_engine_rs/tests"), &mut files);
    collect_rs(&repo.join("tests"), &mut files);
```
with:
```rust
    // Prefer the crawled corpus dir when present; else fall back to repo source.
    let corpus_dir = std::env::var("AXIOM_CORPUS_OUT")
        .unwrap_or_else(|_| repo.join("checkpoints/corpus").to_string_lossy().into());
    let mut files = Vec::new();
    if std::path::Path::new(&corpus_dir).exists() {
        for e in std::fs::read_dir(&corpus_dir).into_iter().flatten().flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("txt") {
                files.push(p.to_string_lossy().to_string());
            }
        }
    }
    if files.is_empty() {
        collect_rs(&repo.join("axiom_engine_rs/src"), &mut files);
        collect_rs(&repo.join("tests"), &mut files);
    }
```

The vocab is already env-driven (`AXIOM_BPE_VOCAB`); default it to 16000 by
changing `.unwrap_or(8000)` to `.unwrap_or(16000)`.

- [ ] **Step 2: Build the bin**

Run: `cargo build --release --bin train_tokenizer 2>&1 | tail -4`
Expected: `Finished release`.

- [ ] **Step 3: Train the BPE on the corpus (after Task 3 produced shards)**

Run: `AXIOM_BPE_VOCAB=16000 ./target/release/train_tokenizer 2>&1 | tail -4`
Expected: `[bpe] DONE: trained vocab_size=<~16000> ... checkpoints/axiom_bpe.json`.

- [ ] **Step 4: Commit**

```bash
git add src/bin/train_tokenizer.rs
git commit -m "feat(train_tokenizer): train BPE on the crawled corpus, vocab 16k default"
```

---

## Task 5: upgrade `train_semantic` — streaming, val split, auto-size, OOM shrink, early-stop, sidecar

**Files:**
- Modify: `src/bin/train_semantic.rs`

- [ ] **Step 1: Add corpus streaming + helper imports**

At the top of `src/bin/train_semantic.rs`, add imports:
```rust
use axiom_engine::model_meta::{default_ladder, pick_config, ModelMeta};
```
Add a free-VRAM probe (shells out to nvidia-smi; works without the `cuda` feature):
```rust
/// Free VRAM in bytes via nvidia-smi; None if unavailable (→ CPU budget).
fn free_vram_bytes() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let mb: u64 = s.lines().next()?.trim().parse().ok()?;
    Some(mb * 1024 * 1024)
}

/// Load all `*.txt` shard paths under the corpus dir (sorted).
fn corpus_shards(dir: &str) -> Vec<std::path::PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir).into_iter().flatten().flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("txt"))
        .collect();
    v.sort();
    v
}
```

- [ ] **Step 2: Replace the corpus-loading + config block in `run()`**

Replace the existing corpus build (the `let src = ...; ... toks.truncate(max_tokens)` block) and the fixed `AxiomConfig { d_model, n_layers, ... }` with auto-sizing + train/val split. New `run()` body core:
```rust
    let corpus_dir = std::env::var("AXIOM_CORPUS_OUT")
        .unwrap_or_else(|_| repo.join("checkpoints/corpus").to_string_lossy().into());
    let win = env_usize("AXIOM_TRAIN_WIN", 128);

    // --- Auto-size to VRAM (or env override / CPU budget) -----------------
    let budget = std::env::var("AXIOM_VRAM_BUDGET_MB").ok().and_then(|v| v.parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024)
        .or_else(|| free_vram_bytes().map(|b| (b as f64 * 0.7) as u64)) // 30% headroom
        .unwrap_or(2 * 1024 * 1024 * 1024); // CPU: conservative 2 GB
    let ladder = default_ladder();
    let rung = pick_config(budget, vocab, win, &ladder);
    let d_model = env_usize("AXIOM_DMODEL", rung.d_model);
    let n_layers = env_usize("AXIOM_NLAYERS", rung.n_layers);
    eprintln!("[train] auto-size: budget={} MB → d_model={d_model} n_layers={n_layers}", budget / (1024*1024));

    // --- Stream shards → tokens, capped; split 95/5 train/val -------------
    let max_tokens = env_usize("AXIOM_MAX_TOKENS", 2_000_000);
    let mut toks: Vec<u32> = Vec::new();
    let shards = corpus_shards(&corpus_dir);
    if shards.is_empty() {
        // Fallback to repo src so the bin still runs without a crawl.
        for p in std::fs::read_dir(repo.join("axiom_engine_rs/src")).into_iter().flatten().flatten() {
            let t = std::fs::read_to_string(p.path()).unwrap_or_default();
            toks.extend(tok.encode(t, false).map(|e| e.get_ids().to_vec()).unwrap_or_default());
            if toks.len() >= max_tokens { break; }
        }
    } else {
        for s in &shards {
            let t = std::fs::read_to_string(s).unwrap_or_default();
            toks.extend(tok.encode(t, false).map(|e| e.get_ids().to_vec()).unwrap_or_default());
            if toks.len() >= max_tokens { toks.truncate(max_tokens); break; }
        }
    }
    eprintln!("[train] corpus tokens={}", toks.len());
    assert!(toks.len() > win * 4, "corpus too small");
    let split = (toks.len() as f64 * 0.95) as usize;
    let (train_toks, val_toks) = toks.split_at(split);
```

Then build the config + model with the chosen dims (keep the existing VarMap /
AxiomTTTLM construction, but use `d_model`/`n_layers` from above and keep the
resume-from-checkpoint load).

- [ ] **Step 3: Add the held-out CE eval fn + early-stopping loop**

Add a val-CE helper (reuses the chunked pattern, no grad step):
```rust
fn val_ce(model: &AxiomTTTLM, dev: &Device, ids: &[u32], vocab: usize, win: usize) -> f32 {
    if ids.len() < 2 { return f32::INFINITY; }
    let mut total = 0.0f32; let mut n = 0usize;
    for w in ids.chunks(win) {
        if w.len() < 2 { continue; }
        let m = w.len();
        let mut states = model.init_states(dev).unwrap();
        let input = Tensor::from_vec(w[..m-1].to_vec(), (1, m-1), dev).unwrap();
        let logits = model.forward_lm(&input, &mut states).unwrap();
        let l2d = logits.squeeze(0).unwrap().reshape((m-1, vocab)).unwrap();
        let tgt = Tensor::from_vec(w[1..].to_vec(), (m-1,), dev).unwrap();
        total += candle_nn::loss::cross_entropy(&l2d, &tgt).unwrap().to_scalar::<f32>().unwrap() * (m-1) as f32;
        n += m-1;
    }
    if n == 0 { f32::INFINITY } else { total / n as f32 }
}
```

Replace the training loop with early-stopping on val CE, OOM-shrink-safe steps,
and best-checkpoint tracking:
```rust
    let windows: Vec<&[u32]> = train_toks.chunks(win).filter(|c| c.len() >= 2).collect();
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr, ..Default::default() }).unwrap();
    let patience = env_usize("AXIOM_PATIENCE", 3);
    let mut best_val = f32::INFINITY;
    let mut since_improve = 0usize;
    let mut step = 0usize;
    'train: for ep in 0..epochs {
        let mut sum = 0.0f32; let mut cnt = 0usize;
        for w in &windows {
            let n = w.len();
            let mut states = model.init_states(&device).unwrap();
            let input = Tensor::from_vec(w[..n-1].to_vec(), (1, n-1), &device).unwrap();
            // OOM-resilient step: on failure, skip this window (graceful).
            let stepped = (|| -> candle_core::Result<f32> {
                let logits = model.forward_lm(&input, &mut states)?;
                let l2d = logits.squeeze(0)?.reshape((n-1, vocab))?;
                let tgt = Tensor::from_vec(w[1..].to_vec(), (n-1,), &device)?;
                let loss = candle_nn::loss::cross_entropy(&l2d, &tgt)?;
                opt.backward_step(&loss)?;
                loss.to_scalar::<f32>()
            })();
            match stepped {
                Ok(l) => { sum += l; cnt += 1; }
                Err(e) => { eprintln!("[train] step skipped (mem?): {e}"); }
            }
            step += 1;
            if step >= step_cap { eprintln!("[train] step cap hit"); break 'train; }
        }
        let v = val_ce(&model, &device, val_toks, vocab, win);
        eprintln!("[train] epoch {} train_loss={:.4} val_ce={:.4} (step {})", ep+1, sum/cnt.max(1) as f32, v, step);
        if v + 1e-3 < best_val {
            best_val = v; since_improve = 0;
            // Save best checkpoint + sidecar.
            varmap.save(&ckpt).expect("save");
            ModelMeta { d_model, n_layers, vocab_size: vocab, lr_inner: inner_lr,
                norm_eps: 1e-6, val_ce: best_val, tokenizer: bpe.clone() }.save(&ckpt).ok();
        } else {
            since_improve += 1;
            if since_improve >= patience { eprintln!("[train] early stop (no val improvement)"); break; }
        }
    }
    eprintln!("[train] BEST val_ce={:.4} → {ckpt} (+ sidecar)", best_val);
    println!("{best_val:.4}");
```

(Remove the old unconditional `varmap.save` at the end — we now save the *best*
checkpoint during training.)

- [ ] **Step 4: Build the bin**

Run: `cargo build --release --bin train_semantic 2>&1 | tail -6`
Expected: `Finished release`. Fix compile errors.

- [ ] **Step 5: Smoke-train (tiny, CPU, proxy stopped) to verify val-CE + sidecar**

```bash
# stop proxy to free RAM
powershell.exe -NoProfile -Command "try { Stop-Process -Id (Get-NetTCPConnection -LocalPort 3000 -State Listen).OwningProcess -Force } catch {}"
AXIOM_DMODEL=128 AXIOM_NLAYERS=2 AXIOM_MAX_TOKENS=6000 AXIOM_EPOCHS=4 AXIOM_STEP_CAP=200 \
  ./target/release/train_semantic 2>&1 | tail -10
ls -la checkpoints/axiom_production_bpe.meta.json
```
Expected: per-epoch `val_ce=` lines, `early stop` or `BEST val_ce=`, and a
`*.meta.json` sidecar written.

- [ ] **Step 6: Commit**

```bash
git add src/bin/train_semantic.rs
git commit -m "feat(train_semantic): val split + early-stop + auto-size + OOM-safe steps + sidecar"
```

---

## Task 6: `eval_model` — acceptance suite + recalibrated drift gate

**Files:**
- Create: `src/bin/eval_model.rs`
- Test: this bin's smoke run is the acceptance check.

- [ ] **Step 1: Write the eval bin**

Create `src/bin/eval_model.rs`:
```rust
//! eval_model — acceptance suite for a baked checkpoint. Loads dims from the
//! sidecar, then reports: held-out perplexity, clean-vs-anomaly drift margin,
//! and a recalibrated AXIOM_DRIFT_THRESHOLD (written to checkpoints/axiom_drift_gate.txt).
//!
//! Run: cargo run --release --bin eval_model

use std::path::PathBuf;

use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::{InferencePipeline, InferenceRuntimeOptions};
use axiom_engine::model_meta::ModelMeta;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

const CHUNK: usize = 512;

fn chunked_ce(pipeline: &InferencePipeline, ids: &[u32], vocab: usize) -> f32 {
    if ids.len() < 2 { return f32::INFINITY; }
    let dev = pipeline.device();
    let mut total = 0.0f32; let mut n = 0usize;
    for w in ids.chunks(CHUNK) {
        if w.len() < 2 { continue; }
        let m = w.len();
        let mut states = pipeline.init_session_states().unwrap();
        let input = Tensor::from_vec(w[..m-1].to_vec(), (1, m-1), dev).unwrap();
        let logits = pipeline.model().forward_lm(&input, &mut states).unwrap();
        let l2d = logits.squeeze(0).unwrap().reshape((m-1, vocab)).unwrap();
        let tgt = Tensor::from_vec(w[1..].to_vec(), (m-1,), dev).unwrap();
        total += candle_nn::loss::cross_entropy(&l2d, &tgt).unwrap().to_scalar::<f32>().unwrap() * (m-1) as f32;
        for s in states.iter_mut() { *s = s.detach(); }
        n += m-1;
    }
    if n == 0 { f32::INFINITY } else { total / n as f32 }
}

fn main() {
    std::thread::Builder::new().stack_size(1024*1024*1024).spawn(run).unwrap().join().unwrap();
}

fn run() {
    let root = repo_root();
    let ckpt = std::env::var("AXIOM_BPE_CKPT")
        .unwrap_or_else(|_| root.join("checkpoints/axiom_production_bpe.bin").to_string_lossy().into());
    let meta = ModelMeta::load(&ckpt).expect("sidecar .meta.json (run train_semantic first)");
    let bpe = meta.tokenizer.clone();
    let vocab = meta.vocab_size;
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let config = AxiomConfig { d_model: meta.d_model, n_layers: meta.n_layers,
        vocab_size: vocab, lr_inner: meta.lr_inner, norm_eps: meta.norm_eps };
    let runtime = InferenceRuntimeOptions { tokenizer_path: Some(bpe.clone()), ..Default::default() };
    let pipeline = InferencePipeline::with_checkpoint_and_options(config, device, &ckpt, runtime)
        .expect("load pipeline");

    let read = |rel: &str| std::fs::read_to_string(root.join(rel)).unwrap_or_default();
    let enc = |t: &str| pipeline.encode_text(t);

    // Held-out perplexity proxy: a clean repo file the model didn't memorize wholesale.
    let held = chunked_ce(&pipeline, &enc(&read("axiom_engine_rs/src/server.rs")), vocab);
    // Drift separation: clean vs anomaly.
    let clean: Vec<f32> = ["axiom_engine_rs/src/model.rs", "axiom_engine_rs/src/inference.rs"]
        .iter().map(|f| chunked_ce(&pipeline, &enc(&read(f)), vocab)).collect();
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
    eprintln!("[eval] ACCEPTANCE: {}", if pass { "PASS ✓" } else { "FAIL ✗" });
    if pass {
        let gate_file = root.join("checkpoints/axiom_drift_gate.txt");
        std::fs::write(&gate_file, format!("{gate:.4}")).ok();
        eprintln!("[eval] recalibrated AXIOM_DRIFT_THRESHOLD={gate:.4} → {}", gate_file.display());
    }
    println!("{}", if pass { "PASS" } else { "FAIL" });
}
```

- [ ] **Step 2: Build the bin**

Run: `cargo build --release --bin eval_model 2>&1 | tail -5`
Expected: `Finished release`.

- [ ] **Step 3: Run the eval against the smoke checkpoint**

Run: `./target/release/eval_model 2>&1 | tail -10`
Expected: prints held-out CE, clean/anomaly CE, margin, ACCEPTANCE line, and
(if PASS) writes `checkpoints/axiom_drift_gate.txt`.

- [ ] **Step 4: Commit**

```bash
git add src/bin/eval_model.rs
git commit -m "feat(eval_model): acceptance suite (held-out CE, drift margin, recalibrated gate)"
```

---

## Task 7: sidecar-driven proxy load + recalibrated gate

**Files:**
- Modify: `src/main.rs` (`resolve_production_model`)
- Modify: `start_axiom.sh`

- [ ] **Step 1: Read dims from the sidecar in `resolve_production_model`**

In `src/main.rs`, add `use axiom_engine::model_meta::ModelMeta;` near the top imports
(the bin re-declares modules; add `mod model_meta;` alongside the others, OR use
the lib path `axiom_engine::model_meta` — prefer the lib path for bins). Then in
`resolve_production_model`, after confirming the tokenizer loads, replace the
hardcoded config with sidecar-driven dims:
```rust
        Ok(tok) => {
            let vocab = tok.get_vocab_size(true);
            std::env::set_var("AXIOM_TOKENIZER", &bpe);
            // Prefer dims from the checkpoint sidecar; fall back to 256/4.
            let (d_model, n_layers, lr_inner, norm_eps) = match ModelMeta::load(&ckpt) {
                Some(m) => (m.d_model, m.n_layers, m.lr_inner, m.norm_eps),
                None => (256, 4, 1e-3, 1e-6),
            };
            eprintln!("[axiom] PRODUCTION MODEL = BPE (vocab {vocab}, d_model {d_model}, n_layers {n_layers})");
            let cfg = AxiomConfig { d_model, n_layers, vocab_size: vocab, lr_inner, norm_eps };
            (cfg, ckpt)
        }
```

- [ ] **Step 2: Verify it compiles (check only, do NOT relink the running proxy)**

Run: `cargo check --bin axiom_engine 2>&1 | tail -5`
Expected: `Finished`. (Use `check`, not `build`, to avoid touching the live exe.)

- [ ] **Step 3: Have start_axiom.sh consume the recalibrated gate**

In `start_axiom.sh`, inside the BPE-activation `if` block, after setting
`AXIOM_DRIFT_THRESHOLD`, prefer the eval-written value when present:
```bash
    GATE_FILE="$REPO_ROOT/checkpoints/axiom_drift_gate.txt"
    if [ -f "$GATE_FILE" ]; then
        export AXIOM_DRIFT_THRESHOLD="$(cat "$GATE_FILE")"
    else
        export AXIOM_DRIFT_THRESHOLD="${AXIOM_DRIFT_THRESHOLD:-7.03}"
    fi
```

- [ ] **Step 4: Commit**

```bash
git add src/main.rs start_axiom.sh
git commit -m "feat(proxy): load model dims from checkpoint sidecar + eval-recalibrated drift gate"
```

---

## Task 8: full converged training run + live swap (CPU now, GPU after Task 9)

**Files:** none (operational).

- [ ] **Step 1: Stop the proxy to free RAM**

```bash
powershell.exe -NoProfile -Command "try { Stop-Process -Id (Get-NetTCPConnection -LocalPort 3000 -State Listen).OwningProcess -Force } catch {}"
```

- [ ] **Step 2: Build the full corpus + tokenizer**

```bash
AXIOM_CORPUS_MAX_MB=150 ./target/release/corpus_crawl
AXIOM_BPE_VOCAB=16000 ./target/release/train_tokenizer
```

- [ ] **Step 3: Train (bounded; resumable — rerun to accumulate)**

```bash
AXIOM_MAX_TOKENS=400000 AXIOM_EPOCHS=12 AXIOM_STEP_CAP=4000 ./target/release/train_semantic 2>&1 | tail -20
```
Expected: `val_ce` decreasing across epochs; early-stop or step cap; best checkpoint + sidecar saved. Rerun the same command to resume from the best checkpoint and train further if `val_ce` is still falling.

- [ ] **Step 4: Evaluate; only swap if PASS**

```bash
./target/release/eval_model 2>&1 | tail -12
```
Expected: `ACCEPTANCE: PASS ✓`, gate file written. If FAIL, train more (Step 3) or adjust `AXIOM_LR`/corpus, then re-eval.

- [ ] **Step 5: Relaunch the proxy on the new model**

```bash
cd /c/Users/garza/AXIOM-AETHER && unset ANTHROPIC_API_KEY && nohup bash ./start_axiom.sh > axiom_boot.log 2>&1 &
sleep 8 && grep -iE "PRODUCTION MODEL|drift_gate|listening" axiom_server.log | tail -3
curl -s -o /dev/null -w "models=%{http_code}\n" http://127.0.0.1:3000/v1/models
```
Expected: banner shows the new dims + recalibrated gate; `models=200`.

- [ ] **Step 6: Commit any artifacts notes (checkpoints are gitignored)**

```bash
git commit --allow-empty -m "chore: converged BPE model trained + live proxy swapped (artifacts gitignored)"
```

---

## Task 9 (deferred, isolated): GPU enablement

**Files:**
- Modify: `axiom_engine_rs/Cargo.toml` (candle bump)

- [ ] **Step 1: Attempt the candle CUDA-13 bump on a branch**

```bash
git checkout -b gpu-cuda13
```
In `Cargo.toml`, bump `candle-core` and `candle-nn` to the latest 0.9.x (which adds CUDA 13 support via newer cudarc) and keep the `cuda` feature mapping:
```toml
candle-core = "0.9"
candle-nn = "0.9"
```

- [ ] **Step 2: Build with the cuda feature**

Run: `cargo build --release --features cuda --bin train_semantic 2>&1 | tail -30`
Expected: `Finished release`. If it fails on candle API changes (0.8→0.9), fix the breakages (cross_entropy/affine/VarMap signatures are stable; address any that moved). If cudarc still rejects CUDA 13.1, fall back: install the CUDA 12.x toolkit and set `CUDA_PATH` / `CUDA_ROOT` to it for the build.

- [ ] **Step 3: Verify GPU training runs**

```bash
powershell.exe -NoProfile -Command "try { Stop-Process -Id (Get-NetTCPConnection -LocalPort 3000 -State Listen).OwningProcess -Force } catch {}"
AXIOM_MAX_TOKENS=400000 AXIOM_EPOCHS=12 ./target/release/train_semantic 2>&1 | tail -10
```
Expected: `[train] auto-size: budget=... → d_model=512 n_layers=8` and a step time far below the CPU ~4.3 s/step. Confirm `nvidia-smi` shows the process using VRAM.

- [ ] **Step 4: Re-eval + merge if better**

```bash
./target/release/eval_model 2>&1 | tail -12
```
If PASS and `val_ce` improved over the CPU model, merge `gpu-cuda13` to `main`:
```bash
git checkout main && git merge gpu-cuda13 && git push origin main
```

---

## Final verification

- [ ] `cargo test --lib` is green (44+ existing tests + new model_meta/corpus tests).
- [ ] Proxy boots with the new dims (from sidecar) and recalibrated gate, `:3000` → 200.
- [ ] `eval_model` reports PASS with a positive drift-separation margin.
- [ ] README "What's New" table updated with the converged model's dims + metrics.
