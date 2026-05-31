# Smarter Core Model — Design Spec

**Date:** 2026-05-31
**Status:** Approved (brainstorming → ready for implementation plan)
**Scope:** A single, focused upgrade — replace Axiom's small, memorized model with a
properly-converged, multi-language **code-specialist** model, auto-sized to the local
GPU, GPU-first at runtime with CPU fallback, trained on a real corpus, and accepted on
held-out cross-entropy + downstream evals. Built **pipeline-first** so progress never
blocks on the CUDA build.

---

## 1. Background & Problem

Current production model (committed): ByteLevel BPE vocab≈5068, `d_model=256`,
`n_layers=4`, baked from ~4,000 tokens of repo Rust → final train loss **0.78**, which
was **memorization** of a tiny corpus, not generalization. It separates the hand-built
anomaly from clean code, but its competence is narrow and the convergence number is not
trustworthy as a quality signal.

Goal: a genuinely smarter base model that lifts every downstream surface at once —
context-compression fidelity (the proxy), the drift/anomaly gate, "codebase DNA," and
the JIT search node's recall.

## 2. Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Direction | Smarter core model | Highest-leverage; improves compression, drift, recall together. |
| Compute | **GPU-first, CPU fallback** | RTX 2060 is the only way to train a bigger model in reasonable time; CPU stays for small-config validation. |
| Competence | **Multi-language code specialist** | Cleanest convergence on 6 GB; maximizes core proxy/drift/DNA value. |
| Corpus | **Hybrid**: local crawl + small curated download | Local cargo registry is a large free Rust corpus; download balances other languages. |
| Model size | **Auto-size to VRAM** (graceful OOM shrink) | Adapts to whatever the box can handle that session. |
| Acceptance | **Held-out CE + downstream evals** | Held-out CE prevents overfit; downstream evals prove real gains. |
| Build order | **Pipeline-first, GPU last** (Approach A) | The CUDA-13 build is the riskiest item; everything else is CPU-validatable and valuable regardless. |

## 3. Detected Hardware Profile (ground truth, 2026-05-31)

| Resource | Detected | Design impact |
|---|---|---|
| GPU | RTX 2060, 6144 MiB total, **5460 MiB free**, driver 596.36 | Auto-size VRAM budget ≈ **3.7 GB** (≈30% headroom of free) → ceiling ≈ `d512/8L/32k` (~50M params). |
| CUDA | toolkit **13.1**, driver supports 13.2 | candle 0.8 (cudarc ~0.12) **does not support CUDA 13.x** → GPU build needs candle 0.9+ *or* a side-by-side CUDA 12.x toolkit. Confirms deferring GPU. |
| CPU | i5-9300H, **4c/8t**, 2.4 GHz mobile | ~4.3 s/step on CPU → CPU is validation-only for the big model. |
| RAM | 20 GB total, **~7.3 GB free** | Binding constraint. Corpus disk-streamed; one heavy job at a time; **stop the proxy during heavy training**. |
| Corpus on disk | cargo registry **276 crates / 7,400+ `.rs`**; Python site-packages present; node_modules minimal | Local corpus is Rust-dominant → small download adds JS/TS/Go; Python from site-packages. |

## 4. Components

Extend existing code; do not rewrite.

| Component | New/changed | Purpose |
|---|---|---|
| `src/bin/corpus_crawl.rs` | **new** | Crawl local code (`~/.cargo/registry/src`, Python `site-packages`, `node_modules`, user projects) + a small curated download; clean, content-hash dedup, language-tag; write size-capped shards to `corpus/`. Streams to disk — never holds the whole corpus in RAM. |
| `src/bin/train_tokenizer.rs` | extend | Train BPE on the crawled corpus; vocab configurable (target 16k–32k). |
| `src/bin/train_semantic.rs` | **major upgrade** | Stream shards → BPE-tokenize → train/val split (~95/5) → **auto-size** config to VRAM with graceful OOM shrink → `cuda_if_available` → **early-stop on held-out CE** → bake checkpoint **+ sidecar `*.meta.json`** (dims/vocab/val-CE). Resumable. |
| `src/bin/eval_model.rs` | **new** | Acceptance suite: held-out perplexity, clean-vs-anomaly drift separation margin, compression ratio / recall_norm. Emits a report. |
| `resolve_production_model` (`src/main.rs`) | change | Read model dims from the checkpoint **sidecar** instead of hardcoding `256/4`, so any baked size loads in the live proxy. |
| `start_axiom.sh` | change | Write the eval-recalibrated `AXIOM_DRIFT_THRESHOLD` instead of the hardcoded 7.03. |
| candle CUDA build | **deferred (last)** | Resolve CUDA-13 compat (prefer candle bump); CPU fallback always works. |

**Linchpin:** the sidecar metadata file decouples model dimensions from hardcoded
constants, making "auto-size to VRAM" deployable end-to-end (trainer picks dims →
records them → proxy/eval read them).

## 5. Data Flow

```
corpus_crawl ──► corpus/ shards (deduped, lang-tagged, size-capped, on disk)
      │              ├─► train_tokenizer ──► checkpoints/axiom_bpe.json  (vocab 16k–32k)
      └──────────────┴─► train_semantic
                              ├─ stream shards → BPE-tokenize → train/val split (~95/5 by shard)
                              ├─ auto-size config → train (early-stop on val CE)
                              └─► axiom_production_bpe.bin  +  axiom_production_bpe.meta.json
                                        ├─► eval_model ──► acceptance report (+ recalibrated gate)
                                        └─► live proxy auto-loads (dims from .meta.json)
```

## 6. Auto-Sizing to VRAM

- On train start, probe the device: CUDA → query free VRAM (cudarc); CPU → conservative RAM budget.
- Pick the largest config from a tier ladder that fits with ~30% headroom. Footprint estimate:
  `4 × params (AdamW moments + grad, fp32) + activation budget for one 512-token window`.
  Ladder e.g. `d384/6L/16k → d512/8L/32k → d640/8L/32k`, ceiling ≈ 3.7 GB on this GPU.
- **Graceful OOM shrink:** wrap each train step; on allocation failure, step down one rung
  (or halve window/batch), reload, continue — never crash the box. Builds on the proven
  `.detach()` + ≤512-token-chunk + large-stack-thread pattern.

## 7. Memory Discipline (20 GB RAM / 6 GB VRAM, volatile; ~7.3 GB RAM free)

- Corpus lives on **disk**, streamed shard-by-shard — RAM holds only the current shard.
- Hard token budget + per-file size cap during crawl; **content-hash dedup** (cargo registry has heavy duplication).
- Training stays in detached ≤512-token windows on a large-stack thread.
- **One heavy job at a time** (concurrent builds + train caused OOM earlier); eval runs after the bake, not during.
- **Stop the proxy** during heavy training passes; restart after. Resumable trainer accumulates across bounded sessions.

## 8. Convergence & Acceptance

- Train/val split ~95/5 by shard; **early-stop** on held-out CE (patience ~3 evals).
- `eval_model` acceptance bar:
  1. Held-out perplexity below a target threshold.
  2. **Drift separation margin** — clean-code CE < anomaly CE with a real gap; recalibrate
     `AXIOM_DRIFT_THRESHOLD` from the clean/anomaly distributions (replacing the old 7.03).
  3. Compression ratio / recall_norm sane.
- The live proxy swaps to the new model **only if** the bar is met.

## 9. GPU Enablement (deferred, last)

- Prefer **bumping candle 0.8 → a CUDA-13-capable release** behind the `cuda` feature
  (cleaner than a second toolkit install); validate one GPU train step end-to-end.
- Fallback if the bump ripples too far: install CUDA 12.x toolkit and point the build at it.
- CPU remains the always-works path; `--device auto` already selects CUDA when present.

## 10. Testing & Integration

- Each new bin gets a tiny-config **CPU smoke test**; `cargo test` stays green.
- `eval_model` is the acceptance test.
- Sidecar `*.meta.json` → `resolve_production_model` reads dims → live proxy auto-loads.
- Recalibrated drift gate written into `start_axiom.sh`.
- **Commit per milestone:** crawler → tokenizer scale → trainer upgrade → eval → sidecar/proxy → GPU.

## 11. Milestones / Sequencing (pipeline-first)

1. `corpus_crawl` (local + small download, disk shards, dedup) + smoke test → commit.
2. `train_tokenizer` on crawled corpus, vocab 16k–32k → commit.
3. `train_semantic` upgrade: shard streaming, train/val split, auto-size, OOM shrink, early-stop, sidecar → commit.
4. `eval_model` acceptance suite → commit.
5. Sidecar-driven `resolve_production_model` + `start_axiom.sh` recalibrated gate → commit.
6. Train to convergence (CPU small-config validation first, then GPU once enabled).
7. GPU enablement (candle bump) → commit. Swap live proxy when acceptance bar met.

## 12. Risks

- **CUDA 13 vs candle 0.8** (highest): mitigated by deferral + candle-bump-or-toolkit fallback; CPU path unaffected.
- **RAM pressure** (~7.3 GB free): mitigated by disk streaming, stopping the proxy during training, one-job-at-a-time.
- **CPU convergence time**: GPU is the real path; CPU only validates small configs.
- **Corpus skew toward Rust**: small curated download balances languages.

## 13. Out of Scope (YAGNI)

- General natural-language competence (chose code-specialist).
- Distributed/multi-machine training (single box).
- Serving-time quantization, throughput dashboards (separate "speed" track).
- Reasoning-agent / long-term-memory upgrades (separate future tracks).
