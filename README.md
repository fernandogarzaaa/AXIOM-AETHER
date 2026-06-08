# AXIOM-AETHER

**Axiom-TTT** is an inference engine with **online Test-Time Training** — every
token updates the model's per-layer dynamic weight matrices (W̃) in real time,
so the engine _learns from context_ during generation without a fine-tuning
pipeline.

It runs as a local-first Rust runtime with several compatible surfaces:

1. **OpenAI/Codex + Anthropic/Claude context-compression proxy** - strips heavy
   context, absorbs it into fast-weight tensors, and forwards a lean
   fingerprinted payload.
2. **Native MCP server** (`--mode mcp`) exposing `axiom_compress_path`,
   `axiom_evaluate_drift`, and `axiom_expand` tools over JSON-RPC stdio.
3. **JIT search-reasoning node** - scrapes the live web with Rust-native
   ingestion, absorbs results via online TTT, and emits a dense
   `<axiom_search_fingerprint>` semantic pointer.
4. **Closed-loop hypervisor runtime** - user-mode Neural VFS, Poly JIT repair,
   Q-TTT simulated tensor optimization, SR-TTT exact residual memory, DWE binary
   tensor deltas, and localized swarm telemetry.

[![Release Binaries](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/release.yml/badge.svg)](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/release.yml)
[![Docker](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/docker.yml/badge.svg)](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/docker.yml)

---

## Current Runtime Snapshot

Current `main` includes the Phase 7/8 runtime surfaces:

| Layer | Status |
|---|---|
| Active model | BPE production stack, `d_model=256`, `n_layers=4`, `vocab_size=16000` |
| Context proxy | Native Anthropic `/v1/messages` and OpenAI/Codex `/v1/chat/completions` compression paths |
| Persistence | On-disk `bincode` fast-weight cache under `checkpoints/memory/` |
| Local routing | Ollama-first swarm router with cloud fallback when configured |
| Hypervisor | Safe user-mode VFS loopback, Poly JIT fault capture/repair, sandbox feedback |
| Q-TTT | Bounded simulated MPS manifold with paired real/imag tensor layout `[2, branches, bond_dim]` |
| SR-TTT | Surprisal-aware dual track: normal fast-weight updates plus exact residual cache for hashes/API keys/schema tokens |
| DWE | Binary differential weight exchange using compact `bincode` tensor delta fragments |
| Swarm matrix | VFS-driven local model-domain telemetry with RTX 2060-safe VRAM accounting |
| Tests | Full Rust unit/integration matrix passes locally |

Important telemetry endpoints:

```text
GET  /v1/hypervisor/jit_status
GET  /v1/hypervisor/quantum_coherent_state
GET  /v1/swarm/matrix_state
POST /v1/hypervisor/mount
POST /v1/ttt/feedback
POST /v1/cluster/merge
```

---

## Why Axiom-TTT?

| Capability | llama.cpp / Ollama / vLLM | **Axiom-TTT** |
|---|---|---|
| Model weights frozen at serve time | ✅ | ✅ |
| Per-session dynamic weight adaptation (W̃) | ❌ | ✅ |
| In-context learning that _persists_ across turns | ❌ | ✅ |
| One-call corpus adaptation (`/v1/adapt`) | ❌ | ✅ |
| **Semantic BPE tokenizer + scaled model** | — | ✅ |
| **Native MCP tool server** | ❌ | ✅ |
| **Live web → TTT search node** | ❌ | ✅ |
| OpenAI / Anthropic-compatible API | ✅ | ✅ |
| Subscription (OAuth) passthrough — no API key needed | ❌ | ✅ |

---

## What's New — Semantic v2

The engine moved from a 256-bucket character-hash tokenizer + tiny toy model to
a **real semantic stack**:

| Area | Before | Now |
|---|---|---|
| Tokenizer | SHA-256 hash → 256 buckets | **ByteLevel BPE**, vocab ~5k (`scripts`/`train_tokenizer`) |
| Compression | n/a | **~3.7 chars / token** |
| Model | d_model=64, 2 layers | **d_model=256, 4 layers** |
| Cross-entropy signal | could not separate clean vs anomalous code | **separates** (clean ℒ ≤ 6.7, anomaly ℒ 7.3; drift gate **7.03**) |
| Autograd safety | deep graph → stack overflow | per-window `.detach()`, strict ≤512-token chunks, large-stack training threads |
| Device | CPU/CUDA/Metal (manual) | adds `--device auto` (`cuda_if_available` → CPU fallback) |

New surfaces: **native MCP server**, **persistent "vibe memory"** + `axiom-wipe`,
a **hybrid pre-commit quality gate** (`axiom-guard`), and a **JIT search node**.

### Update — Semantic v3 (converged, validated, auto-deployed)

A pipeline-first upgrade to a *properly converged* model (see
`docs/superpowers/specs/2026-05-31-smarter-core-model-design.md` and the plan):

| Area | v3 |
|---|---|
| Tokenizer | ByteLevel BPE retrained on a crawled multi-language corpus, **vocab 16,000** |
| Training | `train_semantic` now does a **95/5 train/val split with early-stopping on held-out CE** (not memorization), **auto-sizes** the model to free VRAM, and writes a **`*.meta.json` sidecar** (dims/vocab/val_ce) |
| Converged model | **d_model=256, n_layers=4, val_ce ≈ 3.18** (held-out, 90k-token / 40 MB-corpus run, best-checkpointed at the val minimum) — live in the proxy, dims auto-loaded from the sidecar |
| Drift separation | clean ℒ **4.12–4.20** vs anomaly ℒ **7.23** → **margin +3.04** (vs +0.58 for the old memorized model); held-out code CE **3.47** (down from 5.52 at d128 — the larger model *generalizes* far better); recalibrated gate **5.72** (`eval_model` → `axiom_drift_gate.txt`) |
| New tooling | `corpus_crawl` (on-disk deduped corpus), `eval_model` (acceptance suite), `model_meta`/`corpus` lib modules |
| Deploy | `start_axiom.sh` auto-activates the BPE model + reads dims from the sidecar + the recalibrated gate |

> **Hardware note:** d256/4L is the live model. The GPU training path **works on
> CUDA 12.6** (a d384/6L run trained on an RTX 2060 at ~12 s/step). `candle 0.9 /
> CUDA-13` remains blocked upstream — `cudarc 0.13` does not support CUDA 13.x —
> so CUDA 12.6 is the supported GPU toolchain.

### Update — Semantic v4 (compression that actually works, and is Claude-readable)

The compression path had two latent problems; both are fixed, and the result is
verified end-to-end through the proxy:

| Area | v4 |
|---|---|
| **Compression now fires** | The `/v1/messages` compression ran the TTT model on a tokio worker whose default ~2 MB stack overflowed on candle's deep backward recursion — silently closing the connection, so `heavy_msgs` was **always 0**. The runtime now builds workers with a **256 MB stack**; real heavy context compresses for the first time. |
| **Claude-readable digest** | The opaque neural fingerprint (vocab indices + Frobenius norms) is meaningless to a *different* model, so compressing real code made answers worse. Axiom now ships a **structural skeleton** — doc summary + imports + declaration signatures, bodies elided — that Claude can actually read. Axiom's TTT capability is untouched (the session still absorbs the context); the drift signal (`recall_norm` + `state_hash`) rides along as digest attributes. **~80 % smaller on the wire, still answerable.** (`src/skeleton.rs`) |
| **Multi-language + prose-safe** | The skeletonizer covers Rust/Go/Python/JS-TS/Java/C#, detects brace-language methods with no leading keyword, excludes control-flow headers, and falls back to a head+tail **prose excerpt** for non-code so plain text is never erased. |
| **Hardware auto-optimization** | `src/hardware.rs` + `--mode doctor`: detects GPU/VRAM/CPU/RAM and recommends safe per-role devices. A **co-tenancy guard** keeps the proxy on CPU whenever a training job holds the GPU — the fix for VRAM OOM contention on small cards. `--device auto` honours it. |

---

## Quick Start

### Zero-install binary (pre-built releases)

```bash
# Linux x86-64
curl -LO https://github.com/fernandogarzaaa/AXIOM-AETHER/releases/latest/download/axiom-ttt-<version>-linux-x86_64.tar.gz
tar -xzf axiom-ttt-*.tar.gz
./axiom-ttt-*/axiom_engine --mode server
```

### Docker (multi-arch: linux/amd64 + linux/arm64)

```bash
docker run -p 8080:8080 ghcr.io/fernandogarzaaa/axiom-aether:latest
docker compose up   # alternative
```

### From source (Rust 1.78+)

```bash
git clone https://github.com/fernandogarzaaa/AXIOM-AETHER
cd AXIOM-AETHER/axiom_engine_rs
cargo build --release
./target/release/axiom_engine --mode server --device auto
```

---

## Local Claude Code Integration

Wire Axiom-TTT as the **local inference + context-compression layer for Claude
Code**. Each request is routed through the proxy, which strips heavy context,
trains it into fast-weight tensors (W̃), and forwards a lean fingerprinted
payload upstream — shrinking billed input tokens while keeping the effective
context window large.

**No API key required for Claude Pro/Max.** The proxy runs in **PASSTHROUGH
mode**: it holds no key of its own and relays each client's own
`Authorization` / `x-api-key` headers upstream — the correct mode for a Claude
subscription (OAuth via Claude Code).

> **Platform note:** instructions target Windows + Git Bash. Linux/macOS users
> adapt the auto-start step to a systemd user unit / launchd plist; the rest is
> identical.

### Step 1 — Build the release binary

```bash
cd AXIOM-AETHER/axiom_engine_rs
cargo build --release    # → target/release/axiom_engine(.exe)
```

### Step 2 — Build the semantic model (BPE tokenizer + checkpoint)

The engine now uses a real **ByteLevel BPE** tokenizer and a scaled model. Two
small binaries build them locally (no network, no HF Hub):

```bash
# (a) Train a ByteLevel BPE on your code corpus → checkpoints/axiom_bpe.json
AXIOM_BPE_VOCAB=8000 cargo run --release --bin train_tokenizer

# (b) Train + bake the scaled production checkpoint (resumable; loads + continues
#     from an existing checkpoint). d_model=256, n_layers=4, vocab from the BPE.
AXIOM_EPOCHS=20 AXIOM_MAX_TOKENS=12000 cargo run --release --bin train_semantic
# → checkpoints/axiom_production_bpe.bin
```

`train_semantic` prints the per-epoch loss; a converged run settles into the
**2.0–4.0** range (and lower on a small corpus). It runs on a large-stack thread
and trains in strict ≤512-token detached windows to bound VRAM/RAM.

| Env (train_semantic) | Default | Purpose |
|---|---|---|
| `AXIOM_DMODEL` / `AXIOM_NLAYERS` | 256 / 4 | model dims |
| `AXIOM_EPOCHS` / `AXIOM_STEP_CAP` | 8 / 900 | bounded training budget |
| `AXIOM_TRAIN_WIN` | 128 | backprop window (≤512) |
| `AXIOM_MAX_TOKENS` | 12000 | corpus cap (faster CPU convergence) |
| `AXIOM_LR` | 3e-3 | AdamW learning rate |

> Legacy path: the original `harvest` binary (256-hash, d_model=64) still exists
> for the old `axiom_production.bin`. The proxy uses the BPE model whenever its
> artifacts are present (see Step 3).

### Step 3 — Boot the proxy

```bash
./start_axiom.sh            # binds 127.0.0.1:3000, upstream = real Anthropic API
```

The script:
- Runs in **PASSTHROUGH** mode (no key needed for a Claude subscription).
- **Auto-activates the BPE semantic model** (d_model=256, n_layers=4, drift gate
  read from `checkpoints/axiom_drift_gate.txt` — currently **5.72**, falling back to
  `AXIOM_DRIFT_THRESHOLD=7.03` if the file is absent) when
  `checkpoints/axiom_production_bpe.bin` + `axiom_bpe.json` exist; otherwise falls
  back to the legacy model.
- Enables compression (`AXIOM_TTT_COMPRESS=1`) and tees logs to
  `axiom_server.log`.

Boot banner confirms the active model:
```
[start_axiom] Production model: BPE semantic (d_model=256, n_layers=4, drift_gate=5.72)
[axiom] PRODUCTION MODEL = BPE (vocab 16000, d_model 256, n_layers 4)
[+] Axiom-TTT server listening on http://127.0.0.1:3000
```

### Step 4 — Auto-start at logon (Windows, no admin)

```vbs
' %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\AxiomTTTProxy.vbs
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """C:\Program Files\Git\bin\bash.exe"" -lc " & _
  """cd /c/Users/YOUR_USERNAME/AXIOM-AETHER && " & _
  "./start_axiom.sh >> /c/Users/YOUR_USERNAME/AXIOM-AETHER/axiom_boot.log 2>&1""", 0, False
```

### Step 5 — Route Claude Code through the proxy

**Option A — Per-shell opt-in (safest while evaluating):**
```bash
source ./axiom.env
claude "your prompt here"   # routes through Axiom on 127.0.0.1:3000
```

**Option B — Global default** — add an `env` block to `~/.claude/settings.json`:
```json
{ "env": { "ANTHROPIC_BASE_URL": "http://127.0.0.1:3000" } }
```
> ⚠️ If the proxy is down when Claude Code starts, requests fail on port 3000.
> Use Option A until auto-start is confirmed across reboots.

### Step 6 — Verify token savings (non-billable)

```bash
node scripts/token_savings_test.js   # proxy must be running on :3000
```

### Revert to direct Anthropic routing

Remove the `env` block from `~/.claude/settings.json` (or restore the `.bak`).

---

## Native MCP Server (`--mode mcp`)

Exposes Axiom to a host LLM as a first-class tool provider over JSON-RPC 2.0
stdio (stdout is pure protocol; all logs go to stderr).

```jsonc
// ~/.claude/settings.json → "mcpServers"
"axiom": {
  "command": "C:\\path\\to\\axiom_engine_rs\\target\\release\\axiom_engine.exe",
  "args": ["--mode", "mcp", "--checkpoint", "checkpoints/axiom_production_bpe.bin"]
}
```

| Tool | Input | Behaviour |
|---|---|---|
| `axiom_compress_path` | `path` | absorbs a dir/file through local TTT, returns the `<axiom_context_fingerprint>` block, and commits the session into the persistent master vibe |
| `axiom_evaluate_drift` | `code_content` | cross-entropy of the code vs current fast-weights; a loss spike past `AXIOM_DRIFT_THRESHOLD` (default **7.03**) returns `isError: true` |
| `axiom_expand` | `symbol`, `session_id` | the retrieval half of the skeleton round-trip: returns the **full source body** of a symbol that compression dropped from an `<axiom_context_digest>`. HTTP-calls the proxy's `POST /v1/expand` (`AXIOM_PROXY_URL`, default `127.0.0.1:3000`) |

---

## Persistent "Vibe Memory" + `axiom-wipe`

Adapted session W̃ matrices are EMA-merged into a master tensor
(`axiom_master_vibe.bin`) on session drop / clear / graceful shutdown, so the
engine accumulates "codebase DNA" over time. Persist-only by default; opt-in
priming via `AXIOM_VIBE_PRIME=1`. Disable with `AXIOM_VIBE=0`.

```bash
scripts/axiom-wipe.sh --list      # show backups + live vibe (size + timestamp)
scripts/axiom-wipe.sh             # backup → delete → restart proxy (clean reset)
scripts/axiom-wipe.sh --restore   # restore newest backup, then restart
```

---

## Hybrid Pre-Commit Quality Gate (`axiom-guard`)

A two-phase commit gate (`scripts/axiom-guard.sh`):

1. **Deterministic structural pre-filter (STRICT, blocks):** scans staged `.rs`
   for `unsafe` (outside whitelisted low-level layers), `static mut`, goto-style
   state machines, and extreme nesting — see `scripts/lib/axiom-structural.js`.
2. **TTT semantic pass (ADVISORY, warn-only):** flags files whose cross-entropy
   exceeds the repo-calibrated threshold; never blocks.

```bash
scripts/axiom-guard.sh --install      # install .git/hooks/pre-commit
scripts/axiom-guard.sh --check FILE…  # ad-hoc check
git commit --no-verify                # bypass for one commit
```

---

## JIT Search-Reasoning Node

Scrape the live web, absorb it into local fast-weights via online TTT, and emit
a dense semantic pointer — no external LLM call for the reasoning step.

```bash
cargo build --release --bin search_node
./target/release/search_node "test-time training fast weights"
```

Pipeline: Rust-native `reqwest` + `scraper` search fetch and HTML cleanup
-> BPE tokenize -> online TTT over detached <=512-token chunks
-> `<axiom_search_fingerprint>` with `recall_top_k_topics` and a
`recall_norm` confidence signal. Logic lives in `src/search_ingest.rs` and
`src/search_scrape.rs`.

---

## API Reference

Drop-in OpenAI Chat Completions + Anthropic Messages, plus TTT-specific session
endpoints.

```
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
POST /v1/messages                 (Anthropic Messages API - compression path)
POST /v1/expand                   (expand a dropped symbol body: {session_id, symbol})
POST /v1/sessions                 (create persistent fast-weight session)
POST /v1/adapt                    (in-place TTT adaptation over a corpus)
POST /v1/ttt/feedback             (compiler/runtime feedback -> persistent TTT cache)
POST /v1/cluster/sync             (cluster state synchronization)
POST /v1/cluster/merge            (merge persisted fast-weight checkpoints)
POST /v1/hypervisor/mount         (safe user-mode Neural VFS mount + warm paths)
GET  /v1/hypervisor/jit_status
GET  /v1/hypervisor/quantum_coherent_state
GET  /v1/swarm/matrix_state
GET  /v1/sessions/{id}/checkpoint  PUT .../checkpoint   DELETE /v1/sessions/{id}
GET  /v1/ttt/sessions   DELETE /v1/ttt/sessions        GET /metrics
```

**Example — in-place adaptation:**
```bash
SESSION=$(curl -s -X POST http://localhost:8080/v1/sessions -d '{}' | jq -r .session_id)
curl -X POST http://localhost:8080/v1/adapt -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SESSION\",\"corpus\":[\"candle is a minimalist ML framework for Rust.\"]}"
```
After `/v1/adapt`, generation with that `session_id` reflects the adapted
context — **without retrieval at inference time**.

---

## Hardware Support

| Device | Flag | Requirements |
|---|---|---|
| Auto (CUDA→CPU) | `--device auto` | picks CUDA if available, else CPU — never errors |
| CPU | `--device cpu` (default) | any x86-64 / ARM64 |
| NVIDIA CUDA | `--device cuda` | CUDA toolkit + `cargo build --features cuda` |
| Apple Metal | `--device metal` | macOS 13+ / Apple Silicon + `--features metal` |

---

## CLI Reference

```
axiom_engine --mode <MODE> [OPTIONS]

Modes:  train | generate | server | mcp | lsp | meta-train | doctor

Options:
  --device auto|cpu|cuda|metal   Compute device (default: cpu; 'auto' = CUDA-if-available)
  --checkpoint PATH              Load/save weights
  --tokenizer PATH               BPE/HF tokenizer.json
  --host HOST  --port PORT       Server bind (default 0.0.0.0:8080)
  --epochs N  --steps-per-epoch N
```

Dedicated tool binaries: `train_tokenizer`, `train_semantic`, `search_node`,
`harvest` (legacy).

Key env: `AXIOM_PRODUCTION_BPE`, `AXIOM_TOKENIZER`, `AXIOM_BPE_CKPT`,
`AXIOM_DRIFT_THRESHOLD`, `AXIOM_VIBE` / `AXIOM_VIBE_PRIME`, `AXIOM_TTT_COMPRESS`.

---

## Engine Modules (`axiom_engine_rs/src/`)

| Concern | File |
|---|---|
| Config | `config.rs` |
| TTT block (W̃ update rule) | `ttt_block.rs` |
| Model (embedding → N×TTT → LM head) | `model.rs` |
| Kernels / RMSNorm | `kernel.rs`, `chunk_kernel.rs` |
| Inference pipeline + tokenizer | `inference.rs` |
| Context compression | `context_compressor.rs` |
| Claude-readable digest (skeleton) | `skeleton.rs` |
| Hardware profile + co-tenancy guard | `hardware.rs` |
| Anthropic forwarder (passthrough) | `anthropic_forwarder.rs` |
| Native MCP server | `mcp_stdio.rs` |
| Persistent vibe memory (EMA) | `vibe_memory.rs` |
| JIT search ingestion | `search_ingest.rs` |
| Rust-native search scraping | `search_scrape.rs` |
| Local Ollama router | `swarm_router.rs` |
| Localized swarm telemetry | `swarm_route.rs` |
| User-mode Neural VFS | `vfs.rs` |
| Poly JIT fault recovery | `poly_jit.rs` |
| Sandbox feedback runner | `sandbox.rs` |
| Q-TTT manifold + Hamiltonian optimizer | `q_manifold.rs`, `hamiltonian.rs` |
| Surprisal residual cache | `surprisal.rs` |
| Differential weight exchange | `dwe.rs` |
| LSP daemon | `lsp.rs` |
| Weight checkpoint merge | `weight_merge.rs` |
| HTTP API server | `server.rs` |
| Meta-training | `meta_train.rs` |
| CLI entry-point | `main.rs` |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the architecture diagram, TTT layer
mathematics, and module map.
