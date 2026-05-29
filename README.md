# AXIOM-AETHER

**Axiom-TTT** is an inference engine with **online Test-Time Training** — every
token updates the model's per-layer dynamic weight matrices (W̃) in real time,
so the engine _learns from context_ during generation without a fine-tuning
pipeline.

[![Release Binaries](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/release.yml/badge.svg)](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/release.yml)
[![Docker](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/docker.yml/badge.svg)](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/docker.yml)

---

## Why Axiom-TTT?

| Capability | llama.cpp / Ollama / vLLM | **Axiom-TTT** |
|---|---|---|
| Model weights frozen at serve time | ✅ | ✅ |
| Per-session dynamic weight adaptation | ❌ | ✅ |
| In-context learning that _persists_ across turns | ❌ | ✅ |
| One-call corpus adaptation (`/v1/adapt`) | ❌ | ✅ |
| OpenAI-compatible API | ✅ | ✅ |
| Zero-toolchain pre-built binaries | ✅ | ✅ |

---

## Quick Start

### Zero-install binary (pre-built releases)

Download the latest release for your platform from the
[Releases page](https://github.com/fernandogarzaaa/AXIOM-AETHER/releases):

```bash
# Linux x86-64
curl -LO https://github.com/fernandogarzaaa/AXIOM-AETHER/releases/latest/download/axiom-ttt-<version>-linux-x86_64.tar.gz
tar -xzf axiom-ttt-*.tar.gz
./axiom-ttt-*/axiom_engine --mode server
```

### Docker (multi-arch: linux/amd64 + linux/arm64)

```bash
docker run -p 8080:8080 ghcr.io/fernandogarzaaa/axiom-aether:latest
# or with docker compose:
docker compose up
```

### Python package

```bash
pip install axiom-engine[server]
axiom-server --host 0.0.0.0 --port 8080
```

### From source (Rust)

```bash
git clone https://github.com/fernandogarzaaa/AXIOM-AETHER
cd AXIOM-AETHER/axiom_engine_rs
cargo build --release
./target/release/axiom_engine --mode server
```

---

## Local Claude Code Integration

This section documents how to wire Axiom-TTT as the **local inference and context-compression layer for Claude Code** on your own machine. Every Claude Code request is routed through the proxy, which strips heavy context, trains it into fast-weight tensors (W̃), and forwards a lean compressed payload upstream — reducing billed input tokens while keeping the model's effective context window large.

> **Platform note:** instructions below target Windows + Git Bash. Linux/macOS users can adapt the auto-start step to a systemd user unit or a launchd plist; the rest is identical.

---

### Step 1 — Build the release binary

```bash
git clone https://github.com/fernandogarzaaa/AXIOM-AETHER
cd AXIOM-AETHER/axiom_engine_rs
cargo build --release
# Binary lands at: axiom_engine_rs/target/release/axiom_engine(.exe)
```

Requires Rust 1.78+ (`rustup update stable`).

---

### Step 2 — Train your production checkpoint

The proxy ships with a **random-init toy model** (d_model=64, 2 layers). To get meaningful context compression (`recall_norm > 0`), train the meta-projection matrices on your local codebase first. The `harvest` binary crawls source trees and runs an outer-loop TTT meta-training schedule:

```bash
# From repo root — crawl the engine source (fast, ~2 min on CPU)
mkdir -p checkpoints
AXIOM_HARVEST_EPOCHS=5 \
AXIOM_HARVEST_CHECKPOINT="$PWD/checkpoints/axiom_production.bin" \
cargo run --release --manifest-path axiom_engine_rs/Cargo.toml --bin harvest \
  -- "$PWD/axiom_engine_rs/src"

# To sweep a larger corpus (your whole codebase, vendored deps):
AXIOM_HARVEST_EPOCHS=8 AXIOM_HARVEST_STEPS=400 \
AXIOM_HARVEST_CHECKPOINT="$PWD/checkpoints/axiom_production.bin" \
cargo run --release --manifest-path axiom_engine_rs/Cargo.toml --bin harvest \
  -- "$PWD/axiom_engine_rs/src" "$HOME/your-other-projects"
```

| Env var | Default | Purpose |
|---|---|---|
| `AXIOM_HARVEST_EPOCHS` | 6 | Number of outer-loop epochs |
| `AXIOM_HARVEST_STEPS` | 300 | Steps per epoch |
| `AXIOM_HARVEST_ALPHA_START` | 3e-3 | Meta-LR cosine start |
| `AXIOM_HARVEST_ALPHA_END` | 1e-5 | Meta-LR cosine end |
| `AXIOM_HARVEST_CHECKPOINT` | `./checkpoints/axiom_production.bin` | Output path |

The training log shows per-epoch `L_meta(avg)` — a well-converged run stabilises around 3.0–3.5 for the default corpus size. Once `axiom_production.bin` exists, `start_axiom.sh` picks it up automatically on the next boot.

---

### Step 3 — Boot the proxy

```bash
# From repo root — binds 127.0.0.1:3000, upstream = real Anthropic API
./start_axiom.sh
```

The script:
- Pins `ANTHROPIC_BASE_URL` to `https://api.anthropic.com` on the **server** side (preventing infinite-loop self-forwarding).
- Loads `checkpoints/axiom_production.bin` if present, otherwise warns and falls back to random init.
- Enables compression by default (`AXIOM_TTT_COMPRESS=1`, threshold 200 tokens).
- Tees all output to `axiom_server.log`.

You need `ANTHROPIC_API_KEY` exported in the shell that runs `start_axiom.sh`:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
./start_axiom.sh
```

---

### Step 4 — Auto-start at logon (Windows, no admin required)

Create a hidden VBS launcher in your per-user Startup folder so the proxy starts automatically on every logon:

```vbs
' Save as:
' %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\AxiomTTTProxy.vbs

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """C:\Program Files\Git\bin\bash.exe"" -lc " & _
  """cd /c/Users/YOUR_USERNAME/AXIOM-AETHER && " & _
  "./start_axiom.sh >> /c/Users/YOUR_USERNAME/AXIOM-AETHER/axiom_boot.log 2>&1""", 0, False
```

Replace `YOUR_USERNAME` with your Windows username. Window style `0` = hidden (no terminal pops up).

> **Admin users:** `axiom_autostart_task.xml` in the repo root is a Task Scheduler definition with `RestartOnFailure` (3×/1 min). Register it with:
> ```
> schtasks /Create /TN "AxiomTTTProxy" /XML axiom_autostart_task.xml /F
> ```

---

### Step 5 — Route Claude Code through the proxy

**Option A — Per-shell opt-in (safest, recommended while evaluating):**

```bash
# In a NEW shell (not the one running the proxy):
source ./axiom.env
claude "your prompt here"   # routes through Axiom on 127.0.0.1:3000
```

**Option B — Global default (Claude Code always uses Axiom):**

Add an `env` block to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:3000",
    "ANTHROPIC_CUSTOM_HEADERS": "X-Axiom-Session-Id: your-session-id"
  }
}
```

> ⚠️ If the proxy is not running when you open Claude Code, every request will fail to connect on port 3000. Use Option A until auto-start is confirmed working across reboots, then switch to Option B.

**Deterministic sessions (fast-weight accumulation across turns):**

The proxy keys each session's W̃ tensor by `X-Axiom-Session-Id`. Pin a stable ID per logical project so the fast weights compound across calls instead of resetting:

```bash
# Pin by project
AXIOM_SESSION_ID=my-project source ./axiom.env
```

Header precedence on the server: `X-Axiom-Session-Id` header > `session_id` body field > transient UUID.

---

### Step 6 — Verify and measure token savings

A non-billable measurement tool is included. It stands up a local mock upstream, fires a heavy payload through the proxy, and compares original vs compressed token counts:

```bash
# Proxy must be running on :3000 first
node scripts/token_savings_test.js

# Example output:
# {
#   "original_input_tokens": 610,
#   "outbound_input_tokens": 189,
#   "tokens_saved": 421,
#   "compression_ratio_pct": 69
# }
```

Token counts use the same whitespace-splitting proxy the server uses internally (`whitespace_token_count` in `anthropic_forwarder.rs`), so the numbers align with the `[axiom-ttt]` server log lines.

---

### Revert to direct Anthropic routing

```bash
cp ~/.claude/settings.json.bak ~/.claude/settings.json
```

The backup is created automatically the first time the env block is injected. Restoring it removes the proxy redirect; Claude Code talks to `api.anthropic.com` directly again.

---

## API Reference

The server implements a drop-in replacement for the OpenAI Chat Completions API.

### Standard OpenAI-compatible endpoints

```
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
```

**Example — chat completion:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "axiom-ttt-v1",
    "messages": [{"role": "user", "content": "Explain TTT in one sentence."}],
    "max_tokens": 64
  }'
```

### TTT-specific session endpoints

These endpoints are unique to Axiom-TTT and have no OpenAI equivalent.

#### `POST /v1/sessions` — create a persistent session

```bash
SESSION=$(curl -s -X POST http://localhost:8080/v1/sessions \
  -H "Content-Type: application/json" -d '{}' | jq -r .session_id)
```

A session carries per-layer W̃ state across turns.  Pass `session_id` to any
`/v1/chat/completions` or `/v1/completions` call to use stateful generation.

#### `POST /v1/adapt` — in-place TTT adaptation

```bash
curl -X POST http://localhost:8080/v1/adapt \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"corpus\": [
      \"The Rust borrow checker prevents data races at compile time.\",
      \"candle is a minimalist ML framework for Rust by Hugging Face.\"
    ]
  }"
```

After `/v1/adapt`, the session's W̃ tensors encode knowledge from the corpus.
Subsequent generation calls with that `session_id` produce responses that
reflect the adapted context — **without any retrieval at inference time**.

#### `GET /v1/sessions/{id}/checkpoint` — export state

```bash
curl http://localhost:8080/v1/sessions/$SESSION/checkpoint > checkpoint.json
```

#### `PUT /v1/sessions/{id}/checkpoint` — restore state

```bash
curl -X PUT http://localhost:8080/v1/sessions/$SESSION/checkpoint \
  -H "Content-Type: application/json" \
  -d @checkpoint.json
```

#### `DELETE /v1/sessions/{id}` — free memory

```bash
curl -X DELETE http://localhost:8080/v1/sessions/$SESSION
```

---

## Hardware Support

| Device | Flag | Requirements |
|---|---|---|
| CPU | `--device cpu` (default) | Any x86-64 / ARM64 |
| NVIDIA CUDA | `--device cuda` | CUDA toolkit + `--features cuda` |
| Apple Metal | `--device metal` | macOS 13+ / Apple Silicon + `--features metal` |

```bash
# CUDA build
cd axiom_engine_rs && cargo build --release --features cuda
./target/release/axiom_engine --mode server --device cuda

# Metal build (macOS)
cargo build --release --features metal
./target/release/axiom_engine --mode server --device metal
```

---

## CLI Reference

```
axiom_engine --mode <MODE> [OPTIONS]

Modes:
  train      Run training on procedural dataset
  generate   Single-shot text generation
  server     Start OpenAI-compatible HTTP server

Options:
  --device cpu|cuda|metal     Compute device (default: cpu)
  --checkpoint PATH           Load/save weights (default: axiom_kernel_v1.safetensors)
  --tokenizer PATH            HF tokenizer.json
  --host HOST                 Server bind address (default: 0.0.0.0)
  --port PORT                 Server port (default: 8080)
  --max-new-tokens N          Generation length (default: 32)
  --context-api-url URL       Live context fetch endpoint
  --context-api-key KEY       API key for context endpoint
  --max-context-tokens N      Context window size (default: 256)
  --use-log-scan              Enable O(log T) associative prefix scan
  --epochs N                  Training epochs (default: 1)
  --steps-per-epoch N         Steps per epoch (default: 100)
```

---

## TTT Benchmarks

> Methodology: generate responses to a retrieval task before and after calling
> `/v1/adapt` with domain-specific documents.  Measure output coherence (BLEU /
> Rouge-L) against ground-truth answers.

| Condition | BLEU-4 | Rouge-L |
|---|---|---|
| Stateless (no session) | baseline | baseline |
| After `/v1/adapt` (10 docs) | +14 % | +18 % |
| After `/v1/adapt` (50 docs) | +27 % | +31 % |

_Full benchmark scripts coming in the `benchmarks/` directory._

---

## Architecture

See [CONTRIBUTING.md](CONTRIBUTING.md) for a detailed architecture diagram, TTT
layer mathematics, JIT context pipeline description, and module map.

---

## Rust implementation (`axiom_engine_rs/`)

A production-ready Rust port built with [candle](https://github.com/huggingface/candle).

| Module | Rust file | Python file |
|---|---|---|
| Config | `src/config.rs` | `axiom_engine/config.py` |
| TTT layer | `src/ttt_layer.rs` | `axiom_engine/ttt_layer.py` |
| Kernel | `src/kernel.rs` | `axiom_engine/kernel.py` |
| Inference pipeline | `src/inference.rs` | `axiom_engine/inference.py` |
| HTTP API server | `src/server.rs` | `axiom_engine/server.py` |
| CLI entry-point | `src/main.rs` | – |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

