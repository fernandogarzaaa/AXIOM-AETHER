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

