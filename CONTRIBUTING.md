# Contributing to Axiom-TTT

Welcome!  Axiom-TTT is an open-source inference engine that brings **online
Test-Time Training (TTT)** to production API servers.  Unlike frozen-weight
inference stacks (llama.cpp, Ollama, vLLM), Axiom-TTT can adapt its per-layer
weight matrices _during_ a conversation — without a full fine-tuning pipeline.

This document covers the architecture, TTT math, module boundaries, and
contributor workflow.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [TTT Layer Mathematics](#ttt-layer-mathematics)
3. [JIT Context Pipeline](#jit-context-pipeline)
4. [Module Map](#module-map)
5. [Development Workflow](#development-workflow)
6. [Running Tests](#running-tests)
7. [Pull Request Guidelines](#pull-request-guidelines)

---

## Architecture Overview

```
                   User Prompt
                       │
                       ▼
           ┌───────────────────────┐
           │   JIT Context Fetch   │  (Exa.ai or any search API)
           │   jit_streamer.rs     │
           └───────────┬───────────┘
                       │  context tokens
                       ▼
           ┌───────────────────────┐
           │   Prefill Phase       │  parallel over T tokens
           │   AxiomTTTEngine      │  O(T) with log-scan option
           └───────────┬───────────┘
                       │  W̃₀  (initial dynamic weight per layer)
                       ▼
           ┌───────────────────────┐
           │   Decode Phase        │  autoregressive, one token at a time
           │   TTTLinearLayer      │  W̃ₜ update per step
           └───────────┬───────────┘
                       │  generated tokens
                       ▼
                   Response
```

**Dual-cycle**: every generation starts with a parallel prefill (fast) followed
by an autoregressive decode (TTT weight update per token).

---

## TTT Layer Mathematics

The core innovation lives in `ttt_layer.rs`.  Each `TTTLinearLayer` maintains a
_dynamic weight matrix_ **W̃** ∈ ℝ^(H×D×D) (one per attention head).

### Prefill (parallel)

Given input `X ∈ ℝ^(B×T×D)`:

1. **Linear projections**: `Q = X Wq`,  `K = X Wk`,  `V = X Wv`
2. **TTT target**: minimise reconstruction loss  
   `L = ‖W_curr K − V‖²_F`  (per head, summed over T)
3. **Gradient step**:  
   `W̃ = W_curr − η · ∇_{W_curr} L`  
   where `η = lr_inner` (default 1e-3)
4. **Output**: `Y = W̃ Q`

### Decode (step-wise)

For a single token `x ∈ ℝ^(B×1×D)` and current state `W̃`:

1. Project: `q = x Wq`,  `k = x Wk`,  `v = x Wv`
2. Compute gradient:  
   `err = W̃ k − v`  
   `∇ = err ⊗ k`  (outer product, shape H×D×D)
3. Update:  `W̃' = W̃ − η · ∇`
4. Output:  `y = W̃' q`
5. Return `(y, W̃')` — the updated state is _the new session memory_

### Why this matters

The `W̃` update is a **per-session, per-token gradient step** on the keys of the
current context.  After calling `/v1/adapt` with domain-specific examples, the
session's `W̃` encodes that knowledge.  Subsequent generation queries that same
knowledge _without re-running retrieval or fine-tuning_.

### Logarithmic prefix scan

When `use_log_scan = true`, the prefill phase compresses `T` timesteps into
`O(log T)` parallel reduction depth using an associative merge operator:

```
a ⊕ b  =  a + b + a*b   ≡   (1+a)(1+b) − 1
```

This enables sub-linear-time context ingestion on long documents (>4 K tokens).

---

## JIT Context Pipeline

`jit_streamer.rs` (Rust) / `jit_streamer.py` (Python):

1. **Decompose** the user query into sub-queries (max 4).
2. **Fetch** raw text from a configurable search endpoint  
   (default: Exa.ai `POST /search`; any URL with `{query}` placeholder works).
3. **Rank** retrieved lines by query-term overlap (TF-style score).
4. **Pack** ranked lines into `max_context_tokens` token IDs using SHA-256
   byte-level hashing as a vocabulary-agnostic tokeniser.

The result is a dense context tensor injected before the user prompt during
prefill — giving the model live knowledge without requiring a fixed knowledge
base.

---

## Module Map

### Rust (`axiom_engine_rs/src/`)

| File | Role |
|---|---|
| `main.rs` | CLI entry-point: `--mode train / generate / server`, `--device cpu/cuda/metal` |
| `config.rs` | `AxiomConfig` (d_model, n_layers, num_heads, vocab_size, …) |
| `kernel.rs` | `AxiomTTTEngine` (Embedding → N×AxiomBlock → RMSNorm → LM Head) |
| `ttt_layer.rs` | `TTTLinearLayer` — prefill + decode + `forward_decode_with_loss` |
| `log_scan.rs` | `LogosAssociativeScanner::parallel_prefix_reduce` |
| `jit_streamer.rs` | Live context fetch, ranking, packing |
| `inference.rs` | `InferencePipeline`: `generate`, `generate_with_session`, `adapt_on_corpus` |
| `server.rs` | `axum`-based OpenAI-compatible HTTP API with TTT session management |
| `train.rs` | `AxiomTrainer` — AdamW training loop with procedural dataset |
| `data_gen.rs` | `ProceduralDataset` — variable-trace + logic-tree synthetic sequences |

### Python (`axiom_engine/`)

| File | Role |
|---|---|
| `__init__.py` | Package exports |
| `config.py` | `AxiomConfig` dataclass |
| `kernel.py` | PyTorch `AxiomTTTEngine` |
| `ttt_layer.py` | `TTTLinearLayer` — PyTorch implementation |
| `log_scan.py` | Logarithmic associative scan |
| `jit_streamer.py` | Async context fetch via aiohttp |
| `inference.py` | `InferencePipeline`, `AxiomInferenceRunner` |
| `server.py` | FastAPI ASGI server — OpenAI-compatible API |

---

## Development Workflow

### Prerequisites

- **Rust** ≥ 1.75 (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **Python** ≥ 3.10

### Clone and build

```bash
git clone https://github.com/fernandogarzaaa/AXIOM-AETHER
cd AXIOM-AETHER

# Rust
cd axiom_engine_rs
cargo build --release

# Python (editable install)
cd ..
pip install -e ".[server,dev]"
```

### Hardware flags

| Flag | Description |
|---|---|
| `--device cpu` | Default — runs everywhere |
| `--device cuda` | NVIDIA GPU (build with `--features cuda`) |
| `--device metal` | Apple Silicon (build with `--features metal`) |

```bash
# CUDA build
cargo build --release --features cuda

# Metal build (macOS only)
cargo build --release --features metal
```

### Running the server

```bash
# Rust server (OpenAI-compatible API on :8080)
./target/release/axiom_engine --mode server --port 8080

# Python ASGI server (same API, backed by PyTorch)
axiom-server --port 8080

# Docker
docker compose up
```

---

## Running Tests

### Rust

```bash
cd axiom_engine_rs
cargo fmt          # format
cargo test         # unit + integration tests (5 server tests, 0 non-server tests)
```

### Python

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Pull Request Guidelines

1. **One concern per PR** — keep changes focused.
2. **Run `cargo fmt` and `cargo test`** before opening a PR.
3. **Update `CONTRIBUTING.md`** if you add a new module or change the architecture.
4. **Add tests** for new server endpoints.
5. **Document public API** with doc-comments (`///` in Rust, docstrings in Python).

---

## Architecture Diagram (Text)

```
┌────────────────────────────────────────────────────────────┐
│                        API Layer                           │
│  GET /v1/models  POST /v1/completions                      │
│  POST /v1/chat/completions   POST /v1/sessions             │
│  POST /v1/adapt  GET|PUT /v1/sessions/{id}/checkpoint      │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│               InferencePipeline (inference.rs)             │
│  generate()  generate_with_session()  adapt_on_corpus()    │
│  init_session_states()                                     │
└────────────┬──────────────────────────────────┬────────────┘
             │                                  │
┌────────────▼──────────┐         ┌─────────────▼───────────┐
│  JitContextStreamer    │         │  AxiomTTTEngine          │
│  (jit_streamer.rs)    │         │  (kernel.rs)             │
│  Live context fetch   │         │  N × AxiomBlock          │
└───────────────────────┘         └──────────┬───────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │  TTTLinearLayer (ttt_layer) │
                              │  W̃ update per token         │
                              └─────────────────────────────┘
```
