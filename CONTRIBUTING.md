# Contributing to AXIOM-AETHER

AXIOM-AETHER (`axiom_engine` crate, `axiom` command) is a local-first Rust
runtime for online **Test-Time Training (TTT)**: it keeps per-session fast
weights and updates them while processing context, then uses those weights for
context compression, drift/grounding checks, and self-healing execution loops.

This document is the contributor reference: how to build and validate a fresh
clone, the module map as it actually is in the tree today, the TTT math, and PR
expectations.

---

## Table of Contents

1. [Validate a fresh clone](#validate-a-fresh-clone)
2. [Toolchain](#toolchain)
3. [Module map](#module-map)
4. [TTT layer mathematics](#ttt-layer-mathematics)
5. [Experimental surfaces](#experimental-surfaces)
6. [The proof loop](#the-proof-loop)
7. [Pull request guidelines](#pull-request-guidelines)

---

## Validate a fresh clone

These are the exact commands CI runs (`.github/workflows/ci.yml`, job `test`).
If they pass on a clean checkout, your environment is good.

```bash
git clone https://github.com/fernandogarzaaa/AXIOM-AETHER
cd AXIOM-AETHER/axiom_engine_rs

cargo build --release --locked --bin axiom      # the shipped binary
cargo clippy --lib --locked -- -D warnings      # library lints, warning-free
cargo test  --release --locked                  # full suite (lib unit tests + tests/)
```

Then the end-to-end smoke, from the repo root — CPU-only, no network, throwaway
state dir:

```bash
./scripts/demo_end_to_end.sh
```

It drives compression, the self-healing runtime + acquired immunity, the
epistemic swarm, and grounding verification, printing `PASS`/`FAIL` per step.

### Zero-setup path (no Rust toolchain)

If you only need to run AXIOM (not modify it), use the container — it needs
nothing but Docker:

```bash
docker run -p 3000:8080 ghcr.io/fernandogarzaaa/axiom-aether:latest
curl -fsS http://localhost:3000/healthz          # {"status":"ok"}
```

`.github/workflows/ci.yml` also builds the image on every PR (`docker-build`
job, no push), so a broken `Dockerfile` is caught before merge.

---

## Toolchain

| Requirement | Version | Notes |
|---|---|---|
| Rust | **stable** | CI pins via `dtolnay/rust-toolchain@stable`; the Docker builder uses `rust:1.88-bookworm`. `edition = "2021"`. |
| C toolchain | platform native | `tree-sitter`, `candle`, and `tower-lsp` build C. **Linux:** `pkg-config libssl-dev build-essential`. **macOS:** Xcode CLT. **Windows:** Visual Studio Build Tools with the *"Desktop development with C++"* workload — without it `cargo build` fails at the link step (`link.exe` on `PATH` is not the MSVC linker). |
| Python | ≥ 3.11 | Only for the `axiom_engine/` reference implementation and the `protocol/cp1` validator — not required to build or run the Rust engine. |

Linux and Docker are the primary supported dev paths; both are exercised by CI
on every push. macOS arm64 is release-built but not CI-tested. Windows is
release-built; local development there needs the MSVC C++ workload above.

Cargo features (`axiom_engine_rs/Cargo.toml`):

| Feature | Default | Purpose |
|---|---|---|
| *(none)* | ✅ | Ships only the user-facing `axiom` command. |
| `cuda` | | NVIDIA GPU backend (`--features cuda`, needs the CUDA toolkit). |
| `metal` | | Apple GPU backend (`--features metal`, macOS only). |
| `tools` | | Dev/training/eval binaries under `src/bin/*` (`axiombench`, `train_*`, …). |
| `live-eval` | | `tests/cvm_eval.rs` — drives real `claude -p` traffic and bills your account. |
| `experimental` | | *(planned — see [Experimental surfaces](#experimental-surfaces))* gates ChimeraLang, the VFS hypervisor, and the DWE swarm/fleet. |

---

## Module map

`axiom_engine_rs/src/` — declared in `lib.rs` (~90 modules). The load-bearing
ones, grouped by pillar:

### Core engine

| File | Role |
|---|---|
| `main.rs`, `bin/axiom.rs` | Binary entrypoints (`axiom_engine` alias + canonical `axiom`). |
| `entrypoint.rs` | `--mode` / subcommand dispatch. |
| `cli.rs` | `clap` subcommand + argument definitions. |
| `config.rs`, `model_meta.rs` | `AxiomConfig` (d_model, n_layers, vocab_size…); checkpoint sidecar metadata. |
| `kernel.rs`, `model.rs` | `AxiomTTTEngine` — Embedding → N×block → RMSNorm → LM head. |
| `ttt_block.rs` | `NativeTTTBlock` — prefill + per-token fast-weight (`W̃`) update. |
| `ttt_mlp.rs`, `ttt_mlp_model.rs` | Opt-in MLP TTT backbone + wrapper. |
| `inference.rs` | `InferencePipeline`: `generate`, `generate_with_session`, `adapt_on_corpus`. |
| `train.rs`, `data_gen.rs`, `meta_train.rs` | AdamW training loop, procedural dataset, meta-training. |
| `bootstrap.rs` | Offline checkpoint bootstrap for `axiom init`. |

### Compression / cost (CVM + PSS)

| File | Role |
|---|---|
| `context_compressor.rs`, `skeleton.rs` | Context partitioning + readable structural digest. |
| `responses_compressor.rs` | `/v1/responses` input compression (on by default). |
| `cache_safety.rs`, `cvm_store.rs`, `digest.rs`, `cost_ledger.rs` | Cache-safe rewriting, content-addressed L2 store, tool-result digest admission, dollar-true telemetry. |
| `tool_defer.rs`, `local_trivial.rs`, `rebase.rs`, `prefix_diet.rs`, `keepalive.rs` | PSS levers L-A / L-B / R2 / S4 / opt-in TTL pings. |
| `anthropic_forwarder.rs`, `openai_forwarder.rs` | Upstream passthrough. |

### Autonomy (the spearhead — see [`docs/BENCH-REPAIR.md`](docs/BENCH-REPAIR.md))

| File | Role |
|---|---|
| `self_heal.rs`, `heal_memory.rs` | Supervised command runner + learned environment immunity. |
| `solve.rs`, `poly_jit.rs`, `sandbox.rs`, `fault_locate.rs` | Verifier-driven repair loop, deterministic source-repair patterns, localization. |
| `agentic.rs` | Verifier-gated, reversible, held-out-split agentic loop. |
| `agentic_eval.rs` | `axiom eval-agentic` — the built-in seeded broken-repo benchmark. |

### Grounding / drift

| File | Role |
|---|---|
| `hallucination.rs` | Evidence-relative grounding checks + calibrated conformal trust gate. |
| `epistemic_drift.rs` | Semantic LLM-judge gate over proxied responses. |
| `surprisal.rs` | Token-surprisal signals. |

### Memory / belief

| File | Role |
|---|---|
| `memory_store.rs`, `memory_recall.rs`, `graph_memory.rs` | Long-term store + recall + directed-edge graph with bounded spreading activation. |
| `vibe_memory.rs` | Persistent fast-weight "master vibe" (EMA-merged on session drop). |
| `belief.rs`, `provenance.rs` | `BetaBelief` peer confidence; signed-export verification. |
| `weight_merge.rs` | Checkpoint + peer weight merge (DARE/TIES). |

### Server / protocol

| File | Role |
|---|---|
| `server.rs`, `server/*.rs` | `axum` OpenAI- + Anthropic-compatible HTTP API, TTT session management. |
| `mcp_stdio.rs` | Native MCP server (stdio + HTTP), 20 tools. |
| `lsp.rs` | Language Server Protocol daemon. |
| `metrics.rs` | Prometheus exposition. |
| `cp1.rs` | CP/1 normative protocol source (see `protocol/cp1/`). |

### Experimental (see next section)

`chimera.rs` · `dwe.rs` · `cluster.rs` · `swarm_route.rs` · `swarm_router.rs` ·
`mesh_router.rs` · `daemon.rs` · `vfs.rs` · `state_predictor.rs` ·
`trajectory_sampler.rs` · `predictive_tools.rs` · `alignment_loop.rs` ·
`q_manifold.rs` · `hamiltonian.rs`

---

## TTT layer mathematics

The core lives in `ttt_block.rs`. Each `NativeTTTBlock` maintains a dynamic
weight matrix **W̃** ∈ ℝ^(H×D×D) (one per head).

### Prefill (parallel over T tokens)

Given `X ∈ ℝ^(B×T×D)`:

1. Projections: `Q = X Wq`, `K = X Wk`, `V = X Wv`
2. TTT target: minimise `L = ‖W_curr K − V‖²_F` (per head, summed over T)
3. Gradient step: `W̃ = W_curr − η · ∇_{W_curr} L`, `η = lr_inner` (default `1e-3`)
4. Output: `Y = W̃ Q`

### Decode (per token)

For `x ∈ ℝ^(B×1×D)` and current `W̃`:

1. Project `q, k, v`
2. `err = W̃ k − v`; `∇ = err ⊗ k` (outer product, H×D×D)
3. `W̃' = W̃ − η · ∇`
4. `y = W̃' q`; return `(y, W̃')` — the updated state **is** the session memory

### Why it matters

The `W̃` update is a per-session, per-token gradient step on the current
context's keys. After `/v1/adapt` with domain examples, the session's `W̃`
encodes that knowledge and later generation queries it without re-running
retrieval or fine-tuning.

> A logarithmic associative prefix scan was designed but is **not implemented**
> in either the Rust or Python tree today (verified by grep). Prefill is `O(T)`.

---

## Experimental surfaces

Three subsystems are wired and reachable but **not covered by the proof loop**,
not benchmarked, and not part of the supported product surface:

| Subsystem | Entrypoints |
|---|---|
| **ChimeraLang DSL** | `axiom chimera check\|run\|prove\|verify`, `POST /v1/chimera/run`, `chimera.rs` |
| **VFS hypervisor** | `axiom daemon\|mount`, `POST/GET /v1/hypervisor/*`, `daemon.rs`, `vfs.rs` |
| **DWE swarm / fleet** | `axiom swarm\|fleet`, `/v1/fleet/*`, `/v1/cluster/*`, `/v1/swarm/*`, `dwe.rs`, `cluster.rs`, `swarm_*.rs`, `mesh_router.rs` |

They currently compile unconditionally. The plan is to gate them behind a
`cargo` feature named `experimental` (default off), so `cargo install`, the pip
wheel, Docker, and release builds ship only the proven surface, and the full
surface is `cargo build --features experimental`. The file-by-file gating
checklist is in [`docs/EXPERIMENTAL.md`](docs/EXPERIMENTAL.md).

**If you add a feature that is not yet benchmarked, put it here** — in the
`experimental` set and out of the main README capability table — until it has a
repeatable evaluation and a clear operational reason to be on by default.

---

## The proof loop

Every release republishes three headline numbers, each with the command that
reproduces it. See [`docs/PROOF-LOOP.md`](docs/PROOF-LOOP.md). Before tagging a
version, run `./scripts/proof_loop.sh` and paste its table into the release
notes and the README's *Proof Loop* section. A PR that changes engine behavior
should note any expected movement in those numbers.

---

## Pull request guidelines

1. **One concern per PR.**
2. `cargo fmt`, `cargo clippy --lib --locked -- -D warnings`, and
   `cargo test --release --locked` pass locally before opening.
3. **Update this file** when you add a module or move a surface between
   core and experimental.
4. **Add tests** for new server endpoints and new repair patterns.
5. **Keep public claims evidence-backed.** If a feature is experimental, label
   it that way. If a metric depends on a local run, include the command that
   reproduces it — the README and `docs/` are audited against the code, not
   against intentions.
6. **Update the HTTP API list** in `README.md` if you add or remove a
   `.route(...)`.
