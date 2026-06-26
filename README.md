# AXIOM-AETHER

[![CI](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/ci.yml)
[![Release Binaries](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/release.yml/badge.svg)](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/release.yml)
[![Docker](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/docker.yml/badge.svg)](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/docker.yml)

Axiom is a local-first Rust research runtime for online Test-Time Training (TTT),
context compression, grounding checks, and self-healing execution loops.

The core idea is simple: Axiom keeps per-session fast weights, called
`W_tilde`, and updates them while processing context. Those updates are used by
the local runtime to compress large inputs, measure drift, preserve session
state, and feed bounded automation loops.

This README has been rewritten from a line-by-line audit of the previous page,
the Rust source, the workflow files, the release assets, and the current docs.
It separates shipped surfaces from experimental research and avoids AGI, SGI, or
general answer-quality claims that are not proven by the repository.

## What Is Implemented

| Surface | What it does | Main code |
|---|---|---|
| Online TTT engine | Updates per-session fast weights during inference and adaptation. | `axiom_engine_rs/src/ttt_block.rs`, `model.rs`, `inference.rs` |
| Context compression proxy | Accepts Anthropic Messages and OpenAI Chat Completions style traffic, absorbs heavy context locally, and forwards a smaller readable payload. | `server.rs`, `context_compressor.rs`, `skeleton.rs`, `anthropic_forwarder.rs`, `openai_forwarder.rs` |
| MCP server | Exposes Axiom as stdio JSON-RPC tools for compression, drift checks, expansion, memory, immunity, and grounding. | `mcp_stdio.rs` |
| Self-healing runner | Runs a command, detects supported environment failures, applies bounded heals, and records learned immunity. | `self_heal.rs`, `heal_memory.rs`, `main.rs` |
| Autonomous solve loop | Uses the runner plus source repair attempts to drive a verifier command toward green. | `solve.rs`, `poly_jit.rs`, `sandbox.rs` |
| Grounding verification | Checks whether response claims are supported by supplied evidence and can expand dropped symbols when a session digest is available. | `hallucination.rs`, `/v1/verify` in `server.rs` |
| Swarm and provenance | Shares selected learned state through signed immunity export/merge and weighted belief logic. Patch gossip is Byzantine-robust (bounded per-peer trust). | `belief.rs`, `provenance.rs`, `weight_merge.rs`, `patch_memory.rs` |
| ChimeraLang DSL | In-tree Rust implementation of the [ChimeraLang](https://github.com/fernandogarzaaa/ChimeraLang) AI-cognition language: `belief/inquire/resolve/guard/evolve` programs run on the same `BetaBelief` + provenance substrate, with tamper-evident run certificates. | `chimera.rs`, CLI `axiom chimera`, `/v1/chimera/run` |
| Search ingestion node | Scrapes web pages, ingests text through local TTT, and emits an Axiom fingerprint for downstream use. | `src/bin/search_node.rs`, `search_scrape.rs`, `search_ingest.rs` |

For a compact index of surfaces, see [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md).
For research upgrades and what is still planned, see [`docs/UPGRADES.md`](docs/UPGRADES.md).

## Verification Status

The default CI workflow builds and tests the Rust workspace on Ubuntu:

```text
cargo build --release --locked
cargo test --release --locked
./scripts/demo_end_to_end.sh
```

The demo script runs the major local surfaces together on CPU: hardware doctor,
compression benchmark, self-healing runtime, learned immunity, solve loop, and
grounding verification.

Release CI also builds archives for Linux x86-64, Windows x86-64, and macOS
arm64, then publishes checkpoint assets. The Docker workflow publishes a
multi-architecture GHCR image for `linux/amd64` and `linux/arm64`.

Local verification requires a working Rust toolchain. If `RUSTUP_HOME` or
`CARGO_HOME` points to a missing drive, fix that before running local `cargo`
commands.

## Honest Scope

Axiom is not presented here as AGI, SGI, or a universal factuality engine.

Current grounding checks are evidence-relative: they test whether claims are
supported by the context you provide. They do not prove external truth. The
search node ingests live web text, but it is a local ingestion primitive, not a
replacement for source review. Compression benchmarks report token savings and
round-trip behavior; answer-quality gains require separate upstream evaluation.

## Quick Start

### Install

All three install the `axiom` command:

```bash
pip install axiom-aether            # prebuilt binary, no Rust toolchain (recommended)
cargo install axiom_engine          # build from source (needs a Rust toolchain)
```

Or the release helper (no package manager needed):

Linux/macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/fernandogarzaaa/AXIOM-AETHER/main/scripts/install.sh | bash
```

Windows PowerShell:

```powershell
iwr https://raw.githubusercontent.com/fernandogarzaaa/AXIOM-AETHER/main/scripts/install.ps1 -UseB | iex
```

> `pip install axiom-aether` ships the precompiled `axiom` engine binary — it is
> **not** the same as the pure-Python `axiom-engine` reference package. Publishing
> is automated on version tags; see [`docs/PUBLISHING.md`](docs/PUBLISHING.md).

### Build from source

```bash
git clone https://github.com/fernandogarzaaa/AXIOM-AETHER
cd AXIOM-AETHER/axiom_engine_rs
cargo build --release --locked
cargo test --release --locked
```

### Generation backends & models

Axiom's own model handles compression/recall; **text generation is pluggable and
optional** — a tiny model is bootstrapped offline by `axiom init`, and you can
point axiom at a bigger brain when you want one:

- **Local, zero-cost** — serve an open model (default **Qwen2.5-Coder-32B**) via
  [OpenDrop](https://github.com/fernandogarzaaa/OpenDrop): `AXIOM_BACKEND=opendrop`
  auto-targets the local server, or use `AXIOM_BACKEND=openai` + `OPENAI_BASE_URL`
  for any other host/port (Ollama, vLLM).
- **Cloud** — `AXIOM_BACKEND=openai` or `AXIOM_BACKEND=anthropic` with an API key.
- **GPT + Claude together** — `AXIOM_BACKEND=router` routes by task
  (code→Claude, else→GPT) with failover and opt-in consensus.
- **Your ChatGPT/Claude subscription** — drive axiom's tools from the app via MCP
  (zero API cost); works from Claude over stdio and from the ChatGPT connector
  over remote HTTP (`AXIOM_MCP_HTTP=1`).

See [`docs/AGENT-SETUP.md`](docs/AGENT-SETUP.md) (**set up Axiom with your
Codex/Claude agent** — copy-paste prompts + troubleshooting),
[`docs/BACKENDS.md`](docs/BACKENDS.md) (backends + model recommendations) and
[`docs/MCP-CLIENTS.md`](docs/MCP-CLIENTS.md) (MCP transport details). The trained
model is an enhancement, not a requirement.

Run the local hardware check:

```bash
./target/release/axiom_engine --mode doctor
```

Start the HTTP server:

```bash
./target/release/axiom_engine --mode server --host 127.0.0.1 --port 3000
```

The repo also includes `start_axiom.sh`, which configures the local proxy path
used by the project scripts.

### Run the full local smoke demo

```bash
./scripts/demo_end_to_end.sh
```

The script builds the release binary if needed and uses a temporary state
directory, so it does not mutate your normal Axiom memory.

### Simulate a brand-new user (blank slate)

```bash
./scripts/new_user_simulation.sh
```

Stands up a pristine `HOME` with no `~/.axiom`, runs `axiom init` from scratch
(offline checkpoint bootstrap included), then walks the real first-run journey —
doctor → compression → self-healing runtime → learned immunity → the ChimeraLang
DSL (run + offline certificate) → the reproducible capability score — printing
PASS/FAIL per step. Useful as a first-run regression check.

## Release Assets

The latest GitHub release publishes:

| Asset | Purpose |
|---|---|
| `axiom-ttt-{version}-linux-x86_64.tar.gz` | Linux binary archive |
| `axiom-ttt-{version}-windows-x86_64.zip` | Windows binary archive |
| `axiom-ttt-{version}-macos-arm64.tar.gz` | macOS Apple Silicon binary archive |
| `axiom_production_bpe.bin` | Release checkpoint |
| `axiom_production_bpe.meta.json` | Checkpoint metadata |
| `axiom_bpe.json` | Tokenizer |
| `axiom_drift_gate.txt` | Drift threshold produced by the checkpoint job |
| `SHA256SUMS.txt` | Release asset checksums |

Release page: [latest AXIOM-AETHER release](https://github.com/fernandogarzaaa/AXIOM-AETHER/releases/latest)

## HTTP API

Start the server with `--mode server`. Main endpoints:

```text
GET    /healthz
GET    /readyz
GET    /metrics
GET    /v1/models
POST   /v1/completions
POST   /v1/chat/completions
POST   /v1/messages
POST   /v1/adapt
POST   /v1/expand
POST   /v1/verify
POST   /v1/sessions
GET    /v1/sessions/{id}/checkpoint
PUT    /v1/sessions/{id}/checkpoint
DELETE /v1/sessions/{id}
GET    /v1/ttt/sessions
DELETE /v1/ttt/sessions
POST   /v1/ttt/feedback
POST   /v1/cluster/sync
POST   /v1/cluster/merge
GET    /v1/immunity
POST   /v1/immunity/merge
POST   /v1/hypervisor/mount
POST   /v1/hypervisor/read
POST   /v1/hypervisor/jit_run
GET    /v1/hypervisor/jit_status
GET    /v1/hypervisor/quantum_coherent_state
GET    /v1/swarm/matrix_state
GET    /v1/config
POST   /v1/config
```

Example adaptation call:

```bash
SESSION=$(curl -s -X POST http://127.0.0.1:3000/v1/sessions -d '{}' | jq -r .session_id)
curl -X POST http://127.0.0.1:3000/v1/adapt \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SESSION\",\"corpus\":[\"candle is a Rust ML framework.\"]}"
```

## MCP Tools

Run Axiom as an MCP server:

```bash
axiom_engine --mode mcp --checkpoint checkpoints/axiom_production_bpe.bin
```

Tool catalog:

| Tool | Purpose |
|---|---|
| `axiom_compress_path` | Absorb a file or directory and return an Axiom context digest. |
| `axiom_evaluate_drift` | Score code against the current fast weights and flag drift above threshold. |
| `axiom_expand` | Retrieve a symbol body that compression dropped from a session digest. |
| `axiom_remember` | Store a long-term memory item. |
| `axiom_recall` | Search long-term memory. |
| `axiom_forget` | Tombstone a remembered item. |
| `axiom_verify` | Check response claims against supplied evidence. |
| `axiom_immunity` | Report learned self-healing experience for a command. |

Claude-style MCP config example:

```jsonc
{
  "mcpServers": {
    "axiom": {
      "command": "C:\\path\\to\\axiom_engine.exe",
      "args": ["--mode", "mcp", "--checkpoint", "checkpoints/axiom_production_bpe.bin"]
    }
  }
}
```

## CLI Commands

Release installers expose the command as `axiom`. When building from source in
this repository, use `./target/release/axiom_engine` or
`target\release\axiom_engine.exe`.

The runtime supports both legacy `--mode` operation and newer subcommands:

```text
axiom --mode generate|train|server|mcp|lsp|meta-train|doctor
axiom init [--no-fetch] [--no-train]
axiom prime {dir}
axiom bench {dir}
axiom run [--dry-run] [--max-restarts N] -- {cmd}
axiom solve [--source PATH] [--max-rounds N] -- {cmd}
axiom immunity [query] [--prune]
axiom swarm connect {peer}
axiom swarm immunity {host:port}
axiom daemon start|stop|status
axiom mount {dir}
```

Useful environment variables:

| Variable | Effect |
|---|---|
| `AXIOM_DEVICE` | `cpu`, `cuda`, `metal`, or `auto`. |
| `AXIOM_PRODUCTION_BPE` | Enable the trained BPE checkpoint path. |
| `AXIOM_TOKENIZER` | Path to `axiom_bpe.json`. |
| `AXIOM_BPE_CKPT` | Path to `axiom_production_bpe.bin`. |
| `AXIOM_DRIFT_THRESHOLD` | Override drift threshold. |
| `AXIOM_TTT_COMPRESS` | Enable proxy compression. |
| `AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS` | Compression threshold. |
| `AXIOM_VIBE` / `AXIOM_VIBE_PRIME` | Persistent fast-weight memory controls. |
| `AXIOM_HEAL_MEMORY` | Path for learned heal memory, or `0`/`off` to disable. |
| `AXIOM_FLEET_KEY` | HMAC key for swarm immunity exchange. |
| `AXIOM_VERIFY_RESPONSES` | Opt in to response grounding advisories. |

## Docker

```bash
docker run -p 3000:8080 ghcr.io/fernandogarzaaa/axiom-aether:latest
```

The image is intentionally lean. For a trained checkpoint at boot, mount or
download the release assets and set:

```bash
docker run -p 3000:8080 \
  -e AXIOM_CHECKPOINT_URL=https://github.com/fernandogarzaaa/AXIOM-AETHER/releases/latest/download/axiom_production_bpe.bin \
  -e AXIOM_TOKENIZER_URL=https://github.com/fernandogarzaaa/AXIOM-AETHER/releases/latest/download/axiom_bpe.json \
  ghcr.io/fernandogarzaaa/axiom-aether:latest
```

## Kubernetes

The Helm chart lives in [`deploy/helm/axiom`](deploy/helm/axiom).

```bash
helm install axiom deploy/helm/axiom \
  --namespace axiom \
  --create-namespace \
  --set checkpoint.url=https://github.com/fernandogarzaaa/AXIOM-AETHER/releases/latest/download/axiom_production_bpe.bin \
  --set checkpoint.tokenizerUrl=https://github.com/fernandogarzaaa/AXIOM-AETHER/releases/latest/download/axiom_bpe.json \
  --set secrets.fleetKey=$(openssl rand -hex 32)
```

The chart includes probes, optional Prometheus `ServiceMonitor`, a conditional
PodDisruptionBudget, checkpoint download support, and a GPU values overlay:

```bash
helm install axiom deploy/helm/axiom -f deploy/helm/axiom/values-gpu.yaml
```

Full guide: [`docs/DEPLOY_K8S.md`](docs/DEPLOY_K8S.md).

## Research And Upgrade Notes

[`docs/UPGRADES.md`](docs/UPGRADES.md) tracks methods that were audited against
current research and then classified as implemented, scaffolded, or planned.
Examples include guarded self-correction, DARE/TIES merge logic, Dempster-Shafer
conflict handling, opt-in learned gating, and planned factuality/compression
upgrades.

Experimental pieces remain opt-in or documented as research until they have
repeatable evaluation and clear operational value.

## Repository Map

```text
axiom_engine_rs/src/
  adaptive.rs              adaptive compression thresholding
  anthropic_forwarder.rs   Anthropic Messages passthrough/forwarding
  belief.rs                Beta belief and conflict-aware peer confidence
  cli.rs                   clap subcommands
  context_compressor.rs    context partitioning and digest construction
  hallucination.rs         evidence-relative grounding checks
  heal_memory.rs           learned self-healing memory
  inference.rs             tokenizer plus inference pipeline
  main.rs                  binary entrypoint and mode dispatch
  mcp_stdio.rs             native MCP stdio server
  memory_store.rs          long-term memory store
  model.rs                 model layers
  model_meta.rs            checkpoint sidecar metadata
  openai_forwarder.rs      OpenAI-compatible forwarding
  poly_jit.rs              source repair loop support
  provenance.rs            signed export verification
  search_ingest.rs         live search ingestion into TTT
  search_scrape.rs         Rust web scraping helper
  self_heal.rs             supervised command runner
  server.rs                HTTP API
  skeleton.rs              readable structural compression digest
  solve.rs                 verifier-driven solve loop
  ttt_block.rs             fast-weight update rule
  weight_merge.rs          checkpoint and peer merge logic
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for architecture notes and development
guidance. Keep public claims evidence-backed: if a feature is experimental,
label it that way; if a metric depends on a local run, include the command that
reproduces it.
