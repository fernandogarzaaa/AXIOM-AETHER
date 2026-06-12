# Axiom — Capabilities Map

A single index of everything Axiom does, by surface. The README narrates *how*
each was built and validated; this is the *what* and *where*, for orientation.

Everything rests on one mechanism — **online Test-Time Training (TTT)**: a
per-layer fast-weight matrix `W̃` takes a real gradient step on every token at
inference (`ttt_block.rs`), so the engine learns from context by changing its
weights, O(1)/step, no fine-tune pipeline.

---

## The four pillars

| Pillar | What it does | Where |
|---|---|---|
| **1. Context compression** | Absorb heavy context into `W̃`, forward a Claude-readable structural skeleton (~80% fewer input tokens), recover bodies on demand. | `context_compressor.rs`, `skeleton.rs`, `/v1/messages` |
| **2. Self-healing runtime** | Run a program; on failure, feel the tension (CE), absorb it, heal the environment, and learn immunity. | `self_heal.rs`, `heal_memory.rs`, `axiom run` |
| **3. Autonomy** | Drive a failing verify command to green by chaining environment-heal + source-repair + verify. | `solve.rs`, `axiom solve` |
| **Grounding (anti-hallucination)** | Flag response claims unsupported by the evidence; spend tokens back only where grounding needs them. | `hallucination.rs`, `/v1/verify` |

Cross-cutting: a **verifiable epistemic swarm** (Beta-belief confidence +
Dempster-Shafer merge + tamper-evident provenance) lets nodes share what they
learn safely (`belief.rs`, `provenance.rs`, `/v1/immunity`).

---

## CLI subcommands (`axiom <cmd>`)

| Command | Purpose |
|---|---|
| `init [--no-fetch] [--no-train]` | Scaffold `~/.axiom`; bootstrap a local checkpoint so the proxy never boots on random weights. |
| `--mode server` | The OpenAI/Anthropic-compatible compression proxy. |
| `--mode mcp` | Native MCP stdio server (tools below). |
| `--mode doctor` | Hardware detection + safe per-role device recommendation (GPU co-tenancy guard). |
| `--mode generate \| train \| lsp` | Local generation, training, and an LSP surface. |
| `prime <dir>` | Warm-start the persistent vibe memory by absorbing a codebase. |
| `bench <dir>` | Measure compression: token savings + lossless round-trip rate. |
| `run [--dry-run] [--max-restarts N] -- <cmd>` | Self-healing supervision (Pillar 2). |
| `solve [--source PATH] [--max-rounds N] -- <cmd>` | Autonomous drive-to-green (Pillar 3). |
| `immunity [query] [--prune]` | Report/curate acquired immunity (heals + Beta confidence). |
| `swarm connect <peer>` / `swarm immunity <host:port>` | Register a DWE peer / pull+merge a peer's immunity (provenance-verified). |
| `daemon start\|stop\|status`, `mount <dir>` | Background hypervisor + Neural VFS. |

## HTTP endpoints (`--mode server`)

**Inference / compression:** `/v1/messages`, `/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/v1/adapt`, `/v1/expand`, `/v1/config`, `/metrics`.
**TTT sessions:** `/v1/sessions[/:id[/checkpoint]]`, `/v1/ttt/sessions[/:id]`, `/v1/ttt/feedback`.
**Grounding:** `/v1/verify` (modes: lexical, `neural:true`, `expand:true`).
**Swarm immunity:** `/v1/immunity` (signed export), `/v1/immunity/merge` (verify-before-trust), `/v1/cluster/sync`, `/v1/cluster/merge`.
**Hypervisor:** `/v1/hypervisor/mount`, `/v1/hypervisor/read` (VFS→TTT prefill), `/v1/hypervisor/jit_run` (reversible source repair), `/v1/hypervisor/jit_status`, `/v1/swarm/matrix_state`.

## MCP tools (`--mode mcp`)

`axiom_compress_path`, `axiom_evaluate_drift`, `axiom_expand`,
`axiom_remember` / `axiom_recall` / `axiom_forget` (persistent memory),
`axiom_immunity` (what Axiom has learned about a command),
`axiom_verify` (grounding-check a response against evidence).

---

## Key environment variables

| Var | Effect |
|---|---|
| `AXIOM_DEVICE` | `cpu` / `cuda` / `metal` / `auto` (co-tenancy guard). |
| `AXIOM_TTT_COMPRESS`, `AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS` | Enable/threshold compression. |
| `AXIOM_PRODUCTION_BPE` + `AXIOM_TOKENIZER` + `AXIOM_BPE_CKPT` | Use a trained BPE checkpoint. |
| `AXIOM_HEAL_MEMORY` | Heal-memory path (`0`/`off` disables). |
| `AXIOM_RUN_VIBE`, `AXIOM_RUN_REMEMBER` | Persist a run's W̃ / write novel fixes to recall memory. |
| `AXIOM_FLEET_KEY` | HMAC peer-auth for swarm-immunity exchange. |
| `AXIOM_IMMUNITY_INJECT` | Active-immunity advisory injection (`0` to disable). |
| `AXIOM_VERIFY_RESPONSES` | Opt-in auto grounding-verification of responses. |
| `AXIOM_SWARM_LOCAL` + `AXIOM_OLLAMA_*` | Ollama-first local routing. |
| `AXIOM_DRIFT_THRESHOLD` | Drift gate (auto-set by `eval_model`). |

---

## Supporting subsystems

- **Vibe memory** (`vibe_memory.rs`) — EMA-merged persistent `W̃` (codebase DNA).
- **Tiered recall memory** (`memory_store.rs`, `memory_recall.rs`) — embedded store the proxy and `axiom_recall` share; the self-heal bridge writes novel fixes here.
- **SR-TTT** (`surprisal.rs`) — exact residual cache for high-surprisal identifiers (secrets/hashes).
- **DWE** (`dwe.rs`) — binary `W̃`-delta exchange between peers.
- **Hypervisor** — Neural VFS (`vfs.rs`), Poly JIT source repair (`poly_jit.rs`) ranked by a Q-TTT simulator (`q_manifold.rs`, `hamiltonian.rs`), compile-check sandbox (`sandbox.rs`).
- **Training** — `train_tokenizer`, `train_semantic` (early-stopping on held-out CE), `eval_model` (acceptance + drift-gate calibration); `scripts/train_cpu_quickstart.sh`.

## Try it

```bash
./scripts/demo_end_to_end.sh   # drives every pillar on CPU, no network, PASS/FAIL per step
```
