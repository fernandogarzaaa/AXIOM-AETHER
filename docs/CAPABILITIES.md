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
| **1. Context compression** | Absorb heavy context into `W̃`, forward compact fingerprints, recover bodies on demand, and compress safe-prefix `/v1/responses` transcripts by replacing contiguous assistant runs in place. | `context_compressor.rs`, `responses_compressor.rs`, `skeleton.rs`, `/v1/messages`, `/v1/responses` |
| **2. Self-healing runtime** | Run a program; on failure, feel the tension (CE), absorb it, heal the environment, and learn immunity. | `self_heal.rs`, `heal_memory.rs`, `axiom run` |
| **3. Autonomy** | Drive a failing verify command to green by chaining environment-heal + source-repair + verify. | `solve.rs`, `axiom solve` |
| **Grounding (anti-hallucination)** | Flag response claims unsupported by the evidence with a shipped conformal threshold, and benchmark neural-tier contradiction catch-rate against a lexical baseline. | `hallucination.rs`, `bench/trust/claims.jsonl`, `/v1/verify` |

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
| `fleet status [--offline] [--base-url URL]` / `fleet join <peer>` | Inspect live fleet telemetry (`/metrics` + `/v1/fleet/status`) or print env-only wiring; emit join env for a peer. |
| `daemon start\|stop\|status`, `mount <dir>` | Background hypervisor + Neural VFS. |

## HTTP endpoints (`--mode server`)

**Inference / compression:** `/v1/messages`, `/v1/responses`, `/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/v1/adapt`, `/v1/expand`, `/v1/config`, `/metrics`.
**CVM cost stack:** `/v1/awareness/:id` (dollar-true cost summary, cache hit rate),
`/v1/prefix-diet/report/:session_id` (last request's dedup stats, `AXIOM_PREFIX_DEDUP=1`
only). `/v1/expand` also resolves `AXIOM-PAGE` ids from digest admission control (S3),
not just skeleton symbol names.
**TTT sessions:** `/v1/sessions[/:id[/checkpoint]]`, `/v1/ttt/sessions[/:id]`, `/v1/ttt/feedback`.
**Grounding:** `/v1/verify` (modes: lexical, `neural:true`, `expand:true`).
**Swarm immunity / fleet:** `/v1/fleet/status`, `/v1/immunity` (signed export), `/v1/immunity/merge` (verify-before-trust), `/v1/cluster/sync`, `/v1/cluster/merge`; `/metrics` exports `axiom_dwe_sent`, `axiom_dwe_received`, `axiom_dwe_applied`, and `axiom_dwe_rejected`.
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
| `AXIOM_FLEET_KEY`, `AXIOM_FLEET_KEY_PREV` | Current and previous HMAC keys for swarm-immunity exchange and DWE fragments; previous key is accepted only during rotation. |
| `AXIOM_IMMUNITY_INJECT` | Active-immunity advisory injection (`0` to disable). |
| `AXIOM_VERIFY_RESPONSES` | Opt-in auto grounding-verification of responses. |
| `AXIOM_SWARM_LOCAL` + `AXIOM_OLLAMA_*` | Ollama-first local routing. |
| `AXIOM_DRIFT_THRESHOLD` | Drift gate (auto-set by `eval_model`). |
| `AXIOM_CACHE_SAFE` | CVM cache-safety hardening; `0` disables (default `1`). |
| `AXIOM_CVM_DIGEST` | CVM digest admission control: `off` / `skeleton` (default) / `haiku`. |
| `AXIOM_CVM_DIGEST_THRESHOLD_TOKENS` | Token threshold before a `tool_result` block is digested (default `4000`). |
| `AXIOM_PREFIX_DEDUP` | CVM lossless system-prefix dedup; `1` enables (default `0` — measured 0% real gain so far). |
| `AXIOM_CVM_DIR` | CVM L2 store root (default `checkpoints/cvm`). |
| `AXIOM_CVM_RETAIN` | `1` keeps a session's CVM store file after session drop (default: deleted). |
| `AXIOM_KEEPALIVE` | CVM actuarial cache-refresh pings; `1` enables. Default `0` forever unless explicitly opted in — replays your own credentials on a timer. |
| `AXIOM_TOOL_DEFER` | PSS L-A tool deferral via `defer_loading` (default `on`; `off` disables). |
| `AXIOM_LOCAL_TRIVIAL` | PSS L-B local trivial-turn short-circuit (default `on`; `off` disables). |
| `AXIOM_REBASE_ON_BREAK` | PSS R2 free-window rebasing at genuine cache breaks (default `on`; `off` disables). |
| `AXIOM_ADAPTIVE_TTL` | PSS R3 1-hour cache TTL election after repeated long gaps (default `on`; `off` disables). |
| `AXIOM_MODEL_ROUTE` | PSS R1 high-tier routing: `off` / `auto` (default — Opus/Fable only) / `on` (all Claude tiers). |

---

## Supporting subsystems

- **Vibe memory** (`vibe_memory.rs`) — EMA-merged persistent `W̃` (codebase DNA).
- **Tiered recall memory** (`memory_store.rs`, `memory_recall.rs`) — embedded store the proxy and `axiom_recall` share; the self-heal bridge writes novel fixes here.
- **SR-TTT** (`surprisal.rs`) — exact residual cache for high-surprisal identifiers (secrets/hashes).
- **DWE** (`dwe.rs`, `server/routes_fleet.rs`) — binary `W̃`-delta exchange between peers, HMAC authenticated, replay guarded, and observable through live counters/status.
- **AxiomBench** (`src/bin/axiombench/`) — reproducible cognition, trust, fleet, and live cost evidence; `RESULTS.md` carries the current headline table.
- **CVM cost stack** (`cache_safety.rs`, `cvm_store.rs`, `digest.rs`, `prefix_diet.rs`, `keepalive.rs`, `cost_ledger.rs`) — dollar-true cost reduction for `/v1/messages` traffic: never rewrites bytes at/before a client cache breakpoint, digests heavy tool results into a recoverable content-addressed store, and tracks real USD cost (not byte counts) per session. Design + measured-vs-simulated status: [`docs/superpowers/plans/2026-07-10-cvm-cost-stack.md`](superpowers/plans/2026-07-10-cvm-cost-stack.md), README's [CVM Cost Stack](../README.md#cvm-cost-stack) section.
- **Hypervisor** — Neural VFS (`vfs.rs`), Poly JIT source repair (`poly_jit.rs`) ranked by a Q-TTT simulator (`q_manifold.rs`, `hamiltonian.rs`), compile-check sandbox (`sandbox.rs`).
- **Training** — `train_tokenizer`, `train_semantic` (early-stopping on held-out CE), `eval_model` (acceptance + drift-gate calibration); `scripts/train_cpu_quickstart.sh`.

## Try it

```bash
./scripts/demo_end_to_end.sh   # drives every pillar on CPU, no network, PASS/FAIL per step
```
