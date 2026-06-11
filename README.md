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

[![CI](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogarzaaa/AXIOM-AETHER/actions/workflows/ci.yml)
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
| **Graceful degradation** | A compression-side fault never costs you a turn. If a *compressed* `/v1/messages` or `/v1/chat/completions` forward fails in a recoverable way (transient network, upstream `5xx`, or a `400` the injected digest could have caused), the proxy retries **once** with the original uncompressed payload. Auth / rate-limit / permission failures (`401/403/407/429`) are surfaced immediately — never retried. The fallback count is exposed at `GET /v1/config` (`counters.degraded_fallbacks`). |
| **Readable wire payload** | The opaque neural fingerprint (vocab-id `recall_top_k_indices`, layer Frobenius norms) is **no longer forwarded** — it is noise to a different upstream model and only burns tokens. Only the readable structural skeleton + a short provenance header go on the wire; the TTT drift signal stays server-side. |
| **Never boots on random weights** | `axiom init` now bootstraps a real local checkpoint (offline, bounded) when none exists, so the proxy loads converged-enough weights instead of noise. `--no-train` opts out. |

---

## Operator commands

Beyond the proxy/MCP/LSP runtimes, the `axiom` binary exposes batch commands:

```bash
# Bootstrap ~/.axiom + a real local checkpoint (no network, no random weights).
axiom init                       # --no-train skips the checkpoint bootstrap

# Warm-start: absorb a codebase into persistent vibe memory (codebase DNA) so
# new sessions start pre-adapted. Bounded crawl; AXIOM_PRIME_MAX_* to tune.
axiom prime ./my-repo            # then run the proxy with AXIOM_VIBE_PRIME=1

# Measure compression: token savings + structural round-trip fidelity, offline.
axiom bench ./my-repo
# → token savings 76.9% (60313 -> 13946 tokens); signatures 874/874 round-trip
```

`axiom bench` reports only what is locally verifiable — token savings and the
expand round-trip rate (every kept signature must recover its body). It does
**not** claim an answer-quality delta, which needs an upstream model.

---

## Self-healing runtime — `axiom run`

The core Axiom thesis: software running *inside* Axiom is not static text that
crashes — its failures become state the engine can feel and react to. `axiom
run` is that thesis as a working primitive:

```bash
axiom run -- sh -c 'echo 42 > /data/out/result.txt'   # /data/out does not exist
```

```text
sh: 1: cannot create /data/out/result.txt: Directory nonexistent
[axiom-run] attempt 1: exit=Some(2) — absorbing failure into W̃
[axiom-run]   tension: CE 5.545 -> 5.545 after absorption (29 tokens)
[axiom-run]   heal: created directory /data/out
[axiom-run] environment healed — restarting
[axiom-run] attempt 2: exited cleanly in 0.0s
[axiom-run] result: SUCCESS after 2 attempt(s); 1 heal(s); 29 failure token(s) absorbed
```

What actually happens on a failure:

1. **Tension** — the failure trace is scored through the model (cross-entropy,
   the same signal as the drift gate). An anomaly is a loss spike in the
   network, and the number is printed.
2. **Absorption (tension-gated)** — the trace is wrapped in the
   execution-feedback schema and streamed through the TTT stack: real gradient
   steps move the session's W̃ toward the failure. The *depth* of absorption is
   gated by the failure's novelty — a **FIRST/NOVEL** fault is absorbed deeply
   (the engine concentrates gradient effort on the surprising tension), a
   **KNOWN** fault is reinforced lightly. One session spans all restarts, so the
   program's failure history compounds.
3. **Environmental heal** — deterministic, safe policies repair what the
   process cannot survive: missing directories (`ENOENT` / `Directory
   nonexistent` → `mkdir -p`); a **missing execute bit** (`Permission denied`
   on a file the program tried to run → `chmod +x`, contents never touched);
   and recognised **transient faults** (`Connection refused/reset/timed out`,
   DNS, EAGAIN) where waiting *is* the heal — a bounded backoff-retry (max 2).
   Heals only ever create directories, add an execute bit, or wait — never
   delete, overwrite, or fabricate file content. Non-correctable conditions are
   **diagnosed** instead of faked: a **missing executable** (`command not
   found`) and a **full disk** (`ENOSPC`) are surfaced as actionable messages.
   A **missing required
   environment variable** is *diagnosed* (not fixed — a value can't be safely
   fabricated): Axiom surfaces an actionable message, remembers the requirement,
   and feeds it to `axiom immunity` and the proxy advisory so the operator (and
   Claude) know to export it.
4. **Continue** — restart up to `--max-restarts` (default 3), but **only when a
   new heal was applied**: an unhealed environment is never blindly replayed,
   and the child's exit code is preserved on give-up.
5. **Persist (opt-in)** — with `AXIOM_RUN_VIBE=1`, the run's adapted W̃ is
   EMA-merged into the master vibe on completion: the program's failure history
   becomes memory that outlives the process.
6. **Learned immunity** — every successful heal is remembered against a stable
   fingerprint of the command (`~/.axiom/heal_memory.json`, human-auditable;
   `AXIOM_HEAL_MEMORY` overrides the path, `0`/`off` disables). The next run of
   the same program — even in a **fresh environment** — is immunized first:
   remembered directories are re-created *before* the first attempt, so the
   failure never recurs. The memory also tracks each program's failure-tension
   history (running CE mean) and classifies every new failure as
   **FIRST / KNOWN / NOVEL** against it — the drift gate aimed at a program's
   own life history. Run a program once: crash → heal → success. Wipe the
   environment and run it again: `immunity: pre-created remembered directory …`
   → success on attempt 1 with **zero failure tokens absorbed**.
7. **Portable (location-invariant) immunity** — a heal under the working
   directory is remembered *relative* to it and re-anchored on immunize, so a
   fix learned in one checkout applies in another — and, via swarm immunity,
   on another machine. Learn `dist/` in `/home/alice/projA`; the same command
   in `/home/bob/projB` is immunized to `/home/bob/projB/dist` and succeeds on
   the first attempt, a location it never ran in. Heals outside the working
   directory stay absolute.
8. **Swarm immunity** — heals learned anywhere immunize the whole fleet. The
   server exports its heal memory at `GET /v1/immunity` and accepts a peer's at
   `POST /v1/immunity/merge`; `axiom swarm immunity <host:port>` pulls a peer
   and merges in one command. Merges are conservative: directory lists are
   unioned, tension histories combine as a count-weighted mean (a peer with 100
   observed failures outweighs one with 2), and local learning is never
   weakened. Demonstrated: a program crashes once on node A; node B runs
   `axiom swarm immunity nodeA:3000`; the same program then succeeds
   **first-try on node B, where it had never run before**.

### Anticipatory immunity (pre-failure prediction)

Acquired immunity also lets Axiom predict a failure *before the command runs* —
no model, no execution, just learned prerequisites checked against the current
environment:

```bash
axiom run --dry-run -- cargo build   # predict only, don't execute
# [axiom-run] dry-run prediction: LIKELY TO FAIL — 1 missing learned
#             prerequisite(s): /repo/target
```

Every real run does this as a silent pre-flight: if a learned prerequisite is
missing it logs `pre-flight: predicting failure …` and then immunizes
proactively, so the predicted failure never happens. This turns the immune
system from reactive + prophylactic into genuinely *anticipatory*.

### Cross-reactive immunity (generalization by analogy)

Like antibodies that recognize a pathogen similar to one seen before, Axiom
generalizes a heal across a **program family**: if `cargo build` learned it
needs `target/`, then a never-seen `cargo test` referenced in the conversation
gets an analogical **hint** in the proxy's `<axiom_immunity>` block —

```text
- cross-reactive hint: a sibling `cargo build` previously needed directory
  target; a different `cargo …` invocation here may need the same (Axiom has
  not applied it).
```

These are advisory **only** — never auto-applied — so a wrong analogy costs
nothing. Fires only for multi-sub-command families, never for a directly-known
command (that's a direct advisory) or an unrelated program.

### Adaptive immune confidence (maturation + waning)

Heals carry a confidence that follows an adaptive-immunity lifecycle:

- **Tentative** when first learned (0.50).
- **Affinity maturation** — each time immunizing a program precedes a
  successful run, confidence matures toward 1.0 (`tentative → proven →
  established`). Established fixes are asserted in proxy advisories; tentative
  ones are offered as possibilities.
- **Waning** — confidence decays with time since last reinforcement (30-day
  half-life), so heals never exercised again fade.
- **Forgetting** — `axiom immunity --prune` drops faded records (clonal
  deletion). Fleet merges combine confidence (stronger wins, immunizations sum).

```text
$ axiom run -- sh -c 'echo x > artifacts/o.bin'   # learn
  confidence: 0.50 (tentative, immunizations: 0)
# …after three successful reuses…
  confidence: 0.86 (established, immunizations: 3)
```

### Verifiable epistemic swarm (provenance + Beta beliefs)

Swarm-immunity exchange is tamper-evident and epistemically honest — combining
ideas absorbed from ChimeraLang (epistemics) and chimeralang-mcp (integrity):

- **Provenance (verify-before-trust).** `GET /v1/immunity` returns a *signed
  export* — the heal-memory payload wrapped with a full SHA-256 and, when
  `AXIOM_FLEET_KEY` is set, an HMAC-SHA256 for peer authentication. The merge
  endpoint and `axiom swarm immunity` verify the hash (and HMAC) **before**
  trusting a peer; a tampered payload or a wrong/missing key is rejected.
- **Beta-belief confidence.** Each heal's reliability is a `Beta(α,β)` belief
  carrying estimate *and* uncertainty. "Established" requires high mean AND low
  variance, so one lucky 1/1 success stays tentative; staleness decays the
  belief toward the uniform prior (uncertainty), not toward zero.
- **Dempster-Shafer merge.** Peer beliefs combine via DS evidence combination:
  agreeing peers compound evidence; irreconcilable peers raise a conflict and
  the local belief is kept (never silently averaged). A byzantine gate rejects
  fabricated-certainty peer beliefs.

### Inspecting acquired immunity

What Axiom has learned is queryable by both the operator and an AI agent:

```bash
axiom immunity            # everything learned: heals + per-program tension history
axiom immunity cargo      # filter by command substring
```

The same report is exposed to agents as the **`axiom_immunity`** MCP tool — so
Claude, debugging a command that fails in your environment, can ask Axiom what
it already knows about that program's failures and the heals it now applies.

### Closing the loop: runtime experience → reasoning context

With `AXIOM_RUN_REMEMBER=1`, a **novel** failure that the supervisor heals is
written into the recall memory store (`AXIOM_MEMORY_DIR`, default
`checkpoints/memory`) as a `Fix` memory — using the measured **tension (CE) as
its salience**. That is the same store `axiom_recall` / the proxy's recall
layer reads from, so a fault the *runtime* lived through becomes knowledge the
*reasoning layer* surfaces later:

```text
[axiom-run]   heal: created directory /build/dist
[axiom-run]   remembered fix for the reasoning layer (recall id=…)
# → "Program `…` failed (exit 2) and Axiom self-healed it: created directory
#    /build/dist. If this command fails again in this environment, apply that fix."
```

This is the bridge the project is named for: the self-healing runtime and the
cognitive layer share one memory.

### Active immunity in the proxy

The loop also runs *without anyone asking*. When a compressed `/v1/messages`
request references a command Axiom has already learned to heal, the proxy
injects a short `<axiom_immunity>` advisory into the outbound payload:

```text
<axiom_immunity>
Axiom has prior self-healing experience with commands referenced here:
- `cargo build` has failed in this environment before; Axiom's learned fix:
  create directory ./target. Apply preemptively if it fails again.
</axiom_immunity>
```

Matching is deliberately precise — a program-name + sub-command signature must
appear in the conversation **and** Axiom must hold a concrete learned heal for
it — so it never fires on prose or bare shell snippets. Disable with
`AXIOM_IMMUNITY_INJECT=0`.

Honesty notes: source-artifact patching lives in the Poly JIT hypervisor path,
not here; restarting a process is not literally resuming a suspended thread
(v1 targets batch/idempotent programs); and with an unbaked checkpoint the CE
sits near `ln(vocab)` (uniform) — the tension *plumbing* is always real, the
sharpness of the signal comes from the trained semantic model.

### Making the tension signal sharp (CPU training)

The tension/drift signal is only as discriminative as the model behind it. One
command trains a real, converged checkpoint on a commodity CPU (no GPU):

```bash
./scripts/train_cpu_quickstart.sh
```

It stages a corpus from the repo's own source + docs, trains a vocab-8000
ByteLevel BPE tokenizer, trains a d128/2-layer TTT model under early-stopping on
held-out cross-entropy, and runs the acceptance eval. A validated 4-core CPU run
(~17 min, step cap 4000):

| Metric | Value |
|---|---|
| val_ce (held-out, train split) | **4.93** (vs uniform `ln(8000)=8.99`) |
| held-out CE (unseen `server.rs`) | 4.41 |
| clean code CE | ~4.7 |
| **anomaly CE** (high-entropy fixture) | **9.74** |
| **drift separation margin** | **+4.93** → ACCEPTANCE **PASS** |
| recalibrated drift gate | 7.28 |

With this model active (`AXIOM_PRODUCTION_BPE=1`), a process-failure trace in
`axiom run` scores a real CE (~7.6, above the 7.28 gate → correctly flagged
anomalous) instead of the flat `ln(vocab)` of the bootstrap model — so the
FIRST/KNOWN/NOVEL classification and the drift gate become genuinely meaningful.
For a larger/lower-CE model, raise `AXIOM_DMODEL`/`AXIOM_NLAYERS`/`AXIOM_MAX_TOKENS`
(slower per step on CPU), or use `scripts/train_d384.sh` on a 6 GB GPU.

---

## The drivable hypervisor

The closed-loop hypervisor is exposed over the API, not just observable:

- `POST /v1/hypervisor/mount` + `POST /v1/hypervisor/read` — mount a directory
  into the safe user-mode Neural VFS and absorb files into a session's `W̃`
  incrementally (structural digest → TTT prefill), returning drift telemetry.
- `POST /v1/hypervisor/jit_run` — drive the **Poly JIT** closed loop: run a
  command; on failure feed the fault trace into `W̃` and apply a bounded,
  Q-TTT-ranked, **reversible** source patch, then retry. A given `source_path`
  is backed up first and **restored byte-for-byte** if the repair doesn't pass,
  so a failed attempt never corrupts the artifact. This is *source* healing —
  the complement to `axiom run`'s *environment* healing.
- `GET /v1/hypervisor/jit_status`, `/v1/hypervisor/quantum_coherent_state` —
  Poly JIT + Q-TTT manifold telemetry.

---

## Pillar 3 — Autonomy: `axiom solve`

The third pillar is the autonomous orchestrator that chains every subsystem into
one closed loop that drives a failing target to green and remembers how:

```bash
axiom solve --source src/lib.rs -- cargo test    # drive `cargo test` to green
```

Each round: (1) run the verify command under **environment self-healing**
(Pillar 2 — missing dirs / exec-bit / transients, with immunity + tension
absorption); if still red and `--source` is set, (2) drive **Poly JIT source
repair** (reversible, Q-TTT-ranked); (3) on success, what worked is persisted
(heal memory). The loop stops early when a round makes no progress, and on
failure the source is restored. One report unifies the provenance:

```text
[axiom-solve] round 1: environment heal solved it
[axiom-solve] result: SOLVED after 1 round(s); 1 env-heal(s); source_patched=false; …
# or, when code is the problem:
[axiom-solve] round 1: source repair solved it (patched=true)
```

Honest scope: environment heals are corrective; source repair is the bounded
deterministic patch set (Q-TTT-ranked, always reversible); whole-project Pillar-1
priming remains `axiom prime` (the supervisor absorbs each fault trace into `W̃`
during the loop).

---

## Grounding verification (hallucination flagging)

Axiom's honest answer to hallucination is **grounding verification**: flag the
claims in a response that aren't supported by the supplied evidence/context —
the material Axiom already absorbs. Default tier is deterministic and
model-free (lexical containment); confidence is a Beta belief carrying
uncertainty.

```bash
curl -s -XPOST localhost:3000/v1/verify -d '{
  "response": "Axiom uses online test-time training. It was funded by NASA in 1972.",
  "evidence": "Axiom is an inference engine with online test-time training ..."
}'
# → "It was funded by NASA in 1972" : UNSUPPORTED (flagged); grounded_fraction 0.5
```

Exposed as `POST /v1/verify` and the **`axiom_verify`** MCP tool (which an agent
calls before asserting facts from a document/codebase; it returns isError when
any claim is unsupported, so the agent notices).

### Grounding-gated expansion — saving tokens *while* reducing hallucination

The keystone that unifies Axiom's two goals: **the hallucination check controls
the token budget.** Compress aggressively (forward only the skeleton), then let
grounding decide where it was unsafe to drop detail:

```bash
curl -s -XPOST localhost:3000/v1/verify -d '{
  "response": "checksum computes the crc32 polynomial fold over the data.",
  "evidence": "<the lean skeleton: signatures only>",
  "session_id": "<compression session>", "expand": true
}'
# → grounded_fraction_before 0.0 → after 1.0
#   expanded_symbols: ["checksum"]   (only the claim's dependency was un-compressed)
```

For each claim the skeleton cannot ground, Axiom expands **only that claim's
referenced symbols** (via `/v1/expand` over the stored source) and re-verifies.
Tokens are spent back *surgically* — never across the board — so compression
stays maximal and precision is restored exactly where grounding proves it was
needed. (`verify_with_gated_expansion`.)

**Honest scope:** this checks *support against the supplied evidence*, not
universal fact-checking. The lexical tier flags **unsupported** claims (no
overlap) but — like every lexical verifier — does not reliably catch fluent
*contradictions* that reuse the evidence's wording (the
`verdict_contradiction_blind_spot` test pins this). An optional neural surprisal
tier (against the context-adapted W̃) is the next rung.

---

## Quick Start

### One-line install

Linux/macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/fernandogarzaaa/AXIOM-AETHER/main/scripts/install.sh | bash
```

Windows PowerShell:

```powershell
iwr https://raw.githubusercontent.com/fernandogarzaaa/AXIOM-AETHER/main/scripts/install.ps1 -UseB | iex
```

The installer downloads the latest GitHub Release binary, installs it as
`axiom`, and runs `axiom init` to scaffold `~/.axiom` or
`%USERPROFILE%\.axiom`.

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
