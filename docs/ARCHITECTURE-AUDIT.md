# AXIOM-AETHER — Architecture Audit (2026-08-23)

Scope: the whole repository at branch tip, reconstructed from the actual code,
tests, git history, and open branches/PRs — not from README claims. Every
"works" / "broken" / "wired" / "dead" verdict below was checked against the
source or an executable test; where it wasn't, it is explicitly marked
**unverified**. Numbers were measured on this pass (`cargo 1.94.1`,
`rustc 1.94.1`, CPU-only Linux):

| Axis | Result |
|---|---|
| `axiom_engine_rs` (Rust) | 51,482 lines across 92 top-level modules + `server/`, `bin/`, `cp1/` |
| `axiom_mesh_rs` (Rust, standalone workspace) | 3,204 lines across `axiom_core`, `axiom_prime`, `axiom_mcp` |
| `axiom_engine/` (Python reference) | 1,964 lines |
| `axiom_engine_rs` test suite | **812 passed, 0 failed**, 39 binaries (up from 688 at the 2026-07-24 audit) |
| `axiom_mesh_rs` test suite | **72 passed, 0 failed** |
| `cargo clippy --lib -- -D warnings` | clean |
| `cargo fmt --check` | **not** a CI gate; large pre-existing diff repo-wide (not treated as a defect here — see §8) |

This document supersedes nothing — [`docs/AUDIT_2026-07.md`](AUDIT_2026-07.md)
and [`docs/ARCHITECTURE-UNIFIED.md`](ARCHITECTURE-UNIFIED.md) are prior audits
of the same repository and are still accurate for what they cover (Docker
build, CI coverage, the AXIOM/EVE/ADAM ownership map, CP/1). This pass goes
deeper on the areas the mission brief asked about specifically: TTT/checkpoint
lifecycle, memory, self-healing, security boundaries, and concurrency — and
disposes of every branch/PR that was open and unmerged at the start of this
pass.

---

## 1. What AXIOM-AETHER actually is

Strip the terminology and the codebase is: **a local-first Rust HTTP/MCP
server that sits between an agent (Claude Code, Codex, a raw OpenAI/Anthropic
client) and either a local test-time-trained language model or a real
upstream provider, and does four things to the traffic passing through it**:

1. **Compresses context** — structurally (AST-aware skeleton extraction,
   deterministic, no model required) and adaptively (folds heavy context into
   a per-session fast-weight state and returns a short "memory fingerprint"
   instead of the raw text).
2. **Persists cross-session memory** — a lossless JSONL store of
   remembered facts (`memory_store.rs`/`memory_recall.rs`), separate from an
   EMA-merged persistent copy of the model's own adapted state
   (`vibe_memory.rs`).
3. **Repairs its own environment and, within a bounded scope, source code** —
   `self_heal.rs`/`solve.rs`/`poly_jit.rs`, gated by re-verification, never by
   trust.
4. **Grounds and forwards** — a hallucination/epistemic-drift check
   (`hallucination.rs`, `epistemic_drift.rs`) and a set of provider-compatible
   forwarders (`anthropic_forwarder.rs`, `openai_forwarder.rs`) so it can act
   as a drop-in proxy.

The conceptual pipeline in the mission brief (Context → Adaptation → Memory →
Verification → Execution → Provenance) is a **reasonable relabeling of what
already exists**, not a rewrite target. The actual call order in the hot path
(`server/routes_messages.rs::create_message`) is closer to: **Context
(compress) → Adaptation (TTT fast-weight update) → Memory (recall +
persist) → Execution (self-heal/repair, when invoked as `axiom solve`/`axiom
run`) → Verification (grounding/epistemic checks, opt-in) → Provenance
(session recorder, cost ledger, signed exports)** — Verification sits later
and is opt-in rather than a mandatory gate before Execution. That is a real,
minor divergence from the target diagram, not a contradiction of it; see §7.

---

## 2. CORE / EXPERIMENTAL / INTEGRATION / INFRASTRUCTURE / DEAD

| Tier | Modules | Basis |
|---|---|---|
| **CORE** | `inference.rs`, `ttt_block.rs`, `ttt_mlp*.rs`, `context_compressor.rs`, `responses_compressor.rs`, `prefix_diet.rs`, `skeleton.rs`, `digest.rs`, `memory_store.rs`, `memory_recall.rs`, `vibe_memory.rs`, `self_heal.rs`, `solve.rs`, `poly_jit.rs`, `provenance.rs`, `cost_ledger.rs`, `session_recorder.rs`, `server/*`, `mcp_stdio.rs`, `backend_router.rs`, `*_forwarder.rs`, `hallucination.rs`, `epistemic_drift.rs` | Reachable from the shipped binary's default code paths, covered by the bulk of the 812 tests, load-bearing for the README's headline features |
| **INTEGRATION** | `axiom_mesh_rs/axiom_core` (consumed by `mesh_router.rs`, opt-in `AXIOM_MESH_ROUTING=1`), `axiom_mcp/` plugin surface, `.mcp.json`, Docker/Helm/K8s deploy | A real dependency edge exists, but every consumer is opt-in and off by default |
| **EXPERIMENTAL** | `axiom_mesh_rs/axiom_prime` + `axiom_mesh_rs/axiom_mcp`'s sidecar/worker system (the multi-agent orchestrator — FSM, `aether_worker`, `MeshSupervisor`), `chimera.rs` (1,226-line ChimeraLang DSL port), `hamiltonian.rs`/`q_manifold.rs` (Q-TTT — a classical, deterministic tensor-network simulator the code itself calls "not literal quantum computing"), `predictive_tools.rs`/`state_predictor.rs`/`trajectory_sampler.rs` (untrained heads, self-report `trained: false`), `axiom_engine/` (Python reference implementation) | Real, tested code with no load-bearing caller in the shipped product; each is honestly labeled in its own doc comments or API responses |
| **DEAD / REDUNDANT** | `memory_pool.rs` (one-line stub, zero callers) | Removed in this pass — see §8 |

**Why this mapping, not a bigger rewrite:** every "EXPERIMENTAL" item above
already documents its own status accurately in-repo (untrained-state flags in
API responses, "classical, deterministic" in the Q-TTT module doc, "reference
implementation, not the shipped product" in `axiom_engine/README.md`). The
gap is not that these are mislabeled internally — it's that none of that
honesty surfaces in the top-level README's feature list, which is fixed in
§9/final report. Moving these into a literal `experimental/` directory would
be a large, purely cosmetic diff for no behavior change; the recommendation
(§7 of the roadmap) is to do it opportunistically, not as a one-shot rename.

---

## 3. Subsystem-by-subsystem

### 3.1 Context / Compression

- **Structural**: `skeleton.rs` builds an AST-aware digest (signatures kept,
  bodies elided); `bench.rs`/`axiom bench <path>` measures round-trip fidelity
  deterministically, no model required. Verified this pass: `cargo build
  --release` + `axiom bench axiom_engine_rs/src` reproduces the
  bench/RESULTS.md numbers structurally (compressor logic unchanged this
  pass) — see [AXIOMBENCH.md](AXIOMBENCH.md) for what "verified" means here.
- **Adaptive**: `context_compressor.rs`'s `TttSessionStore` holds one
  `Arc<AsyncMutex<Vec<Tensor>>>` fast-weight state per session in a
  `DashMap`; `adapt_session_blocking`/`extract_memory_vector_blocking` fold
  context into it and return a `MemoryFingerprint` (state hash + per-layer
  Frobenius norms + top-k recall tokens + a `confidence_tier()` that
  downgrades on non-finite norms or low-signal decoded output). This is a
  real, tested contract — `MemoryFingerprint::confidence_tier` is exactly the
  kind of "don't oversell it" guardrail the mission brief asks for elsewhere.
- **Concurrency finding (fixed in this pass, §8):** `responses_run_fingerprint`
  fans independent runs out via a bounded `tokio::task::JoinSet`
  (`server/routes_responses.rs:78`), but every fanned-out task's *entire*
  compute (`encode_text` → `adapt_session_blocking` →
  `extract_memory_vector_blocking`) held one process-wide
  `std::sync::Mutex<InferencePipeline>` for its full duration. The JoinSet's
  parallelism was real in shape and fake in effect — every task serialized on
  the same writer-only lock regardless of how many were spawned. This is
  fixed; see §8.

### 3.2 Adaptation (TTT)

- `ttt_block.rs`/`ttt_mlp.rs`/`ttt_mlp_model.rs` implement the fast-weight
  update as a **closed-form** step (no `.backward()` — see the comment at
  `context_compressor.rs:337`), with an explicit **autograd-truncation**
  (`.detach()` after every chunk) to stop an unbounded computation graph from
  crashing the process on long corpora — a real, previously-hit bug the
  comment documents by name (24k-token saturation run).
- **Numerical safety is real, not aspirational.** `generate_with_session`
  (`inference.rs:224`) snapshots `states` before each token and **discards
  the update and restores the snapshot** if `session_states_are_finite`
  fails, logging `[emergency] non-finite state detected`. `vibe_memory.rs`
  independently rejects non-finite master state on load *and* on EMA merge
  (tests: `commit_session_rejects_non_finite_merge_result`,
  `load_or_init_rejects_non_finite_file` — both green). This is two
  independent NaN/Inf firewalls at two different layers (session state,
  persistent master state), not one.
- **`InferencePipeline` has zero `&mut self` methods** (checked: `grep -n
  "fn .*(&mut self" src/inference.rs` → no matches). Every adaptation
  mutates an explicitly-threaded `states: Vec<Tensor>` parameter, never the
  pipeline itself. This is the fact that makes the RwLock fix in §8 sound,
  and it is also good evidence the "TTT adaptation" is architecturally a
  pure function of `(frozen base weights, session state) → (output, new
  session state)` — closer to a well-factored inference server than to an
  online-training system with global mutable state.
- **Does adaptation demonstrably influence output?** Yes, mechanically —
  `adapt_session_blocking` changes `states` in place and every subsequent
  `forward_lm` call in the same session consumes the changed `states`; tests
  like `repeated_token_drives_reconstruction_error_down` and
  `passkey_recall_breaks_past_zero_after_convergence` (a 223-second test,
  still green) exercise this end-to-end. What is **not** established anywhere
  in this repo is that this changes *downstream agent task success* — see
  [AXIOMBENCH.md](AXIOMBENCH.md). The engine-level mechanism works; the
  product-level "does it make agents better" claim is unmeasured.

### 3.3 Checkpoint lifecycle

Training → checkpoint → metadata → compatibility → distribution → integrity
→ loading → runtime, traced end to end:

- **Training**: `train.rs`/`meta_train.rs`/`bin/train_semantic.rs` etc.
- **Metadata**: `model_meta.rs`'s `ModelMeta` sidecar (`<ckpt>.meta.json`)
  records `d_model`/`n_layers`/`vocab_size`/`val_ce`/tokenizer id plus three
  architecture-affecting flags (`stabilize`, `last_token_only`,
  `learned_gate`) with `#[serde(default)]` so old sidecars still load. This
  *is* explicit compatibility metadata — dimension mismatches fail at tensor
  construction, not silently.
- **Distribution**: `config::ensure_production_checkpoint` (very recent —
  merged 2026-08 per git log) fetches a checkpoint from
  `AXIOM_CHECKPOINT_URL` over plain `reqwest::blocking::get`.
  **Before this pass, there was no integrity verification of the download at
  all** — a corrupted or substituted release asset would install silently.
  **Fixed in this pass**: optional `AXIOM_CHECKPOINT_SHA256` /
  `AXIOM_TOKENIZER_SHA256` pinning that fails closed (deletes the partial
  download, returns `Err`) on mismatch, and always logs the computed digest
  so an operator can capture and pin it. See §8 and
  [SECURITY-AUDIT.md](SECURITY-AUDIT.md).
- **Fresh-install cliff, already closed**: `bootstrap.rs` trains a small
  model locally and offline (procedural dataset, no network, seconds) on
  `axiom init` whenever no checkpoint exists yet, specifically so — per its
  own doc comment — "the neural recall fingerprint is [not] pure noise." This
  directly answers the mission brief's "a learned system is not
  production-ready if fresh installations silently use random/untrained
  state" concern: verified true positive, not a gap. Test:
  `ensure_checkpoint_trains_then_is_noop` (green), which asserts the trained
  checkpoint actually loads back into a dimension-correct pipeline.
- **Gap that remains** (P2, not fixed this pass): no version/architecture
  compatibility matrix beyond the three boolean flags above — a checkpoint
  trained with a future flag `ModelMeta` doesn't know about loads silently
  with that feature's default behavior rather than a clear "this checkpoint
  needs a newer binary" error. Low risk today (only three such flags exist),
  worth a `min_engine_version` field before a fourth is added.

### 3.4 Memory

Five modules, genuinely different designs, not accidental duplication (this
matches [ARCHITECTURE-UNIFIED.md](ARCHITECTURE-UNIFIED.md) gap G-2's own
conclusion, confirmed again here by reading each module's actual contract):

| Module | What it stores | Unit |
|---|---|---|
| `memory_store.rs` + `memory_recall.rs` | Lossless, user-facing facts (`axiom_remember`/`axiom_recall`) | `MemoryRecord { scope, kind, body, embedding, drift_at_ingest }`, JSONL, one file/scope |
| `vibe_memory.rs` | EMA-merged fast-weight tensors — the model's *numeric* accumulated state, not text | `Vec<Tensor>` per layer |
| `heal_memory.rs` | Environment-repair experience (missing dirs, permission fixes), keyed by a command fingerprint | `ProgramRecord { dirs, confidence, ce_history }` |
| `patch_memory.rs` | Verified source-code fixes, locally *and* fleet-shared | `Candidate { sha256, content, ... }`, trust boundary: never applied without a fresh local verify (see §3.5) |
| `graph_memory.rs` **(new, merged this pass — see §8)** | Directed, weighted edges between `MemoryRecord` ids (`Supersedes`, `CausedBy`, `GeneralizesTo`, `Contradicts`, `CoOccurred`, …) + bounded spreading activation | `MemoryEdge`, append-only `edges.jsonl` |

**Graph-memory branch disposition** (mission brief §5 explicitly asks for
this): `claude/graph-b1-edge-store` (PR #143) and `claude/graph-b2-spread`
(PR #144, superset of #143) were open, draft, unmerged. Read in full,
compared against `main`, and unit-tested in isolation (14/14 pass, including
cycle-safety, tombstone handling, determinism, and a `max_visited`-bound
stress case over 1,000 edges). Verdict: **complete and correct as a
building block, not yet integrated** — `graph_memory.rs` is a standalone
module; nothing in `memory_recall.rs` or the MCP tool surface calls
`spread()` yet to widen a recall result. Disposition: **merged** (§8) because
it is additive-only (one new file + one `pub mod` line, zero touches to
existing files, zero behavior change until something calls it), well-tested,
and directly reusable; **wiring it into `axiom_recall`/`axiom_remember` is
left as the next PR** (tracked in [ROADMAP.md](ROADMAP.md)) rather than done
here, because that integration decision (which edge kinds to seed
automatically, how much spread to add to a recall query, how it interacts
with the existing recency/supersession rerank) deserves its own review, not
a rider on an audit pass.

### 3.5 Self-healing

Full trace, failure → diagnosis → repair → verification → persistence →
reuse:

1. **Diagnosis**: `self_heal.rs`'s `extract_missing_paths`/
   `extract_permission_denied_paths`/`extract_required_env_vars` parse a
   failure trace with plain string/regex matching — deterministic, no model.
2. **Repair (environment)**: `heal_missing_path`/`heal_permission_denied`
   create the directory / `chmod +x` the file. Scope is deliberately narrow
   — this is not general code repair.
3. **Repair (source)**: `solve.rs`'s `solve()` only reaches this tier if
   environment healing didn't suffice. It drives `poly_jit.rs`'s bounded
   (`MAX_POLY_JIT_STEPS = 3`) loop: run → on failure, `synthesize_patch` a
   deterministic candidate (or an LLM-proposed one, gated the same way) →
   apply → **re-run the real verify command** → keep only if it now passes.
4. **Verification**: the *same* command that originally failed is the judge,
   every time — `run_verify`/`run_verify_capture`. There is no separate
   "trust the patch because an LLM said so" path anywhere in this code:
   `solve.rs`'s own module doc states the acceptance rule and the tests
   (`solve_reports_unsolved_and_restores_source_on_failure`,
   `apply_verified_patch_iterative`) enforce it.
5. **Rollback**: a source file is **backed up before the first attempt and
   restored byte-for-byte** if the loop ends without a pass — both in the
   library path (`solve.rs`) and independently in the HTTP path
   (`routes_hypervisor.rs::hypervisor_jit_run`, tested by
   `jit_run_restores_source_when_repair_fails`). Two independent
   implementations of the same safety property is a point in this
   subsystem's favor, not a duplication concern — they protect two different
   entry points (CLI `axiom solve` vs. the HTTP hypervisor route).
6. **Persistence + generalization**: `finalize_memory`/`remember_failure_fix`
   write the successful heal back to `heal_memory.rs`. **Cross-program
   generalization is real and shipped this pass** (§8: merged from
   `claude/generalizable-heal-rules`, discovered to already be on `main`
   under different commit hashes — see §8 for the exact disposition): a
   directory heal only generalizes to a program that has *never run before*
   once **two independent, unrelated programs** have each independently
   needed it (`STRUCTURAL_COROBORATION_THRESHOLD = 2`), and the successful
   structural application becomes that third program's own earned
   experience too, not a permanent dependency on the corroborating records.
7. **Regression risk**: bounded by construction — every repair loop (env
   heal, source patch, HTTP `jit_run`) is gated by an independent re-run of
   the real verify command before being kept, and non-passing attempts are
   discarded/restored. What is **not** covered: false-negative structural
   generalization (a directory two unrelated programs both happened to need
   for unrelated reasons gets pre-created for a third — harmless by
   construction here since it only ever *creates* a directory, never removes
   or overwrites one, but the corroboration threshold of 2 is a tunable,
   unvalidated constant, not something calibrated against real repair data).
8. **Isolation**: **none, by design and by current implementation** — see
   [SECURITY-AUDIT.md](SECURITY-AUDIT.md) §"Execution surfaces." Every verify
   command, patch check, and JIT run executes as a real child process of the
   Axiom server/CLI, with the process's own filesystem/network access. The
   safety properties above (verify-gated, reversible, bounded) are real and
   valuable, but they are **not** an isolation boundary — a verify command
   that itself does something destructive (not just fails) is not contained
   by anything in this subsystem.

### 3.6 Mesh / distributed routing

`axiom_mesh_rs` is a genuinely separate, standalone Cargo workspace (own
`Cargo.toml`, doesn't build from the repo root) with 72 passing tests
including property-based invariants (`proptest`). Only `axiom_core` (the
`KineticNeuralMesh` routing primitive) has a real dependency edge into the
shipped product, via `mesh_router.rs`, and only when `AXIOM_MESH_ROUTING=1`.
`axiom_prime` (the FSM orchestrator) and `axiom_mcp` (the sidecar/worker
system, `aether_worker`) are a complete, well-tested, **standalone** research
system with no caller in `axiom_engine_rs` at all — they run their own demo
binary (`cargo run --bin axiom_prime`) and nothing else. This is EXPERIMENTAL
by the same test the mission brief proposes: real dependency relationships
exist for `axiom_core`, don't exist for `axiom_prime`/`axiom_mcp`.

---

## 4. Concurrency and performance (fixed this pass)

**Before this pass**: `AppState.pipeline` and `McpContext.pipeline` were
`Arc<std::sync::Mutex<InferencePipeline>>`. Every one of the 27 call sites
that touch it across `mcp_stdio.rs`, `server/routes_{sessions,messages,
responses,verify}.rs`, `lsp.rs`, `backend_live.rs`, and `vfs.rs` only ever
calls `&self` methods on `InferencePipeline` (confirmed: zero `&mut self`
methods exist on the type at all). The `Mutex` therefore serialized every
concurrent request through the pipeline — including the bounded `JoinSet`
fan-out in `responses_run_fingerprint` (§3.1) that was specifically built to
run runs in parallel.

**Fixed**: `Arc<Mutex<InferencePipeline>>` → `Arc<RwLock<InferencePipeline>>`
everywhere (`std::sync::RwLock`, matching the existing pattern already used
for `AppState.sessions`), `.lock()` → `.read()` at all 27 sites (one
`.try_lock()` → `.try_read()` in the `/readyz` probe). This was mechanical
and fully compiler-verified: since `InferencePipeline` has no `&mut self`
surface, there was no call site that could legitimately need `.write()` —
`cargo check --all-targets` caught the one place that needed fixing (a test
helper's explicit `Mutex` type annotation) on the first try, with zero
follow-up errors. Confirmed with a new regression test
(`server::tests::pipeline_lock_allows_concurrent_readers`) that a second
`try_read()` succeeds while a first `read()` guard is still held — this
would fail immediately if the field were ever changed back to a `Mutex`.

**Why not the full `TokenizerService`/`ModelRuntime`/`AdaptationRuntime`/
`SessionStateStore`/`InferenceExecutor` service split the mission brief
raises as an option**: the investigation above shows the actual contention
was a **lock-granularity bug**, not a **service-boundary problem** — the
session state was already correctly factored out into
`TttSessionStore`/`DashMap` + per-session `AsyncMutex`, and the tokenizer is
already just a read-only field on an otherwise-immutable pipeline. A 5-service
decomposition would restate an architecture that already exists in
substance, at real cost (new trait boundaries, new inter-service call
overhead, a much larger and riskier diff) for no additional concurrency this
repo doesn't already have once the lock type is corrected. Recommending it
anyway would be exactly the kind of unjustified growth the mission brief
warns against in its closing instruction.

---

## 5. Security boundary (see [SECURITY-AUDIT.md](SECURITY-AUDIT.md) for the full audit)

Headline: `POST /v1/hypervisor/jit_run` executed an arbitrary caller-supplied
command with zero capability gate beyond the general (opt-in, off-by-default)
`AXIOM_API_KEY` — on a server that binds `0.0.0.0` by default. This is fixed
in this pass with a dedicated, off-by-default `AXIOM_ENABLE_JIT_EXEC`
capability gate, independent of `AXIOM_API_KEY`. Full writeup, threat model,
and the rest of the execution-surface inventory (sandbox.rs naming, poly_jit,
self_heal's subprocess execution, patch trust model) are in
[SECURITY-AUDIT.md](SECURITY-AUDIT.md).

---

## 6. Terminology map

| Legacy/internal term | Canonical meaning | Where it lives |
|---|---|---|
| Vibe memory | Persistent EMA-merged adaptation state | `vibe_memory.rs` |
| Heal memory / immunity | Learned environment-repair experience | `heal_memory.rs` |
| Patch memory | Verified source-fix substrate (local + fleet) | `patch_memory.rs` |
| CVM | Cost/verification metering store (the "cost stack") | `cvm_store.rs`, `cost_ledger.rs` |
| DWE | Delta-weight exchange (fleet immunity fragments) | `dwe.rs` |
| PSS | (Local-answer short-circuit gate; see `routes_messages.rs`) | `routes_messages.rs` |
| Hypervisor | The Poly JIT + Neural VFS surface (`/v1/hypervisor/*`) — **not** a real hypervisor (no VM/OS isolation); the name predates this audit and is kept for API stability | `server/routes_hypervisor.rs` |
| Sandbox (`sandbox.rs`) | A `cargo check`-only compile verifier in an isolated temp directory — **not** a process/OS security boundary (no seccomp, no container, no resource limits beyond a wall-clock timeout) | `sandbox.rs`; see SECURITY-AUDIT.md for the naming recommendation |
| Q-TTT / Hamiltonian / quantum manifold | A classical, deterministic tensor-network simulator using physics-inspired optimization terminology (imaginary-time evolution, variational collapse) — the module doc itself says "not literal quantum computing" | `hamiltonian.rs`, `q_manifold.rs` |
| Axiom Mesh (`axiom_mesh_rs`) | A separate, standalone control-theoretic multi-agent orchestration research workspace; only its `axiom_core` routing primitive has a real (opt-in) dependency edge into the shipped product | `axiom_mesh_rs/` |
| ChimeraLang | A ported cognition-language DSL, integrated but not load-bearing for any shipped default path | `chimera.rs` |

No public API was renamed as part of this pass — per the mission brief,
terminology clarity was prioritized over cosmetic renames. This table is the
map a new contributor needs; it belongs in the top-level README's
architecture section too (done — see the README diff in this change set).

---

## 7. Does the shipped architecture match the target diagram?

Mostly, with one honest divergence: **Verification is opt-in and sits after
Execution in the actual default request path**, not as a mandatory gate
before it, the way the target diagram's linear Context → Adaptation → Memory
→ Verification → Execution → Provenance ordering implies. Concretely:
`AXIOM_VERIFY_RESPONSES` (grounding advisories) and `/v1/verify` (hallucination
support-checking) are both opt-in and additive; nothing in the default
`/v1/messages` or `/v1/responses` path blocks a response pending a
verification pass. This is a defensible product choice for a
latency-sensitive proxy (mandatory verification would add a full second
inference pass to every request), but it does mean "Verification" in the
target diagram currently describes an available capability, not an enforced
stage. Recorded here rather than silently reconciled, per the mission
brief's instruction not to force the repository into the target framing if
the code contradicts it.

---

## 8. Changes made in this pass

1. **`AXIOM_ENABLE_JIT_EXEC` capability gate** for `POST
   /v1/hypervisor/jit_run` (P0 — see SECURITY-AUDIT.md).
2. **`Arc<Mutex<InferencePipeline>>` → `Arc<RwLock<InferencePipeline>>`**
   across all 27 call sites (P1 concurrency fix, §4).
3. **Checkpoint download integrity**: `AXIOM_CHECKPOINT_SHA256` /
   `AXIOM_TOKENIZER_SHA256` optional pinned verification, fail-closed on
   mismatch (P1, §3.3).
4. **Merged `graph_memory.rs`** (edge store + bounded spreading activation)
   from the stale `claude/graph-b2-spread` branch — additive only, 14/14
   tests green (§3.4).
5. **Deleted `memory_pool.rs`** — a one-line dead stub with zero callers
   anywhere in the codebase.
6. **Branch/PR disposition** (mission brief §5/§22 — inspect every unmerged
   implementation, decide merge/rewrite/supersede/abandon):

   | Branch | PR | Disposition | Evidence |
   |---|---|---|---|
   | `claude/graph-b1-edge-store`, `claude/graph-b2-spread` | #143, #144 | **Merged** (this pass) | New, additive, tested; see §3.4 |
   | `claude/generalizable-heal-rules` | — | **Already landed on `main` under different commits** — content is byte-identical (`STRUCTURAL_COROBORATION_THRESHOLD`, `structural_dirs`, `StructurallyImmunized` all present); branch is stale, no action needed | Diffed branch tip against `main`: zero remaining diff |
   | `claude/axiom-aether-audit-e66aji` | — | **Superseded** — `docs/AUDIT_2026-07.md` on this branch is byte-identical (`md5sum` match) to what's already on `main` | |
   | `claude/interactive-tui` | — | **Superseded** — `tui.rs` is byte-identical to `main`'s | |
   | `claude/mesh-router-integration` | — | **Superseded** — `mesh.rs`/`axiom_prime/main.rs` are byte-identical to `main`'s current versions | |
   | `claude/axiom-swarm-architecture-ntenzk` | — | **Superseded** — `hierarchy.rs` byte-identical to `main`'s; branch otherwise 87 files / −11,519 lines stale relative to `main`'s later CP/1 work | |
   | `claude/docs-audit-fixes` | — | **Stale, low value** — 95 lines of documentation diff remain against current `main`; not reviewed line-by-line in this pass, left for a future docs-only PR | |
   | `cvm/s8-rollout` | — | **Stale** — a version-bump-only branch (0.4.0) superseded by `main`'s current 0.4.1 | |
   | `docs/audit-2026-06` | PR #81 | **Stale, superseded by `docs/AUDIT_2026-07.md`** — orphan branch (no merge-base with `main`) | |

   No branches were force-deleted or PRs force-closed in this pass — that is
   a destructive, visible action better left to the repository owner; the
   table above is the recommendation.
7. **This document, plus [SECURITY-AUDIT.md](SECURITY-AUDIT.md),
   [COMPETITIVE-ANALYSIS.md](COMPETITIVE-ANALYSIS.md),
   [AXIOMBENCH.md](AXIOMBENCH.md), and [ROADMAP.md](ROADMAP.md).**

All changes are covered by the existing test suite plus new regression tests
added alongside them (concurrency: `pipeline_lock_allows_concurrent_readers`;
capability gate: `jit_run_endpoint_is_disabled_by_default`; checksum:
`verify_checksum_{passes_when_nothing_pinned,passes_on_case_insensitive_match,
fails_on_mismatch_with_both_hashes_in_message}`). Full suite: **812/812
(engine) + 72/72 (mesh) passing, 0 failed**, `clippy --lib -D warnings` clean.
