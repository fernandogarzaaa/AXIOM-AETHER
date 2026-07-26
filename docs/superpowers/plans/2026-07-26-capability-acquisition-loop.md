# Capability Acquisition Loop — the next evolution of Axiom

> **Status (2026-07-26): BRAINSTORM.** Nothing here is implemented. This document
> exists to argue for a shape, name the prerequisites honestly, and record what
> is already built so we do not pay twice for it. No numbers in this document are
> measurements — there are no measurements yet, because there is no code yet.

## The proposal in one sentence

Axiom already runs a closed loop that turns a *failure* into durable, shareable
immunity. The next evolution is to run the same loop for *capability*: turn a
missing tool into a verified, provenance-tracked, fleet-shareable artifact — and
upgrade memory from a flat list to a graph so the things it acquires compose.

## Where the three seed ideas actually stand

The originating brainstorm proposed three things. An audit of the tree says they
are at very different maturities, and the dependency order between them is the
opposite of the order they were proposed in.

| Seed idea | Real status | Evidence |
|---|---|---|
| 1. A sandbox for the model | **Exists in name; is not an isolation boundary** | `sandbox.rs` |
| 2. Search GitHub/libraries and synthesize a new tool | **Greenfield** | no crates.io / docs.rs / code-search path exists anywhere in `src/` |
| 3a. Per-token weight adaptation on a small LM | **Shipped and validated** | `ttt_block.rs`, `docs/UPGRADES.md` |
| 3b. Graph working memory | **Greenfield** | `memory_recall.rs` is flat cosine top-k |
| 3c. Hardware-aware resource allocation | **Exists, but boot-time only** | `hardware.rs` |

### 1. The sandbox is a temp directory, not a sandbox

`sandbox.rs:110` creates an isolated temp Cargo package, runs
`cargo check --message-format=json` against it (`sandbox.rs:200`), captures
diagnostics, and feeds them to a TTT feedback closure. That is a useful
*compile-checking* harness and it works. It is not isolation:

- **No isolation primitives at all.** No namespaces, no seccomp, no cgroups, no
  network cutoff, no filesystem confinement. It is `tokio::process::Command` in
  a temp dir with a 20-second timeout (`sandbox.rs:18`).
- **Rust only.** `is_rust_language` (`sandbox.rs:264`) accepts `rust`/`rs` and
  drops every other fenced block on the floor.
- **`cargo check` only.** It never builds, never runs, never tests. A synthesized
  tool that type-checks and is semantically wrong passes cleanly.
- **The blocker for idea 2:** the generated manifest at `sandbox.rs:287` is
  hardcoded with **no `[dependencies]` section**. Any harvested code that imports
  a third-party crate cannot compile. Frankenstein-ing library code is
  structurally impossible in today's sandbox, and no amount of clever search
  fixes that.

Worse, the neighbouring executor is unconfined: `poly_jit.rs:213` runs
`Command::new(&request.command)` with caller-supplied args and working directory,
and `poly_jit.rs:187` writes patched bytes straight to a real `source_path` on
disk.

**This is the load-bearing risk in the whole proposal.** Idea 2 is
"download code from the internet and run it." Layering that on top of an
unconfined executor is a supply-chain incident with extra steps. Real isolation
is not phase-two polish; it is the admission ticket.

### 2. Synthesis is greenfield, and the hard half is not the search

A grep for `crates.io`, `docs.rs`, and GitHub code search across
`axiom_engine_rs/src/` returns nothing. What exists is adjacent but different:
`search_ingest.rs:1` scrapes **web prose**, streams it through TTT, and emits an
`<axiom_search_fingerprint>` semantic pointer. That is a retrieval primitive for
natural language, not a code-acquisition path.

Two naming notes before anyone writes a line:

- **`chimera` is taken.** `chimera.rs` (1,226 lines) is the in-tree ChimeraLang
  DSL — `belief/inquire/resolve/guard/evolve` over the `BetaBelief` substrate.
  The synthesis subsystem needs a different name. This document uses **Forge**.
- **`harvest` is taken** by `src/bin/harvest.rs`.

The seductive framing is "search GitHub and stitch." The framing that survives
contact with reality is: *searching is the easy half; knowing whether the result
is correct is the hard half.* A synthesis loop with no verification story
manufactures code that compiles and is wrong, at machine speed, and writes it
into durable memory where it poisons every downstream use.

Fortunately the repo already solved this exact problem for patches, and the
solution is directly reusable. `patch_memory.rs:9` states the invariant:

> **A patch received from a peer is NEVER applied on trust.** It is only ever
> written through `PatchMemory::try_candidates`, which writes the candidate, runs
> *this node's own* verify check, and keeps the patch **only if it goes green
> locally** — otherwise it is rolled back byte-for-byte.

Harvested code is exactly a peer patch with a wider blast radius. Same invariant,
no exceptions.

### 3a. Per-token adaptation is the thing Axiom already is

`ttt_block.rs:22`: "For every incoming token the block performs one
self-supervised gradient step on W_tilde before producing output, achieving O(1)
memory per inference step." On top of that there is a scalar Gated-DeltaNet
forget gate, a *learned* data-dependent per-token gate
(`α_t = α_min + (1−α_min)·σ(w_α·x)`, warm-started near α≈1 at
`ttt_block.rs:GATE_INIT_LOGIT`), and three online-update guards drawn from the
TTA literature (RDumb periodic reset, EATA sample selection, Fisher anchoring).

`docs/UPGRADES.md` carries the de-noised validation: held-out CE −2.4% on unseen
code with the learned gate, at a small, honestly-reported cost to the
drift-separation margin (+3.695 → +3.548) — which is why it ships opt-in.

**So "a small language model that trains on the fly and adapts its weights per
token" is not the next evolution. It is the current one, and it is measured.**
The real deficiency is orthogonal: `W_tilde` is *per-session and ephemeral*.
Everything the engine learns while adapting evaporates when the session drops.
The graph is the thing that makes it persist and compose.

### 3b. Memory is a list where it wants to be a graph

`memory_recall.rs:43` gathers the union of requested scopes into a flat `Vec`,
filters records superseded by a present record, and runs brute-force cosine
`top_k`. That is a competent tier-2 retriever and it returns real stored bodies,
never reconstructions (`memory_store.rs:1`) — a good property worth keeping.

But there is already a **single edge type hiding in the schema**:
`MemoryRecord.supersedes: Option<String>` (`memory_store.rs:31`). Supersession is
a directed edge, stored, used at recall time, and today consumed only as a
tombstone filter (`memory_recall.rs:60`). The graph is not a rewrite. It is the
generalization of a field that already exists.

### 3c. Hardware awareness is a boot-time doctor, not a scheduler

`hardware.rs:97` is a genuinely good pure function: a hardware snapshot in, a
safe recommendation out, fully unit-testable on a GPU-less CI box. It encodes
three hard-won rules — the co-tenancy guard (`hardware.rs:122`, born from an
observed `CUDA_ERROR_OUT_OF_MEMORY` on a 6 GB RTX 2060), a free-VRAM floor, and
OS core reservation so training cannot starve the display compositor.

The gap named in the seed idea — *"allocate resources based on the task"* — is
real and precisely stated. `recommend()` runs once, against a snapshot, and
produces one global answer for the whole process. There is no per-request
arbitration, no admission control, no re-decision when conditions change.

On "OS & hardware-native primitives": the repo has an established and correct
instinct here that should be respected rather than overridden. `vfs.rs:1` —
"This is intentionally not a kernel driver." `poly_jit.rs:1` — "The runner does
not suspend or patch arbitrary OS thread instruction pointers." Both modules were
asked for kernel-level power and shipped the safe user-mode equivalent. The
proposal below stays inside that discipline: cgroups v2, seccomp-bpf, Landlock,
user namespaces, io_uring, NUMA pinning. All real OS primitives. All user-mode.
No driver, no admin install, no `sudo` in the happy path.

---

## The reframe: one loop, run for a second purpose

`solve.rs:1` describes Pillar 3 as the autonomous orchestrator that chains every
subsystem into one closed loop:

> environment self-heal → source repair → verify → **remember**

Seed ideas 1 and 2 together are that same loop pointed at a different target:

> capability gap → harvest → synthesize → **verify in a real jail** → **remember**

That is the whole insight, and it is worth a lot, because it means most of the
machinery is already written and paid for. `self_heal.rs`, `heal_memory.rs`,
`patch_memory.rs`, `provenance.rs`, `belief.rs`, and `solve.rs` are all reusable
as-is. The genuinely new work is three things: a sandbox that is actually a
sandbox, a code harvester with a provenance and license gate, and a graph to
remember into.

Seed idea 3's graph is not a third parallel feature. It is the **substrate** that
makes "remember" compositional instead of flat — the difference between a pile
of acquired tools and a capability set that knows how its parts relate.

---

## Phase A — Make the sandbox real (prerequisite for everything)

Nothing downstream is safe until this lands. It is unglamorous plumbing and it is
the highest-priority item in this document.

**A1. Isolation.** Wrap sandbox execution in real confinement rather than a temp
directory. On Linux: user namespaces + mount namespace (private `/tmp`,
read-only everything else), **network namespace with no interfaces** (the default
for every synthesis build — see A2), seccomp-bpf syscall filter, Landlock for
filesystem reach, and cgroups v2 for `memory.max` / `pids.max` / `cpu.max`.
On Windows: job objects + restricted tokens as the equivalent (the repo cares
about Windows — `.bat` launchers, `wmic` probing in `hardware.rs:262`,
PowerShell fixtures in `poly_jit.rs:273`). Degrade explicitly and loudly when a
primitive is unavailable; never silently fall back to "no isolation."

**A2. Dependency-capable builds.** This is what unblocks idea 2. Vendor
dependencies during a *networked fetch phase*, then build and run in the
*network-off phase*. Concretely: resolve → `cargo vendor` into the jail →
`.cargo/config.toml` pointing at the vendored offline registry → build with no
network namespace. The fetch phase is the only point where the internet is
reachable, it is auditable, and it produces a lockfile that goes into the
artifact's provenance record.

**A3. Verify, do not merely check.** `cargo check` proves types, not behavior.
The sandbox needs `build` + `test` + a bounded `run` with captured stdout/stderr
and a wall-clock and memory ceiling.

**A4. Beyond Rust.** Generalize past `is_rust_language`. Python and TypeScript
are the two that matter for tool synthesis in practice, and both need the same
vendored-offline treatment (`pip download` / `npm ci --offline`).

The clean bonus: **cgroups v2 is simultaneously the isolation mechanism and the
per-task resource allocation mechanism.** A1 and Phase D are the same
primitive viewed from two directions, which is why they should share a
`ResourceArbiter` rather than growing two independent resource models.

## Phase B — Graph working memory (highest value-to-risk, independent of A)

This can proceed in parallel with A, touches nothing dangerous, and is the
largest capability gain per line of code.

**B1. Typed edges over the existing store.** Keep `MemoryRecord` and the
append-only JSONL durability model (`memory_store.rs:37`); add an append-only
edge log with an in-memory adjacency index built at load. Edge types, each of
which already exists implicitly somewhere in the tree:

| Edge | Source in today's code |
|---|---|
| `supersedes` | `MemoryRecord.supersedes` — already stored, already used |
| `caused_by` | fault trace → heal, implicit in `heal_memory` |
| `generalizes_to` | the heal-generalization work landed in #140 |
| `derived_from` | provenance for synthesized artifacts (Phase C) |
| `depends_on` | tool → tool composition (Phase C) |
| `contradicts` | falls out of `hallucination.rs` verification |
| `co_occurred` | same-session coactivation, cheap to record |

**B2. Spreading activation retrieval.** Keep the stage-1 gate
(`memory_recall.rs:35` — free skip on a degenerate embedding; it is good and
costs nothing). Replace stage 2's flat ranking with: cosine top-k to seed the
frontier, then **bounded** k-hop spreading activation with per-edge-type weights
and a decay factor, ranked on combined cosine + accumulated activation. Bound the
frontier by visited-node count, not by hop depth alone, or traversal cost becomes
unpredictable on dense regions.

This is what buys multi-hop recall: *"what did we decide about X"* traverses
decision → supersession chain → the fault that caused the decision, instead of
returning whichever single record happens to sit nearest in embedding space.

**B3. The graph is what the loops write into.** Both the heal loop and the
acquisition loop stop appending flat records and start appending *edges*. That is
the whole point — it is what turns remembered items into a structure that
composes.

**Honest cost:** traversal over embeddings gets expensive fast. Bounded frontier
is mandatory, not optional. Benchmark recall latency against the current flat
path before defaulting it on; the repo's convention is env-flag-off until
measured, and this should follow it.

## Phase C — Forge: harvest and synthesize (requires A)

**C1. Harvest with provenance from the first byte.** Query crates.io / docs.rs /
GitHub code search for a capability. Every retrieved fragment carries its source
URL, commit SHA, SPDX license, and a content hash from the moment it enters the
process — recorded via the existing `provenance.rs` (SHA-256 + optional HMAC).
Provenance is not added at the end; it is the record's identity.

**C2. A license gate, and it is hard-blocking.** An SPDX allowlist checked before
any fragment is admitted. Axiom is Apache-2.0. Pulling GPL/AGPL source into a
synthesized artifact that then ships through the fleet's signed-export path is a
licensing incident that is trivially cheap to prevent now and expensive to unwind
later. Refuse copyleft, log the refusal, keep going.

**C3. Verification-first synthesis.** A synthesized tool is not admitted because
it compiles. It is admitted because it passes a generated property/differential
test in the jail, and it enters service carrying a **low `BetaBelief`**
(`belief.rs:39`) that is promoted only by observed successes in real use. This is
the honest version of "it works": `BetaBelief` exists precisely because a scalar
confidence cannot tell "succeeded 1/1" from "succeeded 50/50," and a
freshly-synthesized tool is the canonical 1/1 case.

**C4. Reuse the peer-patch invariant verbatim.** Harvested code is a peer patch
from a stranger. Never applied on trust; written, verified locally, rolled back
byte-for-byte on failure. `patch_memory.rs` already implements this control flow
— Forge should call into that shape rather than inventing a parallel one.

**C5. Tools land in the graph, not a directory.** A synthesized tool is a node
with `derived_from` edges to its harvested fragments and `depends_on` edges to
the tools it composes. That is what makes the second synthesis cheaper than the
first, and it is the payoff for doing Phase B first.

## Phase D — Resource arbiter (requires A1's cgroup work)

**D1. Promote `recommend()` from boot-time to per-request.** Keep the pure
function exactly as it is — it is well-designed and well-tested. Wrap it in a
`ResourceArbiter` holding a `HardwareProfile` refreshed on a cadence, plus
semaphore-based admission control. Route each request to a `(device, batch size,
thread count)` decision based on its measured cost class. The co-tenancy guard
generalizes cleanly from "is training resident?" to "what else is in flight right
now?"

**D2. Hardware-native primitives that are actually reachable.** cgroups v2 for
enforcement (shared with A1), `io_uring` or `mmap` + `MADV_SEQUENTIAL` for corpus
streaming, NUMA-aware core pinning for TTT threads, huge pages for the fast-weight
matrices. These are **perf polish, not new capability** — worth doing, worth
measuring, not worth overclaiming. Report them as latency numbers or not at all.

## Phase E — Research (explicitly unbudgeted, no promises)

**E1. Graph-conditioned TTT.** Condition or initialize `W_tilde` on the graph
neighborhood of the active task, so fast weights start warm in the right region
instead of cold every session. This is the deepest idea in this document and the
least certain; it needs a real experiment, not a plan entry.

**E2. Graph-region-keyed weight fragments.** The fleet already gossips
HMAC-authenticated DWE weight fragments (`dwe.rs`). Keying fragments by graph
region would let nodes share adaptation for *a topic* rather than *a session*.
Downstream of both B and E1.

---

## Recommended order, and why

```
A (sandbox: isolation + deps + verify)  ──┬──> C (forge)  ──> E2
                                          │
B (graph memory)  ────────────────────────┴──> E1
                                          │
                                          └──> D (arbiter, shares A1's cgroups)
```

**Start B and A1 in parallel.** B is safe, self-contained, and independently
valuable even if Forge is never built. A1 is the security prerequisite and has
zero dependencies of its own.

**Do not start C before A2 lands.** Without vendored offline dependency builds,
harvested code cannot compile, and the synthesis loop will appear to work while
only ever producing dependency-free toy functions.

## A fifth AxiomBench axis: acquisition

`AxiomBench` measures four axes — cognition, trust, fleet, cost — and
`RESULTS.md` carries the headline numbers. This work needs a fifth axis or it
cannot be honestly evaluated:

> **Acquisition.** From a cold start, given a capability description and a
> held-out property test the synthesizer never sees, how often does the loop
> produce an artifact that passes — with the network off at verify time, inside
> the jail, with full provenance and a clean license gate?

Pass rate, wall-clock, and false-green rate. The false-green rate is the one that
matters: a synthesis loop that confidently ships wrong tools is worse than no
synthesis loop, because its output is durable and shared.

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Running harvested internet code on an unconfined executor | **Critical** | Phase A1 gates all of Phase C. Non-negotiable ordering. |
| Synthesized-but-wrong tools poisoning durable memory | **High** | C3 property tests + low initial `BetaBelief`; false-green rate as a tracked bench metric |
| Copyleft contamination of an Apache-2.0 tree | **High** | C2 hard-blocking SPDX allowlist before admission |
| Graph traversal cost regressing recall latency | Medium | Bounded frontier; env-flag-off until measured against the flat path |
| Rebuilding per-token TTT that already exists | Medium | Documented above; `ttt_block.rs` is the implementation, `docs/UPGRADES.md` the validation |
| Arbiter and sandbox growing separate resource models | Medium | One shared `ResourceArbiter` over one cgroup hierarchy |
| `chimera` / `harvest` name collisions | Low | Use `forge`; check `src/bin/` before naming |

## Proposed module map

| New module | Responsibility |
|---|---|
| `sandbox_jail.rs` | Isolation primitives; consumed by `sandbox.rs` and `poly_jit.rs` |
| `graph_memory.rs` | Typed edge log, adjacency index, spreading activation |
| `forge.rs` | Harvest → license gate → synthesize → verify → admit |
| `arbiter.rs` | Per-request device/thread/memory arbitration over cgroups |

Reused unchanged: `provenance.rs`, `belief.rs`, `patch_memory.rs`,
`heal_memory.rs`, `memory_store.rs`, `hardware.rs::recommend`, `ttt_block.rs`.

## Resolved questions

Recorded 2026-07-26. These are recommendations with stated reasoning, not
measurements. Q2 is binding on Tranche 1; the rest bind Phase C and should be
re-examined when it is briefed.

### Q1. Forge's output format → **WASM (`wasm32-wasip1`) by default, subprocess as escape hatch**

The decisive argument is not composition, it is that **it collapses most of
Phase A1**. A WASM runtime *is* the isolation boundary: capability-based WASI is
deny-by-default for filesystem and network, so the common case needs no
namespaces, no seccomp policy, and no cgroup delegation.

That matters disproportionately here because the cgroups/seccomp/Landlock design
is **Linux-only**, while `.clinerules` records the primary dev machine as Windows
and CI runs `ubuntu-22.04`. A Linux-only jail means the sandbox behaves
differently on the machine it is developed on than on the machine it is tested
on — a bad property for a security boundary. WASM is uniform across both.

Cost, stated honestly: syscall reach is limited, so a tool needing real process
spawning or native libraries cannot be a WASM module. Those keep the subprocess
path and get the heavyweight OS jail — which is the right allocation of effort,
since they are the minority and the genuinely dangerous case.

Open sub-question: this implies a `wasmtime` dependency, which is a large addition
to a crate that currently has none of that weight. Evaluate binary-size and
build-time impact before committing.

### Q2. Graph storage → **stay append-only JSONL** *(binding on Tranche 1)*

Keeps the existing durability and auditability model, requires no new dependency,
and needs no migration. Task B5 measures traversal cost; revisit an embedded
store (redb/sled) only if B5 shows a real p95 regression. Do not pre-optimize
this.

### Q3. Property-test authorship → **local generator first; frontier model opt-in**

Beyond the local-first preference, there is a correctness argument that is
easy to miss: **if the same frontier model writes both the tool and its test, the
failures are correlated.** A misunderstanding of the spec shows up identically in
both, and the test passes vacuously while certifying nothing. That is precisely
the false-green case the acquisition benchmark is meant to catch, and it would be
built into the loop by construction.

So: type-directed local generation (proptest-style, derived from the tool's
signature) as the default, with a frontier-authored path behind an explicit flag
for cases local generation cannot reach. Where the frontier path is used, prefer
a *different* model than the one that synthesized the tool.

### Q4. Fleet sharing → **gossip the recipe, never the artifact**

Do not ship executable artifacts across the fleet. Ship the **provenance record
plus the property test**, and let each peer re-synthesize and re-verify locally.

This is a strictly stronger form of the invariant `patch_memory.rs:9` already
enforces. A patch is re-verified before it is applied; a *tool* has a wider blast
radius than a patch to a known file, so it should be re-*derived*, not merely
re-checked. Sharing the recipe rather than the dish means a compromised peer can
at worst waste a neighbor's CPU, never hand it a binary it did not build.

Weight fragments (`dwe.rs`) are data and can keep their existing gossip path
unchanged. This restriction is about executable artifacts only.

## Known repo inconsistency — license declaration

Surfaced during the Tranche 1 audit; **needs an owner decision, not an agent fix.**

The root `LICENSE` file is **Apache-2.0**. Every packaging declaration says
**MIT**:

| Declaration | Value |
|---|---|
| `LICENSE` (root) | Apache-2.0 |
| `axiom_engine_rs/Cargo.toml` | `license = "MIT"` |
| `axiom_mesh_rs/Cargo.toml` (workspace, inherited by all three mesh crates) | `license = "MIT"` |
| `pyproject.toml` | `{ text = "MIT" }` + MIT OSI classifier |

Git history does not disambiguate intent — both files trace to the same commit
(`63e2aac`), so the tree appears to have been squashed or re-initialized.

**Recommendation: reconcile toward MIT** (replace the `LICENSE` file). The crate
publishes to crates.io and the wheel to PyPI, both declaring MIT — so MIT is what
has already been asserted publicly to every downstream consumer at every
distribution channel. Correcting the outlier file matches what shipped.
Reconciling the other direction would mean every published artifact to date was
mislabeled, which is a materially worse position.

This does not block Phase B. It **does** block Phase C's license gate (C2), which
cannot decide what to admit until the project's own license is unambiguous.
