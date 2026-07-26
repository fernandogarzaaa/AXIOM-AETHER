# Implementation prompt — Capability Acquisition Loop, Tranche 1

> Hand this document to the implementing agent verbatim. It is the executable
> counterpart to `2026-07-26-capability-acquisition-loop.md` (the design
> rationale). Read that document first; it explains *why* each constraint below
> exists. This document says *what to build and in what order*.

---

## Your assignment

You are implementing **Tranche 1** of the Capability Acquisition Loop in
`fernandogarzaaa/AXIOM-AETHER`: **Phase B, graph working memory.**

Phase B was chosen to go first because it is the only phase with no
prerequisites, it touches nothing security-sensitive, and it is independently
valuable even if the later phases are never built. Phases A2, C, D, and E are
**explicitly out of scope** — see "Non-goals" below. Do not start them. Do not
scaffold them.

Work task-by-task. Each task is a branch, a PR, and a green CI run before the
next one starts. Do not batch tasks into one PR.

---

## Part 1 — Repository constraints (read before writing any code)

These are not stylistic preferences. Each one is load-bearing and several encode
a bug that has already been paid for once.

### Build and test gates

Run these locally and get them green **before every push**:

```bash
cd axiom_engine_rs
cargo clippy --lib --locked -- -D warnings
cargo test --release --locked
```

CI (`.github/workflows/ci.yml`) additionally runs the mesh workspace tests,
`axiombench`, `./scripts/demo_end_to_end.sh`, a Docker image build, and the
Python reference suite. Your changes should not touch those surfaces; if one of
them breaks, that is a signal you have reached further than intended.

`--locked` is mandatory. It means **`Cargo.lock` must be committed** if
dependencies change — and see the dependency rule below before they do.

### Hard invariants

1. **No UTF-8 BOM in any `src/**/*.rs` file.** There is a regression test that
   scans for this at `axiom_engine_rs/src/mcp_stdio.rs:2050`
   (`no_source_file_has_utf8_bom`). Note that `Cargo.toml` itself *does* carry a
   BOM — do not "fix" that; it is out of scope and unrelated.
2. **ASCII punctuation in source.** No smart quotes, no en-dashes in code or
   code comments.
3. **All diagnostics go to `stderr` via `eprintln!`.** `stdout` is reserved for
   JSON-RPC framing in MCP mode. A stray `println!` anywhere reachable from
   `mcp_stdio.rs` corrupts the protocol stream. This is the single easiest way
   to break this repo.
4. **Every new behavior ships behind an env flag, defaulting OFF, read live
   inside the code path** — not cached in a `lazy_static` or at boot. Existing
   examples: `AXIOM_RESPONSES_COMPRESS`, `AXIOM_FORGET_GATE`, `AXIOM_CVM_DIGEST`.
5. **Tests that mutate process-global env vars must serialize.** Use a
   `tokio::sync::Mutex` env-lock plus an RAII guard. The established pattern is
   at `axiom_engine_rs/tests/rebase_proxy.rs:27` (`struct EnvVarGuard`). Copy it;
   do not invent a new one. This exists because of a real flaky-test incident.
6. **Never use `reqwest::blocking` inside an async context** — wrap in
   `spawn_blocking`. (Not expected to come up in Phase B, but it is a standing
   rule.)
7. **Memory recall returns real stored bodies, never neural reconstructions.**
   `memory_store.rs:1` and `memory_recall.rs:4` both state this. Your graph work
   must preserve it absolutely: traversal changes *which* records are returned
   and in what order, never *what they contain*.

### Dependency rule

The crate has no graph library and CI builds `--locked`.

**Do not add any dependency for Phase B.** Hand-roll the adjacency index with
`std::collections::HashMap` / `BTreeMap`. `petgraph` is not justified for a
bounded-frontier BFS over a few thousand nodes, and adding it expands the audit
surface of a repo that ships signed artifacts to a fleet.

Already available if you need them: `serde`, `serde_json`, `sha2`, `uuid`,
`walkdir`, `dashmap`, `tokio`, and `tempfile` (dev-dependencies only).

### Git workflow

- One task, one branch, one PR. Branch names must start with `claude/` so CI
  triggers on push (`.github/workflows/ci.yml` fires on `claude/**`).
  Suggested: `claude/graph-b1-edge-store`, `claude/graph-b2-spread`, etc.
- Open PRs as **drafts**. CodeRabbit skips draft review by default; mark ready
  for review when you want its pass.
- Commit messages: evidence-backed, label experimental work as such, include
  reproduction commands for any metric you cite. Never cite a number you did not
  personally measure.

---

## Part 2 — What already exists (do not rebuild any of this)

Read these files before you start. The design depends on reusing them.

| File | What it gives you |
|---|---|
| `src/memory_store.rs` | `MemoryRecord` (id, scope, ts, kind, body, embedding, `supersedes`, tombstone), append-only JSONL `MemoryStore` with `open`/`append`/`load_scope`/`get`/`tombstone`/`scopes`, plus `cosine` and `top_k` helpers |
| `src/memory_recall.rs` | Two-stage recall: `should_recall` gate (free skip on a degenerate embedding), then flat cosine `top_k` with a supersession filter and recency tie-break |
| `src/mcp_stdio.rs:1704`, `:1812` | The two `recall(...)` call sites behind the `axiom_recall` and `search` MCP tools |
| `src/self_heal.rs:949` | Already opens a `MemoryStore` — the natural place `caused_by` edges get written later |
| `src/bin/eval_recall.rs` | An existing recall benchmark binary (gated behind the `tools` feature) — extend this rather than writing a new harness |

**The key observation the whole phase rests on:** `MemoryRecord.supersedes:
Option<String>` (`memory_store.rs:31`) is already a stored, directed edge. It is
consumed today only as a tombstone filter (`memory_recall.rs:60`). You are
generalizing an existing field, not replacing the store. **There must be no data
migration.** Existing `.jsonl` memory files must keep working untouched.

---

## Part 3 — The tasks

Work in order. TDD throughout: **write the failing test first, run it, confirm it
fails for the reason you expect, then implement.** A test that passes the moment
you write it is a test that is not testing anything.

---

### Task B1 — Edge store

**Branch:** `claude/graph-b1-edge-store`
**Files:** create `axiom_engine_rs/src/graph_memory.rs`; register `pub mod
graph_memory;` in `src/lib.rs` (insert after `pub mod fault_locate;`, line 30).

Define:

```rust
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Supersedes,    // already implicit in MemoryRecord.supersedes
    CausedBy,      // fault trace -> heal
    GeneralizesTo, // specific heal -> heal class (cf. PR #140)
    DerivedFrom,   // synthesized artifact -> source fragment (Phase C)
    DependsOn,     // tool -> tool composition (Phase C)
    Contradicts,   // falls out of hallucination.rs verification
    CoOccurred,    // same-session coactivation
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemoryEdge {
    pub from: String,   // MemoryRecord.id
    pub to: String,     // MemoryRecord.id
    pub kind: EdgeKind,
    pub weight: f32,    // evidence strength, clamped to 0.0..=1.0 on append
    pub ts: u64,        // memory_store::now_secs()
    #[serde(default)]
    pub tombstone: bool,
}
```

Implement `EdgeStore`, mirroring `MemoryStore`'s shape and durability model:

- `open(root: impl AsRef<Path>) -> std::io::Result<Self>`
- `append(&self, edge: &MemoryEdge) -> std::io::Result<()>` — clamps `weight`
- `load_all(&self) -> Vec<MemoryEdge>` — skips malformed lines rather than
  failing the whole load, matching `MemoryStore::load_scope`'s tolerance
- `tombstone(&self, from: &str, to: &str, kind: EdgeKind) -> std::io::Result<()>`

Storage is a **single** append-only `edges.jsonl` at the memory root, not one
file per scope — edges legitimately cross scopes and a per-scope split would make
traversal require opening every file.

Also implement the zero-migration bridge:

```rust
/// Derive `Supersedes` edges from the `supersedes` field already present on
/// existing records, so the graph is populated on day one with no migration.
pub fn edges_from_records(records: &[MemoryRecord]) -> Vec<MemoryEdge>;
```

**Tests (write first):**
1. `edge_roundtrip_jsonl` — append two edges, reload, both present and equal.
2. `append_clamps_weight` — a weight of `2.5` reloads as `1.0`; `-1.0` as `0.0`.
3. `malformed_line_is_skipped_not_fatal` — write a garbage line into
   `edges.jsonl` by hand, confirm `load_all` still returns the valid edges.
4. `supersedes_backfill_produces_edges` — records where B supersedes A yield
   exactly one `Supersedes` edge B→A.
5. `tombstoned_edge_is_excluded` — tombstone then reload.

Use `std::env::temp_dir()` with a unique suffix for test roots and clean up, as
`memory_recall.rs:110` does.

---

### Task B2 — Bounded spreading activation

**Branch:** `claude/graph-b2-spread`
**Files:** `axiom_engine_rs/src/graph_memory.rs`

Build a bidirectional adjacency index from `Vec<MemoryEdge>`. Traversal must
follow edges in **both** directions — a `CausedBy` edge is informative read from
either end — but out-edges and in-edges may carry different weights.

```rust
pub struct Adjacency { /* HashMap<String, Vec<(EdgeKind, String, f32)>> x2 */ }
impl Adjacency {
    pub fn build(edges: &[MemoryEdge]) -> Self;  // skips tombstoned
}

pub struct SpreadParams {
    pub max_hops: usize,     // default 2
    pub max_visited: usize,  // default 256 -- HARD frontier bound
    pub decay: f32,          // default 0.5
    pub min_activation: f32, // default 1e-3
    pub kind_weights: BTreeMap<EdgeKind, f32>, // default 1.0 each
}

/// Seeds are (node_id, initial_activation) from the cosine stage.
/// Returns every reached node with accumulated activation, seeds included.
pub fn spread(
    adj: &Adjacency,
    seeds: &[(String, f32)],
    params: &SpreadParams,
) -> Vec<(String, f32)>;
```

Semantics:

- `activation(child) += activation(parent) * decay * edge_weight *
  kind_weight(kind)`
- Stop expanding when: `visited >= max_visited`, **or** hop depth exceeds
  `max_hops`, **or** the propagated activation falls below `min_activation`.
- `max_visited` is a **hard bound on nodes visited**, not on hop depth. Depth
  alone does not bound cost — one hop into a dense region can visit thousands of
  nodes. This is the single most important property in the task.
- **Determinism is required.** Order the frontier by `(activation descending, id
  ascending)`. Two runs over identical input must produce byte-identical output,
  or the tests will flake and the benchmark will be meaningless.
- **Cycles must terminate.** `A -> B -> A` and self-loops are legal inputs.

**Tests (write first):**
1. `spread_reaches_two_hops` — A→B→C with A seeded returns all three.
2. `spread_decays_with_distance` — activation(C) < activation(B) < activation(A).
3. `spread_respects_max_hops` — `max_hops: 1` excludes C.
4. `spread_respects_max_visited` — build a star with 1000 leaves,
   `max_visited: 10`, assert `result.len() <= 10`.
5. `spread_is_deterministic` — run twice, assert exact equality including order.
6. `cycle_does_not_hang` — A↔B plus a self-loop; wrap in a test that would fail
   rather than spin (bounded iteration count assertion is fine).
7. `tombstoned_edge_is_not_traversed`.
8. `kind_weights_change_ranking` — zero-weighting one `EdgeKind` drops the nodes
   only reachable through it.
9. `empty_seeds_returns_empty`.

---

### Task B3 — Graph-aware recall

**Branch:** `claude/graph-b3-recall`
**Files:** `axiom_engine_rs/src/memory_recall.rs` (+ `graph_memory.rs` as needed)

Extend `RecallParams` with graph fields, all defaulting to the current behavior:

```rust
pub struct RecallParams {
    pub min_score: f32,   // existing, 0.2
    pub k: usize,         // existing, 5
    pub graph: bool,      // NEW, default false
    pub spread: SpreadParams, // NEW, defaults as B2
    pub graph_weight: f32,    // NEW, default 0.3
}
```

Add:

```rust
pub fn recall_graph(
    store: &MemoryStore,
    edges: &Adjacency,
    scopes: &[String],
    query_embedding: &[f32],
    params: &RecallParams,
) -> Vec<RecallHit>;
```

Pipeline:

1. **Stage-1 gate unchanged.** Keep `should_recall` exactly as-is
   (`memory_recall.rs:35`). It is free and correct.
2. Cosine `top_k` as today to produce **seeds** (seed count may exceed final `k`;
   use `k * 2` as the existing code already does).
3. `spread` from those seeds over the adjacency.
4. Final score: `cosine_score + graph_weight * activation`. A node reached only
   through the graph has `cosine_score = 0.0` and scores
   `graph_weight * activation`.
5. **Apply the supersession filter and the `min_score` floor after scoring**, and
   truncate to `k`. Both existing behaviors must survive — a superseded record
   must not come back through a graph path.
6. Recency tie-break as today.

**Env flag:** `AXIOM_GRAPH_RECALL`, default off, read live inside the call path.
When unset or off, the existing `recall()` must be used and must be **exactly**
as it is today.

**Tests (write first):**
1. `graph_off_is_identical_to_flat_recall` — the regression guard. Same store,
   same query, flag off, assert `recall_graph` output equals `recall` output.
   **This is the most important test in the phase.**
2. `multi_hop_recall_finds_indirect_record` — the capability test. A record with
   a near-zero cosine score, reachable only via two edges from a strong cosine
   match, is returned when the flag is on and absent when it is off.
3. `superseded_record_not_resurrected_via_graph` — the correctness trap. A
   superseded record with a strong graph path must still be dropped.
4. `graph_hit_below_min_score_is_filtered`.
5. `graph_recall_respects_k`.
6. `graph_recall_returns_real_bodies` — assert exact stored text on a
   graph-reached record; the no-reconstruction invariant must hold on the new
   path too.

---

### Task B4 — MCP surface

**Branch:** `claude/graph-b4-mcp`
**Files:** `axiom_engine_rs/src/mcp_stdio.rs`

1. Add an optional `graph: bool` argument to the `axiom_recall` tool that
   overrides the env default per call. Absent ⇒ fall back to `AXIOM_GRAPH_RECALL`.
2. Add a new tool `axiom_link` that writes one edge: `{from, to, kind, weight?}`.
   Without a writer the graph stays empty, so this is what makes the phase
   usable rather than theoretical.
3. Update both `recall(...)` call sites (`mcp_stdio.rs:1704` and `:1812`).
4. Update the tool count in `README.md` (currently documented as 20 tools) and
   add both entries to `docs/CAPABILITIES.md` and the `.clinerules` MCP tool
   list.

**Watch for:** every diagnostic in this file must go to `stderr`. A `println!`
here corrupts JSON-RPC framing. Re-read constraint 3 in Part 1.

**Tests:** argument parsing, `axiom_link` round-trip through the store, and
malformed-argument handling returning a proper JSON-RPC error rather than
panicking.

---

### Task B5 — Measure it

**Branch:** `claude/graph-b5-bench`
**Files:** `axiom_engine_rs/src/bin/eval_recall.rs` (extend; `tools` feature)

Add a graph mode and a **multi-hop fixture** — a synthetic memory set where the
correct answer to a query is two hops from the best cosine match. Report:

- recall@k, flat vs graph
- p50 and p95 recall latency, flat vs graph
- nodes visited per query (mean and max)

Write results to `bench/graph/RESULTS-<date>.md` following the format of the
existing `bench/cvm/RESULTS-*.md` files, and state plainly which numbers are
measured and on what hardware.

**The flag stays OFF at the end of this task regardless of the numbers.**
Flipping a default is a separate, explicit, human decision in this repo — see the
PSS precedent in `2026-07-11-prolonged-session-stack.md`, where defaults flipped
only after a live eval and an explicit user decision. Publish the numbers and
stop.

---

## Part 4 — Non-goals

Do not do any of the following. Each has a specific reason.

| Do not | Why |
|---|---|
| Implement Phase C (Forge / code harvesting / synthesis) | It requires Phase A's sandbox isolation and offline vendored dependency builds. Building it now means running harvested internet code on an unconfined executor (`poly_jit.rs:213`). |
| Touch `ttt_block.rs`, `ttt_mlp.rs`, or any TTT internals | Per-token adaptation already works and is validated in `docs/UPGRADES.md`. It is not part of this phase. |
| Flip any env default to on | Explicit human decision, always, after measurement. |
| Change `poly_jit.rs` or `sandbox.rs` execution semantics | That is Phase A, and it is security-sensitive. |
| Add a graph dependency (`petgraph` et al.) | See the dependency rule in Part 1. |
| Migrate or rewrite existing `.jsonl` memory files | The design is explicitly zero-migration. |
| Rename `chimera.rs` or `src/bin/harvest.rs` | Both names are taken by unrelated shipped features; the synthesis subsystem is named **Forge** precisely to avoid this. |

---

## Part 5 — Stop and ask

Stop and raise the question rather than guessing if:

1. **`cargo test --release --locked` fails on `main` before your change.** Report
   it; do not build on a red base and do not attribute the failure to yourself.
2. **You believe a new dependency is genuinely required.** Make the case; do not
   add it unilaterally.
3. **The `graph_weight` scoring blend produces ranking behavior you think is
   wrong.** The 0.3 default is a starting guess, not a measured value, and it is
   the parameter most likely to need judgment.
4. **A design choice would break the no-reconstruction or supersession
   invariants** even slightly. These are correctness properties, not preferences.
5. **You reach the end of B5 and the numbers are bad** — graph recall slower with
   no recall@k gain. That is a real and publishable result. Report it honestly
   and stop; do not tune the fixture until the numbers look good.

### Known open item you may hit

`LICENSE` is Apache-2.0 but `axiom_engine_rs/Cargo.toml` declares
`license = "MIT"`. These disagree. It does not block Phase B, but **do not
"resolve" it on your own** — it changes what a future license gate must enforce
and is a decision for the repo owner. Flag it if you touch packaging metadata.

---

## Part 6 — Definition of done for Tranche 1

- [ ] B1–B5 each merged via their own green-CI draft PR
- [ ] `AXIOM_GRAPH_RECALL` exists, defaults **off**, and is read live
- [ ] With the flag off, recall behavior is provably unchanged (test B3.1)
- [ ] With the flag on, multi-hop recall demonstrably works (test B3.2)
- [ ] No data migration: pre-existing `.jsonl` memory files load unchanged
- [ ] `bench/graph/RESULTS-<date>.md` exists with measured, hardware-stated
      numbers and an explicit measured-vs-simulated statement
- [ ] `README.md`, `docs/CAPABILITIES.md`, and `.clinerules` reflect the new MCP
      tools and the new flag
- [ ] `2026-07-26-capability-acquisition-loop.md` updated: Phase B marked
      shipped, with a link to the results file

When all boxes are checked, stop and report. **Do not proceed to Phase A or C
without a fresh brief** — Phase A is security-sensitive and needs its own design
review before a line of it is written.
