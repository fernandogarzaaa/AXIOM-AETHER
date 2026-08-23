# AXIOMBench — status, gap analysis, and a spec for what's missing

The mission brief calls building AXIOMBench "the highest-priority research
improvement" and asks for a framework comparing baseline agent / +memory /
+compression / +adaptation / +full AXIOM across coding, multi-file, long-context,
tool-heavy, memory, and failure-recovery tasks, with a hard measured /
simulated / estimated distinction and never presenting simulated numbers as
real. This document has three parts, in order of how much confidence to put
in each: **(1) what already exists and was reverified this pass — measured**,
**(2) a real gap found in the existing tooling — measured**, **(3) a spec for
the comparison the mission brief actually asks for — not built, because
building it credibly requires infrastructure (real agent runs against real
repos, at real cost) this audit pass cannot fabricate**.

Per the mission brief's closing instruction — *"Never manufacture evidence...
if something cannot be verified, explicitly label it unverified"* — part 3 is
a specification and a punch list, not a result. Presenting invented numbers
for a baseline-vs-AXIOM agent comparison would be exactly the dishonesty this
document exists to prevent.

---

## 1. What exists today (reverified this pass — MEASURED)

`axiom_engine_rs/src/bin/axiombench/` is a real, working benchmark binary
(`cargo run --features tools --bin axiombench`) with four pillars:

| Pillar | What it measures | n | Character |
|---|---|---|---|
| cognition | Symbol exact-recovery through the structural compressor | 3 | Smoke check |
| trust | Supported-claim coverage + neural contradiction catch-rate | 240 calibration claims | Calibrated result |
| fleet | Cross-node immunity transfer + signed DWE fragment acceptance | 2 nodes | Pass/fail, not a benchmark |
| cost | Byte reduction over replayed real sessions | 3 replayed records | Indicative only |

Plus the compression benchmark run separately (`axiom bench <path>`),
measured at real scale: **82.4%–86.7% token savings with 100% signature
round-trip fidelity over 120–121 files**, depending on tokenizer
(`RESULTS.md`, `bench/ttt/RESULTS-2026-08-09.md`).

**Re-run this pass** (debug build, `cognition`/`trust`/`fleet` pillars —
`cost` requires either live upstream credentials or the separate
`scripts/run_axiombench_cost_mock.ps1`, not run here):

```
[cognition] 100% symbol exact-recovery (3/3)
[trust] 100% supported coverage, neural contradiction catch-rate 10% -> 100% (threshold 0.750)
[fleet] node B pre-immunized in 0.12 ms; signed DWE applied in 0.02 ms (pass)
```

These match `RESULTS.md`'s existing headline numbers (fleet timings differ
slightly, as expected for a "fast enough, not a benchmark" single-run
timing). **Verdict: the existing benchmark claims in `RESULTS.md` and
`bench/ttt/RESULTS-2026-08-09.md` reproduce.** None of this pass's code
changes (capability gate, RwLock, checksum verification, graph_memory,
dead-code removal) touch the compression, cognition, trust, or fleet
pipelines, so this reverification is a clean confirmation, not a
coincidence.

**What this existing suite already does right, worth preserving as the
standard for anything added to it**: every pillar's headline in `RESULTS.md`
is paired with an `n` and a "read as" column that explicitly says whether a
percentage is a rate or a smoke-check count. That discipline — the exact
thing the mission brief asks for (measured/simulated/estimated, never
inflated) — already exists here and should be the template for any new
pillar, not something this document needs to invent.

## 2. A real gap found in the existing tooling (MEASURED, this pass)

Running `cargo run --features tools --bin axiombench` **without** also
running the `cost` pillar **overwrites `RESULTS.md` and silently drops**:
the `n`/"read as" columns, the entire "How to read these" explanatory
section, the `cost` pillar's row, and the "Reproduce" instructions — the
regenerated file has none of the caveats that make the current `RESULTS.md`
trustworthy. Reproduced directly this pass (`git diff RESULTS.md` after one
`axiombench` run showed the file shrink from 62 to 12 lines and lose the
`cost` row and every caveat paragraph); reverted before commit, so this
pass's repository state is unaffected, but the bug is real and will hit the
next person who runs `axiombench` without the cost pillar and commits the
result unreviewed. **Filed as a P2 fix in [ROADMAP.md](ROADMAP.md)**: the
report writer should treat hand-maintained prose as owned by a human (append
or preserve it, or move it to a separate file the generator doesn't touch)
rather than overwrite the whole file with only the pillars it happened to
run this time.

## 3. What AXIOMBench does not yet measure (gap, not built)

Every existing pillar measures a **mechanism working** (compression ratio,
round-trip fidelity, contradiction catch-rate, immunity transfer, byte
reduction). **None measures whether any of this changes downstream agent
task success.** This is the gap the mission brief's ablation design targets,
and it doesn't exist in this repo in any form — not a stub, not a mock, not
a simulated version. Building it is real infrastructure work:

- Running a real coding agent (with and without AXIOM in the loop) against a
  fixed task suite, multiple times for variance, with real API costs.
- A task suite spanning single-file fixes, multi-file refactors, long
  conversations, tool-heavy workloads, and deliberately injected
  failure/recovery scenarios — none of which exist as fixtures in this repo
  today (the closest is `axiom eval-agentic`'s 9 fixtures, which test
  AXIOM's *own* repair loop in isolation, not an agent's task success with
  vs. without AXIOM as a dependency).
- A cost/latency measurement harness that can attribute tokens/dollars/time
  to each configuration (baseline / +memory / +compression / +adaptation /
  +full) fairly — e.g., accounting for AXIOM's own compute cost, not just
  the upstream model's.

This is genuinely the highest-value thing to build next for AXIOM's
scientific credibility — the mission brief is right that it's the top
research priority — and it is also genuinely not something an audit pass
can produce trustworthy numbers for by Friday afternoon. What follows is the
spec, so the next person who picks this up isn't starting from nothing.

### 3.1 Ablation configuration surface

Every mechanism AXIOM adds already has a real, working env-var off-switch —
confirmed by reading the code, not assumed:

| Mechanism | Off switch | Verified |
|---|---|---|
| Structural/adaptive compression | `AXIOM_TTT_COMPRESS=0` | `context_compressor.rs`/`routes_responses.rs` gate on `state.controls.enabled()` |
| Responses-path compression specifically | `AXIOM_RESPONSES_COMPRESS=0` | `routes_responses.rs::responses_compression_enabled` |
| Vibe (persistent adaptation) memory | `AXIOM_VIBE=0` | `vibe_memory.rs::MasterVibe::from_env` |
| Grounding/epistemic verification | opt-in via `AXIOM_VERIFY_RESPONSES` (default off — "off" is already the baseline) | `routes_verify.rs` |
| Mesh routing | opt-in via `AXIOM_MESH_ROUTING=1` (default off) | `mesh_router.rs` |
| Self-healing | `axiom run`/`axiom solve` are separate commands from the proxy path — "off" is simply not invoking them | `entrypoint.rs` |
| Heal memory specifically | `AXIOM_HEAL_MEMORY=0`/`off` | `heal_memory.rs` env gate (README-documented) |

This means **the AXIOM OFF / compression OFF / TTT OFF / memory OFF /
routing OFF / self-healing OFF ablation matrix the mission brief asks for is
already fully expressible today via environment variables** — the missing
piece is not the toggles, it's the task suite and harness to run against
them. This is worth stating precisely because it reframes the remaining work
correctly: not "add ablation support," but "build the measurement harness
that uses the ablation support that already exists."

### 3.2 Proposed task suite (spec, not fixtures — none of this exists yet)

Grouped by what the mission brief asks for, with an honest note on
difficulty of building each fairly:

1. **Coding / repository tasks** — a fixed set of real GitHub issues with
   known, mergeable fixes (the standard SWE-bench-style approach), run
   through an agent with AXIOM in/out of the loop. Hardest part to get
   right: making sure AXIOM's compression doesn't get credit or blame for
   variance that's actually the agent's own non-determinism — needs enough
   repetitions per cell to say something with a confidence interval, not a
   single run.
2. **Multi-file changes** — tasks requiring edits across 3+ files with a
   dependency between them (e.g., an interface change and its call sites).
   Measures whether compression's structural elision loses information a
   multi-file task actually needs — a real risk this repo's existing
   benchmarks don't probe (they measure round-trip fidelity of what's kept,
   not whether what's dropped mattered).
3. **Long conversations** — session length scaled until context would
   overflow without compression; measures whether AXIOM's compression
   changes answer quality as sessions grow, not just token count. This is
   the single most direct test of the unmeasured claim in
   [COMPETITIVE-ANALYSIS.md](COMPETITIVE-ANALYSIS.md) ("does adaptation
   improve task success").
4. **Tool-heavy workloads** — many tool calls per turn; tests interaction
   with `tool_defer.rs`'s existing tool-deferral mechanism (found during
   this audit's reconnaissance, not previously covered by any benchmark
   here).
5. **Memory tasks** — recall correctness and, specifically, **wrong-memory
   rate** (the mission brief's own metric) — inject known facts across
   sessions, query for them later, and separately measure false-positive
   recalls. This is the one category where `graph_memory.rs`'s new
   `spread()` function (merged this pass, not yet wired into
   `memory_recall.rs`) should be evaluated for whether it improves or hurts
   recall precision once it *is* wired in — a natural first real use of the
   new module.
6. **Failure-recovery tasks** — deliberately broken environments/code,
   measuring repair success/failure/regression rate. `axiom eval-agentic`'s
   9 fixtures are the closest existing artifact but test AXIOM's repair
   loop directly, not an agent using AXIOM as infrastructure — worth
   reusing as seed fixtures, not as the whole suite.

### 3.3 Required metrics, and which are already instrumented

| Metric | Already measured somewhere in this repo? |
|---|---|
| Input/output/total tokens | Yes — `/metrics` Prometheus counters (`axiom_savings_*`), `cost_ledger.rs` |
| Cost | Yes — `cvm_store.rs`/`cost_ledger.rs`, but only "indicative" per the cost pillar's own n=3 label |
| Latency | Yes — `metrics::observe_prefill_latency` and friends | 
| Cache utilization | Partial — `axiom_savings_ratio` exists; no cache-hit-rate metric specifically for the response cache (Python reference implementation has `response_cache.py`/`test_response_cache.py`; the Rust engine's caching is less directly instrumented) |
| Tool calls | Not currently counted as a distinct metric |
| Memory recall / wrong-memory rate | Not measured anywhere — new instrumentation needed |
| Repair success / failure / regression rate | Partial — `SolveReport`/`SandboxRunReport` carry pass/fail per run; no aggregate rate is computed or reported anywhere |
| Task success (the actual outcome metric) | **Does not exist anywhere in this repo.** This is the central gap |
| Context size | Yes — token counts are already tracked throughout |

### 3.4 What "measured / simulated / estimated" must mean once this is built

Stated explicitly so whoever builds this doesn't have to re-derive the
mission brief's rule: a number is **measured** only if it comes from an
actual executed run recorded in this repo's benchmark output (a JSON/log
artifact, reproducible by re-running the documented command — exactly the
existing pillars' `bench/results/*.json` pattern). A number is **simulated**
if it comes from a mock/replay harness standing in for a real dependency
(the existing `cost` pillar's "3 replayed records" is closer to this than to
fully measured — it replays real sessions but isn't a live run). A number is
**estimated** if it's a projection/extrapolation not backed by either (e.g.,
"assuming this scales linearly to 1000 files"). **Every number in any future
AXIOMBench report must carry one of these three labels next to it, not just
in a preamble** — the existing `RESULTS.md`'s per-row `n` column is the
right pattern to extend, not replace.

---

## Reproduce what exists today

```bash
# Deterministic pillars (cognition, trust, fleet):
cargo run --release --features tools --bin axiombench

# Compression at scale:
cargo run --release --bin axiom -- bench axiom_engine_rs/src

# Cost pillar without live upstream credentials:
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_axiombench_cost_mock.ps1
```

Note: running `axiombench` without also running the cost pillar will
overwrite `RESULTS.md`'s hand-maintained caveats (§2) — review the diff
before committing, or restore from git if it drops content you need.
