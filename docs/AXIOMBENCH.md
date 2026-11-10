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

**FIXED (2026-10-29).** The generator now rewrites only the block between
`<!-- axiombench:table:begin -->` / `<!-- axiombench:table:end -->` in
`RESULTS.md`, emits the `n` / "read as" columns itself (every pillar carries
them), carries over rows for pillars that did not run (the `cost` row
survives a deterministic-only run), and leaves everything outside the markers
untouched. Covered by unit tests in `axiom_engine_rs/src/bin/axiombench/main.rs`.

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
  today (the closest is `axiom eval-agentic`'s 32 fixtures, which test
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
   32 fixtures are the closest existing artifact but test AXIOM's repair
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
| Repair success / failure / regression rate | Partial — `SolveReport`/`CompileRunReport` carry pass/fail per run; no aggregate rate is computed or reported anywhere |
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

## 4. Phase 4 item 1 — paired runner complete (2026-11-11)

The prep scaffolding is now the completed Phase 4 item 1 runner
(PR phase4/paired-runner). The full paired comparison HAS run: all
five arms x all five families, one repeat, seed 42, deterministic (no
LLM), 40 cells, 0 pending. Results in
`bench/results/PAIRED-FULL-2026-11-11.md` (the per-run JSON is
git-ignored, like the pillars' JSON). The harness runs, the suites
exist, the guardrail check is implemented and demonstrated, and the
full-run verdicts are recorded, not guessed.

### 4.1 Paired runner

`axiombench paired` (`axiom_engine_rs/src/bin/axiombench/paired.rs`)
implements the audit section 5 matrix: arms A baseline, B retrieval,
C CVM, D repair, E full stack; families single-file, multi-file,
long-session, tool-heavy, failure-recovery; fixed task set, model,
budget, environment, and oracle recorded in the report JSON
(`bench/results/paired-<unix>.json`) with a stable fingerprint;
run order randomized by seed (SplitMix64 + Fisher-Yates, a fixed seed
reproduces identical order); repeats to estimate variance. Primary
measure is verified task-success rate; secondaries are end-to-end
dollar cost, cache-adjusted billed tokens, agent turns, tool calls,
wall-clock time, first-token latency, unsafe patch rate, and successful
rollback rate. Every secondary carries its provenance next to the
value: deterministic arms have no tokens, turns, or first-token
latency to measure, so those are recorded as not-applicable, never as
a fabricated zero; dollar cost is 0.0 measured (no API spend).

What executes today: all five arms on all five families
(`--smoke` still runs only arms A/D on the single-file family, for
CI), via the real `axiom solve` loop in a scratch copy of each
fixture, following the four steps of
`scripts/bench_repair.sh` (no-op baseline, solve with no file named
and isolated heal-memory state, re-verify, pinned-file and rollback
checks). Arm A runs no solver, so the failing verify stays failing.
Arms B, C, and E run the same loop under their own env overlay (arm
B: compression off; arm C: compression on; arm E: compression plus
vibe on). The overlays are recorded labels, not ablations: the three
overlay variables have no read site in the `axiom solve` path
(`AXIOM_TTT_COMPRESS` is read only by the server startup path,
`AXIOM_RESPONSES_COMPRESS` only by the Responses route,
`AXIOM_VIBE` only by the server startup path, and `SolveOptions`
carries no vibe path), and arm B's overlay is byte-identical to arm
D's. B/C/D/E therefore execute the same deterministic loop, and
their identical scores are replication, not a measured null effect
of the toggles. Cross-arm mechanism comparisons stay unevaluable
until the toggles are wired into solve; the paired run's valid
comparisons are arm A vs the executing arms (the guardrails) and
replication across arms. Every family has an on-disk fixture.
Six live in `bench/repair/`, the corpus `bench_repair.sh --check`
scores (every fixture there must be repairable, so the two tasks
the deterministic solver cannot repair do not live there). Those
two, `multi-both-broken` (port of the
`agentic-multi-file-both-broken` builtin: both scripts broken, only
a coordinated two-file fix passes) and `longsession-overflow`
(off-by-one buried in a 257-line module under a 240-entry lookup
table), live in `bench/paired/`, a paired-only diagnostic corpus
the repair gate never sees. The other fixtures:
`multi-interface-and-callsites` (shared-helper contract broken,
two callsites),
`toolheavy-many-calls` (eight modules, one buggy),
`recovery-broken-env` (pure environment fault: `run.sh` lacks the
exec bit; the code is correct, the environment heal fixes it).

Per-stage latency and memory (Phase 3 items 1 and 5, wired at the
harness level): every executed cell records `stage_ms`
(baseline-verify / solve / re-verify / rollback wall time, the
runner's own pipeline stages) and `peak_rss_kb` (the solver
subprocess peak RSS, VmHWM sampled from Linux `/proc/<pid>/status`
every 5 ms; a sampled lower bound, never an exact peak, and `null`
when a sub-10 ms child exits before the first poll). The Phase 3
signals themselves live in the engine's `metrics.rs`
(`axiom_first_token_latency_seconds`,
`axiom_stage_latency_seconds{stage=compress|route|forward|fingerprint}`,
`axiom_process_rss_bytes`, rendered on the server's `/metrics`);
the runner cannot read a dead subprocess's in-process histograms, so
it measures its own stages and the solver's RSS from outside instead.
First-token latency stays not-applicable in the deterministic harness:
the solver emits no streamed tokens, so `first_token_ms` is `null`
with the reason in `notes`, and is measured only by a live-agent run
against the streamed server path (open item, below), never
fabricated from time-to-first-stdout-byte.

```bash
# Smoke run: arms A/D on the single-file family, one repeat, seed 42:
cargo run --release --features tools --bin axiombench -- paired --smoke
# Full matrix (default repeats=3; the recorded run used --repeats 1):
cargo run --release --features tools --bin axiombench -- paired
# Guardrail demonstration (arm D vs a deliberately regressed arm):
cargo run --release --features tools --bin axiombench -- paired --demo-regression
```

### 4.2 Guardrail qualification

`axiom_engine_rs/src/bin/axiombench/guardrail.rs` implements the
audit section 5 promotion rule as part of the bench runner output
(pass/fail per arm vs baseline, printed and written into the paired
report JSON under `guardrails`):

- **Task-success non-inferiority.** The arm's verified task-success
  rate must stay within 5 percentage points of the baseline arm's
  rate. A cheaper arm that solves fewer tasks does not promote.
- **Reliability guardrail.** Unsafe patch rate must be 0 (no edit to
  the oracle, ever) and successful rollback rate over failed attempts
  must be 1.0 (every failed attempt leaves the tree byte-identical).
- **Cost guardrail.** When the baseline has a measured cost per
  success, the arm's cost per success must not exceed it by more than
  2x. When either side has no measured cost (the deterministic harness
  spends $0), the rule is recorded as not-applicable, never as a pass
  on invented numbers.
- An arm with no executed cells cannot be evaluated: the verdict is a
  fail with reason "no executed tasks", not a silent pass.

Demonstration (local run, `--demo-regression`): the baseline arm
passes against itself, arm D passes on the smoke family, and the
deliberately regressed arm (solver sabotaged to edit a pinned file
and skip rollback) fails on the reliability guardrail, whether its
verifier goes green (the JavaScript fixture, where the sabotage is a
valid comment and the cell scores OVERFIT) or not (the Python and
shell fixtures, where it is a syntax error and the cell scores a
rollback violation). See the run output recorded in
`bench/results/PAIRED-SMOKE-2026-11-10.md` for the run that
demonstrates it.

### 4.3 Genuinely unseen bug suites

Two suites under `bench/corpus/`, both excluded from all tuning until
the qualification run:

- **Seeded, seed 424242** (`bench/corpus/unseen-seed424242/`): 24 fresh
  seeded faults in `axiom_mesh_core` from a different seed and a
  revised generator (new `strlit` class) than seeds 7/2026. See its
  README for the unseen argument and the honest caveat that `strlit`
  contributed 0 faults in this crate. Baseline recorded in
  `bench/results/UNSEEN-SEED424242-2026-11-10.md`.
- **Human-written, Codeflaws** (`bench/corpus/unseen-codeflaws/`): 12
  human-written C defects crawled from Codeforces (Tan et al.),
  imported by `scripts/import_codeflaws.py` under a deterministic
  selection rule (first WRONG_ANSWER defects in detail-file order with
  a ≤150-line buggy submission, ≤15-line fix diff, ≥2 repair-test
  inputs; accepted program must pass and buggy program must fail the
  fixture oracle, or the defect is skipped). Unseen because no repair
  operator or ordering rule was ever tuned on Codeflaws, and because
  its programs are C single-file submissions, a different language and
  shape from QuixBugs' Python algorithms. Baseline recorded in
  `bench/results/UNSEEN-CODEFLAWS-2026-11-10.md`.

### 4.4 Open items (after item 1)

- Wire the arm toggles into the `axiom solve` path so the paired
  arms become real ablations. Today the three overlay variables
  have no read site in solve (see section 4): B's overlay is
  byte-identical to D's, and C/E differ only in labels. Until that
  wiring lands, B/C/D/E execute the same deterministic loop and
  cross-arm mechanism comparisons are unevaluable; the recorded
  6/8-each scores are replication, not a toggle effect.
- The live-LLM agent executor for arms B, C, and E (real coding
  agent with and without AXIOM in the loop, with token/cost/turn
  accounting and first-token latency measured on the streamed server
  path). The deterministic harness runs the real solver under each
  arm's recorded overlay; it does not pretend to be that agent.
- Repeats 2-3 of the full matrix for variance: the recorded run is
  repeats=1 (40 cells); the scaffold default is repeats=3.
- A full-matrix run against the release solver binary: the recorded
  run used the debug binary (`target/debug/axiom`); CI runs the
  release binary for the smoke family only.
- Per-stage latency / first-token latency wiring from Phase 3 item 5
  into the paired secondaries is done at the harness level (stage
  timings and solver RSS, above); first-token latency still waits on
  the live-agent run.

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

Note: `axiombench` only rewrites the marked table block in `RESULTS.md` (§2);
the hand-maintained caveats below it are preserved, and the `cost` row is
carried over when the pillar is not run.

## 4. Guardrail qualification verdicts (Phase 4 item 3)

The per-arm promotion verdicts on the Phase 4 paired run (PR #293)
and unseen baselines (PR #294) are recorded in
`bench/results/PAIRED-VERDICTS-2026-11-11.md`. Summary, under the
audit section 5 promotion rule (measured task-success non-inferiority
plus an explicit reliability guardrail):

- D repair: PROMOTE (narrow). Measured 6/8 vs baseline 0/8, unsafe
  0.00, rollback 1.00, complete coverage, rerun reproduced. Scope is
  the deterministic repair loop on the 8-fixture paired set only. No
  cost claim attaches: cost was not measured.
- B retrieval, C CVM, E full stack: HOLD. Each guardrail PASSES, but
  the identical 6/8 is replication of the same deterministic loop, not
  a measured toggle effect: the overlays have no read site in the
  solve path. Each promotes only after a real toggle ablation shows
  it differing from D.
- D-regressed: FAIL by design (negative control, unsafe 1.00,
  rollback 0.00).
- Overlap disclosure: the seed-424242 "genuinely unseen" headline is
  overstated. 18 of 36 fixtures share a (file, line, class) triple
  with the prior tuning tables; the clean held-out rate is 7/18 on
  fresh sites against 12/18 on previously tuned sites, not the
  headline 19/36. Quote 7/18 alongside 19/36, never 19/36 alone,
  until the suite is regenerated at new sites and re-baselined.
- Cost guardrail: not-applicable on every arm. The deterministic
  harness spends $0. A real cost verdict needs a live-agent run with
  billed-token accounting; wall-clock time is not a substitute.

Open item: the demo-regression FAIL is not wired into CI yet. PR
#293's CI runs only `paired --smoke`, without `--fail-on-guardrail`
and without a `--demo-regression` step. That wiring is blocked on
#293 merging (the `paired` subcommand does not exist on main) and is
specified in the verdicts file for the rebase onto post-#293 main.
The live gate today is the unit test
`regressed_arm_fails_the_guardrail_it_is_built_to_trip`, which runs
under `cargo test` on #293's branch.
