# Paired AXIOMBench — full deterministic comparison — 2026-11-11

Phase 4 item 1 (`phase4/paired-runner`). The per-run JSON records
(`bench/results/paired-full-2026-11-11.json`, plus the rerun
`paired-full-2026-11-11-rerun.json` and the smoke
`paired-smoke-2026-11-11.json`) are git-ignored by the repo's
`bench/results/*.json` rule, like the existing pillars' JSON; this
file is the committed record of what ran and what it showed.

## Run

`axiombench paired --repeats 1 --solver-bin <debug axiom binary>`,
seed 42, deterministic (no LLM), oracle
`bench-repair-v1 (fixture verify command + pinned files)`,
fingerprint `906b424646024f854798c8f47cf4ad0a4fae4d1a1c310fa3db4bccbe5e4ca3a2`.
All five arms (A baseline, B retrieval, C CVM, D repair, E full
stack) x all eight tasks (five families) x 1 repeat: 40 cells, 0
pending. Wall time 2m33s for the run; total cell wall time 143,637 ms.

## Results

| arm | executed | successes | success rate | unsafe patch rate | rollback rate |
|---|---|---|---|---|---|
| A baseline | 8 | 0 | 0.0% | 0.00 | 1.00 |
| B retrieval | 8 | 6 | 75.0% | 0.00 | 1.00 |
| C CVM | 8 | 6 | 75.0% | 0.00 | 1.00 |
| D repair | 8 | 6 | 75.0% | 0.00 | 1.00 |
| E full stack | 8 | 6 | 75.0% | 0.00 | 1.00 |

Guardrails (each arm vs baseline A): all four PASS. The cost
guardrail is not-applicable on all four: the deterministic harness
spends $0, so there is no measured cost per success on either side.

Per-task outcome (identical in the rerun, cell for cell):

- single-file (3 tasks): every executing arm 3/3. Baseline 0/3.
- multi-interface-and-callsites: every executing arm 1/1.
- toolheavy-many-calls: every executing arm 1/1.
- recovery-broken-env: every executing arm 1/1 (environment heal:
  `chmod +x run.sh`).
- multi-both-broken: 0/4 executing arms. Expected from the mechanism:
  the deterministic solver repairs one candidate file at a time and
  rolls each attempt back unless the verifier goes green, so fixing
  `a.sh` alone (or `b.sh` alone) never goes green and the coordinated
  two-file fix is never kept. Only the agentic loop's atomic
  multi-file transaction can solve this one; the fixture exists to
  measure exactly that gap.
- longsession-overflow: 0/4 executing arms. The solver localizes
  correctly ("localized 2 candidate file(s); first: .../bigcalc.py")
  but produces no patch and stops for no progress. Likely cause
  (inferred, not measured): the mutation plan is capped at 64
  candidates and the 240-entry lookup table crowds the faulty line
  out of the budget. The fixture discriminates: repair degrades with
  file size under the bounded deterministic layer.

## Secondaries (measured, run 1)

Per-arm stage totals (ms) over the 8 cells: baseline-verify /
solve / re-verify / rollback, plus total wall.

| arm | baseline verify | solve | re-verify | rollback | wall |
|---|---|---|---|---|---|
| A | 789 | 0 | 0 | 0 | 858 |
| B | 933 | 36,617 | 907 | 0 | 38,475 |
| C | 441 | 39,100 | 974 | 0 | 40,542 |
| D | 728 | 27,215 | 705 | 0 | 28,669 |
| E | 612 | 33,944 | 521 | 0 | 35,093 |

Rollback is 0 ms everywhere: the snapshot diff over these small
trees takes under a millisecond, and `as_millis` truncates. That is a
true sub-millisecond measurement, not a missing one.

Solver peak RSS (VmHWM, sampled every 5 ms): 33-63 MB across the
solve cells. Two A cells (no solve step, sub-10 ms verifies) show the
sampling limit: `multi-both-broken` arm A has no sample at all
(`peak_rss_kb` null: the child exited before the first poll) and
`recovery-broken-env` arm A reads 16 KiB (a mid-exec sample). The
value is a sampled lower bound, never an exact peak.

Not measured, by design: billed tokens, agent turns, tool calls, and
first-token latency are `null` with the reason in each cell's notes
(the deterministic solver spends no tokens and emits no streamed
tokens); dollar cost is 0.0 measured. B/C/D/E all score 6/8 with the
same two misses, but that is replication, not a measured null
effect of the toggles: the three overlay variables have no read site
in the `axiom solve` path, and arm B's overlay is byte-identical to
arm D's (see docs/AXIOMBENCH.md section 4). The arms execute the
same deterministic loop under different labels. Variance across
repeats is still open (repeats 2-3 not run).

## Reproduction

A second full run with the same seed produced an identical run order,
identical per-cell task/arm/success/status, identical success and
rollback rates, and identical guardrail verdicts. Only
`total_wall_ms` differs between the runs (wall clock, expected).
The run used the debug solver binary (`target/debug/axiom`); CI runs
the release binary for the smoke family only.

## Smoke run (CI family)

`axiombench paired --smoke` (arms A/D, single-file, one repeat,
seed 42) reproduces the prep result: arm A 0/3, arm D 3/3, guardrail
PASS. The guardrail demonstration (`--demo-regression`) still fails
the deliberately regressed arm on reliability (unsafe patch rate 1.00,
rollback rate 0.00) while arm D passes.
