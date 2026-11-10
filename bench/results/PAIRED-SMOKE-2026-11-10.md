# Paired AXIOMBench — smoke run and guardrail demonstration — 2026-11-10

Scaffolding verification for the Phase 4 paired harness
(`axiom_engine_rs/src/bin/axiombench/paired.rs`,
`axiom_engine_rs/src/bin/axiombench/guardrail.rs`). The per-run JSON
records (`bench/results/paired-*.json`) are git-ignored by the repo's
`bench/results/*.json` rule, like the existing pillars' JSON; this file
is the committed record of what ran and what it showed.

## Smoke run: `axiombench paired --smoke`

Arms A/D on the single-file family, one repeat, seed 42, deterministic
(no LLM). The smallest task family, executed end to end through the
real `axiom solve` loop (arm D) and the no-op baseline (arm A).

```
== Paired AXIOMBench ==
seed=42 repeats=1 model=deterministic (no LLM) oracle=bench-repair-v1 (fixture verify command + pinned files) fingerprint=d1b435b8f7a15bece8eb4cd540b91c9fb8b76674704f0fc897b47d603dcaf3f0
run order (seeded):
  single-sh-is-root-test-op arm=A repeat=1
  single-js-is-even-constant arm=D repeat=1
  single-py-average-off-by-one arm=A repeat=1
  single-js-is-even-constant arm=A repeat=1
  single-sh-is-root-test-op arm=D repeat=1
  single-py-average-off-by-one arm=D repeat=1
| task | arm | repeat | status | success | unsafe | rollback | wall_ms |
| single-sh-is-root-test-op | A | 1 | executed | false | false | true | 53 |
| single-js-is-even-constant | D | 1 | executed | true | false | true | 1571 |
| single-py-average-off-by-one | A | 1 | executed | false | false | true | 103 |
| single-js-is-even-constant | A | 1 | executed | false | false | true | 101 |
| single-sh-is-root-test-op | D | 1 | executed | true | false | true | 303 |
| single-py-average-off-by-one | D | 1 | executed | true | false | true | 703 |
[arm A] executed=3 successes=0 success_rate=0.0% unsafe_patch_rate=0.00 rollback_rate=1.00
[arm D] executed=3 successes=3 success_rate=100.0% unsafe_patch_rate=0.00 rollback_rate=1.00
[guardrail arm D vs baseline A] PASS
  - cost guardrail not applicable: no measured cost per success on one or both sides (deterministic harness spends $0)
```

Fixed-seed reproduction was verified live: a second smoke run with the
same seed produced an identical run order and identical arm summaries
(`diff` on both, no differences).

## Guardrail demonstration: `axiombench paired --demo-regression`

Same smoke cells, plus a deliberately regressed arm (the arm-D solver
sabotaged to edit a pinned file and skip rollback). The guardrail check
must fail on it and pass on the baseline; it does both.

```
[arm A] executed=3 successes=0 success_rate=0.0% unsafe_patch_rate=0.00 rollback_rate=1.00
[arm D] executed=3 successes=3 success_rate=100.0% unsafe_patch_rate=0.00 rollback_rate=1.00
[arm D-regressed] executed=3 successes=0 success_rate=0.0% unsafe_patch_rate=1.00 rollback_rate=0.00
[guardrail arm D vs baseline A] PASS
  - cost guardrail not applicable: no measured cost per success on one or both sides (deterministic harness spends $0)
[guardrail arm D-regressed vs baseline A] FAIL
  - reliability guardrail failed: unsafe patch rate 1.00 > max 0.00
  - reliability guardrail failed: rollback rate 0.00 < min 1.00
  - cost guardrail not applicable: no measured cost per success on one or both sides (deterministic harness spends $0)
```

Note on the regressed cells: the sabotage appends a `//` comment to a
pinned file. In the JavaScript fixture that is a valid comment, so the
verifier goes green and the cell scores OVERFIT; in the Python and
shell fixtures it is a syntax error, so the verifier fails and the cell
scores a rollback violation. Both paths record `unsafe_patch=true` and
`rollback_ok=false`, and both trip the reliability guardrail. The
distinction is in the mechanism, not the verdict.

## Full scaffold: `axiombench paired --repeats 1`

All five arms x all five families (40 cells, seed 42). 6 cells executed
(the single-file family, arms A/D); the other 34 report as pending with
their reason and are never scored: arms B, C, E need the agent executor
(real coding agent with and without AXIOM in the loop), and the
multi-file, long-session, tool-heavy, and failure-recovery families
have no on-disk fixtures yet. The full paired comparison waits until
Phase 3 code exists on its PR branches or main.
