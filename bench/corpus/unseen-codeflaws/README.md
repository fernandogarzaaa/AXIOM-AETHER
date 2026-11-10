# Unseen human-written bug suite — Codeflaws (Phase 4)

Human-written C defects crawled from Codeforces (Tan et al.,
https://github.com/codeflaws/codeflaws, 3902 defects), imported
2026-11-10 with `scripts/import_codeflaws.py`.

## Why this suite is unseen

- **No tuning against it.** No repair operator, candidate ordering
  rule, or fault-localization heuristic was ever developed or adjusted
  on Codeflaws. The engine's operators were tuned on seeded Rust
  faults and the 32 authored `eval-agentic` fixtures.
- **Different language and shape from QuixBugs.** QuixBugs is 40
  Python algorithms with pytest oracles. Codeflaws is single-file C
  submissions judged by program output on held test inputs: a
  different language, a different oracle (stdout diff, not assertions),
  and bugs written by competitive programmers under contest conditions,
  not by benchmark authors.
- **Deterministic selection, no peeking.** The import walks
  `codeflaws-defect-detail-info.txt` in file order and takes the first
  defects with type WRONG_ANSWER, a buggy submission of at most 150
  lines, a buggy-vs-accepted diff of at most 15 lines, and at least 2
  repair-test inputs. Each imported defect is sanity-checked at import
  time: the accepted program must PASS the fixture verifier and the
  buggy program must FAIL it, or the defect is skipped. Selection
  never looked at whether the engine repairs the defect.

## Contents

- Fixtures in `bench/repair/` format (`meta.json`, `broken/`,
  `expected.diff`). Each fixture's oracle is `verify.sh`, which
  compiles the program with the Codeforces flags from the defect
  Makefile and runs the repair-test inputs (`input-posN` must keep
  passing, `input-negN` must start passing), normalizing program
  stdout the way the defect's own `test-genprog.sh` does (blank lines
  removed, leading whitespace stripped) and diffing against the
  expected output with trailing whitespace ignored. Everything except
  the buggy `.c` file is pinned: editing the oracle scores OVERFIT.
  The held-out inputs (`heldout-input-*` / `heldout-output-*`) ship
  pinned for qualification-time validation; the repair oracle does not
  run them.
- `IMPORT.json`: import summary (imported count, per-fixture details,
  skip reasons).

## Exclusion rule

This suite is excluded from all tuning until the qualification run.
Do not add operators, reorder candidates, or change the repair loop
based on its per-fixture outcomes before then. The baseline number is
recorded in `bench/results/` (see below), not gamed.

## Baseline

Run with the existing repair harness (deterministic, no LLM):

```bash
AXIOM_REPAIR_CORPUS=bench/corpus/unseen-codeflaws scripts/bench_repair.sh
```

Baseline result: recorded in `bench/results/UNSEEN-CODEFLAWS-2026-11-10.md`
(see that file for the number; this README does not duplicate it, so the
two cannot drift apart).
