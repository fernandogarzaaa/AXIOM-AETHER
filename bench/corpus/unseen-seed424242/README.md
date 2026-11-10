# Unseen seeded-fault suite — seed 424242 (Phase 4)

24 fresh seeded faults in `axiom_mesh_core`, generated 2026-11-10 with
`scripts/seed_faults.py --seed 424242 --balance --n 24`.

## Why this suite is unseen

- **Different seed.** Seeds 2026 and 7 were the tuning and held-out
  corpora in `bench/results/SEEDED-HELDOUT-v2-2026-11-06.md`. Seed
  424242 was generated after all repair operators were written, and no
  operator or ordering rule was tuned against it.
- **Different generator revision.** `seed_faults.py` gained a new fault
  class for this suite (`strlit`: append `!` to a short plain string
  literal), which did not exist when seeds 7/2026 were generated.
  Honest caveat: the class contributed 0 of the 24 faults, because
  `axiom_mesh_core` has exactly 1 `strlit` candidate site and the
  crate's own test suite does not catch it (kill filter excludes
  survived faults). The generator differs; the faults are new.
- **Same kill filter and oracle as before.** A fault is kept only when
  the crate's own `cargo test --lib` catches it; the oracle is that
  suite, with the test region pinned (any edit at or below the
  `#[cfg(test)]` line scores OVERFIT).

## Contents

- 24 fixtures, `bench/repair/` format (`meta.json`, `broken/`,
  `expected.diff`), class-balanced by `--balance`.
- `SEED.json`: generation summary (24 emitted, 33 attempted, 9
  survived and excluded; by-class counts).

## Exclusion rule

This suite is excluded from all tuning until the qualification run.
Do not add operators, reorder candidates, or change the repair loop
based on its per-fixture outcomes before then. The baseline number is
recorded in `bench/results/` (see below), not gamed.

## Baseline

Run with the existing repair harness (deterministic, no LLM):

```bash
AXIOM_REPAIR_CORPUS=bench/corpus/unseen-seed424242 scripts/bench_repair.sh
```

Baseline result: recorded in `bench/results/UNSEEN-SEED424242-2026-11-10.md`
(see that file for the number; this README does not duplicate it, so the
two cannot drift apart).
