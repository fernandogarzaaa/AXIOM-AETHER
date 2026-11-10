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

- 36 fixtures, `bench/repair/` format (`meta.json`, `broken/`,
  `expected.diff`), class-balanced by `--balance`: batch 1 (seed 424242,
  24 fixtures) plus batch 2 (seed 424243, 12 fixtures, see above).
- `SEED.json`: generation summary per batch (emitted, attempted,
  survived-and-excluded, by-class counts, per-fixture compile split).

## Strengthened suite: batch 2 (2026-11-11, seed 424243)

Batch 1 was thin in places: `shift`, `neg`, `cmp`, `flow`, `const` had 1
fault each, `logic`, `argswap`, `range` had 2, and three single-token
fault shapes had no class at all. Batch 2 adds 12 fixtures (24 → 36)
generated with `scripts/seed_faults.py --seed 424243 --balance --n 12
--classes deref,cast,ret,logic,argswap,range,minmax --exclude-dir
bench/corpus/unseen-seed424242`, so no batch-1 fault was re-emitted.
15 attempts, 3 survived the kill filter and were excluded.

New fault classes (verified against `MutationClass::ALL` in
`axiom_engine_rs/src/mutation_repair.rs`, 12 classes):

- `deref`: drop a leading `&` borrow or `*` deref, expression positions
  only (signature lines, generic type arguments, pattern bindings, and
  closure params are excluded by construction). Outside the repair
  operator set: no operator adds `&`/`*`.
- `cast`: change an `as` target type (`u64` → `u32`, `f32` → `f64`).
  Outside the operator set: no operator rewrites a cast target.
- `ret`: drop `return` from an early-return statement
  (`return Err(...);` → `Err(...);`). Inside the operator set:
  MissingReturn re-adds a dropped `return`.

A `deref`/`cast` fault can fail to compile instead of failing a test;
the oracle is `cargo test --lib`, which fails either way, and the
per-fixture compile split for batch 2 (compiles vs compile-error) is
recorded in `SEED.json` under `batches`. (Batch 1 predates the split
recording and carries none; its fixtures are unchanged.) Batch 2 composition: `cast` 3, `ret` 3,
`minmax` 2, `deref` 2, `logic` 1, `range` 1. `argswap` contributed 0:
its remaining sites did not yield a killed fault in this batch. `shift`
and `flow` have exactly one candidate site each in the crate and both
were already used by batch 1, so they cannot contribute more; `strlit`
still contributes 0 (one candidate site, in `gumbel.rs:85`, which the
suite does not catch).

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
