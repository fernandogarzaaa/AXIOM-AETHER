# Unseen seeded-fault suite — seed 424242, regenerated (Phase 4)

36 seeded faults in `axiom_mesh_core`: 18 retained from batches 1-2
(seeds 424242/424243, generated 2026-11-10/11) plus 18 regenerated
fixtures (batch 3, seed 424244, generated 2026-11-11).

## Why this suite is unseen

- **Zero triple overlap with every tuning table.** Each fixture's
  (file, line, class) triple was checked mechanically against the
  union of all prior repair-seeded-mesh-core tuning tables
  (`bench/results/repair-seeded-mesh-core-2026-11-06.md`,
  `-v2-2026-11-06.md`, `-v3-2026-11-07.md`,
  `repair-seeded-mesh-core-seed7-2026-11-06.md`,
  `-seed7-v0.6.0-baseline-2026-11-06.md`, `-seed7-v3-2026-11-07.md`,
  `repair-seeded-rollback-attribution-2026-11-06.md`: 56 triples).
  The shipped 36 share none of them, and no two fixtures in the suite
  share a triple. The exclusion list is committed as
  `EXCLUDED-TRIPLES.txt` in this directory, and the generator enforces
  it (`scripts/seed_faults.py --exclude-triples`).
- **Different seed and generator revision.** Seeds 2026 and 7 were the
  tuning and held-out corpora in
  `bench/results/SEEDED-HELDOUT-v2-2026-11-06.md`. Seed 424242 was
  generated after all repair operators were written, and the generator
  has since gained fault classes (`strlit`, `deref`, `cast`, `ret`)
  that did not exist when seeds 7/2026 were generated.
- **Same kill filter and oracle as before.** A fault is kept only when
  the crate's own `cargo test --lib` catches it; the oracle is that
  suite, with the test region pinned (any edit at or below the
  `#[cfg(test)]` line scores OVERFIT).

Honest history: a different seed alone did NOT make the first 36
unseen. Codex review of PR #294 (2026-11-10) found exactly 18 of 36
fixtures share a (file, line, class) triple with the tuning tables,
so the suite as first shipped was not genuinely unseen (tuned 18
repaired 12, 66.7%; fresh 18 repaired 7, 38.9%; headline 19/36,
52.8%). Those 18 fixtures were deleted and replaced by batch 3 at
fresh sites; the numbers above are the pre-regeneration baseline,
kept here so the change is auditable, not hidden.

## Contents

- 36 fixtures, `bench/repair/` format (`meta.json`, `broken/`,
  `expected.diff`): 18 retained from batches 1-2 plus 18 batch-3
  fixtures at fresh sites.
- `SEED.json`: generation summary per batch (emitted, attempted,
  survived-and-excluded, by-class counts, per-fixture compile split).
- `EXCLUDED-TRIPLES.txt`: the (file, line, class) triples batch 3
  must never emit (56 tuning triples + 18 retained triples + 18
  batch-3 triples).
- Portable verify commands: every `meta.json` resolves its shared
  target directory from the runtime environment
  (`CARGO_TARGET_DIR="${TMPDIR:-$HOME/.cache}/axiom_seeded_target"`).
  No committed fixture contains a machine-specific absolute path
  (Codex P1, PR #294).

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

## Regenerated suite: batch 3 (2026-11-11, seed 424244)

The 18 overlapping fixtures (listed in the PR #294 body) were deleted
and replaced with 18 fixtures generated with
`scripts/seed_faults.py --seed 424244 --balance --n 18 --exclude-dir
bench/corpus/unseen-seed424242 --exclude-triples
bench/corpus/unseen-seed424242/EXCLUDED-TRIPLES.txt`.
29 attempts, 11 survived the kill filter and were excluded.
Composition: `offbyone` 4, `arith` 3, `deref` 3, `idswap` 3, `cast` 1,
`cmp` 1, `const` 1, `neg` 1, `ret` 1. Compile split: 10 compile, 8
compile-error (recorded per fixture in `SEED.json`).

Class coverage is narrower than the first 36, and the narrowing is
structural, not a sampling accident. Enumerating every candidate site
in the crate against the exclusion list: `bool` has 3 candidate sites
and all 3 are tuning triples; `flow` has 1, tuned; `shift` has 1,
retained. No fresh `bool`, `flow`, or `shift` fixture can exist in
this crate. `range`, `logic`, and `argswap` have exactly 1 available
site each; a top-up run (seed 424245, `--classes
range,logic,argswap`, 3 attempts) found all 3 survive the kill filter,
so the suite has no fixture in those classes either. The shipped
mix is `arith` 5, `cast` 4, `cmp` 2, `const` 2, `deref` 5, `idswap` 6,
`minmax` 1, `neg` 2, `offbyone` 4, `ret` 4, `shift` 1. The headline
rate must be read with the fresh-site split (see the baseline
write-up): the suite measures repair on genuinely unseen sites, in
the classes this crate can still supply.

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

Baseline result: recorded in `bench/results/UNSEEN-SEED424242-2026-11-11.md`
(see that file for the number; this README does not duplicate it, so the
two cannot drift apart). The 2026-11-10 write-up in the same directory
is the pre-regeneration baseline (19/36 with the overlap) and is kept
as history.
