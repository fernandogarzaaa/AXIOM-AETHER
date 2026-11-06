# Held-out repair rate on seeded faults — 2026-11-06

**Result: 19 of 40 seeded faults repaired (47.5%), 0 rollback violations.**
19 of 29 (66%) in fault classes the engine has a mutation operator for; 0 of 11
in the classes it has none for. 14 of the 19 repairs are the exact reversal of
the seeded fault; the other 5 are different edits the crate's own test suite
accepts. Deterministic, CPU-only, no LLM, no network.

Raw table: [`repair-seeded-mesh-core-2026-11-06.md`](repair-seeded-mesh-core-2026-11-06.md).
Rollback attribution evidence: [`repair-seeded-rollback-attribution-2026-11-06.md`](repair-seeded-rollback-attribution-2026-11-06.md).

## What "held-out" means here

The built-in `eval-agentic` fixtures (32/32) were written alongside the repair
engine, so they show the loop works, not how often it works on bugs nobody
designed for it. This corpus is the next step down that road:

- **Code nobody wrote for the engine.** `axiom_mesh_core` (four source files,
  ~1,800 lines of numeric routing code with 41 unit and property tests)
  predates the mutation-repair layer.
- **Faults nobody chose.** `scripts/seed_faults.py --seed 2026 --balance`
  enumerates every single-token fault site in the non-test code, shuffles them
  with a fixed seed, round-robins across fault classes, and keeps a fault only
  if the crate's own test suite catches it. 68 sites were tried; 28 survived the
  tests and were excluded (a fault the oracle cannot see has no defined repair);
  40 were kept.
- **A fault model that includes what the engine cannot do.** Identifier swaps,
  `min`/`max` and `floor`/`ceil` swaps, and `break`/`continue` flips have no
  mutation operator in `mutation_repair.rs`. They are in the sample on purpose,
  so the headline rate cannot be inflated by choosing only faults the engine
  was built to undo.
- **The engine sees only what a user would.** `axiom solve` runs in a scratch
  copy with no file named: it gets the failing `cargo test` output and nothing
  else. Solver state (`~/.axiom` heal and patch memory) is isolated per fixture,
  and the in-file `#[cfg(test)]` region is pinned as the oracle, so editing a
  test scores OVERFIT, not PASS.

Reproduce:

```bash
python3 scripts/seed_faults.py --crate axiom_mesh_core --out /tmp/seeded --n 40 --seed 2026 --balance
AXIOM_REPAIR_CORPUS=/tmp/seeded scripts/bench_repair.sh
```

## Results by fault class

| fault class | engine has an operator? | repaired | of which the exact seeded reversal |
|---|---|---|---|
| comparison (`<` / `<=` / `==` …) | yes | 4/4 | 2 |
| range (`..` ↔ `..=`) | yes | 4/5 | 3 |
| argument swap | yes | 3/4 | 3 |
| boolean literal | yes | 2/3 | 2 |
| logic (`&&` ↔ `\|\|`) | yes | 2/3 | 1 |
| off-by-one (`.len() ± 1`) | yes | 2/3 | 2 |
| arithmetic operator | yes | 1/2 | 1 |
| negated condition | yes | 1/5 | 0 |
| identifier swap | **no** | 0/7 | 0 |
| min/max, floor/ceil | **no** | 0/3 | 0 |
| break/continue | **no** | 0/1 | 0 |
| **total** | | **19/40** | **14** |

Median wall time: 23 s per repaired fixture and 48 s per unrepaired fixture,
where the 64-candidate budget is exhausted. Each candidate is one incremental
`cargo test` rebuild.

## What the numbers do and do not say

- **The 66% is the fair headline for what the engine attempts.** The 0 of 11
  is the honest floor: a single-edit mutator does not repair what it has no
  operator for, and the benchmark shows that instead of hiding it.
- **"Different fix" is not automatically a wrong fix.** In 5 of the 19 the
  accepted edit is not the reversal of the seeded token (for example
  `<=`→`>=` where `&&`→`||` was seeded). The crate's tests accept it, which
  means either an equivalent program or a test suite too weak to tell them
  apart. This benchmark cannot distinguish the two, and that is the general
  test-overfitting problem of automated repair, not something specific here.
- **Negated conditions are the weak spot (1/5).** The seeded fault wraps a
  whole condition in `!(…)`. The engine's boolean operator drops a leading `!`
  before an identifier but does not unwrap `!(…)` around a parenthesised
  condition. This is a concrete next operator to add.
- **Seeded faults are not developer bugs.** Real bugs are often multi-token,
  multi-line, or semantic. This corpus measures single-token faults in real
  code. The developer-bug measurement (QuixBugs, 40 Python programs) is
  prepared (`scripts/import_quixbugs.py`) but was not run here because the
  sandbox blocked executing externally fetched test code.
- **One crate, one language.** The rate will differ on other code. The tooling
  is crate-agnostic (`--crate`), so the next run is a flag away.

## Rollback safety

Zero rollback violations. The first run flagged 12, all of them
`proptest-regressions/`: when a candidate breaks a property test, the
`proptest` library records the failing seed there. The verifier writes that
file, not Axiom, and running the same tests by hand on the same broken
candidate would leave the same file. Those 12 fixtures were re-run with scratch
trees kept, and an independent `diff -r` of each tree against its pristine
fixture, excluding only `proptest-regressions/` and `target/`, found every file
byte-identical. The runner now reports such files separately as "verifier
artifacts" and does not count them as solver edits.

## Two bugs this benchmark found and fixed first

1. **Shared patch memory turned replays into passes.** Fixtures sharing a
   verify command shared `~/.axiom/axiom_patch_memory.json`, so a whole-file
   patch learned on one fixture was replayed on the next. 19 of 23 "passes" in
   the first run were such replays. Solver state is now isolated per fixture.
2. **Past-dated candidate mtimes made every Rust repair fail.** The repair loop
   set each candidate's modification time into the past (to defeat stale
   Python bytecode). Cargo rebuilds only when a source is newer than its output,
   so every candidate ran the old binary and was rejected. That produced a
   spurious 4/40. Timestamps now move forward only (`agentic::mark_fresh`).
   Both fixes are in 0.6.0.
