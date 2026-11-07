# Held-out repair rate on seeded faults, second pass — 2026-11-06

**Result: 39 of 40 (seed 2026) and 38 of 40 (seed 7) seeded faults repaired,
0 rollback violations.** The 0.6.0 engine repaired 19 of 40 on seed 2026
([first write-up](SEEDED-HELDOUT-2026-11-06.md)) and 20 of 40 on seed 7
(published 0.6.0 Linux binary, same runner, same corpus). On the 17 seed-7
faults whose site and class do not occur in the seed-2026 set, the change is
**6 → 15 of 17**. Deterministic, CPU-only, no LLM, no network.

**Correct, not just test-passing: 33/40 and 32/40** (0.6.0: 16/40 and 16/40),
5 → 14 of the 17 fresh faults. Every accepted edit that is not the seeded
reversal was re-run against a strengthened `axiom_mesh_core` test suite (seven
tests added for this audit). The equivalent ones still pass; the wrong ones
are each rejected by the new test written for that gap. See
[Correct vs test-passing](#correct-vs-test-passing).

Raw tables:
[seed 2026](repair-seeded-mesh-core-2026-v2-2026-11-06.md) ·
[seed 7](repair-seeded-mesh-core-seed7-2026-11-06.md) ·
[seed 7, 0.6.0 baseline](repair-seeded-mesh-core-seed7-v0.6.0-baseline-2026-11-06.md).

## How the method guards against fitting the benchmark

Seed 2026 is the corpus the first write-up reported. Every change below was
developed against it, so its 39/40 is an in-sample number. Seed 7 was generated
after the operators were written and was scored with the code unchanged. 23 of
its 40 faults are on a line and class that seed 2026 also has (one crate gives
a finite set of fault sites), so the clean out-of-sample measurement is the
**17 fresh faults: 15/17 repaired, against 6/17 for 0.6.0.**

The fault model, the corpus generator, the oracle pinning and the runner did
not change. `seed_faults.py` still includes identifier swaps, `min`/`max` and
`break`/`continue` faults. The difference is that the engine now has operators
for them.

## Results by fault class

| fault class | 0.6.0, seed 2026 | now, seed 2026 | now, seed 7 | of which the exact seeded reversal (2026 / 7) |
|---|---|---|---|---|
| comparison (`<` / `<=` / `==` …) | 4/4 | 4/4 | 3/3 | 2 / 1 |
| range (`..` ↔ `..=`) | 4/5 | 5/5 | 5/5 | 3 / 3 |
| argument swap | 3/4 | 4/4 | 4/4 | 4 / 4 |
| boolean literal | 2/3 | 3/3 | 3/3 | 2 / 2 |
| logic (`&&` ↔ `\|\|`) | 2/3 | 3/3 | 3/3 | 1 / 1 |
| off-by-one (`.len() ± 1`) | 2/3 | 3/3 | 2/2 | 3 / 2 |
| arithmetic operator | 1/2 | 1/2 | 1/2 | 1 / 1 |
| negated condition | 1/5 | 5/5 | 4/4 | 2 / 2 |
| identifier swap | 0/7 | 7/7 | 7/7 | 7 / 7 |
| min/max, floor/ceil | 0/3 | 3/3 | 2/3 | 3 / 2 |
| break/continue | 0/1 | 1/1 | 1/1 | 1 / 1 |
| integer constant | — | — | 3/3 | — / 2 |
| **total** | **19/40** | **39/40** | **38/40** | **29 / 28** |

Median wall time per repaired fixture: 8 s, against 23 s in the first run.
Unrepaired fixtures take about 37 s, the time to exhaust the 64-candidate
budget. Each candidate is one incremental `cargo test` rebuild. The new runs
drove the solver with a debug `axiom` binary and the baseline with the 0.6.0
release binary; wall time is dominated by the fixture's own build and tests.

## What changed (all in the open, all unit-tested)

1. **Operators for the four classes at 0.** These are `api-swap` (`min`↔`max`,
   `floor`↔`ceil`, `first`↔`last`, `starts_with`↔`ends_with`, `any`↔`all`,
   `break`↔`continue`) and `ident-swap` (exchange two identifiers on a line,
   at most 15 pairs per line). The boolean class also gained three edits: add
   a dropped `!` to an `if`/`while` condition or a closure body, and unwrap
   `!( … )`.
2. **Candidate ordering that follows the call graph.** After the blamed lines,
   the planner tries these in order: the rest of the function that raised, the
   bodies of the functions the failing test calls, what those call, and then
   the rest of the file. Identifier swaps off the blamed lines come last.
3. **Three bugs that were costing repairs, each found by this benchmark:**
   - *The trace was cut to its last 4000 characters.* With `RUST_BACKTRACE=1`
     the frames that cite the faulty line sit near the top, so localization
     worked from noise. The loop now keeps the first 2000 and the last 4000
     characters.
   - *Candidate mtimes rounded down.* A candidate written in the same second
     the previous build started looked no newer than that build, so cargo
     reran the stale binary. This was timing-dependent: a fix that was
     candidate 1 passed by hand and failed in the run.
   - *Debug builds of every model-loading subcommand crashed at start-up*
     (`reqwest::blocking` client built inside the async runtime). This did not
     affect the release binaries the first write-up used.

## A planner diagnostic that predicts the end-to-end result

`tests/mutation_plan_diag.rs` (ignored; needs a corpus and recorded traces)
plans each fixture offline and reports where the reversal of the seeded fault
ranks. It reads the trace bounded exactly as the solve loop bounds it. The
diagnostic runs in seconds, while a benchmark run takes about ten minutes.

| corpus | reversal within the 64-candidate budget (0.6.0 planner) | now | end-to-end now |
|---|---|---|---|
| seed 2026 | 19/40 | 38/40 | 39/40 |
| seed 7 | 19/40 | 36/40 | 38/40 |

Every fixture still unrepaired end to end is one whose reversal the diagnostic
ranks past the budget (ranks 82–85). The end-to-end count can exceed the
diagnostic count because the engine can find an accepted fix that is not the
reversal (see below).

## Correct vs test-passing

A repair counts in the headline when the crate's own 41 tests accept it, which
is the "plausible patch" of the repair literature. 10 of the 39 (seed 2026)
and 10 of the 38 (seed 7) accepted edits are not the seeded reversal, so each
was checked against a stronger oracle:

1. Re-create the exact accepted edit. The planner diagnostic writes every
   candidate with the logged label. A label can name two spans, for example
   when a line holds two `!=`. The one that passes the fixture's original
   tests is the one the benchmark kept.
2. Splice its code in front of the pristine crate's `#[cfg(test)]` region,
   after seven tests were added to `axiom_mesh_core` for the gaps the wrong
   edits exposed. Then re-run the tests
   (`scripts/reverify_accepted_patches.py`).

| accepted edit (seeded fault → edit) | verdict | caught by |
|---|---|---|
| `0..=n` in `sample_gumbel` → its only caller passes `len() - 1` (`len` asserted ≥ 1) | equivalent | — |
| `0..=len` → `0..=len - 1` (`len` ≥ 1 on that path) | equivalent | — |
| `!(g != 0)` → `!(g == 0)`; `!(len != dim)` → `!(len == dim)` | equivalent (De Morgan) | — |
| `!(x)` / `!(!x)` → drop the outer `!` | equivalent (redundant parentheses) | — |
| `a > 0 \|\| a <= 1` → `a > 0 \|\| a >= 1` | **wrong**: accepts `alpha = 1.5` | `smoother_rejects_alpha_above_one` |
| `(h >> 64) & 1` → `(h >> 63) & 1` | **wrong**: changes every hashed sign | `feature_hashing_is_stable` |
| default `hard: false` → negate `if self.config.hard` | **wrong**: inverts an explicit `hard` setting | `hard_routing_is_the_default_and_soft_routing_keeps_adhesion` |
| `g != 0 && e != 0` → `g != 0 && e == 0`; `!(g != 0 \|\| e != 0)` → `!(g == 0 \|\| e != 0)` | **wrong**: bandit terms gated on the wrong knob | `bandit_terms_apply_when_either_knob_is_set` |
| `len == dim` → `len - 1 == dim` | **wrong**: lets most mismatched payloads through | `every_entry_point_rejects_a_payload_of_the_wrong_dimension` |
| `slots > cap` → `slots += 2` | **wrong**: right only for capacity ≤ 2 | `batch_dispatch_fills_a_capacity_above_two_exactly` |

| | 0.6.0, seed 2026 | now, seed 2026 | 0.6.0, seed 7 | now, seed 7 |
|---|---|---|---|---|
| accepted by the original 41 tests | 19 | 39 | 20 | 38 |
| exact reversal | 14 | 29 | 13 | 28 |
| equivalent edit | 2 | 4 | 3 | 4 |
| **correct** | **16** | **33** | **16** | **32** |
| wrong, accepted only because a test was missing | 3 | 6 | 4 | 6 |

The audit agrees exactly with a reading of each diff: all 13 edits judged
equivalent pass the stronger suite, and each of the 19 judged wrong fails it.
The wider search space doubles the correct repairs, and it also finds more
edits that only a weak test lets through (3–4 → 6). For a user that is the
standing caveat of test-driven repair: a green suite certifies only what the
suite checks. The seven new tests now ship with `axiom_mesh_core`.

## What the numbers do and do not say

- **The remaining misses are a budget problem, not an operator problem.** All
  three are faults in a function the failing test reaches only indirectly,
  through the rest of the file (tier 4), and the reversal ranks just past 64.
  Raising `AXIOM_REPAIR_MAX_CANDIDATES` would repair them at the cost of time
  on every unrepairable fault. The default stays at 64.
- **Seeded faults are not developer bugs.** Real bugs are often multi-token,
  multi-line or semantic. Identifier swaps in particular are seeded as
  "exchange the first two names on a line", and the operator undoes exactly
  that family. A developer's wrong variable is often a one-sided substitution,
  which `ident-swap` does not cover. On developer bugs (QuixBugs, 40 Python
  programs) the same engine repairs 9/40, all correct:
  [`QUIXBUGS-2026-11-07.md`](QUIXBUGS-2026-11-07.md).
- **One crate, one language.** Both corpora come from `axiom_mesh_core`, and
  every operator and tier change is exercised here only on Rust. The Python
  `def`-body tier and the Python/JS forms of the new operators are covered by
  unit tests, not by a held-out run.

## Reproduce

The corpora were generated before the seven audit tests were added to
`axiom_mesh_core`. `seed_faults.py` keeps only faults the crate's tests catch,
so regenerating on a later commit yields a different corpus. To reproduce
these exact fixtures, run the generator at commit `a13ddf4`.

```bash
python3 scripts/seed_faults.py --crate axiom_mesh_core --out /tmp/seeded  --n 40 --seed 2026 --balance
python3 scripts/seed_faults.py --crate axiom_mesh_core --out /tmp/seeded7 --n 40 --seed 7    --balance
AXIOM_REPAIR_CORPUS=/tmp/seeded  scripts/bench_repair.sh
AXIOM_REPAIR_CORPUS=/tmp/seeded7 scripts/bench_repair.sh
# baseline: the same with AXIOM_BIN=<0.6.0 release binary>
```

Offline planner ranks for a corpus (`TRACES/<fixture>.txt` = that fixture's
failing `cargo test` output, collected from a fresh `cp -R` copy):

```bash
AXIOM_PLAN_DIAG_CORPUS=/tmp/seeded7 AXIOM_PLAN_DIAG_TRACES=/tmp/traces7 \
AXIOM_PLAN_DIAG_PRISTINE=../axiom_mesh_core \
  cargo test --test mutation_plan_diag -- --ignored --nocapture
```
