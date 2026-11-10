# Human-written bugs: Codeflaws (C) — 2026-11-10

**Result: 7 of 12 real human-written bugs repaired (58.3%), 0 rollback
violations, 0 edits to a test.** This is the second held-out
developer-bug measurement after QuixBugs (9/40, 22.5%), on a different
language and oracle shape: single-file C programs judged by stdout on
test inputs, not Python programs with pytest assertions.

Raw table: [`repair-unseen-codeflaws-2026-11-10-2026-11-09.md`](repair-unseen-codeflaws-2026-11-10-2026-11-09.md).

## Setup

- **Corpus.** [Codeflaws](https://github.com/codeflaws/codeflaws) (Tan et
  al., 3902 human-written C defects crawled from Codeforces), imported
  with `scripts/import_codeflaws.py`. Selection was deterministic and
  made without looking at repair outcomes: the first defects in
  detail-file order with type WRONG_ANSWER, a buggy submission of at
  most 150 lines, a buggy-vs-accepted diff of at most 15 lines, and at
  least 2 repair-test inputs including at least 1 failing one. 1 defect
  was skipped at import because its buggy program already passed the
  repair-test oracle (its bug shows only on held-out tests).
- **Oracle.** Each fixture's `verify.sh`, mirroring the defect's own
  `test-genprog.sh`: compile with the Codeforces flags, run the
  repair-test inputs (`input-posN` must keep passing, `input-negN`
  must start passing), normalize stdout (blank lines removed, leading
  whitespace stripped), diff with trailing whitespace ignored. The
  accepted program is not in the fixture tree. Everything except the
  buggy `.c` file is pinned.
- **Held out.** No repair operator or ordering rule was tuned on
  Codeflaws. The held-out test files ship pinned in each fixture for
  qualification-time validation; the repair oracle does not run them,
  so a PASS here is a repair-test pass, not a held-out-validated pass.
- **Run.** `scripts/bench_repair.sh` with
  `AXIOM_REPAIR_CORPUS=bench/corpus/unseen-codeflaws`,
  `AXIOM_SOLVE_VERIFY_TIMEOUT_SECS=30`. Deterministic, CPU-only, no
  LLM, no network, candidate cap 64. One fixture (271-A) took 66.5 s;
  the rest took 1-6 s each.

## The seven repairs

All seven are single-token fixes inside the engine's operator set:

| program | accepted edit | shape |
|---|---|---|
| 131-A | off-by-one, `1 -> 0` | off-by-one |
| 133-A | off-by-one, `4 -> 3` | off-by-one |
| 136-A | off-by-one, `2 -> 3` | off-by-one |
| 231-A | identifier swap, `p` <-> `v` | ident-swap |
| 467-A | comparison, `>=` -> `<=` | comparison |
| 58-A-17762757 | off-by-one, `0 -> 1` | off-by-one |
| 96-A-18284844 | off-by-one, drop `+ 1` | off-by-one |

## The five misses, by fix shape

None of the five is a multi-line restructure. Three are string-literal
fixes the mutator has no operator for, and two are constants whose
correct value the trace does not supply:

| program | reference fix | why it is out of reach |
|---|---|---|
| 271-A | `while(i<=9000)` -> `while(i<=9012)` | stale constant; 9012 is not within ±1 and no file supplies it (the open gap in docs/BENCH-REPAIR.md) |
| 4-A | `w>=w` -> `w>=4` | wrong constant; the value 4 is not derivable from the trace |
| 58-A-18163826 | `printf("No")` -> `printf("NO")` | string literal; no operator guesses the original literal |
| 82-A | `printf("sheldon")` -> `printf("Sheldon")` | string literal; same |
| 96-A-18173471 | `printf("No")` -> `printf("NO")` | string literal; same |

The string-literal misses are the same class as the `strlit` fault
class added to `scripts/seed_faults.py` for the seed-424242 suite:
outside the operator set by construction, so the rate is not
tautological.

## Caveats

- n=12 is small. The 58.3% is a baseline number for the qualification
  run, not a general repair rate. The programs are short (11-54 lines)
  and the selection favors small diffs, so this suite is easier than
  arbitrary Codeforces submissions.
- The oracle is the repair-test set only. Held-out validation
  (`heldout-input-*`) was not run as part of this baseline; a PASS
  could in principle fail a held-out input, and the qualification run
  should check that.
