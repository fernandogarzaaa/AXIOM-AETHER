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

## Held-out validation (2026-11-11, run Nov 10 UTC / Nov 11 Manila)

**4 of the 7 repair-test passes also pass every held-out input.**
Full per-fixture table:
[`heldout-codeflaws-2026-11-11.md`](heldout-codeflaws-2026-11-11.md).
Method: `scripts/validate_heldout_codeflaws.py` re-ran each accepted
patch (from the `AXIOM_BENCH_KEEP=1` harness re-run,
[`repair-unseen-codeflaws-rerun-2026-11-10.md`](repair-unseen-codeflaws-rerun-2026-11-10.md))
against the pinned `heldout-input-*` / `heldout-output-*` files with the
same compile flags and stdout normalization as `verify.sh`, and ran the
same oracle on the reference program (`expected.diff` applied) as a
sanity check.

| program | repair-test | held-out | verdict |
|---|---|---|---|
| 131-A | PASS (authored fix) | 55/55 | HELDOUT-PASS |
| 133-A | PASS (authored fix) | 84/84 | HELDOUT-PASS |
| 136-A | PASS (different fix, MXS=300 vs reference 1024) | 100/100 | HELDOUT-PASS |
| 231-A | PASS (different fix, ident-swap vs reference octal `011`→`11`) | 7/10 | HELDOUT-FAIL |
| 467-A | PASS (different fix, `(p+2)<=q` vs reference `(q-p)>=2`, algebraically identical) | 24/24 | HELDOUT-PASS |
| 58-A-17762757 | PASS (different fix, `g=0`→`g=1` vs reference `N0`→`NO`) | solver 25/39, reference 30/39 | REFERENCE-INCONSISTENT |
| 96-A-18284844 | PASS (different fix, `s[i]==s[i]` vs reference `count=0`→`count=1`) | 21/43 | HELDOUT-FAIL |

The two HELDOUT-FAILs are genuine overfits the repair tests cannot see:
231-A swaps the `scanf` argument order instead of fixing the octal
literal, and 96-A-18284844 replaces the comparison with the tautology
`s[i]==s[i]`, which passes the 3 repair tests and fails 22 of 43
held-out inputs. The 58-A-17762757 verdict is different: the REFERENCE
program itself fails 9 of 39 held-out inputs, so the held-out set
cannot discriminate solver from reference there. Spot-checked by hand:
heldout-input-pos25 is a 100-character string, and the reference reads
it with `gets(a)` into `char a[100]`, so glibc aborts on buffer
overflow and the program prints nothing (expected `YES`). The solver
patch fails the same 9 inputs plus 5 the reference passes (pos12, 15,
18, 32, 38), so it is strictly worse than the reference on held-out
inputs, but the fixture cannot yield a clean pass/fail verdict. The
qualification run should treat held-out validation as a second score
(4/7 validated) alongside the repair-test score (7/12), not as a
replacement for it.

## Provenance notes (2026-11-11)

- `expected.diff` repair: the 12 shipped `expected.diff` files were
  unappliable blobs (see `scripts/import_codeflaws.py`: the importer
  used `splitlines()` without `keepends`, so difflib emitted body lines
  with no separators and `patch` could not apply them). Each file was
  reconstructed from its blob plus the buggy program and verified by
  patch-apply plus `verify.sh` on the reference program. The importer
  is fixed to keep line endings. Consequence: the harness's
  authored-fix/different-fix labelling now works for Codeflaws (it
  silently printed "—" for every PASS before); the re-run table above
  labels 2 authored fixes and 5 different fixes. The 7/12 number is
  unchanged by the repair: the oracle never reads `expected.diff`.
- Harness fix: `scripts/bench_repair.sh` now resolves the corpus to an
  absolute path, because the labelling step runs `patch` after `cd`
  into a temp dir and a relative corpus path made the `expected.diff`
  redirect miss (same silent "—" symptom as above).
- The re-run reproduced the baseline exactly (7/12, same per-fixture
  outcomes and repair descriptions), so the baseline number stands.

## Caveats

- n=12 is small. The 58.3% is a baseline number for the qualification
  run, not a general repair rate. The programs are short (11-54 lines)
  and the selection favors small diffs, so this suite is easier than
  arbitrary Codeforces submissions.
- The oracle is the repair-test set only. Held-out validation was run
  after the baseline (see the section above) and is reported as a
  separate 4/7 number, not folded into the 7/12.
