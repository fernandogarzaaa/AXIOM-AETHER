# Autonomous repair — the spearhead benchmark

AXIOM's most concrete, most differentiated capability is **verifier-gated,
reversible, autonomous code repair**: given a failing verify command, localize
the fault, apply a fix under an all-or-nothing transaction, and keep it only if
the verifier passes (and, for the agentic path, a held-out check too). This is
the number the project should be judged on, and the one the
[proof loop](PROOF-LOOP.md) leads with.

Today that number comes from `axiom eval-agentic`. This document defines what it
measures now, how to grow it into a real held-out benchmark, and how to compare
it against alternatives.

---

## What runs today: `axiom eval-agentic`

`src/agentic_eval.rs` → `builtin_cases()` materializes seeded broken projects in
temp dirs and drives each through the real `solve` / `agentic_loop` path.
Deterministic, offline, no model required, so the score is reproducible in CI.

```bash
cargo run --release --bin axiom -- eval-agentic --require-all
# → [PASS] shell-exit-flip … score: 32/32 = 100%   (exit 0; non-zero on any regression)
```

The repair itself comes from two deterministic layers in `solve.rs`:

1. **Poly-JIT flips** (`poly_jit.rs`) — the original fixture-shaped patches
   (`exit 1` → `exit 0`, a `FAIL` marker, one `assert_eq!`).
2. **Mutation repair** (`mutation_repair.rs`, 2026-11-05) — the general layer.
   `fault_locate::repair_targets` ranks the blamed files, **demotes test
   files** and hops through their imports (`from x import`, `require('./x')`,
   `require_relative`, `. ./lib.sh`, `#include "x.h"` → `x.c`, Go package
   siblings) to the module under test. For each target, `mutation_repair::plan`
   enumerates at most 64 single-edit candidates — comparison operators,
   off-by-one (`± 1`, `len(x) - 1`, `range(n + 1)`, `..=`), boolean
   inversion, `and`/`or`, arithmetic operators, swapped call arguments, missing
   `return` (incl. Rust's trailing `;`), shell `test` operators, and a
   misspelled import rewritten to the closest sibling file when the trace says
   `No module named` / `Cannot find module` / `cannot load such file` /
   `fatal error: x.h: No such file` / `file not found for module` — blamed
   lines first, class order steered by the trace (`IndexError` → off-by-one
   first, `undefined`/`NoneType` → missing return first, a missing module →
   import path first, …), and **never an assertion line**. Each candidate is one verifier run under the same
   byte-for-byte rollback as the LLM path.

| # | Fixture | Verifier | Exercises |
|---|---|---|---|
| 1–3 | `shell-exit-flip`, `fixture-marker`, `multi-file-pick-failing` | scripted `sh` | exit-code / marker repair; pick the failing file |
| 4–12 | `rust-assert-flip`, `python-frame-localize`, `js-stack-localize`, `go-frame-localize`, `c-compiler-localize`, `ts-frame-localize`, `java-frame-localize`, `ruby-frame-localize`, `nested-path-localize` | scripted `sh` | one trace dialect each → file (+ nested path) |
| 13–14 | `agentic-multi-file-both-broken`, `agentic-holdout-split` | scripted `sh` | atomic multi-file transaction; train **and** held-out gate |
| 15 | `py-off-by-one-range` | `python3 test_total.py` | `range(len(xs) - 1)`; import hop; test pinned |
| 16 | `py-wrong-comparison` | `python3` | `>` → `>=` |
| 17 | `py-missing-return` | `python3` | bare expression → `return …` |
| 18 | `py-swapped-args` | `python3` | swapped call arguments |
| 19 | `py-boolean-inversion` | `python3` | stray `not` (3 asserts, so the overfit `- 1` variant is rejected) |
| 20 | `py-logical-op` | `python3` | `or` → `and` |
| 21 | `py-index-exception` | `python3` | `IndexError` inside the module: direct blame + trace-steered class order |
| 22 | `js-off-by-one-loop` | `node sum.test.js` | `<=` over `.length`; `require('./sum')` hop |
| 23 | `js-missing-return` | `node` | `undefined` → missing `return` first |
| 24 | `js-wrong-arith` | `node` | `+` → `*` |
| 25 | `sh-wrong-test-op` | `sh test.sh` | `[ "$1" -lt 0 ]` → `-gt`; `. ./lib.sh` hop |
| 26 | `go-off-by-one` | `go test ./...` | `len(xs)-1`; `_test.go` → package sibling |
| 27 | `rust-missing-return-semicolon` | `rustc --crate-type lib` | tail `;` → `()`; rustc's "remove this semicolon" steers the order |
| 28 | `rust-off-by-one-range` | `rustc --test && ./t` | `0..xs.len() - 1` in the same file as its `#[test]`; the assertion line is never mutated |
| 29 | `c-wrong-comparison` | `gcc … && ./t` | `>` → `>=`; `#include "calc.h"` → `calc.c` |
| 30 | `rb-off-by-one` | `ruby test.rb` | `xs[xs.length]`; `require_relative` hop |
| 31 | `py-wrong-import` | `python3` | `from calcualtor import add` → `calculator` (sibling file, Levenshtein 2); the correct module is pinned too |
| 32 | `js-wrong-import` | `node` | `require('./calcualtor')` → `./calculator` |

Fixtures 15–32 pin their test file: a run where the verifier goes green because
the test was edited is scored **FAIL**. A fixture whose toolchain is not on
`PATH` is reported `[SKIP]` and left out of the denominator; `--require-all`
(or `AXIOM_EVAL_REQUIRE_ALL=1`) makes any skip fail the run, which is what
`.github/workflows/eval.yml` and the proof loop use. Whole suite: ~15 s on a
4-core CPU.

**Limitation:** 32 fixtures, hand-built, each with a known one-edit fix. It is
a capability *gate* ("the loop still works end-to-end, on real toolchains"),
not a *rate* ("AXIOM repairs X% of real bugs"). Several passes land an
*equivalent* fix rather than the authored one (`age > 17` for `age >= 18`,
`i <= xs.length - 1` for `i < xs.length`); that is the verifier deciding, which
is the point. CI enforces the gate on every push (`.github/workflows/eval.yml`).

---

## Growing it into a held-out benchmark

The goal is a rate on bugs AXIOM's authors did not design for.

### Phase 1 — expand the deterministic set (no LLM) — **done 2026-11-05**

`builtin_cases()` now holds 32 fixtures across Python, JavaScript, shell, Go,
Rust, C and Ruby, covering off-by-one loop bounds, wrong comparison operators,
swapped arguments, missing `return`, inverted booleans, `and`/`or`, arithmetic
operators, shell `test` operators and misspelled import paths, each verified
by the real toolchain and each pinning its test file. The repair patterns are
implemented generally in `mutation_repair.rs` rather than as per-fixture flips.
Still open from the original list: a stale constant whose correct value is not
within ±1 (that needs a value the trace or another file supplies, which the
single-edit mutator does not read yet). Each fixture is
an `EvalCase` (single-file or module+test) or `EvalCase::new_agentic`
(multi-file, optionally `.with_holdout(...)`), with `.requires(&[...])` naming
its toolchain and `.unchanged(&[...])` pinning its oracle.

Keep these deterministic and offline — they stay in CI as the regression gate.

### Phase 2 — a real held-out corpus

**Second pass (2026-11-06, unreleased): 39/40 on seed 2026 and 38/40 on a
fresh seed 7, 0 rollback violations.** That is 15/17 on the seed-7 faults
outside the tuning set, against 6/17 for the 0.6.0 binary. New operators for
the four classes that were at 0, call-graph candidate ordering, and three
repair-loop bugs the benchmark exposed:
[`bench/results/SEEDED-HELDOUT-v2-2026-11-06.md`](../bench/results/SEEDED-HELDOUT-v2-2026-11-06.md).

**First held-out measurement (2026-11-06): 19/40 seeded faults repaired, 0
rollback violations.** That is 19 of 29 (66%) in the fault classes the engine
has an operator for, and 0 of 11 in the ones it has none for. The faults are
random single-token faults in `axiom_mesh_core` code that predates the repair
layer, kept only when the crate's own tests catch them. The engine runs with
no LLM. Method, per-class table and caveats:
[`bench/results/SEEDED-HELDOUT-2026-11-06.md`](../bench/results/SEEDED-HELDOUT-2026-11-06.md).
Seeded faults are not developer bugs. The developer-bug corpus below is still
the next step: QuixBugs is importable today (`scripts/import_quixbugs.py`) but
has not been run.


Wire `axiom task` (the goal-directed, LLM-backed path) against an external,
versioned bug set so the fixtures are genuinely out-of-distribution:

- **SWE-bench Verified** subset (start with the 20–50 cheapest instances),
- or **Defects4J** / **QuixBugs** for smaller, language-scoped runs.

Each instance = a repo checkout at the buggy commit + the project's own test
command as the verifier. Score = fraction driven green under the reversible
transaction loop. This run needs an API key and is **not** part of the CI gate;
it is a periodic measurement recorded in `bench/results/` and the proof-loop
table with its date and the exact instance list.

### Fixture format for an on-disk corpus (`bench/repair/<name>/`) — **runner shipped 2026-11-05**

`bench/repair/` and `scripts/bench_repair.sh` exist now (see
[`bench/repair/README.md`](../bench/repair/README.md)); three seed fixtures prove
the runner and CI runs it with `--check` on every push (`eval.yml`). Fixtures
that outgrow `builtin_cases()` — and any external bug set — go here as
directories:

```
bench/repair/<name>/
  meta.json        # { "lang", "verify": ["cmd", "args"...], "kind": "single|agentic", "source": "swe-bench:<id>" }
  broken/          # the project tree that fails `verify`
  expected.diff    # a known-good patch, for scoring "did it find *a* fix" vs "the canonical fix"
```

`scripts/bench_repair.sh` iterates `bench/repair/*`, and for each fixture:
confirms the **No-op** baseline (`verify` must fail on the untouched tree, so a
pass means something), runs `axiom solve` in a scratch copy with no file named
(the loop must localize and repair on its own), re-runs `verify`, and on failure
checks **rollback correctness** (the tree is byte-identical to `broken/`). When
`expected.diff` is present a pass is labelled *authored fix* or *different fix*.
It prints a markdown table, writes `bench/results/repair-<date>.md`, and with
`--check` exits non-zero on any unrepaired, not-actually-broken, or
tree-modifying fixture (`AXIOM_EVAL_REQUIRE_ALL=1` also fails a skip). Driving
it with `axiom task --goal …` against SWE-bench checkouts is the remaining Phase
2 step; the harness it needs is in place.

---

## Baseline protocol

A rate means nothing without a comparison. For any corpus above, run the same
fixtures through:

| Baseline | Command shape |
|---|---|
| **AXIOM** | `axiom solve -- <verify>` (deterministic) or `axiom task --goal "<g>" -- <verify>` (LLM) |
| **Aider** | `aider --yes --message "fix the failing test" <files>` then re-run `<verify>` |
| **Plain model** | one-shot: give the model the failing output + files, apply its diff, re-run `<verify>` |
| **No-op** | run `<verify>` unchanged — confirms every fixture actually fails first |

Report all columns in one table: pass rate, mean attempts, and — AXIOM's
specific claim — **rollback correctness** (after a rejected attempt, is the tree
byte-identical to the start?). Same model backend across LLM rows so the
comparison is about the loop, not the model.

Record each run as `bench/results/repair-<corpus>-<date>.md` and carry the
headline into the proof-loop table.

---

## Why this is the spearhead

- It **works now** — `32/32` on real toolchains, reversible, verifier-gated,
  never editing the test, with cross-session immunity memory.
- It has an **obvious external yardstick** (SWE-bench, Defects4J).
- The competition (SWE-agent, Aider, OpenHands, coding agents generally) is
  legible, so a delta is a sentence people repeat.
- Every other pillar — compression, drift, memory — is *in service of* doing
  this cheaply and safely, so improving them shows up here.
