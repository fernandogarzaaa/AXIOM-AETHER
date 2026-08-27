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
temp dirs and drives each through the real
`solve` / `agentic_loop` path. Deterministic, offline, no model required — every
fixture is repairable by the deterministic Poly-JIT layer, so the score is
reproducible in CI.

```bash
cargo run --release --bin axiom -- eval-agentic
# → [PASS] shell-exit-flip … score: 9/9 = 100%   (exit 0; non-zero on any regression)
```

| Fixture | Exercises |
|---|---|
| `shell-exit-flip` | shell exit-code localization + repair |
| `fixture-marker` | marker-based FAIL → PASS repair |
| `multi-file-pick-failing` | pick the failing file out of several |
| `rust-assert-flip` | Rust `--> file:line` localization + `assert_eq!` flip |
| `python-frame-localize` | Python traceback frame → file |
| `js-stack-localize` | JS stack frame → file |
| `go-frame-localize` | Go build error → file |
| `agentic-multi-file-both-broken` | coordinated multi-file edit (no single-file fix passes) |
| `agentic-holdout-split` | fix must pass train **and** held-out verifier |

**Limitation:** 9 fixtures, hand-built, all deterministically repairable. It is
a capability *gate* ("the loop still works end-to-end"), not a *rate* ("AXIOM
repairs X% of real bugs"). CI enforces the gate on every push
(`.github/workflows/eval.yml`).

---

## Growing it into a held-out benchmark

The goal is a rate on bugs AXIOM's authors did not design for.

### Phase 1 — expand the deterministic set (no LLM)

Add fixtures to `builtin_cases()` covering repair patterns the Poly-JIT layer
already claims: off-by-one in loop bounds, wrong comparison operator, swapped
arguments, missing `return`, wrong import path, a stale constant. Target ~30
fixtures across Rust/Python/JS/Go/shell. Each is a `EvalCase` (single-file) or
`EvalCase::new_agentic` (multi-file, optionally `.with_holdout(...)`).

Keep these deterministic and offline — they stay in CI as the regression gate.

### Phase 2 — a real held-out corpus (LLM path)

Wire `axiom task` (the goal-directed, LLM-backed path) against an external,
versioned bug set so the fixtures are genuinely out-of-distribution:

- **SWE-bench Verified** subset (start with the 20–50 cheapest instances),
- or **Defects4J** / **QuixBugs** for smaller, language-scoped runs.

Each instance = a repo checkout at the buggy commit + the project's own test
command as the verifier. Score = fraction driven green under the reversible
transaction loop. This run needs an API key and is **not** part of the CI gate;
it is a periodic measurement recorded in `bench/results/` and the proof-loop
table with its date and the exact instance list.

### Fixture format for an on-disk corpus (`bench/repair/<name>/`)

When Phase 2 fixtures outgrow `builtin_cases()`, store them as directories:

```
bench/repair/<name>/
  meta.json        # { "lang", "verify": ["cmd", "args"...], "kind": "single|agentic", "source": "swe-bench:<id>" }
  broken/          # the project tree that fails `verify`
  expected.diff    # a known-good patch, for scoring "did it find *a* fix" vs "the canonical fix"
```

A thin `scripts/bench_repair.sh` runner (Phase 2 deliverable) iterates
`bench/repair/*`, runs `axiom solve` (or `axiom task --goal …`) in a copy of
`broken/`, checks `verify`, and prints a pass-rate table.

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

- It **works now** — `9/9`, reversible, verifier-gated, with cross-session
  immunity memory.
- It has an **obvious external yardstick** (SWE-bench, Defects4J).
- The competition (SWE-agent, Aider, OpenHands, coding agents generally) is
  legible, so a delta is a sentence people repeat.
- Every other pillar — compression, drift, memory — is *in service of* doing
  this cheaply and safely, so improving them shows up here.
