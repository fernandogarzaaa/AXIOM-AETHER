# AXIOM Agentic Core

The Agentic Core is the capstone of AXIOM's autonomy pillar. It turns the
verifier-gated repair loop into a **goal-directed, self-evaluating autonomous
coding** system, and — crucially — reports its capability as a *measured number*
rather than a claim.

> Honest scope: this is not "AGI." It is a measurable, reproducible autonomous
> coding loop. The benchmark below is the number it actually achieves; there is
> no hidden claim beyond it.

## The loop

```text
broken repo ──▶ detect language ──▶ self-select verify command
            ──▶ localize faulty file(s) across languages
            ──▶ propose an edit-set (deterministic Poly-JIT · fleet-shared patch · LLM)
            ──▶ apply atomically, run the verifier
            ──▶ keep ONLY if green, else roll back byte-for-byte and retry
            ──▶ remember what worked (and what failed) · share across the fleet
```

Every stage was built and merged incrementally:

| Capability | Where |
|---|---|
| Fleet patch sharing (signed, re-verified before trust) | `patch_memory.rs` |
| Cross-language fault localization | `fault_locate.rs` |
| Self-selected verify command per language | `fault_locate::default_verify` + CLI |
| Line-targeted, multi-candidate repair | `solve.rs` |
| **Goal-directed loop, atomic multi-file edits, attempt memory** | `agentic.rs` |
| **Capability benchmark (measured score)** | `agentic_eval.rs` |

## Safety invariants (unchanged from `solve`)

1. **Verifier-gated.** No proposed change is ever kept unless *this machine's*
   verify command passes after it. The model never gets to "fix" anything that
   isn't independently confirmed.
2. **Reversible & atomic.** `agentic::Transaction` snapshots every targeted file,
   applies the whole edit-set, and on rejection (or drop/panic) restores all of
   them byte-for-byte; newly created files are removed. A rejected multi-file
   change never half-applies.
3. **Bounded blast radius.** The agent may only write files it was explicitly
   given or that were localized from the verifier's own output and resolve
   *under the project root* (canonicalized containment; `..`/dependency/stdlib
   paths are rejected).
4. **No wasted work.** `agentic::AttemptMemory` remembers rejected edit-sets by
   content hash, so an identical failed change is never retried.

## Commands

```sh
# Goal-directed coding (requires an LLM backend; verifier still gates acceptance)
axiom task --goal "add a --json flag to the CLI" -- cargo test
# Pure repair is the special case (no --goal); files auto-localized if omitted
axiom task -- pytest -q
axiom task --file src/lib.rs -- cargo test

# Measure the autonomous loop's capability (deterministic, no LLM needed)
axiom eval-agentic
```

## Measured capability

`axiom eval-agentic` runs the built-in seeded broken-repo fixtures through the
real `solve` loop (localize → deterministic Poly-JIT repair → verify, all
reversible) and reports the fraction it fixes end-to-end. The fixtures are
deterministic so the score is reproducible in CI with **no model or network**:

```text
[axiom-eval-agentic] autonomous repair capability:
  [PASS] shell-exit-flip
  [PASS] fixture-marker
  [PASS] multi-file-pick-failing
  [PASS] rust-assert-flip
  [PASS] python-frame-localize
  [PASS] js-stack-localize
  [PASS] go-frame-localize
  [PASS] agentic-multi-file-both-broken
score: 8/8 = 100%
```

This is the honest headline: **the deterministic autonomous loop repairs 100% of
the built-in fixtures with no LLM.** The suite spans fault localization across
five trace dialects (shell, Rust `-->`, Python `File "…"`, JS stack, Go
`file.go:line`) and multiple deterministic repair patterns, plus a multi-file
case where the loop must pick the failing file among several. The final case
certifies the strictly harder class of **coordinated multi-file repair**: the
verifier requires two independently-broken files to both go green, so no
single-file fix passes — only the agentic loop's atomic multi-file transaction
(fix both → verify → commit, else roll back) can solve it. The LLM-driven `axiom
task` extends the same verified loop to open-ended objectives; its success rate
depends on the backing model and is bounded by the same verifier gate. Grow the
suite (LLM-required cases) to make the number mean more.
