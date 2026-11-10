# Paired guardrail verdicts — 2026-11-11 (Phase 4 item 3)

Qualification of the Phase 4 paired arms under the audit section 5
promotion rule: an optimization promotes only on measured task-success
non-inferiority plus an explicit reliability guardrail. The guardrail
check itself is implemented in PR #293
(`axiom_engine_rs/src/bin/axiombench/guardrail.rs`, `evaluate_guardrail`,
wired into the paired runner output and `--fail-on-guardrail`). This
file is the promotion decision on that check's output, not a second
implementation. Nothing here re-runs the paired or unseen suites; the
numbers are quoted from the committed records listed under Sources,
and the one new measurement in this file (the overlap pass split) is
derived mechanically from those committed tables, method stated below.

## Sources

- Paired run: `bench/results/PAIRED-FULL-2026-11-11.md` (PR #293:
  seed 42, 40 cells, arms A-E, 0 pending; rerun reproduced run order
  and all 40 outcomes). The file is on main since #293 merged
  (2026-11-11).
- Seed-424242 baseline: `bench/results/UNSEEN-SEED424242-2026-11-10.md`
  (24-fixture class-balanced subset, 17/24) and
  `bench/results/repair-unseen-seed424242-36-2026-11-10.md` (full
  36-fixture run, 19/36) (PR #294). Both files are not on main yet;
  read them at the #294 head:
  https://github.com/fernandogarzaaa/AXIOM-AETHER/blob/20f1c6037dc3478b60d44818e7beb7f50a5b4b5a/bench/results/UNSEEN-SEED424242-2026-11-10.md
  https://github.com/fernandogarzaaa/AXIOM-AETHER/blob/20f1c6037dc3478b60d44818e7beb7f50a5b4b5a/bench/results/repair-unseen-seed424242-36-2026-11-10.md
- Codeflaws baseline: `bench/results/UNSEEN-CODEFLAWS-2026-11-10.md`
  (7/12 repair-test passes) and
  `bench/results/heldout-codeflaws-2026-11-11.md` (4/7 held-out
  validated) (PR #294), same #294 head as above.
- Prior held-out anchor: QuixBugs 9/40 (22.5%), from the Phase 4 plan.

## Inputs: the paired run

| arm | executed | successes | success rate | unsafe patch rate | rollback rate | guardrail |
|---|---|---|---|---|---|---|
| A baseline | 8 | 0 | 0.0% | 0.00 | 1.00 | PASS (vs itself) |
| B retrieval | 8 | 6 | 75.0% | 0.00 | 1.00 | PASS |
| C CVM | 8 | 6 | 75.0% | 0.00 | 1.00 | PASS |
| D repair | 8 | 6 | 75.0% | 0.00 | 1.00 | PASS |
| E full stack | 8 | 6 | 75.0% | 0.00 | 1.00 | PASS |
| D-regressed (demo) | 3 | 0 | 0.0% | 1.00 | 0.00 | FAIL |

Cost guardrail: not-applicable on every arm. The deterministic harness
spends $0 (see Cost below). The two paired misses are the same for
every executing arm: `multi-both-broken` (needs a coordinated two-file
fix the one-file-at-a-time solver never keeps) and
`longsession-overflow` (candidate cap crowds the faulty line out of the
budget). Per-arm stage wall times are secondaries, not cost: D solved
fastest (27,215 ms) against B 36,617 ms, C 39,100 ms, E 33,944 ms.

## Verdicts

| arm | verdict | basis |
|---|---|---|
| A baseline | N/A (control) | No-op control by design, 0/8. Not a promotion candidate. |
| B retrieval | HOLD | Guardrail PASS, but the 6/8 is replication, not a measured retrieval effect. The B/C/E env overlays have no read site in the `axiom solve` path, and arm B's overlay is byte-identical to arm D's (PAIRED-FULL). The arms execute the same deterministic loop under different labels. Promoting "retrieval" on this evidence would attribute the repair loop's result to retrieval. Promotes only after a real toggle ablation: overlays need read sites in the solve path, then a rerun must show B differing from D. |
| C CVM | HOLD | Same as B. No measured CVM effect: identical 6/8 with the same two misses, same mechanism. |
| D repair | PROMOTE (narrow, see limits) | Measured improvement over baseline, 0/8 to 6/8, with a clean reliability guardrail (unsafe 0.00, rollback 1.00), complete coverage (0 pending), and a rerun that reproduced all 40 outcomes. The repair loop is the mechanism actually exercised here, so this verdict attributes the result to the thing that produced it. |
| E full stack | HOLD | Same as B/C. The "full stack" combination effect is unmeasured: E scores the same 6/8 through the same loop, so no combination gain (or interaction) is evidenced. |
| D-regressed | FAIL (by design) | Negative control. The deliberately regressed arm fails the reliability guardrail (unsafe 1.00, rollback 0.00), which is the proof the guardrail trips when it should. Recorded in PAIRED-SMOKE-2026-11-10. |

Limits on the D promotion, all load-bearing:

- Scope is the deterministic repair loop on the 8-fixture paired set
  only. It is a promotion of the repair capability as measured, not
  of any cost-saving claim: no cost was measured, so no cost claim
  attaches to this promotion (see Cost).
- n=8, one repeat. Repeats 2-3 were not run, so variance is open;
  the rerun reproduced outcomes exactly, but that is determinism,
  not a variance estimate.
- The two known failure modes above bound the claim: coordinated
  multi-file repair and large-file localization under the candidate
  cap are measured gaps, not edge cases.
- Generalization is bounded by the unseen suites below. The paired
  6/8 does not by itself support a held-out repair-rate claim.

## Overlap disclosure: the seed-424242 "genuinely unseen" headline is overstated

PR #294 presents seed 424242 as a genuinely unseen suite. Codex
confirmed on #294, and the Phase 4 coordinator re-derived
mechanically, that 18 of the 36 fixtures share a (file, line, class)
triple with the prior repair-seeded-mesh-core tuning tables (seed 7
and seed 2026). I reproduced that count from the committed tables
and split the repair outcomes by it:

- Tuning sites: 46 distinct (file, line) sites across
  `bench/results/repair-seeded-mesh-core-2026-v2-2026-11-06.md` and
  `bench/results/repair-seeded-mesh-core-seed7-2026-11-06.md`
  (56 distinct (file, line, class) triples on those sites).
  Keys are parsed from fixture names (`<file>-L<line>-<class>-<n>`).
- Unseen 36: 21 fixtures sit at a previously tuned site, 15 at fresh
  sites. By the narrower (file, line, class) triple, 18 overlap and
  18 do not. The 3 fixtures that change key are idc-L78-cast (FAIL),
  idc-L80-shift (FAIL), and mesh-L471-minmax (PASS): same site as a
  tuning fixture, different class. The site key is the stricter and
  correct one for a "genuinely unseen" claim, so the site split is
  the headline and the triple split is reported alongside it.
- Repair split by site: tuned sites 13/21 repaired (61.9%); fresh
  sites 6/15 repaired (40.0%). Headline 19/36 (52.8%) mixes the two.
  By triple: tuned 12/18 (66.7%), fresh 7/18 (38.9%). Both keys agree
  on the fresh-site rate within about 1 point; the counts differ.
- Exact repeat: `mesh-L489-flow-04` is the identical fixture in both
  the seed-2026 tuning table and the unseen table: same name, same
  mutation (api-swap, L489 `break` -> `continue`), both PASS.
  `mesh-L491-bool-20` (unseen) repeats the L491 `false` -> `true`
  mutation tuned as `-09` (seed 2026) and `-19` (seed 7). The overlap
  is therefore not only same-site-same-class: at least one fault is
  the same fault. An earlier draft of this file claimed overlapping
  faults were distinct instances; that claim was wrong and is
  withdrawn here.

What this does to the claim: the 19/36 number cannot be read as a
clean held-out rate. It overstates generalization by about 13
percentage points against the fresh-site rate (52.8% vs 40.0%). The
honest held-out number from this suite is 6/15 on fresh sites. The
"genuinely unseen" label applies to 15 fixtures by site (18 by
triple), not 36.

What it does not do: it does not overturn the D promotion above,
which rests on the paired set, a different fixture set. It does not
imply the overlapping repairs are all invalid: most overlapping
fixtures are same-site-same-class faults rather than exact repeats
(only the L489 case is a full-name repeat). It does mean the suite
cannot carry a generalization claim at face value.

Remediation (not done here, named as the follow-up): regenerate the
seeded suite at sites with no tuning history (file, line), not just
no matching triple, drop the repeated fixtures, re-baseline, and
report the fresh-site rate as the headline. Until then, quote 6/15
fresh alongside 19/36, never 19/36 alone.

The Codeflaws number is the cleaner generalization signal and is
unaffected by this overlap: different language (C), different oracle
shape, no repair operator tuned on it. Repair-test passes 7/12
(58.3%), 0 rollback violations, 0 edits to a test; held-out validated
4/7. Caveat: fixture 58-A is reference-inconsistent (its reference
fix fails 9 of 39 held-out inputs), so that fixture's held-out oracle
disagrees with its own reference; the 4/7 counts it as not validated.

## Cost guardrail: not-applicable, and what would unblock it

Status: N/A on every arm, by measurement rather than by assumption.
The deterministic harness spends $0: `cost_usd` is hardcoded `0.0`
at every construction site in the paired runner
(`axiom_engine_rs/src/bin/axiombench/paired.rs`, lines 479, 1084,
1982, 2024 in PR #293), and `evaluate_guardrail` records the cost
rule as not-applicable instead of passing on invented numbers
(`guardrail.rs`). No arm promotes on cost, and no cost saving is
claimed anywhere in these verdicts.

What would unblock a real cost verdict: a live-agent run with
billed-token accounting, meaning per-cell model plus billed tokens
(token counts from the provider, priced at the provider rate),
recorded into `cost_usd` instead of the hardcoded 0.0. That needs the
agent executor (a Phase 4 prep open item: a real coding agent with
and without AXIOM in the loop) and provider credentials, neither of
which exists in CI today. Wall-clock time is not a cost substitute
and is not presented as one here; the cost pillar (`cost.rs`)
measures corpus byte reduction through the live proxy, not LLM
dollars, and is not a substitute either.

## Open items (not verified or not yet wired)

- Repeats 2-3 of the full paired run were not run; variance across
  repeats is open. The rerun reproduced outcomes exactly, which
  verifies determinism, not variance.
- The demo-regression FAIL is not wired into CI. PR #293's CI runs
  only `paired --smoke` (`.github/workflows/ci.yml`), without
  `--fail-on-guardrail` and without a `--demo-regression` step, so a
  guardrail failure in the smoke run would not fail CI today. The
  live gate today is the unit test
  `regressed_arm_fails_the_guardrail_it_is_built_to_trip`
  (`paired.rs`, in `#[cfg(test)]`), which runs under `cargo test`.
  Wiring the demo into CI is unblocked now that #293 has merged: the
  `paired` subcommand exists on main, so add `--fail-on-guardrail`
  to the smoke step and a step that runs `paired --demo-regression
  --fail-on-guardrail` and requires a nonzero exit.
- The paired and unseen numbers above are crew-measured and recorded
  in #293/#294; the coordinator verified the PR heads, diffs, and CI
  live. This file re-derives only the overlap count and pass split
  from the committed tables. No run was re-executed for this verdict.
- Codeflaws fixture 58-A needs an oracle decision (its reference
  fix fails held-out inputs); held out of the 4/7 as not validated
  until that is resolved.
