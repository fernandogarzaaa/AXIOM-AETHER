# Performance regression gate (Phase 5 item 3)

The `perf-regression` job in `.github/workflows/ci.yml` wires the Phase 3
stress harness and the Phase 4 paired bench into CI as a regression gate.
It runs on every pull request and on pushes to `main` and `claude/**`,
like the other CI jobs.

## What the job runs

1. **Phase 3 stress harness, both tests.** `cargo test --test stress`
   (128 concurrent multi-tenant sessions, `tests/stress.rs`) and
   `cargo test --test session_concurrency_stress -- --nocapture`
   (graded p95 / RSS / registry bounds at concurrency 1, 4, 16, 32).
   The graded test already asserts its own ratio bounds in-test; the
   job additionally saves its `--nocapture` log so the comparison
   script can check the measured numbers against the baseline.
2. **Phase 4 paired smoke with the guardrail enforced.**
   `axiombench paired --smoke --fail-on-guardrail`: arms A/D on the
   single-file family, seed 42, one repeat. A guardrail FAIL (or an arm
   with zero executed cells) exits nonzero and fails the job. The
   previous smoke step in the `test` job ran without the flag, so a
   guardrail failure there would not have failed CI.
3. **Demo-regression negative control.**
   `axiombench paired --demo-regression --fail-on-guardrail` must exit
   exactly 1, and its fresh report must show arm D passing and
   D-regressed failing the reliability guardrail. A nonzero exit alone
   is not enough: exit 2 is a usage error, and a crash before the
   guardrail runs also exits nonzero, so the job requires the exit
   status and the verdict, not just a failure. This is the same
   demonstration recorded in
   `bench/results/PAIRED-SMOKE-2026-11-10.md`.
4. **Baseline comparison.** `scripts/check_perf_regression.py` compares
   the fresh paired JSON and stress log against
   `bench/results/perf-baseline.json` and fails on regression beyond
   the documented tolerances.

## What the comparison checks

The gate has a sharp part and a coarse part.

Sharp part, exact and machine-independent: per-arm `executed`,
`pending`, `successes`, `unsafe_patch_rate`, and `rollback_rate` must
equal the baseline, and every recorded guardrail verdict must be PASS.
Any difference fails the job.

Coarse part, wall-clock time with wide tolerances: CI runners are not
a perf lab, so wall times compare with a factor, not an absolute
bound. Paired per-arm `total_wall_ms` allows 3x the baseline.
Stress per-level p95 growth allows 2x the baseline ratio, capped at
the in-test absolute bound of 20x. Stress RSS delta is recorded but
not compared across runs: two back-to-back runs on this sandbox
measured 3,368 KiB vs 11,708 KiB at level 32 on identical code, a 3.5x
spread from allocator retention, so a cross-run RSS comparison would
fail the gate on noise. RSS is gated inside the run instead, by the
graded test's own per-level bound (1.25x plus 16 MiB over the
pre-level RSS). A genuine regression (deadlock-adjacent
serialization shows as 50x+ in the stress harness's own notes) still
trips the coarse checks; ordinary machine noise does not.

What the gate deliberately does not check: absolute p95 latency
against the baseline. The baseline p95 was measured on the sandbox
(1.5 shared CPUs, in-process load generator sharing the tokio
runtime); a CI runner is a different machine, so an absolute
comparison would measure the machine, not the code. The ratio checks
(p95 growth over the same run's warm baseline, RSS delta over the
same run's pre-level RSS) are the machine-independent form of the
same numbers.

## Baseline

`bench/results/perf-baseline.json` was measured from real runs on
current main (commit `10ee468`) on this sandbox, 2026-11-11. It is the
one JSON exception to the `bench/results/*.json` git-ignore rule (see
`.gitignore`): per-run paired JSONs stay ignored, the baseline is
committed.

Paired smoke (seed 42, repeats 1, arms A/D, single-file family,
fingerprint `33423f7d235668247a0626cd97105d076e57b4a4ba568532425d4434f4fc65f5`):

| arm | executed | successes | success rate | unsafe | rollback | total wall ms |
|-----|----------|-----------|--------------|--------|----------|---------------|
| A | 3 | 0 | 0.0% | 0.00 | 1.00 | 905 |
| D | 3 | 3 | 100.0% | 0.00 | 1.00 | 12396 |

Guardrail D vs baseline A: PASS. Wall times are the median of three
smoke runs (A: 1401/709/905 ms; D: 13290/12396/9500 ms); the per-run
values are in the measurement log linked below.

Graded stress (warm sequential baseline p95 158.3 ms, warm RSS 15932 KiB):

| level | p95 ms | p95 growth | RSS delta KiB |
|-------|--------|------------|---------------|
| 1 | 116.6 | 0.7x | 2200 |
| 4 | 365.5 | 2.3x | 9164 |
| 16 | 633.6 | 4.0x | 19496 |
| 32 | 1476.2 | 9.3x | 3368 |

A second run on the same machine measured baseline p95 133.5 ms and
growth 0.9x/3.3x/5.9x/11.7x, inside the 2x tolerance at every level;
its RSS deltas were 2236/7268/16156/11708 KiB, which is why RSS delta
is recorded but not gated (see above).

What limits these numbers: the load generator is in-process
`oneshot`, so client and server share the tokio runtime and the
sandbox's CPUs; the tiny pipeline's `forward_native` gradient loops
run on the `spawn_blocking` pool, so latency at higher concurrency is
bounded by pool/CPU contention, not by the network. RSS grows from
one-time warm-up (allocator arenas, candle, tokio) plus the live
sessions of the current level, not a leak: per-level deltas stay flat
and the registry returns to 0 after every clear.

## Re-baselining

Re-baseline only when the task set, the harness, or the intended
performance profile changes on purpose (new fixture family, new arm,
deliberate solver speedup). Never re-baseline to make a failing gate
pass: a failing gate means the code regressed or the baseline is
stale for a documented reason, and the reason goes in the PR body.

To re-baseline:

```bash
cargo build --release --locked --features tools   # in axiom_engine_rs/
cargo run --release --features tools --bin axiombench -- paired --smoke \
  --out ../bench/results/paired-baseline-candidate.json
cargo test --release --locked --test session_concurrency_stress -- --nocapture \
  2>&1 | tee /tmp/stress-baseline-candidate.log
```

Then update `bench/results/perf-baseline.json` from the fresh numbers
(success counts and guardrail verdicts exact, wall times median of
three smoke runs), update the table above, and say in the PR body why
the old baseline no longer applies.

## Reproduce locally

```bash
# From the repository root. The cargo commands run in axiom_engine_rs/;
# the checker runs from the root, with the stress log at the root.
cd axiom_engine_rs
cargo build --release --locked --features tools
cargo test --release --locked --test stress
cargo test --release --locked --test session_concurrency_stress -- --nocapture \
  2>&1 | tee ../perf-stress.log
cargo run --release --features tools --bin axiombench -- paired --smoke \
  --fail-on-guardrail --out ../bench/results/paired-ci-smoke.json
cargo run --release --features tools --bin axiombench -- paired \
  --demo-regression --fail-on-guardrail --out ../bench/results/paired-ci-demo.json
# The demo run must exit exactly 1 (guardrail failure, not a usage
# error or a crash), and its report must show D passing and D-regressed
# failing the reliability guardrail; CI checks both, not just the exit.
cd ..
python3 scripts/check_perf_regression.py \
  --paired bench/results/paired-ci-smoke.json \
  --stress-log perf-stress.log
```

## Open items

- The gate runs the smoke matrix only (arms A/D, single-file family).
  The full 40-cell matrix (arms A-E, all five families) is too heavy
  for every PR; it stays a manual release-time run until someone
  measures its CI cost.
- Cost guardrail stays not-applicable: the deterministic harness
  spends $0, so no cost verdict attaches. A live-agent run with
  billed-token accounting is the unblocker (see
  `docs/AXIOMBENCH.md` section 5).
- The wall-time tolerances are coarse by design (3x paired wall,
  2x stress p95 growth). If a
  future change makes the solver materially faster on purpose, the
  baseline tightens at re-baseline time; the gate never auto-tightens.
