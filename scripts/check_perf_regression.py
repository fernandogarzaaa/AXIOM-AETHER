#!/usr/bin/env python3
"""Performance regression gate: compare a fresh run against the stored baseline.

Compares a fresh paired smoke run (``axiombench paired --smoke`` JSON) and
a graded session-concurrency stress log (``cargo test --test
session_concurrency_stress -- --nocapture``) against
``bench/results/perf-baseline.json`` and fails on regression beyond the
documented tolerances.

The gate has a sharp part and a coarse part. The sharp part is exact and
machine-independent: per-arm success counts, unsafe-patch and rollback
rates, pending-cell counts, and the guardrail verdicts recorded in the
paired JSON must match the baseline exactly. The coarse part is wall-clock
time, which is machine-dependent: CI runners are not a perf lab, so wall
times compare against the baseline with a wide factor (3x for paired
totals, 2x for stress p95-growth ratios). A genuine regression
(deadlock-adjacent serialization shows as 50x+ in the stress harness's
own notes) still trips the coarse checks; ordinary machine noise does
not.

Deliberately not compared across runs: stress RSS delta. Two
back-to-back runs on this sandbox measured 3,368 KiB vs 11,708 KiB at
level 32, a 3.5x spread on the same code, because the allocator retains
arenas and the delta measures what it happened to retain, not leakage.
A cross-run RSS comparison would fail the gate on noise. RSS is still
gated, inside the run: the graded test itself asserts each level's RSS
stays within 1.25x plus 16 MiB of the pre-level RSS, and that assertion
failing fails the job before this script runs. The RSS deltas are
parsed and recorded in the baseline for the record, not gated.

Usage (from the repo root, as CI runs it)::

    python3 scripts/check_perf_regression.py \
        --paired bench/results/paired-ci-smoke.json \
        --stress-log perf-stress.log \
        --baseline bench/results/perf-baseline.json

Exit codes: 0 pass, 1 regression or uncomparable input, 2 usage error.
A missing baseline field, a missing log line, or a fingerprint mismatch
fails the run instead of passing silently: an uncomparable gate is a
failed gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Documented tolerances. Also recorded in bench/results/perf-baseline.json;
# the JSON values win when they differ, so a re-baseline carries its own
# tolerances. These defaults apply only when the baseline omits them.
DEFAULT_TOLERANCES = {
    "wall_ms_factor": 3.0,
    "p95_growth_factor": 2.0,
    # Absolute ceiling on stress p95 growth, mirroring the in-test bound
    # MAX_P95_GROWTH in session_concurrency_stress.rs. The comparison
    # never allows more than the test itself allows.
    "p95_growth_max": 20.0,
}

LEVEL_LINE = re.compile(
    r"\[graded-stress\] level=(?P<level>\d+)"
    r" p95=(?P<p95_ms>[\d.]+)ms"
    r" mean=[\d.]+ms"
    r" p95_growth=(?P<growth>[\d.]+)x"
    r"(?: rss_before=(?P<rss_before>\d+)KiB"
    r" rss_after=(?P<rss_after>\d+)KiB"
    r" rss_delta=(?P<rss_delta>-?\d+)KiB)?"
)


class GateFailure(Exception):
    """A single failed check, reported as FAIL and counted toward exit 1."""


def load_json(path: Path) -> dict:
    """Load a JSON file, failing the gate on a missing or invalid file."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise GateFailure(f"missing file: {path}")
    except json.JSONDecodeError as exc:
        raise GateFailure(f"invalid JSON in {path}: {exc}")


def parse_stress_log(text: str) -> dict[int, dict[str, float | None]]:
    """Parse [graded-stress] level lines into {level: {p95_ms, growth, rss_delta}}.

    The RSS fields are optional: on non-Linux the test prints
    `rss=unmeasured (non-Linux)` instead of numeric RSS fields, and the
    level still parses with rss_delta None. RSS is recorded, not gated,
    so an unmeasured RSS changes nothing about the verdict.
    """
    levels: dict[int, dict[str, float | None]] = {}
    for line in text.splitlines():
        match = LEVEL_LINE.search(line)
        if match:
            level = int(match.group("level"))
            rss_delta = match.group("rss_delta")
            levels[level] = {
                "p95_ms": float(match.group("p95_ms")),
                "growth": float(match.group("growth")),
                "rss_delta_kib": float(rss_delta) if rss_delta is not None else None,
            }
    return levels


def check_paired(
    fresh: dict, baseline: dict, tolerances: dict, failures: list[str]
) -> None:
    """Exact success/guardrail checks plus the coarse wall-time check."""
    fresh_cfg = fresh.get("config", {})
    base_cfg = baseline.get("paired", {}).get("config", {})
    for key in ("fingerprint", "seed", "repeats", "arms", "families"):
        if fresh_cfg.get(key) != base_cfg.get(key):
            raise GateFailure(
                f"paired config mismatch on {key}: fresh={fresh_cfg.get(key)!r} "
                f"baseline={base_cfg.get(key)!r}; re-baseline instead of comparing "
                "across different task sets"
            )
    if fresh.get("kind") != "paired":
        raise GateFailure(
            f"paired JSON kind is {fresh.get('kind')!r}, expected 'paired'"
        )

    fresh_arms = {s["arm"]: s for s in fresh.get("arm_summaries", [])}
    base_arms = baseline.get("paired", {}).get("arm_summaries", {})
    if not base_arms:
        raise GateFailure("baseline has no paired arm_summaries; cannot compare")
    # A baseline that omits a configured arm would silently skip that
    # arm's exact-count checks below, so the summary keys must match
    # the configured arms exactly before any comparison runs.
    if set(base_arms) != set(base_cfg.get("arms", [])):
        raise GateFailure(
            f"baseline arm_summaries keys {sorted(base_arms)} do not match "
            f"configured arms {base_cfg.get('arms')}; re-baseline"
        )
    for arm, base in base_arms.items():
        got = fresh_arms.get(arm)
        if got is None:
            failures.append(f"paired arm {arm}: missing from fresh run")
            continue
        for key in ("executed", "pending", "successes"):
            if got.get(key) != base.get(key):
                failures.append(
                    f"paired arm {arm}: {key} {got.get(key)} != baseline {base.get(key)}"
                )
        for key in ("unsafe_patch_rate", "rollback_rate"):
            if got.get(key) != base.get(key):
                failures.append(
                    f"paired arm {arm}: {key} {got.get(key)} != baseline {base.get(key)}"
                )
        wall_factor = tolerances["wall_ms_factor"]
        fresh_wall = got.get("total_wall_ms")
        base_wall = base.get("total_wall_ms")
        if fresh_wall is None or base_wall is None:
            failures.append(f"paired arm {arm}: total_wall_ms missing, cannot compare")
        elif fresh_wall > base_wall * wall_factor:
            failures.append(
                f"paired arm {arm}: total_wall_ms {fresh_wall} > baseline "
                f"{base_wall} x {wall_factor} (={base_wall * wall_factor:.0f})"
            )
        else:
            print(
                f"PASS paired arm {arm}: total_wall_ms {fresh_wall} "
                f"vs baseline {base_wall} (limit {base_wall * wall_factor:.0f})"
            )

    fresh_guardrails = fresh.get("guardrails", {})
    for arm, verdict in baseline.get("paired", {}).get("guardrails", {}).items():
        got = fresh_guardrails.get(arm)
        if got is None:
            failures.append(f"paired guardrail {arm}: missing from fresh run")
        elif got.get("pass") is not True:
            failures.append(
                f"paired guardrail {arm}: FAIL in fresh run "
                f"({'; '.join(got.get('reasons', []))})"
            )
        elif verdict is True:
            print(f"PASS paired guardrail {arm}: PASS, matches baseline")


def check_stress(
    fresh_levels: dict[int, dict[str, float | None]],
    baseline: dict,
    tolerances: dict,
    failures: list[str],
) -> None:
    """Ratio checks per concurrency level against the stored stress baseline."""
    base_levels = baseline.get("stress", {}).get("levels", {})
    if not base_levels:
        raise GateFailure("baseline has no stress levels; cannot compare")
    if not fresh_levels:
        raise GateFailure("no [graded-stress] level lines found in stress log")
    # A harness that adds or drops a level must not silently pass: the
    # level sets must match exactly, or the baseline no longer describes
    # the run and needs a re-baseline.
    base_set = {int(level) for level in base_levels}
    if set(fresh_levels) != base_set:
        raise GateFailure(
            f"stress level sets differ: fresh {sorted(fresh_levels)} vs "
            f"baseline {sorted(base_set)}; re-baseline"
        )
    for level_str, base in base_levels.items():
        level = int(level_str)
        got = fresh_levels[level]
        growth_factor = tolerances["p95_growth_factor"]
        growth_max = tolerances["p95_growth_max"]
        allowed_growth = min(base["growth"] * growth_factor, growth_max)
        if got["growth"] > allowed_growth:
            failures.append(
                f"stress level {level}: p95_growth {got['growth']:.1f}x > "
                f"baseline {base['growth']:.1f}x x {growth_factor} "
                f"(allowed {allowed_growth:.1f}x, absolute max {growth_max:.0f}x)"
            )
        else:
            print(
                f"PASS stress level {level}: p95_growth {got['growth']:.1f}x "
                f"vs baseline {base['growth']:.1f}x (allowed {allowed_growth:.1f}x)"
            )
        # RSS delta is recorded, not gated: two back-to-back runs on the
        # same machine measured a 3.5x spread at level 32 (allocator
        # retention noise), so a cross-run RSS comparison would fail on
        # noise. The in-test per-level RSS bound is the RSS gate.


def main(argv: list[str] | None = None) -> int:
    """Run the gate from the command line; exit 0 pass, 1 fail, 2 usage error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", required=True, help="fresh paired smoke JSON")
    parser.add_argument(
        "--stress-log", required=True, help="graded stress --nocapture log"
    )
    parser.add_argument(
        "--baseline",
        default="bench/results/perf-baseline.json",
        help="stored baseline JSON (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []
    try:
        baseline = load_json(Path(args.baseline))
        fresh = load_json(Path(args.paired))
        try:
            stress_text = Path(args.stress_log).read_text()
        except FileNotFoundError:
            raise GateFailure(f"missing file: {args.stress_log}")
        tolerances = {**DEFAULT_TOLERANCES, **baseline.get("tolerances", {})}
        check_paired(fresh, baseline, tolerances, failures)
        check_stress(parse_stress_log(stress_text), baseline, tolerances, failures)
    except GateFailure as exc:
        print(f"FAIL {exc}")
        return 1

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("perf-regression gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
