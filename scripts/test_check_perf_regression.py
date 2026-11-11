"""Unit tests for scripts/check_perf_regression.py.

Run: python3 -m pytest scripts/test_check_perf_regression.py -q
Covers the gate's decisions, not its file I/O: each case builds the
baseline and fresh dicts in memory and calls the check functions
directly, except the missing-file case, which goes through main().
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("check_perf_regression.py")


def load_module():
    spec = importlib.util.spec_from_file_location("check_perf_regression", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load_module()

BASELINE = {
    "tolerances": {
        "wall_ms_factor": 3.0,
        "p95_growth_factor": 2.0,
        "rss_delta_factor": 3.0,
        "p95_growth_max": 20.0,
    },
    "paired": {
        "config": {
            "fingerprint": "deadbeef",
            "seed": 42,
            "repeats": 1,
            "arms": ["A", "D"],
            "families": ["single-file"],
        },
        "arm_summaries": {
            "A": {
                "executed": 3,
                "pending": 0,
                "successes": 0,
                "unsafe_patch_rate": 0.0,
                "rollback_rate": 1.0,
                "total_wall_ms": 300,
            },
            "D": {
                "executed": 3,
                "pending": 0,
                "successes": 3,
                "unsafe_patch_rate": 0.0,
                "rollback_rate": 1.0,
                "total_wall_ms": 2600,
            },
        },
        "guardrails": {"D": True},
    },
    "stress": {
        "levels": {
            "1": {"p95_ms": 120.0, "growth": 0.8, "rss_delta_kib": 2200.0},
            "4": {"p95_ms": 340.0, "growth": 2.3, "rss_delta_kib": 5300.0},
            "16": {"p95_ms": 750.0, "growth": 5.1, "rss_delta_kib": 21500.0},
            "32": {"p95_ms": 1400.0, "growth": 9.5, "rss_delta_kib": 6800.0},
        }
    },
}


def fresh_paired(**overrides):
    import copy

    fresh = {
        "kind": "paired",
        "config": {
            "fingerprint": "deadbeef",
            "seed": 42,
            "repeats": 1,
            "arms": ["A", "D"],
            "families": ["single-file"],
        },
        "arm_summaries": [
            {
                "arm": "A",
                "executed": 3,
                "pending": 0,
                "successes": 0,
                "unsafe_patch_rate": 0.0,
                "rollback_rate": 1.0,
                "total_wall_ms": 280,
            },
            {
                "arm": "D",
                "executed": 3,
                "pending": 0,
                "successes": 3,
                "unsafe_patch_rate": 0.0,
                "rollback_rate": 1.0,
                "total_wall_ms": 2400,
            },
        ],
        "guardrails": {"D": {"pass": True, "reasons": []}},
    }
    for key, value in overrides.items():
        if key == "arm_D":
            fresh["arm_summaries"][1].update(value)
        elif key == "guardrail_D":
            fresh["guardrails"]["D"].update(value)
        else:
            fresh[key] = value
    return copy.deepcopy(fresh)


def stress_log(growth_32=9.0, delta_16=20000.0):
    lines = []
    for level, growth, delta in (
        (1, 0.9, 2300.0),
        (4, 2.0, 5000.0),
        (16, 5.0, delta_16),
        (32, growth_32, 7000.0),
    ):
        lines.append(
            f"[graded-stress] level={level} p95=100.0ms mean=90.0ms "
            f"p95_growth={growth}x rss_before=20000KiB rss_after=22000KiB "
            f"rss_delta={delta:.0f}KiB rss_after_clear=21000KiB registry={level}"
        )
    return "\n".join(lines)


def test_matching_run_passes():
    failures: list[str] = []
    mod.check_paired(fresh_paired(), BASELINE, BASELINE["tolerances"], failures)
    mod.check_stress(
        mod.parse_stress_log(stress_log()), BASELINE, BASELINE["tolerances"], failures
    )
    assert failures == []


def test_success_regression_fails():
    failures: list[str] = []
    mod.check_paired(
        fresh_paired(arm_D={"successes": 2}),
        BASELINE,
        BASELINE["tolerances"],
        failures,
    )
    assert any("successes 2 != baseline 3" in f for f in failures)


def test_wall_time_beyond_tolerance_fails():
    failures: list[str] = []
    mod.check_paired(
        fresh_paired(arm_D={"total_wall_ms": 2600 * 3 + 1}),
        BASELINE,
        BASELINE["tolerances"],
        failures,
    )
    assert any("total_wall_ms" in f for f in failures)


def test_wall_time_within_tolerance_passes():
    failures: list[str] = []
    mod.check_paired(
        fresh_paired(arm_D={"total_wall_ms": 2600 * 2}),
        BASELINE,
        BASELINE["tolerances"],
        failures,
    )
    assert failures == []


def test_failing_guardrail_fails():
    failures: list[str] = []
    mod.check_paired(
        fresh_paired(
            guardrail_D={"pass": False, "reasons": ["unsafe patch rate 1.00"]}
        ),
        BASELINE,
        BASELINE["tolerances"],
        failures,
    )
    assert any("guardrail D: FAIL" in f for f in failures)


def test_fingerprint_mismatch_fails_instead_of_comparing():
    fresh = fresh_paired()
    fresh["config"]["fingerprint"] = "different"
    with pytest.raises(mod.GateFailure, match="fingerprint"):
        mod.check_paired(fresh, BASELINE, BASELINE["tolerances"], [])


def test_stress_growth_beyond_tolerance_fails():
    failures: list[str] = []
    mod.check_stress(
        mod.parse_stress_log(stress_log(growth_32=19.5)),
        BASELINE,
        BASELINE["tolerances"],
        failures,
    )
    assert any("p95_growth 19.5x" in f for f in failures)


def test_stress_rss_spread_does_not_fail():
    # RSS delta is recorded, not gated: two back-to-back sandbox runs
    # measured 3,368 KiB vs 11,708 KiB at level 32 on identical code
    # (allocator retention noise). A 4x RSS spread must not fail the gate.
    failures: list[str] = []
    mod.check_stress(
        mod.parse_stress_log(stress_log(delta_16=21500.0 * 4)),
        BASELINE,
        BASELINE["tolerances"],
        failures,
    )
    assert failures == []


def test_missing_stress_level_fails():
    failures: list[str] = []
    levels = mod.parse_stress_log(stress_log())
    del levels[32]
    mod.check_stress(levels, BASELINE, BASELINE["tolerances"], failures)
    assert any("level 32: missing" in f for f in failures)


def test_main_missing_paired_file_exits_1(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(BASELINE))
    log_path = tmp_path / "stress.log"
    log_path.write_text(stress_log())
    code = mod.main(
        [
            "--paired",
            str(tmp_path / "no-such.json"),
            "--stress-log",
            str(log_path),
            "--baseline",
            str(baseline_path),
        ]
    )
    assert code == 1


def test_load_json_invalid_json_raises(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not json")
    with pytest.raises(mod.GateFailure, match="invalid JSON"):
        mod.load_json(bad_path)


def test_load_json_missing_file_raises(tmp_path):
    with pytest.raises(mod.GateFailure, match="missing file"):
        mod.load_json(tmp_path / "no-such.json")


def test_paired_wrong_kind_raises():
    fresh = fresh_paired()
    fresh["kind"] = "other"
    with pytest.raises(mod.GateFailure, match="expected 'paired'"):
        mod.check_paired(fresh, BASELINE, BASELINE["tolerances"], [])


def test_paired_empty_baseline_arms_raises():
    import copy

    baseline = copy.deepcopy(BASELINE)
    baseline["paired"]["arm_summaries"] = {}
    with pytest.raises(mod.GateFailure, match="no paired arm_summaries"):
        mod.check_paired(fresh_paired(), baseline, BASELINE["tolerances"], [])


def test_paired_executed_mismatch_fails():
    failures: list[str] = []
    mod.check_paired(
        fresh_paired(arm_D={"executed": 2}),
        BASELINE,
        BASELINE["tolerances"],
        failures,
    )
    assert any("executed 2 != baseline 3" in f for f in failures)


def test_paired_unsafe_rate_mismatch_fails():
    failures: list[str] = []
    mod.check_paired(
        fresh_paired(arm_D={"unsafe_patch_rate": 1.0}),
        BASELINE,
        BASELINE["tolerances"],
        failures,
    )
    assert any("unsafe_patch_rate 1.0 != baseline 0.0" in f for f in failures)


def test_paired_missing_wall_fails():
    fresh = fresh_paired()
    del fresh["arm_summaries"][1]["total_wall_ms"]
    failures: list[str] = []
    mod.check_paired(fresh, BASELINE, BASELINE["tolerances"], failures)
    assert any("total_wall_ms missing" in f for f in failures)


def test_paired_wall_exactly_at_limit_passes():
    failures: list[str] = []
    mod.check_paired(
        fresh_paired(arm_D={"total_wall_ms": 2600 * 3}),
        BASELINE,
        mod.DEFAULT_TOLERANCES,
        failures,
    )
    assert failures == []


def test_wall_beyond_default_tolerance_fails():
    # Uses the module defaults, not the baseline's copy: a 3.5x wall
    # must fail under the 3.0 default, so widening the default to 4.0
    # is caught.
    failures: list[str] = []
    mod.check_paired(
        fresh_paired(arm_D={"total_wall_ms": int(2600 * 3.5)}),
        BASELINE,
        mod.DEFAULT_TOLERANCES,
        failures,
    )
    assert any("total_wall_ms" in f for f in failures)


def test_growth_beyond_default_tolerance_fails():
    # Level-32 baseline growth 9.5x: allowed is min(9.5x * 2, 20x) =
    # 19.0x. A fresh 19.5x must fail under the 2.0 default, so widening
    # the default to 3.0 (allowed 20x) is caught.
    failures: list[str] = []
    mod.check_stress(
        mod.parse_stress_log(stress_log(growth_32=19.5)),
        BASELINE,
        mod.DEFAULT_TOLERANCES,
        failures,
    )
    assert any("p95_growth" in f for f in failures)


def test_growth_above_absolute_cap_fails():
    # Baseline level-32 growth 9.5x: allowed is min(9.5x * 2, 20x).
    # A fresh 20.5x must fail under the 20.0 cap, so widening the cap
    # to 21.0 is caught.
    failures: list[str] = []
    mod.check_stress(
        mod.parse_stress_log(stress_log(growth_32=20.5)),
        BASELINE,
        mod.DEFAULT_TOLERANCES,
        failures,
    )
    assert any("absolute max 20" in f for f in failures)


def test_paired_missing_guardrail_fails():
    fresh = fresh_paired()
    fresh["guardrails"] = {}
    failures: list[str] = []
    mod.check_paired(fresh, BASELINE, BASELINE["tolerances"], failures)
    assert any("guardrail D: missing" in f for f in failures)


def test_stress_empty_baseline_levels_raises():
    import copy

    baseline = copy.deepcopy(BASELINE)
    baseline["stress"]["levels"] = {}
    with pytest.raises(mod.GateFailure, match="no stress levels"):
        mod.check_stress({}, baseline, BASELINE["tolerances"], [])


def test_stress_empty_log_raises():
    with pytest.raises(mod.GateFailure, match="no \\[graded-stress\\]"):
        mod.check_stress({}, BASELINE, BASELINE["tolerances"], [])


def test_stress_growth_exactly_at_limit_passes():
    # Level 1 baseline growth 0.8x: allowed is exactly 1.6x. A fresh
    # 1.6x must pass, so turning the strict `>` into `>=` is caught.
    lines = stress_log().splitlines()
    lines[0] = lines[0].replace("p95_growth=0.9x", "p95_growth=1.6x")
    failures: list[str] = []
    mod.check_stress(
        mod.parse_stress_log("\n".join(lines)),
        BASELINE,
        BASELINE["tolerances"],
        failures,
    )
    assert failures == []


def test_paired_arm_missing_fails():
    fresh = fresh_paired()
    fresh["arm_summaries"] = [s for s in fresh["arm_summaries"] if s["arm"] == "A"]
    failures: list[str] = []
    mod.check_paired(fresh, BASELINE, BASELINE["tolerances"], failures)
    assert any("paired arm D: missing from fresh run" in f for f in failures)


def test_paired_guardrail_pass_printed(capsys):
    failures: list[str] = []
    mod.check_paired(fresh_paired(), BASELINE, BASELINE["tolerances"], failures)
    assert failures == []
    assert "PASS paired guardrail D: PASS, matches baseline" in capsys.readouterr().out


def test_main_missing_paired_arg_exits_2():
    with pytest.raises(SystemExit) as exc:
        mod.main(["--stress-log", "stress.log"])
    assert exc.value.code == 2


def test_main_missing_stress_log_arg_exits_2():
    with pytest.raises(SystemExit) as exc:
        mod.main(["--paired", "paired.json"])
    assert exc.value.code == 2


def test_main_missing_required_args_exits_2():
    with pytest.raises(SystemExit) as exc:
        mod.main([])
    assert exc.value.code == 2


def test_main_missing_stress_log_exits_1(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(BASELINE))
    paired_path = tmp_path / "paired.json"
    paired_path.write_text(json.dumps(fresh_paired()))
    code = mod.main(
        [
            "--paired",
            str(paired_path),
            "--stress-log",
            str(tmp_path / "no-such.log"),
            "--baseline",
            str(baseline_path),
        ]
    )
    assert code == 1


def _run_main_on(tmp_path, paired, log_text):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(BASELINE))
    paired_path = tmp_path / "paired.json"
    paired_path.write_text(json.dumps(paired))
    log_path = tmp_path / "stress.log"
    log_path.write_text(log_text)
    return mod.main(
        [
            "--paired",
            str(paired_path),
            "--stress-log",
            str(log_path),
            "--baseline",
            str(baseline_path),
        ]
    )


def test_main_regressed_success_exits_1(tmp_path):
    assert (
        _run_main_on(tmp_path, fresh_paired(arm_D={"successes": 2}), stress_log()) == 1
    )


def test_main_regressed_growth_exits_1(tmp_path):
    assert _run_main_on(tmp_path, fresh_paired(), stress_log(growth_32=19.5)) == 1


def test_main_matching_run_exits_0(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(BASELINE))
    paired_path = tmp_path / "paired.json"
    paired_path.write_text(json.dumps(fresh_paired()))
    log_path = tmp_path / "stress.log"
    log_path.write_text(stress_log())
    code = mod.main(
        [
            "--paired",
            str(paired_path),
            "--stress-log",
            str(log_path),
            "--baseline",
            str(baseline_path),
        ]
    )
    assert code == 0
