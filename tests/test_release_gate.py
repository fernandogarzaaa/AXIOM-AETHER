"""Release-gate aggregation behavior (scripts/release_gate.py).

The gate is load-bearing for releases, so its verdicts are tested
directly: a red check flips every gate it proves, a missing
conditional check never fails its gate, and a stale red run that a
later re-run fixed does not fail the gate.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "release_gate", REPO_ROOT / "scripts" / "release_gate.py"
)
release_gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release_gate)


def run(name, status="completed", conclusion="success", run_id=1):
    return {"id": run_id, "name": name, "status": status, "conclusion": conclusion}


ALL_CHECKS = [
    "CP/1 normative source",
    "Lossy-context invariant guard",
    "Build & test (Rust)",
    "Build & test (--features experimental)",
    "MSRV (Rust 1.94)",
    "Python reference engine (tests)",
    "Agentic eval suite",
    "Session concurrency stress",
    "Pinned container images",
    "Helm chart (lint + auth renders)",
    "Docker image build (no push)",
    "Build linux/amd64",
    "Build linux/arm64",
    "Merge into multi-arch manifest",
    "Regenerate proof-loop metrics",
]


def test_main_head_all_green_is_blocked_only_on_missing_evidence():
    verdicts, unmapped = release_gate.evaluate_checks(
        [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
    )
    assert verdicts["docker-build"][0] == "PASS"
    assert verdicts["full-tests"][0] == "PASS"
    assert verdicts["session-concurrency-stress"][0] == "PASS"
    assert verdicts["perf-regression"][0] == "NOT_EVALUATED"
    assert verdicts["provenance"][0] == "NOT_EVALUATED"
    assert unmapped == ["Regenerate proof-loop metrics"]


def test_pr_head_without_multi_arch_checks_still_passes_docker_build():
    pr_checks = [
        c
        for c in ALL_CHECKS
        if not c.startswith("Build linux") and c != "Merge into multi-arch manifest"
    ]
    verdicts, _ = release_gate.evaluate_checks(
        [run(n, run_id=i) for i, n in enumerate(pr_checks, start=1)]
    )
    verdict, detail = verdicts["docker-build"]
    assert verdict == "PASS"
    assert "not run on this ref" in detail


def test_red_build_flips_every_gate_it_proves():
    checks = [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
    for c in checks:
        if c["name"] == "Build & test (Rust)":
            c["conclusion"] = "failure"
    verdicts, _ = release_gate.evaluate_checks(checks)
    assert verdicts["fmt-clippy"][0] == "FAIL"
    assert verdicts["full-tests"][0] == "FAIL"
    assert verdicts["cp1-source"][0] == "PASS"


def test_red_multi_arch_check_fails_docker_build_when_present():
    checks = [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
    for c in checks:
        if c["name"] == "Build linux/arm64":
            c["conclusion"] = "failure"
    verdicts, _ = release_gate.evaluate_checks(checks)
    assert verdicts["docker-build"][0] == "FAIL"


def test_conditional_check_skipped_or_neutral_is_informational():
    for conclusion in ("skipped", "neutral"):
        checks = [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
        for c in checks:
            if c["name"] == "Build linux/arm64":
                c["conclusion"] = conclusion
        verdict, detail = release_gate.evaluate_checks(checks)[0]["docker-build"]
        assert verdict == "PASS"
        assert f"Build linux/arm64: {conclusion} (informational)" in detail


def test_required_check_skipped_fails_its_gate():
    checks = [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
    for c in checks:
        if c["name"] == "Build & test (Rust)":
            c["conclusion"] = "skipped"
    verdicts, _ = release_gate.evaluate_checks(checks)
    assert verdicts["fmt-clippy"][0] == "FAIL"
    assert verdicts["full-tests"][0] == "FAIL"


def test_latest_by_name_equal_ids_keeps_first():
    checks = [
        run("Build & test (Rust)", conclusion="failure", run_id=5),
        run("Build & test (Rust)", conclusion="success", run_id=5),
    ]
    verdicts, _ = release_gate.evaluate_checks(checks)
    assert verdicts["fmt-clippy"][0] == "FAIL"


def test_latest_by_name_missing_id_defaults_to_zero():
    checks = [
        {"name": "Build & test (Rust)", "status": "completed", "conclusion": "failure"},
        run("Build & test (Rust)", conclusion="success", run_id=1),
    ]
    verdicts, _ = release_gate.evaluate_checks(checks)
    assert verdicts["fmt-clippy"][0] == "PASS"


def test_latest_by_name_incoming_missing_id_defaults_to_zero():
    checks = [
        run("Build & test (Rust)", conclusion="failure", run_id=0),
        {"name": "Build & test (Rust)", "status": "completed", "conclusion": "success"},
    ]
    verdicts, _ = release_gate.evaluate_checks(checks)
    assert verdicts["fmt-clippy"][0] == "FAIL"
    checks = [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
    checks.append(run("Build & test (Rust)", conclusion="failure", run_id=100))
    checks.append(run("Build & test (Rust)", conclusion="success", run_id=101))
    verdicts, _ = release_gate.evaluate_checks(checks)
    assert verdicts["fmt-clippy"][0] == "PASS"
    assert verdicts["full-tests"][0] == "PASS"


def test_missing_required_check_fails_its_gate():
    checks = [
        run(n, run_id=i)
        for i, n in enumerate(ALL_CHECKS, start=1)
        if n != "CP/1 normative source"
    ]
    verdicts, _ = release_gate.evaluate_checks(checks)
    assert verdicts["cp1-source"][0] == "FAIL"


def test_review_gate_bot_approved_and_empty():
    bots = [{"user": {"login": "coderabbitai[bot]"}, "state": "COMMENTED"}]
    verdict, detail = release_gate.evaluate_reviews(bots)
    assert verdict == "PASS"
    assert detail == "bot review by coderabbitai[bot]"
    codex = [{"user": {"login": "chatgpt-codex-connector[bot]"}, "state": "COMMENTED"}]
    verdict, detail = release_gate.evaluate_reviews(codex)
    assert verdict == "PASS"
    assert detail == "bot review by chatgpt-codex-connector[bot]"
    human = [{"user": {"login": "fernandogarzaaa"}, "state": "APPROVED"}]
    verdict, detail = release_gate.evaluate_reviews(human)
    assert verdict == "PASS"
    assert detail == "approved by fernandogarzaaa"
    verdict, detail = release_gate.evaluate_reviews(bots + human)
    assert verdict == "PASS"
    assert detail == ("bot review by coderabbitai[bot]; approved by fernandogarzaaa")
    assert release_gate.evaluate_reviews([])[0] == "FAIL"
    assert release_gate.evaluate_reviews(None)[0] == "NOT_EVALUATED"


def test_main_exit_codes(tmp_path, capsys):
    payload = {
        "check_runs": [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
    }
    checks_file = tmp_path / "check-runs.json"
    checks_file.write_text(json.dumps(payload))
    reviews_file = tmp_path / "reviews.json"
    reviews_file.write_text(
        json.dumps([{"user": {"login": "coderabbitai[bot]"}, "state": "COMMENTED"}])
    )
    # BLOCKED (perf/provenance have no evidence yet), not PASS.
    assert (
        release_gate.main(
            [
                "--check-runs",
                str(checks_file),
                "--reviews",
                str(reviews_file),
                "--pr",
                "1",
            ]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "VERDICT: BLOCKED" in out
    assert "2 gate(s) without evidence: perf-regression, provenance" in out
    assert release_gate.main(["--check-runs", str(checks_file)]) == 1


def test_main_pass_verdict_returns_0(monkeypatch, tmp_path, capsys):
    # The PASS path is unreachable under the real gate table while
    # perf-regression and provenance have no evidence, so exercise it
    # under a gate table where every gate has evidence.
    monkeypatch.setattr(
        release_gate,
        "GATES",
        {
            "cp1-source": {
                "description": "CP/1 normative source check",
                "checks": ["CP/1 normative source"],
            },
        },
    )
    checks_file = tmp_path / "check-runs.json"
    checks_file.write_text(
        json.dumps({"check_runs": [run("CP/1 normative source", run_id=1)]})
    )
    reviews_file = tmp_path / "reviews.json"
    reviews_file.write_text(
        json.dumps([{"user": {"login": "coderabbitai[bot]"}, "state": "COMMENTED"}])
    )
    assert (
        release_gate.main(
            [
                "--check-runs",
                str(checks_file),
                "--reviews",
                str(reviews_file),
                "--pr",
                "1",
            ]
        )
        == 0
    )
    assert "VERDICT: PASS" in capsys.readouterr().out


def test_main_fail_verdict_returns_1(tmp_path, capsys):
    checks = [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
    for c in checks:
        if c["name"] == "Build & test (Rust)":
            c["conclusion"] = "failure"
    checks_file = tmp_path / "check-runs.json"
    checks_file.write_text(json.dumps({"check_runs": checks}))
    reviews_file = tmp_path / "reviews.json"
    reviews_file.write_text(
        json.dumps([{"user": {"login": "coderabbitai[bot]"}, "state": "COMMENTED"}])
    )
    assert (
        release_gate.main(
            [
                "--check-runs",
                str(checks_file),
                "--reviews",
                str(reviews_file),
                "--pr",
                "1",
            ]
        )
        == 1
    )
    assert "VERDICT: FAIL" in capsys.readouterr().out


def test_main_unreadable_reviews_returns_2(tmp_path):
    checks_file = tmp_path / "check-runs.json"
    checks_file.write_text(
        json.dumps(
            {
                "check_runs": [
                    run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)
                ]
            }
        )
    )
    assert (
        release_gate.main(
            [
                "--check-runs",
                str(checks_file),
                "--reviews",
                str(tmp_path / "missing.json"),
            ]
        )
        == 2
    )
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert (
        release_gate.main(["--check-runs", str(checks_file), "--reviews", str(corrupt)])
        == 2
    )


def test_main_requires_check_runs_arg():
    with pytest.raises(SystemExit) as exc:
        release_gate.main([])
    assert exc.value.code == 2


def test_main_usage_error_on_unreadable_payload(tmp_path):
    assert release_gate.main(["--check-runs", str(tmp_path / "missing.json")]) == 2
