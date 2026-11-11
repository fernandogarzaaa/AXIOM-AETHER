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
    "Performance regression gate",
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
    assert verdicts["perf-regression"][0] == "PASS"
    assert verdicts["provenance"][0] == "NOT_EVALUATED"
    assert unmapped == ["Regenerate proof-loop metrics"]


def test_unmapped_in_progress_sibling_run_check_does_not_fail_gate():
    # A push-triggered sibling run for the same head schedules the same
    # jobs under a different concurrency group. Its newest check runs
    # share the head SHA, so an unmapped one still in progress must not
    # fail any gate: only explicitly mapped checks affect the verdict.
    checks = [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
    checks.append(
        run(
            "Release gate (aggregate verdict)",
            status="in_progress",
            conclusion=None,
            run_id=100,
        )
    )
    checks.append(
        run("Some new CI job", status="in_progress", conclusion=None, run_id=101)
    )
    verdicts, unmapped = release_gate.evaluate_checks(checks)
    assert verdicts["docker-build"][0] == "PASS"
    assert verdicts["full-tests"][0] == "PASS"
    assert verdicts["fmt-clippy"][0] == "PASS"
    assert "Release gate (aggregate verdict)" in unmapped
    assert "Some new CI job" in unmapped


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
    # BLOCKED (provenance has no evidence yet), not PASS.
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
    assert "1 gate(s) without evidence: provenance" in out
    assert release_gate.main(["--check-runs", str(checks_file)]) == 1


def test_main_pass_verdict_returns_0(monkeypatch, tmp_path, capsys):
    # The PASS path is unreachable under the real gate table while
    # provenance has no evidence, so exercise it under a gate table
    # where every gate has evidence.
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


def without_multi_arch_checks():
    return [
        c
        for c in ALL_CHECKS
        if not c.startswith("Build linux") and c != "Merge into multi-arch manifest"
    ]


@pytest.mark.parametrize(
    "event_name,ref,required",
    [
        ("push", "refs/heads/main", True),
        ("push", "refs/tags/v1.2.3", True),
        ("push", "refs/heads/claude/work", False),
        ("push", "refs/heads/feature-x", False),
        ("workflow_dispatch", "refs/heads/main", True),
        ("workflow_dispatch", "refs/tags/v0.1.0", True),
        ("workflow_dispatch", "refs/heads/work", False),
        ("pull_request", "refs/pull/296/merge", False),
        ("push", "", False),
        (None, None, False),
    ],
)
def test_docker_checks_required_rule(event_name, ref, required):
    assert release_gate.docker_checks_required(event_name, ref) is required


def test_docker_checks_missing_fails_gate_on_main_push():
    checks = without_multi_arch_checks()
    verdicts, _ = release_gate.evaluate_checks(
        [run(n, run_id=i) for i, n in enumerate(checks, start=1)],
        "push",
        "refs/heads/main",
    )
    verdict, detail = verdicts["docker-build"]
    assert verdict == "FAIL"
    assert "Build linux/amd64: missing" in detail


def test_docker_checks_missing_fails_gate_on_tag_push():
    checks = without_multi_arch_checks()
    verdicts, _ = release_gate.evaluate_checks(
        [run(n, run_id=i) for i, n in enumerate(checks, start=1)],
        "push",
        "refs/tags/v1.0.0",
    )
    assert verdicts["docker-build"][0] == "FAIL"


def test_docker_checks_missing_passes_gate_on_claude_push():
    checks = without_multi_arch_checks()
    verdicts, _ = release_gate.evaluate_checks(
        [run(n, run_id=i) for i, n in enumerate(checks, start=1)],
        "push",
        "refs/heads/claude/work",
    )
    verdict, detail = verdicts["docker-build"]
    assert verdict == "PASS"
    assert "not run on this ref" in detail


def test_docker_checks_missing_passes_gate_on_dispatch_working_branch():
    checks = without_multi_arch_checks()
    verdicts, _ = release_gate.evaluate_checks(
        [run(n, run_id=i) for i, n in enumerate(checks, start=1)],
        "workflow_dispatch",
        "refs/heads/work",
    )
    assert verdicts["docker-build"][0] == "PASS"


def test_docker_check_skipped_fails_gate_when_required():
    checks = [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
    for c in checks:
        if c["name"] == "Build linux/arm64":
            c["conclusion"] = "skipped"
    verdicts, _ = release_gate.evaluate_checks(checks, "push", "refs/heads/main")
    verdict, detail = verdicts["docker-build"]
    assert verdict == "FAIL"
    assert "Build linux/arm64: skipped" in detail


def test_main_event_and_ref_flags_drive_docker_rule(tmp_path, capsys):
    checks = without_multi_arch_checks()
    checks_file = tmp_path / "check-runs.json"
    checks_file.write_text(
        json.dumps(
            {"check_runs": [run(n, run_id=i) for i, n in enumerate(checks, start=1)]}
        )
    )
    assert (
        release_gate.main(
            [
                "--check-runs",
                str(checks_file),
                "--event-name",
                "push",
                "--ref",
                "refs/heads/main",
            ]
        )
        == 1
    )
    assert "VERDICT: FAIL" in capsys.readouterr().out
    assert (
        release_gate.main(
            [
                "--check-runs",
                str(checks_file),
                "--event-name",
                "push",
                "--ref",
                "refs/heads/claude/work",
            ]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "VERDICT: BLOCKED" in out


def without_eval_check():
    return [c for c in ALL_CHECKS if c != "Agentic eval suite"]


@pytest.mark.parametrize(
    "event_name,ref,required",
    [
        ("pull_request", "refs/pull/296/merge", True),
        ("push", "refs/heads/main", True),
        ("push", "refs/heads/claude/work", True),
        ("push", "refs/heads/feature-x", False),
        ("push", "refs/tags/v1.0.0", False),
        ("workflow_dispatch", "refs/heads/main", True),
        ("workflow_dispatch", "refs/heads/claude/work", True),
        ("workflow_dispatch", "refs/heads/work", False),
        ("push", "", True),
        (None, None, True),
    ],
)
def test_eval_check_required_rule(event_name, ref, required):
    assert release_gate.eval_check_required(event_name, ref) is required


def test_eval_check_missing_fails_gate_on_pr():
    checks = without_eval_check()
    verdicts, _ = release_gate.evaluate_checks(
        [run(n, run_id=i) for i, n in enumerate(checks, start=1)],
        "pull_request",
        "refs/pull/296/merge",
    )
    verdict, detail = verdicts["full-tests"]
    assert verdict == "FAIL"
    assert "Agentic eval suite: missing" in detail


def test_eval_check_missing_fails_gate_on_main_push():
    checks = without_eval_check()
    verdicts, _ = release_gate.evaluate_checks(
        [run(n, run_id=i) for i, n in enumerate(checks, start=1)],
        "push",
        "refs/heads/main",
    )
    assert verdicts["full-tests"][0] == "FAIL"


def test_eval_check_missing_passes_gate_on_dispatch_working_branch():
    checks = without_eval_check()
    verdicts, _ = release_gate.evaluate_checks(
        [run(n, run_id=i) for i, n in enumerate(checks, start=1)],
        "workflow_dispatch",
        "refs/heads/work",
    )
    verdict, detail = verdicts["full-tests"]
    assert verdict == "PASS"
    assert "not run on this ref" in detail


def test_eval_check_missing_passes_gate_on_feature_push():
    checks = without_eval_check()
    verdicts, _ = release_gate.evaluate_checks(
        [run(n, run_id=i) for i, n in enumerate(checks, start=1)],
        "push",
        "refs/heads/feature-x",
    )
    assert verdicts["full-tests"][0] == "PASS"


def test_eval_check_red_fails_gate_even_when_not_required():
    checks = [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
    for c in checks:
        if c["name"] == "Agentic eval suite":
            c["conclusion"] = "failure"
    verdicts, _ = release_gate.evaluate_checks(
        checks, "workflow_dispatch", "refs/heads/work"
    )
    assert verdicts["full-tests"][0] == "FAIL"


def test_eval_check_skipped_passes_gate_when_not_required():
    checks = [run(n, run_id=i) for i, n in enumerate(ALL_CHECKS, start=1)]
    for c in checks:
        if c["name"] == "Agentic eval suite":
            c["conclusion"] = "skipped"
    verdicts, _ = release_gate.evaluate_checks(
        checks, "workflow_dispatch", "refs/heads/work"
    )
    verdict, detail = verdicts["full-tests"]
    assert verdict == "PASS"
    assert "Agentic eval suite: skipped (informational)" in detail


def test_main_event_and_ref_flags_drive_eval_rule(tmp_path, capsys):
    checks = without_eval_check()
    checks_file = tmp_path / "check-runs.json"
    checks_file.write_text(
        json.dumps(
            {"check_runs": [run(n, run_id=i) for i, n in enumerate(checks, start=1)]}
        )
    )
    assert (
        release_gate.main(
            [
                "--check-runs",
                str(checks_file),
                "--event-name",
                "workflow_dispatch",
                "--ref",
                "refs/heads/work",
            ]
        )
        == 1
    )
    assert "VERDICT: BLOCKED" in capsys.readouterr().out
    assert (
        release_gate.main(
            [
                "--check-runs",
                str(checks_file),
                "--event-name",
                "workflow_dispatch",
                "--ref",
                "refs/heads/main",
            ]
        )
        == 1
    )
    assert "VERDICT: FAIL" in capsys.readouterr().out
