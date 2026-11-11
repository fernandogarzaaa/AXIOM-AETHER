#!/usr/bin/env python3
"""release_gate.py: evaluate the AXIOM-AETHER release gate locally.

The gate is an aggregate verdict over the existing CI gates plus the
Phase 3/4 evidence once it exists. The gate table here is the single
source of truth for the local evaluator. docs/RELEASE-GATES.md mirrors
it in prose, and the `release-gate` job in .github/workflows/ci.yml
enforces the same table in CI (via `needs` plus a check-runs query).

Usage:
    python3 scripts/release_gate.py --check-runs FILE [--reviews FILE]
    python3 scripts/release_gate.py --check-runs FILE --reviews FILE --pr 288

Inputs are GitHub API payloads, saved to disk first. Fetch them with
curl and a token:
    check-runs: GET /repos/fernandogarzaaa/AXIOM-AETHER/commits/<sha>/check-runs
    reviews:    GET /repos/fernandogarzaaa/AXIOM-AETHER/pulls/<n>/reviews

Pass the PR head SHA, not the merge-commit SHA: PR-triggered check
runs are listed under the head SHA. Verified live 2026-11-11 against
PR #295 head 60866d2 (13 runs) and main head 10ee468 (15 runs).

Exit codes: 0 release gate PASS, 1 gate FAIL or BLOCKED, 2 usage error.
No network access. No cargo. Read-only on the repo.
"""

import argparse
import json
import sys

# Each gate names the check runs that prove it. A gate passes only when
# every mapped check run is completed with conclusion "success". A check
# run that is still running, or completed with any other conclusion,
# fails its gate. Check runs with no mapping are reported as UNMAPPED
# and do not affect the verdict.
#
# "checks_if_present" names checks that only exist on some refs. The
# docker.yml multi-arch checks run only on pushes to main and on tags,
# so they are absent from every PR head SHA. On a PR the docker-build
# gate is proved by the ci.yml no-push build alone; on main the
# multi-arch checks must also pass when present. A missing conditional
# check never fails its gate, but a present red one does.
GATES = {
    "cp1-source": {
        "description": "CP/1 normative source check",
        "checks": ["CP/1 normative source"],
    },
    "lossy-context-guard": {
        "description": "Lossy-context invariant guard",
        "checks": ["Lossy-context invariant guard"],
    },
    "fmt-clippy": {
        "description": "rustfmt and clippy -D warnings (steps of Build & test)",
        "checks": ["Build & test (Rust)"],
    },
    "full-tests": {
        "description": "Full test suites, all crate graphs and features",
        "checks": [
            "Build & test (Rust)",
            "Build & test (--features experimental)",
            "MSRV (Rust 1.94)",
            "Python reference engine (tests)",
            "Agentic eval suite",
        ],
    },
    "session-concurrency-stress": {
        "description": "Graded session-concurrency stress bounds",
        "checks": ["Session concurrency stress"],
    },
    "pinned-images": {
        "description": "Every deployable image reference digest-pinned",
        "checks": ["Pinned container images"],
    },
    "helm": {
        "description": "Helm chart lint plus authenticated render tests",
        "checks": ["Helm chart (lint + auth renders)"],
    },
    "docker-build": {
        "description": "Docker image builds on every platform, no push",
        "checks": ["Docker image build (no push)"],
        "checks_if_present": [
            "Build linux/amd64",
            "Build linux/arm64",
            "Merge into multi-arch manifest",
        ],
    },
    # Evidence that does not exist yet. These gates are NOT_EVALUATED until
    # the named work lands. A NOT_EVALUATED gate blocks the release. There
    # is no silent pass for missing evidence.
    "perf-regression": {
        "description": "Performance regression vs stored baselines",
        "checks": [],
        "blocked_on": "Phase 3 stress harness and Phase 4 paired bench baselines",
    },
    "provenance": {
        "description": "SBOM plus provenance attestation per release artifact",
        "checks": [],
        "blocked_on": "Phase 5 enforcement PR wiring SBOM/provenance attestation into release.yml and docker.yml",
    },
}

# Bot reviewers whose presence satisfies the review gate. Verified live
# against PR #289 (coderabbitai[bot], chatgpt-codex-connector[bot]) and
# PR #295 (coderabbitai[bot]) on 2026-11-11. A human APPROVED review
# also satisfies it.
REVIEW_BOTS = {"coderabbitai[bot]", "chatgpt-codex-connector[bot]"}


def latest_by_name(check_runs):
    """Keep the newest check run per name (greatest id wins).

    listForRef returns every run for the ref, including superseded
    re-runs. Evaluating all of them would fail the gate on a stale
    red run that a later re-run already fixed.
    """
    by_name = {}
    for run in check_runs:
        name = run["name"]
        if name not in by_name or run.get("id", 0) > by_name[name].get("id", 0):
            by_name[name] = run
    return by_name


def evaluate_checks(check_runs):
    """Map check runs onto gates. Returns (verdicts, unmapped)."""
    by_name = latest_by_name(check_runs)
    verdicts = {}
    for gate, spec in GATES.items():
        if not spec["checks"] and not spec.get("checks_if_present"):
            verdicts[gate] = ("NOT_EVALUATED", spec.get("blocked_on", ""))
            continue
        problems = []
        notes = []
        for name in spec["checks"]:
            if name not in by_name:
                problems.append(f"{name}: missing")
            else:
                status = by_name[name].get("status")
                conclusion = by_name[name].get("conclusion")
                if status != "completed" or conclusion != "success":
                    problems.append(f"{name}: {status}/{conclusion}")
        for name in spec.get("checks_if_present", []):
            if name not in by_name:
                notes.append(f"{name}: not run on this ref")
            else:
                status = by_name[name].get("status")
                conclusion = by_name[name].get("conclusion")
                if status != "completed" or conclusion != "success":
                    problems.append(f"{name}: {status}/{conclusion}")
        if problems:
            verdicts[gate] = ("FAIL", "; ".join(problems))
        else:
            verdicts[gate] = ("PASS", "; ".join(notes))
    mapped = {
        name
        for spec in GATES.values()
        for name in spec["checks"] + spec.get("checks_if_present", [])
    }
    unmapped = sorted(set(by_name) - mapped)
    return verdicts, unmapped


def evaluate_reviews(reviews):
    """Review gate. Returns (verdict, detail)."""
    if reviews is None:
        return "NOT_EVALUATED", "no --reviews file given"
    bots = sorted({r["user"]["login"] for r in reviews} & REVIEW_BOTS)
    approved = sorted(
        {r["user"]["login"] for r in reviews if r.get("state") == "APPROVED"}
    )
    if bots or approved:
        detail = []
        if bots:
            detail.append("bot review by " + ", ".join(bots))
        if approved:
            detail.append("approved by " + ", ".join(approved))
        return "PASS", "; ".join(detail)
    return "FAIL", "no CodeRabbit, Codex, or approving human review found"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the release gate.")
    parser.add_argument(
        "--check-runs", required=True, help="check-runs JSON from the GitHub API"
    )
    parser.add_argument("--reviews", help="PR reviews JSON from the GitHub API")
    parser.add_argument("--pr", default="", help="PR number, shown in the report only")
    args = parser.parse_args(argv)

    try:
        with open(args.check_runs) as f:
            payload = json.load(f)
    except (OSError, ValueError) as e:
        print(f"release-gate: cannot read check-runs file: {e}", file=sys.stderr)
        return 2
    check_runs = payload.get("check_runs", [])
    reviews = None
    if args.reviews:
        try:
            with open(args.reviews) as f:
                reviews = json.load(f)
        except (OSError, ValueError) as e:
            print(f"release-gate: cannot read reviews file: {e}", file=sys.stderr)
            return 2

    verdicts, unmapped = evaluate_checks(check_runs)
    review_verdict, review_detail = evaluate_reviews(reviews)
    verdicts["bot-review"] = (review_verdict, review_detail)

    print(f"Release gate verdict for commit (PR {args.pr or 'n/a'})")
    print(f"Check runs evaluated: {len(check_runs)}")
    for gate in list(GATES) + ["bot-review"]:
        verdict, detail = verdicts[gate]
        desc = GATES.get(gate, {}).get("description", "Bot review present")
        line = f"  [{verdict}] {gate}: {desc}"
        if detail:
            line += f" ({detail})"
        print(line)
    if unmapped:
        print(
            f"Unmapped check runs (informational, do not affect verdict): "
            f"{', '.join(unmapped)}"
        )

    failing = [g for g, (v, _) in verdicts.items() if v == "FAIL"]
    blocked = [g for g, (v, _) in verdicts.items() if v == "NOT_EVALUATED"]
    if failing:
        print(f"VERDICT: FAIL ({len(failing)} failing gate(s): {', '.join(failing)})")
        return 1
    if blocked:
        print(
            f"VERDICT: BLOCKED ({len(blocked)} gate(s) without evidence: "
            f"{', '.join(blocked)})"
        )
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
