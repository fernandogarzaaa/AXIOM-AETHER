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
# gate is proved by the ci.yml no-push build alone. On a ref that
# schedules docker.yml (pushes to main and v* tags, and a dispatch on
# one of those refs) the multi-arch checks are required: a missing one
# fails the gate, matching the CI job. On any other ref a missing
# conditional check never fails its gate, and a present one that
# concludes skipped or neutral counts as not run. A present red one
# fails its gate. The same conditional rule applies to the Agentic
# eval suite inside full-tests: eval.yml runs on every PR and on
# pushes to main and claude/**, so the check is required only on
# those events and refs, never on a standalone dispatch elsewhere.
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
    # Evidence that does not exist yet. This gate is NOT_EVALUATED until
    # the named work lands. A NOT_EVALUATED gate blocks the release. There
    # is no silent pass for missing evidence. The perf-regression gate
    # left this block when PR #299 merged: it is now proved by the
    # "Performance regression gate" check run.
    "perf-regression": {
        "description": "Performance regression vs stored baselines",
        "checks": ["Performance regression gate"],
    },
    "provenance": {
        "description": "SBOM plus provenance attestation per release artifact",
        "checks": [],
        "blocked_on": "Phase 5 enforcement PR landed (#297); no CI check run proves it per push, evidence is the release attestation verified at release time",
    },
}

# Bot reviewers whose presence satisfies the review gate. Verified live
# against PR #289 (coderabbitai[bot], chatgpt-codex-connector[bot]) and
# PR #295 (coderabbitai[bot]) on 2026-11-11. A human APPROVED review
# also satisfies it.
REVIEW_BOTS = {"coderabbitai[bot]", "chatgpt-codex-connector[bot]"}

# Conclusions that mean "did not run" rather than "failed". A
# conditional check that concludes skipped or neutral counts as not
# run, matching the CI job. Required checks stay strict: anything
# other than completed/success fails their gate.
BENIGN_CONCLUSIONS = {"skipped", "neutral"}


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


def docker_checks_required(event_name, ref):
    """Whether docker.yml schedules its checks for this event/ref.

    Mirrors the release-gate job in .github/workflows/ci.yml.
    docker.yml runs on pushes to main and v* tags. A push to
    claude/**, a pull request, or a dispatch on a working branch
    never schedules it, so its checks must not be required there.
    A dispatch counts only when its ref is main or a v* tag. With
    no ref given the checks stay optional, preserving the old
    local-only behavior.
    """
    if not ref:
        return False
    on_docker_ref = ref == "refs/heads/main" or ref.startswith("refs/tags/v")
    return on_docker_ref and event_name != "pull_request"


def eval_check_required(event_name, ref):
    """Whether eval.yml schedules the Agentic eval suite for this event/ref.

    Mirrors the release-gate job in .github/workflows/ci.yml.
    eval.yml runs on every pull request and on pushes to main and
    claude/**. A standalone CI dispatch on any other branch never
    schedules it, so its check must not be required there. A dispatch
    counts only when its ref is main or claude/**. With no ref given
    the check stays required, preserving the old local-only behavior.
    """
    if event_name == "pull_request":
        return True
    if not ref:
        return True
    return ref == "refs/heads/main" or ref.startswith("refs/heads/claude/")


def evaluate_checks(check_runs, event_name=None, ref=None):
    """Map check runs onto gates. Returns (verdicts, unmapped)."""
    by_name = latest_by_name(check_runs)
    verdicts = {}
    eval_required = eval_check_required(event_name, ref)
    for gate, spec in GATES.items():
        if not spec["checks"] and not spec.get("checks_if_present"):
            verdicts[gate] = ("NOT_EVALUATED", spec.get("blocked_on", ""))
            continue
        problems = []
        notes = []
        for name in spec["checks"]:
            # The Agentic eval suite is conditionally required: it is
            # scheduled only on refs where eval.yml runs. Elsewhere a
            # missing one never fails its gate, and a benign conclusion
            # is informational. A present red one still fails its gate,
            # matching the CI job.
            conditional_eval = name == "Agentic eval suite" and not eval_required
            if name not in by_name:
                if conditional_eval:
                    notes.append(f"{name}: not run on this ref")
                else:
                    problems.append(f"{name}: missing")
            else:
                status = by_name[name].get("status")
                conclusion = by_name[name].get("conclusion")
                if status != "completed" or conclusion != "success":
                    if conditional_eval and conclusion in BENIGN_CONCLUSIONS:
                        notes.append(f"{name}: {conclusion} (informational)")
                    else:
                        problems.append(f"{name}: {status}/{conclusion}")
        # Conditional checks are required only on refs that schedule
        # their workflow (docker.yml: pushes to main and v* tags, and
        # a dispatch on one of those refs). Elsewhere a missing one
        # never fails its gate, and a benign conclusion is
        # informational, not a failure. A required check that
        # concludes skipped or neutral fails: it did not succeed.
        conditional_required = gate == "docker-build" and docker_checks_required(
            event_name, ref
        )
        for name in spec.get("checks_if_present", []):
            if name not in by_name:
                if conditional_required:
                    problems.append(
                        f"{name}: missing (required cross-workflow check "
                        f"never ran on this ref)"
                    )
                else:
                    notes.append(f"{name}: not run on this ref")
            else:
                status = by_name[name].get("status")
                conclusion = by_name[name].get("conclusion")
                if conclusion in BENIGN_CONCLUSIONS:
                    if conditional_required:
                        problems.append(
                            f"{name}: {conclusion} (required cross-workflow "
                            f"check did not succeed)"
                        )
                    else:
                        notes.append(f"{name}: {conclusion} (informational)")
                elif status != "completed" or conclusion != "success":
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
    """Parse arguments, evaluate the gate, print the verdict.

    Returns 0 on PASS, 1 on FAIL or BLOCKED, 2 on a usage error.
    """
    parser = argparse.ArgumentParser(description="Evaluate the release gate.")
    parser.add_argument(
        "--check-runs", required=True, help="check-runs JSON from the GitHub API"
    )
    parser.add_argument("--reviews", help="PR reviews JSON from the GitHub API")
    parser.add_argument("--pr", default="", help="PR number, shown in the report only")
    parser.add_argument(
        "--event-name",
        default="push",
        help="GitHub event name: push, pull_request, workflow_dispatch",
    )
    parser.add_argument(
        "--ref",
        default="",
        help="full git ref, e.g. refs/heads/main; drives the docker rule",
    )
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

    verdicts, unmapped = evaluate_checks(check_runs, args.event_name, args.ref)
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
