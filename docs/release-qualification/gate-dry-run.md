# Gate aggregation dry run

Run 2026-11-11 against main head `10ee468` and PR #295 head
`60866d2`, before any push. No CI was triggered. The check-runs
payloads came from the live GitHub API
(`GET /repos/fernandogarzaaa/AXIOM-AETHER/commits/<sha>/check-runs`).
The reviews payloads came from PR #295
(`GET /repos/fernandogarzaaa/AXIOM-AETHER/pulls/295/reviews`).

## Run 1: real CI state, main head

Command:

```sh
python3 scripts/release_gate.py --check-runs /tmp/check-runs-10ee468.json --reviews /tmp/reviews-295.json --pr 295
```

Output:

```text
Release gate verdict for commit (PR 295)
Check runs evaluated: 15
  [PASS] cp1-source: CP/1 normative source check
  [PASS] lossy-context-guard: Lossy-context invariant guard
  [PASS] fmt-clippy: rustfmt and clippy -D warnings (steps of Build & test)
  [PASS] full-tests: Full test suites, all crate graphs and features
  [PASS] session-concurrency-stress: Graded session-concurrency stress bounds
  [PASS] pinned-images: Every deployable image reference digest-pinned
  [PASS] helm: Helm chart lint plus authenticated render tests
  [PASS] docker-build: Docker image builds on every platform, no push
  [NOT_EVALUATED] perf-regression: Performance regression vs stored baselines (Phase 3 stress harness and Phase 4 paired bench baselines)
  [NOT_EVALUATED] provenance: SBOM plus provenance attestation per release artifact (Phase 5 enforcement PR wiring SBOM/provenance attestation into release.yml and docker.yml)
  [PASS] bot-review: Bot review present (bot review by coderabbitai[bot])
Unmapped check runs (informational, do not affect verdict): Regenerate proof-loop metrics
VERDICT: BLOCKED (2 gate(s) without evidence: perf-regression, provenance)
```

Exit code 1. The verdict is BLOCKED, not PASS, because the two
gates with no evidence yet block the release by design. All eight
evaluated CI gates passed on the real `10ee468` state, and the review
gate passed on the real PR #295 reviews.

## Run 2: real CI state, PR head

Command:

```sh
python3 scripts/release_gate.py --check-runs /tmp/check-runs-60866d2.json --reviews /tmp/reviews-295.json --pr 295
```

Output:

```text
Release gate verdict for commit (PR 295)
Check runs evaluated: 13
  [PASS] cp1-source: CP/1 normative source check
  [PASS] lossy-context-guard: Lossy-context invariant guard
  [PASS] fmt-clippy: rustfmt and clippy -D warnings (steps of Build & test)
  [PASS] full-tests: Full test suites, all crate graphs and features
  [PASS] session-concurrency-stress: Graded session-concurrency stress bounds
  [PASS] pinned-images: Every deployable image reference digest-pinned
  [PASS] helm: Helm chart lint plus authenticated render tests
  [PASS] docker-build: Docker image builds on every platform, no push (Build linux/amd64: not run on this ref; Build linux/arm64: not run on this ref; Merge into multi-arch manifest: not run on this ref)
  [NOT_EVALUATED] perf-regression: Performance regression vs stored baselines (Phase 3 stress harness and Phase 4 paired bench baselines)
  [NOT_EVALUATED] provenance: SBOM plus provenance attestation per release artifact (Phase 5 enforcement PR wiring SBOM/provenance attestation into release.yml and docker.yml)
  [PASS] bot-review: Bot review present (bot review by coderabbitai[bot])
Unmapped check runs (informational, do not affect verdict): CodeRabbit, Regenerate proof-loop metrics
VERDICT: BLOCKED (2 gate(s) without evidence: perf-regression, provenance)
```

Exit code 1. Same BLOCKED verdict as Run 1. This run matters
because the prep draft of the script FAILED here: docker.yml runs
only on pushes to main and tags, so its three multi-arch checks do
not exist on a PR head, and the draft required them. The fixed
script treats them as conditional (present and red fails, absent is
noted and does not fail), so the docker-build gate passes on the
PR head via the ci.yml no-push build alone. The CodeRabbit app check
(conclusion skipped) is informational, not a failure.

## Run 3: mutated CI state (red-flip proof)

The Run 1 payload with one check run flipped to failure
(`Build & test (Rust)` to `completed/failure`). Output, abridged to
the changed lines:

```text
  [FAIL] fmt-clippy: rustfmt and clippy -D warnings (steps of Build & test) (Build & test (Rust): completed/failure)
  [FAIL] full-tests: Full test suites, all crate graphs and features (Build & test (Rust): completed/failure)
VERDICT: FAIL (2 failing gate(s): fmt-clippy, full-tests)
```

Exit code 1. A single red check flips every gate it proves, and the
aggregate goes FAIL. The aggregation logic is demonstrated. The CI
job itself runs for the first time on the PR for this work; its
red-flip on a live PR was not demonstrated, because that would need
a deliberately broken gate on a scratch branch and a full CI run.
The local mirror plus `tests/test_release_gate.py` cover the flip
behavior instead.
