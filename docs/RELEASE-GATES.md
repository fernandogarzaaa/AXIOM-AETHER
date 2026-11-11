# Release gates

Audit section 7, Phase 5. A release is production-ready only when every
gate below passes. One red gate blocks the release. Missing evidence
blocks the release too. There is no silent pass for a gate that was
never evaluated.

## The aggregate verdict

The release gate is one verdict computed from the gates below. The CI
job `release-gate` in `.github/workflows/ci.yml` enforces it. The local
mirror is `scripts/release_gate.py`. The two share the gate table in
definition, but they are not the same verdict. The CI job evaluates
the CI-provable subset and reports PASS when every evaluated gate
passed. The local evaluator is the full verdict: it reports BLOCKED
when `perf-regression` or `provenance` is unevaluated. A green CI
aggregate does not mean shippable.

- `PASS` means every evaluated gate passed.
- `FAIL` means at least one gate failed.
- `BLOCKED` means at least one gate has no evidence yet. A blocked
  release does not ship until the evidence exists or a waiver is
  recorded. A waiver names the gate, the reason, the owner, and the
  expiry. Waivers live in the release notes, not in the gate table.

## Gate list

| Gate | Proved by (CI check runs) | Notes |
| --- | --- | --- |
| cp1-source | CP/1 normative source | Protocol fixtures and manifest hashes. |
| lossy-context-guard | Lossy-context invariant guard | The six invariant gate tests plus gate calls in request paths. |
| fmt-clippy | Build & test (Rust) | rustfmt and clippy `-D warnings` run as steps of this job. A fmt or clippy failure fails the whole job, which fails this gate. Job granularity is the limit here. Splitting fmt and clippy into their own jobs is a Phase 5 enforcement follow-up. |
| full-tests | Build & test (Rust), Build & test (--features experimental), MSRV (Rust 1.94), Python reference engine (tests), Agentic eval suite | Every crate graph, both feature surfaces, MSRV, the Python reference suite, and the agentic eval suite. |
| session-concurrency-stress | Session concurrency stress | The graded Phase 3 stress harness bounds (p95 latency, RSS delta, registry size per concurrency level). The full `test` job also runs this suite, but the standalone job keeps its measured baseline visible in its own log. |
| pinned-images | Pinned container images | Every deployable image reference carries a sha256 digest pin. |
| helm | Helm chart (lint + auth renders) | Lint plus the authenticated and unauthenticated render tests. |
| docker-build | Docker image build (no push) on every ref; plus Build linux/amd64, Build linux/arm64, Merge into multi-arch manifest when present | The multi-arch checks run only on pushes to main and tags (docker.yml has no pull_request trigger), so they never exist on a PR head. On a PR the gate is proved by the no-push build alone; on main the multi-arch checks must also pass when present. A present red multi-arch check fails the gate. A multi-arch check that concludes skipped or neutral counts as not run, not as a failure. |
| bot-review | PR reviews API, not a check run | At least one review by `coderabbitai[bot]` or `chatgpt-codex-connector[bot]`, or an APPROVED review by a human. Bot logins verified live against PR #289 (both bots) and PR #295 (CodeRabbit) on 2026-11-11. A review that only comments still counts as present. Addressing its findings is a human decision, not part of this gate. |
| perf-regression | None yet | Blocked on Phase 3 and Phase 4. See below. |
| provenance | None yet | Blocked on the Phase 5 enforcement PR. See below. |

Check runs with no mapping (Regenerate proof-loop metrics, the
CodeRabbit app check) are informational and never affect the verdict.
A conditional check that concludes skipped or neutral (a docker
multi-arch check that did not run) counts as not run, not as a
failure. Required checks must be completed with conclusion success;
anything else fails their gate.

## Gates blocked on Phase 3 and Phase 4 results

These gates are defined now and evaluated later. They stay
NOT_EVALUATED until the named work lands. Do not guess their verdicts.

- **perf-regression.** The Phase 3 stress harness and the Phase 4
  paired bench produce stored baselines in `bench/results/`. The gate
  compares a run against the stored baseline and fails on regression
  beyond the documented tolerance. Until the harness, the bench, the
  baselines, and the tolerance exist, this gate is NOT_EVALUATED and
  blocks the release.
- **provenance.** The gate passes only when `release.yml` and
  `docker.yml` run SBOM plus provenance-attestation tooling and
  attach the outputs to the release. That tooling and wiring is the
  Phase 5 enforcement PR. Until it lands, this gate is
  NOT_EVALUATED and blocks the release.

## How the CI job works

The `release-gate` job in `.github/workflows/ci.yml` has three parts.

1. `needs` lists every other job in the workflow. The job runs with
   `if: ${{ !cancelled() }}`, so a failing needed job makes this job
   render an explicit FAIL instead of a silent skip. The script step
   reads the `needs` results and fails on any result that is not
   `success`. A needed job that was skipped (because a job it needed
   failed) is an unevaluated gate, not a pass.
2. A script step queries the GitHub API for the head commit check
   runs and the PR reviews. It fails when a mapped check run from
   another workflow (for example the agentic eval suite, or the
   docker.yml multi-arch publish on main) is red or still running,
   and when no bot or approving human review is present on the PR.
   Two rules keep the verdict honest. First, the required
   cross-workflow check (Agentic eval suite on refs where eval.yml
   runs: every PR, pushes to main and claude/**, and a dispatch on
   one of those refs; plus the docker multi-arch checks on main and
   tag pushes) must be present:
   absence fails the gate instead of passing silently. Second, the
   script waits a bounded time (20 polls, 30 seconds apart) for
   those required checks to finish, because the other workflows run
   on their own schedule and the gate must fail on a real red gate,
   not on timing. After the wait the verdict still fails closed on
   a missing or unfinished required check. On pushes to main there
   is no PR, so the review check reports skipped. The review gate
   only bites on pull requests, which is where reviews happen. A
   review that lands after the gate ran needs a gate rerun: reviews
   have no running state to wait on, so no bounded wait can cover
   them. The query resolves the ref explicitly: on a PR it uses the
   PR head SHA, because `context.sha` is the merge commit there and
   the runs live under the head SHA. It keeps only the newest run
   per check name, so a stale red run that a re-run already fixed
   cannot fail the gate.
3. The verdict is written to the job summary. A green verdict means
   every EVALUATED gate passed. It does not mean shippable: the two
   NOT_EVALUATED gates are named in the summary and still block a
   release until their evidence lands.

## Local dry run

The aggregation logic is demonstrated without pushing anything.
Fetch the real CI state for a commit, then evaluate it. Pass the PR
head SHA, not the merge-commit SHA.

```sh
curl -s "https://api.github.com/repos/fernandogarzaaa/AXIOM-AETHER/commits/<sha>/check-runs?per_page=100" -o /tmp/check-runs.json
curl -s "https://api.github.com/repos/fernandogarzaaa/AXIOM-AETHER/pulls/<n>/reviews?per_page=100" -o /tmp/reviews.json
python3 scripts/release_gate.py --check-runs /tmp/check-runs.json --reviews /tmp/reviews.json --pr <n>
```

Exit 0 means PASS. Exit 1 means FAIL or BLOCKED. The dry-run
transcripts for main head `10ee468` and PR #295 head `60866d2` are in
`docs/release-qualification/gate-dry-run.md`. The aggregation behavior
is also covered by `tests/test_release_gate.py`, which runs in the
Python reference CI job.
