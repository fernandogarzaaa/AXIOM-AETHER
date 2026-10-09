# Benchmark v2 Checkpoint

**Current stage:** 5 (single test run)
**Branch:** retrieval-benchmark
**Ranker pinned:** 7c373819b24b783ea457fa547a9c13762c8174a5 (frozen, no changes)
**Freeze commit:** 4699776

## Done
- Stage 0: pilot closeout committed (0cfa28b). Headline: no detectable difference (pilot).
- Stage 1: spec v2 committed (0fa994a). H1 primary on variants B and C, Holm-corrected.
- Stage 2: dataset mining complete (2111574). 54 verified test tasks from 6 repos. Gate PASSED.
- Stage 3: harness v2-ready (b99f18b). --manifest flag, 6 new repos, leakage gate 0 violations.
- Stage 4: FREEZE committed (4699776). Ranker verified, tree clean, no amendments after.

## Stage 5 tasks (single test run)
- [x] Run attempt 1: started, infrastructure crash (service restart) at task 39/54 (urllib3-test-04). Partial output 1287/1782 lines discarded. Documented per one-run rule.
- [x] Run attempt 2: rerun with identical code and inputs after infra crash. Exit 0, 1782 lines (54 x 3 x 11).
- [x] Committed raw JSONL unedited (6a44b24)

## Stage 6 tasks (report)
- [x] (a) primary comparison (H1: Recall@5 on variants B and C, Holm-corrected): PARTIAL (C significant, B n.s.)
- [x] (b) all variants and baselines per task, per-repo and per-language breakdowns
- [x] (c) sensitivity analysis
- [x] (d) AXIOM case-study table (separate, in retrieval-case-studies.md)
- [x] Write limitations
- [x] Report written: docs/research/benchmark-v2/test-results-v2.md
- [ ] Open PR; do NOT merge

## Next step
Execute the single test run.

## Decisions log
- Stage 0: pilot closed with "no detectable difference" headline. 16 pilot tasks now dev-only.
- Stage 1: H1 moved to variants B and C (Holm-corrected); variant A secondary.
- Stage 2: 54 tasks / 6 repos, gate passed. H3 stays exploratory (all logic tags).
- Stage 3: --manifest flag added; v2 dev refs resolve against v1 manifest.
- Stage 4: frozen. No amendments after 4699776.

## Attempts used
- Stage 0: 1 (success)
- Stage 1: 1 (success)
- Stage 2: 1 (success, delegated)
- Stage 3: 1 (success)
- Stage 4: 1 (success)
- Stage 5: attempt 1 interrupted by service restart at task 39/54 (infra crash, documented); attempt 2 in progress
