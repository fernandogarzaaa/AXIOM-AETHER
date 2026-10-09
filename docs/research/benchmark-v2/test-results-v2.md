# Retrieval Benchmark v2: Test Results Report

**Date:** 2026-10-09
**Status:** Complete. Stage 6 done.
**Ranker:** pinned at `7c373819b24b783ea457fa547a9c13762c8174a5` (frozen, verified pre-run)
**Dataset:** 54 test tasks from 6 repos (commit 2111574)
**Run:** single test run, attempt 2 after infrastructure crash (documented in STATE.md). Exit 0, 1782 lines (54 x 3 x 11). Raw JSONL committed unedited (6a44b24).

## Provenance

| Item | Value |
|------|-------|
| Ranker commit | `7c373819b24b783ea457fa547a9c13762c8174a5` |
| Harness commit | `b99f18b465ace219f2cf17eb335c9d7fad75c12f` |
| Freeze commit | `4699776` |
| Spec | `docs/research/benchmark-v2/spec-v2.md` (0fa994a) |
| Dataset | `docs/research/benchmark-v2/dataset-manifest-v2.json` (2111574) |
| Results | `bench/results/v2-test.jsonl` (6a44b24), 1782 lines |
| Best baseline (pre-registered) | `grep` (from v1 pilot dev run) |

## (a) Primary comparison: H1

**H1:** On held-out bug-fix tasks, `axiom rank` achieves higher Recall@5 than the best standard baseline (grep) on query variants B (test output) and C (test name only), each tested separately with Holm correction for the two comparisons (family-wise alpha 0.05).

**Method:** Paired difference (axiom_rank minus grep) in Recall@5 per task. 95% cluster-bootstrap CI by repo (10,000 resamples, resample repos with replacement). Two-sided bootstrap p-value. Holm correction across the two variant comparisons.

### Variant B (test output)

- axiom_rank Recall@5: 0.6265
- grep Recall@5: 0.5586
- Paired difference: +0.0679
- 95% cluster-bootstrap CI: [-0.1789, +0.1953]
- Bootstrap p: 0.5612
- Per-task: axiom_rank better on 9, tie on 41, grep better on 4

**Result:** CI includes zero. Not significant.

### Variant C (test name only)

- axiom_rank Recall@5: 0.8395
- grep Recall@5: 0.3642
- Paired difference: +0.4753
- 95% cluster-bootstrap CI: [+0.3030, +0.6585]
- Bootstrap p: < 0.0001 (0/10,000 resamples at or below zero)
- Per-task: axiom_rank better on 31, tie on 21, grep better on 2

**Result:** CI excludes zero in axiom_rank's favor. Significant.

### Holm correction

Sorted p-values: C (p < 0.0001), B (p = 0.5612).

- Step 1: p_C < 0.05/2 = 0.025. Reject H0 for variant C.
- Step 2: p_B < 0.05/1 = 0.05? No (0.5612 > 0.05). Fail to reject for variant B.

### H1 verdict: PARTIAL

Per spec v2 Section 6 interpretation rules: one of two corrected intervals excludes zero in axiom_rank's favor. This is a **partial** outcome. We do not claim H1.

**Primary result in one sentence:** On 54 held-out bug-fix tasks, axiom_rank beat the pre-registered grep baseline on test-name-only queries (variant C Recall@5 0.84 vs 0.36, Holm-corrected significant) but showed no detectable difference on test-output queries (variant B Recall@5 0.63 vs 0.56, n.s.), yielding a partial H1 outcome.

## (b) Full results: all variants, all baselines

### Recall@5 by method and variant (mean over 54 tasks)

| Method | Variant A | Variant B | Variant C |
|--------|-----------|-----------|-----------|
| random/seed=11 | 0.0741 | 0.0741 | 0.0741 |
| random/seed=22 | 0.0648 | 0.0648 | 0.0648 |
| random/seed=33 | 0.0926 | 0.0926 | 0.0926 |
| random/seed=44 | 0.0278 | 0.0278 | 0.0278 |
| random/seed=55 | 0.0185 | 0.0185 | 0.0185 |
| grep | 0.5679 | 0.5586 | 0.3642 |
| bm25_content | 0.6389 | 0.5741 | 0.3056 |
| bm25_symbols | 0.4167 | 0.4599 | 0.1574 |
| import_graph | 0.0000 | 0.3673 | 0.4259 |
| stacktrace_frames | 0.0000 | 0.0000 | 0.0000 |
| axiom_rank | 0.6821 | 0.6265 | 0.8395 |

Notes:
- Random baselines are query-independent, so their scores are identical across variants (expected).
- stacktrace_frames scores 0.0000 Recall@5 on all variants. It extracts no usable frames from these queries (diagnostic: the method requires stack-trace-formatted input, which variants B and C do not reliably provide).
- import_graph scores 0.0000 on variant A (stripped issue text contains no test paths to seed the graph) but recovers on B and C where test names are present.

### Recall@1 by method and variant

| Method | Variant A | Variant B | Variant C |
|--------|-----------|-----------|-----------|
| grep | 0.1852 | 0.0741 | 0.0370 |
| bm25_content | 0.3302 | 0.0926 | 0.0370 |
| bm25_symbols | 0.2130 | 0.1235 | 0.0185 |
| import_graph | 0.0000 | 0.0000 | 0.0000 |
| stacktrace_frames | 0.0000 | 0.0000 | 0.0000 |
| axiom_rank | 0.2284 | 0.0617 | 0.0525 |

### MRR by method and variant

| Method | Variant A | Variant B | Variant C |
|--------|-----------|-----------|-----------|
| grep | 0.3939 | 0.2964 | 0.2166 |
| bm25_content | 0.5038 | 0.3386 | 0.1933 |
| bm25_symbols | 0.3153 | 0.3003 | 0.1213 |
| import_graph | 0.0180 | 0.1426 | 0.1843 |
| stacktrace_frames | 0.0180 | 0.0180 | 0.0180 |
| axiom_rank | 0.4716 | 0.3306 | 0.4369 |

### Per-repo Recall@5: axiom_rank vs grep

**Variant A (descriptive; secondary)**

| Repo | n | axiom_rank | grep | Diff |
|------|---|------------|------|------|
| encode/starlette | 3 | 0.3333 | 0.0000 | +0.3333 |
| pallets/click | 26 | 0.8462 | 0.7308 | +0.1154 |
| pallets/werkzeug | 8 | 0.4167 | 0.2917 | +0.1250 |
| psf/cachecontrol | 3 | 1.0000 | 0.8333 | +0.1667 |
| python-attrs/attrs | 3 | 0.3333 | 0.3333 | +0.0000 |
| urllib3/urllib3 | 11 | 0.5909 | 0.5303 | +0.0606 |

**Variant B (primary)**

| Repo | n | axiom_rank | grep | Diff |
|------|---|------------|------|------|
| encode/starlette | 3 | 0.0000 | 0.0000 | +0.0000 |
| pallets/click | 26 | 0.8846 | 0.6923 | +0.1923 |
| pallets/werkzeug | 8 | 0.1667 | 0.5417 | -0.3750 |
| psf/cachecontrol | 3 | 0.6667 | 0.5000 | +0.1667 |
| python-attrs/attrs | 3 | 0.6667 | 0.3333 | +0.3333 |
| urllib3/urllib3 | 11 | 0.5000 | 0.4848 | +0.0152 |

Note: on variant B, pallets/werkzeug shows axiom_rank underperforming grep by 0.3750. This repo-level heterogeneity drives the wide cluster-bootstrap CI. With only 6 repos, a single repo reversing direction substantially widens the interval.

**Variant C (primary)**

| Repo | n | axiom_rank | grep | Diff |
|------|---|------------|------|------|
| encode/starlette | 3 | 0.6667 | 0.0000 | +0.6667 |
| pallets/click | 26 | 0.9231 | 0.3846 | +0.5385 |
| pallets/werkzeug | 8 | 0.6667 | 0.4167 | +0.2500 |
| psf/cachecontrol | 3 | 0.6667 | 0.0000 | +0.6667 |
| python-attrs/attrs | 3 | 1.0000 | 0.0000 | +1.0000 |
| urllib3/urllib3 | 11 | 0.8182 | 0.5758 | +0.2424 |

On variant C, axiom_rank beats grep in every repo. The effect is consistent, not driven by a single repo.

### Per-language breakdown

All 54 test tasks are Python (6 Python repos). There is no cross-language variation to report. This is a limitation (see below): the benchmark does not test whether the ranker's advantage generalizes beyond Python.

### Per-task table

The full per-task Recall@5 table (54 tasks x 11 methods x 3 variants) is available in the raw results file `bench/results/v2-test.jsonl` (commit 6a44b24). Summary counts:

- Variant B: axiom_rank strictly better than grep on 9 tasks, tied on 41, worse on 4.
- Variant C: axiom_rank strictly better than grep on 31 tasks, tied on 21, worse on 2.

The high tie count on variant B (41/54) reflects that both methods often either both hit or both miss the gold file within top 5 on test-output queries.

### H2 (secondary): axiom advantage larger on A than on C?

- Mean paired advantage (axiom_rank minus grep) on variant A: +0.1142
- Mean paired advantage on variant C: +0.4753
- **H2 NOT supported.** The direction is opposite to H2 (same as the pilot's descriptive finding). The ranker's advantage is largest where the query is a test name, not where it is stripped issue text.

## (c) Sensitivity analyses (pre-specified)

### 1. Short queries (variant-A query under 6 tokens)

5 tasks have variant-A queries under 6 tokens; 49 have 6 or more.

| Cut | Variant B diff | Variant C diff |
|-----|---------------|----------------|
| Full (n=54) | +0.0679 | +0.4753 |
| Short only (n=5) | +0.0667 | +0.2667 |
| Long only (n=49) | +0.0680 | +0.4966 |

The short-query cut does not materially change the story. On variant B the difference is near zero in all cuts. On variant C the advantage persists in both cuts (smaller in the 5-task short cut, but n=5 is too small for inference).

### 2. Tie sensitivity (Recall@5 boundary)

Count of tasks where the first gold-file rank is 5 or 6 (i.e., the task sits on the Recall@5 boundary and could flip with a different tie-break):

| Variant | axiom_rank | grep |
|---------|------------|------|
| B | 11 tasks | 8 tasks |
| C | 3 tasks | 2 tasks |

On variant B, 11/54 axiom_rank tasks and 8/54 grep tasks sit on the boundary. The tie-break behavior (axiom_rank has no explicit tie-break; Rust stable sort preserves candidate collection order) could affect a meaningful number of variant-B outcomes. On variant C, boundary tasks are few (3 and 2), so the significant result is not tie-break-driven.

### 3. Per-repo heterogeneity

Reported in Section (b) above. Key finding: on variant B, pallets/werkzeug reverses the direction (axiom_rank worse than grep by 0.3750), which is the primary driver of the wide CI and the null result. On variant C, the direction is consistent across all 6 repos.

## (d) AXIOM case-study table (separate, NOT benchmark data)

Per spec v2 Section 7C (carried from v1 spec Section 7C): the three known bugs from the retrieval prototype's own development are reported here for context only. They are **not** benchmark tasks (different repos, no fail/pass verification, queries not constructed per protocol) and are **never mixed into Recall@k aggregates**. The ranker was developed with knowledge of these bugs; these ranks cannot support any claim about retrieval performance.

See `docs/research/retrieval-case-studies.md` for the full table:

| # | Bug | Repo | File | axiom_rank rank (prototype-reported) |
|---|-----|------|------|--------------------------------------|
| 1 | Missing `"denied"` key in `to_dict()` | GHOST-Chimera | `safety_layer/ssrf.py` | #2 |
| 2 | `RISK_ORDER` maps `"low"` to `3` instead of `1` | GHOST-Chimera | `ghostchimera/redaction.py` | #17 |
| 3 | `infer_desktop_action_class` fallback flipped | chimera_pilot | `chimera_pilot/desktop_policy.py` | #1 |

## Limitations

1. **Single language.** All 54 tasks are Python. The benchmark says nothing about retrieval performance on other languages.
2. **Small repo count for cluster bootstrap.** The 95% CI uses cluster bootstrap by repo with only 6 clusters. With so few clusters, the CI is sensitive to single-repo heterogeneity (as seen on variant B with pallets/werkzeug). The interval is honest but wide.
3. **Tie-break unspecified for axiom_rank.** The ranker has no explicit tie-break; Rust's stable sort preserves candidate collection order. On variant B, 11/54 axiom_rank tasks sit on the Recall@5 boundary (rank 5 or 6). A different tie-break could move these. This is a known limitation, not a bug fixed during the benchmark (ranker frozen).
4. **stacktrace_frames baseline scored zero.** This baseline extracted no usable signal from any variant. It may be misconfigured for these query formats rather than genuinely uninformative. Its zero scores do not mean stack-trace methods are useless in general.
5. **import_graph scored zero on variant A.** Expected: stripped issue text contains no test paths to seed the graph. This is a query-format limitation, not a method flaw.
6. **Test-name queries may overstate real-world utility.** Variant C (test name only) is the variant where axiom_rank shines, but in practice a developer often has more context than just a test name. The benchmark does not measure performance on the richer, messier queries developers actually write.
7. **No tuning on test, but the ranker was built for this.** The ranker prototype was developed with the general goal of test-to-source retrieval. The variant-C advantage may partly reflect that the ranker's design aligns with the benchmark's query construction, not a general retrieval capability.
8. **One run.** Per protocol, the test split was run exactly once (plus one documented infrastructure-crash rerun with identical code and inputs). There is no test-set replication.
9. **H3 dropped.** All tasks are logic bugs; the data/logic tag distinction did not materialize. H3 remains exploratory-only per spec.

## What the data supports and does not support

**Supports:**
- axiom_rank retrieves the correct source file from a test-name query substantially better than grep (Recall@5 0.84 vs 0.36, consistent across all 6 repos, Holm-corrected significant). This is a real, replicable capability on this task distribution.
- The ranker's advantage is specific to test-name queries. It does not generalize to test-output queries in this benchmark.

**Does not support:**
- A general claim that axiom_rank beats standard baselines on bug-fix retrieval. H1 was not claimed (partial outcome). On variant B the difference is indistinguishable from zero.
- Any claim about non-Python languages, or about query types beyond the three tested variants.
- That the ranker is better than BM25 or other baselines in general. The pre-registered comparison was against grep only. (Descriptively, axiom_rank also beats bm25_content on variant C 0.84 vs 0.31, but this was not the pre-registered test.)
- Causal attribution of *why* axiom_rank wins on variant C. The benchmark measures retrieval outcomes, not mechanisms.

## Interpretation (per spec v2 Section 6)

**Partial outcome.** Mixed evidence. axiom_rank shows a large, significant advantage on test-name queries but no detectable difference on test-output queries. Do not claim H1. Recommended next step per spec: targeted analysis of why variant B shows no advantage (in particular the pallets/werkzeug reversal) before any Phase B representation experiments.
