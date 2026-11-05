# JSON digest — measured on real MCP tool results (2026-11-05)

**What was measured.** Five JSON `tool_result` blocks recorded from a real
Claude Code session on this repository (GitHub MCP server: `list_pull_requests`
×2, `actions_list(list_workflow_runs)` ×2, `get_job_logs` ×1), extracted from
the session transcript as-is. Numbers are the engine's own estimates
(`whitespace tokens` = `split_whitespace().count()`, the crate's convention;
`chars/4` = the ranked skeletonizer's convention), not provider billing.
Reproduce with the fixtures in a directory and:

```bash
AXIOM_JSON_DIGEST_FIXTURES=<dir> cargo test --lib -- --ignored --nocapture print_measurement_table
```

**Why it matters.** Digest admission (`apply_digest_admission`,
`routes_messages.rs`) compared **whitespace tokens** against the 4000-token
threshold. Minified JSON has 5–10× fewer whitespace tokens than model tokens,
so none of these five blocks — 1.7k to 9.3k model tokens each — was ever
digested: each reached upstream verbatim and then sat in the cached transcript
for every later turn of the session. Admission now estimates JSON as
`max(whitespace, chars/4)`, and the digest is structure-preserving
(`json_digest.rs`) rather than the prose head+tail excerpt.

| fixture | chars | ws tokens | chars/4 | admitted before? | budget (15%) | digest tokens | reduction |
|---|---|---|---|---|---|---|---|
| `list_pull_requests` (20 PRs) | 8561 | 272 | 2141 | **no** | 321 | 318 | 85.1% |
| `list_workflow_runs` (15 runs, long commit messages) | 37073 | 2758 | 9269 | **no** | 1390 | 971 | 89.5% |
| `list_workflow_runs` (6 runs) | 22652 | 2090 | 5663 | **no** | 849 | 783 | 86.2% |
| `list_pull_requests` (21 PRs, fewer fields) | 6784 | 289 | 1696 | **no** | 254 | 230 | 86.4% |
| `get_job_logs` (one 13 KB string field) | 13379 | 566 | 3345 | **no** | 502 | 70 | 97.9% |

Total: 88,449 chars ≈ 22.1k estimated tokens in, 2,372 out — **89.3% reduction**
on content that previously was not reduced at all. At the default threshold of
4000 only the two `list_workflow_runs` blocks would be admitted (the others sit
between 1.7k and 3.3k tokens); the table shows what the digest would do to each
if admitted, which is what `AXIOM_CVM_DIGEST_THRESHOLD_TOKENS=1500` buys.

**What the digest keeps.** For `list_workflow_runs` (15 runs): `total_count:
637`, `workflow_runs: array · 15 object(s)`, the union of 20 item keys with
types, value histograms for the enum-like fields (`status (string:
"completed"×15)`, `event (string: "dynamic"×10 "push"×5)`), three complete
items with nested objects collapsed and the 1.6 KB `head_commit.message`
truncated with its length marked, then `… +12 more item(s) with the same keys`
and the `axiom_expand` trailer. The prose fallback kept the first 1,200 chars
(half of item 1) and the last 500 (the tail of item 15).

**Caveats, which travel with these numbers.**

- Five blocks from one session, one MCP server. The shape (arrays of
  homogeneous objects with long string fields) is typical of API-backed MCP
  servers, but the reduction will be lower for deeply nested, heterogeneous
  JSON and for JSON that is mostly one long string (the `get_job_logs` row is
  the degenerate case: the digest cannot summarise a log, it can only show the
  string was 13 KB; the skeleton digest's prose excerpt may be preferable for
  log-shaped strings and that is a possible refinement).
- Token counts are estimates. The whitespace count is the crate's existing
  convention and the chars/4 estimate is the ranked skeletonizer's; neither is
  the provider's tokenizer. The *direction* (JSON undercounted by 5–10×) is
  robust to that.
- Admission savings compound: a block admitted once is smaller on every later
  turn's cached-prefix read for the rest of the session, which is where
  long-session cost lives (README § Prolonged-Session Stack). That compounding
  is not in the table.
- Correctness: the digest is recoverable verbatim via `axiom_expand` (tested
  end-to-end in `tests/digest_proxy.rs`), so the fault mode is an extra expand
  round-trip, the same as for code digests. No live answer-quality eval was
  run for JSON specifically; the S5 eval (`RESULTS-2026-07-11.md`) covered
  code digests.
