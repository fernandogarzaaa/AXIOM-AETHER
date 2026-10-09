# AXIOM Skeletonizer: Negative Results and Research Pivot

**Date:** 2026-10-08
**Status:** Findings frozen. This document records what we tested, what failed,
and where the research goes next. It is deliberately written before the
retrieval system succeeds, so the negative results stand on their own.

## Abstract

We tested whether AXIOM's code skeletonizer — a deterministic structural
compressor achieving 80-89% token reduction — improves AI agent bug-fixing
outcomes. Across three controlled experiments, skeletons never improved
outcomes: they added a round trip for localized bugs, matched grep for
import-chain bugs, and were 3x slower with 16x the tokens for decoupled bugs.
A follow-up safety investigation found the digest default is
misleading-but-recoverable when expansion is available. We are pivoting from
"source to skeleton" to "repository to hierarchical evidence representation,"
with the skeleton demoted from the solution to one layer in a query-conditioned
retrieval architecture.

## Original Hypothesis

"Give the agent a compressed structural representation of the repository and
it will debug more efficiently."

## Representation

The skeletonizer (`axiom skeleton`, PR #196) parses source files and emits a
structural summary:

- **Kept:** function/method signatures, class definitions, imports, type
  annotations, module structure.
- **Dropped:** all function bodies, module-level constant values, comments,
  docstrings.
- **Compression:** 80-89% token reduction measured on GHOST-Chimera
  (3.9 MB source to 420 KB skeletons across 270 files; 79.8% on the
  single-file case).
- **Ranked variant:** `skeletonize_ranked()` extracts symbols, builds a
  reference graph, applies PageRank, and emits higher-ranked symbols first
  within a token budget. Used by the production digest path
  (`AXIOM_CVM_DIGEST=skeleton`), which triggers on `tool_result` blocks above
  4000 tokens with a 15% budget.

The compression number is real and deterministic. What was unproven — and is
now disproven for the tested scenarios — is that it improves agent outcomes.

## Experiment 1: Diagnosis

**Hypothesis:** Skeletons provide enough signal to diagnose a localized bug.

**Method:** Realistic bug planted in `ghostchimera/safety_layer/ssrf.py`:
`SSRFPolicy.to_dict()` omitted the `"denied"` key from its returned dict.
Verifier: `pytest
tests/test_ssrf_policy.py::SSRFPolicyTests::test_to_dict`. Two conditions,
same bug, same agent model:

- **A (full source):** failing test output + complete 12,476-char source in
  briefing (13,853 chars total, ~3,463 tokens).
- **B (skeleton):** failing test output + `axiom skeleton` output
  (2,518 chars, 79.8% smaller than source) in briefing (4,227 chars total,
  ~1,056 tokens). Agent retained filesystem access and was told to read full
  files as needed.

**Result:**

| Metric | A (full source) | B (skeleton) |
|---|---|---|
| Briefing input | 13,853 chars (~3,463 tok) | 4,227 chars (~1,056 tok) |
| Wall time | ~46 s | ~17 s |
| Fix correct? | Yes, byte-identical | Yes, byte-identical |
| Tests after fix | 24/24 pass | 24/24 pass |
| Full files read | 1 | 2 |

Agent B's briefing was 69.5% smaller, but B then read the full 12.5 KB source
file plus the 6.2 KB test file from disk. Effective tokens consumed by B
(~5,700) **exceeded** A's upfront ~3,463.

**Failure analysis:** The skeleton showed `def to_dict(self) -> dict[str,
Any]:` but elided the dict literal body — the missing key was invisible. The
skeleton provided *navigation signal* (where `to_dict` lives) but zero
*diagnostic signal* (what is wrong inside it). This tested the weakest case
for skeletons: a localized bug in an already-identified file, where the
elided content is exactly what is needed for diagnosis.

## Experiment 2: Navigation

**Hypothesis:** Skeletons help cross-file bug hunting across 270 files.

**Method:** Bug planted in `ghostchimera/redaction.py` line 32: `RISK_ORDER`
changed from `{"low": 1, ...}` to `{"low": 3, ...}`. The failing test
(`test_summary_flags_high_risk_unreviewed_records`, `2 != 1`) targets
`capability_admission.py`, which imports `RISK_ORDER` from `.redaction`. The
redaction module's own 13 tests still pass. Two conditions:

- **A:** file listing + grep + full file reads.
- **B:** 420 KB skeleton of all files upfront, then full reads of chosen
  files.

**Result:**

| Metric | A (grep) | B (skeletons) |
|---|---|---|
| Wall time | ~28 s | ~40 s |
| Full files read | 2 | 2 |
| Fix correct? | Yes | Yes |

Both agents followed the identical 2-hop path: failing test →
`capability_admission.py` → saw `RISK_ORDER` imported from `.redaction` →
read `redaction.py` → found wrong values. The 12-second difference is noise.

**Failure analysis:** The bug was too easy for a navigation test. The import
chain was visible in both conditions, making this a 2-hop trace rather than a
genuine "needle in 270 files" search. **Secondary finding:** the skeleton
elides module-level constant values entirely, so for data-level bugs (wrong
constant), skeletons actively *hide* the defect. They are strictly a
navigation aid.

## Experiment 3: Decoupled Retrieval

**Hypothesis:** Skeletons help when the symptom and cause have no visible
import-chain connection.

**Method:** Bug planted in
`ghostchimera/chimera_pilot/desktop_policy.py` line 96: the fallback in
`infer_desktop_action_class` flipped from `MUTATING` to `DESTRUCTIVE`. The
failing test never imports `desktop_policy`; the error message
("'destructive' is disabled by policy") misleadingly blames the allowlist.
Grep for "destructive" returns 11 files. Two conditions:

- **A:** file listing + grep + full file reads.
- **B:** 420 KB skeleton of all files upfront (8,862 lines), then full reads.

**Result:**

| Metric | A (grep) | B (skeletons) |
|---|---|---|
| Wall time | **31 s** | 96 s |
| Full files read | 3 | 2 |
| Fix correct? | Yes, root cause | Yes, root cause |
| Est. tokens | ~7,000 | ~111,000 |

Condition A was **3x faster and used 16x fewer tokens**. The skeleton agent
took a marginally more direct path (one fewer file read), but the 105k-token
upfront cost of the skeleton dwarfed that benefit.

**Failure analysis:** Broadcasting a compressed representation of the whole
repository can be **more expensive than retrieving the relevant raw
evidence**. We compressed each file individually but did not compress the
*search problem*. A 20% representation of 500 files is still a huge amount
of irrelevant context. For a bug requiring 2-3 file reads, the skeleton is
massive overkill. A competent agent follows imports quickly; skeletons do not
add information that `grep "infer_desktop_action_class"` would not reveal in
seconds.

## Cross-Experiment Synthesis

| Experiment | Scenario | Skeleton benefit |
|---|---|---|
| Exp 1 | Localized bug, known file | Negative (added round trip) |
| Exp 2 | 2-hop import chain | None (matched grep) |
| Exp 3 | Decoupled, misleading error | Negative (3x slower, 16x tokens) |

Three experiments, three failures of the original hypothesis. The compression
number (80-89%) is real but never translated to better outcomes.

The key insight: **the agent's cost is dominated by what it reads, and a
capable agent with grep already reads selectively.** Compression only wins
when the context was going to be paid for anyway. Skeletons do not replace
reading; at best, they replace a worse orientation step — and in our tests,
they did not even do that.

## Safety Investigation

**Question:** Does `AXIOM_CVM_DIGEST=skeleton` (on by default) actively
mislead agents? Do agents trust digests and silently miss bugs invisible in
the compressed view?

**Mechanism (verified in code):** `AXIOM_CVM_DIGEST` defaults to `"skeleton"`
(`routes_messages.rs:593`). Any `tool_result` block above 4000 tokens is
replaced with a PageRank-ranked digest at 15% budget. The digest ends with
`[AXIOM-PAGE-END expand with axiom_expand("{page_id}")]`, pointing the agent
at the `axiom_expand` MCP tool for the full text.

**Method:** 8 Python modules with planted bugs, digested with the real Rust
`SkeletonDigestor`. 16 agent runs (8 cases x 2 rounds), each seeing only the
digest + expansion hint, tasked to find the bug.

**Results:**

| Outcome | Count |
|---|---|
| Requested expansion | 15/16 |
| Correctly identified bug from digest (control) | 1/16 |
| Confident wrong answer | **0/16** |

The silent failure mode **did not materialize** in this harness. Zero
confident hallucinations.

**Sharp edge:** For 3 of 8 cases, the buggy *function* was elided entirely
from the digest — not just its body. The agent sees no trace that the
function exists. An agent cannot expand what it does not know is missing.
(Test agents still expanded, prompted by the "N lines elided" marker.)

**External critique (Claude):** The 0/16 is weaker than it looks — agents
were told not to guess, n=16 gives a 19% upper bound on the failure rate
(rule of three), and all runs used the same model family. More importantly:
**if the client never registered the Axiom MCP server, `axiom_expand` does
not exist, and the digest is unrecoverable.** The proxy sees the `tools`
array in each request, so digesting should be gated on the tool's presence.
Non-code payloads (test logs, stack traces, JSON) were not tested and may be
higher risk than source code.

**Revised rating:** Unproven under realistic conditions; recoverable only in
a best-case harness.

## P0 Safety Invariant

**`LOSSY_CONTEXT ⇒ RECOVERY_CAPABILITY_PRESENT`**

A lossy representation must never be served without the corresponding
recovery mechanism. PR #207 implements this mechanically:

1. **Expand gate:** digesting is skipped entirely when `axiom_expand` (or
   the MCP-namespaced `*__axiom_expand`) is absent from the request's
   `tools[]` array. Missing or empty tools array fails closed. Applies to
   all digest modes.
2. **Symbol name retention:** `skeletonize_ranked` no longer lets symbols
   vanish under token budget. Elided symbols emit `// name (… body elided
   …)` stubs. Python `UPPER_SNAKE_CASE` module constants are now detected
   and retained as stubs.

This invariant should govern every future compression decision in AXIOM, not
just the current digest path.

## Diagnostic Elision (formerly `--smart`)

Direct response to Experiment 1's failure mode. PR #205 adds a
`--diagnostic` flag (renamed from `--smart` because "smart" overclaims —
the mode preserves manually-defined diagnostic signals, not learned
relevance) that preserves:

- Error handling paths (`try`/`except`, `raise`, `unwrap()`, `?` operator)
- Boundary conditions (comparisons, bracket-aware to exclude `Vec<T>`)
- Suspicious patterns (TODO/FIXME, bare `except:`, asserts)
- Complex conditionals (3+ boolean connectors)
- Return statements and Rust tail expressions

At 51.6% of source (vs 19.9% for the default skeleton), it **would have
revealed the Experiment 1 bug** — verified by reintroducing the bug and
confirming the missing key is visible in the output.

**Caveats:** This fixes one observed information-loss failure mode. It is a
heuristic salience classifier, not a relevance system. It has not been
tested on held-out bugs. We will not keep adding heuristic rules; the next
investment is retrieval, not better elision.

## Revised Hypothesis

"Structural representations can help an agent acquire relevant repository
evidence more efficiently when combined with task-conditioned retrieval and
exact source expansion."

The skeleton is demoted from "the solution" to "one representation in an
evidence acquisition architecture":

```
repository → hierarchical evidence representation → query-conditioned context
```

## Query-Conditioned Retrieval (in progress)

Prototype (`axiom rank`, `axiom context`) implementing:

- Lexical overlap with identifier splitting and IDF weighting
- Module-level constant extraction
- Import-graph proximity
- Tiered context assembly: top files get full source, next tier gets
  skeletons, rest get signatures — within a token budget
- Wired to the existing `/v1/expand` mechanism

Early results on the 3 known experiment bugs: 2 ranked in top 3, 1 at #17
(informative failure — lexical + shallow import proximity is insufficient
for semantic decoupling). Per the agreed discipline: add minimal
test-to-code mapping and ranking provenance, then **freeze the algorithm**
before benchmarking. No more heuristic tuning until the benchmark decides.

## Current Evidence

| Claim | Status | Source |
|---|---|---|
| Skeletonizer achieves 80-89% token reduction | **Established** | Measured on GHOST-Chimera (270 files) |
| Signature round-trip fidelity | **Established** | Existing benchmark suite |
| Skeletons improve bug-fix success or speed | **Falsified** (for tested scenarios) | Exp 1 (negative), Exp 2 (neutral), Exp 3 (negative) |
| Skeletons are safe as a default digest | **Unproven** under realistic conditions | Safety investigation + Claude critique |
| Digest without expansion capability is harmful | **Accepted as invariant** | `LOSSY_CONTEXT ⇒ RECOVERY_CAPABILITY_PRESENT`, PR #207 |
| Diagnostic elision preserves bug-relevant info | **Unproven** (single-case verification only) | PR #205 |
| Query-conditioned retrieval beats grep | **Unknown** | Prototype exists, not benchmarked |

## What Remains Unknown

- Does query-conditioned retrieval beat grep on held-out bugs?
- Does diagnostic elision help on bugs it was not designed for?
- What happens when non-code payloads (logs, traces, JSON) are digested?
- Does the safety case hold for weaker or less disciplined models?
- At what repository scale does structural indexing pay for itself?
- Is compression value conditional on retrieval precision (the 2x2
  hypothesis)?

## Preregistered Next Experiment

**Retrieval-only benchmark** (before any end-to-end agent evaluation):

- **Dataset:** 30-50 real bugs (mined from bug-fix commits, not planted),
  each with repository, buggy commit, issue description, gold file, gold
  symbol, and failing tests.
- **Conditions:** lexical only, lexical + IDF, lexical + import graph,
  lexical + test mapping, skeleton, repo map, combined AXIOM retrieval.
- **Metrics:** Recall@1, Recall@3, Recall@5, Recall@10, MRR, NDCG, tokens
  required to expose gold evidence.
- **Discipline:** Freeze the ranking algorithm before benchmarking. Maintain
  development / held-out / frozen-test splits. Paired per-task analysis
  (McNemar-style), not pooled means.
- **Token accounting:** measured tokenizer counts (input, output, tool
  results, cache), not chars/4.

Only after retrieval is characterized do we run end-to-end repair
experiments, measuring verified task success and cost per successful task —
with compression ratio as a constraint, not the objective.

## Claims We Explicitly Do NOT Make

- Skeletonization improves bug-fixing success.
- Token reduction implies task improvement.
- Diagnostic elision preserves bug-relevant information for arbitrary bugs.
- Query-conditioned retrieval improves repair performance.
- AXIOM retrieval currently outperforms standard repository search.
- The 80-89% compression number means agents work better or cheaper in
  practice.

These are open research questions. This document exists so that future work
is measured against what we actually found, not what we hoped to find.
