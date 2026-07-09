# Functional Superiority: five capability upgrades with a benchmark proof layer

**Date:** 2026-07-02
**Status:** Approved design, pending implementation plan
**Goal:** Make Axiom measurably superior to LLM gateways, memory layers, agent-reliability
tools, and (uncontested) fleet-learning systems — by shipping five functional upgrades and
wrapping them in a reproducible benchmark harness whose published numbers carry the claim.

## Framing

The intelligence stays in the cloud models (Claude and Codex traffic both route through the
proxy). Axiom's local layer is deterministic compression, verification, and healing — no
local model training and no GPU are required for anything in this design. Benchmarks are the
proof layer, not the product: if the harness were deleted afterward, Axiom would still be
materially better on every axis.

## Non-goals

- Training or scaling the local TTT checkpoint.
- A hosted/cloud service; everything here runs on user hardware.
- Dashboards or growth tooling (later cycle; the savings receipt is the seed).
- Multi-provider routing beyond the existing Anthropic + OpenAI upstreams.

## Deliverable 1 — Compression on by default (and correct)

**Today:** Responses compression is opt-in and has a correctness bug (issue #85: merging
non-contiguous assistant items reorders retained items in multi-turn transcripts).

**Change:**
- Fix #85 in `responses_compressor.rs`: replace each *contiguous run* of eligible assistant
  items independently, anchored in place, so intervening user/tool items keep their positions.
  Property test: compressed transcript preserves the relative order of all retained items.
- Flip the default: safe compression ON for both `/v1/messages` and `/v1/responses`
  (`AXIOM_TTT_COMPRESS=0` remains the kill switch). "Safe" = the structural/skeleton path;
  anything that fails a fidelity precondition passes through untouched.
- Existing degraded-fallback behavior (retry original payload on upstream 4xx/5xx) is kept
  and becomes load-bearing; it is the error boundary for default-on.

**Interface:** unchanged for clients. **Depends on:** nothing. **Blocks:** cost pillar.

## Deliverable 2 — Session record/replay

**Today:** sessions leave only log lines; nothing can be replayed or diffed.

**Change:**
- `AXIOM_SESSION_RECORD=1` (default off) makes the proxy persist per-session JSONL under
  `~/.axiom/sessions/<session-id>.jsonl`: one record per request/response pair
  `{ts, endpoint, request, response, tokens_in, tokens_out, compressed}` with secrets
  scrubbed at write time (Authorization and any allowlisted-relay header values are never
  persisted; a regex pass drops apparent keys/tokens from bodies).
- `axiombench record --scrub` promotes recorded sessions into corpus format: content-hash
  named, PII/secret scrub re-run, provenance stamped.
- Replay consumer #1 is the bench; consumer #2 is regression debugging ("did this proxy
  change alter what the model saw?") via `axiombench replay --diff <a> <b>`.

**Error handling:** recording failures never block the request path (log-and-continue).
**Depends on:** nothing. **Blocks:** corpus, cost pillar.

## Deliverable 3 — Calibrated trust gate shipped by default

**Today:** the conformal gate exists but runs uncalibrated unless env vars are hand-set.

**Change:**
- Build a labeled claim/evidence dataset (~500 claims): seeded from existing verify test
  fixtures, extended with claims harvested from recorded sessions (hand-labeled) and
  synthetic contradiction pairs. Stored under `bench/trust/` with a held-out split.
- Calibrate via the existing `/v1/verify` calibrate mode; ship the resulting
  `AXIOM_CONFORMAL_THRESHOLD` + `AXIOM_CONFORMAL_DELTA` in release artifacts
  (`axiom.env` defaults + documented in README), so `/v1/verify` verdicts carry a stated
  coverage guarantee out of the box.
- ChimeraLang pairing benefits automatically (its verify calls hit the same gate).

**Depends on:** deliverable 2 (claim harvesting), but fixture-seeded work can start first.
**Blocks:** trust pillar.

## Deliverable 4 — Fleet mode productized

**Today:** DWE listener and immunity merge are wired and unit-tested but unauthenticated
(issue #86) and never exercised across real nodes.

**Change:**
- Authenticate inbound DWE fragments and immunity merges: shared-secret HMAC over the
  fragment bytes (`AXIOM_FLEET_SECRET`), reusing the sequence-version replay rejection the
  cluster sync path already has. No secret configured = listener refuses to start.
- `axiom fleet join <peer>` CLI: wires DWE peers + immunity gossip in one command;
  `axiom fleet status` shows peer health and last-merge stats.
- Two-node integration test (both nodes in-process): node A fails a command, heals, exports
  immunity; node B merges and pre-heals the same failure. This test is the fleet pillar's
  engine.

**Depends on:** nothing. **Blocks:** fleet pillar.

## Deliverable 5 — Savings receipts

**Today:** compression savings are visible only in scattered stderr lines.

**Change:**
- Extend `/metrics` with per-session and lifetime counters: `tokens_in`, `tokens_forwarded`,
  `tokens_saved`, `saved_ratio`, plus verify/heal counters.
- On session drop (existing lifecycle hook), emit a one-line receipt to the proxy log:
  `session <id>: 41k in, 17k forwarded, 58% saved`. The same numbers power the cost pillar.

**Depends on:** deliverable 1 (meaningful savings to report).

## Proof layer — AxiomBench

`axiombench` binary in the workspace (`src/bin/axiombench.rs`, `tools` feature), `bench/`
directory for corpus and results. Four pillars, one per competitive axis; each consumes a
deliverable above:

| Pillar | Consumes | Headline number | Runs in CI? |
|---|---|---|---|
| Cost | D1 + D2 corpus | token reduction % at task-quality delta | no (`--live`, needs API keys) |
| Cognition | skeleton + `/v1/expand` | symbol exact-recovery rate | yes (deterministic) |
| Trust | D3 dataset + gate | catch-rate at stated coverage vs ungated | yes (deterministic) |
| Fleet | D4 two-node test | time-to-immunity, pre-heal success rate | yes (deterministic) |

- **Corpus:** versioned, content-hashed JSONL under `bench/corpus/` (~30 Claude + ~30 Codex
  sessions, scrubbed); every published number cites the corpus hash. Synthetic sessions use
  pinned commits of public OSS repos.
- **Run config:** a checked-in `bench/config.toml` pins everything a number depends on —
  judge model id, judge prompt hash, confidence floor, retry/backoff params, and corpus
  hash. The config hash is stamped into every results file so a published table is fully
  reproducible from `(config hash, corpus hash)`.
- **Quality judge (cost pillar):** exact checks where outcomes are verifiable (tests pass,
  patches apply); a fixed cloud LLM judge for open-ended outputs, identical judge for both
  arms; judge scores under a confidence floor are flagged, not averaged.
- **Reporting:** `bench/results/<timestamp>-<corpus-hash>.json` + generated markdown;
  `RESULTS.md` at repo root holds the current headline table; README links it. CI runs the
  three deterministic pillars on every release.
- **Error handling:** per-case isolation; API retries with backoff; errored cases counted
  and reported separately, never silently dropped; corpus hash-verified before every run.

## Testing

- Unit tests for metrics math and the #85 ordering property (proptest-style over synthetic
  transcripts).
- A 3-case golden mini-corpus ships in-repo; `cargo test` exercises record→replay→report
  offline against a mock upstream.
- The two-node fleet integration test runs in the normal suite.
- CI smoke-runs the mini-corpus on PRs; full deterministic pillars on release.

## Phasing

- **Phase 0:** fix #85; flip compression default. (Everything else is additive.)
- **Phase 1:** session record/replay (D2) + savings receipts (D5).
- **Phase 2:** trust dataset + shipped calibration (D3); fleet auth + CLI (D4). Independent,
  parallelizable.
- **Phase 3:** AxiomBench harness + corpus v1 + RESULTS.md + CI wiring.

## Risks

- **Cost-pillar quality judging is the weakest link:** LLM judges drift. Mitigation: pin the
  judge model + prompt in the corpus hash; lead with exact-check cases.
- **Recorded-session privacy:** scrubbing must be conservative; corpus promotion is manual
  (`--scrub` + human review) before anything is committed.
- **Default-on compression regressions:** the pass-through fallback and kill switch bound the
  blast radius; receipts (D5) make any quality change visible immediately.
