# CVM Cost Stack — Construction Blueprint

*Objective: make Axiom genuinely reduce real dollar cost for its users — target ≥70% vs
uncached baseline, ≥35% vs stock Claude Code with prompt caching — with no mid-session
latency or quality trade-offs.*

*Generated 2026-07-10 from a cited research pass + 4,000-session Monte-Carlo simulation
(`cvm_sim.py`, results in §0). Executor: Sonnet 5 agents, cold-start per step.*

---

## 0. Executor contract (read first, every step)

You are implementing one step of this plan in the AXIOM-AETHER repo (`D:\AXIOM-AETHER`,
GitHub `fernandogarzaaa/AXIOM-AETHER`). Rules that apply to EVERY step:

1. **Branch → PR → review → merge.** One branch per step (`cvm/s<N>-<slug>`), squash-merge
   after CI green. CodeRabbit + Codex post automated reviews on every PR: verify each
   finding against the code, fix real ones, reply with a triage comment. Never merge with
   unresolved CRITICAL/HIGH findings.
2. **TDD.** Write the failing test first, watch it fail for the right reason, then
   implement. Every acceptance criterion below maps to at least one test.
3. **CI gates** (must pass locally before pushing):
   ```
   cd axiom_engine_rs
   cargo test --lib
   cargo clippy --lib --locked -- -D warnings
   ```
   Full CI runs `cargo test --release --locked` on Ubuntu; the local Windows release
   build has a known unrelated `ppv-lite86` AVX2 quirk — do NOT chase it, debug profile
   locally is fine.
4. **Repo idioms:**
   - `src/server.rs` is 12 `include!` lines; server code lives in `src/server/*.rs`
     sharing ONE module scope. New server files must be added as an `include!` line.
   - MCP tools: catalogue in `mcp_stdio.rs::tools_list()`, dispatch in
     `handle_tools_call()`. The test `tools_list_exposes_tools_with_schemas` asserts the
     exact tool count — update it when adding tools.
   - Long-running work inside handlers uses `tokio::task::spawn_blocking`.
   - Never introduce a UTF-8 BOM (regression test `no_source_file_has_utf8_bom` scans
     `src/**/*.rs`). ASCII punctuation in source files.
   - Honesty convention: any output produced by an untrained/heuristic component must
     say so in-band (see `predictive_tools.rs` `trained: false` / `state_source`).
5. **Config convention:** every new behavior ships behind an env flag, default matching
   the "Defaults" column in §2. Flags are read once at context build
   (`mcp_stdio.rs::build_context` / `server/run.rs`), not per-request.
6. **Version/publishing:** do not bump `Cargo.toml` version in feature PRs. Step S8 does
   one bump; pushing a `v*` tag triggers the three publish workflows (crates.io+PyPI,
   release binaries, Docker/GHCR).

**Anchor facts** (measured 2026-07-10, this repo's live proxy): Claude Code request
prefix ≈ 319,248 bytes ≈ 80K tokens (tools+system); heavy tool result ≈ 16K tokens;
TTT compression ≈ 15.8 s per heavy event on CPU; Sonnet 4.6 pricing per MTok: input $3,
5-min cache write $3.75, cache read $0.30, output $15 (cache read is 0.1×: **rewriting
one cached token costs 10× carrying it**). Anthropic cache = byte-exact prefix match,
`tools → system → messages` render order, 4 breakpoints max, 20-block lookback, 5-min
TTL refreshed free on every hit. Claude Code uses caching by default.

**Why this plan exists** (simulation, 4,000 sessions, mean 34.6 turns):

| Scenario | $/session | vs uncached | vs stock CC |
|---|---|---|---|
| Uncached | 15.72 | — | — |
| Stock Claude Code (caching) | 7.24 | 54% | — |
| Axiom v0.2.9 (per-turn rewrite) | 6.58 | 58% | 9% + 4.6 min stalls |
| **Full CVM stack (this plan)** | **4.49** | **71.5%** | **38%** |

Two mechanisms carry nearly all the win: **prefix diet** (23% alone) and **digest
admission control** (24% alone). Scheduled eviction was simulated and REJECTED (worse
than baseline v1) — do not implement it. Digest quality is the critical risk: at 10%
page-fault rate the digest path is net-negative; the eval gate (S5) and prefetch hedge
(S7) exist because of this.

---

## 1. Dependency graph

```
S0 telemetry ──┬── S1 cache-safety ──┬── S2 L2 store ── S3 digest admission ──┐
               │                     └── S6 keepalive (after S1: both edit    ├── S5 eval gate ── S8 rollout
               │                          anthropic_forwarder.rs)             │
               └── S4 prefix diet (truly parallel: new isolated module)       │
                                       S7 prefetch (after S3+S5) ─────────────┘
```

Parallel-safe: S4 alongside anything after S0 (isolated new `prefix_diet.rs`). S1 and
S6 both modify `anthropic_forwarder.rs` and per-session state — run S6 AFTER S1, not in
parallel, to avoid merge conflicts. S2→S3 serial. S5 blocks default-flips. Model tier:
S5 and S7 want the strongest available model; S0–S4, S6, S8 are Sonnet-5 sized.

## 2. Flags and defaults

| Flag | Values | Default | Flipped by |
|---|---|---|---|
| `AXIOM_COST_TELEMETRY` | 0/1 | **1** | — (always safe) |
| `AXIOM_CACHE_SAFE` | 0/1 | **1** | — (S1; stops prefix-mutating compression for cache-bearing clients) |
| `AXIOM_CVM_DIGEST` | `off`/`skeleton`/`haiku` | **off** | S5 pass → `skeleton` |
| `AXIOM_CVM_DIGEST_THRESHOLD_TOKENS` | int | **4000** | — |
| `AXIOM_PREFIX_DEDUP` | 0/1 | **0** | S5 pass → 1 |
| `AXIOM_KEEPALIVE` | 0/1 | **0** | never auto (security: S6) |
| `AXIOM_CVM_PREFETCH` | 0/1 | **0** | S7's own eval |

---

## S0 — Dollar-true cache-aware telemetry

**Context brief.** Axiom's proxy (`src/anthropic_forwarder.rs`,
`src/server/routes_messages.rs`) forwards Anthropic `/v1/messages` traffic and today
counts only *bytes* saved. Precise geography (verified): the `/metrics` route is
*registered* in `server/routes_verify.rs:39` but the handler `export_metrics` and all
`axiom_savings_*` emission live in `server/routes_core.rs`; the byte receipt to extend
is `emit_savings_receipt` in `server/routes_sessions.rs`; the server-side session state
to extend is `AppState.awareness` (`server/prelude_state.rs`) — NOT the separate
`McpContext.awareness` in `mcp_stdio.rs`, which lives in a different process. Bytes lie:
rewriting a cached prefix "saves bytes" while multiplying real cost by 10. Anthropic
returns ground truth in every response: `usage.input_tokens` (uncached),
`usage.cache_creation_input_tokens` (1.25×), `usage.cache_read_input_tokens` (0.1×),
`usage.output_tokens`. Everything later in this plan is judged by this step's numbers.

**Task.**
1. New module `src/cost_ledger.rs` (declare in `lib.rs`):
   - `struct PriceTable { input, cache_write_5m, cache_read, output: f64 }` per MTok;
     `PriceTable::for_model(model: &str) -> (PriceTable, bool /*estimated*/)` with a
     static table for current Claude models (Sonnet 4.6/5: 3.00/3.75/0.30/15.00;
     Opus 4.7/4.8: 5/6.25/0.50/25; Haiku 4.5: 1/1.25/0.10/5; unknown model → Sonnet
     prices + estimated=true).
   - `struct TurnCost { uncached_in, cache_write, cache_read, output: u64, usd: f64, estimated: bool }`
   - `fn turn_cost(model: &str, usage: &serde_json::Value) -> Option<TurnCost>` parsing
     the four usage fields (absent field = 0; None only if `usage` missing/not-object).
   - Per-session accumulator keyed by session id, plus a counterfactual: what the same
     turn would have cost fully uncached (`(uncached_in+write+read) × input price`).
     Store alongside the existing session awareness state.
2. Hook: in the non-streaming response path AND the streaming path of the Anthropic
   forwarder. Streaming reality check: `forward_messages_stream` currently returns the
   upstream `reqwest::Response` whose `bytes_stream()` is piped straight to the client —
   there is no interception point today. You must wrap the stream in a scanning adapter
   (e.g. `futures::StreamExt::inspect` over chunks feeding a line-accumulator) that
   watches for the `message_start` SSE event (input-side usage) and the final
   `message_delta` event (output usage). The adapter may hold only the current partial
   SSE line for parsing; every received chunk is forwarded downstream immediately and
   unmodified — never delay or reorder emission.
3. Export: `/metrics` gains `axiom_cost_usd_total`, `axiom_cost_uncached_usd_total`
   (counterfactual), `axiom_cache_read_tokens_total`, `axiom_cache_write_tokens_total`,
   `axiom_uncached_input_tokens_total`. The `axiom_status` MCP tool output gains a
   `cost` section: session USD, counterfactual USD, cache hit-rate
   (`read/(read+write+uncached)`), and an `estimated` label when the price table guessed.
4. On session drop, log a one-line dollar receipt (extend the existing byte receipt).

**Not in scope:** changing any request. This step is read-only w.r.t. traffic.

**Acceptance.**
- Unit: `turn_cost` on fixture usage JSONs (all 4 fields; fields absent; streaming pair)
  produces exact USD (assert to 1e-9).
- Integration (`tests/`): mock upstream returning a usage block → `/metrics` shows the
  expected totals; hit-rate computed correctly.
- `axiom_status` shows the cost section (extend its existing test).
- No new writes into the outbound request body anywhere in this diff.

**Abort criteria.** If streaming usage proves unparseable without buffering, ship
non-streaming-only telemetry, expose `streaming_usage: unavailable` in `axiom_status`,
and file a follow-up — never buffer SSE.

---

## S1 — Cache-safety hardening (stop breaking the user's cache)

**Context brief.** Axiom's messages-path compression (`anthropic_forwarder.rs`,
`context_compressor.rs`) rewrites heavy message content into TTT fingerprints that can
change between turns → byte-instability → the client's `cache_control` prefix misses.
Simulation says this costs more than it saves whenever the client caches (Claude Code
always does). Anthropic renders `tools → system → messages`; any byte change at or
before a breakpoint invalidates everything after it.

**Task.**
1. `fn request_uses_cache(body: &Value) -> bool`: any `cache_control` key anywhere in
   the request JSON, or the top-level automatic-caching field.
2. When `AXIOM_CACHE_SAFE=1` (default) and the request uses cache:
   - Compression may only touch content **strictly after the last `cache_control`
     breakpoint** (the mutable tail). Content at/before any breakpoint is byte-frozen.
   - **Determinism:** if a message block was ever transformed and sent, record
     (session store: SHA-256(original) → transformed string) and reuse verbatim on every
     later turn. Same input bytes in → same output bytes out, for the session's life.
3. When the request does not use cache: existing behavior unchanged.
4. Decision telemetry: debug log line per request:
   `cache_safe: frozen_blocks=N mutable_blocks=M compressed=K`.

**Acceptance.**
- Byte-stability test: two consecutive turns (turn 2 = turn 1 + one appended message)
  through the transform with heavy content in turn 1 → the serialized bytes of turn 1's
  portion are IDENTICAL across both requests.
- Breakpoint-respect test: `cache_control` on message index 3, heavy content at index 2
  → untouched; heavy content at index 5 → may transform.
- Existing compression tests pass with `AXIOM_CACHE_SAFE=0`.

**Abort criteria.** If the transform cannot be made deterministic (TTT state evolves),
freeze-on-first-send (record the transformed string) is the REQUIRED mechanism — do not
weaken to "usually stable".

---

## S2 — L2 store + stubs + expand wiring

**Context brief.** CVM needs a local backing store: full-fidelity content that was
digested out of the API transcript, recoverable on demand. The repo already has a
memory store (`src/memory_store.rs`), an expansion endpoint (`POST /v1/expand`), and the
MCP tool `axiom_expand` that HTTP-calls it (`mcp_stdio.rs::expand_symbol_blocking`).

**Task.**
1. New module `src/cvm_store.rs`: content-addressed store
   `put(session_id, kind, original_text) -> PageId` (PageId = first 16 hex chars of
   SHA-256(text)), `get(session_id, page_id) -> Option<String>`; persisted at
   `checkpoints/cvm/<session_id>.jsonl`, append-only JSONL rows
   `{"page_id":"a1b2c3d4e5f60718","created_unix_s":1783990000,"kind":"tool_result","bytes":64213,"text":"..."}`;
   in-memory index built on open; per-session cap 64 MiB (drop oldest rows on rewrite,
   log evictions).
2. Canonical stub format (single line; must survive model round-trips verbatim):
   `[AXIOM-PAGE <page_id> <orig_tokens>tok <kind>] <first 120 chars of original>...`
   with `kind ∈ {tool_result, file, output}`. Builder + parser + round-trip tests in
   `cvm_store.rs`.
3. Extend `/v1/expand`: when `symbol` matches an `AXIOM-PAGE` id for the session, return
   the stored page (same response shape as symbol expansion). Mention page ids in
   `axiom_expand`'s tool description (tool count is unchanged).
4. Session cleanup: when the proxy drops a session, delete its CVM file unless
   `AXIOM_CVM_RETAIN=1`.

**Acceptance.** put/get round-trip; stub build/parse round-trip incl. unicode; expand-by-
page-id integration test through the HTTP route; cap-eviction test; cleanup test.

---

## S3 — Digest admission control (the big lever)

**Context brief.** Prevention beats compression: heavy tool results should never enter
the (expensive, cached-forever) transcript at full size. On arrival — in the **newest
turn only**, which is by definition after every cache breakpoint and not yet cached —
replace heavy content with digest + stub; full text goes to the L2 store (S2). Cache-safe
by construction (S1 froze the prefix) and needs no determinism trick: the digest is
created exactly once and then it IS the history. Simulation: 24% cost reduction alone;
net-negative if fault rate ≥ ~10%, hence default-off until S5 passes.

**Task.**
1. Digest backend trait in new `src/digest.rs`:
   `trait Digestor { fn digest(&self, text: &str, budget_tokens: usize) -> String; fn name(&self) -> &'static str; }`
   - `SkeletonDigestor` (default): reuse `src/skeleton.rs` (code-aware,
     signature-preserving, prose-fallback) targeting `budget = orig_tokens × 0.15`.
     Zero API cost, deterministic, no auth concerns.
   - `HaikuDigestor` (only when `AXIOM_CVM_DIGEST=haiku`): calls Anthropic Haiku 4.5
     **re-using the client's own auth headers from the current request**. Constraints:
     headers never persisted; 10 s timeout; any error → fall back to `SkeletonDigestor`.
     Flag docs must say this bills the user's account (~$0.02 per heavy event) — that is
     why it is opt-in.
2. Hook in the messages path (after S1's cache-safe gate): for each `tool_result` block
   in the newest turn whose token estimate ≥ `AXIOM_CVM_DIGEST_THRESHOLD_TOKENS`:
   `page_id = cvm_store.put(...)`; replace the content with:
   `<stub line>\n<digest text>\n[AXIOM-PAGE-END expand with axiom_expand("<page_id>")]`.
3. In-band honesty: the stub tells the model the content was digested and how to recover
   it. No silent elision, ever.
4. Telemetry (uses S0): digested block count, bytes in/out, and **faults** — every
   `/v1/expand` call that resolves a page id appends
   `{"session":"s","page_id":"p","turns_since_digest":3}` to
   `checkpoints/cvm/faults.jsonl` (this is S7's training signal).
5. Latency budget: skeleton digest of a 16K-token input completes < 500 ms on CPU (it is
   pattern-based, not neural; if slower, cap the input scan and truncate-digest).

**Acceptance.**
- Unit: threshold respected; digest ≤ 20% of original tokens; stub parseable; page
  recoverable via expand; fault row written.
- Integration: full proxy turn with a 16K-token tool_result → upstream mock receives
  digest+stub, not the original; `/v1/expand` returns the original.
- Composition with S1: with `cache_control` present, digestion touches only the newest
  turn (test).
- Flags off ⇒ bytes pass through unchanged (test).

**Abort criteria.** If `skeleton.rs` output on non-code prose is unusable (manually
check 3 real tool outputs), digest only code-like content (heuristic: >30% of lines
match code patterns) and pass prose through untouched. Do not invent a new summarizer
inside this step.

---

## S4 — Prefix diet, dedup tier (parallel with S1–S3)

**Context brief.** The 80K-token fixed prefix is paid on every cache write (1.25×) and
read (0.1×) forever. Observed on this machine: Claude Code injects CLAUDE.md/rules
content **duplicated** (identical file content appears more than once in system
context), plus large plugin/skill catalogs. Lossless-first diet: deterministic
deduplication of byte-identical repeated blocks, replacing later copies with a one-line
reference marker. As a pure function of the request bytes it is byte-stable across turns
→ cache-safe. NOTE: the simulated 23% came from a 40% *lossy* diet; this step ships only
the lossless subset and measures honestly — the lossy tier is a separate maintainer
decision recorded at the end of this step.

**Task.**
1. New `src/prefix_diet.rs`: split system text blocks on double-newline/markdown-heading
   boundaries; find repeated blocks ≥ 400 bytes occurring ≥ 2 times; replace occurrences
   after the first with `[AXIOM-DEDUP: identical to an earlier block in this prompt]`.
   Pure function: stateless, no session store, `diet(diet(x)) == diet(x)`.
2. Gate: `AXIOM_PREFIX_DEDUP=1` only (default 0 until S5) and only when
   `request_uses_cache` (S1). Because it is a pure function applied identically every
   turn, the output prefix is byte-stable — apply on every request when enabled.
3. Never dedup across `cache_control` tiers: only dedup within a single system text
   block's tier (simplest safe rule).
4. Telemetry: tokens removed per request into S0's counterfactual; debug endpoint
   `GET /v1/prefix-diet/report` returning
   `{"original_tokens":N,"dedup_tokens":M,"blocks_deduped":K}` for the session's last
   request.
5. Measure on THIS machine's real Claude Code traffic and put the real numbers in the
   PR description.

**Acceptance.** Purity + idempotency property tests; dedup correctness on a sanitized
fixture captured from real Claude Code system content; tier-boundary test; measured
report attached to the PR.

**Abort criteria.** If real measured dedup gain < 5% of prefix tokens, mark this step
DONE-BUT-WEAK here (strike-through + annotation), leave the default off, and record that
only the lossy tier (trimming tool descriptions — eval-gated, maintainer sign-off) can
reach the simulated 23%.

---

### ~~S4 status: DONE-BUT-WEAK~~ (2026-07-10, PR #109)

Implemented and tested (`prefix_diet.rs`, gated `AXIOM_PREFIX_DEDUP=1`, default **0**,
unchanged). Real measurement on this machine's actual `~/.claude` rule set (`CLAUDE.md` +
all 29 `rules/**/*.md` files, 59,349 bytes / 6,871 approx. tokens, the same file set
rendered verbatim into this very session's system prompt): **0.00% gain** — this specific
30-file set contains no byte-identical repeated blocks ≥ 400 bytes; every file is
distinct content (English rules, Chinese translations, and web-specific overrides all
differ). Below the 5% bar → abort criteria triggered, default stays off.

The mechanism itself is verified correct and effective when duplication is actually
present: a constructed scenario re-injecting `CLAUDE.md` a second time (simulating the
historically-observed "file included twice" pattern this step was designed for)
measured **32.21% token reduction** (2,201 → 1,492 tokens, 4 blocks deduped) on the same
real file content. The gap between 0% (this machine, today, this file set) and 32%
(when duplication exists) confirms the lossless dedup tier is sound but its real-world
yield is entirely conditional on Claude Code/the user's config actually duplicating
content — not guaranteed, and not present in the current baseline. Only the lossy tier
(trimming tool descriptions, eval-gated, maintainer sign-off) can reach the simulated
23% unconditionally.

---

## S5 — Behavior eval gate (blocks all default flips)

**Context brief.** No simulation can answer: do digests/stubs/dedup markers confuse the
model? This step builds the harness that answers it with live traffic and is the ONLY
authority for flipping `AXIOM_CVM_DIGEST` → `skeleton` and `AXIOM_PREFIX_DEDUP` → 1.
Use the strongest available model to design/refine the tasks; the harness itself runs
cheap (Haiku).

**Task.**
1. `scripts/cvm_eval.sh` + `axiom_engine_rs/tests/cvm_eval.rs` (ignored by default,
   gated behind `--features live-eval`): drive 12 scripted agentic tasks through the
   local proxy twice (flags off / flags on) using headless `claude -p` with
   `--model claude-haiku-4-5`, each task requiring ≥ 1 heavy file read from this repo
   (e.g. "what does the GET /v1/responses handler return for a non-websocket request?").
2. Score per task: (a) correctness — each task has a grep-able expected fact;
   (b) fault rate = expand calls / digested pages; (c) $ per task from S0 telemetry.
3. Pass bar (hard asserts in the script): correctness parity (flags-on ≥ flags-off − 1
   task), fault rate ≤ 5%, flags-on cost strictly lower.
4. Write a markdown report to `bench/cvm/RESULTS-<date>.md`, committed with the run.
5. On pass: the same PR flips the two defaults and updates README (which must stay
   honest about what is measured vs simulated).

**Acceptance.** Harness runs end-to-end on this machine; report has real numbers;
defaults flipped only on pass. On FAIL: file per-task issues, leave defaults off, mark
S3/S4 "shipped, gated-off" here.

---

## S6 — Actuarial keepalive (opt-in, security-sensitive; parallel)

**Context brief.** ~15% of inter-turn gaps exceed the 5-min TTL → full cache re-write
(1.25× on ~160K tokens ≈ $0.60) where a ping would cost ~$0.05 (0.1× read). Recent
Anthropic docs describe `max_tokens: 0` prefill pings (rejected combos: stream,
thinking, forced tool_choice); older API behavior required `max_tokens >= 1`. Sources
conflict, so the implementation must handle BOTH (see task 2). Pinging replays the
client's auth headers autonomously: a security decision only the user can make, hence
default-off forever.

**Task.**
1. `AXIOM_KEEPALIVE=1` (default 0, never auto-flipped). Boot banner when enabled:
   "keepalive re-uses your API credentials for cost-saving cache pings".
2. Per session: hold the last request's auth + relay headers **in memory only**; a tokio
   timer fires at TTL−30 s after last activity. **Ping body, precisely:** take the last
   real request JSON; keep `model`, `tools`, `system`, and `messages` truncated so the
   last retained content block is the last block carrying `cache_control` (drop
   everything after it); append one placeholder user message
   `{"role":"user","content":"."}`; remove `stream`, `thinking`, `tool_choice`,
   `metadata`; set `max_tokens: 0`. If the API returns 400 mentioning `max_tokens`,
   retry once with `max_tokens: 1` and remember the working value for the session; any
   other 4xx → disable keepalive for this session (log once). Stop after
   `ceil(horizon/TTL)` pings (`AXIOM_KEEPALIVE_HORIZON_S`, default 1800).
3. Actuarial gate: reuse `src/belief.rs::BetaBelief` per session over "another request
   arrived within the horizon after a gap"; ping only while
   `belief.mean() × 1.25 > 0.1 × pings_planned` (derivation in a code comment; the
   prefix-token factor cancels).
4. Telemetry: pings sent, estimated $ saved (counterfactual re-write minus ping cost)
   into S0's ledger, labeled `estimated`.

**Acceptance.** Unit tests for the inequality + belief update; integration with mock
upstream asserting ping shape (`max_tokens:0`, no stream/thinking/tool_choice); the
header-holder type implements neither Serialize nor value-revealing Debug and lives only
in the in-memory session map (grep test); disabled ⇒ zero timers.

**Abort criteria.** If pings get 401 (anti-replay OAuth), log once per session and
disable keepalive for that session — never retry-spam an auth endpoint.

---

## S7 — Speculative prefetch (phase 2; needs S3 fault data + S5 harness)

**Context brief.** The fault hedge: at 10% fault rate digestion is net-negative, but 80%
prefetch accuracy restores the economics ($7.91 → $4.37 in sim). S3 logs every fault in
`checkpoints/cvm/faults.jsonl` — the first real training objective for the predictive
engine (`state_predictor.rs`/`predictive_tools.rs`, currently honest-but-untrained).
Strongest model tier for this step.

**Task (gated outline — refine against real fault data).**
1. Collect ≥ 2 weeks of fault logs. If total faults < 50: STOP — fault rate is already
   low, prefetch is unnecessary; record the finding here and close the step as a
   success.
2. Heuristic baseline before any learning: prefetch pages whose stub text was mentioned
   in the model's previous output, plus pages originating from files edited this
   session. Measure recall against the logged faults. Recall ≥ 70% → ship the heuristic
   behind `AXIOM_CVM_PREFETCH=1` after an S5-style eval, and skip learning entirely.
3. Only if the heuristic falls short: train the state-predictor head on
   (session embedding → faulted page) pairs; wire as a prefetch ranker; publish honest
   precision/recall in `bench/cvm/`.
4. Prefetched pages are appended as new content blocks in the mutable tail — never
   inserted into the frozen prefix.

**Acceptance.** A decision recorded here with real numbers at each gate; whichever
branch ships, measure faults/session for ≥ 1 week before vs after.

---

## S8 — Rollout

**Task.** README gains a CVM section (each flag, and **measured** numbers from S5/S0 —
never the simulation numbers presented as fact); update `docs/CAPABILITIES.md`; bump
`Cargo.toml` to `0.4.0` (new user-facing subsystem) + `cargo update -p axiom_engine
--offline`; PR; after merge push tag `v0.4.0` → three publish workflows; verify with
`gh release view v0.4.0` and `gh run list`.

**Acceptance.** Release live on all three channels; README claims match S0 telemetry
semantics (dollar-true, `estimated` labeling preserved).

---

## Anti-pattern catalog (explicitly forbidden)

1. **Rewriting cached bytes to "save" tokens.** Pricing says this is a 10× loss. If a
   step seems to need it, the step is wrong.
2. **Scheduled eviction / deliberate cache breaks.** Simulated (scenario E): worse than
   doing nothing. Do not resurrect without new simulation evidence.
3. **Silent elision.** Every removed/digested byte leaves an in-band, model-visible,
   recoverable marker.
4. **Byte counters as success metrics.** Only S0's dollar ledger counts.
5. **Buffering the SSE stream** for any purpose (v0.2.1 regression history).
6. **Persisting client credentials.** Memory-only, opt-in, per-session (S6).
7. **Big-bang PRs.** One step, one PR; > ~600 non-test diff lines → split (protocol
   below).

## Mutation protocol

Any executor may SPLIT a step (acceptance criteria redistributed), INSERT a step (with
measurement/simulation justification), or ABANDON a step (with measured evidence). Every
mutation edits THIS file in the same PR as the work. Never delete history — strike
through and annotate.

## Verification of the whole (after S8)

Run one week with shipped defaults, then compute from S0 telemetry:
`1 − axiom_cost_usd_total / axiom_cost_uncached_usd_total`. Publish the real number in
the README. Target ≥ 70%. If reality lands materially below simulation, that is a
finding to publish, not to hide.
