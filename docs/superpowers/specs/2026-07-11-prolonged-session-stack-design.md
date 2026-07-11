# Prolonged-Session Stack (PSS v2) — Design

*Objective: reduce subscription quota burn by an additional ≥50% for prolonged
(100+ turn, multi-hour) Claude Code sessions, on top of the shipped CVM cost
stack (v0.4.0). Validated by Monte-Carlo simulation first, then a live
S5-style eval; defaults flip only on a live pass. If the live eval fails, the
result is published honestly and the design returns to brainstorming.*

Date: 2026-07-11. Builds on `docs/superpowers/plans/2026-07-10-cvm-cost-stack.md`.

## Simulation result (the gate that shaped this design)

A quota-units Monte-Carlo (`bench/cvm/pss_sim.py`, 4,000 sessions, mean 187
turns) over CVM-v1-shipped baseline produced (reduction vs baseline):

| Design | Sonnet-5 | Opus-4.x |
|---|---|---|
| R1+R2+R3 (first draft) | +31.1% | +48.0% |
| **+ L-A tool elision** | **+55.3%** | +65.8% |
| **+ L-A + L-B (PSS v2)** | **+56.8%** | **+66.4%** |
| PSS v2 without R1 | +56.3% | +56.6% |

Three findings drove the final design:
1. **L-A tool-schema elision is the anchor** — it alone clears 50% on Sonnet-5.
   The dominant prolonged-session cost is the ~80K static prefix (mostly
   tool/MCP schemas) re-read every turn; shrink it at the root and the rest
   compounds.
2. **R1 model-routing is dead weight on Sonnet** (+56.3% without it ≈ +56.8%
   with it) — Haiku's *own* prompt cache makes routing net-neutral once the
   prefix is cheap. R1 earns its complexity only on Opus/Fable tier (+~10
   points there), so it is **gated to high-tier requests only**.
3. **PSS v2 clears ≥56% on both tiers**, past the 50% goal, with a simpler
   design than the first draft.

The sim's L-A used a flat 30%-keep; the real gain depends on how safely tools
can be elided, which only the live eval settles.

## Objective function (differs from CVM v1)

CVM v1 optimized **API dollars**. PSS optimizes **subscription rate-limit
consumption** (Claude Pro/Max usage windows). Anthropic's usage accounting
weights tokens roughly like API prices — cache reads cheap, output tokens
heaviest, Opus ≈ 5x Sonnet per token — so the S0 dollar ledger is a usable
proxy, but PSS adds an explicit **quota-units ledger** so the two are never
conflated. Model routing, worthless for a flat-rate subscriber in dollars, is
the single biggest lever in quota units.

## What dominates prolonged-session cost (post-CVM-v1)

1. **Tier multiplier**: every turn — including mechanical "here's the tool
   output, continue" turns — is answered by the top-tier requested model
   (Fable/Opus/Sonnet), burning 5–10x the quota Haiku would.
2. **Growing-history re-reads**: turn N re-reads the whole cached transcript;
   0.1x per token but × N turns × growing length.
3. **Wasted cache breaks**: compaction / TTL expiry / session start re-write
   the entire prefix at premium anyway — and nothing exploits that already-paid
   moment.
4. **Output tokens**: heaviest weight, uncontrolled.

## Components

### R0 — Quota ledger + Fable-5/Opus-4.8 pricing (foundation)

- `cost_ledger.rs`: add `FABLE` price row (`fable-5` match) and an `opus-4-8`
  match if its pricing differs from the OPUS row; keep `estimated` semantics.
- New `quota_units` accounting alongside USD: same per-tier token weights,
  normalized so 1 unit = 1 Sonnet-5 uncached input token. Surfaces in
  `AwarenessState`/`CostSummary`, `/metrics`
  (`axiom_quota_units_total`, `axiom_quota_units_uncached_total`), and
  `/v1/awareness/:id`.
- Success: Fable traffic priced first-class (`estimated=false`); quota ledger
  visible end-to-end.

### L-A — Tool-schema elision (THE ANCHOR LEVER, break-window-only)

New `tool_elision.rs`, hooked into `compressed_messages_path`. Flag:
`AXIOM_TOOL_ELIDE` = `off` (default until eval passes) / `on`.
The ~80K prefix is dominated by `tools[]` schemas (many connected MCP
servers). Most turns use 2–3 tools. Present only the **working set** — tools
actually invoked in the session's recent history — plus a compact
`axiom_load_tools` affordance describing how to request the full set. Full
schemas for elided tools go to the L2 store (S2 machinery); a request for an
elided tool triggers exactly one reload.

**Cache-safety is the binding constraint.** The `tools[]` array renders BEFORE
system and messages in Anthropic's cache prefix, so changing it mid-session
would break the entire cache (violating S1's whole premise). Therefore
elision is **break-window-only**: the presented tool set is recomputed ONLY at
a natural cache break (session start, compaction, TTL expiry — detected via the
S1 memo, same trigger as R2), never mid-session. Between breaks the tool set
is frozen and byte-stable. A needed-but-elided tool causes at most one extra
break to reload, counted as an elision fault (telemetry, S7-style).

- Working-set policy: union of (tools invoked in the last K turns) ∪ (a small
  always-keep core: Read/Edit/Bash/Glob/Grep or their session analogues).
- Determinism: the elided set is a pure function of the frozen recent-history
  window, so the rendered prefix is identical across turns within a window.
- Success: measured prefix-token reduction reported per session; elision-fault
  rate (`AXIOM-LOAD-TOOLS` reload events / sessions) tracked.

### L-B — Local trivial-turn short-circuit

New path in `compressed_messages_path`. Flag: `AXIOM_LOCAL_TRIVIAL` = `off`
(default until eval passes) / `on`. A **confidently-trivial** turn — a
mechanical continuation with no heavy tool result, no error, and low local-TTT
surprisal (reusing `mean_surprisal` under the drift gate) — is answered by the
local engine with a minimal acknowledgment and NO upstream call. The turn
leaves the quota ledger entirely (strictly better than routing it to Haiku:
no second cache, no upstream tokens).
- Extreme conservatism: only fires when ALL of {mechanical, light, no error,
  surprisal below gate} hold. Fail-closed — any doubt forwards upstream.
- Honesty: the local answer is annotated in-band as locally generated; a
  `local_answered_turns` counter surfaces in telemetry. Never silent.
- Abort criterion: if the live eval shows local answers ever break task
  continuity, L-B ships gated-off and only L-A/R2/R3 carry the win (sim shows
  L-B is only ~1.5 points on Sonnet — expendable if risky).

### R1 — Difficulty-aware model routing (HIGH-TIER-GATED)

New `model_router.rs`, hooked into `compressed_messages_path` before forward.
Flag: `AXIOM_MODEL_ROUTE` = `off` (default) / `auto` / `on`. In `auto` (the
value the eval may flip to), routing engages ONLY when the requested model is
Opus- or Fable-tier — the simulation showed it is net-neutral on Sonnet
(Haiku's own cache cancels the win) and only worth its complexity on the 5x+
tiers. `on` forces it for any tier (for experimentation).
Routable-set guard: only reroute within Anthropic Claude models, and only
**downward** (never upgrade; never touch a request already targeting Haiku).

Per-turn classification, all free/local signals:
- **Structural**: newest turn is tool_result-dominated with no fresh user
  prose → mechanical; contains a real user question / no tool results → hard.
- **Local-TTT surprisal**: CE of the newest turn's text through the local
  engine (existing `mean_surprisal`); low surprisal = predictable →
  routable down. Threshold from the existing drift-gate calibration.
- **Sticky escalation**: any error signature in tool results, user correction,
  or a routed turn followed by a retry → force the requested model for the
  next `AXIOM_MODEL_ROUTE_COOLDOWN_TURNS` (default 3).
- **Fail-open**: ambiguous → requested model. Quality risk concentrates only
  in confidently-mechanical turns.

Honesty: routed responses gain an in-band annotation (`model` field already
reflects the answering model in the response; additionally log + count per
session: `routed_turns`, `routed_quota_saved_units`). No silent substitution
in telemetry.

Cache interaction: rewriting `model` does NOT invalidate Anthropic's prompt
cache prefix (cache is per-model!). **Critical caveat**: Anthropic caches are
model-specific, so a Haiku-routed turn re-writes its own Haiku-side cache.
The router must account for this: routing is only net-positive when
`haiku_write_cost(prefix) + haiku_turn < requested_read_cost(prefix) +
requested_turn`, amortized over consecutive routed turns. The router therefore
prefers **runs** of mechanical turns (hysteresis: once routed, stay routed
until a hard turn appears) rather than alternating, and the simulation must
model the dual-cache economics explicitly. If simulation shows alternation
kills the win, the router ships with run-length-aware gating.

### R2 — Free-window rebasing

When the S1 cache-safety memo shows the incoming prefix ≠ the last-sent prefix
(compaction happened, TTL expired, or new session), the client's cache is
already broken — the full prefix will be re-written at premium regardless.
In exactly that window, restructure deeply at zero marginal cache cost:
- Digest ALL heavy tool_result blocks older than the newest turn (not just
  newest — S3 only ever touched the newest) into stubs + L2 pages.
- Apply prefix dedup across the whole transcript (S4 machinery, which is
  byte-stable so the rebased prefix stays cache-safe going forward).
- Never triggered by the proxy itself — piggyback only. This respects the
  anti-pattern catalog (scheduled eviction stays forbidden).
Flag: `AXIOM_REBASE_ON_BREAK` = `off` default until eval passes.

### R3 — Adaptive cache TTL

Sessions with long thinking gaps lose the 5-minute cache repeatedly. Anthropic
offers 1-hour TTL at 2x write (vs 1.25x). Per-session choice using the gap
statistics the keepalive module's BetaBelief already learns: if observed
inter-turn gaps repeatedly exceed ~4 minutes, annotate outbound
`cache_control` with `ttl: "1h"` on the newest breakpoint. Pure economics:
one-time +0.75x write premium vs repeated full re-writes.
Flag: `AXIOM_ADAPTIVE_TTL` = `off` default until eval passes.

### R4 — Validation: simulation first (DONE), then live eval

1. Monte-Carlo (`bench/cvm/pss_sim.py`) — **complete**, results in the table
   above. Committed with the first implementation PR. Any future lever tweak
   re-runs it and updates the table.
2. Live eval: extend `scripts/cvm_eval.sh` with a third condition (PSS flags
   on) and longer multi-turn tasks (sequences of dependent questions in one
   session, not single-shot lookups) so elision/short-circuit actually engage.
   Report **quota units** (R0's ledger), not just dollars. Pass bar:
   correctness parity (same rule as S5), elision-fault + local-continuity
   faults ≤5%, and **quota units strictly lower with PSS on**, targeting ≥50%
   reduction on the long-session tasks. On pass: flip `AXIOM_TOOL_ELIDE`,
   `AXIOM_LOCAL_TRIVIAL`, `AXIOM_REBASE_ON_BREAK`, `AXIOM_ADAPTIVE_TTL`
   defaults, and set `AXIOM_MODEL_ROUTE=auto`. On fail: publish the numbers,
   leave defaults off, return to brainstorming (user-directed).

## Error handling

- L-A elision: a request for an elided tool → one break to reload the full set
  (or just the requested tool), counted as an elision fault; never a hard
  failure. Any elision computation error → present the full tool set (fail
  toward correctness).
- L-B local short-circuit: fail-closed — any uncertainty forwards upstream. A
  locally-answered turn that the client immediately retries/corrects →
  disable L-B for the next cooldown window and forward upstream.
- R1 router: any classification error → fail-open to requested model. Upstream
  4xx on a routed turn → retry once unrouted (mirrors the existing
  compression-fallback pattern), count a `route_fallback`.
- R2 rebase: any digest/dedup failure mid-rebase → forward the original bytes
  (never a half-rebased prefix); log once.
- R3 TTL: malformed/unsupported `ttl` field upstream → strip and retry once.

## Testing

Same regime as CVM v1: TDD per component, unit + integration (mock upstream)
suites per module, env-var race hygiene (async env locks, RAII guards),
`cargo test --lib` + `cargo clippy --lib --locked -- -D warnings` gates,
one PR per step, CodeRabbit triage, live eval gated behind `live-eval`
feature + explicit human run.

## Non-goals

- No output-token shaping in v1 (deferred; quality-risk against uncertain
  gain — revisit only if R1–R3 undershoot).
- No cross-provider routing (Anthropic models only).
- No proxy-initiated cache breaks, ever.

## Build order (one PR per step, TDD, CI + CodeRabbit gates)

- **P0** R0 quota ledger + Fable-5/Opus-4.8 pricing rows + committed
  `bench/cvm/pss_sim.py`. Foundation; no traffic change.
- **P1** L-A tool elision (the anchor). Depends on P0 (measure in quota units)
  and reuses S2's L2 store for elided schemas.
- **P2** R2 free-window rebasing + R3 adaptive TTL (both piggyback on the S1
  break-detection memo; grouped as they share that trigger).
- **P3** L-B local trivial short-circuit.
- **P4** R1 high-tier-gated model routing.
- **P5** Live eval harness extension (quota-units, long multi-turn tasks) +
  the deliberate human-run + default flips on pass. Mirrors S5.

## Success criteria

Simulation predicts ≥50% quota reduction on the long-session profile
(**achieved: +56.8% Sonnet-5, +66.4% Opus** with PSS v2). Live eval confirms
correctness parity, fault rate ≤5%, and a strict quota reduction with all PSS
flags on, ≥50% on long-session tasks. Defaults flip only on a live pass; on
fail, publish honestly and return to brainstorming.
