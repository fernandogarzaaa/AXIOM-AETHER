# Prolonged-Session Stack (PSS) — Design

*Objective: reduce subscription quota burn by an additional ≥50% for prolonged
(100+ turn, multi-hour) Claude Code sessions, on top of the shipped CVM cost
stack (v0.4.0). Validated by Monte-Carlo simulation first, then a live
S5-style eval; defaults flip only on a live pass. If the live eval fails, the
result is published honestly and the design returns to brainstorming.*

Date: 2026-07-11. Builds on `docs/superpowers/plans/2026-07-10-cvm-cost-stack.md`.

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

### R1 — Difficulty-aware model routing (the big lever)

New `model_router.rs`, hooked into `compressed_messages_path` before forward.
Flag: `AXIOM_MODEL_ROUTE` = `off` (default until eval passes) / `on`.
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

### R4 — Validation: simulation first, then live eval

1. Extend the Monte-Carlo (`cvm_sim.py` → committed as `bench/cvm/pss_sim.py`)
   to model 100–300-turn sessions in quota units with: per-model dual-cache
   economics, compaction events, gap distributions, mechanical/hard turn mix
   (measured proportions from real Claude Code session recordings where
   available; sensitivity-swept otherwise). Publish the predicted lever sizes
   in the PR. Abort any lever the simulation shows <5% or negative.
2. Live eval: extend `scripts/cvm_eval.sh` with a third condition (PSS flags
   on) and longer multi-turn tasks (sequences of dependent questions in one
   session, not single-shot lookups) so routing/rebase actually engage.
   Pass bar: correctness parity (same rule as S5), fault rate ≤5%, and
   **quota units strictly lower with PSS on**, targeting ≥50% reduction on
   the long-session tasks specifically. On pass: flip `AXIOM_MODEL_ROUTE`,
   `AXIOM_REBASE_ON_BREAK`, `AXIOM_ADAPTIVE_TTL` defaults. On fail: publish
   the numbers, leave defaults off, return to brainstorming (user-directed).

## Error handling

- Router: any classification error → fail-open to requested model. Upstream
  4xx on a routed turn → retry once unrouted (mirrors the existing
  compression-fallback pattern), count a `route_fallback`.
- Rebase: any digest/dedup failure mid-rebase → forward the original bytes
  (never a half-rebased prefix); log once.
- TTL: malformed/unsupported `ttl` field upstream → strip and retry once.

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

## Success criteria

Simulation predicts ≥50% quota reduction on the long-session profile with
R1+R2+R3 combined; live eval confirms correctness parity and a strict
quota reduction with all PSS flags on, ≥50% on long-session tasks.
