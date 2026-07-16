# Prolonged-Session Stack (PSS v2) Implementation Plan

> **Status (2026-07-16, COMPLETE):** P0–P5 all shipped. P0–P4 merged as #118/#120/#121/#122/#123; live-eval harness #124/#127; R2 break-detection root-cause fix #125 (the 2026-07-12 FAIL); **defaults flipped ON by explicit user decision** after the first valid measurement (2026-07-16: parity 12/13 = 12/13, 0% faults, 11.0% quota savings — L-A only could fire in that harness shape; the ≥50% target remains unproven on real long-session traffic). Opt out per-lever: `AXIOM_TOOL_DEFER/LOCAL_TRIVIAL/REBASE_ON_BREAK/ADAPTIVE_TTL=off`, `AXIOM_MODEL_ROUTE=off`. Real measurements in `bench/cvm/PSS-RESULTS-*.md`; README + docs/CAPABILITIES.md updated.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut subscription quota burn ≥50% for prolonged (100+ turn) Claude Code sessions, on top of the shipped CVM cost stack (v0.4.0), validated by a live eval before any default flips.

**Architecture:** Five levers hooked into the existing `/v1/messages` compression path (`server/routes_messages.rs::compressed_messages_path`), each behind an env flag defaulting off: L-A tool-schema elision (the anchor — shrink the ~80K static prefix at break windows only), L-B local trivial-turn short-circuit, R2 free-window rebasing, R3 adaptive cache TTL, and R1 high-tier-gated model routing. A new quota-units ledger (R0) measures the win. All reuse shipped machinery: S1's cache-break memo, S2's L2 store, S3's digestor, S4's dedup, `mean_surprisal`, and `BetaBelief`.

**Tech Stack:** Rust (axum 0.7, tokio, serde_json), candle for the local TTT engine, `cargo test`/`cargo clippy` gates, Python 3 for the Monte-Carlo, `claude -p` for the live eval.

## Global Constraints

- Branch → PR → CodeRabbit triage → squash-merge per step; branch name `pss/p<N>-<slug>`.
- TDD: failing test first, watch it fail for the right reason, then implement.
- CI gates (must pass locally before pushing): `cd axiom_engine_rs && cargo test --lib && cargo clippy --lib --locked -- -D warnings`.
- Every new behavior ships behind an env flag, default **off**, read live (not cached at boot) inside the request path — matches S1/S3/S4.
- Never introduce a UTF-8 BOM (regression test `no_source_file_has_utf8_bom` scans `src/**/*.rs`). ASCII punctuation in source.
- Never break Anthropic's cache mid-session: any change to `tools[]`, `system`, or a frozen message may only happen at a natural cache break (session start / compaction / TTL expiry), detected via the S1 memo. This is the binding constraint for L-A, R2, R3.
- Integration tests that mutate a process-global env var must serialize via a `tokio::sync::Mutex` env-lock + an RAII `EnvVarGuard` (the S1 race lesson).
- The live eval (P5) spends real Anthropic credits: gated behind the `live-eval` Cargo feature + `#[ignore]`, never run in CI, only by a deliberate human `scripts/cvm_eval.sh` invocation.
- Defaults flip ONLY on a live-eval pass (correctness parity, fault ≤5%, quota strictly lower, ≥50% on long-session tasks). On fail: publish numbers, leave off, return to brainstorming.

---

### Task P0: Quota-units ledger + Fable-5 / Opus-4.8 pricing

**Files:**
- Modify: `axiom_engine_rs/src/cost_ledger.rs` (add `FABLE` const, extend `for_model`, add `quota_units` fn)
- Modify: `axiom_engine_rs/src/session_awareness.rs` (add quota atomics + `CostSummary` fields, mirroring the S6 `keepalive_*` pattern)
- Modify: `axiom_engine_rs/src/server/routes_core.rs` (`export_metrics`: 1 new line)
- Test: inline `#[cfg(test)]` in `cost_ledger.rs`

**Interfaces:**
- Consumes: existing `PriceTable`, `TurnCost`, `turn_cost(model, usage) -> Option<TurnCost>`.
- Produces: `PriceTable::FABLE`; `fn quota_units(tc: &TurnCost, prices: &PriceTable) -> f64` (normalized so 1 unit = 1 Sonnet-5 uncached input token); `AwarenessState::record_turn_quota(&self, units: f64)`; `CostSummary.quota_units_total: f64`.

- [ ] **Step 1: Write the failing test** in `cost_ledger.rs` tests module:

```rust
#[test]
fn fable_5_is_priced_first_class_not_estimated() {
    let (p, estimated) = PriceTable::for_model("claude-fable-5");
    assert!(!estimated, "fable-5 must be a known model, not an estimate");
    assert_eq!(p.input_per_mtok, 10.00); // real Anthropic rate, verified 2026-07
    assert_eq!(p.output_per_mtok, 50.00);
}

#[test]
fn sonnet5_pricing_is_date_aware_across_the_sep_2026_change() {
    use chrono::TimeZone; // or std date shim per repo convention
    let before = PriceTable::for_model_at("claude-sonnet-5", date(2026, 8, 31));
    let after  = PriceTable::for_model_at("claude-sonnet-5", date(2026, 9, 1));
    assert_eq!(before.0.input_per_mtok, 2.00);
    assert_eq!(after.0.input_per_mtok, 3.00);
}

#[test]
fn quota_units_normalize_sonnet5_input_to_one() {
    let (prices, _) = PriceTable::for_model("claude-sonnet-5");
    let tc = TurnCost { uncached_in: 1_000_000, cache_write: 0, cache_read: 0,
                        output: 0, usd: 0.0, estimated: false };
    let u = quota_units(&tc, &prices);
    assert!((u - 1_000_000.0).abs() < 1e-6, "got {u}");
}

#[test]
fn quota_units_weight_output_heaviest() {
    let (prices, _) = PriceTable::for_model("claude-sonnet-5");
    let out = TurnCost { uncached_in: 0, cache_write: 0, cache_read: 0,
                         output: 1000, usd: 0.0, estimated: false };
    let inp = TurnCost { uncached_in: 1000, cache_write: 0, cache_read: 0,
                         output: 0, usd: 0.0, estimated: false };
    assert!(quota_units(&out, &prices) > quota_units(&inp, &prices));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib cost_ledger 2>&1 | tail -20`
Expected: FAIL — `for_model("claude-fable-5")` returns `estimated=true`; `quota_units` not defined.

- [ ] **Step 3: Implement.** Add `FABLE` with the **real** Anthropic-published rate (verified 2026-07 via deep-research: input $10, 5m-write $12.50, cache-read $1, output $50) and extend `for_model` (fable arm BEFORE the opus/sonnet arms):

```rust
const FABLE: PriceTable = PriceTable {
    input_per_mtok: 10.00, cache_write_5m_per_mtok: 12.50,
    cache_read_per_mtok: 1.00, output_per_mtok: 50.00,
};
// in for_model:
} else if m.contains("fable-5") || m.contains("mythos-5") {
    (Self::FABLE, false)
} else if m.contains("opus-4") {
    (Self::OPUS, false)
```

Also make Sonnet-5 **date-aware**: Anthropic raises Sonnet-5 pricing on 2026-09-01 (input $2→$3, 5m-write $2.50→$3.75, cache-read $0.20→$0.30, output $10→$15). Add a `SONNET5_POST_SEP2026` const and select it in `for_model` when the system clock is ≥ 2026-09-01 (UTC). Add a unit test that asserts the boundary both sides (mock the date via an injectable `now` param on a private `for_model_at(model, date)` so the public `for_model` stays a thin wrapper — keeps the test deterministic, no clock dependency in CI):

```rust
const SONNET5_POST_SEP2026: PriceTable = PriceTable {
    input_per_mtok: 3.00, cache_write_5m_per_mtok: 3.75,
    cache_read_per_mtok: 0.30, output_per_mtok: 15.00,
};
```

Module-level quota fn:

```rust
const SONNET5_INPUT_PER_MTOK: f64 = 2.00;
/// Quota units for a turn, normalized so 1 unit == 1 Sonnet-5 uncached input
/// token. Subscription usage windows weight tokens ~ like price, so reuse the
/// per-tier price ratios as quota weights against the Sonnet-5 input rate.
pub fn quota_units(tc: &TurnCost, prices: &PriceTable) -> f64 {
    let w = |per_mtok: f64| per_mtok / SONNET5_INPUT_PER_MTOK;
    tc.uncached_in as f64 * w(prices.input_per_mtok)
        + tc.cache_write as f64 * w(prices.cache_write_5m_per_mtok)
        + tc.cache_read as f64 * w(prices.cache_read_per_mtok)
        + tc.output as f64 * w(prices.output_per_mtok)
}
```

- [ ] **Step 4: Run to verify pass** — `cargo test --lib cost_ledger 2>&1 | tail -20` → PASS.

- [ ] **Step 5: Wire the session ledger + /metrics.** In `session_awareness.rs` add `quota_units_micros: AtomicUsize` (units×1e6 as integer, mirroring `cost_usd_micros`), `record_turn_quota(&self, units: f64)`, `CostSummary.quota_units_total: f64`, populate in `cost_summary()` — following the exact S6 `keepalive_*` pattern (field → Default init → CostSummary field → cost_summary() line → recorder). Call `record_turn_quota` beside the existing `record_turn_cost` in `routes_messages.rs`. Add to `routes_core.rs::export_metrics` an `axiom_quota_units_total` lifetime counter (mirror `LIFETIME_COST_USD_MICROS`).

- [ ] **Step 6: Verify + clippy + commit**

Run: `cargo test --lib && cargo clippy --lib --locked -- -D warnings`
```bash
git add -A && git commit -m "feat(pss): P0 quota-units ledger + Fable-5/Opus-4.8 pricing"
```

---

### Task P1: L-A tool deferral via native `defer_loading` (the anchor lever)

**Design note (from deep-research 2026-07-11):** Anthropic's tool-use docs describe a native mechanism that supersedes break-window elision. Marking a tool with `defer_loading: true` keeps it OUT of the cached `tools[]` prefix; when the model needs it, tool-search appends its schema as a `tool_reference` block in `messages` (after the breakpoint), so **the prefix cache is never broken, even mid-session.** This is strictly better than swapping the tools array (which invalidates the whole cache): higher gain, cache-safe by construction, no break-window constraint. `tools[]` order/contents must still be byte-stable, so the proxy only ever *sets `defer_loading`*, never adds/removes/reorders tools.

**Files:**
- Create: `axiom_engine_rs/src/tool_defer.rs`
- Modify: `axiom_engine_rs/src/lib.rs` (`pub mod tool_defer;`, alphabetical)
- Modify: `axiom_engine_rs/src/server/routes_messages.rs` (`compressed_messages_path`: apply before forward)
- Test: `axiom_engine_rs/tests/tool_defer_proxy.rs` + inline

**Interfaces:**
- Consumes: `body["tools"]` (array of `{"name":..}`), `body["messages"]` (for the working set).
- Produces: `tool_defer::working_set(messages: &[Value], recent_k: usize) -> HashSet<String>`; `tool_defer::mark_deferred(tools: &[Value], keep: &HashSet<String>) -> (Vec<Value>, usize)` returning `(tools_with_defer_loading_set, deferred_count)` — the array is the SAME tools in the SAME order, only with `"defer_loading": true` added to each tool not in `keep` (never removed/reordered, so the byte-stability rule holds while still shrinking the cached prefix, since deferred tools drop out of it).

- [ ] **Step 1: Failing unit test** in `tool_defer.rs`:

```rust
#[test]
fn working_set_keeps_recently_invoked_tools_plus_core() {
    let messages = vec![json!({"role":"assistant","content":[
        {"type":"tool_use","name":"WebFetch","id":"t1","input":{}}]})];
    let ws = working_set(&messages, 8);
    assert!(ws.contains("WebFetch"));
    assert!(ws.contains("Read"), "core tool always kept");
}

#[test]
fn mark_deferred_sets_flag_on_unused_tools_preserving_order() {
    let tools = vec![json!({"name":"Read"}), json!({"name":"WebFetch"}),
                     json!({"name":"ObscureTool"})];
    let mut keep = std::collections::HashSet::new();
    keep.insert("Read".to_string());
    let (out, deferred) = mark_deferred(&tools, &keep);
    assert_eq!(deferred, 2);
    // order + count unchanged (byte-stability rule): still 3 tools, same names, same order
    assert_eq!(out.len(), 3);
    assert_eq!(out[0]["name"], json!("Read"));
    assert_eq!(out[2]["name"], json!("ObscureTool"));
    // kept tool: no defer_loading; unused tools: defer_loading true
    assert!(out[0].get("defer_loading").is_none());
    assert_eq!(out[1]["defer_loading"], json!(true));
    assert_eq!(out[2]["defer_loading"], json!(true));
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test --lib tool_defer 2>&1 | tail -20` → FAIL.

- [ ] **Step 3: Implement `tool_defer.rs`.** `working_set` walks messages for `tool_use` blocks' `name` within the last `recent_k` turns, unions `const CORE: [&str; 6] = ["Read","Edit","Write","Bash","Glob","Grep"]`. `mark_deferred` maps over `tools` preserving order: for each tool whose `name` is NOT in `keep`, clone it and insert `"defer_loading": true`; kept tools pass through unchanged. Returns `(out, deferred_count)`.

- [ ] **Step 4: Run to verify pass** — `cargo test --lib tool_defer` → PASS.

- [ ] **Step 5: Wire** in `compressed_messages_path`, gated `AXIOM_TOOL_DEFER == Ok("on")`: compute `working_set` from the full message history, `mark_deferred(body["tools"], keep)`, set `outbound["tools"]` to the result. Because only `defer_loading` flags are added (order/names identical), and deferred tools leave the cached prefix, the cached prefix SHRINKS while staying byte-stable turn-over-turn. Record the deferred-prefix-token delta via P0's quota ledger. No break-window logic needed — `defer_loading` is inherently cache-safe.

- [ ] **Step 6: Integration test** `tests/tool_defer_proxy.rs` (mirror `digest_proxy.rs`'s mock upstream + `Capture` + env-lock/`EnvVarGuard`): (a) `AXIOM_TOOL_DEFER=on`, 10 tools with 2 recently used → upstream receives 10 tools but 8 carry `defer_loading:true`; (b) two consecutive turns with the same working set → outbound `tools[]` byte-identical both (cache-safe); (c) flag unset → tools unchanged (no `defer_loading` added).

- [ ] **Step 7: Verify + clippy + commit**

Run: `cargo test --lib && cargo test --test tool_defer_proxy && cargo clippy --lib --locked -- -D warnings && cargo clippy --test tool_defer_proxy --locked -- -D warnings`
```bash
git add -A && git commit -m "feat(pss): P1 L-A tool deferral via native defer_loading (cache-safe)"
```

---

### Task P2: R2 free-window rebasing + R3 adaptive cache TTL

**Files:**
- Create: `axiom_engine_rs/src/rebase.rs`
- Modify: `axiom_engine_rs/src/lib.rs` (`pub mod rebase;`)
- Modify: `axiom_engine_rs/src/server/routes_messages.rs`
- Modify: `axiom_engine_rs/src/server/prelude_state.rs` (per-session gap tracker: `RwLock<HashMap<String,(u64,u32)>>` = (last_ts, long_gap_count))
- Test: `axiom_engine_rs/tests/rebase_proxy.rs` + inline

**Interfaces:**
- Consumes: S3 `digest::SkeletonDigestor` + `DEFAULT_DIGEST_THRESHOLD_TOKENS`, `cvm_store::{CvmStore, build_stub}`, S4 `prefix_diet::diet_system_field`, S1 break detection.
- Produces: `rebase::rebase_transcript(messages: &[Value], store: &CvmStore, session_id: &str) -> Vec<Value>`; `rebase::choose_ttl(long_gap_count: u32, threshold: u32) -> Option<&'static str>`.

- [ ] **Step 1: Failing test** in `rebase.rs`:

```rust
#[test]
fn choose_ttl_picks_1h_only_after_repeated_long_gaps() {
    assert_eq!(choose_ttl(0, 3), None);
    assert_eq!(choose_ttl(2, 3), None);
    assert_eq!(choose_ttl(3, 3), Some("1h"));
}

#[test]
fn rebase_digests_all_old_heavy_but_never_the_newest_turn() {
    let dir = std::env::temp_dir().join(format!("pss-rebase-{}", std::process::id()));
    let store = CvmStore::open(&dir).unwrap();
    let big = "x ".repeat(9000); // > digest threshold
    let messages = vec![
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":big}]}),
        json!({"role":"user","content":"newest turn stays whole"}),
    ];
    let out = rebase_transcript(&messages, &store, "s1");
    assert!(out[0].to_string().contains("AXIOM-PAGE"), "old heavy digested");
    assert!(out[1].to_string().contains("newest turn stays whole"));
    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test --lib rebase 2>&1 | tail -20` → FAIL.

- [ ] **Step 3: Implement `rebase.rs`.** `choose_ttl` = the threshold above. `rebase_transcript` iterates all messages except the last; for each `tool_result` block whose whitespace-token estimate ≥ `DEFAULT_DIGEST_THRESHOLD_TOKENS`, `store.put(session_id,"tool_result",text)` + replace with `build_stub(...)` + `SkeletonDigestor.digest(...)` (exactly S3's mechanism, applied to ALL old turns); then run each message's system-shaped text through `diet_system_field` for dedup.

- [ ] **Step 4: Run to verify pass** — `cargo test --lib rebase` → PASS.

- [ ] **Step 5: Wire** into `compressed_messages_path`, ONLY when a break is detected (S1 memo mismatch), never proxy-initiated. `AXIOM_REBASE_ON_BREAK=on` → replace `mutable_messages` with `rebase_transcript(...)` before the existing S3 newest-turn digest. `AXIOM_ADAPTIVE_TTL=on` → update the per-session gap tracker from inter-turn timestamps (a gap > 240s increments `long_gap_count`); if `choose_ttl(count, 3)` is `Some(ttl)`, set `["cache_control"]["ttl"] = ttl` on the newest breakpoint block in `outbound`.

- [ ] **Step 6: Integration test** `tests/rebase_proxy.rs`: (a) a simulated break (2nd request, changed prefix) with `AXIOM_REBASE_ON_BREAK=on` → older heavy blocks arrive upstream as stubs; (b) no break → transcript untouched; (c) repeated long gaps with `AXIOM_ADAPTIVE_TTL=on` → outbound newest `cache_control` gains `"ttl":"1h"`.

- [ ] **Step 7: Verify + clippy + commit**
```bash
git add -A && git commit -m "feat(pss): P2 R2 free-window rebasing + R3 adaptive cache TTL"
```

---

### Task P3: L-B local trivial-turn short-circuit

**Files:**
- Create: `axiom_engine_rs/src/local_trivial.rs`
- Modify: `axiom_engine_rs/src/lib.rs` (`pub mod local_trivial;`)
- Modify: `axiom_engine_rs/src/server/routes_messages.rs`
- Modify: `axiom_engine_rs/src/session_awareness.rs` (`local_answered_turns` counter)
- Test: `axiom_engine_rs/tests/local_trivial_proxy.rs` + inline

**Interfaces:**
- Consumes: existing `mean_surprisal(state, text) -> Option<f32>`, drift gate (`AXIOM_DRIFT_THRESHOLD`, default 7.03).
- Produces: `local_trivial::has_error_signature(text: &str) -> bool` (shared with P4); `local_trivial::is_trivial(messages: &[Value], surprisal: Option<f32>, gate: f32) -> bool`; `local_trivial::local_ack() -> Value`.

- [ ] **Step 1: Failing test** in `local_trivial.rs`:

```rust
#[test]
fn is_trivial_true_for_clean_mechanical_low_surprisal_turn() {
    let m = vec![json!({"role":"user","content":[
        {"type":"tool_result","tool_use_id":"x","content":"ok, exit 0"}]})];
    assert!(is_trivial(&m, Some(1.0), 7.03));
}
#[test]
fn is_trivial_false_when_surprisal_above_gate() {
    let m = vec![json!({"role":"user","content":[
        {"type":"tool_result","tool_use_id":"x","content":"ok"}]})];
    assert!(!is_trivial(&m, Some(9.0), 7.03));
}
#[test]
fn is_trivial_false_on_error_signature() {
    let m = vec![json!({"role":"user","content":[
        {"type":"tool_result","tool_use_id":"x","content":"Error: panicked"}]})];
    assert!(!is_trivial(&m, Some(1.0), 7.03));
}
#[test]
fn has_error_signature_is_case_insensitive() {
    assert!(has_error_signature("Traceback (most recent call last)"));
    assert!(has_error_signature("FAILED"));
    assert!(!has_error_signature("all green"));
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test --lib local_trivial 2>&1 | tail -20` → FAIL.

- [ ] **Step 3: Implement.** `has_error_signature` = case-insensitive match of any of `["error","panic","failed","exception","traceback"]`. `is_trivial` is fail-closed: returns `false` unless ALL hold — newest turn is tool_result-only with no fresh user prose, `!has_error_signature(text)`, no heavy content, and `surprisal.map_or(false,|s| s < gate)`. `local_ack` returns an Anthropic-shaped message response with `"model":"axiom-local"`, `stop_reason":"end_turn"`, and content noting it was locally generated.

- [ ] **Step 4: Run to verify pass** — `cargo test --lib local_trivial` → PASS.

- [ ] **Step 5: Wire** — gated `AXIOM_LOCAL_TRIVIAL=on`; when `is_trivial`, return `local_ack()` WITHOUT calling the forwarder, increment `local_answered_turns`, append the exchange to session history. Client retry of a locally-answered turn (identical inbound within a short window) → disable L-B for a cooldown, forward upstream.

- [ ] **Step 6: Integration test** `tests/local_trivial_proxy.rs`: (a) `AXIOM_LOCAL_TRIVIAL=on` + clean mechanical turn → upstream receives ZERO requests, client gets 200 `model=axiom-local`; (b) error-bearing tool_result → upstream IS called; (c) flag off → always forwarded.

- [ ] **Step 7: Verify + clippy + commit**
```bash
git add -A && git commit -m "feat(pss): P3 L-B local trivial-turn short-circuit"
```

---

### Task P4: R1 high-tier-gated model routing

**Rationale (deep-research 2026-07-11):** subscription metering uses per-bucket utilization, and Opus has its OWN weekly bucket (`seven_day_opus`) that is far scarcer than Sonnet's (~15–35 vs ~140–280 hrs on Max 5x). Routing an Opus/Fable turn to Haiku therefore relieves the tightest bucket — value beyond raw token weight — which is exactly why routing is gated to high tiers and left off for Sonnet. Note: Anthropic's own guidance ("spawn a separate call rather than switching the main loop's model") endorses this pattern, and caches are model-scoped so a Haiku turn does not destroy the top-tier cache (it just can't read it).

**Files:**
- Create: `axiom_engine_rs/src/model_router.rs`
- Modify: `axiom_engine_rs/src/lib.rs` (`pub mod model_router;`)
- Modify: `axiom_engine_rs/src/server/routes_messages.rs`
- Modify: `axiom_engine_rs/src/session_awareness.rs` (`routed_turns`, `route_fallbacks`, `routed_quota_saved_units` — the quota units saved by each downgrade, = `quota_units(requested_tier) - quota_units(haiku)` for the turn; surfaced in `CostSummary` and the eval report so the live gate can attribute R1's contribution)
- Test: `axiom_engine_rs/tests/model_router_proxy.rs` + inline

**Interfaces:**
- Consumes: `cost_ledger::PriceTable::for_model`, `local_trivial::has_error_signature` (from P3), `body["model"]`.
- Produces: `model_router::is_high_tier(model: &str) -> bool`; `model_router::route(model: &str, mechanical: bool, cooldown: u32, mode: &str) -> Option<&'static str>`.

- [ ] **Step 1: Failing test** in `model_router.rs`:

```rust
#[test]
fn auto_mode_routes_only_high_tier_mechanical_turns() {
    assert_eq!(route("claude-opus-4-8", true, 0, "auto"), Some("claude-haiku-4-5"));
    assert_eq!(route("claude-sonnet-5", true, 0, "auto"), None);
    assert_eq!(route("claude-fable-5", true, 0, "auto"), Some("claude-haiku-4-5"));
    assert_eq!(route("claude-opus-4-8", false, 0, "auto"), None); // hard turn
    assert_eq!(route("claude-opus-4-8", true, 2, "auto"), None);  // cooldown
    assert_eq!(route("claude-haiku-4-5", true, 0, "auto"), None); // never touch haiku
}

#[test]
fn is_high_tier_matches_opus_and_fable_not_sonnet() {
    assert!(is_high_tier("claude-opus-4-8"));
    assert!(is_high_tier("claude-fable-5"));
    assert!(!is_high_tier("claude-sonnet-5"));
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test --lib model_router 2>&1 | tail -20` → FAIL.

- [ ] **Step 3: Implement.** First a Claude-provider guard: `is_claude(m) = m.contains("claude") || m.starts_with("opus-") || m.starts_with("sonnet-") || m.starts_with("haiku-") || m.starts_with("fable-") || m.starts_with("mythos-")` — the routable set is Anthropic Claude models ONLY, so an arbitrary id like `openai-fable-5` is never rewritten. `is_high_tier` = `is_claude(m) && (m.contains("opus-4") || m.contains("fable-5") || m.contains("mythos-5"))`. `route` returns `Some("claude-haiku-4-5")` only when `is_claude(model) && mechanical && cooldown==0 && !model.contains("haiku") && (mode=="on" || (mode=="auto" && is_high_tier(model)))`; else `None`. Add a test asserting `route("openai-fable-5", true, 0, "on") == None` (non-Claude never routed even in `on` mode).

- [ ] **Step 4: Run to verify pass** — `cargo test --lib model_router` → PASS.

- [ ] **Step 5: Wire** — gated `AXIOM_MODEL_ROUTE` ∈ {`off`(default),`auto`,`on`}. When `route` returns `Some(m)`, set `outbound["model"] = m` before forward; on any upstream 4xx for a routed turn, retry once with the original model (mirror S1 compression-fallback), count `route_fallbacks`. Sticky escalation: `has_error_signature` on any tool_result → 3-turn cooldown.

- [ ] **Step 6: Integration test** `tests/model_router_proxy.rs`: (a) `AXIOM_MODEL_ROUTE=auto` + Opus mechanical turn → upstream receives `model=claude-haiku-4-5`; (b) same on Sonnet → upstream receives `model=claude-sonnet-5`; (c) routed turn that 4xxs → exactly one retry with the original model.

- [ ] **Step 7: Verify + clippy + commit**
```bash
git add -A && git commit -m "feat(pss): P4 R1 high-tier-gated model routing"
```

---

### Task P5: Live eval harness extension + default flips (the gate)

**Files:**
- Modify: `scripts/cvm_eval.sh` (PSS-on condition; report quota units from `/metrics axiom_quota_units_total`; long multi-turn dependent-task sequences)
- Add: `bench/cvm/pss-eval-tasks.tsv`
- On PASS only: flip each lever default in `routes_messages.rs`; update the digest-style tests that assumed unset==off; update `README.md` + `docs/CAPABILITIES.md` + this plan's status with the real measured quota numbers.

**Interfaces:**
- Consumes: all P0–P4 flags; P0's quota ledger. Produces: `bench/cvm/PSS-RESULTS-<date>.md`.

**Agent SDK credit caveat (deep-research 2026-07-11):** since 2026-06-15, non-interactive `claude -p` usage draws from a SEPARATE monthly Agent-SDK credit pool ($20 Pro / $100 Max 5x / $200 Max 20x), not the main subscription window. The live eval's spend hits that pool; note this in `PSS-RESULTS-<date>.md` so the numbers aren't misread as main-window savings.

- [ ] **Step 1:** Extend `cvm_eval.sh` with a third proxy condition exporting all PSS flags on (`AXIOM_TOOL_DEFER=on AXIOM_LOCAL_TRIVIAL=on AXIOM_REBASE_ON_BREAK=on AXIOM_ADAPTIVE_TTL=on AXIOM_MODEL_ROUTE=auto`) and long multi-turn task sequences (a dependent chain in ONE session so history grows and deferral/rebase engage — e.g. "read file A" → "compare to B" → "summarize both" → 15+ follow-ups). Score: correctness parity (S5 rule), deferral-fault (a deferred tool that had to be loaded) + local-continuity fault rate ≤5%, quota units (not just USD) strictly lower with PSS on, target ≥50% on the long-session tasks.
- [ ] **Step 2:** Human runs `./scripts/cvm_eval.sh` deliberately (real credits). Capture `bench/cvm/PSS-RESULTS-<date>.md`.
- [ ] **Step 3 (on PASS):** One PR flips the four lever defaults on (`AXIOM_TOOL_DEFER`, `AXIOM_LOCAL_TRIVIAL`, `AXIOM_REBASE_ON_BREAK`, `AXIOM_ADAPTIVE_TTL`) + `AXIOM_MODEL_ROUTE` default `auto`; fix the tests that assumed unset==off (mirror the S5→flip `digest_proxy.rs` fix); update README/CAPABILITIES/this-plan-status with real measured quota numbers, honestly separated from the simulation's +56.8% Sonnet / +66.4% Opus / +69.1% Fable.
- [ ] **Step 3 (on FAIL):** Commit the results file, leave all defaults off, mark status "shipped, gated-off, live-eval FAILED — returned to brainstorming", report to the user.
- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "feat(pss): P5 live eval + default flips (on pass)"
```

---

## Self-Review

- **Spec coverage:** R0→P0, L-A→P1, R2+R3→P2, L-B→P3, R1→P4, R4 validation→P5. All spec components mapped.
- **Cache-safety constraint:** enforced in P1 (break-window-only tools), P2 (break-only rebase), P3 (short-circuit never touches the prefix).
- **Type consistency:** `quota_units` (P0) consumed by P1/P4 telemetry; `has_error_signature` defined in P3, reused in P4 (noted at both). `CvmStore`/`build_stub`/`diet_system_field`/`SkeletonDigestor`/`DEFAULT_DIGEST_THRESHOLD_TOKENS`/`mean_surprisal` all shipped and verified present this session.
- **Dependency order:** P0 foundation; P1 depends on P0 + S2; P2/P3/P4 independent of each other, all depend on S1 break detection; P5 last. No forward references.
- **No placeholders:** the Fable-5 pricing note is a directive to confirm the real rate before merge (a concrete `FABLE` const is given), not a deferred TODO.
