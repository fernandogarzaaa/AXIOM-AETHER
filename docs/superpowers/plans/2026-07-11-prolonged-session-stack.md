# Prolonged-Session Stack (PSS v2) Implementation Plan

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
    let (_, estimated) = PriceTable::for_model("claude-fable-5");
    assert!(!estimated, "fable-5 must be a known model, not an estimate");
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

- [ ] **Step 3: Implement.** Add `FABLE` (Fable 5 is Mythos-class, above Opus; set to the real published Fable-5 per-MTok rate — confirm before merge, do not leave a placeholder) and extend `for_model` (fable arm BEFORE the opus/sonnet arms):

```rust
const FABLE: PriceTable = PriceTable {
    input_per_mtok: 6.00, cache_write_5m_per_mtok: 7.50,
    cache_read_per_mtok: 0.60, output_per_mtok: 30.00,
};
// in for_model:
} else if m.contains("fable-5") || m.contains("mythos-5") {
    (Self::FABLE, false)
} else if m.contains("opus-4") {
    (Self::OPUS, false)
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

### Task P1: L-A tool-schema elision (the anchor lever)

**Files:**
- Create: `axiom_engine_rs/src/tool_elision.rs`
- Modify: `axiom_engine_rs/src/lib.rs` (`pub mod tool_elision;`, alphabetical)
- Modify: `axiom_engine_rs/src/server/prelude_state.rs` (`AppState` field `tool_elide_frozen: Arc<RwLock<HashMap<String, Value>>>` — frozen `tools[]` per session; + `pub(crate)` get/set mirroring `cache_safety_memo`)
- Modify: `axiom_engine_rs/src/server/routes_messages.rs` (`compressed_messages_path`: apply after the S1 frozen/mutable split, before forward)
- Test: `axiom_engine_rs/tests/tool_elision_proxy.rs` + inline

**Interfaces:**
- Consumes: S1 break detection (a prefix mismatch via `cache_safety_memo` signals a break), S2 `CvmStore` for stashing elided schemas, `body["tools"]` (array of `{"name":..}`) and `body["messages"]`.
- Produces: `tool_elision::working_set(messages: &[Value], recent_k: usize) -> HashSet<String>`; `tool_elision::elide(tools: &[Value], keep: &HashSet<String>) -> (Vec<Value>, usize)` returning `(kept_plus_affordance, elided_count)`.

- [ ] **Step 1: Failing unit test** in `tool_elision.rs`:

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
fn elide_drops_unused_tools_and_adds_affordance() {
    let tools = vec![json!({"name":"Read"}), json!({"name":"WebFetch"}),
                     json!({"name":"ObscureTool"})];
    let mut keep = std::collections::HashSet::new();
    keep.insert("Read".to_string());
    let (kept, elided) = elide(&tools, &keep);
    assert_eq!(elided, 2);
    let names: Vec<_> = kept.iter()
        .filter_map(|t| t.get("name").and_then(|v| v.as_str())).collect();
    assert!(names.contains(&"Read"));
    assert!(names.contains(&"axiom_load_tools"));
    assert!(!names.contains(&"ObscureTool"));
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test --lib tool_elision 2>&1 | tail -20` → FAIL.

- [ ] **Step 3: Implement `tool_elision.rs`.** `working_set` walks messages for `tool_use` blocks' `name` within the last `recent_k` turns, unions `const CORE: [&str; 6] = ["Read","Edit","Write","Bash","Glob","Grep"]`. `elide` filters `tools` to `keep`, appends the affordance `{"name":"axiom_load_tools","description":"Some tools were omitted to save context. If you need a tool not listed, state which and it will be provided.","input_schema":{"type":"object","properties":{}}}`, returns `(kept, elided_count)`.

- [ ] **Step 4: Run to verify pass** — `cargo test --lib tool_elision` → PASS.

- [ ] **Step 5: Wire break-window-only application** in `compressed_messages_path`, gated `AXIOM_TOOL_ELIDE == Ok("on")` AND `uses_cache`: if a frozen `tools[]` exists for the session and no break occurred → reuse it verbatim (byte-stable); if a break occurred (S1 memo mismatch) or none exists → recompute `working_set`+`elide`, stash the full original `tools[]` into `CvmStore` under `tools:<session>`, store the elided array in `tool_elide_frozen`. Splice the frozen elided array into `outbound["tools"]`. Record the prefix-token delta via P0's quota ledger.

- [ ] **Step 6: Integration test** `tests/tool_elision_proxy.rs` (mirror `digest_proxy.rs`'s mock upstream + `Capture` + env-lock/`EnvVarGuard`): (a) `AXIOM_TOOL_ELIDE=on`, 10 tools with 2 recently used → upstream receives ≤3 tools incl. `axiom_load_tools`; (b) two consecutive no-break turns → outbound `tools[]` byte-identical both (cache-safe); (c) flag unset → tools unchanged.

- [ ] **Step 7: Verify + clippy + commit**

Run: `cargo test --lib && cargo test --test tool_elision_proxy && cargo clippy --lib --locked -- -D warnings && cargo clippy --test tool_elision_proxy --locked -- -D warnings`
```bash
git add -A && git commit -m "feat(pss): P1 L-A tool-schema elision (break-window-only)"
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

**Files:**
- Create: `axiom_engine_rs/src/model_router.rs`
- Modify: `axiom_engine_rs/src/lib.rs` (`pub mod model_router;`)
- Modify: `axiom_engine_rs/src/server/routes_messages.rs`
- Modify: `axiom_engine_rs/src/session_awareness.rs` (`routed_turns`, `route_fallbacks`)
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

- [ ] **Step 3: Implement.** `is_high_tier` = `m.contains("opus-4") || m.contains("fable-5") || m.contains("mythos-5")`. `route` returns `Some("claude-haiku-4-5")` only when `mechanical && cooldown==0 && model != haiku && (mode=="on" || (mode=="auto" && is_high_tier(model)))`; else `None`.

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

- [ ] **Step 1:** Extend `cvm_eval.sh` with a third proxy condition exporting all PSS flags on (`AXIOM_TOOL_ELIDE=on AXIOM_LOCAL_TRIVIAL=on AXIOM_REBASE_ON_BREAK=on AXIOM_ADAPTIVE_TTL=on AXIOM_MODEL_ROUTE=auto`) and long multi-turn task sequences (a dependent chain in ONE session so history grows and elision/rebase engage — e.g. "read file A" → "compare to B" → "summarize both" → 15+ follow-ups). Score: correctness parity (S5 rule), elision + local-continuity fault rate ≤5%, quota units (not just USD) strictly lower with PSS on, target ≥50% on the long-session tasks.
- [ ] **Step 2:** Human runs `./scripts/cvm_eval.sh` deliberately (real credits). Capture `bench/cvm/PSS-RESULTS-<date>.md`.
- [ ] **Step 3 (on PASS):** One PR flips the four lever defaults on + `AXIOM_MODEL_ROUTE` default `auto`; fix the tests that assumed unset==off (mirror the S5→flip `digest_proxy.rs` fix); update README/CAPABILITIES/this-plan-status with real measured quota numbers, honestly separated from the simulation's +56.8%/+66.4%.
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
