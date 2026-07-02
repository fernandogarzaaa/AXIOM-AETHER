# Phase 0: Compression On By Default (with #85 ordering fix) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the multi-turn reordering bug (#85) in Responses input compression, then make Responses compression on-by-default (opt-out), so every Claude and Codex session saves tokens without a flag.

**Architecture:** `plan_compression` groups eligible assistant messages into *maximal contiguous runs* instead of one flat list. `apply_plan` rebuilds the input array in place — each run collapses to a single fingerprint at the run's first position; every non-selected item (user/tool/structural) keeps its exact position. The server generates one fingerprint per run. A single gate (`responses_compression_enabled`) flips from default-off to default-on.

**Tech Stack:** Rust, `serde_json::Value`, `sha2`, axum, tokio. Tests: `cargo test` (workspace), hand-rolled case tests (no new deps).

## Global Constraints

- Rust workspace `axiom_engine_rs`, build with `cargo build --release`.
- Run all builds/tests with `CARGO_TARGET_DIR=target-test` to avoid relink contention with the running proxy/MCP binary (the live proxy locks `target/release/axiom_engine.exe`).
- No new crate dependencies.
- Commit style: conventional commits (`fix:`, `feat:`, `test:`, `docs:`), no attribution footer.
- The kill switch `AXIOM_RESPONSES_COMPRESS=0` (and `AXIOM_TTT_COMPRESS=0`) must always fully disable compression.
- Pass-through fallback is load-bearing: any body that cannot be safely compressed must be forwarded unmodified.

---

### Task 1: Group eligible assistant messages into contiguous runs

Refactor `ResponsesCompressionPlan` to hold runs, and rewrite `plan_compression` to build them. To keep this task independently green, `apply_plan` is minimally adapted at the end (Task 2 rewrites it properly).

**Files:**
- Modify: `axiom_engine_rs/src/responses_compressor.rs` (struct + `plan_compression`; whole file is 113 lines)
- Test: `axiom_engine_rs/src/responses_compressor.rs` (new `#[cfg(test)]` module — file currently has none)

**Interfaces:**
- Produces:
  - `pub struct AssistantRun { pub indices: Vec<usize>, pub source_hashes: Vec<String>, pub context: String }`
  - `pub struct ResponsesCompressionPlan { pub runs: Vec<AssistantRun>, pub query: String }`
  - `impl ResponsesCompressionPlan { pub fn item_indices(&self) -> Vec<usize>; pub fn total_context(&self) -> String; }`
  - `pub fn plan_compression(body: &serde_json::Value) -> Option<ResponsesCompressionPlan>`

- [ ] **Step 1: Write the failing unit tests**

Add to the bottom of `axiom_engine_rs/src/responses_compressor.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assistant(text: &str) -> serde_json::Value {
        json!({"type":"message","role":"assistant","content":text})
    }
    fn user(text: &str) -> serde_json::Value {
        json!({"type":"message","role":"user","content":text})
    }

    #[test]
    fn contiguous_assistants_form_one_run() {
        let body = json!({"input":[
            assistant("A0"), assistant("A1"), user("latest")
        ]});
        let plan = plan_compression(&body).unwrap();
        assert_eq!(plan.runs.len(), 1);
        assert_eq!(plan.runs[0].indices, vec![0, 1]);
        assert_eq!(plan.runs[0].context, "A0\n\nA1");
        assert_eq!(plan.query, "latest");
    }

    #[test]
    fn assistants_split_by_user_form_separate_runs() {
        let body = json!({"input":[
            assistant("A0"), user("U1"), assistant("A2"), user("latest")
        ]});
        let plan = plan_compression(&body).unwrap();
        assert_eq!(plan.runs.len(), 2, "non-contiguous assistants are distinct runs");
        assert_eq!(plan.runs[0].indices, vec![0]);
        assert_eq!(plan.runs[1].indices, vec![2]);
        assert_eq!(plan.item_indices(), vec![0, 2]);
    }

    #[test]
    fn no_eligible_assistant_returns_none() {
        assert!(plan_compression(&json!({"input":"hello"})).is_none());
        assert!(plan_compression(&json!({"input":[user("only user")]})).is_none());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CARGO_TARGET_DIR=target-test cargo test --lib responses_compressor 2>&1 | tail -20`
Expected: FAIL — `no field \`runs\`` / `no method item_indices` (struct still old shape).

- [ ] **Step 3: Replace the struct and `plan_compression`**

Replace lines 9–65 of `axiom_engine_rs/src/responses_compressor.rs` (the `ResponsesCompressionPlan` struct through the end of `plan_compression`) with:

```rust
/// A maximal run of consecutive eligible assistant items (consecutive
/// original `input` indices). Each run compresses to ONE fingerprint at its
/// first index; because a run is contiguous, no non-selected item ever sits
/// between its members, so replacing it in place cannot reorder the transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssistantRun {
    pub indices: Vec<usize>,
    pub source_hashes: Vec<String>,
    pub context: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponsesCompressionPlan {
    pub runs: Vec<AssistantRun>,
    pub query: String,
}

impl ResponsesCompressionPlan {
    /// All compressed item indices, flattened in ascending order (metrics/logging).
    pub fn item_indices(&self) -> Vec<usize> {
        self.runs.iter().flat_map(|r| r.indices.iter().copied()).collect()
    }

    /// Concatenated context across every run — the total the threshold check
    /// weighs and the total the receipt accounting attributes as absorbed.
    pub fn total_context(&self) -> String {
        self.runs
            .iter()
            .map(|r| r.context.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

pub fn plan_compression(body: &Value) -> Option<ResponsesCompressionPlan> {
    let input = body.get("input")?.as_array()?;
    let latest_user = input
        .iter()
        .enumerate()
        .rev()
        .find(|(_, item)| item.get("role").and_then(Value::as_str) == Some("user"))
        .map(|(index, _)| index)?;
    let first_structural = input
        .iter()
        .position(|item| {
            !matches!(
                item.get("type").and_then(Value::as_str),
                None | Some("message")
            )
        })
        .unwrap_or(input.len());
    let safe_end = latest_user.min(first_structural);

    let selected: Vec<(usize, String)> = input[..safe_end]
        .iter()
        .enumerate()
        .filter(|(_, item)| item.get("role").and_then(Value::as_str) == Some("assistant"))
        .filter_map(|(index, item)| text_content(item.get("content")?).map(|text| (index, text)))
        .filter(|(_, text)| !text.trim().is_empty())
        .collect();
    if selected.is_empty() {
        return None;
    }

    // Group into maximal contiguous runs by consecutive original index.
    let mut runs: Vec<AssistantRun> = Vec::new();
    for (index, text) in selected {
        let hash = format!("{:x}", Sha256::digest(text.as_bytes()));
        match runs.last_mut() {
            Some(run) if run.indices.last().copied() == Some(index - 1) => {
                run.indices.push(index);
                run.source_hashes.push(hash);
                run.context.push_str("\n\n");
                run.context.push_str(&text);
            }
            _ => runs.push(AssistantRun {
                indices: vec![index],
                source_hashes: vec![hash],
                context: text,
            }),
        }
    }

    let query = text_content(input[latest_user].get("content")?).unwrap_or_default();
    Some(ResponsesCompressionPlan { runs, query })
}
```

- [ ] **Step 4: Temporarily adapt `apply_plan` to the new struct so the crate compiles**

`apply_plan` (lines 67–95) still references `plan.item_indices` and `plan.source_hashes` as fields. Change its body's opening so Task 1 compiles standalone (Task 2 rewrites it). Replace:

```rust
    let mut output = body.clone();
    let input = output.get_mut("input")?.as_array_mut()?;
    let insertion_index = *plan.item_indices.first()?;
    let manifest = plan
        .item_indices
        .iter()
        .zip(&plan.source_hashes)
```

with:

```rust
    let mut output = body.clone();
    let input = output.get_mut("input")?.as_array_mut()?;
    let flat_indices = plan.item_indices();
    let flat_hashes: Vec<String> =
        plan.runs.iter().flat_map(|r| r.source_hashes.iter().cloned()).collect();
    let insertion_index = *flat_indices.first()?;
    let manifest = flat_indices
        .iter()
        .zip(&flat_hashes)
```

and change the removal loop below from `for index in plan.item_indices.iter().rev()` to `for index in flat_indices.iter().rev()`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `CARGO_TARGET_DIR=target-test cargo test --lib responses_compressor 2>&1 | tail -20`
Expected: PASS — 3 passed.

- [ ] **Step 6: Commit**

```bash
git add axiom_engine_rs/src/responses_compressor.rs
git commit -m "refactor: model Responses compression as contiguous assistant runs"
```

---

### Task 2: Rebuild the input array in place (the #85 fix)

Rewrite `apply_plan` to take one fingerprint per run and rebuild the `input` array, preserving every non-selected item's position.

**Files:**
- Modify: `axiom_engine_rs/src/responses_compressor.rs` (`apply_plan`)
- Test: `axiom_engine_rs/src/responses_compressor.rs` (extend `#[cfg(test)]`)

**Interfaces:**
- Consumes: `ResponsesCompressionPlan`, `AssistantRun` (Task 1).
- Produces: `pub fn apply_plan(body: &serde_json::Value, plan: &ResponsesCompressionPlan, fingerprints: &[String]) -> Option<serde_json::Value>` — `fingerprints.len()` must equal `plan.runs.len()`.

- [ ] **Step 1: Write the failing ordering test**

Add to the `#[cfg(test)] mod tests` block in `axiom_engine_rs/src/responses_compressor.rs`:

```rust
    #[test]
    fn apply_preserves_position_of_interleaved_user_message() {
        // Regression for #85: A0, U1, A2 — the user turn between two
        // non-contiguous assistants must NOT move.
        let body = json!({"input":[
            assistant("A0"), user("U1"), assistant("A2"), user("latest")
        ]});
        let plan = plan_compression(&body).unwrap();
        let out = apply_plan(&body, &plan, &["FP0".into(), "FP2".into()]).unwrap();
        let items = out.get("input").unwrap().as_array().unwrap();

        assert_eq!(items.len(), 4, "two runs replaced 1-for-1, nothing collapsed away");
        assert_eq!(items[1], user("U1"), "interleaved user stays at position 1");
        assert_eq!(items[3], user("latest"));
        assert_eq!(items[0]["role"], "assistant");
        assert!(items[0]["content"].as_str().unwrap().contains("FP0"));
        assert!(items[2]["content"].as_str().unwrap().contains("FP2"));
    }

    #[test]
    fn apply_collapses_a_contiguous_run_to_one_fingerprint() {
        let body = json!({"input":[
            assistant("A0"), assistant("A1"), user("latest")
        ]});
        let plan = plan_compression(&body).unwrap();
        let out = apply_plan(&body, &plan, &["FP".into()]).unwrap();
        let items = out.get("input").unwrap().as_array().unwrap();
        assert_eq!(items.len(), 2, "the 2-message run collapses to one fingerprint");
        assert!(items[0]["content"].as_str().unwrap().contains("0:"));
        assert!(items[0]["content"].as_str().unwrap().contains("1:"));
        assert_eq!(items[1], user("latest"));
    }

    #[test]
    fn apply_rejects_fingerprint_count_mismatch() {
        let body = json!({"input":[
            assistant("A0"), user("U1"), assistant("A2"), user("latest")
        ]});
        let plan = plan_compression(&body).unwrap();
        assert!(apply_plan(&body, &plan, &["only-one".into()]).is_none());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CARGO_TARGET_DIR=target-test cargo test --lib responses_compressor 2>&1 | tail -20`
Expected: FAIL — `apply_plan` takes `&str`, not `&[String]` (arg type mismatch / arity).

- [ ] **Step 3: Rewrite `apply_plan`**

Replace the entire `apply_plan` function (from `pub fn apply_plan` through its closing `}`) in `axiom_engine_rs/src/responses_compressor.rs` with:

```rust
/// Replace each run with its fingerprint, in place. Non-selected items keep
/// their exact positions; a contiguous run collapses to one fingerprint at the
/// run's first index. `fingerprints[i]` corresponds to `plan.runs[i]`.
pub fn apply_plan(
    body: &Value,
    plan: &ResponsesCompressionPlan,
    fingerprints: &[String],
) -> Option<Value> {
    if fingerprints.len() != plan.runs.len() {
        return None;
    }
    let mut output = body.clone();
    let input = output.get_mut("input")?.as_array_mut()?;

    use std::collections::{HashMap, HashSet};
    // first-index -> run ordinal; trailing run indices -> dropped (folded into
    // the run's leading fingerprint).
    let mut run_start: HashMap<usize, usize> = HashMap::new();
    let mut dropped: HashSet<usize> = HashSet::new();
    for (ordinal, run) in plan.runs.iter().enumerate() {
        let first = *run.indices.first()?;
        run_start.insert(first, ordinal);
        for &idx in &run.indices[1..] {
            dropped.insert(idx);
        }
    }

    let mut rebuilt: Vec<Value> = Vec::with_capacity(input.len());
    for (idx, item) in input.iter().enumerate() {
        if let Some(&ordinal) = run_start.get(&idx) {
            let run = &plan.runs[ordinal];
            let manifest = run
                .indices
                .iter()
                .zip(&run.source_hashes)
                .map(|(i, h)| format!("{i}:{h}"))
                .collect::<Vec<_>>()
                .join(",");
            rebuilt.push(json!({
                "type": "message",
                "role": "assistant",
                "content": format!(
                    "<axiom_source_manifest compressed_item_sha256=\"{manifest}\" />\n{}",
                    fingerprints[ordinal]
                )
            }));
        } else if !dropped.contains(&idx) {
            rebuilt.push(item.clone());
        }
    }
    *input = rebuilt;
    Some(output)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CARGO_TARGET_DIR=target-test cargo test --lib responses_compressor 2>&1 | tail -20`
Expected: PASS — 6 passed.

- [ ] **Step 5: Commit**

```bash
git add axiom_engine_rs/src/responses_compressor.rs
git commit -m "fix: preserve transcript order when compressing Responses input (#85)"
```

---

### Task 3: Generate one fingerprint per run in the server

Update `compressed_responses_payload` to adapt each run's context in its own sub-session and call the new `apply_plan`. Extract the inline fingerprint-generation into a named helper.

**Files:**
- Modify: `axiom_engine_rs/src/server.rs:1387-1462` (`compressed_responses_payload`)
- Test: `axiom_engine_rs/tests/responses_compression.rs` (update to new `apply_plan` signature)

**Interfaces:**
- Consumes: `plan.runs`, `plan.total_context()`, `plan.item_indices()`, `apply_plan(body, plan, &[String])`.
- Produces: `async fn responses_run_fingerprint(state: &AppState, session_id: &str, context: &str, query: &str) -> Result<String, ApiError>` (module-private).

- [ ] **Step 1: Update the existing integration test to the new signature**

In `axiom_engine_rs/tests/responses_compression.rs`, the round-trip test (~line 46) calls `apply_plan(&body, &plan, "<axiom_context_fingerprint />")`. Replace that call with:

```rust
    let plan = plan_compression(&body).unwrap();
    let fingerprints: Vec<String> =
        plan.runs.iter().map(|_| "<axiom_context_fingerprint />".to_string()).collect();
    let output = apply_plan(&body, &plan, &fingerprints).unwrap();
```

Replace any `plan.item_indices` field access in this file with the method call `plan.item_indices()`.

- [ ] **Step 2: Run the integration test to verify it fails to compile**

Run: `CARGO_TARGET_DIR=target-test cargo test --test responses_compression 2>&1 | tail -20`
Expected: FAIL — server still calls `apply_plan(body, &plan, &fingerprint.to_prompt_block())` with a single `&str`, so the crate does not build.

- [ ] **Step 3: Extract the fingerprint helper and loop over runs**

In `axiom_engine_rs/src/server.rs`, replace the body of `compressed_responses_payload` from the `let threshold = state.controls.threshold();` line (~1398) through the `Ok(Some(compressed))` line (~1461) with:

```rust
    let threshold = state.controls.threshold();
    let context_tokens = state
        .pipeline
        .lock()
        .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?
        .token_count(&plan.total_context());
    if context_tokens < threshold {
        return Ok(None);
    }

    let session_id = session_override
        .map(str::to_string)
        .unwrap_or_else(|| format!("responses-{}", Uuid::new_v4()));

    // One fingerprint per contiguous run, each adapted in its own sub-session
    // so a run's recall vector reflects only that run's context.
    let mut fingerprints = Vec::with_capacity(plan.runs.len());
    for (ordinal, run) in plan.runs.iter().enumerate() {
        let run_session = format!("{session_id}#r{ordinal}");
        let fp = responses_run_fingerprint(state, &run_session, &run.context, &plan.query).await?;
        fingerprints.push(fp);
    }

    let compressed = apply_plan(body, &plan, &fingerprints)
        .ok_or_else(|| ApiError::Internal("Responses compression transform failed".into()))?;
    let bytes_in = serde_json::to_vec(body).map(|value| value.len()).unwrap_or(0) as u64;
    let bytes_out = serde_json::to_vec(&compressed)
        .map(|value| value.len())
        .unwrap_or(0) as u64;
    let compressed_items = plan.item_indices().len() as u64;
    state.controls.record(compressed_items, bytes_in, bytes_out);
    eprintln!(
        "[axiom-ttt] responses compressed session={} runs={} assistant_items={} tokens={} bytes={}=>{}",
        session_id,
        plan.runs.len(),
        compressed_items,
        context_tokens,
        bytes_in,
        bytes_out
    );
    Ok(Some(compressed))
}

/// Adapt one run's context into a fresh TTT sub-session and return the recall
/// fingerprint block. Extracted from the former inline body of
/// `compressed_responses_payload` so each run gets its own adaptation.
async fn responses_run_fingerprint(
    state: &AppState,
    session_id: &str,
    context: &str,
    query: &str,
) -> Result<String, ApiError> {
    let pipeline_arc = state.pipeline.clone();
    let store = state.ttt_sessions.clone();
    let context = context.to_string();
    let query = query.to_string();
    let top_k = state.compressor_config.recall_top_k;
    let session_for_task = session_id.to_string();
    let started = Instant::now();
    let fingerprint = spawn_blocking(move || -> Result<_, ApiError> {
        let pipeline = pipeline_arc
            .lock()
            .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
        let session = store
            .get_or_create(&session_for_task, &pipeline)
            .map_err(|error| ApiError::Internal(format!("session allocation failed: {error}")))?;
        let mut states = session.blocking_lock();
        let context_ids = pipeline.encode_text(&context);
        adapt_session_blocking(&pipeline, &mut states, &context_ids)
            .map_err(|error| ApiError::Internal(format!("TTT adapt failed: {error}")))?;
        let query_ids = pipeline.encode_text(&query);
        extract_memory_vector_blocking(
            &pipeline,
            &mut states,
            &query_ids,
            &session_for_task,
            context_ids.len(),
            started,
            top_k,
        )
        .map_err(|error| ApiError::Internal(format!("memory extraction failed: {error}")))
    })
    .await
    .map_err(|error| ApiError::Internal(format!("blocking task join failed: {error}")))??;
    Ok(fingerprint.to_prompt_block())
}
```

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `CARGO_TARGET_DIR=target-test cargo test --test responses_compression 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add axiom_engine_rs/src/server.rs axiom_engine_rs/tests/responses_compression.rs
git commit -m "feat: per-run fingerprints for Responses compression"
```

---

### Task 4: Flip Responses compression to default-on (opt-out)

**Files:**
- Modify: `axiom_engine_rs/src/server.rs:1381-1385` (`responses_compression_enabled`)
- Test: `axiom_engine_rs/src/server.rs` (extend the `#[cfg(test)] mod tests` near `relayable_response_headers_skips_hop_by_hop_and_managed`)

**Interfaces:**
- Consumes: env `AXIOM_RESPONSES_COMPRESS`.
- Produces: `fn responses_compression_enabled() -> bool` (default true; false only on explicit off value).

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `axiom_engine_rs/src/server.rs`:

```rust
    #[test]
    fn responses_compression_defaults_on_and_opts_out() {
        std::env::remove_var("AXIOM_RESPONSES_COMPRESS");
        assert!(responses_compression_enabled(), "on by default when unset");

        std::env::set_var("AXIOM_RESPONSES_COMPRESS", "0");
        assert!(!responses_compression_enabled(), "explicit 0 disables");
        std::env::set_var("AXIOM_RESPONSES_COMPRESS", "off");
        assert!(!responses_compression_enabled(), "explicit off disables");

        std::env::set_var("AXIOM_RESPONSES_COMPRESS", "1");
        assert!(responses_compression_enabled(), "explicit 1 enables");
        std::env::remove_var("AXIOM_RESPONSES_COMPRESS");
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CARGO_TARGET_DIR=target-test cargo test --lib responses_compression_defaults_on_and_opts_out 2>&1 | tail -20`
Expected: FAIL — current impl returns false when unset.

- [ ] **Step 3: Flip the default**

Replace `responses_compression_enabled` (lines 1381–1385) in `axiom_engine_rs/src/server.rs` with:

```rust
/// Responses compression is ON by default (opt-out). It still requires general
/// compression (`state.controls.enabled()` / AXIOM_TTT_COMPRESS) to be active;
/// this gate only lets an operator disable the Responses path specifically via
/// AXIOM_RESPONSES_COMPRESS in {0,false,no,off}.
fn responses_compression_enabled() -> bool {
    match std::env::var("AXIOM_RESPONSES_COMPRESS") {
        Ok(value) => !matches!(value.to_lowercase().as_str(), "0" | "false" | "no" | "off"),
        Err(_) => true,
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `CARGO_TARGET_DIR=target-test cargo test --lib responses_compression_defaults_on_and_opts_out 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add axiom_engine_rs/src/server.rs
git commit -m "feat: Responses compression on by default (opt-out via AXIOM_RESPONSES_COMPRESS=0)"
```

---

### Task 5: Full-suite verification and operator-facing docs

**Files:**
- Modify: `axiom_engine_rs/src/server.rs` (startup banner block near line 4617)
- Modify: `README.md` (compression section)
- Modify: `axiom.env` (document `AXIOM_RESPONSES_COMPRESS`)

- [ ] **Step 1: Run the entire test suite**

Run: `CARGO_TARGET_DIR=target-test cargo test --release 2>&1 | grep -E "^test result|FAILED"`
Expected: every line `ok`, zero `FAILED`. The only env-touching integration test (`tests/responses_compression_proxy.rs`) sets `AXIOM_RESPONSES_COMPRESS=1` explicitly and remains valid.

- [ ] **Step 2: Add the startup banner line**

In `axiom_engine_rs/src/server.rs`, in the endpoint banner `println!` block, add after the `/v1/responses` line:

```rust
    println!(
        "[+] Responses input compression: {} (opt out with AXIOM_RESPONSES_COMPRESS=0)",
        if responses_compression_enabled() { "ON (default)" } else { "OFF" }
    );
```

- [ ] **Step 3: Document in README and axiom.env**

Add to `README.md` under the compression section:

```markdown
### Responses (Codex/OpenAI) input compression

On by default whenever compression is enabled. Axiom replaces old, text-only
assistant turns in the safe transcript prefix with a dense recall fingerprint,
preserving every user/tool/structural item in place. Disable with
`AXIOM_RESPONSES_COMPRESS=0`.
```

Add to `axiom.env` (near the other `AXIOM_TTT_COMPRESS*` lines):

```bash
# Responses (Codex/OpenAI) input compression. On by default; set to 0 to opt out.
# export AXIOM_RESPONSES_COMPRESS=0
```

- [ ] **Step 4: Rebuild the release binary and confirm it compiles**

Run: `CARGO_TARGET_DIR=target-test cargo build --release 2>&1 | grep -E "^error|Finished"`
Expected: `Finished`.

- [ ] **Step 5: Commit and close the issue**

```bash
git add axiom_engine_rs/src/server.rs README.md axiom.env
git commit -m "docs: document default-on Responses compression; banner + axiom.env"
```

Then:
```bash
gh issue close 85 --repo fernandogarzaaa/AXIOM-AETHER --comment "Fixed: Responses compression now models eligible assistants as contiguous runs and rebuilds the input in place, preserving the position of every non-selected item. Covered by apply_preserves_position_of_interleaved_user_message."
```

---

## Self-Review

**Spec coverage (Deliverable 1 / Phase 0):**
- "Fix #85 … replace each contiguous run independently, anchored in place" → Tasks 1–3. ✓
- "Property test: compressed transcript preserves relative order of retained items" → Task 2 `apply_preserves_position_of_interleaved_user_message`. ✓
- "Flip the default … AXIOM_TTT_COMPRESS=0 remains kill switch" → Task 4 (general kill switch untouched; still gates via `state.controls.enabled()`). ✓
- "Existing degraded-fallback behavior kept" → untouched; verified by `tests/responses_compression_proxy.rs`. ✓
- Deliverables 2–5 + AxiomBench → separate plans (spec phasing).

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command shows expected output. ✓

**Type consistency:** `ResponsesCompressionPlan { runs, query }`, `AssistantRun { indices, source_hashes, context }`, `item_indices()`/`total_context()`, and `apply_plan(body, plan, &[String])` are used identically across Tasks 1–3 and the integration-test update. `responses_run_fingerprint` signature matches its call site. ✓

**Scope:** Single subsystem, five right-sized tasks, each independently testable and committable. ✓
