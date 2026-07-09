# Phase 1: Session Record/Replay + Savings Receipts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The proxy can persist scrubbed per-session request/response JSONL (`AXIOM_SESSION_RECORD=1`), and every session emits a token-savings receipt (per-session + lifetime counters on `/metrics`, one-line receipt at session drop).

**Architecture:** A new `session_recorder` module owns file layout, scrubbing, and append semantics; handlers call one `record_exchange(...)` function (fire-and-forget — recording failures never block the request path). Savings accounting adds a per-session `SavingsLedger` (`Mutex<HashMap<String, (u64, u64)>>`) in `AppState`, fed at the three existing `state.controls.record(...)` call sites, rendered in `/metrics`, and drained into a receipt log line on session drop.

**Tech Stack:** Rust, serde_json, std::fs (append), existing `metrics.rs` render. No new crate dependencies.

## Global Constraints

- Build/test with `CARGO_TARGET_DIR=target-test` (live proxy locks `target/release/axiom_engine.exe`).
- No new crate dependencies.
- Conventional commits, no attribution footer.
- Recording is default OFF (`AXIOM_SESSION_RECORD=1` enables); receipts are always on (they cost one HashMap update per compressed request).
- Secrets never persist: Authorization header values are never written; body strings matching key patterns are redacted at write time.

---

### Task 1: `session_recorder` module (scrub + append + read-back)

**Files:**
- Create: `axiom_engine_rs/src/session_recorder.rs`
- Modify: `axiom_engine_rs/src/lib.rs` (add `pub mod session_recorder;` after `pub mod server;`)
- Test: unit tests inside `session_recorder.rs`

**Interfaces:**
- Produces:
  - `pub fn recording_enabled() -> bool` (env `AXIOM_SESSION_RECORD` in {1,true,yes,on})
  - `pub struct ExchangeRecord { pub ts: u64, pub endpoint: String, pub session_id: String, pub request: serde_json::Value, pub response: serde_json::Value, pub compressed: bool }`
  - `pub fn record_exchange(record: ExchangeRecord)` — scrubs, appends JSONL under `sessions_dir()/<session-id>.jsonl`; all errors log-and-continue
  - `pub fn scrub(value: &serde_json::Value) -> serde_json::Value` — deep-copies with secret-looking strings redacted
  - `pub fn sessions_dir() -> std::path::PathBuf` — `AXIOM_SESSIONS_DIR` env or `~/.axiom/sessions`
  - `pub fn read_session(path: &std::path::Path) -> std::io::Result<Vec<ExchangeRecord>>`

- [ ] **Step 1: Write the failing tests**

Create `axiom_engine_rs/src/session_recorder.rs` containing ONLY the test module first:

```rust
//! Per-session request/response recording (opt-in via AXIOM_SESSION_RECORD).
//!
//! One JSONL file per session under `sessions_dir()`. Secrets are scrubbed at
//! write time; recording failures never block the request path.

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scrub_redacts_secret_looking_strings_and_auth_keys() {
        let v = json!({
            "headers": {"authorization": "Bearer sk-live-abc123def456ghi789", "accept": "json"},
            "body": {"note": "use key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX please",
                      "ok": "plain text stays"}
        });
        let s = scrub(&v);
        assert_eq!(s["headers"]["authorization"], "[REDACTED]");
        assert_eq!(s["headers"]["accept"], "json");
        assert!(!s["body"]["note"].as_str().unwrap().contains("sk-proj-"));
        assert_eq!(s["body"]["ok"], "plain text stays");
    }

    #[test]
    fn record_and_read_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "axiom_rec_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::env::set_var("AXIOM_SESSIONS_DIR", &dir);
        let rec = ExchangeRecord {
            ts: 1234,
            endpoint: "/v1/responses".into(),
            session_id: "s1".into(),
            request: json!({"input":"hi"}),
            response: json!({"output":"there"}),
            compressed: true,
        };
        record_exchange(rec);
        let file = sessions_dir().join("s1.jsonl");
        let records = read_session(&file).unwrap();
        std::env::remove_var("AXIOM_SESSIONS_DIR");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].endpoint, "/v1/responses");
        assert_eq!(records[0].response["output"], "there");
    }

    #[test]
    fn session_id_is_sanitized_for_filesystem() {
        assert_eq!(session_file_name("a/b\\c:d"), "a_b_c_d.jsonl");
        assert_eq!(session_file_name("normal-id_1"), "normal-id_1.jsonl");
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `CARGO_TARGET_DIR=target-test cargo test --lib session_recorder 2>&1 | tail -5`
Expected: FAIL — functions undefined. (Add `pub mod session_recorder;` to `lib.rs` in this step so the failure is about functions, not the module.)

- [ ] **Step 3: Implement the module**

Prepend above the test module in `session_recorder.rs`:

```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeRecord {
    pub ts: u64,
    pub endpoint: String,
    pub session_id: String,
    pub request: Value,
    pub response: Value,
    pub compressed: bool,
}

pub fn recording_enabled() -> bool {
    std::env::var("AXIOM_SESSION_RECORD")
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

pub fn sessions_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("AXIOM_SESSIONS_DIR") {
        if !dir.trim().is_empty() {
            return PathBuf::from(dir);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join(".axiom")
        .join("sessions")
}

/// Replace every value under a secret-named key, and every string that looks
/// like a bearer token / API key, with "[REDACTED]". Conservative by design.
pub fn scrub(value: &Value) -> Value {
    fn looks_secret(s: &str) -> bool {
        let lower = s.to_ascii_lowercase();
        lower.starts_with("bearer ")
            || s.len() >= 20
                && ["sk-", "ghp_", "gho_", "eyj"]
                    .iter()
                    .any(|p| lower.contains(p))
    }
    fn scrub_string(s: &str) -> Value {
        if looks_secret(s) {
            return Value::String("[REDACTED]".into());
        }
        Value::String(s.to_string())
    }
    fn walk(v: &Value) -> Value {
        match v {
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, val)| {
                        let lk = k.to_ascii_lowercase();
                        let redact = [
                            "authorization", "x-api-key", "api_key", "apikey", "token",
                            "secret", "password",
                        ]
                        .contains(&lk.as_str());
                        if redact {
                            (k.clone(), Value::String("[REDACTED]".into()))
                        } else {
                            (k.clone(), walk(val))
                        }
                    })
                    .collect(),
            ),
            Value::Array(items) => Value::Array(items.iter().map(walk).collect()),
            Value::String(s) => scrub_string(s),
            other => other.clone(),
        }
    }
    walk(value)
}

fn session_file_name(session_id: &str) -> String {
    let safe: String = session_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    format!("{safe}.jsonl")
}

/// Append one scrubbed record; every failure is log-and-continue.
pub fn record_exchange(record: ExchangeRecord) {
    let dir = sessions_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("[record] cannot create {}: {e}", dir.display());
        return;
    }
    let scrubbed = ExchangeRecord {
        request: scrub(&record.request),
        response: scrub(&record.response),
        ..record
    };
    let path = dir.join(session_file_name(&scrubbed.session_id));
    let line = match serde_json::to_string(&scrubbed) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[record] serialize failed: {e}");
            return;
        }
    };
    use std::io::Write;
    match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{line}") {
                eprintln!("[record] append failed for {}: {e}", path.display());
            }
        }
        Err(e) => eprintln!("[record] open failed for {}: {e}", path.display()),
    }
}

pub fn read_session(path: &Path) -> std::io::Result<Vec<ExchangeRecord>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect())
}
```

Note: the scrub test asserts the inline `sk-proj-…` token inside prose is removed — `looks_secret` covers it because the whole string contains `sk-` and is ≥20 chars, redacting the entire string. That is the conservative behavior we want.

- [ ] **Step 4: Run to verify pass**

Run: `CARGO_TARGET_DIR=target-test cargo test --lib session_recorder 2>&1 | tail -5`
Expected: PASS — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add axiom_engine_rs/src/session_recorder.rs axiom_engine_rs/src/lib.rs
git commit -m "feat: session recorder module (scrubbed per-session JSONL)"
```

---

### Task 2: Wire recording into the three proxy handlers

**Files:**
- Modify: `axiom_engine_rs/src/server.rs` — `create_message` (Anthropic), `create_chat_completion`, `create_response`

**Interfaces:**
- Consumes: `crate::session_recorder::{recording_enabled, record_exchange, ExchangeRecord}` (Task 1).
- Produces: no new public surface. For JSON (non-streaming) responses the full body is recorded; for streaming/SSE responses the record's `response` is `{"streamed": true, "status": <code>}`.

- [ ] **Step 1: Add a helper near `relayable_response_headers` in `server.rs`**

```rust
/// Fire-and-forget session recording (AXIOM_SESSION_RECORD=1). `response` is
/// the full JSON body when available, else `{"streamed":true,"status":N}`.
fn record_proxy_exchange(
    endpoint: &str,
    session_id: &str,
    request: &Value,
    response: Value,
    compressed: bool,
) {
    if !crate::session_recorder::recording_enabled() {
        return;
    }
    let record = crate::session_recorder::ExchangeRecord {
        ts: unix_now(),
        endpoint: endpoint.to_string(),
        session_id: session_id.to_string(),
        request: request.clone(),
        response,
        compressed,
    };
    tokio::task::spawn_blocking(move || crate::session_recorder::record_exchange(record));
}
```

- [ ] **Step 2: Call it from `create_response`**

After the upstream response's status/content-type are known (the handler branches on streaming vs JSON — read the handler body first; variable names below must be adjusted to the actual in-scope names):
- JSON branch: `record_proxy_exchange("/v1/responses", session_override.unwrap_or("anonymous"), &body, response_json.clone(), compressed.is_some());`
- Streaming branch: `record_proxy_exchange("/v1/responses", session_override.unwrap_or("anonymous"), &body, serde_json::json!({"streamed": true, "status": status.as_u16()}), compressed.is_some());`

- [ ] **Step 3: Call it from `create_chat_completion` and `create_message`**

Same pattern with endpoints `"/v1/chat/completions"` and `"/v1/messages"`; session id from the `x-axiom-session-id` header variable in each handler; `compressed` = true inside the compression branch, false on passthrough.

- [ ] **Step 4: Integration test**

Add a new `#[tokio::test]` to `axiom_engine_rs/tests/responses_compression_proxy.rs` (it already boots the router against a mock upstream and mutates env): set `AXIOM_SESSION_RECORD=1` and `AXIOM_SESSIONS_DIR=<unique tmp>` at the start, run one request through the proxy with header `x-axiom-session-id: rectest`, then assert `sessions_dir().join("rectest.jsonl")` exists with one record whose `endpoint == "/v1/responses"`. Remove both env vars and the tmp dir at the end.

- [ ] **Step 5: Run, verify, commit**

Run: `CARGO_TARGET_DIR=target-test cargo test --test responses_compression_proxy 2>&1 | tail -5`
Expected: PASS.

```bash
git add axiom_engine_rs/src/server.rs axiom_engine_rs/tests/responses_compression_proxy.rs
git commit -m "feat: record scrubbed request/response exchanges per session"
```

---

### Task 3: Savings ledger + receipts

**Files:**
- Modify: `axiom_engine_rs/src/server.rs` — `AppState` (add ledger field), the three `state.controls.record(...)` sites, `export_metrics`, `delete_session`, `ttt_session_drop`, shutdown flush path
- Test: `axiom_engine_rs/src/server.rs` `#[cfg(test)]` module

**Interfaces:**
- Produces:
  - `AppState.savings: Arc<Mutex<HashMap<String, (u64, u64)>>>` — session_id → (bytes_in, bytes_forwarded)
  - `fn record_savings(state: &AppState, session_id: &str, bytes_in: u64, bytes_out: u64)`
  - `fn savings_receipt(bytes_in: u64, bytes_out: u64) -> String`
  - `/metrics` gains `axiom_savings_bytes_in_total`, `axiom_savings_bytes_forwarded_total`, `axiom_savings_ratio`

- [ ] **Step 1: Failing unit test for the receipt formatter**

```rust
    #[test]
    fn savings_receipt_formats_bytes_and_ratio() {
        assert_eq!(savings_receipt(41_200, 17_300), "41.2k in, 17.3k forwarded, 58% saved");
        assert_eq!(savings_receipt(0, 0), "0.0k in, 0.0k forwarded, 0% saved");
        // Never negative even if out > in (fingerprint overhead on tiny bodies).
        assert_eq!(savings_receipt(100, 150), "0.1k in, 0.2k forwarded, 0% saved");
    }
```

Run: `CARGO_TARGET_DIR=target-test cargo test --lib savings_receipt_formats 2>&1 | tail -3` → FAIL (undefined).

- [ ] **Step 2: Implement formatter + ledger + wiring**

```rust
/// One-line human receipt: thousands of bytes with one decimal + saved ratio.
fn savings_receipt(bytes_in: u64, bytes_out: u64) -> String {
    let saved_pct = if bytes_in > 0 && bytes_out < bytes_in {
        ((bytes_in - bytes_out) * 100 / bytes_in) as u32
    } else {
        0
    };
    format!(
        "{:.1}k in, {:.1}k forwarded, {}% saved",
        bytes_in as f64 / 1000.0,
        bytes_out as f64 / 1000.0,
        saved_pct
    )
}

fn record_savings(state: &AppState, session_id: &str, bytes_in: u64, bytes_out: u64) {
    if let Ok(mut ledger) = state.savings.lock() {
        let entry = ledger.entry(session_id.to_string()).or_insert((0, 0));
        entry.0 += bytes_in;
        entry.1 += bytes_out;
    }
}
```

- Add `pub savings: Arc<Mutex<HashMap<String, (u64, u64)>>>` to `AppState` and initialize with `Arc::new(Mutex::new(HashMap::new()))` in its constructor.
- At each `state.controls.record(...)` site add `record_savings(&state, <session-id-in-scope>, bytes_in, bytes_out);` (`session_override.unwrap_or("anonymous")` where none).
- In `export_metrics`, sum the ledger and append the three metric lines to the rendered text.
- In `delete_session` and `ttt_session_drop`, remove the ledger entry and, when non-zero, `eprintln!("[receipt] session {id}: {}", savings_receipt(bin, bout));`. In the shutdown flush path, print all remaining receipts.

- [ ] **Step 3: Run tests + full compile**

`CARGO_TARGET_DIR=target-test cargo test --lib savings_receipt_formats 2>&1 | tail -3` → PASS.
`CARGO_TARGET_DIR=target-test cargo check --release 2>&1 | grep -c "^error"` → 0.

- [ ] **Step 4: Commit**

```bash
git add axiom_engine_rs/src/server.rs
git commit -m "feat: per-session savings ledger, /metrics counters, drop receipts"
```

---

### Task 4: Full-suite verification + docs

- [ ] **Step 1:** `CARGO_TARGET_DIR=target-test cargo test --release 2>&1 | grep -E "^test result|FAILED"` — all `ok`.
- [ ] **Step 2:** README: add a row to the "What Is Implemented" table:

```markdown
| Session recording & receipts | Opt-in scrubbed per-session request/response JSONL (`AXIOM_SESSION_RECORD=1`, `~/.axiom/sessions/`), plus always-on token-savings receipts (per-session `/metrics` counters and a one-line receipt at session drop). | `session_recorder.rs`, `server.rs` |
```

- [ ] **Step 3:** `start_axiom.sh`: document `AXIOM_SESSION_RECORD` next to the compression envs.
- [ ] **Step 4: Commit**

```bash
git add README.md start_axiom.sh
git commit -m "docs: session recording and savings receipts"
```

---

## Self-Review

**Spec coverage (Deliverables 2 + 5):** JSONL layout/fields/scrubbing → Task 1; never-block error handling → Task 1 (log-and-continue) + Task 2 (spawn_blocking); handler coverage (messages, chat, responses; streamed marker) → Task 2; `/metrics` counters + drop receipt → Task 3; corpus promotion (`axiombench record --scrub`) and `replay --diff` belong to the Phase 3 bench binary and consume `read_session` — deferred per spec phasing. ✓

**Placeholders:** Task 2 Steps 2–3 name in-scope variables generically ("adjust to the actual names") because the handler bodies are long — the executor must read each handler first; flagged deliberately. All other steps carry complete code. ✓

**Type consistency:** `ExchangeRecord` fields match module, helper, and tests; `record_savings`/`savings_receipt` signatures consistent throughout. ✓
