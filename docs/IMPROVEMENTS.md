# AXIOM AETHER — Improvement Recommendations from Live Testing

## Audit Date: 2026-07-09
## Environment: Windows 11, CPU-only, PowerShell, VS Code + Cline

---

## Critical Issues (Fix Immediately)

### 1. Dockerfile Corrupted
**Severity:** Critical
**Symptom:** `Dockerfile` contains binary/garbled content instead of valid Dockerfile syntax.
**Impact:** Docker builds fail completely; CI Docker workflow cannot produce images.
**Fix:** Regenerate the Dockerfile from the `docker-compose.yml` and the repo's known
structure. The image should be a multi-stage Rust build that produces the `axiom` binary
and serves it on port 8080.

### 2. Autostart Task Path Mismatch (FIXED)
**Severity:** High
**Symptom:** `axiom_autostart_task.xml` pointed to `C:\Users\garza\AXIOM-AETHER` but the
repo is at `D:\AXIOM-AETHER`.
**Impact:** Autostart on login silently failed; proxy was never started automatically.
**Fix Applied:** Updated XML to use `powershell.exe` with
`D:\AXIOM-AETHER\scripts\start_axiom_proxy.ps1 -Watchdog`.

### 3. Vibe Memory Disabled in Proxy
**Severity:** High
**Symptom:** `restart_proxy.ps1` sets `AXIOM_VIBE=0`, disabling persistent fast-weight
memory across proxy restarts.
**Impact:** All accumulated session context is lost on restart; the proxy starts from
scratch every time, reducing compression quality over time.
**Recommendation:** Enable `AXIOM_VIBE=1` in `restart_proxy.ps1` (the new
`start_axiom_proxy.ps1` already defaults to `1`). The EMA-merge on shutdown is designed
to be safe — it flushes sessions to disk on graceful shutdown.

### 4. No Watchdog Running
**Severity:** High
**Symptom:** `axiom_proxy_watchdog.ps1` exists but no watchdog log file is present,
indicating the Task Scheduler watchdog is not active.
**Impact:** Proxy has "died twice" per the watchdog script comments. Without the
watchdog, a crash requires manual restart.
**Recommendation:** Register the watchdog in Task Scheduler to run every 5 minutes:
```powershell
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File D:\AXIOM-AETHER\scripts\axiom_proxy_watchdog.ps1"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName "AxiomProxyWatchdog" -Action $action -Trigger $trigger
```

---

## Operational Improvements

### 5. Log Encoding Issues
**Severity:** Medium
**Symptom:** Em-dashes in server log output render as `â€"` (UTF-8 vs Windows console
encoding mismatch).
**Impact:** Log readability reduced; automated log parsing may fail.
**Recommendation:** Either:
- Set `PYTHONUTF8=1` and `RUST_BACKTRACE=1` in the proxy environment, or
- Replace em-dashes in `eprintln!`/`println!` calls with ASCII equivalents (`--`), or
- Configure PowerShell to use UTF-8: `chcp 65001` before launching the binary.

### 6. Compression Threshold Too Aggressive
**Severity:** Medium
**Symptom:** `AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS=200` (whitespace words, ~800 real
tokens) may compress normal conversation that isn't actually "heavy" context.
**Impact:** Over-compression of normal dialogue; potential information loss in
medium-length messages.
**Recommendation:** Raise to `400` (~1600 real tokens) for the proxy. The skeleton
compressor excels on large code blocks and whole files, not on medium-length prose.
Monitor `axiom_savings_ratio` in `/metrics` to tune.

### 7. Missing Drift Gate File
**Severity:** Low
**Symptom:** `checkpoints/axiom_drift_gate.txt` is not present; using hardcoded default
`7.03` instead of an eval-calibrated threshold.
**Impact:** Drift detection may not be optimally calibrated for the current checkpoint.
**Recommendation:** Run `eval_model` to produce a calibrated drift gate:
```powershell
cd d:\AXIOM-AETHER\axiom_engine_rs
cargo run --release --features tools --bin eval_model -- --checkpoint ..\checkpoints\axiom_production_bpe.bin
```
This writes `axiom_drift_gate.txt` which `start_axiom_proxy.ps1` will auto-detect.

### 8. No Health Check in Cline Config
**Severity:** Low
**Symptom:** Cline has no way to detect if the Axiom MCP server process is alive.
**Impact:** If the MCP server crashes, Cline silently loses tool access until reload.
**Recommendation:** This is a Cline limitation, not an Axiom issue. The `alwaysAllow`
list in the MCP settings reduces friction. Cline will respawn the MCP process on
session start.

---

## Documentation Improvements (Partially Fixed)

### 9. MCP Tool Count Mismatch (FIXED)
**Severity:** Low
**Symptom:** `AGENT-SETUP.md` says "10 tools" but `mcp_stdio.rs` exposes 16.
**Impact:** Users may expect fewer tools than available; setup verification fails.
**Fix Applied:** Updated `MCP-CLIENTS.md` to list all 16 tools. `AGENT-SETUP.md` should
also be updated to reflect the full count.

### 10. No Cline-Specific Docs (FIXED)
**Severity:** Low
**Symptom:** No documentation for Cline setup.
**Fix Applied:** Created `docs/CLINE-SETUP.md` with full configuration, verification,
and troubleshooting guide.

### 11. No Windows-Native Startup Guide (FIXED)
**Severity:** Low
**Symptom:** All docs reference `start_axiom.sh` (bash), but Windows machines may not
have Git Bash reliably.
**Fix Applied:** Created `scripts\start_axiom_proxy.ps1` — a native PowerShell launcher
with watchdog support, proper env var setup, and health check.

---

## Architecture Improvements (Future Work)

### 12. MCP Server Pipeline Mutex Contention
**Severity:** Medium
**Symptom:** `Arc<Mutex<InferencePipeline>>` in `McpContext` serializes all tool calls.
A long-running `axiom_compress_path` blocks `axiom_recall` and all other tools.
**Impact:** Under concurrent tool calls (e.g., Cline calling multiple tools in
parallel), latency spikes.
**Recommendation:** Consider `RwLock` for read-heavy access patterns, or per-session
pipelines. The pipeline is mostly read during inference (forward pass) and only
mutated during adaptation (TTT update). An `RwLock` would allow concurrent reads
while serializing writes.

### 13. No Graceful MCP Shutdown Vibe Flush
**Severity:** Medium
**Symptom:** The stdio MCP server (`run_stdio_server`) drops the pipeline on a blocking
thread but does not flush the vibe memory. The HTTP proxy does flush on shutdown.
**Impact:** If `AXIOM_VIBE=1` is enabled for MCP, accumulated session context is lost
when Cline closes the MCP process.
**Recommendation:** Add a vibe flush before the pipeline drop in `run_stdio_server`:
```rust
// Before dropping the pipeline, flush vibe memory
if let Some(vibe) = ctx.vibe.lock().unwrap().try_flush() {
    eprintln!("[mcp] vibe memory flushed on shutdown");
}
```

### 14. Compression Cache Not Versioned
**Severity:** Low
**Symptom:** `axiom_compression_cache.bin` is hydrated on startup without checking
checkpoint version compatibility.
**Impact:** After a checkpoint upgrade, stale adapted sessions may be loaded, causing
mismatched fast-weight dimensions.
**Recommendation:** Store a checkpoint hash alongside the compression cache and
invalidate on mismatch. The `model_meta.rs` sidecar already records architecture
metadata — extend it to gate cache hydration.

### 15. No MCP Tool Timeout
**Severity:** Low
**Symptom:** `axiom_compress_path` on a very large directory can take minutes with no
timeout. Cline may kill the MCP process if it appears hung.
**Impact:** Large directory compression may be interrupted, leaving partial state.
**Recommendation:** Add a configurable timeout (e.g., `AXIOM_MCP_TIMEOUT_SECS=120`)
that returns a partial result if exceeded. The `max_files` and `max_bytes` limits
already bound the work, but a wall-clock timeout is a safety net.

### 16. Memory Store Not Backed Up
**Severity:** Low
**Symptom:** `checkpoints/memory/` contains the persistent memory store but has no
backup mechanism.
**Impact:** Accidental deletion or corruption loses all `axiom_remember` data.
**Recommendation:** Add a periodic backup (copy `*.jsonl` to a timestamped backup
directory) or integrate with the existing `.vibe_backups/` pattern.

---

## Performance Observations from Live Logs

### Server Startup
- Pipeline assembly: ~1-2 seconds (CPU, d_model=256, 2 layers)
- Compression cache hydration: 2 sessions loaded successfully
- Port binding: immediate after sanity check

### Compression Metrics (from `/metrics`)
- `axiom_savings_bytes_in_total: 0` — no live traffic has been compressed yet
- `axiom_savings_ratio: 0.0` — no baseline established
- All DWE counters at 0 (no fleet peers configured)

### CUDA Diagnostics
- Phase 3 d512/4-layer bake completed on CUDA:0 with val_ce=9.6776
- Step times: 824ms → 457ms (warming up), avg 622ms
- No CUDA errors in current logs (CPU mode is correct for this machine)

---

## Summary Priority Matrix

| Priority | Item | Effort | Impact |
|---|---|---|---|
| P0 | Fix Dockerfile | Medium | Docker CI broken |
| P0 | Enable vibe memory in proxy | Trivial | Context persistence |
| P0 | Register watchdog in Task Scheduler | Trivial | Auto-recovery |
| P1 | Fix log encoding | Low | Log readability |
| P1 | Raise compression threshold | Trivial | Reduce over-compression |
| P1 | Run eval_model for drift gate | Low | Drift calibration |
| P2 | MCP pipeline RwLock | Medium | Concurrent tool latency |
| P2 | MCP vibe flush on shutdown | Low | MCP context persistence |
| P2 | Compression cache versioning | Medium | Stale cache safety |
| P3 | MCP tool timeout | Low | Large dir safety |
| P3 | Memory store backup | Low | Data safety |