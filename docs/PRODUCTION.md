# AXIOM-AETHER — Production Readiness

Operational reference for running AXIOM-AETHER as a fleet of autonomous,
verifier-gated code-repair nodes. Pairs with the engine survey/roadmap in
`docs/research/ttt-engine-frontier-and-roadmap.md`.

## Build & run

**⚠️ The port-3000 proxy is a live, shared, supervised process — multiple
agent sessions on this machine may route through it concurrently. Killing or
overwriting the running `axiom_engine.exe` drops everyone's connection
(including your own, if `ANTHROPIC_BASE_URL` points at it) and can race with
another session's rebuild.**

- **Never `taskkill` / `Stop-Process` the running proxy directly** to "force"
  a rebuild. `cargo build` will fail with `Access is denied` while the binary
  is running (Windows file locking) — that failure is expected and is not a
  reason to kill the process.
- **To restart** (same binary, e.g. after an env var or checkpoint change),
  use the supervised script — it stops only the port-3000 instance (never
  MCP stdio or ChatGPT-connector instances) and relaunches it:
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File D:\AXIOM-AETHER\scripts\restart_proxy.ps1
  ```
- **To deploy a rebuilt binary** (source changed), the running `.exe` is
  locked — rename it aside to a **unique, timestamped** path first (never a
  fixed name like `axiom_engine.old.exe`: two concurrent sessions doing this
  at once will rename or overwrite each other's staged binary, potentially
  leaving neither a valid target to restart), then build and restart:

  Git Bash:
  ```bash
  STAMP=$(date +%Y%m%d-%H%M%S)
  mv axiom_engine_rs/target/release/axiom_engine.exe \
     "axiom_engine_rs/target/release/axiom_engine.$STAMP.exe"
  cd axiom_engine_rs && cargo build --release --locked && cd ..
  ```

  PowerShell:
  ```powershell
  $Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  Move-Item axiom_engine_rs\target\release\axiom_engine.exe `
            "axiom_engine_rs\target\release\axiom_engine.$Stamp.exe"
  Push-Location axiom_engine_rs; cargo build --release --locked; Pop-Location
  ```

  Then restart the same way regardless of which shell built it:
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File D:\AXIOM-AETHER\scripts\restart_proxy.ps1
  ```
  If `cargo build` fails because the source path was already moved by another
  session, someone else's deploy is in flight — wait and retry rather than
  re-renaming.
- If your own requests start failing with connection-refused after any of
  the above, the proxy is mid-restart or another session's — retry in a few
  seconds rather than intervening again.

```bash
# Release binary (the CI gate: cargo build --release --locked)
cd axiom_engine_rs
cargo build --release --locked          # → target/release/axiom_engine

# Verify before deploy
cargo test --lib                        # full library suite (287 tests)
cargo clippy --lib                      # production code lints (clean)
```

The binary is a single self-contained server/CLI; no attention KV-cache, O(1)
state per inference step (TTT fast-weights).

## Runtime configuration (environment)

| Variable | Purpose | Production guidance |
|---|---|---|
| `AXIOM_FLEET_KEY` | Current HMAC key gating patch/immunity gossip provenance and DWE fragments | **Required for fleet mode.** Set the same secret on every trusted node; `AXIOM_DWE_LISTEN` refuses to start without it. |
| `AXIOM_FLEET_KEY_PREV` | Previous fleet key accepted during rotation | Set only during a graceful key rotation window. Nodes sign outbound fragments with `AXIOM_FLEET_KEY` and accept inbound fragments signed by either current or previous key. Remove after all peers rotate. |
| `AXIOM_HEAL_MEMORY` | Path to persistent heal memory (`axiom_heal_memory.json`); patch store sits alongside it | Point at durable storage; `0`/`off` disables persistence. The `/v1/patches*` endpoints return **503** when unset. |
| `AXIOM_BACKEND` | LLM backend selection | Set per deployment. |
| `AXIOM_DEVICE` | Compute device (`cpu`/`cuda`) | Match the host. |

## Fleet gossip endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/fleet/status` | Live fleet status: DWE telemetry, configured peers, listen address, and current/previous key configuration booleans. |
| `GET /v1/patches` | This node's verified-patch store as a provenance-signed export. |
| `POST /v1/patches/merge` | Fold a peer's signed export in — **Byzantine-robust** (bounded per-peer trust), 32 MiB body limit. |
| `GET /v1/immunity` · `POST /v1/immunity/merge` | Environmental-immunity gossip (Dempster–Shafer, Byzantine-resistant). |

## Safety invariants (do not regress)

`GET /metrics` also exports `axiom_dwe_sent`, `axiom_dwe_received`, `axiom_dwe_applied`, and `axiom_dwe_rejected`. Use `axiom fleet status` for the live HTTP view, or `axiom fleet status --offline` when the server is not running and you only need local environment wiring.

1. **Re-verify before trust.** A peer's patch is *never* applied on trust. It is
   only written through `PatchMemory::try_candidates`, which writes, runs *this
   node's* verifier, and keeps it **only if it passes locally** — otherwise rolls
   back byte-for-byte. `verified_count` is a ranking signal, never authorization.
2. **Provenance gates ingestion.** SHA-256 + optional HMAC (`AXIOM_FLEET_KEY`)
   decides which exports are even considered.
3. **Byzantine robustness (FleetTrustPolicy::robust).** Even a key-holding peer
   cannot inflate trust (≤ +1 per candidate per merge), flood the store (≤ 8 new
   per fingerprint), or ship oversized candidates (≤ 8 MiB). Reported via
   `byzantine_rejected`.
4. **Held-out verification.** `agentic_loop_with_holdout` commits a fix only if
   it passes both the train and a held-out check, rejecting test-overfit patches.
5. **Byte-identical defaults.** Every TTT control (online guards, contrastive
   inner loss, learned/forget gate) defaults to a no-op; enabling is opt-in and
   reversible. The proven linear backbone is unchanged.

## TTT engine controls (opt-in)

Tunable at runtime across the whole stack without a checkpoint reload:

- `AxiomTTTLM::set_online_guards(drift_reset_norm, update_min_error, anchor_strength)`
  — bound persistent online adaptation (RDumb reset / EATA token-selection /
  anti-forgetting anchor). All `0.0` ⇒ disabled.
- `AxiomTTTLM::set_aux_loss_normalized(bool)` — contrastive multi-view inner loss.
- `AxiomMlpLM` — opt-in TTT-MLP backbone for expressivity; benchmark with
  `ttt_mlp_model::mlp_vs_linear_reconstruction` before switching ("measure first").
- Hindsight self-improvement: `hindsight::fine_tune` trains on the node's own
  verified fixes and returns a `FineTuneReport`; **gate checkpoint promotion
  behind the agentic-eval benchmark** before serving it.

## Pre-deploy checklist

- [ ] `cargo build --release --locked` succeeds.
- [ ] `cargo test --lib` green; `cargo clippy --lib` clean.
- [ ] `AXIOM_FLEET_KEY` set identically on all trusted nodes (or fleet mode off).
- [ ] `AXIOM_HEAL_MEMORY` on durable storage.
- [ ] Patch-merge endpoint confirmed using the robust trust policy (default).
- [ ] Any fine-tuned checkpoint passed the agentic-eval benchmark before promotion.
