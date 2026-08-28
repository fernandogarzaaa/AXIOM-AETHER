# AXIOM-AETHER — Production Readiness

Operational reference for running AXIOM-AETHER as a service: a single inference
node, or a fleet of autonomous verifier-gated code-repair nodes that share
learned state. Pairs with the engine roadmap in
`docs/research/ttt-engine-frontier-and-roadmap.md`.

The self-contained binary is both the server and the CLI. It holds **O(1) state
per inference step** (TTT fast-weights, no attention KV-cache), so memory does
not grow with context length.

---

## Glossary

| Term | Meaning here |
|---|---|
| **TTT** | Test-Time Training — the engine updates per-session fast weights (`W̃`) while processing context, instead of keeping frozen weights. |
| **fast-weights / `W̃`** | The per-session weight matrices TTT updates. They *are* the session's memory; there is no token KV-cache. |
| **CVM** | Context Virtual Memory — the proxy's cost design: provider prompt cache as L1, a local content-addressed store as L2. |
| **DWE** | Distributed Weight Exchange — fleet nodes gossip signed fast-weight *fragments* and verified patches. |
| **fleet mode** | Two or more nodes sharing immunity/patch state over the `/v1/fleet/*`, `/v1/cluster/*`, `/v1/immunity/*` endpoints. Off unless `AXIOM_FLEET_KEY` is set. **Experimental** — see [`EXPERIMENTAL.md`](EXPERIMENTAL.md). |
| **immunity** | Learned environment heals (e.g. "create this missing directory") remembered across runs and, in fleet mode, across nodes. |
| **HMAC** | Keyed hash (`AXIOM_FLEET_KEY`) that authenticates fleet gossip so only key-holders are considered. |
| **held-out split** | A fix is committed only if it passes both the train verifier and a separate held-out check — rejects test-overfit patches. |

---

## 1. Build

From a clean checkout, in `axiom_engine_rs/`:

```bash
cargo build --release --locked --bin axiom     # → target/release/axiom  (alias: axiom_engine)
```

The release CI gate is exactly `cargo build --release --locked`
(`.github/workflows/ci.yml`). Prebuilt binaries and the multi-arch container
image are published per release (`ghcr.io/fernandogarzaaa/axiom-aether`).

---

## 2. Verify before deploy

Run these four checks in order. Each states the command, the expected output,
and what "pass" means. All are CPU-only and need no network or API key.

### 2.1 Library + integration tests

```bash
cargo test --release --locked
```

**Expected:** `test result: ok.` for every binary; zero failures.
**Pass:** the whole suite is green (this includes the epistemic-drift and
server-route integration tests under `tests/`).

### 2.2 Lints

```bash
cargo clippy --lib --locked -- -D warnings
```

**Expected:** `Finished` with no warnings.
**Pass:** the library (the shipped product) is warning-free. Bin/test targets
carry known dead-code warnings and are not gated here.

### 2.3 End-to-end smoke

```bash
./scripts/demo_end_to_end.sh          # from the repo root
```

**Expected:** a `PASS` line for every pillar — doctor, compression, self-healing
runtime, learned immunity, solve loop, grounding — and a final
`PASS n  FAIL 0`.
**Pass:** `FAIL 0`. Runs in a throwaway temp dir; does not touch `~/.axiom`.

### 2.4 Autonomous-repair benchmark

```bash
cargo run --release --bin axiom -- eval-agentic
```

**Expected:** one `[PASS]` per built-in fixture and a final
`score: N/N = 100%`.
**Pass:** `N/N` (exit 0). The command exits non-zero if any fixture regresses.
This is the spearhead metric — see [`BENCH-REPAIR.md`](BENCH-REPAIR.md).

If a checkpoint is present it is loaded (`[+] Loaded checkpoint …`); if not, the
loop still runs — the built-in fixtures are repaired by the deterministic
Poly-JIT layer with no model required.

---

## 3. Runtime configuration

Full environment reference: README → *CLI Commands* → *Useful environment
variables*. The production-relevant subset:

| Variable | Purpose | Production guidance |
|---|---|---|
| `AXIOM_API_KEY` | Data-plane auth (`X-Axiom-Key` header). | **Set it before binding off `127.0.0.1`.** Unset ⇒ open. Serve over HTTPS/TLS — it is a bearer secret. |
| `AXIOM_BACKEND` | LLM backend for generation (`anthropic`, `openai`, `opendrop`, `router`). | Set per deployment. `axiom run`/`solve`/`eval-agentic` need no backend. |
| `AXIOM_DEVICE` | `cpu` / `cuda` / `metal` / `auto`. | Match the host. |
| `AXIOM_HEAL_MEMORY` | Path to persistent learned-immunity JSON. | Point at durable storage. `0`/`off` disables persistence; `/v1/patches*` then return `503`. |
| `AXIOM_CHECKPOINT_URL` / `AXIOM_TOKENIZER_URL` | Fetch the trained checkpoint + tokenizer at boot. | Pair with `AXIOM_CHECKPOINT_SHA256` / `AXIOM_TOKENIZER_SHA256` so a substituted asset fails closed. |
| `AXIOM_ENABLE_JIT_EXEC` | Enables `POST /v1/hypervisor/jit_run` (runs a caller-supplied command, can overwrite files). | **Leave unset** unless you specifically need it; if set, `AXIOM_API_KEY` must be set too. Experimental. |
| `AXIOM_FLEET_KEY` | HMAC key for fleet gossip + DWE fragments. | **Required for fleet mode**, identical on every trusted node. Experimental. |

---

## 4. Deploy

### Single node (container)

```bash
docker run -p 3000:8080 \
  -e AXIOM_API_KEY="$(openssl rand -hex 32)" \
  -e AXIOM_CHECKPOINT_URL=https://github.com/fernandogarzaaa/AXIOM-AETHER/releases/latest/download/axiom_production_bpe.bin \
  -e AXIOM_TOKENIZER_URL=https://github.com/fernandogarzaaa/AXIOM-AETHER/releases/latest/download/axiom_bpe.json \
  ghcr.io/fernandogarzaaa/axiom-aether:latest
```

Health: `GET /healthz` → `{"status":"ok"}`; readiness: `GET /readyz`; metrics:
`GET /metrics` (Prometheus). The image `HEALTHCHECK` already probes `/healthz`.

### Kubernetes

`helm install axiom deploy/helm/axiom` — probes, optional Prometheus
`ServiceMonitor`, conditional PodDisruptionBudget, checkpoint download, and a
`values-gpu.yaml` overlay. Full guide: [`DEPLOY_K8S.md`](DEPLOY_K8S.md).

### Fleet mode (experimental)

Set `AXIOM_FLEET_KEY` identically on all nodes and configure peers via
`axiom fleet join <peer>`. Gossip endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /v1/fleet/status` | DWE telemetry, configured peers, key config. |
| `GET /v1/patches` · `POST /v1/patches/merge` | Verified-patch store as a provenance-signed export; merge is Byzantine-robust, 32 MiB body limit. |
| `GET /v1/immunity` · `POST /v1/immunity/merge` | Environment-immunity gossip (Dempster–Shafer, Byzantine-resistant). |

`GET /metrics` also exports `axiom_dwe_{sent,received,applied,rejected}`. Use
`axiom fleet status` for the live view, or `--offline` for local wiring only.

---

## 5. Safety invariants (do not regress)

1. **Re-verify before trust.** A peer's patch is never applied on trust — only
   through `PatchMemory::try_candidates`, which writes it, runs *this node's*
   verifier, and keeps it only if it passes locally, else rolls back
   byte-for-byte. `verified_count` is a ranking signal, never authorization.
2. **Provenance gates ingestion.** SHA-256 + optional HMAC (`AXIOM_FLEET_KEY`)
   decides which exports are even considered.
3. **Byzantine robustness** (`FleetTrustPolicy::robust`, the default). A
   key-holding peer still cannot inflate trust (≤ +1 per candidate per merge),
   flood the store (≤ 8 new per fingerprint), or ship oversized candidates
   (≤ 8 MiB). Reported via `byzantine_rejected`.
4. **Held-out verification.** `agentic_loop_with_holdout` commits a fix only if
   it passes both the train and a held-out check.
5. **Byte-identical defaults.** Every TTT control (online guards, contrastive
   inner loss, learned/forget gate) defaults to a no-op; enabling is opt-in and
   reversible. The proven linear backbone is unchanged.

### TTT engine controls (opt-in, runtime-tunable, no reload)

- `AxiomTTTLM::set_online_guards(drift_reset_norm, update_min_error, anchor_strength)`
  — bound persistent online adaptation (reset / token-selection / anti-forgetting
  anchor). All `0.0` ⇒ disabled.
- `AxiomTTTLM::set_aux_loss_normalized(bool)` — contrastive multi-view inner loss.
- `AxiomMlpLM` — opt-in TTT-MLP backbone; benchmark with
  `ttt_mlp_model::mlp_vs_linear_reconstruction` before switching.
- `hindsight::fine_tune` trains on the node's own verified fixes and returns a
  `FineTuneReport`. **Gate checkpoint promotion behind the autonomous-repair
  benchmark** (`eval-agentic`) before serving a fine-tuned checkpoint —
  `scripts/promote_d384_on_pass.sh` is the reference gate.

---

## 6. Pre-deploy checklist

- [ ] `cargo build --release --locked` succeeds on a clean checkout.
- [ ] `cargo test --release --locked` green; `cargo clippy --lib --locked -- -D warnings` clean.
- [ ] `./scripts/demo_end_to_end.sh` ends `FAIL 0`.
- [ ] `axiom eval-agentic` reports `N/N = 100%`.
- [ ] `AXIOM_API_KEY` set (any bind other than `127.0.0.1`), served over TLS.
- [ ] `AXIOM_HEAL_MEMORY` on durable storage.
- [ ] `AXIOM_ENABLE_JIT_EXEC` unset unless required (and `AXIOM_API_KEY` set if it is).
- [ ] Fleet: `AXIOM_FLEET_KEY` identical on all nodes, or fleet mode off.
- [ ] Any fine-tuned checkpoint passed `eval-agentic` before promotion.

---

## Appendix: single-host shared-proxy operator notes

*Applies only to a developer machine where several agent sessions route through
one long-lived local proxy (`ANTHROPIC_BASE_URL` pointed at it). Not relevant to
a container or K8s deployment, where each pod owns its process.*

The running binary is a shared process: killing or overwriting it drops every
session routed through it and can race a concurrent rebuild.

- **Do not `taskkill` / `Stop-Process` the proxy to force a rebuild.** On
  Windows `cargo build` fails with `Access is denied` while the binary runs —
  that is expected file locking, not a reason to kill it.
- **Restart in place** (after an env or checkpoint change) with the supervised
  script, which stops only the proxy instance (never MCP-stdio or connector
  instances) and relaunches it:
  ```bash
  scripts/restart_proxy.ps1          # PowerShell; path is repo-relative
  ```
- **Deploy a rebuilt binary:** the running `.exe` is locked, so move it aside to
  a **unique timestamped** path (never a fixed name — two concurrent deploys
  would clobber each other's staged binary), then build and restart:
  ```bash
  STAMP=$(date +%Y%m%d-%H%M%S)
  mv axiom_engine_rs/target/release/axiom_engine.exe \
     "axiom_engine_rs/target/release/axiom_engine.$STAMP.exe"
  ( cd axiom_engine_rs && cargo build --release --locked )
  scripts/restart_proxy.ps1
  ```
  If `cargo build` fails because the source path was already moved, another
  session's deploy is in flight — wait and retry rather than re-renaming.
- If your requests start failing with connection-refused after any of the above,
  the proxy is mid-restart — retry in a few seconds rather than intervening
  again.
