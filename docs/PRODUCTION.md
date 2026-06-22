# AXIOM-AETHER — Production Readiness

Operational reference for running AXIOM-AETHER as a fleet of autonomous,
verifier-gated code-repair nodes. Pairs with the engine survey/roadmap in
`docs/research/ttt-engine-frontier-and-roadmap.md`.

## Build & run

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
| `AXIOM_FLEET_KEY` | HMAC key gating patch/immunity gossip provenance | **Required for fleet mode.** Set the same secret on every trusted node; absent ⇒ exports are unsigned and merges of signed exports are rejected. |
| `AXIOM_HEAL_MEMORY` | Path to persistent heal memory (`axiom_heal_memory.json`); patch store sits alongside it | Point at durable storage; `0`/`off` disables persistence. The `/v1/patches*` endpoints return **503** when unset. |
| `AXIOM_BACKEND` | LLM backend selection | Set per deployment. |
| `AXIOM_DEVICE` | Compute device (`cpu`/`cuda`) | Match the host. |

## Fleet gossip endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/patches` | This node's verified-patch store as a provenance-signed export. |
| `POST /v1/patches/merge` | Fold a peer's signed export in — **Byzantine-robust** (bounded per-peer trust), 32 MiB body limit. |
| `GET /v1/immunity` · `POST /v1/immunity/merge` | Environmental-immunity gossip (Dempster–Shafer, Byzantine-resistant). |

## Safety invariants (do not regress)

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
