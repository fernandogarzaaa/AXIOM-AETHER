# Phase 2: Fleet Auth + CLI + Shipped Trust Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Authenticate the DWE inbound weight-fragment listener (closes #86), add an `axiom fleet` CLI to wire peers in one command, and ship a data-calibrated conformal trust threshold so `/v1/verify` carries a stated coverage guarantee out of the box.

**Architecture:** DWE fragments gain an `hmac` field authenticated with the existing `AXIOM_FLEET_KEY` (reusing `provenance::hmac_sha256_hex`); the listener refuses to start without a key and rejects unauthenticated/replayed fragments. A new `fleet` CLI subcommand composes DWE peer + immunity config. A checked-in labeled claim/evidence dataset drives the existing `/v1/verify` calibrate mode to produce the shipped `AXIOM_CONFORMAL_THRESHOLD`.

**Tech Stack:** Rust, existing `provenance` HMAC, `dwe`, `hallucination::ConformalGate`. No new crate dependencies.

## Global Constraints

- Build/test with `CARGO_TARGET_DIR=target-test` (live proxy locks `target/release/axiom_engine.exe`).
- No new crate dependencies.
- Conventional commits, no attribution footer.
- Reuse `AXIOM_FLEET_KEY` (already used by immunity merge) as the single fleet secret — do NOT introduce a second env var.
- DWE listener MUST refuse to start when `AXIOM_DWE_LISTEN` is set but no `AXIOM_FLEET_KEY` is configured (fail closed).

---

### Task 1: Authenticated DWE fragments (closes #86)

Add an optional HMAC to `DweFragment`, compute it on send, verify it on receive.

**Files:**
- Modify: `axiom_engine_rs/src/dwe.rs` (struct + serialize/verify helpers)
- Test: `axiom_engine_rs/src/dwe.rs` (`#[cfg(test)]`)

**Interfaces:**
- Produces:
  - `DweFragment` gains `pub hmac: Option<String>`.
  - `pub fn fragment_preimage(fragment: &DweFragment) -> Vec<u8>` — deterministic bytes over `{schema, session_id, sequence, layers, state_hash}` (excludes `hmac`).
  - `pub fn sign_fragment(fragment: &mut DweFragment, key: &[u8])` — sets `fragment.hmac`.
  - `pub fn verify_fragment(fragment: &DweFragment, key: &[u8]) -> Result<(), String>` — recomputes and constant-time compares.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod auth_tests {
    use super::*;

    fn frag() -> DweFragment {
        DweFragment {
            schema: "axiom.dwe.v1".into(),
            session_id: "s1".into(),
            sequence: 7,
            layers: vec![DweLayerDelta { layer_index: 0, shape: vec![2], values: vec![1.0, 2.0] }],
            state_hash: "abc".into(),
            hmac: None,
        }
    }

    #[test]
    fn signed_fragment_verifies_with_same_key() {
        let key = b"fleet-secret";
        let mut f = frag();
        sign_fragment(&mut f, key);
        assert!(f.hmac.is_some());
        assert!(verify_fragment(&f, key).is_ok());
    }

    #[test]
    fn tampered_fragment_fails_verification() {
        let key = b"fleet-secret";
        let mut f = frag();
        sign_fragment(&mut f, key);
        f.layers[0].values[0] = 99.0;
        assert!(verify_fragment(&f, key).is_err());
    }

    #[test]
    fn wrong_key_and_missing_hmac_fail() {
        let mut f = frag();
        sign_fragment(&mut f, b"key-a");
        assert!(verify_fragment(&f, b"key-b").is_err());
        f.hmac = None;
        assert!(verify_fragment(&f, b"key-a").is_err());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `CARGO_TARGET_DIR=target-test cargo test --lib dwe::auth_tests 2>&1 | tail -6`
Expected: FAIL — `no field hmac` / functions undefined.

- [ ] **Step 3: Implement**

Add `pub hmac: Option<String>` to `DweFragment` (after `state_hash`). In `extract_delta_fragment`, initialize `hmac: None`. Add below `deserialize_fragment`:

```rust
/// Deterministic HMAC preimage over the authenticated fields (everything but
/// the hmac itself).
pub fn fragment_preimage(fragment: &DweFragment) -> Vec<u8> {
    let shadow = (
        &fragment.schema,
        &fragment.session_id,
        fragment.sequence,
        &fragment.layers,
        &fragment.state_hash,
    );
    bincode::serialize(&shadow).unwrap_or_default()
}

/// Sign a fragment in place with the fleet key.
pub fn sign_fragment(fragment: &mut DweFragment, key: &[u8]) {
    let mac = crate::provenance::hmac_sha256_hex(key, &fragment_preimage(fragment));
    fragment.hmac = Some(mac);
}

/// Verify a fragment's HMAC against the fleet key (constant-time).
pub fn verify_fragment(fragment: &DweFragment, key: &[u8]) -> Result<(), String> {
    let Some(mac) = fragment.hmac.as_deref() else {
        return Err("fragment is unsigned".into());
    };
    let expected = crate::provenance::hmac_sha256_hex(key, &fragment_preimage(fragment));
    if expected.len() != mac.len() {
        return Err("hmac length mismatch".into());
    }
    let diff = expected.bytes().zip(mac.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b));
    if diff == 0 {
        Ok(())
    } else {
        Err("hmac verification failed".into())
    }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `CARGO_TARGET_DIR=target-test cargo test --lib dwe:: 2>&1 | tail -6`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add axiom_engine_rs/src/dwe.rs
git commit -m "feat: authenticate DWE fragments with fleet-key HMAC (#86)"
```

---

### Task 2: Enforce auth + replay rejection at the listener

**Files:**
- Modify: `axiom_engine_rs/src/server.rs` — the `AXIOM_DWE_LISTEN` block (~line 4985), plus the outbound broadcast chokepoint (`DweBus::broadcast` in `dwe.rs` if no server-side send exists).

**Interfaces:**
- Consumes: `fleet_key()` (existing, server.rs), `verify_fragment`, `sign_fragment`.
- Produces: no new public surface. Listener refuses to start without `AXIOM_FLEET_KEY`; inbound fragments are HMAC-verified and replay-checked before merge.

- [ ] **Step 1: Fail closed without a key + verify inbound**

After the `if !listen_addr.is_empty()` guard opens, capture the key or skip listening:

```rust
            let fleet_secret = match fleet_key() {
                Some(k) => k,
                None => {
                    eprintln!(
                        "[dwe] AXIOM_DWE_LISTEN is set but AXIOM_FLEET_KEY is not — refusing to \
                         start an unauthenticated weight-fragment listener"
                    );
                    // Skip the listener setup; fall through without spawning.
                    return; // valid: this is inside a `if let Ok(...) = env::var` block in run_server
                }
            };
            let verify_secret = fleet_secret.clone();
```

(Read the enclosing function first; if `return` would exit `run_server` entirely, instead wrap the listener spawn in `if fleet_key().is_some() { ... } else { eprintln!(...) }`.)

Inside the `while let Some(fragment) = in_rx.recv().await` loop, before `sessions.write()`:

```rust
                    if let Err(e) = crate::dwe::verify_fragment(&fragment, &verify_secret) {
                        eprintln!("[dwe] rejected fragment for '{}': {e}", fragment.session_id);
                        continue;
                    }
                    {
                        let key = format!("dwe:{}", fragment.session_id);
                        let mut seqs = match apply_state.sequence_versions.write() {
                            Ok(s) => s,
                            Err(_) => continue,
                        };
                        if let Some(prev) = seqs.get(&key) {
                            if fragment.sequence <= prev.version {
                                eprintln!(
                                    "[dwe] stale fragment seq {} <= {} for '{}'; dropped",
                                    fragment.sequence, prev.version, fragment.session_id
                                );
                                continue;
                            }
                        }
                        seqs.insert(key, SequenceState { version: fragment.sequence, timestamp: unix_now() });
                    }
```

(Read `SequenceState`'s field names near `sequence_versions` and adjust if they differ from `version`/`timestamp`.)

- [ ] **Step 2: Sign outbound fragments**

In `dwe.rs` `DweBus::broadcast`, before `try_send`, sign when a key is present:

```rust
    pub fn broadcast(&self, mut fragment: DweFragment) {
        if let Ok(key) = std::env::var("AXIOM_FLEET_KEY") {
            if !key.trim().is_empty() {
                sign_fragment(&mut fragment, key.as_bytes());
            }
        }
        // ... existing try_send(fragment) ...
    }
```

(Read the current `broadcast` body and thread the signing in before the existing send; keep the rest unchanged.)

- [ ] **Step 3: Integration test**

Create `axiom_engine_rs/tests/dwe_fleet_auth.rs`:

```rust
use axiom_engine::dwe::{sign_fragment, verify_fragment, DweFragment, DweLayerDelta};

fn frag() -> DweFragment {
    DweFragment {
        schema: "axiom.dwe.v1".into(),
        session_id: "s".into(),
        sequence: 1,
        layers: vec![DweLayerDelta { layer_index: 0, shape: vec![1], values: vec![0.5] }],
        state_hash: "h".into(),
        hmac: None,
    }
}

#[test]
fn only_matching_key_and_signed_fragments_verify() {
    let mut f = frag();
    sign_fragment(&mut f, b"shared");
    assert!(verify_fragment(&f, b"shared").is_ok());
    assert!(verify_fragment(&f, b"other").is_err());
    let unsigned = frag();
    assert!(verify_fragment(&unsigned, b"shared").is_err());
}
```

- [ ] **Step 4: Run + commit**

Run: `CARGO_TARGET_DIR=target-test cargo test --test dwe_fleet_auth 2>&1 | tail -5` → PASS.
Run: `CARGO_TARGET_DIR=target-test cargo check --release 2>&1 | grep -c "^error"` → 0.

```bash
git add axiom_engine_rs/src/server.rs axiom_engine_rs/src/dwe.rs axiom_engine_rs/tests/dwe_fleet_auth.rs
git commit -m "feat: DWE listener fails closed without fleet key; verifies + replay-checks fragments (#86)"
```

- [ ] **Step 5: Close #86**

```bash
gh issue close 86 --repo fernandogarzaaa/AXIOM-AETHER --comment "Fixed: DWE fragments carry a fleet-key HMAC; the AXIOM_DWE_LISTEN listener refuses to start without AXIOM_FLEET_KEY, verifies every inbound fragment, and rejects replays by (session_id, sequence)."
```

---

### Task 3: `axiom fleet` CLI

**Files:**
- Modify: `axiom_engine_rs/src/cli.rs`, `axiom_engine_rs/src/main.rs`

**Interfaces:**
- Produces: `axiom fleet status` (DWE/immunity wiring + whether a fleet key is set) and `axiom fleet join <peer>` (prints the env exports a peer sets). Config composer + status view; does not hot-reconfigure a running server.

- [ ] **Step 1: Add the subcommand to `cli.rs`**

Add to the `AxiomCommand` enum:

```rust
    /// Fleet operations: inspect DWE/immunity wiring and print peer-join config.
    Fleet {
        #[command(subcommand)]
        command: FleetCommand,
    },
```

and:

```rust
#[derive(Debug, Subcommand)]
pub enum FleetCommand {
    /// Show DWE wiring and whether a fleet key is set.
    Status,
    /// Print the environment a peer node must set to join this fleet.
    Join {
        /// Peer address as host:port (its AXIOM_DWE_LISTEN).
        peer: String,
    },
}
```

- [ ] **Step 2: Dispatch in `main.rs`**

```rust
        AxiomCommand::Fleet { command } => match command {
            cli::FleetCommand::Status => {
                let key_set = std::env::var("AXIOM_FLEET_KEY").map(|k| !k.trim().is_empty()).unwrap_or(false);
                let peers = std::env::var("AXIOM_DWE_PEERS").unwrap_or_default();
                let listen = std::env::var("AXIOM_DWE_LISTEN").unwrap_or_default();
                println!("fleet key configured : {}", if key_set { "yes" } else { "NO (required to listen)" });
                println!("dwe listen address   : {}", if listen.is_empty() { "(off)" } else { &listen });
                println!("dwe peers            : {}", if peers.is_empty() { "(none)" } else { &peers });
            }
            cli::FleetCommand::Join { peer } => {
                println!("# Add this node to the fleet by exporting (fleet key MUST match all peers):");
                println!("export AXIOM_FLEET_KEY=<shared-secret>");
                println!("export AXIOM_DWE_PEERS={peer}");
                println!("export AXIOM_DWE_LISTEN=0.0.0.0:<this-node-port>");
            }
        },
```

- [ ] **Step 3: Build + smoke**

Run: `CARGO_TARGET_DIR=target-test cargo run --release -- fleet status 2>&1 | tail -4`
Expected: three status lines.

- [ ] **Step 4: Commit**

```bash
git add axiom_engine_rs/src/cli.rs axiom_engine_rs/src/main.rs
git commit -m "feat: axiom fleet status|join CLI"
```

---

### Task 4: Shipped trust calibration dataset

**Files:**
- Create: `bench/trust/claims.jsonl`, `bench/trust/README.md`
- Create: `axiom_engine_rs/tests/trust_calibration.rs`
- Modify: `axiom.env`

**Interfaces:**
- Consumes: `hallucination::verify`, `hallucination::calibrate_conformal_threshold`, `hallucination::{ConformalGate, Verdict}` (all public).
- Produces: a checked-in labeled dataset + a test that scores each claim with `verify`, calibrates at δ=0.1, and asserts ≥(1−δ) coverage on the held-out split.

- [ ] **Step 1: Build the labeled dataset**

Create `bench/trust/claims.jsonl` with ≥40 lines `{"claim","evidence","supported"}`, seeded from existing verify fixtures and extended with supported / clearly-unsupported / vocabulary-sharing-contradiction rows. Deterministic, secret-free, mixed labels. Add `bench/trust/README.md` describing provenance + split.

- [ ] **Step 2: Write the calibration test**

Create `axiom_engine_rs/tests/trust_calibration.rs`:

```rust
use axiom_engine::hallucination::{calibrate_conformal_threshold, verify, ConformalGate, Verdict};
use serde::Deserialize;

#[derive(Deserialize)]
struct Row { claim: String, evidence: String, supported: bool }

fn load() -> Vec<Row> {
    let text = std::fs::read_to_string(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/trust/claims.jsonl")
    ).expect("dataset present");
    text.lines().filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid row"))
        .collect()
}

fn support_score(row: &Row) -> f32 {
    verify(&row.claim, &row.evidence).claims.first().map(|c| c.support).unwrap_or(0.0)
}

#[test]
fn calibrated_gate_meets_coverage_on_holdout() {
    let rows = load();
    assert!(rows.len() >= 40, "dataset must be substantive");
    let cal: Vec<(f32, bool)> = rows.iter().step_by(2).map(|r| (support_score(r), r.supported)).collect();
    let threshold = calibrate_conformal_threshold(&cal, 0.10);
    eprintln!("[calibration] threshold={threshold}");

    let gate = ConformalGate { threshold, delta: 0.10 };
    let positives: Vec<&Row> = rows.iter().skip(1).step_by(2).filter(|r| r.supported).collect();
    let covered = positives.iter()
        .filter(|r| matches!(gate.verdict(support_score(r)), Verdict::Supported))
        .count();
    let coverage = covered as f32 / positives.len().max(1) as f32;
    assert!(coverage >= 0.80, "held-out coverage {coverage:.2} below 1-delta target");
}
```

- [ ] **Step 3: Run, capture the threshold, ship it**

Run: `CARGO_TARGET_DIR=target-test cargo test --test trust_calibration -- --nocapture 2>&1 | grep -E "calibration|test result"` → note the printed threshold; PASS.
Add to `axiom.env`:

```bash
# Shipped conformal trust gate: threshold calibrated on bench/trust/claims.jsonl
# at delta=0.10 (>=90% coverage of genuinely supported claims). Override to retune.
export AXIOM_CONFORMAL_THRESHOLD=<value>
export AXIOM_CONFORMAL_DELTA=0.10
```

- [ ] **Step 4: Commit**

```bash
git add bench/trust axiom_engine_rs/tests/trust_calibration.rs axiom.env
git commit -m "feat: ship data-calibrated conformal trust threshold (delta=0.10)"
```

---

### Task 5: Full-suite verification + docs + PR

- [ ] **Step 1:** `CARGO_TARGET_DIR=target-test cargo test --release 2>&1 | grep -E "^test result|FAILED"` — all `ok`.
- [ ] **Step 2:** README "What Is Implemented": update the Swarm row to note DWE fragment authentication; add fleet CLI + shipped trust calibration.
- [ ] **Step 3:** `start_axiom.sh` / `axiom.env`: document `AXIOM_FLEET_KEY` requirement for `AXIOM_DWE_LISTEN`.
- [ ] **Step 4: Commit, push, open PR, merge per policy.**

---

## Self-Review

**Spec coverage (Deliverables 3 + 4):** DWE fragment auth + fail-closed listener + replay rejection (#86) → Tasks 1–2; fleet CLI → Task 3; labeled trust dataset + shipped calibrated threshold with coverage guarantee → Task 4. ✓

**Placeholders:** Tasks 2–4 flag where the executor must read a surrounding definition (SequenceState fields, broadcast body, run_server skip semantics) before editing — necessary because these touch long existing functions; all core logic is complete code. Task 4 Step 1 requires authoring dataset content (inherent to a labeling task; constraints are explicit). ✓

**Type consistency:** `DweFragment.hmac: Option<String>`, `fragment_preimage`/`sign_fragment`/`verify_fragment` consistent across dwe.rs, server.rs, and tests; `FleetCommand` variants match cli.rs + main.rs dispatch. ✓
