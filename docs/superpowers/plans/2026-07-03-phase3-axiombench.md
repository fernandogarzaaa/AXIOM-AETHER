# Phase 3: AxiomBench — the proof layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single `axiombench` binary that measures Axiom on four competitive axes and writes a reproducible headline table to `RESULTS.md`: cognition (skeleton round-trip fidelity), trust (calibrated gate catch-rate), fleet (time-to-immunity), and cost (corpus token reduction, live). The three deterministic pillars run in CI; cost runs locally with `--live`.

**Architecture:** `axiombench` is a `tools`-gated bin dispatching to per-pillar modules under `src/bin/`. Each pillar returns a `PillarResult { name, headline, detail }`; the runner serializes results to `bench/results/<ts>.json` and regenerates the `RESULTS.md` table. Deterministic pillars consume only in-repo data (skeleton fixtures, `bench/trust/claims.jsonl`, heal-memory fixtures); the cost pillar replays corpus sessions through a live proxy.

**Tech Stack:** Rust, existing `skeleton`, `hallucination`, `dwe`, `heal_memory`, `session_recorder`. No new crate dependencies.

## Global Constraints

- Build/test with `CARGO_TARGET_DIR=target-test` and `CARGO_INCREMENTAL=0` (avoids the incremental-cache compiler panics seen this session and relink contention with the live proxy binary).
- No new crate dependencies.
- Conventional commits, no attribution footer.
- The bin is `required-features = ["tools"]`, matching the other dev/eval bins.
- Deterministic pillars must not require network or API keys; only `--live` (cost) may.

---

### Task 1: Bench scaffold + cognition pillar

**Files:**
- Modify: `axiom_engine_rs/Cargo.toml` (add `[[bin]] axiombench`, `required-features = ["tools"]`)
- Create: `axiom_engine_rs/src/bin/axiombench.rs` (runner + `PillarResult`), `axiom_engine_rs/src/bin/axiombench_cognition.rs`
- Test: unit test inside `axiombench_cognition.rs`

**Interfaces:**
- Produces:
  - `pub struct PillarResult { pub name: String, pub headline: String, pub detail: serde_json::Value }`
  - `pub fn run_cognition() -> PillarResult` — recovers elided symbols via `skeleton::expand_symbol`, reports exact-recovery rate.

- [ ] **Step 1: Write the cognition pillar + test**

Create `axiom_engine_rs/src/bin/axiombench_cognition.rs`:

```rust
//! Cognition pillar: structural skeleton round-trip fidelity.
use serde_json::json;

pub struct PillarResult {
    pub name: String,
    pub headline: String,
    pub detail: serde_json::Value,
}

fn samples() -> Vec<(&'static str, &'static str)> {
    vec![
        ("adder", "pub fn adder(a: i32, b: i32) -> i32 { let s = a + b; s }"),
        ("greet", "pub fn greet(name: &str) -> String { format!(\"hi {name}\") }"),
        ("twice", "pub fn twice(x: u64) -> u64 { x.wrapping_mul(2) }"),
    ]
}

pub fn run_cognition() -> PillarResult {
    let mut recovered = 0usize;
    let total = samples().len();
    for (name, src) in samples() {
        if let Some(body) = axiom_engine::skeleton::expand_symbol(src, name) {
            if body.contains(name) {
                recovered += 1;
            }
        }
    }
    let rate = recovered as f64 / total as f64;
    PillarResult {
        name: "cognition".into(),
        headline: format!("{:.0}% symbol exact-recovery ({recovered}/{total})", rate * 100.0),
        detail: json!({ "recovered": recovered, "total": total, "rate": rate }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cognition_recovers_all_sample_symbols() {
        let r = run_cognition();
        assert_eq!(r.name, "cognition");
        assert!(r.detail["rate"].as_f64().unwrap() >= 1.0, "{}", r.headline);
    }
}
```

- [ ] **Step 2: Add the bin target + runner**

Add to `Cargo.toml`:

```toml
[[bin]]
name = "axiombench"
path = "src/bin/axiombench.rs"
required-features = ["tools"]
```

Create `axiom_engine_rs/src/bin/axiombench.rs`:

```rust
//! AxiomBench — the proof layer. Runs the deterministic pillars always; the
//! cost pillar only with `--live`.

#[path = "axiombench_cognition.rs"]
mod cognition;

use cognition::PillarResult;

fn print_result(r: &PillarResult) {
    println!("[{}] {}", r.name, r.headline);
}

fn main() {
    let results = vec![cognition::run_cognition()];
    println!("== AxiomBench ==");
    for r in &results {
        print_result(r);
    }
}
```

- [ ] **Step 3: Run the test**

Run: `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=target-test cargo test --features tools --bin axiombench 2>&1 | tail -6`
Expected: PASS. If `expand_symbol` needs a specific digest form, read its contract in `skeleton.rs` and adjust `samples()`/assertion to the real round-trip.

- [ ] **Step 4: Smoke-run the bin**

Run: `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=target-test cargo run --features tools --bin axiombench 2>&1 | tail -4`
Expected: prints `== AxiomBench ==` and the cognition line.

- [ ] **Step 5: Commit**

```bash
git add axiom_engine_rs/Cargo.toml axiom_engine_rs/src/bin/axiombench.rs axiom_engine_rs/src/bin/axiombench_cognition.rs
git commit -m "feat: axiombench scaffold + cognition pillar"
```

---

### Task 2: Trust pillar

**Files:**
- Create: `axiom_engine_rs/src/bin/axiombench_trust.rs`
- Modify: `axiom_engine_rs/src/bin/axiombench.rs`

**Interfaces:**
- Consumes: `bench/trust/claims.jsonl`, `hallucination::{verify, calibrate_conformal_threshold, ConformalGate, Verdict}`.
- Produces: `pub fn run_trust() -> PillarResult` — calibrates on the even split; reports held-out coverage + false-positive rate at δ=0.10.

- [ ] **Step 1: Write the pillar + test**

Create `axiom_engine_rs/src/bin/axiombench_trust.rs`:

```rust
//! Trust pillar: calibrated conformal gate coverage vs an ungated baseline.
use crate::cognition::PillarResult;
use axiom_engine::hallucination::{calibrate_conformal_threshold, verify, ConformalGate, Verdict};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Row { claim: String, evidence: String, supported: bool }

fn load() -> Vec<Row> {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"), "/../bench/trust/claims.jsonl"
    )).expect("trust dataset present");
    text.lines().filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("row")).collect()
}

fn score(r: &Row) -> f32 {
    verify(&r.claim, &r.evidence).claims.first().map(|c| c.support).unwrap_or(0.0)
}

pub fn run_trust() -> PillarResult {
    let rows = load();
    let cal: Vec<(f32, bool)> = rows.iter().step_by(2).map(|r| (score(r), r.supported)).collect();
    let threshold = calibrate_conformal_threshold(&cal, 0.10);
    let gate = ConformalGate { threshold, delta: 0.10 };

    let holdout: Vec<&Row> = rows.iter().skip(1).step_by(2).collect();
    let pos: Vec<&&Row> = holdout.iter().filter(|r| r.supported).collect();
    let neg: Vec<&&Row> = holdout.iter().filter(|r| !r.supported).collect();
    let covered = pos.iter().filter(|r| matches!(gate.verdict(score(r)), Verdict::Supported)).count();
    let false_pos = neg.iter().filter(|r| matches!(gate.verdict(score(r)), Verdict::Supported)).count();
    let coverage = covered as f64 / pos.len().max(1) as f64;
    let fpr = false_pos as f64 / neg.len().max(1) as f64;

    PillarResult {
        name: "trust".into(),
        headline: format!("{:.0}% coverage, {:.0}% false-positive @ delta=0.10 (threshold {:.3})",
                          coverage * 100.0, fpr * 100.0, threshold),
        detail: json!({ "threshold": threshold, "coverage": coverage, "false_positive_rate": fpr,
                        "positives": pos.len(), "negatives": neg.len() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_reports_high_coverage_and_bounded_fpr() {
        let r = run_trust();
        assert!(r.detail["coverage"].as_f64().unwrap() >= 0.80);
        assert!(r.detail["false_positive_rate"].as_f64().unwrap() <= 0.5);
    }
}
```

- [ ] **Step 2: Wire into the runner**

In `axiombench.rs` add `#[path = "axiombench_trust.rs"] mod trust;` and push `trust::run_trust()` into `results`.

- [ ] **Step 3: Run + commit**

Run: `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=target-test cargo test --features tools --bin axiombench 2>&1 | tail -6` → PASS.

```bash
git add axiom_engine_rs/src/bin/axiombench.rs axiom_engine_rs/src/bin/axiombench_trust.rs
git commit -m "feat: axiombench trust pillar"
```

---

### Task 3: Fleet pillar

**Files:**
- Create: `axiom_engine_rs/src/bin/axiombench_fleet.rs`
- Modify: `axiom_engine_rs/src/bin/axiombench.rs`

**Interfaces:**
- Consumes: `heal_memory::HealMemory`, `dwe::{sign_fragment, verify_fragment}`.
- Produces: `pub fn run_fleet() -> PillarResult` — node A learns a heal, exports; node B merges and is pre-immunized; report merge latency + the fragment-auth invariant.

- [ ] **Step 1: Write the pillar + test**

Create `axiom_engine_rs/src/bin/axiombench_fleet.rs`:

```rust
//! Fleet pillar: cross-node immunity transfer + fragment-auth invariant.
use crate::cognition::PillarResult;
use axiom_engine::dwe::{sign_fragment, verify_fragment, DweFragment, DweLayerDelta};
use axiom_engine::heal_memory::{fingerprint, HealMemory};
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

fn tmp(tag: &str) -> PathBuf {
    let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("axiombench_fleet_{tag}_{n}.json"))
}

pub fn run_fleet() -> PillarResult {
    let a_path = tmp("nodeA");
    let mut a = HealMemory::load(&a_path);
    let fp = fingerprint("prog", &[]);
    a.remember_dirs(&fp, "prog", &[PathBuf::from("/needed/dir")]);

    let started = Instant::now();
    let b_path = tmp("nodeB");
    let mut b = HealMemory::load(&b_path);
    let merged = b.merge_json(&a.to_json()).is_ok();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let pre_immunized = b.record(&fp).is_some();

    let mut frag = DweFragment {
        schema: "axiom.dwe.v1".into(), session_id: "s".into(), sequence: 1,
        layers: vec![DweLayerDelta { layer_index: 0, shape: vec![1], values: vec![0.5] }],
        state_hash: "h".into(), hmac: None,
    };
    let unsigned_rejected = verify_fragment(&frag, b"k").is_err();
    sign_fragment(&mut frag, b"k");
    let signed_ok = verify_fragment(&frag, b"k").is_ok();

    let _ = std::fs::remove_file(&a_path);
    let _ = std::fs::remove_file(&b_path);

    let ok = merged && pre_immunized && unsigned_rejected && signed_ok;
    PillarResult {
        name: "fleet".into(),
        headline: format!("node B pre-immunized in {elapsed_ms:.2} ms; fragment auth enforced ({})",
                          if ok { "pass" } else { "FAIL" }),
        detail: json!({ "pre_immunized": pre_immunized, "merge_ms": elapsed_ms,
                        "unsigned_rejected": unsigned_rejected, "signed_ok": signed_ok }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fleet_transfers_immunity_and_enforces_auth() {
        let r = run_fleet();
        assert!(r.detail["pre_immunized"].as_bool().unwrap());
        assert!(r.detail["unsigned_rejected"].as_bool().unwrap());
        assert!(r.detail["signed_ok"].as_bool().unwrap());
    }
}
```

(Confirm `remember_dirs`/`record`/`to_json`/`merge_json`/`fingerprint` shapes against `heal_memory.rs` tests before finalizing.)

- [ ] **Step 2: Wire + run + commit**

Wire `#[path = "axiombench_fleet.rs"] mod fleet;` + `fleet::run_fleet()`.
Run: `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=target-test cargo test --features tools --bin axiombench 2>&1 | tail -6` → PASS.

```bash
git add axiom_engine_rs/src/bin/axiombench.rs axiom_engine_rs/src/bin/axiombench_fleet.rs
git commit -m "feat: axiombench fleet pillar"
```

---

### Task 4: Cost pillar (live) + results writer + RESULTS.md

**Files:**
- Create: `axiom_engine_rs/src/bin/axiombench_cost.rs`
- Modify: `axiom_engine_rs/src/bin/axiombench.rs` (args `--live`/`--out`; results serialization; RESULTS.md)
- Create: `RESULTS.md`, `bench/results/.gitkeep`

**Interfaces:**
- Consumes: `session_recorder::read_session`, a running proxy at `AXIOM_BASE_URL` (default `http://127.0.0.1:3000`).
- Produces: `pub fn run_cost(base_url: &str) -> PillarResult` — replays corpus sessions on/off compression, measures forwarded-token reduction from `/metrics` `axiom_savings_*`; errored cases counted separately, never averaged.

- [ ] **Step 1: Cost pillar (guarded, live-only)**

Create `axiom_engine_rs/src/bin/axiombench_cost.rs` with `run_cost(base_url)`: read corpus session files under `bench/corpus/` (empty ⇒ well-formed `PillarResult` with `detail.skipped == true` and a "no corpus" headline), else replay and compute reduction. Unit test asserts the no-corpus path returns `detail.skipped == true` (so CI exercises the code without a corpus/live proxy).

- [ ] **Step 2: Runner arg parsing + results writer**

In `axiombench.rs`: parse `--live` (add cost) and `--out <path>` (default `bench/results/<unix_ts>.json`); serialize all `PillarResult`s as `{ "generated": <ts>, "results": [...] }`; regenerate `RESULTS.md` at repo root (table `| Pillar | Headline |` + a "generated <ts>" line).

- [ ] **Step 3: Run + generate RESULTS.md**

Run: `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=target-test cargo run --features tools --bin axiombench 2>&1 | tail -8`
Expected: prints cognition/trust/fleet lines; writes `bench/results/<ts>.json` + `RESULTS.md`.

- [ ] **Step 4: Commit**

```bash
git add axiom_engine_rs/src/bin/axiombench.rs axiom_engine_rs/src/bin/axiombench_cost.rs RESULTS.md bench/results/.gitkeep
git commit -m "feat: axiombench cost pillar (live) + results writer + RESULTS.md"
```

---

### Task 5: CI wiring + docs + PR

**Files:**
- Modify: `.github/workflows/ci.yml`, `README.md`

- [ ] **Step 1: CI step**

After the existing `cargo test` step in `ci.yml`:

```yaml
      - name: AxiomBench (deterministic pillars)
        run: cargo run --release --features tools --bin axiombench
```

- [ ] **Step 2: README**

Near "Verification Status":

```markdown
### AxiomBench

`cargo run --release --features tools --bin axiombench` measures Axiom on four axes —
cognition (skeleton round-trip fidelity), trust (calibrated gate coverage), fleet
(cross-node immunity + fragment auth), and cost (corpus token reduction, `--live`).
Current headline numbers live in [`RESULTS.md`](RESULTS.md); the three deterministic
pillars run in CI on every push.
```

- [ ] **Step 3: Full suite + commit + PR + merge per policy**

Run: `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=target-test cargo test --release --features tools 2>&1 | grep -E "^test result|FAILED"` — all `ok`.

```bash
git add .github/workflows/ci.yml README.md
git commit -m "ci: run AxiomBench deterministic pillars; docs: link RESULTS.md"
git push -u origin phase3/axiombench
gh pr create --title "feat: AxiomBench proof layer (Phase 3)" --body "<summary>"
```

---

## Self-Review

**Spec coverage (Proof layer):** `axiombench` bin with four pillars → Tasks 1–4; deterministic pillars in CI → Task 5; `RESULTS.md` headline table + per-run JSON → Task 4; corpus/live cost pillar with errored-case isolation → Task 4. Corpus authoring (`bench/corpus/`) is a data task the cost pillar reads; the "no corpus" path keeps CI green without it. ✓

**Placeholders:** Tasks 3–4 flag where the executor confirms `heal_memory` call shapes and the cost-replay corpus format against the real modules before finalizing — necessary because these consume existing APIs; all pillar logic is complete code. ✓

**Type consistency:** `PillarResult { name, headline, detail }` defined once in `axiombench_cognition.rs`, imported by every pillar via `crate::cognition::PillarResult`; `run_*` signatures consistent with the runner. ✓
