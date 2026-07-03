//! Fleet pillar: cross-node immunity transfer + fragment-auth invariant.

use crate::cognition::PillarResult;
use axiom_engine::dwe::{sign_fragment, verify_fragment, DweFragment, DweLayerDelta};
use axiom_engine::heal_memory::{fingerprint, HealMemory};
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

fn tmp(tag: &str) -> PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("axiombench_fleet_{tag}_{n}.json"))
}

pub fn run_fleet() -> PillarResult {
    // Node A learns a directory heal.
    let a_path = tmp("nodeA");
    let mut a = HealMemory::load(&a_path);
    let fp = fingerprint("prog", &[]);
    a.remember_dirs(&fp, "prog", &[PathBuf::from("/needed/dir")]);

    // Node B merges A's export → should be pre-immunized after the merge.
    let started = Instant::now();
    let b_path = tmp("nodeB");
    let mut b = HealMemory::load(&b_path);
    let merged = b.merge_json(&a.to_json()).is_ok();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let pre_immunized = b.record(&fp).is_some();

    // Security invariant: an unsigned fragment is rejected; a signed one verifies.
    let mut frag = DweFragment {
        schema: "axiom.dwe.v1".into(),
        session_id: "s".into(),
        sequence: 1,
        layers: vec![DweLayerDelta {
            layer_index: 0,
            shape: vec![1],
            values: vec![0.5],
        }],
        state_hash: "h".into(),
        hmac: None,
    };
    let unsigned_rejected = verify_fragment(&frag, b"k").is_err();
    sign_fragment(&mut frag, b"k");
    let signed_ok = verify_fragment(&frag, b"k").is_ok();

    let _ = std::fs::remove_file(&a_path);
    let _ = std::fs::remove_file(&b_path);

    let ok = merged && pre_immunized && unsigned_rejected && signed_ok;
    PillarResult {
        name: "fleet".into(),
        headline: format!(
            "node B pre-immunized in {elapsed_ms:.2} ms; fragment auth enforced ({})",
            if ok { "pass" } else { "FAIL" }
        ),
        detail: json!({
            "ok": ok,
            "pre_immunized": pre_immunized,
            "merge_ms": elapsed_ms,
            "unsigned_rejected": unsigned_rejected,
            "signed_ok": signed_ok,
        }),
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
