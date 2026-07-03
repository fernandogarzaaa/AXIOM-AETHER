//! Fleet pillar: cross-node immunity transfer + DWE fragment apply/auth invariant.

use crate::cognition::PillarResult;
use axiom_engine::dwe::{
    apply_fragment, sign_fragment, verify_fragment, DweFragment, DweLayerDelta,
};
use axiom_engine::heal_memory::{fingerprint, HealMemory};
use candle_core::{DType, Device, Tensor};
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

    // Node B applies the signed DWE fragment into its local fast-weight state.
    let device = Device::Cpu;
    let mut node_b_layers = vec![Tensor::zeros((1,), DType::F32, &device).unwrap()];
    let apply_started = Instant::now();
    let dwe_applied = signed_ok && apply_fragment(&mut node_b_layers, &frag, &device).is_ok();
    let dwe_apply_ms = apply_started.elapsed().as_secs_f64() * 1000.0;
    let applied_value = node_b_layers[0]
        .to_vec1::<f32>()
        .ok()
        .and_then(|values| values.first().copied())
        .unwrap_or_default();

    let _ = std::fs::remove_file(&a_path);
    let _ = std::fs::remove_file(&b_path);

    let ok = merged && pre_immunized && unsigned_rejected && signed_ok && dwe_applied;
    PillarResult {
        name: "fleet".into(),
        headline: format!(
            "node B pre-immunized in {elapsed_ms:.2} ms; signed DWE applied in {dwe_apply_ms:.2} ms ({})",
            if ok { "pass" } else { "FAIL" }
        ),
        detail: json!({
            "ok": ok,
            "pre_immunized": pre_immunized,
            "merge_ms": elapsed_ms,
            "dwe_apply_ms": dwe_apply_ms,
            "dwe_applied": dwe_applied,
            "applied_value": applied_value,
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
        assert!(r.detail["dwe_applied"].as_bool().unwrap());
        assert_eq!(r.detail["applied_value"].as_f64().unwrap(), 0.5);
        assert!(r.detail["unsigned_rejected"].as_bool().unwrap());
        assert!(r.detail["signed_ok"].as_bool().unwrap());
    }
}
