use axiom_engine::hallucination::{calibrate_conformal_threshold, verify, ConformalGate, Verdict};
use serde::Deserialize;

#[derive(Deserialize)]
struct Row {
    claim: String,
    evidence: String,
    supported: bool,
}

fn load() -> Vec<Row> {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../bench/trust/claims.jsonl"
    ))
    .expect("trust dataset present at bench/trust/claims.jsonl");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid dataset row"))
        .collect()
}

fn support_score(row: &Row) -> f32 {
    verify(&row.claim, &row.evidence)
        .claims
        .first()
        .map(|c| c.support)
        .unwrap_or(0.0)
}

#[test]
fn calibrated_gate_meets_coverage_on_holdout() {
    let rows = load();
    assert!(rows.len() >= 40, "dataset must be substantive ({} rows)", rows.len());

    // Deterministic split: even indices calibrate, odd indices are held out.
    let cal: Vec<(f32, bool)> = rows
        .iter()
        .step_by(2)
        .map(|r| (support_score(r), r.supported))
        .collect();
    let threshold = calibrate_conformal_threshold(&cal, 0.10);
    eprintln!("[calibration] threshold={threshold}");

    let gate = ConformalGate {
        threshold,
        delta: 0.10,
    };
    let positives: Vec<&Row> = rows
        .iter()
        .skip(1)
        .step_by(2)
        .filter(|r| r.supported)
        .collect();
    assert!(!positives.is_empty(), "held-out set must contain supported claims");
    let covered = positives
        .iter()
        .filter(|r| matches!(gate.verdict(support_score(r)), Verdict::Supported))
        .count();
    let coverage = covered as f32 / positives.len() as f32;
    assert!(
        coverage >= 0.80,
        "held-out coverage {coverage:.2} below the 1-delta (0.90) target minus slack"
    );
}
