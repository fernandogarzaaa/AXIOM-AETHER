use axiom_engine::hallucination::{
    calibrate_conformal_threshold, deterministic_grounding_lift, verify, verify_with_signals,
    ConformalGate, Verdict,
};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Family {
    Supported,
    Unsupported,
    Contradiction,
}

#[derive(Deserialize)]
struct Row {
    claim: String,
    evidence: String,
    supported: bool,
    family: Family,
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

fn lexical_supported(row: &Row, gate: &ConformalGate) -> bool {
    matches!(gate.verdict(support_score(row)), Verdict::Supported)
}

fn neural_supported(row: &Row) -> bool {
    verify_with_signals(&row.claim, &row.evidence, |claim| {
        deterministic_grounding_lift(claim, &row.evidence)
    })
    .claims
    .first()
    .map(|c| matches!(c.verdict, Verdict::Supported))
    .unwrap_or(false)
}

fn family_count(rows: &[Row], family: Family) -> usize {
    rows.iter().filter(|r| r.family == family).count()
}

#[test]
fn calibrated_gate_meets_family_coverage_on_holdout() {
    let rows = load();
    assert!(
        rows.len() >= 200,
        "dataset must be substantive ({} rows)",
        rows.len()
    );
    assert!(
        family_count(&rows, Family::Supported) > 0,
        "dataset must contain supported rows"
    );
    assert!(
        family_count(&rows, Family::Unsupported) > 0,
        "dataset must contain unsupported rows"
    );
    assert!(
        family_count(&rows, Family::Contradiction) * 3 >= rows.len(),
        "contradictions must be at least one third of the dataset"
    );

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
    let holdout: Vec<&Row> = rows.iter().skip(1).step_by(2).collect();
    for family in [
        Family::Supported,
        Family::Unsupported,
        Family::Contradiction,
    ] {
        assert!(
            holdout.iter().any(|r| r.family == family),
            "held-out set must contain {family:?} rows"
        );
    }

    let positives: Vec<&Row> = holdout.iter().copied().filter(|r| r.supported).collect();
    assert!(
        !positives.is_empty(),
        "held-out set must contain supported claims"
    );
    let covered = positives
        .iter()
        .filter(|r| lexical_supported(r, &gate))
        .count();
    let coverage = covered as f32 / positives.len() as f32;
    assert!(
        coverage >= 0.80,
        "held-out coverage {coverage:.2} below the 1-delta (0.90) target minus slack"
    );

    let unsupported: Vec<&Row> = holdout
        .iter()
        .copied()
        .filter(|r| r.family == Family::Unsupported)
        .collect();
    let unsupported_false_positive_rate = unsupported
        .iter()
        .filter(|r| lexical_supported(r, &gate))
        .count() as f32
        / unsupported.len() as f32;
    assert!(
        unsupported_false_positive_rate <= 0.50,
        "unsupported false-positive rate {unsupported_false_positive_rate:.2} is too high"
    );

    let contradictions: Vec<&Row> = holdout
        .iter()
        .copied()
        .filter(|r| r.family == Family::Contradiction)
        .collect();
    let lexical_catch_rate = contradictions
        .iter()
        .filter(|r| !lexical_supported(r, &gate))
        .count() as f32
        / contradictions.len() as f32;
    let neural_catch_rate = contradictions
        .iter()
        .filter(|r| !neural_supported(r))
        .count() as f32
        / contradictions.len() as f32;
    assert!(
        neural_catch_rate >= lexical_catch_rate,
        "neural tier worsened contradiction catch-rate ({neural_catch_rate:.2} < {lexical_catch_rate:.2})"
    );
    assert!(
        neural_catch_rate > lexical_catch_rate,
        "neural tier should improve contradiction catch-rate on the held-out split"
    );
}
