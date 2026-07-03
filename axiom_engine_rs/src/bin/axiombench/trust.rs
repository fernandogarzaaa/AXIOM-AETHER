//! Trust pillar: calibrated conformal gate coverage on a held-out split.

use crate::cognition::PillarResult;
use axiom_engine::hallucination::{calibrate_conformal_threshold, verify, ConformalGate, Verdict};
use serde::Deserialize;
use serde_json::json;

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

fn score(r: &Row) -> f32 {
    verify(&r.claim, &r.evidence)
        .claims
        .first()
        .map(|c| c.support)
        .unwrap_or(0.0)
}

pub fn run_trust() -> PillarResult {
    let rows = load();
    // Even indices calibrate; odd indices are held out (matches the
    // trust_calibration integration test's split).
    let cal: Vec<(f32, bool)> = rows.iter().step_by(2).map(|r| (score(r), r.supported)).collect();
    let threshold = calibrate_conformal_threshold(&cal, 0.10);
    let gate = ConformalGate { threshold, delta: 0.10 };

    let holdout: Vec<&Row> = rows.iter().skip(1).step_by(2).collect();
    let pos: Vec<&&Row> = holdout.iter().filter(|r| r.supported).collect();
    let neg: Vec<&&Row> = holdout.iter().filter(|r| !r.supported).collect();
    let covered = pos
        .iter()
        .filter(|r| matches!(gate.verdict(score(r)), Verdict::Supported))
        .count();
    let false_pos = neg
        .iter()
        .filter(|r| matches!(gate.verdict(score(r)), Verdict::Supported))
        .count();
    let coverage = covered as f64 / pos.len().max(1) as f64;
    let fpr = false_pos as f64 / neg.len().max(1) as f64;

    PillarResult {
        name: "trust".into(),
        headline: format!(
            "{:.0}% supported-claim coverage, {:.0}% false-positive @ delta=0.10 (threshold {:.3})",
            coverage * 100.0,
            fpr * 100.0,
            threshold
        ),
        detail: json!({
            "threshold": threshold,
            "coverage": coverage,
            "false_positive_rate": fpr,
            "positives": pos.len(),
            "negatives": neg.len(),
        }),
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
