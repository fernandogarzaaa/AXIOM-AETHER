//! Trust pillar: calibrated conformal gate coverage on a held-out split.

use crate::cognition::PillarResult;
use axiom_engine::hallucination::{
    calibrate_conformal_threshold, deterministic_grounding_lift, verdict_after_neural_lift, verify,
    verify_with_signals, ConformalGate, Verdict,
};
use serde::Deserialize;
use serde_json::json;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Family {
    Supported,
    Unsupported,
    Contradiction,
}

impl Family {
    fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::Unsupported => "unsupported",
            Self::Contradiction => "contradiction",
        }
    }
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

fn score(r: &Row) -> f32 {
    verify(&r.claim, &r.evidence)
        .claims
        .first()
        .map(|c| c.support)
        .unwrap_or(0.0)
}

fn lexical_supported(row: &Row, gate: &ConformalGate) -> bool {
    matches!(gate.verdict(score(row)), Verdict::Supported)
}

fn neural_supported(row: &Row, gate: &ConformalGate) -> bool {
    verify_with_signals(&row.claim, &row.evidence, |claim| {
        deterministic_grounding_lift(claim, &row.evidence)
    })
    .claims
    .first()
    .map(|c| {
        matches!(
            verdict_after_neural_lift(gate.verdict(c.support), c.lift),
            Verdict::Supported
        )
    })
    .unwrap_or(false)
}

fn family_count(rows: &[&Row], family: Family) -> usize {
    rows.iter().filter(|r| r.family == family).count()
}

pub fn run_trust() -> PillarResult {
    let rows = load();
    // Even indices calibrate; odd indices are held out (matches the
    // trust_calibration integration test's split).
    let cal: Vec<(f32, bool)> = rows
        .iter()
        .step_by(2)
        .map(|r| (score(r), r.supported))
        .collect();
    let threshold = calibrate_conformal_threshold(&cal, 0.10);
    let gate = ConformalGate {
        threshold,
        delta: 0.10,
    };

    let holdout: Vec<&Row> = rows.iter().skip(1).step_by(2).collect();
    let pos: Vec<&Row> = holdout.iter().copied().filter(|r| r.supported).collect();
    let neg: Vec<&Row> = holdout.iter().copied().filter(|r| !r.supported).collect();
    let unsupported: Vec<&Row> = holdout
        .iter()
        .copied()
        .filter(|r| r.family == Family::Unsupported)
        .collect();
    let covered = pos
        .iter()
        .filter(|r| matches!(gate.verdict(score(r)), Verdict::Supported))
        .count();
    let false_pos = unsupported
        .iter()
        .filter(|r| lexical_supported(r, &gate))
        .count();
    let contradictions: Vec<&Row> = holdout
        .iter()
        .copied()
        .filter(|r| r.family == Family::Contradiction)
        .collect();
    let lexical_contradictions_caught = contradictions
        .iter()
        .filter(|r| !lexical_supported(r, &gate))
        .count();
    let neural_contradictions_caught = contradictions
        .iter()
        .filter(|r| !neural_supported(r, &gate))
        .count();
    let coverage = covered as f64 / pos.len().max(1) as f64;
    let fpr = false_pos as f64 / unsupported.len().max(1) as f64;
    let lexical_contradiction_catch_rate =
        lexical_contradictions_caught as f64 / contradictions.len().max(1) as f64;
    let neural_contradiction_catch_rate =
        neural_contradictions_caught as f64 / contradictions.len().max(1) as f64;

    let all_rows: Vec<&Row> = rows.iter().collect();

    PillarResult {
        name: "trust".into(),
        headline: format!(
            "{:.0}% supported coverage, neural contradiction catch-rate {:.0}% -> {:.0}% (threshold {:.3})",
            coverage * 100.0,
            lexical_contradiction_catch_rate * 100.0,
            neural_contradiction_catch_rate * 100.0,
            threshold
        ),
        detail: json!({
            "threshold": threshold,
            "coverage": coverage,
            "false_positive_rate": fpr,
            "positives": pos.len(),
            "negatives": neg.len(),
            "unsupported": unsupported.len(),
            "contradictions": contradictions.len(),
            "lexical_contradiction_catch_rate": lexical_contradiction_catch_rate,
            "neural_contradiction_catch_rate": neural_contradiction_catch_rate,
            "family_counts": {
                Family::Supported.as_str(): family_count(&all_rows, Family::Supported),
                Family::Unsupported.as_str(): family_count(&all_rows, Family::Unsupported),
                Family::Contradiction.as_str(): family_count(&all_rows, Family::Contradiction),
            },
            "holdout_family_counts": {
                Family::Supported.as_str(): family_count(&holdout, Family::Supported),
                Family::Unsupported.as_str(): family_count(&holdout, Family::Unsupported),
                Family::Contradiction.as_str(): family_count(&holdout, Family::Contradiction),
            },
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
        assert!(
            r.detail["neural_contradiction_catch_rate"]
                .as_f64()
                .unwrap()
                >= r.detail["lexical_contradiction_catch_rate"]
                    .as_f64()
                    .unwrap()
        );
    }
}
