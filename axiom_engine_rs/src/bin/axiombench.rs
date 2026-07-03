//! AxiomBench — the proof layer.
//!
//! Runs the deterministic pillars (cognition, trust, fleet) always; the cost
//! pillar only with `--live`. Each pillar returns a `PillarResult`; the runner
//! prints them, writes `bench/results/<ts>.json`, and regenerates `RESULTS.md`.

#[path = "axiombench_cognition.rs"]
mod cognition;
#[path = "axiombench_fleet.rs"]
mod fleet;
#[path = "axiombench_trust.rs"]
mod trust;

use cognition::PillarResult;

fn print_result(r: &PillarResult) {
    println!("[{}] {}", r.name, r.headline);
}

/// Serialize all pillar results to a JSON summary (consumed by the results
/// writer in a later task; also printed so `detail` is always surfaced).
fn results_json(results: &[PillarResult]) -> serde_json::Value {
    serde_json::json!({
        "results": results.iter().map(|r| serde_json::json!({
            "name": r.name,
            "headline": r.headline,
            "detail": r.detail,
        })).collect::<Vec<_>>(),
    })
}

fn main() {
    let results = vec![
        cognition::run_cognition(),
        trust::run_trust(),
        fleet::run_fleet(),
    ];
    println!("== AxiomBench ==");
    for r in &results {
        print_result(r);
    }
    println!("{}", results_json(&results));
}
