//! Cognition pillar: structural skeleton round-trip fidelity.
//!
//! Axiom's compression elides function bodies into a digest; `expand_symbol`
//! must recover the exact declaration+body on demand. This pillar measures the
//! exact-recovery rate over a set of source samples.

use serde_json::json;

/// One competitive-axis measurement. Shared by every pillar.
#[derive(Default)]
pub struct PillarResult {
    pub name: String,
    pub headline: String,
    pub detail: serde_json::Value,
    /// Sample size behind `headline`, when meaningful -- lets a report
    /// distinguish a measured rate from a smoke-check count. `None` when the
    /// pillar was skipped or a count isn't meaningful.
    pub sample_n: Option<u64>,
    /// One short phrase on how to read `headline` -- "smoke check, not a
    /// rate", "calibrated result", etc. Carried alongside the number so a
    /// regenerated RESULTS.md can't drop the caveat that makes the number
    /// trustworthy (see `write_results_md` in `main.rs`).
    pub read_as: Option<String>,
}

fn samples() -> Vec<(&'static str, &'static str)> {
    vec![
        ("adder", "pub fn adder(a: i32, b: i32) -> i32 {\n    let s = a + b;\n    s\n}"),
        ("greet", "pub fn greet(name: &str) -> String {\n    format!(\"hi {name}\")\n}"),
        ("twice", "pub fn twice(x: u64) -> u64 {\n    x.wrapping_mul(2)\n}"),
    ]
}

/// A recovery is *exact* when the block `expand_symbol` returns reproduces the
/// original single-declaration source verbatim (whitespace-normalized, so
/// trailing-newline handling doesn't count as a mismatch) — not merely mentions
/// the symbol name.
fn is_exact_recovery(recovered: &str, original: &str) -> bool {
    recovered.split_whitespace().eq(original.split_whitespace())
}

pub fn run_cognition() -> PillarResult {
    let mut recovered = 0usize;
    let total = samples().len();
    for (name, src) in samples() {
        if let Some(body) = axiom_engine::skeleton::expand_symbol(src, name) {
            if is_exact_recovery(&body, src) {
                recovered += 1;
            }
        }
    }
    let rate = recovered as f64 / total as f64;
    PillarResult {
        name: "cognition".into(),
        headline: format!("{:.0}% symbol exact-recovery ({recovered}/{total})", rate * 100.0),
        detail: json!({ "recovered": recovered, "total": total, "rate": rate }),
        sample_n: Some(total as u64),
        read_as: Some("smoke check, not a rate".into()),
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
