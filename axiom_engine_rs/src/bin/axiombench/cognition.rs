//! Cognition pillar: structural skeleton round-trip fidelity.
//!
//! Axiom's compression elides function bodies into a digest; `expand_symbol`
//! must recover the exact declaration+body on demand. This pillar measures the
//! exact-recovery rate over a set of source samples.

use serde_json::json;

/// One competitive-axis measurement. Shared by every pillar.
pub struct PillarResult {
    pub name: String,
    pub headline: String,
    pub detail: serde_json::Value,
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
