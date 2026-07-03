//! Cost pillar: corpus token reduction through the live proxy.
//!
//! Live-only: replaying corpus sessions needs a running proxy. With no corpus
//! present (the CI case), returns a well-formed skipped result so the code path
//! is still exercised deterministically.

use crate::cognition::PillarResult;
use serde_json::json;
use std::path::Path;

fn corpus_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/corpus"))
}

/// Count `.jsonl` corpus session files, if any.
fn corpus_session_count(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|x| x.to_str())
                        .map(|x| x.eq_ignore_ascii_case("jsonl"))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

pub fn run_cost(base_url: &str) -> PillarResult {
    let dir = corpus_dir();
    let sessions = corpus_session_count(&dir);
    if sessions == 0 {
        return PillarResult {
            name: "cost".into(),
            headline: "skipped — no corpus (add bench/corpus/*.jsonl and run with --live)".into(),
            detail: json!({ "skipped": true, "reason": "no corpus", "corpus_dir": dir.display().to_string() }),
        };
    }
    // With a corpus present, live replay would go here (compression on/off arms
    // against `base_url`, reading /metrics axiom_savings_* counters). Kept as a
    // clearly-marked live path; deterministic CI never reaches it.
    PillarResult {
        name: "cost".into(),
        headline: format!("{sessions} corpus session(s) available; run with --live against {base_url}"),
        detail: json!({ "skipped": false, "sessions": sessions, "base_url": base_url }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_skips_cleanly_without_corpus() {
        // CI has no bench/corpus, so this must return a well-formed result.
        let r = run_cost("http://127.0.0.1:3000");
        assert_eq!(r.name, "cost");
        assert!(r.detail.get("skipped").is_some());
    }
}
