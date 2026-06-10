//! `axiom bench <path>` — measure what compression actually buys.
//!
//! The project's headline numbers (cross-entropy, drift margins) are *internal*
//! signals. The number that proves the product is downstream: how much smaller
//! is the forwarded payload, and how much of the answerable structure survives?
//! A full answer-quality bench needs an upstream model to compare answers
//! with/without compression; that requires network egress and a key. This bench
//! measures the two things that are locally verifiable and deterministic:
//!
//! 1. **Token savings** — original heavy context vs the structural skeleton the
//!    proxy forwards, in real tokenizer tokens.
//! 2. **Structural fidelity** — every signature kept in the skeleton must round-
//!    trip: `expand_symbol` recovers its full body from the retained source.
//!    A 100% round-trip means no part of the API surface is lost — Claude can
//!    always recover any elided body on demand.
//!
//! It is intentionally honest about its limits: it does not claim an answer-
//! quality delta it cannot measure offline.

use std::path::Path;

use candle_core::Result;

use crate::inference::InferencePipeline;
use crate::prime::collect_source_files;
use crate::skeleton::{build_digest, expand_symbol};

const DEFAULT_MAX_FILES: usize = 2000;

/// Declaration keywords whose following identifier is a candidate symbol to
/// round-trip through `expand_symbol`.
const DECL_KEYWORDS: &[&str] = &[
    "fn ",
    "func ",
    "def ",
    "function ",
    "struct ",
    "enum ",
    "trait ",
    "interface ",
    "class ",
    "type ",
];

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

/// Pull the identifier that follows a declaration keyword on a signature line,
/// e.g. `pub fn run(` → `run`, `class Beta {` → `Beta`.
fn symbol_from_signature(line: &str) -> Option<String> {
    let t = line.trim_start();
    for kw in DECL_KEYWORDS {
        if let Some(rest) = t.split_once(kw).map(|(_, r)| r) {
            // Only treat it as this keyword if `kw` starts at a word boundary
            // (the prefix is empty or ends in a non-identifier char).
            let prefix = &t[..t.len() - rest.len() - kw.len()];
            if !prefix.is_empty()
                && prefix
                    .chars()
                    .last()
                    .map(|c| c.is_alphanumeric() || c == '_')
                    .unwrap_or(false)
            {
                continue;
            }
            let name: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    None
}

/// Aggregate bench result over a crawled tree.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BenchReport {
    pub files: usize,
    pub original_tokens: usize,
    pub skeleton_tokens: usize,
    pub symbols_total: usize,
    pub symbols_recovered: usize,
}

impl BenchReport {
    /// Fraction of tokens removed by skeletonization (0.0–1.0).
    pub fn savings_ratio(&self) -> f64 {
        if self.original_tokens == 0 {
            return 0.0;
        }
        1.0 - (self.skeleton_tokens as f64 / self.original_tokens as f64)
    }

    /// Fraction of skeleton signatures whose body round-trips via expand (0.0–1.0).
    pub fn fidelity_ratio(&self) -> f64 {
        if self.symbols_total == 0 {
            return 1.0;
        }
        self.symbols_recovered as f64 / self.symbols_total as f64
    }
}

/// Crawl `target`, skeletonize each source file, and measure token savings plus
/// expand round-trip fidelity.
pub fn run_bench(target: &Path, pipeline: &InferencePipeline) -> Result<BenchReport> {
    let max_files = env_usize("AXIOM_BENCH_MAX_FILES", DEFAULT_MAX_FILES);
    let files = collect_source_files(target, max_files);

    let mut report = BenchReport::default();

    for path in &files {
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if source.trim().is_empty() {
            continue;
        }
        let original_tokens = pipeline.token_count(&source);
        let digest = build_digest(&source, "bench", original_tokens, 0.0, "bench", 3);
        let skeleton_tokens = pipeline.token_count(&digest);

        report.files += 1;
        report.original_tokens += original_tokens;
        report.skeleton_tokens += skeleton_tokens;

        // Round-trip every signature retained in the digest.
        for line in digest.lines() {
            if let Some(symbol) = symbol_from_signature(line) {
                report.symbols_total += 1;
                if expand_symbol(&source, &symbol).is_some() {
                    report.symbols_recovered += 1;
                }
            }
        }
    }

    print_report(target, &report);
    Ok(report)
}

fn print_report(target: &Path, r: &BenchReport) {
    println!("[bench] target: {}", target.display());
    println!("[bench] files measured        : {}", r.files);
    println!("[bench] original tokens        : {}", r.original_tokens);
    println!("[bench] skeleton tokens        : {}", r.skeleton_tokens);
    println!(
        "[bench] token savings          : {:.1}%  ({} -> {} tokens)",
        r.savings_ratio() * 100.0,
        r.original_tokens,
        r.skeleton_tokens
    );
    println!(
        "[bench] signatures kept        : {} (expand round-trip {}/{} = {:.1}%)",
        r.symbols_total,
        r.symbols_recovered,
        r.symbols_total,
        r.fidelity_ratio() * 100.0
    );
    println!(
        "[bench] note: token savings + structural fidelity are measured offline. \
         Answer-quality delta needs an upstream model and is not asserted here."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_extraction_handles_keywords_and_boundaries() {
        assert_eq!(symbol_from_signature("pub fn run(a: u8) {").as_deref(), Some("run"));
        assert_eq!(symbol_from_signature("    def go(self):").as_deref(), Some("go"));
        assert_eq!(symbol_from_signature("class Beta {").as_deref(), Some("Beta"));
        // `transform` contains the substring "fn " mid-word — must NOT match it.
        assert_eq!(symbol_from_signature("let transform = 1;"), None);
        assert_eq!(symbol_from_signature("// just a comment"), None);
    }

    #[test]
    fn report_ratios_are_well_defined_on_empty() {
        let r = BenchReport::default();
        assert_eq!(r.savings_ratio(), 0.0);
        assert_eq!(r.fidelity_ratio(), 1.0); // no symbols → vacuously perfect
    }
}
