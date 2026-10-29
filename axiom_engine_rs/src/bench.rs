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

use std::path::{Path, PathBuf};

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
    // Doc/comment lines are kept verbatim in the digest — they are not
    // expandable code signatures. Skip them so a decl keyword appearing as an
    // ordinary English word in prose (e.g. "…trait to Axiom", "elides function
    // bodies…") isn't mis-parsed into a bogus symbol ("to", "bodies") that then
    // fails to round-trip and understates fidelity. This is a benchmark-harness
    // fix only; the compressor itself always kept these lines correctly.
    const COMMENT_PREFIXES: &[&str] = &["//", "/*", "*", "#", "--", "<!--"];
    if COMMENT_PREFIXES.iter().any(|p| t.starts_with(p)) {
        return None;
    }
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

/// Extract the candidate symbols from a digest, skipping comment content.
///
/// `symbol_from_signature` is line-local and drops lines with a leading comment
/// marker, but a `/* … */` block comment can carry bare interior lines (prose
/// with no `*` prefix) that would otherwise be mis-parsed as a signature — e.g.
/// a line reading `trait to Axiom` inside a header block. Tracking block-comment
/// state across lines here closes that gap so the fidelity denominator counts
/// only real signatures.
fn symbols_in_digest(digest: &str) -> Vec<String> {
    let mut symbols = Vec::new();
    let mut in_block_comment = false;
    for line in digest.lines() {
        let trimmed = line.trim_start();
        if in_block_comment {
            if trimmed.contains("*/") {
                in_block_comment = false;
            }
            continue;
        }
        // A block comment that opens and does not close on the same line starts
        // a skipped region (single-line `/* … */` is handled by the comment
        // prefix guard in `symbol_from_signature`).
        if trimmed.starts_with("/*") && !trimmed.contains("*/") {
            in_block_comment = true;
            continue;
        }
        if let Some(symbol) = symbol_from_signature(line) {
            symbols.push(symbol);
        }
    }
    symbols
}

/// Aggregate bench result over a crawled tree.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BenchReport {
    pub files: usize,
    pub original_tokens: usize,
    pub skeleton_tokens: usize,
    pub symbols_total: usize,
    pub symbols_recovered: usize,
    /// Signatures `expand_symbol` could not recover, with their source files.
    pub unrecovered: Vec<UnrecoveredSymbol>,
}

/// One signature that failed to round-trip through `expand_symbol`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnrecoveredSymbol {
    /// Source file the unrecoverable signature was extracted from.
    pub file: PathBuf,
    /// The symbol `expand_symbol` could not recover.
    pub symbol: String,
}

/// Options for `run_bench`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BenchOptions {
    /// Print one `[bench] UNRECOVERED <file>: <symbol>` line per signature
    /// that failed to round-trip.
    pub verbose: bool,
    /// Return an error (after printing the report) when any signature failed
    /// to round-trip, so CI can gate on 100% fidelity.
    pub strict: bool,
    /// Use PageRank-ranked skeletonization instead of the default unranked.
    pub ranked: bool,
    /// Token budget for ranked skeleton output. Keeps highest-ranked symbols.
    pub budget: Option<usize>,
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

    /// True when every retained signature round-tripped (vacuously true when
    /// no signatures were kept).
    pub fn is_lossless(&self) -> bool {
        self.symbols_recovered == self.symbols_total
    }
}

/// Round-trip every signature in `digest` against `source`, returning the
/// ones `expand_symbol` could not recover (with the file they came from).
fn unrecovered_in_file(source: &str, digest: &str, file: &Path) -> Vec<UnrecoveredSymbol> {
    symbols_in_digest(digest)
        .into_iter()
        .filter(|symbol| expand_symbol(source, symbol).is_none())
        .map(|symbol| UnrecoveredSymbol {
            file: file.to_path_buf(),
            symbol,
        })
        .collect()
}

/// Crawl `target`, skeletonize each source file, and measure token savings plus
/// expand round-trip fidelity.
///
/// With `BenchOptions::verbose`, unrecoverable signatures are printed as
/// `[bench] UNRECOVERED <file>: <symbol>`. With `BenchOptions::strict`, a
/// non-lossless result is returned as an error after the report is printed.
pub fn run_bench(
    target: &Path,
    pipeline: &InferencePipeline,
    opts: BenchOptions,
) -> Result<BenchReport> {
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
        let digest = if opts.ranked {
            // PageRank-ranked skeletonization: symbols ordered by importance.
            let lang = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            crate::skeleton::skeletonize_ranked(&source, lang, opts.budget)
        } else {
            build_digest(&source, "bench", original_tokens, 0.0, "bench", 3)
        };
        let skeleton_tokens = pipeline.token_count(&digest);

        report.files += 1;
        report.original_tokens += original_tokens;
        report.skeleton_tokens += skeleton_tokens;

        // Round-trip every signature retained in the digest.
        let kept = symbols_in_digest(&digest).len();
        let failures = unrecovered_in_file(&source, &digest, path);
        report.symbols_total += kept;
        report.symbols_recovered += kept - failures.len();
        report.unrecovered.extend(failures);
    }

    print_report(target, &report, opts.verbose);

    if opts.strict && !report.is_lossless() {
        let failed = report.symbols_total - report.symbols_recovered;
        let total = report.symbols_total;
        println!("[bench] STRICT: {failed}/{total} signatures failed round-trip");
        return Err(candle_core::Error::Msg(format!(
            "bench --strict: {failed}/{total} signatures failed round-trip"
        )));
    }

    Ok(report)
}

fn print_report(target: &Path, r: &BenchReport, verbose: bool) {
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
    if verbose {
        for u in &r.unrecovered {
            println!("[bench] UNRECOVERED {}: {}", u.file.display(), u.symbol);
        }
    }
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
        assert_eq!(
            symbol_from_signature("pub fn run(a: u8) {").as_deref(),
            Some("run")
        );
        assert_eq!(
            symbol_from_signature("    def go(self):").as_deref(),
            Some("go")
        );
        assert_eq!(
            symbol_from_signature("class Beta {").as_deref(),
            Some("Beta")
        );
        // `transform` contains the substring "fn " mid-word — must NOT match it.
        assert_eq!(symbol_from_signature("let transform = 1;"), None);
        assert_eq!(symbol_from_signature("// just a comment"), None);
        // Doc-comment prose that happens to contain a decl keyword as an
        // ordinary word must NOT be mis-parsed into a symbol (these two lines
        // were the only "failures" in the src/ round-trip benchmark).
        assert_eq!(
            symbol_from_signature("//! Bridges the router's trait to Axiom's real backend"),
            None
        );
        assert_eq!(
            symbol_from_signature("//! Axiom's compression elides function bodies into a digest"),
            None
        );
        assert_eq!(symbol_from_signature("/// doc for a class Widget"), None);
    }

    #[test]
    fn digest_scan_skips_block_comment_interiors() {
        // A bare interior line of a /* … */ block (no `*` prefix) must not be
        // mis-parsed as a signature, while real signatures around it still are.
        let digest = "\
/*
 header prose mentioning a trait to Axiom and a class of things
 more prose describing function bodies
*/
pub fn real_fn(x: i32) -> i32 { … }
struct RealStruct { … }";
        let symbols = symbols_in_digest(digest);
        assert_eq!(
            symbols,
            vec!["real_fn".to_string(), "RealStruct".to_string()]
        );
    }

    #[test]
    fn report_ratios_are_well_defined_on_empty() {
        let r = BenchReport::default();
        assert_eq!(r.savings_ratio(), 0.0);
        assert_eq!(r.fidelity_ratio(), 1.0); // no symbols → vacuously perfect
    }

    #[test]
    fn unrecovered_symbols_carry_file_and_name() {
        let source = "pub fn real_fn() {}\n";
        let digest = "pub fn real_fn() { … }\nfn ghost_fn() { … }\n";
        let failures = unrecovered_in_file(source, digest, Path::new("src/lib.rs"));
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].file, PathBuf::from("src/lib.rs"));
        assert_eq!(failures[0].symbol, "ghost_fn");
    }

    #[test]
    fn strict_decision_is_lossless_only() {
        let mut r = BenchReport::default();
        assert!(r.is_lossless()); // vacuously true with no signatures
        r.symbols_total = 4;
        r.symbols_recovered = 4;
        assert!(r.is_lossless());
        r.symbols_recovered = 3;
        assert!(!r.is_lossless());
    }
}
