//! AxiomBench — the proof layer.
//!
//! Runs the deterministic pillars (cognition, trust, fleet) always; the cost
//! pillar only with `--live`. Each pillar returns a `PillarResult`; the runner
//! prints them, writes `bench/results/<ts>.json`, and regenerates `RESULTS.md`.
//!
//! Usage: `axiombench [--live] [--out <path>]`
//!   --live         include the cost pillar (needs a running proxy + corpus)
//!   --out <path>   results JSON path (default bench/results/<unix_ts>.json)

mod cognition;
mod corpus;
mod cost;
mod fleet;
mod trust;

use cognition::PillarResult;
use std::io::Write;
use std::path::{Path, PathBuf};

fn print_result(r: &PillarResult) {
    println!("[{}] {}", r.name, r.headline);
}

fn unix_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn repo_root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
}

fn results_json(results: &[PillarResult], generated: u64) -> serde_json::Value {
    serde_json::json!({
        "generated": generated,
        "results": results.iter().map(|r| serde_json::json!({
            "name": r.name,
            "headline": r.headline,
            "detail": r.detail,
        })).collect::<Vec<_>>(),
    })
}

/// Markers delimiting the generated headline table inside `RESULTS.md`.
/// Everything outside them (the "How to read these" caveats, the compression-
/// at-scale paragraph, the reproduce instructions) is hand-maintained and
/// must survive a regeneration untouched.
const TABLE_BEGIN: &str = "<!-- axiombench:table:begin -->";
const TABLE_END: &str = "<!-- axiombench:table:end -->";

/// Parse the pillar rows out of a previously generated table block, keyed by
/// pillar name. Used to carry over rows for pillars that did not run this
/// invocation (typically `cost` without `--live`) so a deterministic-only run
/// never erases the last live measurement.
fn previous_rows(block: &str) -> std::collections::BTreeMap<String, String> {
    block
        .lines()
        .filter(|l| l.starts_with("| "))
        .filter(|l| !l.starts_with("| Pillar") && !l.starts_with("|---"))
        .filter_map(|l| {
            let name = l.trim_start_matches('|').split('|').next()?.trim();
            (!name.is_empty()).then(|| (name.to_string(), l.to_string()))
        })
        .collect()
}

fn render_table(
    results: &[PillarResult],
    generated: u64,
    carried: &std::collections::BTreeMap<String, String>,
) -> String {
    let mut md = String::new();
    md.push_str(TABLE_BEGIN);
    md.push('\n');
    md.push_str(&format!("Generated (unix): {generated}\n\n"));
    md.push_str("| Pillar | Headline | n | Read as |\n|---|---|---|---|\n");
    let ran: std::collections::BTreeSet<&str> = results.iter().map(|r| r.name.as_str()).collect();
    for r in results {
        md.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            r.name,
            r.headline.replace('|', "\\|"),
            r.n,
            r.read_as
        ));
    }
    let mut carried_over = Vec::new();
    for (name, row) in carried {
        if !ran.contains(name.as_str()) {
            md.push_str(row);
            md.push('\n');
            carried_over.push(name.as_str());
        }
    }
    if !carried_over.is_empty() {
        md.push_str(&format!(
            "\nRows carried over from the previous table (pillar not run this invocation): {}.\n",
            carried_over.join(", ")
        ));
    }
    md.push_str(TABLE_END);
    md
}

/// Merge a fresh table into the existing `RESULTS.md` text.
///
/// * Markers present: only the block between them is replaced; rows for
///   pillars that did not run are carried over from the old block.
/// * Markers absent but a file exists: the block is inserted after the first
///   `# ` title line (or at the top), and the rest of the file is kept.
/// * No file: a minimal document is produced.
fn merge_results_md(existing: Option<&str>, results: &[PillarResult], generated: u64) -> String {
    match existing {
        Some(text) => match (text.find(TABLE_BEGIN), text.find(TABLE_END)) {
            (Some(b), Some(e)) if e > b => {
                let old_block = &text[b..e + TABLE_END.len()];
                let table = render_table(results, generated, &previous_rows(old_block));
                format!("{}{}{}", &text[..b], table, &text[e + TABLE_END.len()..])
            }
            _ => {
                let table = render_table(results, generated, &Default::default());
                let mut out = String::new();
                let mut inserted = false;
                for line in text.lines() {
                    out.push_str(line);
                    out.push('\n');
                    if !inserted && line.starts_with("# ") {
                        out.push('\n');
                        out.push_str(&table);
                        out.push('\n');
                        inserted = true;
                    }
                }
                if !inserted {
                    out = format!("{table}\n\n{out}");
                }
                out
            }
        },
        None => format!(
            "# AxiomBench Results\n\n{}\n\nReproduce deterministic pillars: \
             `cargo run --release --features tools --bin axiombench`.\n",
            render_table(results, generated, &Default::default())
        ),
    }
}

/// Regenerate the headline table inside the repo-root RESULTS.md, preserving
/// every hand-maintained line outside the marker block.
fn write_results_md(root: &Path, results: &[PillarResult], generated: u64) -> std::io::Result<()> {
    let path = root.join("RESULTS.md");
    let existing = std::fs::read_to_string(&path).ok();
    std::fs::write(
        &path,
        merge_results_md(existing.as_deref(), results, generated),
    )
}

enum Command {
    Bench { live: bool, out: Option<PathBuf> },
    Corpus(Vec<String>),
}

fn parse_args() -> Command {
    let mut live = false;
    let mut out: Option<PathBuf> = None;
    let argv = std::env::args().skip(1).collect::<Vec<_>>();
    if argv.first().map(String::as_str) == Some("corpus") {
        return Command::Corpus(argv.into_iter().skip(1).collect());
    }
    let mut args = argv.into_iter();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--live" => live = true,
            "--out" => out = args.next().map(PathBuf::from),
            _ => {}
        }
    }
    Command::Bench { live, out }
}

fn main() {
    let command = parse_args();
    let (live, out) = match command {
        Command::Bench { live, out } => (live, out),
        Command::Corpus(args) => {
            if let Err(error) = corpus::run(&args) {
                eprintln!("[axiombench] {error}");
                std::process::exit(2);
            }
            return;
        }
    };
    let base_url =
        std::env::var("AXIOM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:3000".to_string());

    let mut results = vec![
        cognition::run_cognition(),
        trust::run_trust(),
        fleet::run_fleet(),
    ];
    if live {
        results.push(cost::run_cost(&base_url));
    }

    println!("== AxiomBench ==");
    for r in &results {
        print_result(r);
    }

    let generated = unix_ts();
    let root = repo_root();
    let out_path =
        out.unwrap_or_else(|| root.join("bench/results").join(format!("{generated}.json")));
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::File::create(&out_path)
        .and_then(|mut f| f.write_all(results_json(&results, generated).to_string().as_bytes()))
    {
        Ok(()) => println!("results -> {}", out_path.display()),
        Err(e) => eprintln!(
            "[axiombench] could not write results {}: {e}",
            out_path.display()
        ),
    }
    if let Err(e) = write_results_md(&root, &results, generated) {
        eprintln!("[axiombench] could not write RESULTS.md: {e}");
    } else {
        println!("RESULTS.md updated");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pillar(name: &str, headline: &str) -> PillarResult {
        PillarResult {
            name: name.into(),
            headline: headline.into(),
            n: "3".into(),
            read_as: "smoke check, not a rate".into(),
            detail: json!({}),
        }
    }

    #[test]
    fn merge_replaces_only_the_marked_block_and_keeps_prose() {
        let existing = format!(
            "# AxiomBench Results\n\n{TABLE_BEGIN}\nGenerated (unix): 1\n\n\
             | Pillar | Headline | n | Read as |\n|---|---|---|---|\n\
             | cognition | old | 3 | smoke |\n\
             | cost | 75.1% byte reduction | **3 records** | indicative only |\n\
             {TABLE_END}\n\n## How to read these\n\nHand-written caveats.\n"
        );
        let out = merge_results_md(Some(&existing), &[pillar("cognition", "new")], 42);
        assert!(out.contains("Hand-written caveats."), "{out}");
        assert!(out.contains("## How to read these"), "{out}");
        assert!(out.contains("Generated (unix): 42"), "{out}");
        assert!(
            out.contains("| cognition | new | 3 | smoke check, not a rate |"),
            "{out}"
        );
        assert!(!out.contains("| cognition | old |"), "{out}");
        // The cost row did not run this invocation and is carried over verbatim.
        assert!(
            out.contains("| cost | 75.1% byte reduction | **3 records** | indicative only |"),
            "{out}"
        );
        assert!(
            out.contains("carried over from the previous table"),
            "{out}"
        );
        assert_eq!(out.matches(TABLE_BEGIN).count(), 1);
        assert_eq!(out.matches(TABLE_END).count(), 1);
    }

    #[test]
    fn merge_inserts_markers_after_title_when_absent() {
        let existing = "# AxiomBench Results\n\nSome prose that must survive.\n";
        let out = merge_results_md(Some(existing), &[pillar("fleet", "pass")], 7);
        assert!(
            out.starts_with("# AxiomBench Results\n\n<!-- axiombench:table:begin -->"),
            "{out}"
        );
        assert!(out.contains("Some prose that must survive."), "{out}");
        assert!(out.contains("| fleet | pass |"), "{out}");
        // Re-merging is idempotent on structure: still exactly one block.
        let again = merge_results_md(Some(&out), &[pillar("fleet", "pass again")], 8);
        assert_eq!(again.matches(TABLE_BEGIN).count(), 1);
        assert!(
            again.contains("pass again") && !again.contains("| fleet | pass |"),
            "{again}"
        );
    }

    #[test]
    fn merge_without_file_produces_minimal_document() {
        let out = merge_results_md(None, &[pillar("trust", "ok")], 1);
        assert!(out.starts_with("# AxiomBench Results"));
        assert!(out.contains(TABLE_BEGIN) && out.contains(TABLE_END));
        assert!(out.contains("| trust | ok |"));
    }

    #[test]
    fn pipe_in_headline_is_escaped() {
        let out = merge_results_md(None, &[pillar("x", "a | b")], 1);
        assert!(out.contains("| x | a \\| b |"), "{out}");
    }
}
