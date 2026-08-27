//! AxiomBench — the proof layer.
//!
//! Runs the deterministic pillars (cognition, trust, fleet) always; the cost
//! pillar only with `--live`; the ablation pillar only with `--ablation`
//! (deterministic and offline, but builds a real inference pipeline and runs
//! 9 fixtures through 2 arms, so it's opt-in rather than part of the fast
//! default run). Each pillar returns a `PillarResult`; the runner prints
//! them, writes `bench/results/<ts>.json`, and regenerates `RESULTS.md`.
//!
//! Usage: `axiombench [--live] [--ablation] [--out <path>]`
//!   --live         include the cost pillar (needs a running proxy + corpus)
//!   --ablation     include the self-heal ablation pillar (baseline vs +AXIOM
//!                  repair loop over the eval-agentic fixture suite)
//!   --out <path>   results JSON path (default bench/results/<unix_ts>.json)

mod ablation;
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

const RESULTS_TABLE_START: &str = "<!-- axiombench:table:start -->";
const RESULTS_TABLE_END: &str = "<!-- axiombench:table:end -->";

/// Default template used only when RESULTS.md doesn't exist yet. The marker
/// pair is what makes every later run non-destructive (see below); the prose
/// around it is written once and then belongs to whoever edits it by hand.
fn default_results_template() -> String {
    format!(
        "# AxiomBench Results\n\n\
         {RESULTS_TABLE_START}\n{RESULTS_TABLE_END}\n\n\
         ## How to read these\n\n\
         The `n` column is a sample size, not a rate qualifier -- a small `n` \
         means the headline is a smoke check or an indicative measurement, not \
         evidence of a general rate. See each row's `read as`.\n\n\
         For the compression figure that *is* measured at scale, run \
         `axiom bench <path>` -- see `docs/AXIOMBENCH.md`.\n\n\
         ## Reproduce\n\n\
         Deterministic pillars: `cargo run --release --features tools --bin axiombench`.\n\n\
         Live cost pillar without upstream credentials: \
         `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\run_axiombench_cost_mock.ps1`.\n"
    )
}

/// Render just the `| Pillar | Headline | n | Read as |` table for `results`,
/// preserving any row for a pillar *not* present in `results` (e.g. `cost`
/// when this run wasn't `--live`) by carrying it over from `previous_table`
/// verbatim -- a non-`--live` run must not erase the last recorded cost
/// figure, only refresh the pillars it actually ran.
fn render_results_table(results: &[PillarResult], previous_table: &str) -> String {
    let mut rows: Vec<(String, String)> = Vec::new();
    for r in results {
        let n = r.sample_n.map(|n| n.to_string()).unwrap_or_else(|| "-".into());
        let read_as = r.read_as.as_deref().unwrap_or("-");
        rows.push((r.name.clone(), format!("| {} | {} | {} | {} |", r.name, r.headline, n, read_as)));
    }
    let seen: std::collections::HashSet<String> = rows.iter().map(|(name, _)| name.clone()).collect();
    for line in previous_table.lines() {
        let Some(name) = line.trim_start().strip_prefix('|').map(str::trim).and_then(|s| s.split('|').next()) else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() || name == "Pillar" || name.starts_with('-') || seen.contains(name) {
            continue;
        }
        rows.push((name.to_string(), line.to_string()));
    }
    let mut table = String::from("| Pillar | Headline | n | Read as |\n|---|---|---|---|\n");
    for (_, line) in &rows {
        table.push_str(line);
        table.push('\n');
    }
    table
}

/// Regenerate the repo-root RESULTS.md headline table from the latest run,
/// **without** clobbering hand-maintained prose. Everything outside the
/// `axiombench:table` marker pair is read back verbatim from the existing
/// file (or, if RESULTS.md doesn't exist yet, written once from
/// `default_results_template`); only the table between the markers is
/// replaced. A pillar this run didn't execute keeps its last-recorded row
/// instead of disappearing (see `render_results_table`).
fn write_results_md(root: &Path, results: &[PillarResult], generated: u64) -> std::io::Result<()> {
    let path = root.join("RESULTS.md");
    let existing = std::fs::read_to_string(&path).unwrap_or_else(|_| default_results_template());
    let (before, rest) = match existing.split_once(RESULTS_TABLE_START) {
        Some(parts) => parts,
        None => {
            // No markers (e.g. a pre-existing hand-written file from before
            // this mechanism existed): prepend a fresh marked table rather
            // than guess where to splice, so nothing hand-written is lost.
            let table = render_results_table(results, "");
            let md = format!(
                "# AxiomBench Results\n\nGenerated (unix): {generated}\n\n{RESULTS_TABLE_START}\n{table}{RESULTS_TABLE_END}\n\n{existing}"
            );
            return std::fs::write(&path, md);
        }
    };
    let (previous_table, after) = rest.split_once(RESULTS_TABLE_END).unwrap_or(("", rest));
    let table = render_results_table(results, previous_table);
    let md = format!("{before}{RESULTS_TABLE_START}\n{table}{RESULTS_TABLE_END}{after}");
    // Refresh the "Generated (unix)" line if present, leaving everything else untouched.
    let md = update_generated_line(&md, generated);
    std::fs::write(&path, md)
}

/// Replace a `Generated (unix): <n>` line in place, or leave the file
/// untouched if the hand-written template doesn't have one.
fn update_generated_line(md: &str, generated: u64) -> String {
    let mut out = String::with_capacity(md.len());
    let mut replaced = false;
    for line in md.lines() {
        if !replaced && line.starts_with("Generated (unix):") {
            out.push_str(&format!("Generated (unix): {generated}"));
            replaced = true;
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

enum Command {
    Bench { live: bool, ablation: bool, out: Option<PathBuf> },
    Corpus(Vec<String>),
}

fn parse_args() -> Command {
    let mut live = false;
    let mut ablation = false;
    let mut out: Option<PathBuf> = None;
    let argv = std::env::args().skip(1).collect::<Vec<_>>();
    if argv.first().map(String::as_str) == Some("corpus") {
        return Command::Corpus(argv.into_iter().skip(1).collect());
    }
    let mut args = argv.into_iter();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--live" => live = true,
            "--ablation" => ablation = true,
            "--out" => out = args.next().map(PathBuf::from),
            _ => {}
        }
    }
    Command::Bench { live, ablation, out }
}

fn main() {
    let command = parse_args();
    let (live, ablation, out) = match command {
        Command::Bench { live, ablation, out } => (live, ablation, out),
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
    if ablation {
        results.push(ablation::run_ablation());
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

    fn pillar(name: &str, headline: &str, n: u64, read_as: &str) -> PillarResult {
        PillarResult {
            name: name.into(),
            headline: headline.into(),
            detail: serde_json::json!({}),
            sample_n: Some(n),
            read_as: Some(read_as.into()),
        }
    }

    #[test]
    fn write_results_md_preserves_hand_written_prose() {
        let dir = std::env::temp_dir().join(format!(
            "axiombench_results_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("RESULTS.md"),
            "# AxiomBench Results\n\nGenerated (unix): 1\n\n\
             <!-- axiombench:table:start -->\n\
             | Pillar | Headline | n | Read as |\n|---|---|---|---|\n\
             | cognition | old headline | 3 | smoke check, not a rate |\n\
             | cost | old cost headline | 3 | indicative only |\n\
             <!-- axiombench:table:end -->\n\n\
             ## How to read these\n\nSome carefully hand-written caveat text.\n",
        )
        .unwrap();

        let results = vec![pillar("cognition", "new headline", 3, "smoke check, not a rate")];
        write_results_md(&dir, &results, 2).unwrap();

        let md = std::fs::read_to_string(dir.join("RESULTS.md")).unwrap();
        assert!(md.contains("Generated (unix): 2"), "timestamp must refresh:\n{md}");
        assert!(md.contains("new headline"), "run pillar must update:\n{md}");
        assert!(
            md.contains("old cost headline"),
            "a pillar not in this run must keep its last row, not disappear:\n{md}"
        );
        assert!(
            md.contains("Some carefully hand-written caveat text."),
            "hand-written prose outside the table markers must survive verbatim:\n{md}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_results_md_creates_a_template_when_no_file_exists() {
        let dir = std::env::temp_dir().join(format!(
            "axiombench_results_test_new_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let results = vec![pillar("cognition", "100% (3/3)", 3, "smoke check, not a rate")];
        write_results_md(&dir, &results, 42).unwrap();

        let md = std::fs::read_to_string(dir.join("RESULTS.md")).unwrap();
        assert!(md.contains("100% (3/3)"));
        assert!(md.contains(RESULTS_TABLE_START) && md.contains(RESULTS_TABLE_END));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_results_table_drops_stale_header_and_separator_lines() {
        let table = render_results_table(
            &[pillar("cognition", "h", 3, "smoke check, not a rate")],
            "| Pillar | Headline | n | Read as |\n|---|---|---|---|\n| cost | old | 3 | indicative only |\n",
        );
        assert_eq!(table.matches("| Pillar |").count(), 1, "must not duplicate the header:\n{table}");
        assert!(table.contains("| cost | old | 3 | indicative only |"));
    }
}
