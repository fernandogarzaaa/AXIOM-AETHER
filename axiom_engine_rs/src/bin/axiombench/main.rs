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

/// Regenerate the repo-root RESULTS.md headline table from the latest run.
fn write_results_md(root: &Path, results: &[PillarResult], generated: u64) -> std::io::Result<()> {
    let mut md = String::new();
    md.push_str("# AxiomBench Results\n\n");
    md.push_str(&format!("Generated (unix): {generated}\n\n"));
    md.push_str("| Pillar | Headline |\n|---|---|\n");
    for r in results {
        md.push_str(&format!("| {} | {} |\n", r.name, r.headline));
    }
    md.push_str(
        "\nReproduce deterministic pillars: `cargo run --release --features tools --bin axiombench`.\n\
         Reproduce the live cost pillar without upstream credentials: \
         `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\run_axiombench_cost_mock.ps1`.\n",
    );
    std::fs::write(root.join("RESULTS.md"), md)
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
