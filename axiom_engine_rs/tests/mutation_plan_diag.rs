//! Offline diagnostic for the mutation-repair planner on a seeded corpus.
//!
//! For each fixture written by `scripts/seed_faults.py`, plan candidates for
//! the faulted file against a recorded failure trace and report where (if
//! anywhere) the pristine file appears in the ranked list. Rank < budget means
//! the engine would try the reversal; "absent" means no operator produces it.
//! It does not run any verifier, so it iterates in seconds; the end-to-end rate
//! still comes from `scripts/bench_repair.sh`.
//!
//! ```bash
//! AXIOM_PLAN_DIAG_CORPUS=/path/to/seeded \
//! AXIOM_PLAN_DIAG_TRACES=/path/to/traces \
//! AXIOM_PLAN_DIAG_PRISTINE=../axiom_mesh_core \
//!   cargo test --test mutation_plan_diag -- --ignored --nocapture
//! ```
//! `TRACES/<fixture>.txt` holds the failing `cargo test` output for that fixture.

use std::path::{Path, PathBuf};

#[test]
#[ignore = "needs a seeded corpus on disk; see module docs"]
fn print_plan_ranks() {
    let corpus = PathBuf::from(std::env::var("AXIOM_PLAN_DIAG_CORPUS").expect("corpus dir"));
    let traces = PathBuf::from(std::env::var("AXIOM_PLAN_DIAG_TRACES").expect("traces dir"));
    let pristine_root =
        PathBuf::from(std::env::var("AXIOM_PLAN_DIAG_PRISTINE").expect("pristine crate dir"));
    let budget = axiom_engine::mutation_repair::max_candidates();

    let mut names: Vec<String> = std::fs::read_dir(&corpus)
        .expect("read corpus")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("meta.json").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();

    let (mut within, mut beyond, mut absent) = (0, 0, 0);
    println!("| fixture | candidates | rank of reversal | blamed lines |");
    println!("|---|---|---|---|");
    for name in &names {
        let dir = corpus.join(name);
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        let rel = meta["fault"]["file"].as_str().unwrap();
        let broken_path = dir.join("broken").join(rel);
        let broken = std::fs::read_to_string(&broken_path).unwrap();
        let pristine = std::fs::read_to_string(pristine_root.join(rel)).unwrap();
        // Bounded exactly as the solve loop bounds a verifier's output.
        let trace = axiom_engine::solve::bound_trace(
            &std::fs::read_to_string(traces.join(format!("{name}.txt"))).unwrap_or_default(),
        );
        let plan = axiom_engine::mutation_repair::plan(&broken, Path::new(rel), &trace, 100_000);
        let rank = plan.iter().position(|c| c.source == pristine);
        let hints = axiom_engine::fault_locate::line_hints_for(&trace, Path::new(rel));
        let shown = match rank {
            Some(r) if r < budget => {
                within += 1;
                format!("{} ({})", r + 1, plan[r].label)
            }
            Some(r) => {
                beyond += 1;
                format!("**{}** ({})", r + 1, plan[r].label)
            }
            None => {
                absent += 1;
                "absent".to_string()
            }
        };
        println!(
            "| {name} | {} | {shown} | {:?} |",
            plan.len(),
            &hints[..hints.len().min(5)]
        );
    }
    println!(
        "\nreversal within budget {budget}: {within}; beyond budget: {beyond}; not generated: {absent}"
    );
}
