//! Integration test for `axiom bench` (compression measurement).

use std::fs;
use std::path::PathBuf;

use axiom_engine::bench::run_bench;
use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use candle_core::Device;

fn tiny_pipeline() -> InferencePipeline {
    let cfg = AxiomConfig {
        d_model: 16,
        n_layers: 2,
        vocab_size: 64,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    InferencePipeline::new(cfg, Device::Cpu).expect("tiny pipeline must build")
}

fn unique_tmp(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("axiom_bench_{tag}_{nanos}"))
}

#[test]
fn bench_reports_savings_and_full_fidelity_on_code() {
    let root = unique_tmp("repo");
    fs::create_dir_all(&root).unwrap();
    // Real signatures with deliberately LARGE bodies — on real-size files
    // elision wins decisively (a tiny file's digest wrapper would dominate).
    let big_body = (0..40)
        .map(|i| format!("    acc += compute_step({i}) * {i} + offset;"))
        .collect::<Vec<_>>()
        .join("\n");
    let src = format!(
        "use std::collections::HashMap;\n\n\
         /// Adds two numbers.\n\
         pub fn add(a: i32, b: i32) -> i32 {{\n    let mut acc = 0;\n{big_body}\n    acc\n}}\n\n\
         pub struct Widget {{\n    pub name: String,\n}}\n\n\
         impl Widget {{\n    pub fn render(&self) -> String {{\n        let mut s = String::new();\n{big_body}\n        s\n    }}\n}}\n"
    );
    fs::write(root.join("widget.rs"), src).unwrap();

    let pipeline = tiny_pipeline();
    let report = run_bench(&root, &pipeline).expect("bench must succeed");

    assert_eq!(report.files, 1);
    assert!(report.original_tokens > 0);
    assert!(
        report.skeleton_tokens < report.original_tokens,
        "skeleton must be smaller than source"
    );
    assert!(report.savings_ratio() > 0.0, "must report positive savings");
    assert!(report.symbols_total >= 2, "add/render/Widget signatures detected");
    assert_eq!(
        report.symbols_recovered, report.symbols_total,
        "every kept signature must round-trip through expand"
    );
    assert_eq!(report.fidelity_ratio(), 1.0);

    let _ = fs::remove_dir_all(&root);
}
