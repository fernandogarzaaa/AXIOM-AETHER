//! Agentic self-evaluation harness — capability as a *measured number*.
//!
//! The point of this whole pillar is that AXIOM's coding ability should be
//! reported, not asserted. [`run_eval`] materializes a set of seeded broken-repo
//! fixtures, runs the real autonomous [`crate::solve`] loop on each (localize →
//! deterministic Poly-JIT repair → verify, all reversible), and reports the
//! fraction it drives green.
//!
//! The built-in fixtures ([`builtin_cases`]) are deliberately **deterministic
//! and offline**: each failure is one the Poly-JIT layer can repair without any
//! model or network, so the score is reproducible in CI and means exactly what
//! it says — "the autonomous loop fixed N of M broken repos end-to-end."

use std::path::PathBuf;

use crate::inference::InferencePipeline;
use crate::solve::{solve, SolveOptions};

/// A seeded broken project: files to materialize and the verify command that
/// must be driven to green. The fix is reachable by the deterministic repair
/// layer, so no LLM is required.
#[derive(Debug, Clone)]
pub struct EvalCase {
    pub name: String,
    /// (relative path, contents) written under a fresh temp project root.
    pub files: Vec<(String, String)>,
    /// Verify program + args, run with the project root as working dir.
    pub command: String,
    pub args: Vec<String>,
}

impl EvalCase {
    fn new(
        name: &str,
        files: &[(&str, &str)],
        command: &str,
        args: &[&str],
    ) -> Self {
        EvalCase {
            name: name.to_string(),
            files: files
                .iter()
                .map(|(p, c)| (p.to_string(), c.to_string()))
                .collect(),
            command: command.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Result of one case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseResult {
    pub name: String,
    pub solved: bool,
}

/// Aggregate evaluation report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalReport {
    pub results: Vec<CaseResult>,
}

impl EvalReport {
    pub fn total(&self) -> usize {
        self.results.len()
    }
    pub fn solved(&self) -> usize {
        self.results.iter().filter(|r| r.solved).count()
    }
    /// Success rate in [0, 1]; an empty suite scores 0.
    pub fn score(&self) -> f64 {
        if self.results.is_empty() {
            0.0
        } else {
            self.solved() as f64 / self.total() as f64
        }
    }
    /// Human-readable one-liner per case plus the headline score.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        for r in &self.results {
            s.push_str(&format!(
                "  [{}] {}\n",
                if r.solved { "PASS" } else { "FAIL" },
                r.name
            ));
        }
        s.push_str(&format!(
            "score: {}/{} = {:.0}%",
            self.solved(),
            self.total(),
            self.score() * 100.0
        ));
        s
    }
}

/// The built-in deterministic fixture suite. Each case's failure is repairable
/// by the Poly-JIT layer (e.g. an `exit 1` flip or a fixture marker), so the
/// whole suite is offline and reproducible.
pub fn builtin_cases() -> Vec<EvalCase> {
    vec![
        // 1. A script that fails with a self-naming frame; Poly-JIT flips exit 1.
        EvalCase::new(
            "shell-exit-flip",
            &[(
                "run.sh",
                "#!/bin/sh\necho 'run.sh:2:1: boom' >&2\nexit 1\n",
            )],
            "sh",
            &["run.sh"],
        ),
        // 2. Poly-JIT fixture marker repair.
        EvalCase::new(
            "fixture-marker",
            &[(
                "check.sh",
                "#!/bin/sh\necho 'check.sh:1:1: AXIOM_POLYJIT_FIXTURE_FAIL'\nexit 1\n",
            )],
            "sh",
            &["check.sh"],
        ),
        // 3. Multi-file: a passing helper cited first, the failing file second;
        //    the loop must localize both and repair the right one.
        EvalCase::new(
            "multi-file-pick-failing",
            &[
                ("helper.sh", "#!/bin/sh\necho 'helper.sh:1:1: note'\nexit 0\n"),
                ("main.sh", "#!/bin/sh\necho 'main.sh:3:1: boom' >&2\nexit 1\n"),
            ],
            "sh",
            &["-c", "sh helper.sh; sh main.sh"],
        ),
        // 4. Rust-format localization (`--> file:line`) + a second deterministic
        //    repair pattern (the `assert_eq!(1, 2)` -> `(1, 1)` flip). The grep
        //    guard fails while the bad assert is present and passes once flipped.
        EvalCase::new(
            "rust-assert-flip",
            &[("widget.rs", "pub fn f() { assert_eq!(1, 2); }\n")],
            "sh",
            &[
                "-c",
                "if grep -q 'assert_eq!(1, 2)' widget.rs; then \
                 echo 'error[E0001]'; echo '  --> widget.rs:1:14'; exit 1; else exit 0; fi",
            ],
        ),
        // 5. Python-traceback localization (`File \"x.py\", line N`) + fixture
        //    marker repair (FAIL -> PASS).
        EvalCase::new(
            "python-frame-localize",
            &[("core.py", "# AXIOM_POLYJIT_FIXTURE_FAIL\n")],
            "sh",
            &[
                "-c",
                "if grep -q AXIOM_POLYJIT_FIXTURE_FAIL core.py; then \
                 echo '  File \"core.py\", line 1, in <module>'; exit 1; else exit 0; fi",
            ],
        ),
        // 6. JS stack-frame localization (`at fn (file:line:col)`) + exit flip.
        EvalCase::new(
            "js-stack-localize",
            &[("index.js", "// index.js\nprocess.exit 1\n")],
            "sh",
            &[
                "-c",
                "if grep -q 'exit 1' index.js; then \
                 echo '    at Object.<anonymous> (index.js:2:9)'; exit 1; else exit 0; fi",
            ],
        ),
        // 7. Go-format localization (`./pkg/file.go:line:col:`) + exit flip.
        EvalCase::new(
            "go-frame-localize",
            &[("server.go", "package main\n// exit 1 placeholder\n")],
            "sh",
            &[
                "-c",
                "if grep -q 'exit 1' server.go; then \
                 echo './server.go:2:4: build failed'; exit 1; else exit 0; fi",
            ],
        ),
    ]
}

/// Materialize `cases` into fresh temp roots, run the autonomous loop on each,
/// and collect pass/fail. Each case gets its own throwaway directory which is
/// removed afterward. `solve` is run with no caller-named source, so the loop
/// must localize the fault itself — exactly the production path.
pub fn run_eval(pipeline: &InferencePipeline, cases: &[EvalCase]) -> EvalReport {
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let solved = run_one(pipeline, case).unwrap_or(false);
        results.push(CaseResult {
            name: case.name.clone(),
            solved,
        });
    }
    EvalReport { results }
}

fn run_one(pipeline: &InferencePipeline, case: &EvalCase) -> std::io::Result<bool> {
    let root = unique_root(&case.name);
    std::fs::create_dir_all(&root)?;
    for (rel, content) in &case.files {
        // EvalCase is public; never let a case write outside its sandbox root via
        // an absolute path or a `..` escape.
        let rel_path = std::path::Path::new(rel);
        if rel_path.is_absolute()
            || rel_path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("eval case path escapes the sandbox root: {rel}"),
            ));
        }
        let path = root.join(rel_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, content)?;
    }
    let opts = SolveOptions {
        max_rounds: 2,
        max_restarts: 1,
        anchor: Some(root.clone()),
        source_path: None,         // force autonomous localization
        patch_memory_path: None,   // isolate the eval from real fleet memory
        ..SolveOptions::default()
    };
    let report = solve(pipeline, &case.command, &case.args, &opts);
    let solved = report.map(|r| r.solved).unwrap_or(false);
    let _ = std::fs::remove_dir_all(&root);
    Ok(solved)
}

fn unique_root(name: &str) -> PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let safe: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    std::env::temp_dir().join(format!("axiom_eval_{safe}_{n}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AxiomConfig;
    use candle_core::Device;

    fn tiny_pipeline() -> InferencePipeline {
        let cfg = AxiomConfig {
            d_model: 16,
            n_layers: 2,
            vocab_size: 64,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        };
        InferencePipeline::new(cfg, Device::Cpu).expect("tiny pipeline")
    }

    #[test]
    fn builtin_suite_is_fully_solved_by_the_deterministic_loop() {
        // The headline guarantee: the autonomous loop repairs every built-in
        // fixture end-to-end with no LLM — a real, reproducible capability score.
        let report = run_eval(&tiny_pipeline(), &builtin_cases());
        assert_eq!(
            report.solved(),
            report.total(),
            "every deterministic fixture should be auto-repaired:\n{}",
            report.summary()
        );
        assert!((report.score() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_score_math() {
        let report = EvalReport {
            results: vec![
                CaseResult { name: "a".into(), solved: true },
                CaseResult { name: "b".into(), solved: false },
            ],
        };
        assert_eq!(report.total(), 2);
        assert_eq!(report.solved(), 1);
        assert!((report.score() - 0.5).abs() < f64::EPSILON);
        assert!(EvalReport { results: vec![] }.score() == 0.0);
    }
}
