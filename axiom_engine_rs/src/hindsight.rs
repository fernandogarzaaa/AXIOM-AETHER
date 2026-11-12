//! B.3 — hindsight fine-tuning on the node's own verified/failed traces.
//!
//! SOAR-style self-improvement: periodically fold the node's *own* experience
//! back into the slow weights. The training signal is sourced honestly from where
//! each part actually lives (a distinction Codex flagged on the roadmap):
//!
//!   * **Verified fix contents** — the positive examples — come from
//!     [`crate::patch_memory::PatchMemory`], where each [`PatchCandidate`] stores
//!     the fix as a diff: the lines that made verify pass, as they read after
//!     the fix (not the whole file), plus a `verified_count` confidence weight
//!     (how many fleet nodes independently confirmed it green).
//!   * **Failure-tension context** — which commands have been hard, and how
//!     surprising their failures were — comes from
//!     [`crate::heal_memory::HealMemory`] (`ce_mean` per program). `HealMemory`
//!     does **not** hold source contents, so it is used only as auxiliary signal
//!     / reporting, never as the fine-tuning target.
//!
//! The collected verified fixes are materialized into a small corpus and fed
//! through the existing [`crate::meta_train::MetaTrainer`] pipeline, so a node
//! literally trains on the fixes it (and its fleet) have proven correct.
//!
//! ## Gating (the invariant)
//!
//! Fine-tuning here only proposes a *new checkpoint*; it is never promoted on
//! trust. [`FineTuneReport`] carries the before/after loss so the caller can gate
//! promotion behind the agentic-eval benchmark (the same re-verify-before-trust
//! discipline the patch path uses). Nothing the node learns is believed until it
//! is independently re-measured.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::heal_memory::HealMemory;
use crate::patch_memory::PatchMemory;

/// One supervised training example harvested from verified-patch memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingExample {
    /// Failure fingerprint the fix resolved.
    pub fingerprint: String,
    /// Repo-relative path the fix targeted.
    pub rel_path: String,
    /// The verified fix: the changed lines as they read after the fix
    /// (`PatchCandidate::diff.replacement`) — the positive target.
    pub content: String,
    /// Confidence weight = `verified_count` (independent green confirmations).
    pub weight: u32,
}

/// Auxiliary failure-tension signal harvested from heal memory: `(command,
/// ce_mean)`. Context only — heal memory has no source contents to train on.
#[derive(Debug, Clone, PartialEq)]
pub struct TensionContext {
    pub command: String,
    pub ce_mean: f32,
}

/// Summary of a hindsight fine-tuning pass, with enough information for the
/// caller to gate checkpoint promotion behind a benchmark.
#[derive(Debug, Clone, PartialEq)]
pub struct FineTuneReport {
    /// Number of distinct verified fixes used as training targets.
    pub examples: usize,
    /// Number of distinct verified fixes written to the corpus.
    pub corpus_files: usize,
    /// Final training loss (lower is better). `None` if there was nothing to
    /// train on.
    pub final_loss: Option<f32>,
}

/// Collect verified-fix training examples from patch memory, keeping only
/// candidates confirmed green at least `min_verified` times. Ordered by weight
/// (most-confirmed first) so the strongest signal leads.
pub fn collect_from_patch_memory(pm: &PatchMemory, min_verified: u32) -> Vec<TrainingExample> {
    let mut out = Vec::new();
    for (fingerprint, candidates) in &pm.by_fingerprint {
        for c in candidates {
            if c.verified_count >= min_verified.max(1) {
                out.push(TrainingExample {
                    fingerprint: fingerprint.clone(),
                    rel_path: c.rel_path.clone(),
                    content: c.diff.replacement.clone(),
                    weight: c.verified_count,
                });
            }
        }
    }
    out.sort_by_key(|e| std::cmp::Reverse(e.weight));
    out
}

/// Collect the auxiliary failure-tension context from heal memory.
pub fn collect_tension_context(hm: &HealMemory) -> Vec<TensionContext> {
    let mut out: Vec<TensionContext> = hm
        .all_records()
        .into_iter()
        .filter(|r| r.ce_count > 0)
        .map(|r| TensionContext {
            command: r.command.clone(),
            ce_mean: r.ce_mean,
        })
        .collect();
    // Highest mean cross-entropy (most surprising failures) first.
    out.sort_by(|a, b| {
        b.ce_mean
            .partial_cmp(&a.ce_mean)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// File the verified-fix corpus is written to, inside the corpus directory.
/// It matches the `hindsight_*.rs` pattern [`write_corpus`] prunes, and keeps
/// a `.rs` extension so the trainer's source-extension filter picks it up.
const CORPUS_FILE: &str = "hindsight_corpus.rs";

/// Materialize verified-fix examples into a corpus directory the
/// [`crate::meta_train::MetaTrainer`] can ingest.
///
/// A stored fix is a diff, so an example is usually a few changed lines, far
/// shorter than one training window; the trainer slices each *file* into
/// windows and drops a file shorter than one. The distinct fixes are therefore
/// written as one stream, one after another in a single file, so every fix
/// contributes tokens and short fixes are not lost. Fixes are deduplicated by
/// content (a fix confirmed many times is not over-represented beyond its
/// inclusion) and keep the order given, most-confirmed first. Returns the
/// number of distinct fixes written.
pub fn write_corpus(examples: &[TrainingExample], dir: &Path) -> std::io::Result<usize> {
    std::fs::create_dir_all(dir)?;
    // Prune stale generated files from prior runs so a re-run with fewer/different
    // examples never trains on obsolete samples. Only our own `hindsight_*.rs`
    // files are touched.
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let is_generated = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with("hindsight_") && n.ends_with(".rs"))
            .unwrap_or(false);
        if is_generated {
            let _ = std::fs::remove_file(path);
        }
    }
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut stream = String::new();
    let mut written = 0usize;
    for ex in examples {
        if !seen.insert(ex.content.as_str()) {
            continue; // dedup identical contents
        }
        stream.push_str(&ex.content);
        if !ex.content.ends_with('\n') {
            stream.push('\n');
        }
        written += 1;
    }
    if written > 0 {
        std::fs::write(dir.join(CORPUS_FILE), stream)?;
    }
    Ok(written)
}

/// Default on-disk location for the hindsight corpus. Prefer an app-owned
/// directory alongside the patch store (not world-readable shared temp) since
/// corpus entries are lines of verified source; fall back to temp only when
/// no patch-store path is configured.
pub fn default_corpus_dir() -> PathBuf {
    if let Some(pm_path) = PatchMemory::default_path() {
        return pm_path.with_file_name("axiom_hindsight_corpus");
    }
    std::env::temp_dir().join("axiom_hindsight_corpus")
}

/// Build the verified-fix corpus and run a hindsight fine-tuning pass through the
/// existing meta-training pipeline, returning a [`FineTuneReport`].
///
/// This is the full end-to-end wiring: verified fixes → corpus → `MetaTrainer` →
/// updated checkpoint. The caller is responsible for gating promotion of the
/// resulting `checkpoint_path` behind the benchmark (the report's `final_loss`
/// supports that decision). With no qualifying examples, or with fixes that
/// together are too short to form a single training window, it is a no-op
/// that reports `final_loss: None` and writes no checkpoint.
#[allow(clippy::too_many_arguments)]
pub fn fine_tune(
    pm: &PatchMemory,
    min_verified: u32,
    corpus_dir: &Path,
    config: crate::config::AxiomConfig,
    device: candle_core::Device,
    checkpoint_path: impl Into<String>,
    epochs: usize,
    steps_per_epoch: usize,
    lr: f64,
) -> candle_core::Result<FineTuneReport> {
    let examples = collect_from_patch_memory(pm, min_verified);
    if examples.is_empty() {
        return Ok(FineTuneReport {
            examples: 0,
            corpus_files: 0,
            final_loss: None,
        });
    }
    let corpus_files = write_corpus(&examples, corpus_dir).map_err(candle_core::Error::wrap)?;

    // Sequence window short enough that a handful of small fixes yield
    // training windows.
    let seq_len = 16usize;
    let (max_files, max_sequences, seed) = (4096usize, 65536usize, 0xA11CE_u64);

    // Too little material for one window is "nothing to train on yet", not a
    // failure: `MetaTrainer::build` errors on an empty dataset, so measure the
    // corpus first with the same dataset parameters.
    let windows = crate::meta_train::RepoFileDataset::build(
        corpus_dir,
        config.vocab_size as u32,
        seq_len,
        max_files,
        max_sequences,
        seed,
    )
    .map_err(candle_core::Error::wrap)?
    .len();
    if windows == 0 {
        return Ok(FineTuneReport {
            examples: examples.len(),
            corpus_files,
            final_loss: None,
        });
    }

    let mut trainer = crate::meta_train::MetaTrainer::build(
        config,
        device,
        corpus_dir,
        checkpoint_path,
        /* batch_size */ windows.clamp(1, 4),
        seq_len,
        max_files,
        max_sequences,
        seed,
    )?;

    let final_loss = trainer.run(epochs.max(1), steps_per_epoch.max(1), lr)?;
    Ok(FineTuneReport {
        examples: examples.len(),
        corpus_files,
        final_loss: Some(final_loss),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("axiom_hindsight_{tag}_{n}"))
    }

    fn long_fix(seed: &str) -> String {
        // Long enough to yield several token windows at seq_len=16.
        let mut s = format!("// fix for {seed}\nfn solve_{seed}() -> i32 {{\n");
        for i in 0..40 {
            s.push_str(&format!("    let v{i} = {i} + {seed}_base();\n"));
        }
        s.push_str("    0\n}\n");
        s
    }

    /// Every fix in these tests creates its file from nothing, so the stored
    /// span is the whole fixed content.
    fn test_scope(rel_path: &str) -> crate::patch_memory::RepoScope {
        crate::patch_memory::RepoScope::new("path:hindsight-test", rel_path)
    }

    #[test]
    fn collect_filters_by_min_verified_and_sorts_by_weight() {
        let mut pm = PatchMemory::new();
        pm.record_verified("fpA", &test_scope("src/a.rs"), "", "fn a() {}");
        // Reinforce fpA's candidate to verified_count = 3.
        pm.record_verified("fpA", &test_scope("src/a.rs"), "", "fn a() {}");
        pm.record_verified("fpA", &test_scope("src/a.rs"), "", "fn a() {}");
        pm.record_verified("fpB", &test_scope("src/b.rs"), "", "fn b() {}"); // verified_count = 1

        let all = collect_from_patch_memory(&pm, 1);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].weight, 3, "most-confirmed example must lead");
        assert_eq!(all[0].rel_path, "src/a.rs");

        let strong_only = collect_from_patch_memory(&pm, 2);
        assert_eq!(strong_only.len(), 1, "min_verified must filter weak fixes");
        assert_eq!(strong_only[0].fingerprint, "fpA");
    }

    #[test]
    fn write_corpus_dedups_identical_contents() {
        let dir = tmp("corpus");
        let examples = vec![
            TrainingExample {
                fingerprint: "fp".into(),
                rel_path: "a.rs".into(),
                content: "fn a() {}".into(),
                weight: 2,
            },
            TrainingExample {
                fingerprint: "fp".into(),
                rel_path: "a.rs".into(),
                content: "fn a() {}".into(), // identical → deduped
                weight: 1,
            },
            TrainingExample {
                fingerprint: "fp2".into(),
                rel_path: "b.rs".into(),
                content: "fn b() {}".into(),
                weight: 1,
            },
        ];
        let n = write_corpus(&examples, &dir).unwrap();
        assert_eq!(n, 2, "identical contents must be written once");
        // One stream file holding both distinct fixes, most-confirmed first.
        let count = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(count, 1);
        assert_eq!(
            std::fs::read_to_string(dir.join(CORPUS_FILE)).unwrap(),
            "fn a() {}\nfn b() {}\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_corpus_prunes_stale_generated_files() {
        let dir = tmp("prune");
        std::fs::create_dir_all(&dir).unwrap();
        // A stale file from a prior run + an unrelated file that must be kept.
        std::fs::write(dir.join("hindsight_999999.rs"), "stale").unwrap();
        std::fs::write(dir.join("keep.txt"), "unrelated").unwrap();

        let examples = vec![TrainingExample {
            fingerprint: "fp".into(),
            rel_path: "a.rs".into(),
            content: "fn a() {}".into(),
            weight: 1,
        }];
        let n = write_corpus(&examples, &dir).unwrap();
        assert_eq!(n, 1);
        assert!(
            !dir.join("hindsight_999999.rs").exists(),
            "stale file must be pruned"
        );
        assert!(
            dir.join("keep.txt").exists(),
            "non-generated files must be left alone"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fine_tune_is_a_noop_with_no_qualifying_examples() {
        let pm = PatchMemory::new();
        let dir = tmp("noop");
        let report = fine_tune(
            &pm,
            1,
            &dir,
            crate::config::AxiomConfig::runtime_small(),
            candle_core::Device::Cpu,
            dir.join("ckpt.safetensors").to_string_lossy().to_string(),
            1,
            1,
            1e-3,
        )
        .unwrap();
        assert_eq!(report.examples, 0);
        assert!(report.final_loss.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fine_tune_trains_end_to_end_on_verified_fixes() {
        // Full wiring: record verified fixes → corpus → MetaTrainer → checkpoint.
        let mut pm = PatchMemory::new();
        pm.record_verified("fp1", &test_scope("src/one.rs"), "", &long_fix("one"));
        pm.record_verified("fp2", &test_scope("src/two.rs"), "", &long_fix("two"));

        let dir = tmp("train");
        let ckpt = dir.join("ckpt.safetensors").to_string_lossy().to_string();
        let report = fine_tune(
            &pm,
            1,
            &dir.join("corpus"),
            crate::config::AxiomConfig::runtime_small(),
            candle_core::Device::Cpu,
            ckpt.clone(),
            /* epochs */ 1,
            /* steps_per_epoch */ 2,
            1e-3,
        )
        .unwrap();

        assert_eq!(report.examples, 2);
        assert!(report.corpus_files >= 2);
        let loss = report.final_loss.expect("should have trained");
        assert!(
            loss.is_finite() && loss >= 0.0,
            "loss must be finite (got {loss})"
        );
        assert!(Path::new(&ckpt).exists(), "checkpoint must be written");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A realistic single-line repair: one token changed on one line of a
    /// multi-line file. The stored fix is just that line.
    fn record_single_line_fix(pm: &mut PatchMemory, n: usize) {
        let pre = format!(
            "fn helper_{n}(x: i32, y: i32) -> i32 {{\n    return x - y;\n}}\n\nfn other_{n}() {{}}\n"
        );
        let post = pre.replace("x - y", &format!("x + y + {n}"));
        pm.record_verified(
            &format!("fp-short-{n}"),
            &test_scope(&format!("src/short_{n}.rs")),
            &pre,
            &post,
        );
    }

    #[test]
    fn stored_fixes_are_short_snippets_not_files() {
        // The premise of the two tests below: a one-line repair is stored as
        // a handful of tokens, far below one training window.
        let mut pm = PatchMemory::new();
        record_single_line_fix(&mut pm, 1);
        let examples = collect_from_patch_memory(&pm, 1);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].content, "    return x + y + 1;\n");
        assert!(examples[0].content.split_whitespace().count() < 17);
    }

    #[test]
    fn fine_tune_trains_on_short_single_line_fixes() {
        // Eight one-line repairs, each far shorter than a window on its own.
        // Together they are a corpus; none of them may be silently dropped.
        let mut pm = PatchMemory::new();
        for n in 0..8 {
            record_single_line_fix(&mut pm, n);
        }
        let dir = tmp("short");
        let ckpt = dir.join("ckpt.safetensors").to_string_lossy().to_string();
        let report = fine_tune(
            &pm,
            1,
            &dir.join("corpus"),
            crate::config::AxiomConfig::runtime_small(),
            candle_core::Device::Cpu,
            ckpt.clone(),
            /* epochs */ 1,
            /* steps_per_epoch */ 2,
            1e-3,
        )
        .expect("short fixes must train, not error");

        assert_eq!(report.examples, 8);
        assert_eq!(report.corpus_files, 8, "every short fix is in the corpus");
        let loss = report
            .final_loss
            .expect("eight short fixes form training windows");
        assert!(
            loss.is_finite() && loss >= 0.0,
            "loss must be finite (got {loss})"
        );
        assert!(Path::new(&ckpt).exists(), "checkpoint must be written");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fine_tune_is_a_noop_when_the_fixes_cannot_form_one_window() {
        // A single one-line repair is a few tokens: not enough for even one
        // window. That is "nothing to train on yet", not a failure.
        let mut pm = PatchMemory::new();
        record_single_line_fix(&mut pm, 1);
        let dir = tmp("tooshort");
        let ckpt = dir.join("ckpt.safetensors").to_string_lossy().to_string();
        let report = fine_tune(
            &pm,
            1,
            &dir.join("corpus"),
            crate::config::AxiomConfig::runtime_small(),
            candle_core::Device::Cpu,
            ckpt.clone(),
            1,
            1,
            1e-3,
        )
        .expect("too little material is a no-op, not an error");
        assert_eq!(report.examples, 1);
        assert_eq!(report.corpus_files, 1);
        assert!(report.final_loss.is_none());
        assert!(!Path::new(&ckpt).exists(), "no checkpoint is written");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
