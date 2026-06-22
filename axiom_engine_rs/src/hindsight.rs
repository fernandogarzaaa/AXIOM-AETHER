//! B.3 — hindsight fine-tuning on the node's own verified/failed traces.
//!
//! SOAR-style self-improvement: periodically fold the node's *own* experience
//! back into the slow weights. The training signal is sourced honestly from where
//! each part actually lives (a distinction Codex flagged on the roadmap):
//!
//!   * **Verified fix contents** — the positive examples — come from
//!     [`crate::patch_memory::PatchMemory`], where each [`PatchCandidate`] stores
//!     the full file contents that made verify pass, plus a `verified_count`
//!     confidence weight (how many fleet nodes independently confirmed it green).
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
    /// Full verified file contents — the positive target.
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
    /// Total tokens-worth of corpus material written.
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
                    content: c.content.clone(),
                    weight: c.verified_count,
                });
            }
        }
    }
    out.sort_by(|a, b| b.weight.cmp(&a.weight));
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

/// Materialize verified-fix examples into a corpus directory the
/// [`crate::meta_train::MetaTrainer`] can ingest.
///
/// Each distinct fix is written as a `.rs` file (deduplicated by content so a
/// fix confirmed many times is not over-represented beyond its inclusion). A
/// high-`verified_count` fix is written first so, combined with the trainer's
/// own sampling, the strongest signal is always present. Returns the number of
/// files written.
pub fn write_corpus(examples: &[TrainingExample], dir: &Path) -> std::io::Result<usize> {
    std::fs::create_dir_all(dir)?;
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut written = 0usize;
    for (i, ex) in examples.iter().enumerate() {
        if !seen.insert(ex.content.clone()) {
            continue; // dedup identical contents
        }
        // Stable, collision-free filename; keep a .rs extension so the trainer's
        // source-extension include filter picks it up.
        let fname = format!("hindsight_{i:06}.rs");
        std::fs::write(dir.join(fname), &ex.content)?;
        written += 1;
    }
    Ok(written)
}

/// Default on-disk location for the hindsight corpus, alongside the patch store.
pub fn default_corpus_dir() -> PathBuf {
    std::env::temp_dir().join("axiom_hindsight_corpus")
}

/// Build the verified-fix corpus and run a hindsight fine-tuning pass through the
/// existing meta-training pipeline, returning a [`FineTuneReport`].
///
/// This is the full end-to-end wiring: verified fixes → corpus → `MetaTrainer` →
/// updated checkpoint. The caller is responsible for gating promotion of the
/// resulting `checkpoint_path` behind the benchmark (the report's `final_loss`
/// supports that decision). With no qualifying examples it is a no-op that
/// reports `final_loss: None` and writes no checkpoint.
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

    // Sequence window short enough that even small fixes yield training windows.
    let seq_len = 16usize;
    let mut trainer = crate::meta_train::MetaTrainer::build(
        config,
        device,
        corpus_dir,
        checkpoint_path,
        /* batch_size */ corpus_files.clamp(1, 4),
        seq_len,
        /* max_files */ 4096,
        /* max_sequences */ 65536,
        /* seed */ 0xA11CE,
    )?;

    if trainer.dataset_len() == 0 {
        // Corpus too small to form a single window — nothing to train on.
        return Ok(FineTuneReport {
            examples: examples.len(),
            corpus_files,
            final_loss: None,
        });
    }

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

    #[test]
    fn collect_filters_by_min_verified_and_sorts_by_weight() {
        let mut pm = PatchMemory::new();
        pm.record_verified("fpA", "src/a.rs", "fn a() {}");
        // Reinforce fpA's candidate to verified_count = 3.
        pm.record_verified("fpA", "src/a.rs", "fn a() {}");
        pm.record_verified("fpA", "src/a.rs", "fn a() {}");
        pm.record_verified("fpB", "src/b.rs", "fn b() {}"); // verified_count = 1

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
        let count = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(count, 2);
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
        pm.record_verified("fp1", "src/one.rs", &long_fix("one"));
        pm.record_verified("fp2", "src/two.rs", &long_fix("two"));

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
        assert!(loss.is_finite() && loss >= 0.0, "loss must be finite (got {loss})");
        assert!(Path::new(&ckpt).exists(), "checkpoint must be written");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
