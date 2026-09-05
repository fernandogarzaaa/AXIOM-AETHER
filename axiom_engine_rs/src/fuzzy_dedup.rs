//! Fuzzy dedup tier: near-duplicate detection, extending S4 (`prefix_diet`)
//! from byte-identical dedup to similarity-based dedup.
//!
//! ## Why this exists
//!
//! S4's own write-up (`docs/superpowers/plans/2026-07-10-cvm-cost-stack.md`)
//! measured **0.00% real gain** from exact-byte dedup on this project's own
//! `~/.claude` rule set -- because the English, Chinese-translation, and
//! web-specific-override variants of the same rules are *not* byte-identical.
//! They are, however, exactly the kind of content a similarity check should
//! catch: same meaning, different bytes. This module is that check.
//!
//! ## What this is not
//!
//! This is **not** a trained-by-Axiom model. Axiom's own `AxiomTTTLM`
//! (`model.rs`) ships with a random-init checkpoint by default, and its
//! embedding path (`embedder.rs`) already documents the honest consequence:
//! an untrained model's pooled vector is degenerate and must be treated as
//! "no signal" rather than matched against. Fuzzy dedup needs an embedding
//! that actually clusters similar text *today*, without asking the user to
//! train anything first -- so this module takes a small, pretrained,
//! externally-distilled static embedding model (Model2Vec, via the
//! `model2vec-rs` crate) purely as a similarity oracle. It is never used for
//! generation, only for "are these two blocks saying roughly the same
//! thing". See `docs/EXPERIMENTAL.md` for how this is gated.
//!
//! ## Fail-safe by construction
//!
//! [`SimilarityBackend::embed_batch`] returns `Option<Vec<Vec<f32>>>`.
//! `None` (no backend configured, load failed, or the `fuzzy-embed` feature
//! is off) means "no signal" and [`fuzzy_diet`] falls back to the exact-diet
//! result unchanged -- the same convention `embedder.rs` uses for a
//! degenerate all-zero vector. A missing or broken embedder can only make
//! this module do *less*, never fabricate a false match.
//!
//! ## Not cache-safe in the S1 sense -- do not wire this in blindly
//!
//! S1 (`cache_safety.rs`) guarantees the compressor never rewrites bytes at
//! or before a client `cache_control` breakpoint, and S4's exact dedup is a
//! pure, byte-stable function of the input -- so it's safe to apply on every
//! request once enabled. Similarity scores from an embedding model are
//! **not** guaranteed byte-for-byte stable across model versions the way a
//! hash comparison is. Until this has its own S5-style behavior-eval gate
//! and measured real-traffic yield, it should only run on content that is
//! *not* being cached across turns (e.g. per-request tool-result bodies),
//! never on the system-prompt prefix tiers S1/S4 protect. Treat the plumbing
//! here as DONE-BUT-UNGATED, matching S4's own honesty convention for a
//! mechanism that's implemented and tested but not yet proven on real
//! traffic or cleared for a cache-adjacent default.

use crate::prefix_diet::{self, MIN_DEDUP_BYTES};

/// Marker inserted in place of a block judged a near-duplicate of an earlier
/// block. Deliberately distinct from [`crate::prefix_diet::DEDUP_MARKER`] --
/// a reader (or a downstream eval) must be able to tell "provably identical"
/// apart from "similarity-scored, below-certainty" at a glance.
pub const FUZZY_DEDUP_MARKER: &str =
    "[AXIOM-FUZZY-DEDUP: near-duplicate of an earlier block in this prompt]";

/// Conservative default: cosine similarity must clear this before two blocks
/// are considered the same content. High-precision on purpose -- a false
/// positive here silently drops information the model needed, which
/// `context_economics::preserves_evidence` treats as a failed optimization.
/// This has not been empirically tuned against real traffic; treat it as a
/// starting point for the same measure-before-flip process S4/S5 used, not
/// as a validated constant.
pub const DEFAULT_FUZZY_THRESHOLD: f32 = 0.93;

/// Aggregated telemetry for one [`fuzzy_diet`] call, mirroring
/// [`crate::prefix_diet::DietReport`]'s shape so the two tiers can be
/// reported side by side.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FuzzyDietReport {
    pub original_tokens: usize,
    pub fuzzy_dedup_tokens: usize,
    pub blocks_marked: usize,
    /// True only when a real backend returned embeddings this call. False
    /// means the result is identical to plain exact-dedup -- distinguishes
    /// "ran and found nothing to merge" from "didn't run at all".
    pub backend_active: bool,
}

/// A source of sentence/block embeddings, used only as a similarity oracle.
/// Implementors must return `None` rather than a degenerate or fabricated
/// vector when they cannot produce a real embedding -- see the module-level
/// fail-safe note above.
pub trait SimilarityBackend {
    fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>>;
}

/// Default backend: always declines. Wiring `fuzzy_diet` against this is
/// equivalent to running exact dedup alone -- the safe, zero-dependency
/// default until a real backend is explicitly configured.
pub struct NullBackend;

impl SimilarityBackend for NullBackend {
    fn embed_batch(&self, _texts: &[String]) -> Option<Vec<Vec<f32>>> {
        None
    }
}

/// Cosine similarity of two vectors. Returns `0.0` (never matches) for a
/// zero-norm vector on either side, rather than dividing by zero or treating
/// a degenerate embedding as similar to everything -- the same "no signal"
/// convention `embedder.rs::pool_and_normalize` uses.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Run exact dedup (S4) first, then mark near-duplicates among what's left
/// using `backend`. Reconstruction follows the same
/// `blocks[0] + seps[0] + blocks[1] + ...` contract as `prefix_diet::diet`.
///
/// If `backend.embed_batch` returns `None` -- no backend configured, a load
/// failure, or the `fuzzy-embed` feature isn't compiled in -- this returns
/// the exact-dedup result unchanged with `backend_active: false`. Callers
/// should treat that as "ran, found nothing extra", not as an error.
pub fn fuzzy_diet<B: SimilarityBackend>(
    text: &str,
    backend: &B,
    threshold: f32,
) -> (String, FuzzyDietReport) {
    let original_tokens = text.split_whitespace().count();

    // S4 first: cheap, zero-risk, catches true duplicates before we ever
    // call an embedding model.
    let (exact_deduped, _exact_count) = prefix_diet::diet_with_report(text);

    let (blocks, seps) = prefix_diet::split_blocks(&exact_deduped);

    // Candidates: long enough to be worth marking, and not already an
    // exact-dedup marker (never re-embed a marker line, and never let a
    // marker "match" real content).
    let candidate_idx: Vec<usize> = blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            let t = b.trim();
            t.len() >= MIN_DEDUP_BYTES && t != prefix_diet::DEDUP_MARKER
        })
        .map(|(i, _)| i)
        .collect();

    if candidate_idx.is_empty() {
        return (
            exact_deduped,
            FuzzyDietReport {
                original_tokens,
                ..Default::default()
            },
        );
    }

    let candidate_texts: Vec<String> = candidate_idx
        .iter()
        .map(|&i| blocks[i].trim().to_string())
        .collect();

    let Some(embeddings) = backend.embed_batch(&candidate_texts) else {
        // No signal available -- fall back to exact-dedup-only, unchanged.
        return (
            exact_deduped,
            FuzzyDietReport {
                original_tokens,
                ..Default::default()
            },
        );
    };

    // Greedy left-to-right clustering: each kept block is compared against
    // every earlier *kept* representative; the first match wins. Order
    // matches how the text reads, so "earlier block" is unambiguous without
    // needing to encode a back-reference in the marker itself.
    let mut kept_reps: Vec<usize> = Vec::new(); // indices into `embeddings`
    let mut is_dup = vec![false; candidate_idx.len()];
    for i in 0..embeddings.len() {
        let mut matched = false;
        for &rep in &kept_reps {
            if cosine(&embeddings[i], &embeddings[rep]) >= threshold {
                matched = true;
                break;
            }
        }
        if matched {
            is_dup[i] = true;
        } else {
            kept_reps.push(i);
        }
    }

    let mut dup_block_idx = std::collections::HashSet::new();
    for (candidate_pos, &blk_idx) in candidate_idx.iter().enumerate() {
        if is_dup[candidate_pos] {
            dup_block_idx.insert(blk_idx);
        }
    }

    let mut result = String::new();
    let mut blocks_marked = 0usize;
    for (i, b) in blocks.iter().enumerate() {
        if i > 0 {
            result.push_str(&seps[i - 1]);
        }
        if dup_block_idx.contains(&i) {
            result.push_str(FUZZY_DEDUP_MARKER);
            blocks_marked += 1;
        } else {
            result.push_str(b);
        }
    }

    let fuzzy_dedup_tokens = exact_deduped
        .split_whitespace()
        .count()
        .saturating_sub(result.split_whitespace().count());
    (
        result,
        FuzzyDietReport {
            original_tokens,
            fuzzy_dedup_tokens,
            blocks_marked,
            backend_active: true,
        },
    )
}

/// Cascading backend: try the larger GPU-first model, then the same model
/// on CPU, then the small CPU-first Model2Vec model, then no backend at
/// all. This is meant to be the default choice for real wiring -- as
/// opposed to picking [`NullBackend`], [`model2vec_backend::Model2VecBackend`],
/// or [`crate::gpu_embed::bert_backend::BertEmbedBackend`] by hand -- and it
/// composes whichever of the `fuzzy-embed` / `fuzzy-embed-gpu` features were
/// actually compiled in. With neither feature enabled, `build()` degrades
/// to behaving exactly like [`NullBackend`].
///
/// Every step is a graceful fallback, never a panic: a missing GPU, an
/// out-of-memory load, a network failure fetching Hub weights, or a
/// disabled feature all just move to the next candidate.
pub struct AutoBackend {
    inner: Option<Box<dyn SimilarityBackend>>,
    /// Which candidate actually loaded, for logging/telemetry -- e.g.
    /// `"bert-gpu"`, `"bert-cpu"`, `"model2vec-cpu"`, or `"none"`. Not used
    /// for any decision-making inside this type; purely informational.
    pub active_backend: &'static str,
}

impl AutoBackend {
    pub fn build() -> Self {
        #[cfg(feature = "fuzzy-embed-gpu")]
        {
            use crate::gpu_embed::bert_backend::{BertEmbedBackend, DEFAULT_GPU_MODEL_REPO};
            // Only attempt a CUDA device when the `cuda` feature is compiled
            // in (requires CUDA toolkit at build time). A CUDA initialization
            // or model-load failure still retries the same model on CPU before
            // moving on to Model2Vec.
            #[cfg(feature = "cuda")]
            if let Ok(device) = candle_core::Device::new_cuda(0) {
                if let Ok(b) = BertEmbedBackend::load(DEFAULT_GPU_MODEL_REPO, device) {
                    return Self {
                        inner: Some(Box::new(b)),
                        active_backend: "bert-gpu",
                    };
                }
            }
            if let Ok(b) = BertEmbedBackend::load(DEFAULT_GPU_MODEL_REPO, candle_core::Device::Cpu)
            {
                return Self {
                    inner: Some(Box::new(b)),
                    active_backend: "bert-cpu",
                };
            }
        }
        #[cfg(feature = "fuzzy-embed")]
        {
            use model2vec_backend::{Model2VecBackend, DEFAULT_MODEL_REPO};
            if let Ok(b) = Model2VecBackend::load(DEFAULT_MODEL_REPO) {
                return Self {
                    inner: Some(Box::new(b)),
                    active_backend: "model2vec-cpu",
                };
            }
        }
        Self {
            inner: None,
            active_backend: "none",
        }
    }
}

impl SimilarityBackend for AutoBackend {
    fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
        self.inner.as_ref()?.embed_batch(texts)
    }
}

/// Real embedding backend, gated behind the `fuzzy-embed` feature so default
/// builds (`cargo install`, the pip wheel, Docker, release binaries) carry
/// no dependency on an external model. Requires network access to the
/// Hugging Face Hub (or a local cache) the first time a given repo is
/// loaded; there is no bundled checkpoint.
#[cfg(feature = "fuzzy-embed")]
pub mod model2vec_backend {
    use super::SimilarityBackend;
    use model2vec_rs::model::StaticModel;

    /// Small, general-purpose, pretrained static-embedding model. Not
    /// trained by Axiom, not fine-tuned on any Axiom data -- used purely as
    /// an off-the-shelf similarity oracle. ~7.5M params, CPU-only, no GPU
    /// required (per the model2vec-rs benchmarks).
    pub const DEFAULT_MODEL_REPO: &str = "minishlab/potion-base-8M";

    pub struct Model2VecBackend {
        model: StaticModel,
    }

    impl Model2VecBackend {
        /// Loads `repo_or_path` via `StaticModel::from_pretrained`
        /// (Hugging Face Hub repo id or a local path). Returns `Err` on any
        /// load failure -- callers should treat that as "no backend
        /// available" and fall back to [`super::NullBackend`], not panic.
        pub fn load(repo_or_path: &str) -> anyhow::Result<Self> {
            let model = StaticModel::from_pretrained(repo_or_path, None, None, None)?;
            Ok(Self { model })
        }
    }

    impl SimilarityBackend for Model2VecBackend {
        fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
            // model2vec-rs's `encode` does not return a `Result`; if this
            // ever becomes fallible upstream, prefer mapping any error to
            // `None` here over unwrapping -- fail-safe is the whole point
            // of this module.
            Some(self.model.encode(texts))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(byte: char, len: usize) -> String {
        let word: String = std::iter::repeat(byte).take(4).collect();
        let mut s = String::new();
        while s.len() < len {
            if !s.is_empty() {
                s.push(' ');
            }
            s.push_str(&word);
        }
        s.truncate(len);
        s
    }

    #[test]
    fn cosine_identical_vectors_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors_is_zero() {
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector_is_never_a_match() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine(&[], &[]), 0.0);
    }

    #[test]
    fn null_backend_falls_back_to_exact_dedup_only() {
        let repeated = block('a', 500);
        let text = format!("{repeated}\n\nunique\n\n{repeated}");
        let (out, report) = fuzzy_diet(&text, &NullBackend, DEFAULT_FUZZY_THRESHOLD);
        // Exact dedup still ran (S4 always runs first)...
        assert!(out.contains(prefix_diet::DEDUP_MARKER));
        // ...but the fuzzy tier reports it never had signal to act on.
        assert!(!report.backend_active);
        assert_eq!(report.blocks_marked, 0);
    }

    /// A stub backend for exercising the clustering logic deterministically,
    /// without a real model or network access: maps each text to a
    /// hand-assigned vector so the test controls similarity directly.
    struct StubBackend {
        vectors: std::collections::HashMap<String, Vec<f32>>,
    }

    impl SimilarityBackend for StubBackend {
        fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
            texts
                .iter()
                .map(|t| self.vectors.get(t).cloned())
                .collect::<Option<Vec<_>>>()
        }
    }

    #[test]
    fn stub_backend_marks_near_duplicate_blocks() {
        // Two blocks with genuinely different content (so S4's exact-byte
        // dedup can't touch either one) but assigned near-identical vectors
        // by the stub, plus one unrelated block. Deliberately NOT a
        // whitespace-only difference: block('x', 450) happens to end in a
        // trailing space by construction (450 is a multiple of the 5-byte
        // "xxxx "  unit), so two blocks differing only by trailing
        // whitespace trim down to the *same* string and collide as a single
        // HashMap key -- which made S4 dedup them before the fuzzy tier ever
        // ran, defeating the point of this test. Using different letters
        // keeps the two candidate texts distinct at every stage.
        let a = block('x', 450);
        let b = block('y', 450); // different content, not byte-identical to a
        let c = block('z', 450); // unrelated

        let mut vectors = std::collections::HashMap::new();
        vectors.insert(a.trim().to_string(), vec![1.0, 0.0, 0.0]);
        vectors.insert(b.trim().to_string(), vec![0.999, 0.001, 0.0]);
        vectors.insert(c.trim().to_string(), vec![0.0, 1.0, 0.0]);
        let backend = StubBackend { vectors };

        let text = format!("{a}\n\n{c}\n\n{b}");
        let (out, report) = fuzzy_diet(&text, &backend, 0.9);

        assert!(report.backend_active);
        assert_eq!(report.blocks_marked, 1, "only the later near-duplicate should be marked");
        assert!(out.contains(FUZZY_DEDUP_MARKER));
        assert!(out.contains(a.trim()), "first occurrence is kept in full");
        assert!(out.contains(c.trim()), "unrelated block is untouched");
    }

    #[test]
    fn stub_backend_leaves_dissimilar_blocks_untouched() {
        let a = block('p', 450);
        let b = block('q', 450);
        let mut vectors = std::collections::HashMap::new();
        vectors.insert(a.trim().to_string(), vec![1.0, 0.0]);
        vectors.insert(b.trim().to_string(), vec![0.0, 1.0]);
        let backend = StubBackend { vectors };

        let text = format!("{a}\n\n{b}");
        let (out, report) = fuzzy_diet(&text, &backend, 0.9);
        assert_eq!(report.blocks_marked, 0);
        assert_eq!(out, text);
    }

    #[test]
    fn fuzzy_token_savings_exclude_exact_dedup_savings() {
        let repeated = block('r', 500);
        let unique = block('u', 500);
        let mut vectors = std::collections::HashMap::new();
        vectors.insert(repeated.trim().to_string(), vec![1.0, 0.0]);
        vectors.insert(unique.trim().to_string(), vec![0.0, 1.0]);
        let backend = StubBackend { vectors };

        let text = format!("{repeated}\n\n{unique}\n\n{repeated}");
        let (out, report) = fuzzy_diet(&text, &backend, 0.9);

        assert!(out.contains(prefix_diet::DEDUP_MARKER));
        assert!(report.backend_active);
        assert_eq!(report.blocks_marked, 0);
        assert_eq!(report.fuzzy_dedup_tokens, 0);
    }

    #[test]
    fn short_blocks_are_never_candidates_even_with_a_backend() {
        let short = "too short to matter".to_string();
        let text = format!("{short}\n\n{short}");
        // Backend that would match anything to anything, if it were consulted.
        struct AlwaysMatch;
        impl SimilarityBackend for AlwaysMatch {
            fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
                Some(texts.iter().map(|_| vec![1.0]).collect())
            }
        }
        let (out, report) = fuzzy_diet(&text, &AlwaysMatch, 0.5);
        assert_eq!(report.blocks_marked, 0);
        assert_eq!(out, text);
    }

    #[test]
    #[cfg(feature = "fuzzy-embed")]
    #[ignore = "needs network access to the Hugging Face Hub on first load; \
                run explicitly once a model cache is warm, matching the \
                convention semantic_harness.rs / cvm_eval.rs already use for \
                network- or credential-gated tests"]
    fn real_backend_loads_and_embeds() {
        use super::model2vec_backend::Model2VecBackend;
        let backend = Model2VecBackend::load(super::model2vec_backend::DEFAULT_MODEL_REPO)
            .expect("model load");
        let out = backend
            .embed_batch(&["hello world".to_string()])
            .expect("embeddings");
        assert_eq!(out.len(), 1);
        assert!(!out[0].is_empty());
    }

    #[test]
    fn auto_backend_degrades_gracefully_with_no_features_compiled() {
        // With neither `fuzzy-embed` nor `fuzzy-embed-gpu` enabled, build()
        // must not panic and must behave exactly like NullBackend.
        let auto = AutoBackend::build();
        #[cfg(not(any(feature = "fuzzy-embed", feature = "fuzzy-embed-gpu")))]
        {
            assert_eq!(auto.active_backend, "none");
            assert!(auto.embed_batch(&["x".to_string()]).is_none());
        }
        // Regardless of which features are on, embed_batch must never panic.
        let _ = auto.embed_batch(&["a warm-up call".to_string()]);
    }
}
