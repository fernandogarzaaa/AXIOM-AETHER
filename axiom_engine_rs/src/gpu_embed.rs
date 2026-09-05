//! GPU-first embedding backend for the fuzzy dedup tier
//! ([`crate::fuzzy_dedup`]), with the CPU Model2Vec backend as a documented
//! fallback -- see [`crate::fuzzy_dedup::AutoBackend`] for the cascade.
//!
//! ## Why a second, larger model instead of just running Model2Vec on GPU
//!
//! Model2Vec's `potion-base-8M` is a *static* embedding model: it's
//! essentially a lookup table, and running it on a GPU wouldn't make its
//! judgments any better, only irrelevantly faster. Real quality headroom on
//! top of it comes from a full contextual encoder -- a Transformer that
//! actually reads a block as a sequence rather than bag-of-tokens. That's
//! genuinely GPU-shaped work, and it's the model2vec-rs README's own
//! reference point (Model2Vec is explicitly a distillation *of*
//! `bge-base-en-v1.5`): using the teacher model directly, when a GPU is
//! available to afford it, should only ever match or beat the distilled
//! student on similarity discrimination -- never do worse. Whether that
//! translates into materially more caught near-duplicates on real traffic
//! is unmeasured; see the eval-gate note in `docs/EXPERIMENTAL.md`.
//!
//! ## Sizing against a 2GB VRAM budget
//!
//! `BAAI/bge-base-en-v1.5` is 109M params: ~440MB of weights at f32 (what
//! `candle_transformers::models::bert::DTYPE` uses), ~220MB at f16. A CUDA
//! context alone typically costs 300-500MB before a single tensor is
//! allocated, and this module processes one text at a time (see
//! [`bert_backend::BertEmbedBackend::embed_batch`]) specifically to keep
//! activation memory small and predictable rather than trading VRAM
//! headroom for batch throughput. Net: comfortably fits a 2GB card with
//! room for the CUDA context, even before accounting for the fact that
//! Axiom's own TTT engine may want the same GPU concurrently. `bge-large-en
//! -v1.5` (335M params, ~1.3GB f32) is a drop-in alternative -- pass its
//! repo id to [`bert_backend::BertEmbedBackend::load`] -- but is not the
//! default here because it leaves much less headroom on a 2GB card once the
//! CUDA context and any concurrent TTT workload are paid for.
//!
//! Uses `candle-transformers`, pinned to the exact `0.10.2` release that
//! matches this crate's existing `candle-core`/`candle-nn = "0.10.2"` pins
//! (see `Cargo.toml`) -- deliberately *not* the newer `0.11.0` on crates.io,
//! to avoid forcing a candle upgrade across the whole engine (including
//! `model.rs`/`ttt_block.rs`) as a side effect of an embedding feature.

#[cfg(feature = "fuzzy-embed-gpu")]
pub mod bert_backend {
    use crate::fuzzy_dedup::SimilarityBackend;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config, DTYPE};
    use tokenizers::{Tokenizer, TruncationParams};

    /// See the module-level sizing note for why base, not large, is the
    /// default for a 2GB budget.
    pub const DEFAULT_GPU_MODEL_REPO: &str = "BAAI/bge-base-en-v1.5";
    const BERT_MAX_TOKENS: usize = 512;

    fn configure_truncation(tokenizer: &mut Tokenizer) -> anyhow::Result<()> {
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: BERT_MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(anyhow::Error::msg)?;
        Ok(())
    }

    pub struct BertEmbedBackend {
        model: BertModel,
        tokenizer: Tokenizer,
        device: Device,
    }

    impl BertEmbedBackend {
        /// Loads `repo` (a Hugging Face Hub repo id) onto `device`.
        /// `config.json` / `tokenizer.json` / `model.safetensors` are
        /// fetched from the Hub on first use and cached locally after --
        /// there is no bundled checkpoint, matching this crate's existing
        /// convention of never shipping model weights (see `Cargo.toml`'s
        /// `exclude` list, which already excludes `**/*.safetensors`).
        pub fn load(repo: &str, device: Device) -> anyhow::Result<Self> {
            let api = hf_hub::api::sync::Api::new()?;
            let repo_api = api.repo(hf_hub::Repo::new(repo.to_string(), hf_hub::RepoType::Model));

            let config_path = repo_api.get("config.json")?;
            let tokenizer_path = repo_api.get("tokenizer.json")?;
            let weights_path = repo_api.get("model.safetensors")?;

            let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
            let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
            configure_truncation(&mut tokenizer)?;

            // Safety: standard candle pattern (matches candle-examples/bert)
            // -- mmaps the safetensors file directly rather than reading it
            // fully into memory first, which matters more as models get
            // bigger. Requires the file not be mutated while mapped, which
            // holds here: it's a read-only Hub-cache download.
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)?
            };
            let model = BertModel::load(vb, &config)?;

            Ok(Self { model, tokenizer, device })
        }

        /// Convenience: try CUDA device 0 first (only when the `cuda`
        /// feature is compiled in), fall back to CPU (same model) on any
        /// failure -- no GPU present, driver mismatch, out of memory, etc.
        /// This is the GPU-first/CPU-fallback behavior *within* this one
        /// model. For falling back across *different* pretrained models
        /// (this one, then the much smaller Model2Vec model) when even a
        /// CPU forward pass through BERT isn't wanted, see
        /// [`crate::fuzzy_dedup::AutoBackend`].
        pub fn load_cuda_or_cpu(repo: &str) -> anyhow::Result<Self> {
            #[cfg(feature = "cuda")]
            {
                match Device::new_cuda(0) {
                    Ok(device) => {
                        Self::load(repo, device).or_else(|_| Self::load(repo, Device::Cpu))
                    }
                    Err(_) => Self::load(repo, Device::Cpu),
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                Self::load(repo, Device::Cpu)
            }
        }
    }

    impl SimilarityBackend for BertEmbedBackend {
        /// Processes one text at a time rather than batching with padding.
        /// This is a deliberate simplicity-over-throughput choice: getting
        /// an attention mask wrong on padded batches is a well-known way to
        /// silently corrupt embeddings, and this module has not been
        /// validated against a reference implementation's output. One text
        /// at a time needs no attention mask at all (nothing is padded), at
        /// the cost of not using the GPU's batching parallelism. Batched
        /// inference is a reasonable follow-up once this path has a
        /// correctness check to validate against.
        ///
        /// Any tokenization or forward-pass failure returns `None` for the
        /// *whole* call rather than a partial result or a panic -- fail-safe
        /// per this module's convention: a broken backend can only make
        /// `fuzzy_diet` do less, never fabricate a bad embedding.
        fn embed_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(texts.len());
            for text in texts {
                let encoding = self.tokenizer.encode(text.as_str(), true).ok()?;
                let token_ids = Tensor::new(encoding.get_ids(), &self.device)
                    .ok()?
                    .unsqueeze(0)
                    .ok()?;
                let token_type_ids = token_ids.zeros_like().ok()?;
                let hidden = self.model.forward(&token_ids, &token_type_ids, None).ok()?;
                // Reuses the exact pooling/normalization convention
                // Axiom's own embedder already uses (`embedder.rs`), so a
                // degenerate (all-zero) pooled vector is handled identically
                // regardless of which model produced it.
                let pooled = crate::embedder::pool_and_normalize(&hidden).ok()?;
                out.push(pooled);
            }
            Some(out)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use candle_nn::VarBuilder;
        use std::collections::HashMap;
        use tokenizers::{
            models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
            processors::template::TemplateProcessing,
        };

        #[test]
        fn long_input_is_truncated_before_bert_forward() {
            let vocab = HashMap::from([
                ("[UNK]".to_string(), 0),
                ("[CLS]".to_string(), 1),
                ("[SEP]".to_string(), 2),
                ("word".to_string(), 3),
            ]);
            let word_level = WordLevel::builder()
                .vocab(vocab)
                .unk_token("[UNK]".to_string())
                .build()
                .unwrap();
            let mut tokenizer = Tokenizer::new(word_level);
            tokenizer
                .with_pre_tokenizer(Whitespace::default())
                .with_post_processor(
                    TemplateProcessing::builder()
                        .try_single("[CLS] $A [SEP]")
                        .unwrap()
                        .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
                        .build()
                        .unwrap(),
                );
            configure_truncation(&mut tokenizer).unwrap();
            let long_input = std::iter::repeat("word")
                .take(BERT_MAX_TOKENS + 100)
                .collect::<Vec<_>>()
                .join(" ");
            let encoding = tokenizer.encode(long_input.as_str(), true).unwrap();
            assert_eq!(encoding.len(), BERT_MAX_TOKENS);
            assert_eq!(encoding.get_ids().first(), Some(&1));
            assert_eq!(encoding.get_ids().last(), Some(&2));

            let config = Config {
                vocab_size: 4,
                hidden_size: 8,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                intermediate_size: 16,
                ..Default::default()
            };
            let device = Device::Cpu;
            let model = BertModel::load(VarBuilder::zeros(DTYPE, &device), &config).unwrap();
            let backend = BertEmbedBackend {
                model,
                tokenizer,
                device,
            };

            let embeddings = backend
                .embed_batch(&[long_input])
                .expect("embedding succeeds");

            assert_eq!(embeddings.len(), 1);
            assert_eq!(embeddings[0].len(), config.hidden_size);
        }
    }
}
