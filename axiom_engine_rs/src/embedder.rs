//! Axiom-as-embedder: turn text into a dense, L2-normalized vector by
//! mean-pooling the model's final hidden states. No external embedding model.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};

use crate::encoder::{BiEncoder, EmbedderMeta, EncoderConfig};
use crate::inference::InferencePipeline;

/// Max tokens fed through the model for a single embedding. Bounds compute and
/// matches the project's ≤512-token windowing discipline.
pub const EMBED_MAX_TOKENS: usize = 512;

/// Mean-pool a `[1, T, d_model]` hidden-state tensor over the sequence dim and
/// L2-normalize the result into a `[d_model]` unit vector.
///
/// Normalization is exact: a non-zero pooled vector comes back at unit length.
/// A degenerate all-zero pooled vector (e.g. an untrained model) is returned
/// as-is (all zeros) — `memory_recall::should_recall` treats that as "no
/// signal" and skips retrieval rather than matching noise.
pub fn pool_and_normalize(hidden: &Tensor) -> Result<Vec<f32>> {
    // [1, T, d_model] → mean over T → [1, d_model] → [d_model].
    let pooled = hidden.mean(1)?.squeeze(0)?;
    let pooled_vec = pooled.to_vec1::<f32>()?;
    let norm: f32 = pooled_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return Ok(pooled_vec); // degenerate: no signal, leave as zeros
    }
    Ok(pooled_vec.into_iter().map(|x| x / norm).collect())
}

/// Take the LAST token's hidden state from a `[1, T, d_model]` tensor and
/// L2-normalize it into a `[d_model]` unit vector.
///
/// For an autoregressive TTT model the final token has absorbed the whole
/// sequence through the per-layer `W̃` state, so its hidden state is a stronger
/// summary than a mean-pool (which tends to cancel out). Empirically this beats
/// mean-pooling on the recall eval, so it is the default for `embed_text`.
pub fn pool_last_normalize(hidden: &Tensor) -> Result<Vec<f32>> {
    let t = hidden.dim(1)?;
    let idx = if t == 0 { 0 } else { t - 1 };
    let last = hidden.narrow(1, idx, 1)?.squeeze(1)?.squeeze(0)?; // [d_model]
    let v = last.to_vec1::<f32>()?;
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return Ok(v);
    }
    Ok(v.into_iter().map(|x| x / norm).collect())
}

/// Embed `text` into an L2-normalized `[d_model]` vector (last-token pooling).
pub fn embed_text(pipeline: &InferencePipeline, text: &str) -> Result<Vec<f32>> {
    let device = pipeline.device();
    let mut ids = pipeline.encode_text(text);
    if ids.is_empty() {
        ids.push(0);
    }
    ids.truncate(EMBED_MAX_TOKENS);
    let len = ids.len();

    let input = Tensor::from_vec(ids, (1, len), device)?;
    let mut states = pipeline.init_session_states()?;

    // [1, T, d_model] final normed hidden states.
    let hidden = pipeline.model().forward_hidden(&input, &mut states)?;
    pool_last_normalize(&hidden)
}

/// A loaded contrastive bidirectional encoder + tokenizer — the Phase-2.0.1
/// embedder. Produced by `train_embedder` into `axiom_embedder.bin`.
pub struct EmbeddingModel {
    encoder: BiEncoder,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    _varmap: VarMap,
}

impl EmbeddingModel {
    /// Load from `axiom_embedder.bin` (+ sidecar) and its tokenizer. Returns
    /// `None` if the artifact, sidecar, or tokenizer is absent (→ caller falls
    /// back to the TTT pooling path).
    pub fn load(checkpoint: &str, device: Device) -> Option<Self> {
        if !std::path::Path::new(checkpoint).exists() {
            return None;
        }
        let meta = EmbedderMeta::load(checkpoint)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&meta.tokenizer).ok()?;
        let cfg = EncoderConfig {
            vocab_size: meta.vocab_size,
            d_model: meta.d_model,
            n_layers: meta.n_layers,
            n_heads: meta.n_heads,
            ffn_dim: meta.ffn_dim,
            max_seq: meta.max_seq,
            norm_eps: 1e-5,
        };
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder = BiEncoder::new(vb, cfg).ok()?;
        varmap.load(checkpoint).ok()?;
        Some(Self { encoder, tokenizer, device, _varmap: varmap })
    }

    /// Embed `text` into an L2-normalized `[d_model]` vector via the encoder.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let ids =
            self.tokenizer.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default();
        let t = self.encoder.ids_tensor(&ids, &self.device)?;
        self.encoder.encode(&t)?.to_vec1::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AxiomConfig;
    use candle_core::Device;

    #[test]
    fn embedding_model_absent_returns_none() {
        let none = EmbeddingModel::load("____no_embedder____.bin", Device::Cpu);
        assert!(none.is_none());
    }

    fn tiny_pipeline() -> InferencePipeline {
        // No checkpoint on disk → random init; fine for shape/determinism tests.
        let config = AxiomConfig {
            d_model: 16,
            n_layers: 2,
            vocab_size: 64,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        };
        InferencePipeline::with_checkpoint(config, Device::Cpu, "____no_such_checkpoint____")
            .unwrap()
    }

    #[test]
    fn pool_and_normalize_produces_unit_vector() {
        // hidden [1, 2, 3]: tokens (3,4,0) and (0,0,0) → mean (1.5,2,0)
        // → norm 2.5 → normalized (0.6, 0.8, 0.0).
        let h = Tensor::from_vec(
            vec![3f32, 4., 0., 0., 0., 0.],
            (1, 2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let v = pool_and_normalize(&h).unwrap();
        assert_eq!(v.len(), 3);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
        assert!(v[2].abs() < 1e-5);
    }

    #[test]
    fn pool_last_normalize_takes_final_token() {
        // hidden [1, 2, 3]: token0 (1,0,0), token1 (0,3,4) → last (0,3,4)
        // → norm 5 → normalized (0, 0.6, 0.8).
        let h = Tensor::from_vec(vec![1f32, 0., 0., 0., 3., 4.], (1, 2, 3), &Device::Cpu).unwrap();
        let v = pool_last_normalize(&h).unwrap();
        assert_eq!(v.len(), 3);
        assert!(v[0].abs() < 1e-5);
        assert!((v[1] - 0.6).abs() < 1e-5);
        assert!((v[2] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn pool_and_normalize_zero_stays_zero() {
        let h = Tensor::zeros((1, 2, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let v = pool_and_normalize(&h).unwrap();
        assert_eq!(v, vec![0.0; 4]);
    }

    #[test]
    fn embedding_has_d_model_length() {
        let p = tiny_pipeline();
        let v = embed_text(&p, "fn main() { println!(\"hi\"); }").unwrap();
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn embedding_is_deterministic() {
        let p = tiny_pipeline();
        let a = embed_text(&p, "same input text").unwrap();
        let b = embed_text(&p, "same input text").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn empty_text_does_not_panic() {
        let p = tiny_pipeline();
        let v = embed_text(&p, "").unwrap();
        assert_eq!(v.len(), 16);
    }
}
