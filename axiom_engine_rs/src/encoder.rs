//! Bidirectional transformer encoder for sentence/code embeddings.
//! Processes ONE sequence at a time → an L2-normalized `[d_model]` vector.
//! Trained contrastively (see `contrastive.rs`); NOT causal (unlike the TTT LM).

use std::path::Path;

use candle_core::{Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// RMSNorm with gamma initialized to ONES (candle's `vs.get` zero-inits, which
/// would zero a from-scratch model's output). `candle_nn::layer_norm` silently
/// breaks autograd in this candle build, so we roll our own differentiable norm.
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self { weight, eps })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let eps = Tensor::new(self.eps as f32, x.device())?;
        let normed = x.broadcast_div(&variance.broadcast_add(&eps)?.sqrt()?)?;
        normed.broadcast_mul(&self.weight)
    }
}

/// Static hyper-parameters for the encoder.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub max_seq: usize,
    pub norm_eps: f64,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 16000,
            d_model: 384,
            n_layers: 6,
            n_heads: 6,
            ffn_dim: 1536,
            max_seq: 256,
            norm_eps: 1e-5,
        }
    }
}

/// Sidecar persisted alongside `axiom_embedder.bin` as `axiom_embedder.meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbedderMeta {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub max_seq: usize,
    pub vocab_size: usize,
    pub tokenizer: String,
    /// best held-out pair recall@1 during training
    pub val_recall_at_1: f32,
}

impl EmbedderMeta {
    pub fn sidecar_path(checkpoint: &str) -> String {
        let p = Path::new(checkpoint);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("axiom_embedder");
        match p.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => {
                format!("{}/{stem}.meta.json", dir.to_string_lossy())
            }
            _ => format!("{stem}.meta.json"),
        }
    }
    pub fn save(&self, checkpoint: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).expect("serialize embedder meta");
        std::fs::write(Self::sidecar_path(checkpoint), json)
    }
    pub fn load(checkpoint: &str) -> Option<EmbedderMeta> {
        let txt = std::fs::read_to_string(Self::sidecar_path(checkpoint)).ok()?;
        serde_json::from_str(&txt).ok()
    }
}

/// Multi-head self-attention over a single sequence `[1, T, d]`. No causal mask
/// — every position attends to every position (bidirectional).
pub struct SelfAttention {
    w_q: candle_nn::Linear,
    w_k: candle_nn::Linear,
    w_v: candle_nn::Linear,
    w_o: candle_nn::Linear,
    n_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    pub fn new(vb: VarBuilder, d_model: usize, n_heads: usize) -> Result<Self> {
        assert!(d_model % n_heads == 0, "d_model must divide n_heads");
        Ok(Self {
            w_q: candle_nn::linear(d_model, d_model, vb.pp("w_q"))?,
            w_k: candle_nn::linear(d_model, d_model, vb.pp("w_k"))?,
            w_v: candle_nn::linear(d_model, d_model, vb.pp("w_v"))?,
            w_o: candle_nn::linear(d_model, d_model, vb.pp("w_o"))?,
            n_heads,
            head_dim: d_model / n_heads,
        })
    }

    /// `x`: `[1, T, d]` → `[1, T, d]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let h = self.n_heads;
        let hd = self.head_dim;
        let scale = 1.0 / (hd as f64).sqrt();

        // project → [1,T,d] → [1,T,H,hd] → [1,H,T,hd]
        let to_heads = |t_in: Tensor| -> Result<Tensor> {
            t_in.reshape((b, t, h, hd))?.transpose(1, 2)?.contiguous()
        };
        let q = to_heads(self.w_q.forward(x)?)?;
        let k = to_heads(self.w_k.forward(x)?)?;
        let v = to_heads(self.w_v.forward(x)?)?;

        // scores [1,H,T,T] = q @ k^T * scale
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scores = q.matmul(&k_t)?.affine(scale, 0.0)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;

        // ctx [1,H,T,hd] = attn @ v → back to [1,T,d]
        let ctx = attn.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, d))?;
        self.w_o.forward(&ctx)
    }
}

/// One transformer encoder block: self-attention + FFN, each residual + LayerNorm.
pub struct EncoderBlock {
    attn: SelfAttention,
    norm1: RmsNorm,
    ff1: candle_nn::Linear,
    ff2: candle_nn::Linear,
    norm2: RmsNorm,
}

impl EncoderBlock {
    pub fn new(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::new(vb.pp("attn"), cfg.d_model, cfg.n_heads)?,
            // candle_nn::layer_norm silently breaks autograd in this candle build;
            // RMSNorm (kernel.rs) is the in-house, proven-differentiable norm.
            norm1: RmsNorm::new(cfg.d_model, cfg.norm_eps, vb.pp("norm1"))?,
            ff1: candle_nn::linear(cfg.d_model, cfg.ffn_dim, vb.pp("ff1"))?,
            ff2: candle_nn::linear(cfg.ffn_dim, cfg.d_model, vb.pp("ff2"))?,
            norm2: RmsNorm::new(cfg.d_model, cfg.norm_eps, vb.pp("norm2"))?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.attn.forward(x)?;
        let x = self.norm1.forward(&(x + a)?)?;
        let h = self.ff2.forward(&self.ff1.forward(&x)?.gelu()?)?;
        self.norm2.forward(&(x + h)?)
    }
}

/// Full bidirectional encoder. `encode` → an L2-normalized `[d_model]` vector.
pub struct BiEncoder {
    tok_emb: candle_nn::Embedding,
    pos_emb: candle_nn::Embedding,
    blocks: Vec<EncoderBlock>,
    norm_f: RmsNorm,
    pub config: EncoderConfig,
}

impl BiEncoder {
    pub fn new(vb: VarBuilder, config: EncoderConfig) -> Result<Self> {
        let tok_emb = candle_nn::embedding(config.vocab_size, config.d_model, vb.pp("tok_emb"))?;
        let pos_emb = candle_nn::embedding(config.max_seq, config.d_model, vb.pp("pos_emb"))?;
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            blocks.push(EncoderBlock::new(vb.pp(format!("block_{i}")), &config)?);
        }
        let norm_f = RmsNorm::new(config.d_model, config.norm_eps, vb.pp("norm_f"))?;
        Ok(Self { tok_emb, pos_emb, blocks, norm_f, config })
    }

    /// Encode one token sequence `[1, T]` → L2-normalized `[d_model]`.
    pub fn encode(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, t) = input_ids.dims2()?;
        let device = input_ids.device();
        let tok = self.tok_emb.forward(input_ids)?; // [1,T,d]
        let pos_ids = Tensor::arange(0u32, t as u32, device)?.reshape((1, t))?;
        let pos = self.pos_emb.forward(&pos_ids)?; // [1,T,d]
        let mut x = (tok + pos)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        x = self.norm_f.forward(&x)?;
        // mean-pool over T → [1,d] → [d]
        let pooled = x.mean(1)?.squeeze(0)?;
        // L2 normalize
        let norm = pooled.sqr()?.sum_all()?.sqrt()?;
        let norm = norm.broadcast_add(&Tensor::new(1e-8f32, device)?)?;
        pooled.broadcast_div(&norm)
    }

    /// Truncate ids to `max_seq` and wrap as a `[1, T]` tensor.
    pub fn ids_tensor(&self, ids: &[u32], device: &candle_core::Device) -> Result<Tensor> {
        let mut ids = ids.to_vec();
        if ids.is_empty() {
            ids.push(0);
        }
        ids.truncate(self.config.max_seq);
        let t = ids.len();
        Tensor::from_vec(ids, (1, t), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    fn vb_cpu() -> (VarMap, VarBuilder<'static>) {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        (vm, vb)
    }

    #[test]
    fn embedder_meta_sidecar_path() {
        assert_eq!(
            EmbedderMeta::sidecar_path("checkpoints/axiom_embedder.bin"),
            "checkpoints/axiom_embedder.meta.json"
        );
    }

    #[test]
    fn embedder_meta_roundtrips() {
        let m = EmbedderMeta {
            d_model: 384,
            n_layers: 6,
            n_heads: 6,
            ffn_dim: 1536,
            max_seq: 256,
            vocab_size: 16000,
            tokenizer: "t.json".into(),
            val_recall_at_1: 0.9,
        };
        let ckpt = std::env::temp_dir().join("axiom_emb_meta_test.bin");
        let ckpt = ckpt.to_string_lossy().to_string();
        m.save(&ckpt).unwrap();
        assert_eq!(EmbedderMeta::load(&ckpt).unwrap(), m);
        let _ = std::fs::remove_file(EmbedderMeta::sidecar_path(&ckpt));
    }

    #[test]
    fn attention_preserves_shape() {
        let (_vm, vb) = vb_cpu();
        let attn = SelfAttention::new(vb, 16, 4).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1usize, 5usize, 16usize), &Device::Cpu).unwrap();
        let y = attn.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 5, 16]);
    }

    #[test]
    fn attention_is_bidirectional() {
        // Changing the LAST token must change position 0's output (no causal mask).
        let (_vm, vb) = vb_cpu();
        let attn = SelfAttention::new(vb, 16, 4).unwrap();
        let x1 = Tensor::randn(0f32, 1f32, (1usize, 4usize, 16usize), &Device::Cpu).unwrap();
        let mut data: Vec<f32> = x1.flatten_all().unwrap().to_vec1().unwrap();
        for i in (3 * 16)..(4 * 16) {
            data[i] += 5.0;
        }
        let x2 = Tensor::from_vec(data, (1, 4, 16), &Device::Cpu).unwrap();
        let y1 = attn.forward(&x1).unwrap();
        let y2 = attn.forward(&x2).unwrap();
        let p0_1: Vec<f32> = y1.narrow(1, 0, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let p0_2: Vec<f32> = y2.narrow(1, 0, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(p0_1, p0_2, "position 0 output must depend on later tokens");
    }

    fn tiny_encoder() -> (VarMap, BiEncoder) {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = EncoderConfig {
            vocab_size: 64,
            d_model: 16,
            n_layers: 2,
            n_heads: 4,
            ffn_dim: 32,
            max_seq: 32,
            norm_eps: 1e-5,
        };
        let enc = BiEncoder::new(vb, cfg).unwrap();
        (vm, enc)
    }

    #[test]
    fn encode_returns_unit_vector_of_d_model() {
        let (_vm, enc) = tiny_encoder();
        let ids = enc.ids_tensor(&[1, 2, 3, 4, 5], &Device::Cpu).unwrap();
        let v = enc.encode(&ids).unwrap();
        assert_eq!(v.dims(), &[16]);
        let n: f32 = v.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar().unwrap();
        assert!((n - 1.0).abs() < 1e-4, "norm {n}");
    }

    #[test]
    fn encode_is_differentiable_wrt_all_params() {
        // Regression: every encoder Var must receive a gradient. (candle_nn's
        // layer_norm silently severed autograd here — we use RMSNorm instead.)
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let cfg = EncoderConfig {
            vocab_size: 64, d_model: 16, n_layers: 2, n_heads: 4,
            ffn_dim: 32, max_seq: 32, norm_eps: 1e-5,
        };
        let enc = BiEncoder::new(vb, cfg).unwrap();
        let ids = enc.ids_tensor(&[1, 2, 3, 4], &dev).unwrap();
        let loss = enc.encode(&ids).unwrap().sqr().unwrap().sum_all().unwrap();
        let g = loss.backward().unwrap();
        let vars = vm.all_vars();
        let with_grad = vars.iter().filter(|v| g.get(v.as_tensor()).is_some()).count();
        assert_eq!(with_grad, vars.len(), "all {} encoder params must get grad, got {}", vars.len(), with_grad);
    }

    #[test]
    fn different_inputs_give_different_embeddings() {
        let (_vm, enc) = tiny_encoder();
        let a = enc.encode(&enc.ids_tensor(&[1, 2, 3, 4], &Device::Cpu).unwrap()).unwrap();
        let b = enc.encode(&enc.ids_tensor(&[40, 41, 42, 43], &Device::Cpu).unwrap()).unwrap();
        let av: Vec<f32> = a.to_vec1().unwrap();
        let bv: Vec<f32> = b.to_vec1().unwrap();
        let cos: f32 = av.iter().zip(&bv).map(|(x, y)| x * y).sum();
        assert!(cos < 0.999, "untrained encoder embeddings suspiciously identical: cos={cos}");
    }
}
