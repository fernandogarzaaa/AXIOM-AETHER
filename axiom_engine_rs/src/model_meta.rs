//! Sidecar metadata for baked checkpoints + a pure VRAM auto-size estimator.
//!
//! The sidecar (`<checkpoint>.meta.json`) records the dims a checkpoint was
//! trained with so the proxy/eval load the right model without hardcoding.
//! The auto-size estimator picks the largest model config that fits a memory
//! budget, so "auto-size to VRAM" works end-to-end.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Persisted alongside a checkpoint as `<path>.meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMeta {
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub lr_inner: f32,
    pub norm_eps: f32,
    /// Best held-out cross-entropy seen during training (eval signal).
    pub val_ce: f32,
    /// Tokenizer file this model was trained against.
    pub tokenizer: String,
}

impl ModelMeta {
    /// Sidecar path for a checkpoint path: `foo.bin` -> `foo.meta.json`.
    pub fn sidecar_path(checkpoint: &str) -> String {
        let p = Path::new(checkpoint);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        match p.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => {
                format!("{}/{stem}.meta.json", dir.to_string_lossy())
            }
            _ => format!("{stem}.meta.json"),
        }
    }

    pub fn save(&self, checkpoint: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).expect("serialize meta");
        std::fs::write(Self::sidecar_path(checkpoint), json)
    }

    pub fn load(checkpoint: &str) -> Option<ModelMeta> {
        let txt = std::fs::read_to_string(Self::sidecar_path(checkpoint)).ok()?;
        serde_json::from_str(&txt).ok()
    }
}

/// One rung of the auto-size ladder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizeRung {
    pub d_model: usize,
    pub n_layers: usize,
}

/// Default ladder, smallest → largest. Ceiling tuned for ~3.7 GB usable VRAM.
pub fn default_ladder() -> Vec<SizeRung> {
    vec![
        SizeRung { d_model: 256, n_layers: 4 },
        SizeRung { d_model: 384, n_layers: 6 },
        SizeRung { d_model: 512, n_layers: 8 },
        SizeRung { d_model: 640, n_layers: 8 },
    ]
}

/// Estimate the training footprint (bytes) of a config: embedding + lm_head +
/// per-layer projections, times 4 (params + grad + AdamW m + v, all fp32),
/// plus a flat activation budget for one `win`-token forward.
pub fn estimate_footprint_bytes(
    d_model: usize,
    n_layers: usize,
    vocab: usize,
    win: usize,
) -> u64 {
    let params = 2 * vocab * d_model // embedding + lm_head
        + n_layers * 3 * d_model * d_model // w_q, w_k, w_v per layer
        + n_layers * d_model // layer norms (approx)
        + d_model; // final norm
    let param_bytes = params as u64 * 4 * 4; // fp32 × (param + grad + m + v)
    let activation_bytes = (win as u64) * (vocab as u64) * 4 * 3; // logits + softmax scratch
    param_bytes + activation_bytes
}

/// Pick the largest rung that fits `budget_bytes`. Always returns at least the
/// smallest rung, so training never refuses to start.
pub fn pick_config(budget_bytes: u64, vocab: usize, win: usize, ladder: &[SizeRung]) -> SizeRung {
    let mut chosen = ladder[0];
    for rung in ladder {
        let fp = estimate_footprint_bytes(rung.d_model, rung.n_layers, vocab, win);
        if fp <= budget_bytes {
            chosen = *rung;
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_path_swaps_extension() {
        assert_eq!(
            ModelMeta::sidecar_path("checkpoints/axiom_production_bpe.bin"),
            "checkpoints/axiom_production_bpe.meta.json"
        );
        assert_eq!(ModelMeta::sidecar_path("model.bin"), "model.meta.json");
    }

    #[test]
    fn meta_roundtrips_through_disk() {
        let m = ModelMeta {
            d_model: 512,
            n_layers: 8,
            vocab_size: 32000,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
            val_ce: 3.21,
            tokenizer: "t.json".into(),
        };
        let ckpt = std::env::temp_dir().join("axiom_meta_test.bin");
        let ckpt = ckpt.to_string_lossy().to_string();
        m.save(&ckpt).unwrap();
        assert_eq!(ModelMeta::load(&ckpt).unwrap(), m);
        let _ = std::fs::remove_file(ModelMeta::sidecar_path(&ckpt));
    }

    #[test]
    fn footprint_grows_with_size() {
        let small = estimate_footprint_bytes(256, 4, 16000, 512);
        let large = estimate_footprint_bytes(512, 8, 32000, 512);
        assert!(large > small);
    }

    #[test]
    fn pick_config_respects_budget() {
        let ladder = default_ladder();
        let tiny = pick_config(1, 32000, 512, &ladder);
        assert_eq!(tiny, ladder[0]);
        let huge = pick_config(u64::MAX, 32000, 512, &ladder);
        assert_eq!(huge, *ladder.last().unwrap());
    }
}
