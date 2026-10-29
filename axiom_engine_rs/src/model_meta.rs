//! Sidecar metadata for baked checkpoints + a pure VRAM auto-size estimator.
//!
//! The sidecar (`<checkpoint>.meta.json`) records the dims a checkpoint was
//! trained with so the proxy/eval load the right model without hardcoding.
//! The auto-size estimator picks the largest model config that fits a memory
//! budget, so "auto-size to VRAM" works end-to-end.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Sidecar schema version this engine writes and the newest it understands.
///
/// Bump it when a field is added whose *absence* would make an older engine
/// misread a checkpoint (as opposed to merely ignore an optional extra).
/// Sidecars written before this field existed deserialize as `0` and are
/// accepted; a sidecar declaring a version newer than this constant is
/// refused by [`ModelMeta::load_checked`] so the engine fails loudly instead
/// of building a model with the wrong shape.
pub const MODEL_META_SCHEMA_VERSION: u32 = 1;

/// Persisted alongside a checkpoint as `<path>.meta.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelMeta {
    /// Sidecar schema version (see [`MODEL_META_SCHEMA_VERSION`]). `0` means
    /// the sidecar predates versioning. `save` always stamps the current value.
    #[serde(default)]
    pub schema_version: u32,
    /// `CARGO_PKG_VERSION` of the engine that wrote the sidecar. Provenance
    /// only: never used to decide whether a checkpoint loads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_version: Option<String>,
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub lr_inner: f32,
    pub norm_eps: f32,
    /// Best held-out cross-entropy seen during training (eval signal).
    pub val_ce: f32,
    /// Tokenizer file this model was trained against.
    pub tokenizer: String,
    /// Whether this checkpoint was trained with inner-loop stabilization
    /// (normalized keys + state clamp). The proxy/eval must run it the same way.
    /// `#[serde(default)]` → old sidecars (no field) load as `false`.
    #[serde(default)]
    pub stabilize: bool,
    /// Whether training used the detached-chunk final-token loss fast path.
    /// This is training provenance only; inference remains unchanged.
    #[serde(default)]
    pub last_token_only: bool,
    /// Whether this checkpoint was trained with the *learned* data-dependent
    /// Gated-DeltaNet forget gate (a per-layer `w_α` projection that decides how
    /// much memory to retain per token). It adds weights, so the model must be
    /// built with it enabled to load such a checkpoint. `#[serde(default)]` → old
    /// sidecars load as `false`, so the parameter-free path is used and existing
    /// checkpoints are unaffected.
    #[serde(default)]
    pub learned_gate: bool,
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

    /// Whether this engine understands the sidecar's schema. Older (including
    /// unversioned) sidecars are always supported; newer ones are not.
    pub fn is_supported(&self) -> bool {
        self.schema_version <= MODEL_META_SCHEMA_VERSION
    }

    /// Write the sidecar, stamping the current schema and engine versions so
    /// the on-disk file always records what produced it.
    pub fn save(&self, checkpoint: &str) -> std::io::Result<()> {
        let stamped = ModelMeta {
            schema_version: MODEL_META_SCHEMA_VERSION,
            engine_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            ..self.clone()
        };
        let json = serde_json::to_string_pretty(&stamped).expect("serialize meta");
        std::fs::write(Self::sidecar_path(checkpoint), json)
    }

    /// Load the sidecar, distinguishing the three outcomes a caller must treat
    /// differently:
    ///
    /// * `Ok(None)` -- no sidecar (or unparsable): fall back to defaults, the
    ///   historical behaviour for checkpoints that never had one.
    /// * `Ok(Some(meta))` -- a sidecar this engine understands.
    /// * `Err(msg)` -- a sidecar from a *newer* engine. Falling back to
    ///   defaults here would silently build the wrong model shape, so callers
    ///   on a production load path should refuse to start.
    pub fn load_checked(checkpoint: &str) -> Result<Option<ModelMeta>, String> {
        let path = Self::sidecar_path(checkpoint);
        let Ok(txt) = std::fs::read_to_string(&path) else {
            return Ok(None);
        };
        let Ok(meta) = serde_json::from_str::<ModelMeta>(&txt) else {
            return Ok(None);
        };
        if !meta.is_supported() {
            return Err(format!(
                "checkpoint sidecar {path} declares schema_version {} (written by engine {}), \
                 but this engine ({}) only understands schema_version <= {}; \
                 upgrade axiom or use a checkpoint trained by this version",
                meta.schema_version,
                meta.engine_version.as_deref().unwrap_or("unknown"),
                env!("CARGO_PKG_VERSION"),
                MODEL_META_SCHEMA_VERSION
            ));
        }
        Ok(Some(meta))
    }

    /// Permissive load for tooling: a sidecar from a newer engine is reported
    /// on stderr and treated as absent. Production paths use
    /// [`ModelMeta::load_checked`] and fail instead.
    pub fn load(checkpoint: &str) -> Option<ModelMeta> {
        match Self::load_checked(checkpoint) {
            Ok(meta) => meta,
            Err(msg) => {
                eprintln!("[axiom] WARN: {msg}; ignoring sidecar");
                None
            }
        }
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
        SizeRung {
            d_model: 256,
            n_layers: 4,
        },
        SizeRung {
            d_model: 384,
            n_layers: 6,
        },
        SizeRung {
            d_model: 512,
            n_layers: 8,
        },
        SizeRung {
            d_model: 640,
            n_layers: 8,
        },
    ]
}

/// Estimate the training footprint (bytes) of a config: embedding + lm_head +
/// per-layer projections, times 4 (params + grad + AdamW m + v, all fp32),
/// plus a flat activation budget for one `win`-token forward.
pub fn estimate_footprint_bytes(d_model: usize, n_layers: usize, vocab: usize, win: usize) -> u64 {
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
            schema_version: MODEL_META_SCHEMA_VERSION,
            engine_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            d_model: 512,
            n_layers: 8,
            vocab_size: 32000,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
            val_ce: 3.21,
            tokenizer: "t.json".into(),
            stabilize: true,
            last_token_only: true,
            learned_gate: true,
        };
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("axiom_meta_test.bin");
        let ckpt = ckpt.to_string_lossy().to_string();
        m.save(&ckpt).unwrap();
        assert_eq!(ModelMeta::load(&ckpt).unwrap(), m);
    }

    #[test]
    fn save_stamps_schema_and_engine_version() {
        // A caller that leaves the version fields at their defaults still
        // produces a fully stamped sidecar on disk.
        let m = ModelMeta {
            d_model: 256,
            n_layers: 4,
            vocab_size: 8000,
            tokenizer: "t.json".into(),
            ..Default::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("stamp.bin");
        let ckpt = ckpt.to_string_lossy().to_string();
        m.save(&ckpt).unwrap();
        let loaded = ModelMeta::load_checked(&ckpt).unwrap().unwrap();
        assert_eq!(loaded.schema_version, MODEL_META_SCHEMA_VERSION);
        assert_eq!(
            loaded.engine_version.as_deref(),
            Some(env!("CARGO_PKG_VERSION"))
        );
        assert_eq!(loaded.d_model, 256);
    }

    #[test]
    fn unversioned_sidecar_loads_as_schema_zero() {
        // Sidecars written before versioning existed carry no `schema_version`
        // and must keep loading unchanged.
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("legacy.bin");
        let ckpt = ckpt.to_string_lossy().to_string();
        std::fs::write(
            ModelMeta::sidecar_path(&ckpt),
            r#"{"d_model":384,"n_layers":6,"vocab_size":8000,"lr_inner":0.001,
                "norm_eps":1e-6,"val_ce":5.2,"tokenizer":"bpe.json"}"#,
        )
        .unwrap();
        let loaded = ModelMeta::load_checked(&ckpt).unwrap().unwrap();
        assert_eq!(loaded.schema_version, 0);
        assert!(loaded.is_supported());
        assert_eq!((loaded.d_model, loaded.n_layers), (384, 6));
        assert!(!loaded.stabilize && !loaded.last_token_only && !loaded.learned_gate);
    }

    #[test]
    fn newer_sidecar_is_refused_by_checked_load_and_ignored_by_permissive_load() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("future.bin");
        let ckpt = ckpt.to_string_lossy().to_string();
        let newer = MODEL_META_SCHEMA_VERSION + 1;
        std::fs::write(
            ModelMeta::sidecar_path(&ckpt),
            format!(
                r#"{{"schema_version":{newer},"engine_version":"99.0.0","d_model":512,
                    "n_layers":8,"vocab_size":8000,"lr_inner":0.001,"norm_eps":1e-6,
                    "val_ce":1.0,"tokenizer":"bpe.json"}}"#
            ),
        )
        .unwrap();
        let err = ModelMeta::load_checked(&ckpt).unwrap_err();
        assert!(err.contains(&format!("schema_version {newer}")), "{err}");
        assert!(err.contains("99.0.0"), "{err}");
        assert!(ModelMeta::load(&ckpt).is_none());
    }

    #[test]
    fn missing_sidecar_is_ok_none() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("absent.bin");
        assert_eq!(
            ModelMeta::load_checked(&ckpt.to_string_lossy()).unwrap(),
            None
        );
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
