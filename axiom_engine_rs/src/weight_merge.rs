//! Cross-session fast-weight checkpoint merging.
//!
//! Persisted compression caches contain per-session W_tilde tensors. This module
//! merges those tensors through task-vector arithmetic:
//! `merged = W_base + alpha * mean(W_tilde_i - W_base)`.

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerWeights {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionCheckpoint {
    session_id: String,
    version: u32,
    created_at: u64,
    layers: Vec<LayerWeights>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedCompressionEntry {
    session_id: String,
    context_hash: String,
    checkpoint: SessionCheckpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedCompressionCache {
    version: u32,
    entries: Vec<PersistedCompressionEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeSummary {
    pub output_path: String,
    pub input_files: usize,
    pub input_sessions: usize,
    pub layers: usize,
    pub d_model: usize,
    pub alpha: f32,
}

pub fn merge_checkpoint_files(
    inputs: &[PathBuf],
    output: &Path,
    alpha: f32,
) -> Result<MergeSummary, String> {
    if inputs.is_empty() {
        return Err("at least one input checkpoint is required".to_string());
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err("alpha must be in [0, 1]".to_string());
    }

    let mut sessions = Vec::new();
    for input in inputs {
        let bytes = fs::read(input).map_err(|e| format!("read {} failed: {e}", input.display()))?;
        let cache: PersistedCompressionCache = bincode::deserialize(&bytes)
            .map_err(|e| format!("decode {} failed: {e}", input.display()))?;
        if cache.version != 1 {
            return Err(format!(
                "unsupported cache version {} in {}",
                cache.version,
                input.display()
            ));
        }
        sessions.extend(
            cache
                .entries
                .into_iter()
                .map(|entry| entry.checkpoint.layers),
        );
    }
    if sessions.is_empty() {
        return Err("input checkpoints contain no sessions".to_string());
    }

    let merged = merge_layer_stacks(&sessions, alpha)?;
    let d_model = merged
        .first()
        .and_then(|layer| layer.shape.first())
        .copied()
        .unwrap_or(0);
    let payload = PersistedCompressionCache {
        version: 1,
        entries: vec![PersistedCompressionEntry {
            session_id: "merged".to_string(),
            context_hash: merge_hash(inputs, alpha),
            checkpoint: SessionCheckpoint {
                session_id: "merged".to_string(),
                version: 1,
                created_at: unix_now(),
                layers: merged.clone(),
            },
        }],
    };

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir {} failed: {e}", parent.display()))?;
    }
    let bytes =
        bincode::serialize(&payload).map_err(|e| format!("encode merged cache failed: {e}"))?;
    fs::write(output, bytes).map_err(|e| format!("write {} failed: {e}", output.display()))?;

    Ok(MergeSummary {
        output_path: output.display().to_string(),
        input_files: inputs.len(),
        input_sessions: sessions.len(),
        layers: merged.len(),
        d_model,
        alpha,
    })
}

pub fn merge_layer_stacks(
    stacks: &[Vec<LayerWeights>],
    alpha: f32,
) -> Result<Vec<LayerWeights>, String> {
    let first = stacks
        .first()
        .ok_or_else(|| "no layer stacks supplied".to_string())?;
    let layer_count = first.len();
    if layer_count == 0 {
        return Err("layer stacks are empty".to_string());
    }
    for stack in stacks {
        if stack.len() != layer_count {
            return Err("all checkpoints must have the same layer count".to_string());
        }
    }

    (0..layer_count)
        .map(|layer_index| merge_one_layer(stacks, layer_index, alpha))
        .collect()
}

fn merge_one_layer(
    stacks: &[Vec<LayerWeights>],
    layer_index: usize,
    alpha: f32,
) -> Result<LayerWeights, String> {
    let shape = stacks[0][layer_index].shape.clone();
    let d_model = square_d_model(&shape)?;
    for stack in stacks {
        let layer = &stack[layer_index];
        if layer.shape != shape {
            return Err(format!("layer {layer_index} shape mismatch"));
        }
        if layer.data.len() != d_model * d_model {
            return Err(format!(
                "layer {layer_index} data length does not match shape"
            ));
        }
    }

    let device = Device::Cpu;
    let base = Tensor::eye(d_model, DType::F32, &device)
        .map_err(|e| format!("base tensor failed: {e}"))?;
    let mut delta_sum = Tensor::zeros((d_model, d_model), DType::F32, &device)
        .map_err(|e| format!("delta tensor failed: {e}"))?;
    for stack in stacks {
        let layer = Tensor::from_vec(
            stacks_data(&stack[layer_index]),
            (d_model, d_model),
            &device,
        )
        .map_err(|e| format!("layer tensor failed: {e}"))?;
        let delta = layer.sub(&base).map_err(|e| format!("delta failed: {e}"))?;
        delta_sum = delta_sum
            .add(&delta)
            .map_err(|e| format!("delta accumulate failed: {e}"))?;
    }
    let mean_delta = delta_sum
        .affine(1.0 / stacks.len() as f64, 0.0)
        .map_err(|e| format!("delta average failed: {e}"))?;
    let scaled_delta = mean_delta
        .affine(alpha as f64, 0.0)
        .map_err(|e| format!("delta scale failed: {e}"))?;
    let merged = base
        .add(&scaled_delta)
        .map_err(|e| format!("merge failed: {e}"))?;
    let data = merged
        .flatten_all()
        .map_err(|e| format!("flatten failed: {e}"))?
        .to_vec1::<f32>()
        .map_err(|e| format!("extract failed: {e}"))?;
    Ok(LayerWeights { shape, data })
}

fn stacks_data(layer: &LayerWeights) -> Vec<f32> {
    layer.data.clone()
}

fn square_d_model(shape: &[usize]) -> Result<usize, String> {
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(format!(
            "expected square [d_model, d_model] layer, got {shape:?}"
        ));
    }
    Ok(shape[0])
}

fn merge_hash(inputs: &[PathBuf], alpha: f32) -> String {
    let mut hasher = Sha256::new();
    hasher.update(alpha.to_le_bytes());
    for input in inputs {
        hasher.update(input.to_string_lossy().as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[allow(dead_code)]
pub fn write_test_cache(path: &Path, session_id: &str, layers: Vec<LayerWeights>) {
    let payload = PersistedCompressionCache {
        version: 1,
        entries: vec![PersistedCompressionEntry {
            session_id: session_id.to_string(),
            context_hash: format!("sha256:{session_id}"),
            checkpoint: SessionCheckpoint {
                session_id: session_id.to_string(),
                version: 1,
                created_at: 0,
                layers,
            },
        }],
    };
    let bytes = bincode::serialize(&payload).unwrap();
    fs::write(path, bytes).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_preserves_layer_shapes() {
        let a = vec![LayerWeights {
            shape: vec![2, 2],
            data: vec![1.2, 0.0, 0.0, 1.2],
        }];
        let b = vec![LayerWeights {
            shape: vec![2, 2],
            data: vec![1.4, 0.0, 0.0, 1.4],
        }];
        let merged = merge_layer_stacks(&[a, b], 0.5).unwrap();
        assert_eq!(merged[0].shape, vec![2, 2]);
        assert_eq!(merged[0].data.len(), 4);
        assert!((merged[0].data[0] - 1.15).abs() < 1e-5);
        assert!((merged[0].data[3] - 1.15).abs() < 1e-5);
    }
}
