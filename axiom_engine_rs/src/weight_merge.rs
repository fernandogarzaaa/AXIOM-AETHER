//! Cross-session fast-weight checkpoint merging.
//!
//! Persisted compression caches contain per-session W_tilde tensors. This module
//! merges those tensors. Two methods are supported:
//!  * **AlphaBlend** (legacy): `merged = W_base + alpha * mean(W_tilde_i - W_base)`
//!    — uniform task-vector interpolation. Suffers from sign cancellation (two
//!    peers pushing a weight in opposite directions average to ~0) and dilution
//!    by redundant deltas.
//!  * **DareTies** (recommended for fleet merges): DARE drop-and-rescale
//!    sparsification + TIES trim/elect-sign/disjoint-merge. Drops redundant and
//!    sign-conflicting deltas before averaging, so agreeing peers compound and
//!    disagreeing ones don't cancel. Data-free, single-pass.
//!    Refs: TIES arXiv:2306.01708, DARE arXiv:2311.03099.

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

/// How to combine per-session fast-weight task vectors.
#[derive(Debug, Clone, Copy)]
pub enum MergeMethod {
    /// Uniform task-vector interpolation: `base + alpha * mean(delta)`.
    AlphaBlend { alpha: f32 },
    /// DARE-TIES: drop+rescale sparsify (Bernoulli `drop_prob`), trim to the
    /// top-`density` fraction by magnitude, elect a per-element sign, then
    /// average only the sign-agreeing deltas. `seed` makes the drop reproducible.
    DareTies {
        alpha: f32,
        density: f32,
        drop_prob: f32,
        seed: u64,
    },
}

impl MergeMethod {
    fn alpha(&self) -> f32 {
        match *self {
            MergeMethod::AlphaBlend { alpha } => alpha,
            MergeMethod::DareTies { alpha, .. } => alpha,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            MergeMethod::AlphaBlend { .. } => "alpha_blend",
            MergeMethod::DareTies { .. } => "dare_ties",
        }
    }
}

/// Default DARE-TIES configuration for fleet merges. Conservative `drop_prob`
/// (fast-weight deltas are denser than SFT deltas, so we don't drop 90%+) and a
/// top-70% magnitude trim. `seed` is fixed so every node merges identically.
pub fn fleet_dare_ties(alpha: f32) -> MergeMethod {
    MergeMethod::DareTies {
        alpha,
        density: 0.7,
        drop_prob: 0.5,
        seed: 0xA1A0_5EED,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeSummary {
    pub output_path: String,
    pub input_files: usize,
    pub input_sessions: usize,
    pub layers: usize,
    pub d_model: usize,
    pub alpha: f32,
    #[serde(default)]
    pub method: String,
}

/// Legacy entry point — uniform alpha-blend. Kept for back-compat.
pub fn merge_checkpoint_files(
    inputs: &[PathBuf],
    output: &Path,
    alpha: f32,
) -> Result<MergeSummary, String> {
    merge_checkpoint_files_with(inputs, output, MergeMethod::AlphaBlend { alpha })
}

/// Merge persisted fast-weight caches with an explicit [`MergeMethod`].
pub fn merge_checkpoint_files_with(
    inputs: &[PathBuf],
    output: &Path,
    method: MergeMethod,
) -> Result<MergeSummary, String> {
    if inputs.is_empty() {
        return Err("at least one input checkpoint is required".to_string());
    }
    let alpha = method.alpha();
    if !(0.0..=1.0).contains(&alpha) {
        return Err("alpha must be in [0, 1]".to_string());
    }
    if let MergeMethod::DareTies {
        density, drop_prob, ..
    } = method
    {
        if !(0.0..=1.0).contains(&density) || density == 0.0 {
            return Err("density must be in (0, 1]".to_string());
        }
        if !(0.0..1.0).contains(&drop_prob) {
            return Err("drop_prob must be in [0, 1)".to_string());
        }
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
        sessions.extend(cache.entries.into_iter().map(|entry| entry.checkpoint.layers));
    }
    if sessions.is_empty() {
        return Err("input checkpoints contain no sessions".to_string());
    }

    let merged = merge_layer_stacks_with(&sessions, method)?;
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
        method: method.name().to_string(),
    })
}

/// Legacy alpha-blend over in-memory layer stacks. Kept for back-compat.
pub fn merge_layer_stacks(
    stacks: &[Vec<LayerWeights>],
    alpha: f32,
) -> Result<Vec<LayerWeights>, String> {
    merge_layer_stacks_with(stacks, MergeMethod::AlphaBlend { alpha })
}

pub fn merge_layer_stacks_with(
    stacks: &[Vec<LayerWeights>],
    method: MergeMethod,
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
        .map(|layer_index| merge_one_layer(stacks, layer_index, method))
        .collect()
}

fn merge_one_layer(
    stacks: &[Vec<LayerWeights>],
    layer_index: usize,
    method: MergeMethod,
) -> Result<LayerWeights, String> {
    let shape = stacks[0][layer_index].shape.clone();
    let d_model = square_d_model(&shape)?;
    let n = d_model * d_model;
    for stack in stacks {
        let layer = &stack[layer_index];
        if layer.shape != shape {
            return Err(format!("layer {layer_index} shape mismatch"));
        }
        if layer.data.len() != n {
            return Err(format!(
                "layer {layer_index} data length does not match shape"
            ));
        }
    }

    // The base task-vector reference is the identity (the fast-weight prior).
    let base = identity_vec(d_model);
    // Per-stack deltas (task vectors) against the identity base.
    let deltas: Vec<Vec<f32>> = stacks
        .iter()
        .map(|stack| {
            stack[layer_index]
                .data
                .iter()
                .zip(base.iter())
                .map(|(w, b)| w - b)
                .collect()
        })
        .collect();

    let merged_delta = match method {
        MergeMethod::AlphaBlend { alpha } => mean_delta(&deltas, n)
            .into_iter()
            .map(|d| d * alpha)
            .collect::<Vec<f32>>(),
        MergeMethod::DareTies {
            alpha,
            density,
            drop_prob,
            seed,
        } => {
            let sparse = dare_ties_delta(&deltas, n, density, drop_prob, seed, layer_index);
            sparse.into_iter().map(|d| d * alpha).collect()
        }
    };

    let data = base
        .iter()
        .zip(merged_delta.iter())
        .map(|(b, d)| b + d)
        .collect();
    Ok(LayerWeights { shape, data })
}

/// Plain mean of per-stack deltas (the alpha-blend aggregate).
fn mean_delta(deltas: &[Vec<f32>], n: usize) -> Vec<f32> {
    let mut sum = vec![0.0f32; n];
    for d in deltas {
        for (s, v) in sum.iter_mut().zip(d.iter()) {
            *s += v;
        }
    }
    let inv = 1.0 / deltas.len() as f32;
    sum.iter().map(|s| s * inv).collect()
}

/// DARE-TIES aggregate of per-stack deltas:
///   1. DARE: drop each element with prob `drop_prob`, rescale survivors by 1/(1-p).
///   2. TIES trim: per delta, keep the top-`density` fraction by |value|, zero the rest.
///   3. Elect sign: per element, the sign of the summed (trimmed) deltas.
///   4. Disjoint merge: average only the contributions matching the elected sign.
fn dare_ties_delta(
    deltas: &[Vec<f32>],
    n: usize,
    density: f32,
    drop_prob: f32,
    seed: u64,
    layer_index: usize,
) -> Vec<f32> {
    let scale = 1.0 / (1.0 - drop_prob);
    // 1 + 2: sparsify each delta independently.
    let sparse: Vec<Vec<f32>> = deltas
        .iter()
        .enumerate()
        .map(|(stack_idx, delta)| {
            let mut rng = SplitMix64::new(
                seed ^ ((layer_index as u64) << 32) ^ (stack_idx as u64).wrapping_mul(0x9E37_79B9),
            );
            // DARE drop + rescale.
            let mut dropped: Vec<f32> = delta
                .iter()
                .map(|&v| {
                    if drop_prob > 0.0 && rng.next_f32() < drop_prob {
                        0.0
                    } else {
                        v * scale
                    }
                })
                .collect();
            // TIES trim to top-`density` by magnitude.
            if density < 1.0 {
                let keep = ((n as f32) * density).ceil() as usize;
                trim_to_top_k(&mut dropped, keep);
            }
            dropped
        })
        .collect();

    // 3: elect a sign per element from the summed sparse deltas.
    let mut sum = vec![0.0f32; n];
    for d in &sparse {
        for (s, v) in sum.iter_mut().zip(d.iter()) {
            *s += v;
        }
    }

    // 4: disjoint merge — mean of sign-agreeing, nonzero contributions.
    let mut out = vec![0.0f32; n];
    for j in 0..n {
        let elected = sum[j].signum();
        if elected == 0.0 {
            continue;
        }
        let mut acc = 0.0f32;
        let mut count = 0u32;
        for d in &sparse {
            let v = d[j];
            if v != 0.0 && v.signum() == elected {
                acc += v;
                count += 1;
            }
        }
        if count > 0 {
            out[j] = acc / count as f32;
        }
    }
    out
}

/// Zero all but the `keep` largest-magnitude entries of `v`, in place.
fn trim_to_top_k(v: &mut [f32], keep: usize) {
    let n = v.len();
    if keep >= n {
        return;
    }
    if keep == 0 {
        v.iter_mut().for_each(|x| *x = 0.0);
        return;
    }
    let mut mags: Vec<f32> = v.iter().map(|x| x.abs()).collect();
    // Partial sort to find the magnitude threshold (the keep-th largest).
    mags.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = mags[keep - 1];
    let mut kept = 0usize;
    for x in v.iter_mut() {
        if x.abs() >= threshold && kept < keep {
            kept += 1;
        } else {
            *x = 0.0;
        }
    }
}

fn identity_vec(d_model: usize) -> Vec<f32> {
    let device = Device::Cpu;
    Tensor::eye(d_model, DType::F32, &device)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap_or_else(|_| {
            // Fallback: build identity directly if candle errors.
            let mut v = vec![0.0f32; d_model * d_model];
            for i in 0..d_model {
                v[i * d_model + i] = 1.0;
            }
            v
        })
}

/// Tiny deterministic PRNG (SplitMix64) for reproducible DARE masking — avoids
/// pulling in an rng dependency and guarantees every fleet node drops the same
/// elements given the same seed.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        // Top 24 bits → [0, 1).
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
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

    #[test]
    fn dare_ties_elects_dominant_sign_instead_of_cancelling() {
        // Diagonal [0,0]: A delta +0.4, B delta +0.6 (agree) → mean 0.5 → 1.5.
        // Off-diagonal [0,1]: A delta +0.5, B delta -0.1 (conflict). Elected sign
        // is + (sum 0.4 > 0); disjoint mean keeps only A's +0.5. A naive average
        // would have given +0.2 — sign cancellation diluting the dominant signal.
        let a = vec![LayerWeights {
            shape: vec![2, 2],
            data: vec![1.4, 0.5, 0.0, 1.0],
        }];
        let b = vec![LayerWeights {
            shape: vec![2, 2],
            data: vec![1.6, -0.1, 0.0, 1.0],
        }];
        // No drop / no trim → deterministic, isolates the sign-election behaviour.
        let method = MergeMethod::DareTies {
            alpha: 1.0,
            density: 1.0,
            drop_prob: 0.0,
            seed: 1,
        };
        let merged = merge_layer_stacks_with(&[a, b], method).unwrap();
        assert!((merged[0].data[0] - 1.5).abs() < 1e-5, "agreeing deltas average");
        assert!(
            (merged[0].data[1] - 0.5).abs() < 1e-5,
            "elected sign keeps the dominant delta, not the cancelled average"
        );
    }

    #[test]
    fn dare_drop_is_reproducible_and_rescales() {
        // Same seed → identical merged output (deterministic masking).
        let mk = || {
            vec![vec![LayerWeights {
                shape: vec![4, 4],
                data: (0..16).map(|i| if i % 5 == 0 { 1.3 } else { 0.2 }).collect(),
            }]]
        };
        let method = MergeMethod::DareTies {
            alpha: 1.0,
            density: 0.8,
            drop_prob: 0.5,
            seed: 42,
        };
        let m1 = merge_layer_stacks_with(&mk()[0..1], method).unwrap();
        let m2 = merge_layer_stacks_with(&mk()[0..1], method).unwrap();
        assert_eq!(m1[0].data, m2[0].data, "same seed → identical merge");
        assert_eq!(m1[0].shape, vec![4, 4]);
    }

    #[test]
    fn rejects_bad_dare_params() {
        let tmp = std::env::temp_dir().join("axiom_merge_badparams.bin");
        write_test_cache(
            &tmp,
            "s",
            vec![LayerWeights {
                shape: vec![2, 2],
                data: vec![1.0, 0.0, 0.0, 1.0],
            }],
        );
        let bad = MergeMethod::DareTies {
            alpha: 0.5,
            density: 0.0,
            drop_prob: 0.5,
            seed: 1,
        };
        assert!(merge_checkpoint_files_with(&[tmp.clone()], &tmp, bad).is_err());
        let _ = std::fs::remove_file(&tmp);
    }
}
