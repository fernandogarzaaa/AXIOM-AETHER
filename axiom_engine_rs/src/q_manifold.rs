//! Bounded quantum-state manifold simulator for Q-TTT.
//!
//! This is a classical, deterministic tensor-network simulator. Complex
//! amplitudes are represented as paired real-valued dimensions with shape
//! `[2, branches, bond_dim]` to stay compatible with Candle CPU/CUDA tensors.

use candle_core::{Device, Result, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const COMPLEX_COMPONENTS: usize = 2;
pub const MAX_MPS_BRANCHES: usize = 8;
pub const MAX_MPS_BOND_DIM: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldVariation {
    pub label: String,
    pub structural_text: String,
    #[serde(default)]
    pub prior_cost: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumManifoldTelemetry {
    pub active: bool,
    pub branches: usize,
    pub bond_dimension: usize,
    pub tensor_shape: Vec<usize>,
    pub entropy_bits: f32,
    pub collapse_probabilities: Vec<f32>,
    pub collapsed_branch: Option<usize>,
    pub ground_energy: Option<f32>,
    pub iterations: usize,
}

impl Default for QuantumManifoldTelemetry {
    fn default() -> Self {
        Self {
            active: false,
            branches: 0,
            bond_dimension: 0,
            tensor_shape: vec![COMPLEX_COMPONENTS, 0, 0],
            entropy_bits: 0.0,
            collapse_probabilities: Vec::new(),
            collapsed_branch: None,
            ground_energy: None,
            iterations: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantumStateManifold {
    pub branches: Vec<String>,
    pub bond_dimension: usize,
    pub amplitudes: Vec<[f32; COMPLEX_COMPONENTS]>,
    pub feature_norms: Vec<f32>,
    pub tensor: Tensor,
}

impl QuantumStateManifold {
    pub fn encode(
        context: &str,
        variations: &[ManifoldVariation],
        requested_bond_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let branch_count = variations.len().clamp(1, MAX_MPS_BRANCHES);
        let bond_dimension = requested_bond_dim.clamp(1, MAX_MPS_BOND_DIM);
        let mut raw = Vec::with_capacity(branch_count);
        let mut feature_norms = Vec::with_capacity(branch_count);
        let mut branches = Vec::with_capacity(branch_count);

        for variation in variations.iter().take(branch_count) {
            let features = hashed_features(context, variation, bond_dimension);
            let norm = features.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            feature_norms.push(norm);
            branches.push(variation.label.clone());
            let phase_seed = stable_unit(&variation.structural_text, 17);
            raw.push([
                (1.0 / (1.0 + variation.prior_cost.max(0.0))) * (0.75 + phase_seed * 0.25),
                phase_seed * 0.125,
            ]);
        }

        normalize_amplitudes(&mut raw);
        let mut packed = vec![0.0_f32; COMPLEX_COMPONENTS * branch_count * bond_dimension];
        for (branch, amp) in raw.iter().enumerate() {
            let features = hashed_features(context, &variations[branch], bond_dimension);
            let norm = features.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            for bond in 0..bond_dimension {
                let feature = features[bond] / norm;
                packed[index(0, branch, bond, branch_count, bond_dimension)] = amp[0] * feature;
                packed[index(1, branch, bond, branch_count, bond_dimension)] = amp[1] * feature;
            }
        }
        let tensor = Tensor::from_vec(
            packed,
            (COMPLEX_COMPONENTS, branch_count, bond_dimension),
            device,
        )?;
        Ok(Self {
            branches,
            bond_dimension,
            amplitudes: raw,
            feature_norms,
            tensor,
        })
    }

    pub fn probabilities(&self) -> Vec<f32> {
        normalize_probabilities(
            self.amplitudes
                .iter()
                .map(|amp| amp[0] * amp[0] + amp[1] * amp[1])
                .collect(),
        )
    }

    pub fn entropy_bits(&self) -> f32 {
        entropy_bits(&self.probabilities())
    }

    pub fn telemetry(&self) -> QuantumManifoldTelemetry {
        QuantumManifoldTelemetry {
            active: !self.branches.is_empty(),
            branches: self.branches.len(),
            bond_dimension: self.bond_dimension,
            tensor_shape: self.tensor.dims().to_vec(),
            entropy_bits: self.entropy_bits(),
            collapse_probabilities: self.probabilities(),
            collapsed_branch: None,
            ground_energy: None,
            iterations: 0,
        }
    }

    pub fn collapse(
        &mut self,
        branch: usize,
        ground_energy: f32,
        iterations: usize,
    ) -> QuantumManifoldTelemetry {
        let branch = branch.min(self.amplitudes.len().saturating_sub(1));
        for (idx, amp) in self.amplitudes.iter_mut().enumerate() {
            if idx == branch {
                *amp = [1.0, 0.0];
            } else {
                *amp = [0.0, 0.0];
            }
        }
        QuantumManifoldTelemetry {
            active: true,
            branches: self.branches.len(),
            bond_dimension: self.bond_dimension,
            tensor_shape: self.tensor.dims().to_vec(),
            entropy_bits: self.entropy_bits(),
            collapse_probabilities: self.probabilities(),
            collapsed_branch: Some(branch),
            ground_energy: Some(ground_energy),
            iterations,
        }
    }
}

pub fn evolve_probabilities(mut probabilities: Vec<f32>, energies: &[f32], dt: f32) -> Vec<f32> {
    if probabilities.len() != energies.len() || probabilities.is_empty() {
        return Vec::new();
    }
    for (prob, energy) in probabilities.iter_mut().zip(energies) {
        *prob *= (-dt * energy.max(0.0)).exp();
    }
    normalize_probabilities(probabilities)
}

pub fn entropy_bits(probabilities: &[f32]) -> f32 {
    probabilities
        .iter()
        .copied()
        .filter(|p| *p > 1e-9)
        .map(|p| -p * p.log2())
        .sum()
}

fn normalize_amplitudes(amplitudes: &mut [[f32; COMPLEX_COMPONENTS]]) {
    let norm = amplitudes
        .iter()
        .map(|amp| amp[0] * amp[0] + amp[1] * amp[1])
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    for amp in amplitudes {
        amp[0] /= norm;
        amp[1] /= norm;
    }
}

pub fn normalize_probabilities(mut probabilities: Vec<f32>) -> Vec<f32> {
    let total = probabilities.iter().sum::<f32>();
    if total <= 1e-9 {
        let uniform = 1.0 / probabilities.len().max(1) as f32;
        probabilities.fill(uniform);
        return probabilities;
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    probabilities
}

fn hashed_features(context: &str, variation: &ManifoldVariation, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|idx| {
            let value = stable_unit(
                &format!(
                    "{}\n{}\n{}\n{}",
                    context, variation.label, variation.structural_text, idx
                ),
                idx as u64,
            );
            value * 2.0 - 1.0
        })
        .collect()
}

fn stable_unit(text: &str, salt: u64) -> f32 {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.update(salt.to_le_bytes());
    let digest = hasher.finalize();
    let raw = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    raw as f32 / u32::MAX as f32
}

fn index(component: usize, branch: usize, bond: usize, branches: usize, bond_dim: usize) -> usize {
    component * branches * bond_dim + branch * bond_dim + bond
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifold_encodes_paired_real_imag_tensor() {
        let variations = vec![
            ManifoldVariation {
                label: "a".into(),
                structural_text: "exit 1".into(),
                prior_cost: 1.0,
            },
            ManifoldVariation {
                label: "b".into(),
                structural_text: "exit 0".into(),
                prior_cost: 0.1,
            },
        ];
        let manifold = QuantumStateManifold::encode("fault", &variations, 4, &Device::Cpu).unwrap();
        assert_eq!(manifold.tensor.dims(), &[2, 2, 4]);
        assert!((manifold.probabilities().iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(manifold.entropy_bits() > 0.0);
    }

    #[test]
    fn probability_evolution_prefers_lower_energy() {
        let evolved = evolve_probabilities(vec![0.5, 0.5], &[4.0, 0.2], 0.4);
        assert!(evolved[1] > evolved[0]);
    }
}
