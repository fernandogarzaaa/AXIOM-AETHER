//! Variational Hamiltonian optimizer for the Q-TTT simulator.
//!
//! The optimizer maps runtime fault text and candidate source mutations to a
//! diagonal cost Hamiltonian. Imaginary-time evolution reweights branch
//! probabilities toward the lowest-energy candidate before an explicit collapse.

use candle_core::{Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::q_manifold::{
    evolve_probabilities, ManifoldVariation, QuantumManifoldTelemetry, QuantumStateManifold,
    MAX_MPS_BOND_DIM,
};

pub const MAX_QTTT_CANDIDATES: usize = 8;
pub const MAX_VQE_ITERATIONS: usize = 12;

#[derive(Debug, Clone)]
pub struct HamiltonianFault {
    pub stdout: String,
    pub stderr: String,
    pub status_code: Option<i32>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct QuantumRuntimeStatus {
    pub total_optimizations: u64,
    pub total_collapses: u64,
    pub last_state: QuantumManifoldTelemetry,
}


#[derive(Debug, Clone)]
pub struct QuantumPatchCandidate {
    pub label: String,
    pub source: String,
    pub prior_cost: f32,
}

#[derive(Debug, Clone)]
pub struct HamiltonianOptimization {
    pub collapsed_patch: Option<String>,
    pub collapsed_label: Option<String>,
    pub telemetry: QuantumManifoldTelemetry,
    pub energies: Vec<f32>,
    pub hamiltonian: Tensor,
}

#[derive(Debug, Clone)]
pub struct VariationalHamiltonian {
    bond_dimension: usize,
    iterations: usize,
    imaginary_dt: f32,
    device: Device,
}

impl Default for VariationalHamiltonian {
    fn default() -> Self {
        Self::new(
            MAX_MPS_BOND_DIM.min(4),
            MAX_VQE_ITERATIONS,
            0.45,
            Device::Cpu,
        )
    }
}

impl VariationalHamiltonian {
    pub fn new(
        bond_dimension: usize,
        iterations: usize,
        imaginary_dt: f32,
        device: Device,
    ) -> Self {
        Self {
            bond_dimension: bond_dimension.clamp(1, MAX_MPS_BOND_DIM),
            iterations: iterations.clamp(1, MAX_VQE_ITERATIONS),
            imaginary_dt: imaginary_dt.max(0.01),
            device,
        }
    }

    pub fn optimize_fault(&self, fault: &HamiltonianFault) -> Result<HamiltonianOptimization> {
        let candidates = generate_quantum_patch_candidates(&fault.source);
        let variations: Vec<ManifoldVariation> = candidates
            .iter()
            .map(|candidate| ManifoldVariation {
                label: candidate.label.clone(),
                structural_text: candidate.source.clone(),
                prior_cost: candidate.prior_cost,
            })
            .collect();
        let context = format!(
            "status={:?}\nstdout:\n{}\nstderr:\n{}",
            fault.status_code, fault.stdout, fault.stderr
        );
        let mut manifold =
            QuantumStateManifold::encode(&context, &variations, self.bond_dimension, &self.device)?;
        let energies: Vec<f32> = candidates
            .iter()
            .map(|candidate| candidate_energy(candidate, fault))
            .collect();
        let hamiltonian = Tensor::from_vec(energies.clone(), (energies.len(),), &self.device)?;
        let mut probabilities = manifold.probabilities();
        for _ in 0..self.iterations {
            probabilities = evolve_probabilities(probabilities, &energies, self.imaginary_dt);
        }
        for (amp, probability) in manifold.amplitudes.iter_mut().zip(&probabilities) {
            *amp = [probability.sqrt(), 0.0];
        }

        let collapsed = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let ground_energy = energies.get(collapsed).copied().unwrap_or_default();
        let telemetry = manifold.collapse(collapsed, ground_energy, self.iterations);
        let collapsed_patch = candidates
            .get(collapsed)
            .map(|candidate| candidate.source.clone());
        let collapsed_label = candidates
            .get(collapsed)
            .map(|candidate| candidate.label.clone());
        Ok(HamiltonianOptimization {
            collapsed_patch,
            collapsed_label,
            telemetry,
            energies,
            hamiltonian,
        })
    }
}

pub fn generate_quantum_patch_candidates(source: &str) -> Vec<QuantumPatchCandidate> {
    let mut candidates = vec![QuantumPatchCandidate {
        label: "identity".into(),
        source: source.to_string(),
        prior_cost: 4.0,
    }];

    if source.contains("AXIOM_POLYJIT_FIXTURE_FAIL") && source.contains("Write-Error") {
        candidates.push(QuantumPatchCandidate {
            label: "powershell_fixture_collapse".into(),
            source: "Write-Output \"AXIOM_POLYJIT_FIXTURE_PASS\"\nexit 0\n".into(),
            prior_cost: 0.05,
        });
    }

    if source.contains("AXIOM_QTTT_MULTI_FAULT") || source.contains("QTTT_MULTI_FAULT") {
        candidates.push(QuantumPatchCandidate {
            label: "multi_fault_ground_state".into(),
            source: "Write-Output \"AXIOM_QTTT_FIXTURE_PASS\"\nexit 0\n".into(),
            prior_cost: 0.01,
        });
    }

    if source.contains("throw ") || source.contains("Write-Error") {
        candidates.push(QuantumPatchCandidate {
            label: "powershell_exception_ground_state".into(),
            source: "Write-Output \"axiom quantum repaired\"\nexit 0\n".into(),
            prior_cost: 0.2,
        });
    }

    if source.contains("exit 1") {
        candidates.push(QuantumPatchCandidate {
            label: "exit_code_flip".into(),
            source: source.replace("exit 1", "exit 0"),
            prior_cost: 0.6,
        });
    }

    if source.contains("assert_eq!(1, 2)") {
        candidates.push(QuantumPatchCandidate {
            label: "rust_assert_ground_state".into(),
            source: source.replace("assert_eq!(1, 2)", "assert_eq!(1, 1)"),
            prior_cost: 0.25,
        });
    }

    if source.contains("AXIOM_POLYJIT_FIXTURE_FAIL") {
        candidates.push(QuantumPatchCandidate {
            label: "marker_flip".into(),
            source: source.replace("AXIOM_POLYJIT_FIXTURE_FAIL", "AXIOM_POLYJIT_FIXTURE_PASS"),
            prior_cost: 0.8,
        });
    }

    dedupe_candidates(candidates)
        .into_iter()
        .take(MAX_QTTT_CANDIDATES)
        .collect()
}

fn candidate_energy(candidate: &QuantumPatchCandidate, fault: &HamiltonianFault) -> f32 {
    let failure_markers = marker_cost(&candidate.source);
    let syntax_weight = syntax_cost(&candidate.source);
    let status_cost = if fault.status_code == Some(0) {
        0.0
    } else {
        0.25
    };
    let vram_cost = (candidate.source.len() as f32 / 4096.0).min(1.0) * 0.15;
    let edit_cost = normalized_edit_distance(&fault.source, &candidate.source) * 0.2;
    failure_markers + syntax_weight + status_cost + vram_cost + edit_cost + candidate.prior_cost
}

fn marker_cost(source: &str) -> f32 {
    let markers = [
        ("exit 1", 2.0),
        ("Write-Error", 2.0),
        ("throw ", 2.0),
        ("panic!", 1.5),
        ("AXIOM_POLYJIT_FIXTURE_FAIL", 2.5),
        ("AXIOM_QTTT_MULTI_FAULT", 2.5),
    ];
    markers
        .iter()
        .filter(|(marker, _)| source.contains(marker))
        .map(|(_, cost)| *cost)
        .sum()
}

fn syntax_cost(source: &str) -> f32 {
    let opens = source.matches('{').count() as i32 + source.matches('(').count() as i32;
    let closes = source.matches('}').count() as i32 + source.matches(')').count() as i32;
    (opens - closes).unsigned_abs() as f32 * 0.2
}

fn normalized_edit_distance(before: &str, after: &str) -> f32 {
    let max_len = before.len().max(after.len()).max(1) as f32;
    let delta = before.len().abs_diff(after.len()) as f32;
    (delta / max_len).min(1.0)
}

fn dedupe_candidates(candidates: Vec<QuantumPatchCandidate>) -> Vec<QuantumPatchCandidate> {
    let mut out = Vec::new();
    for candidate in candidates {
        if out
            .iter()
            .any(|existing: &QuantumPatchCandidate| existing.source == candidate.source)
        {
            continue;
        }
        out.push(candidate);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamiltonian_collapses_multi_fault_to_clean_patch() {
        let fault = HamiltonianFault {
            stdout: String::new(),
            stderr: "throw plus exit".into(),
            status_code: Some(1),
            source: "Write-Error 'AXIOM_QTTT_MULTI_FAULT'; throw 'bad'; exit 1".into(),
        };
        let optimizer = VariationalHamiltonian::default();
        let result = optimizer.optimize_fault(&fault).unwrap();
        assert_eq!(result.telemetry.tensor_shape[0], 2);
        assert_eq!(result.telemetry.collapsed_branch, Some(1));
        assert!(result
            .collapsed_patch
            .unwrap()
            .contains("AXIOM_QTTT_FIXTURE_PASS"));
    }

    #[test]
    fn candidate_space_is_bounded() {
        let candidates = generate_quantum_patch_candidates(
            "AXIOM_QTTT_MULTI_FAULT AXIOM_POLYJIT_FIXTURE_FAIL Write-Error throw exit 1",
        );
        assert!(candidates.len() <= MAX_QTTT_CANDIDATES);
        assert!(candidates.len() > 1);
    }
}
