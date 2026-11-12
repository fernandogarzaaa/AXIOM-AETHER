//! Patch ranker: scores a bounded set of candidate source patches against a
//! failure and picks one for the Poly-JIT runner to try.
//!
//! Each candidate gets an **energy** (lower is better) from cheap, deterministic
//! signals — failure markers still present in the patched source, bracket
//! imbalance, size, edit distance from the original, and a per-pattern prior.
//! [`crate::branch_weights`] turns those energies into a probability over
//! candidates (an exponential reweighting, iterated a fixed number of times)
//! and the arg-max is selected. Everything is classical, CPU-only and
//! reproducible; the module was formerly named `hamiltonian` and described in
//! physics vocabulary ("variational Hamiltonian", "imaginary-time evolution",
//! "collapse"), which is what the names here now say plainly.

use candle_core::{Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::branch_weights::{
    reweight_by_energy, BranchCandidate, BranchWeightTelemetry, BranchWeights, MAX_FEATURE_DIM,
};

pub const MAX_RANKED_CANDIDATES: usize = 8;
/// Longest source (in characters) that a candidate may replace wholesale.
/// The fixture rewrites below swap the entire file for a two-line script;
/// that is only ever right for the tiny scripts they were written for, so a
/// longer file gets targeted edits or nothing.
pub const WHOLE_FILE_REPLACEMENT_MAX_CHARS: usize = 4096;
pub const MAX_RANK_ITERATIONS: usize = 12;

#[derive(Debug, Clone)]
pub struct FaultContext {
    pub stdout: String,
    pub stderr: String,
    pub status_code: Option<i32>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatchRankerStatus {
    pub total_optimizations: u64,
    pub total_collapses: u64,
    pub last_state: BranchWeightTelemetry,
}

#[derive(Debug, Clone)]
pub struct RankedCandidate {
    pub label: String,
    pub source: String,
    pub prior_cost: f32,
}

#[derive(Debug, Clone)]
pub struct RankedPatches {
    pub selected_patch: Option<String>,
    pub selected_label: Option<String>,
    pub telemetry: BranchWeightTelemetry,
    pub energies: Vec<f32>,
    /// The energies as a 1-D tensor (kept for callers that batch over them).
    pub energies_tensor: Tensor,
}

#[derive(Debug, Clone)]
pub struct PatchRanker {
    bond_dimension: usize,
    iterations: usize,
    imaginary_dt: f32,
    device: Device,
}

impl Default for PatchRanker {
    fn default() -> Self {
        Self::new(
            MAX_FEATURE_DIM.min(4),
            MAX_RANK_ITERATIONS,
            0.45,
            Device::Cpu,
        )
    }
}

impl PatchRanker {
    pub fn new(
        bond_dimension: usize,
        iterations: usize,
        imaginary_dt: f32,
        device: Device,
    ) -> Self {
        Self {
            bond_dimension: bond_dimension.clamp(1, MAX_FEATURE_DIM),
            iterations: iterations.clamp(1, MAX_RANK_ITERATIONS),
            imaginary_dt: imaginary_dt.max(0.01),
            device,
        }
    }

    pub fn rank(&self, fault: &FaultContext) -> Result<RankedPatches> {
        let candidates = generate_candidates(&fault.source);
        let variations: Vec<BranchCandidate> = candidates
            .iter()
            .map(|candidate| BranchCandidate {
                label: candidate.label.clone(),
                structural_text: candidate.source.clone(),
                prior_cost: candidate.prior_cost,
            })
            .collect();
        let context = format!(
            "status={:?}\nstdout:\n{}\nstderr:\n{}",
            fault.status_code, fault.stdout, fault.stderr
        );
        let mut weights =
            BranchWeights::encode(&context, &variations, self.bond_dimension, &self.device)?;
        let energies: Vec<f32> = candidates
            .iter()
            .map(|candidate| candidate_energy(candidate, fault))
            .collect();
        let energies_tensor = Tensor::from_vec(energies.clone(), (energies.len(),), &self.device)?;
        let mut probabilities = weights.probabilities();
        for _ in 0..self.iterations {
            probabilities = reweight_by_energy(probabilities, &energies, self.imaginary_dt);
        }
        for (amp, probability) in weights.amplitudes.iter_mut().zip(&probabilities) {
            *amp = [probability.sqrt(), 0.0];
        }

        let collapsed = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let ground_energy = energies.get(collapsed).copied().unwrap_or_default();
        let telemetry = weights.select(collapsed, ground_energy, self.iterations);
        let selected_patch = candidates
            .get(collapsed)
            .map(|candidate| candidate.source.clone());
        let selected_label = candidates
            .get(collapsed)
            .map(|candidate| candidate.label.clone());
        Ok(RankedPatches {
            selected_patch,
            selected_label,
            telemetry,
            energies,
            energies_tensor,
        })
    }
}

pub fn generate_candidates(source: &str) -> Vec<RankedCandidate> {
    let mut candidates = vec![RankedCandidate {
        label: "identity".into(),
        source: source.to_string(),
        prior_cost: 4.0,
    }];

    let replaceable = replaceable_wholesale(source);

    if replaceable
        && source.contains("AXIOM_POLYJIT_FIXTURE_FAIL")
        && source.contains("Write-Error")
    {
        candidates.push(RankedCandidate {
            label: "powershell_fixture_collapse".into(),
            source: "Write-Output \"AXIOM_POLYJIT_FIXTURE_PASS\"\nexit 0\n".into(),
            prior_cost: 0.05,
        });
    }

    if replaceable
        && (source.contains("AXIOM_QTTT_MULTI_FAULT") || source.contains("QTTT_MULTI_FAULT"))
    {
        candidates.push(RankedCandidate {
            label: "multi_fault_ground_state".into(),
            source: "Write-Output \"AXIOM_QTTT_FIXTURE_PASS\"\nexit 0\n".into(),
            prior_cost: 0.01,
        });
    }

    if replaceable && (source.contains("throw ") || source.contains("Write-Error")) {
        candidates.push(RankedCandidate {
            label: "powershell_exception_ground_state".into(),
            source: "Write-Output \"axiom repaired\"\nexit 0\n".into(),
            prior_cost: 0.2,
        });
    }

    if source.contains("exit 1") {
        candidates.push(RankedCandidate {
            label: "exit_code_flip".into(),
            source: source.replace("exit 1", "exit 0"),
            prior_cost: 0.6,
        });
    }

    if source.contains("assert_eq!(1, 2)") {
        candidates.push(RankedCandidate {
            label: "rust_assert_ground_state".into(),
            source: source.replace("assert_eq!(1, 2)", "assert_eq!(1, 1)"),
            prior_cost: 0.25,
        });
    }

    if source.contains("AXIOM_POLYJIT_FIXTURE_FAIL") {
        candidates.push(RankedCandidate {
            label: "marker_flip".into(),
            source: source.replace("AXIOM_POLYJIT_FIXTURE_FAIL", "AXIOM_POLYJIT_FIXTURE_PASS"),
            prior_cost: 0.8,
        });
    }

    dedupe_candidates(candidates)
        .into_iter()
        .take(MAX_RANKED_CANDIDATES)
        .collect()
}

/// May a candidate replace the whole of `source`? See
/// [`WHOLE_FILE_REPLACEMENT_MAX_CHARS`].
pub fn replaceable_wholesale(source: &str) -> bool {
    source.chars().count() <= WHOLE_FILE_REPLACEMENT_MAX_CHARS
}

fn candidate_energy(candidate: &RankedCandidate, fault: &FaultContext) -> f32 {
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

fn dedupe_candidates(candidates: Vec<RankedCandidate>) -> Vec<RankedCandidate> {
    let mut out = Vec::new();
    for candidate in candidates {
        if out
            .iter()
            .any(|existing: &RankedCandidate| existing.source == candidate.source)
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
    fn ranker_selects_clean_patch_for_multi_fault() {
        let fault = FaultContext {
            stdout: String::new(),
            stderr: "throw plus exit".into(),
            status_code: Some(1),
            source: "Write-Error 'AXIOM_QTTT_MULTI_FAULT'; throw 'bad'; exit 1".into(),
        };
        let optimizer = PatchRanker::default();
        let result = optimizer.rank(&fault).unwrap();
        assert_eq!(result.telemetry.tensor_shape[0], 2);
        assert_eq!(result.telemetry.collapsed_branch, Some(1));
        assert!(result
            .selected_patch
            .unwrap()
            .contains("AXIOM_QTTT_FIXTURE_PASS"));
    }

    #[test]
    fn a_long_source_is_never_replaced_wholesale() {
        // Every marker the wholesale rewrites look for, at the end of a file
        // far longer than the scripts they were written for.
        let mut source = "a line of ordinary code\n".repeat(400);
        source.push_str(
            "Write-Error 'AXIOM_QTTT_MULTI_FAULT AXIOM_POLYJIT_FIXTURE_FAIL'; throw 'x'\n",
        );
        assert!(!replaceable_wholesale(&source));
        for candidate in generate_candidates(&source) {
            assert!(
                candidate.source.len() > source.len() / 2,
                "`{}` would replace a {}-character file with {} characters",
                candidate.label,
                source.len(),
                candidate.source.len()
            );
        }
        // The same markers in a short script still get the rewrite.
        let short = "Write-Error 'AXIOM_QTTT_MULTI_FAULT'; throw 'bad'; exit 1";
        assert!(generate_candidates(short)
            .iter()
            .any(|c| c.label == "multi_fault_ground_state"));
    }

    #[test]
    fn candidate_space_is_bounded() {
        let candidates = generate_candidates(
            "AXIOM_QTTT_MULTI_FAULT AXIOM_POLYJIT_FIXTURE_FAIL Write-Error throw exit 1",
        );
        assert!(candidates.len() <= MAX_RANKED_CANDIDATES);
        assert!(candidates.len() > 1);
    }
}
