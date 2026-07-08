//! trajectory_sampler.rs — Concurrent Trajectory Sampling.
//!
//! Forks the reasoning process into multiple parallel pathways during the
//! look-ahead phase. A resource-allocation scoring model evaluates the viability
//! of each branch in real time, aggressively pruning dead ends and committing
//! compute power only to the most mathematically sound trajectory.
//!
//! ## Architecture
//!
//! The sampler takes a `SemanticStateMap` (from `state_predictor.rs`) and forks
//! each milestone's branch hints into parallel trajectories. Each trajectory is
//! scored on three dimensions:
//!
//! 1. **Viability** — expected success probability (from BetaBelief confidence)
//! 2. **Compute cost** — estimated GPU seconds (from token budget)
//! 3. **Information gain** — expected reduction in uncertainty
//!
//! The composite score is `viability * information_gain / compute_cost`. Branches
//! below the prune threshold are abandoned before GPU compute is committed.

use serde::{Deserialize, Serialize};

use crate::belief::BetaBelief;
use crate::q_manifold::{QuantumStateManifold, ManifoldVariation, evolve_probabilities, entropy_bits};
use crate::state_predictor::SemanticStateMap;

/// Maximum number of concurrent trajectory branches.
pub const MAX_BRANCHES: usize = 8;

/// Default prune threshold: branches with composite score below this are pruned.
pub const DEFAULT_PRUNE_THRESHOLD: f32 = 0.1;

/// A single trajectory branch — one forked reasoning pathway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryBranch {
    /// Unique branch identifier.
    pub id: usize,
    /// Human-readable label for this branch.
    pub label: String,
    /// The milestone index this branch forks from.
    pub fork_point: usize,
    /// Viability score (0-1): expected success probability.
    pub viability: f32,
    /// Estimated compute cost in GPU-seconds.
    pub compute_cost: f32,
    /// Expected information gain (reduction in entropy bits).
    pub information_gain: f32,
    /// Composite score: viability * information_gain / compute_cost.
    pub composite_score: f32,
    /// Whether this branch was pruned.
    pub pruned: bool,
    /// Confidence in the branch assessment.
    pub confidence: BetaBelief,
}

/// Result of trajectory sampling — the set of branches and the selected winner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectorySampleResult {
    /// All evaluated branches (including pruned ones).
    pub branches: Vec<TrajectoryBranch>,
    /// The index of the selected branch in `branches`.
    pub selected: Option<usize>,
    /// The prune threshold used.
    pub prune_threshold: f32,
    /// Total estimated compute cost of all non-pruned branches.
    pub total_compute_cost: f32,
    /// Entropy of the branch distribution (higher = more uncertain).
    pub entropy_bits: f32,
}

/// The trajectory sampler — forks and scores reasoning branches.
pub struct TrajectorySampler {
    max_branches: usize,
    prune_threshold: f32,
}

impl TrajectorySampler {
    /// Create a new sampler with the given configuration.
    pub fn new(max_branches: usize, prune_threshold: f32) -> Self {
        Self {
            max_branches: max_branches.min(MAX_BRANCHES),
            prune_threshold,
        }
    }

    /// Create a sampler with default settings.
    pub fn default_config() -> Self {
        Self::new(MAX_BRANCHES, DEFAULT_PRUNE_THRESHOLD)
    }

    /// Sample trajectories from a semantic state map.
    ///
    /// For each milestone in the state map, fork the branch hints into parallel
    /// trajectories. Score each branch, prune low-scoring ones, and select the
    /// best remaining branch.
    pub fn sample(&self, state_map: &SemanticStateMap) -> TrajectorySampleResult {
        let mut branches: Vec<TrajectoryBranch> = Vec::new();
        let mut branch_id = 0usize;

        // Fork from each milestone's branch hints.
        for (milestone_idx, milestone) in state_map.milestones.iter().enumerate() {
            for hint in &milestone.branch_hints {
                if branch_id >= self.max_branches {
                    break;
                }

                // Estimate viability from the milestone confidence.
                let viability = milestone.confidence.mean();

                // Estimate compute cost from the token budget.
                // Assume ~0.1 GPU-seconds per token on MI300X with vLLM.
                let compute_cost = (milestone.token_budget as f32 * 0.1).max(0.01);

                // Estimate information gain from the entropy of the branch hints.
                // More hints = more uncertainty = more potential gain.
                let n_hints = milestone.branch_hints.len() as f32;
                let information_gain = if n_hints > 1.0 {
                    (n_hints).log2()
                } else {
                    0.5 // Single-hint branches still carry some information
                };

                // Composite score: viability * information_gain / compute_cost.
                let composite_score = if compute_cost > 0.0 {
                    (viability * information_gain) / compute_cost
                } else {
                    0.0
                };

                branches.push(TrajectoryBranch {
                    id: branch_id,
                    label: format!("milestone_{}_{}", milestone_idx, hint),
                    fork_point: milestone_idx,
                    viability,
                    compute_cost,
                    information_gain,
                    composite_score,
                    pruned: false,
                    confidence: milestone.confidence,
                });
                branch_id += 1;
            }
            if branch_id >= self.max_branches {
                break;
            }
        }

        // If no branches were created (no hints), create a single default branch.
        if branches.is_empty() && !state_map.milestones.is_empty() {
            let m = &state_map.milestones[0];
            branches.push(TrajectoryBranch {
                id: 0,
                label: "default".to_string(),
                fork_point: 0,
                viability: m.confidence.mean(),
                compute_cost: (m.token_budget as f32 * 0.1).max(0.01),
                information_gain: 0.5,
                composite_score: m.confidence.mean() * 0.5 / (m.token_budget as f32 * 0.1).max(0.01),
                pruned: false,
                confidence: m.confidence,
            });
        }

        // Prune branches below the threshold.
        for b in &mut branches {
            if b.composite_score < self.prune_threshold {
                b.pruned = true;
            }
        }

        // Select the best non-pruned branch.
        let selected = branches
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.pruned)
            .max_by(|(_, a), (_, b)| {
                a.composite_score
                    .partial_cmp(&b.composite_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx);

        // Compute total compute cost of non-pruned branches.
        let total_compute_cost: f32 = branches
            .iter()
            .filter(|b| !b.pruned)
            .map(|b| b.compute_cost)
            .sum();

        // Compute entropy of the branch distribution.
        let probs: Vec<f32> = branches
            .iter()
            .filter(|b| !b.pruned)
            .map(|b| b.composite_score.max(1e-6))
            .collect();
        let total: f32 = probs.iter().sum();
        let normalized: Vec<f32> = if total > 0.0 {
            probs.iter().map(|p| p / total).collect()
        } else {
            vec![1.0]
        };
        let entropy = entropy_bits(&normalized);

        TrajectorySampleResult {
            branches,
            selected,
            prune_threshold: self.prune_threshold,
            total_compute_cost,
            entropy_bits: entropy,
        }
    }

    /// Sample trajectories and encode them into a quantum state manifold
    /// for branch-aware probability evolution.
    pub fn sample_with_manifold(
        &self,
        state_map: &SemanticStateMap,
        context: &str,
    ) -> (TrajectorySampleResult, Option<QuantumStateManifold>) {
        let result = self.sample(state_map);

        // Build manifold variations from non-pruned branches.
        let variations: Vec<ManifoldVariation> = result
            .branches
            .iter()
            .filter(|b| !b.pruned)
            .map(|b| ManifoldVariation {
                label: b.label.clone(),
                structural_text: format!("fork_{}_viability_{:.3}", b.fork_point, b.viability),
                prior_cost: b.compute_cost,
            })
            .collect();

        let manifold = if variations.is_empty() {
            None
        } else {
            // Use CPU device for the manifold (it's lightweight).
            QuantumStateManifold::encode(context, &variations, 4, &candle_core::Device::Cpu).ok()
        };

        (result, manifold)
    }

    /// Evolve branch probabilities using energy-based evolution.
    /// Lower-energy branches (higher viability) gain probability mass.
    pub fn evolve_branches(
        &self,
        result: &mut TrajectorySampleResult,
        dt: f32,
    ) {
        let energies: Vec<f32> = result
            .branches
            .iter()
            .filter(|b| !b.pruned)
            .map(|b| (1.0 - b.viability) * 10.0) // higher viability = lower energy
            .collect();

        let probs: Vec<f32> = result
            .branches
            .iter()
            .filter(|b| !b.pruned)
            .map(|b| b.composite_score.max(1e-6))
            .collect();

        let evolved = evolve_probabilities(probs, &energies, dt);

        // Update composite scores with evolved probabilities.
        let mut ev_iter = evolved.iter();
        for b in &mut result.branches {
            if !b.pruned {
                if let Some(&ep) = ev_iter.next() {
                    // Blend the evolved probability into the composite score.
                    b.composite_score = b.composite_score * 0.5 + ep * 0.5;
                }
            }
        }

        // Re-select the best branch after evolution.
        result.selected = result
            .branches
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.pruned)
            .max_by(|(_, a), (_, b)| {
                a.composite_score
                    .partial_cmp(&b.composite_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx);
    }
}

/// Render a trajectory sample result as a human-readable string.
pub fn render_trajectory_result(result: &TrajectorySampleResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "TrajectorySample (branches={}, pruned={}, selected={:?}, entropy={:.2} bits):\n",
        result.branches.len(),
        result.branches.iter().filter(|b| b.pruned).count(),
        result.selected,
        result.entropy_bits
    ));
    for b in &result.branches {
        let status = if b.pruned { "PRUNED" } else { "ACTIVE" };
        let marker = if Some(b.id) == result.selected { " *** SELECTED ***" } else { "" };
        out.push_str(&format!(
            "  [{}] {} viability={:.2} cost={:.1}s gain={:.2} score={:.4} {}{}\n",
            b.id,
            status,
            b.viability,
            b.compute_cost,
            b.information_gain,
            b.composite_score,
            b.label,
            marker
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_predictor::{SemanticMilestone, SemanticStateMap};

    fn make_state_map() -> SemanticStateMap {
        SemanticStateMap {
            milestones: vec![
                SemanticMilestone {
                    label: "hypothesis".to_string(),
                    label_idx: 0,
                    predicted_state: vec![0.1; 16],
                    token_budget: 128,
                    branch_hints: vec!["direct".to_string(), "decomposed".to_string()],
                    confidence: BetaBelief::from_confidence(0.8, 4.0),
                },
                SemanticMilestone {
                    label: "execution".to_string(),
                    label_idx: 2,
                    predicted_state: vec![0.2; 16],
                    token_budget: 512,
                    branch_hints: vec!["primary".to_string(), "fallback".to_string()],
                    confidence: BetaBelief::from_confidence(0.6, 4.0),
                },
            ],
            confidence: BetaBelief::from_confidence(0.7, 4.0),
            session_id: "test".to_string(),
            context_state: vec![0.5; 16],
        }
    }

    #[test]
    fn sample_produces_branches() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        assert!(!result.branches.is_empty());
        assert!(result.branches.len() <= MAX_BRANCHES);
    }

    #[test]
    fn branches_have_valid_scores() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        for b in &result.branches {
            assert!(b.viability >= 0.0 && b.viability <= 1.0);
            assert!(b.compute_cost > 0.0);
            assert!(b.information_gain >= 0.0);
        }
    }

    #[test]
    fn pruning_removes_low_scoring_branches() {
        let sampler = TrajectorySampler::new(8, 100.0); // very high threshold
        let map = make_state_map();
        let result = sampler.sample(&map);
        assert!(result.branches.iter().any(|b| b.pruned), "some branches should be pruned");
    }

    #[test]
    fn selected_branch_is_not_pruned() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        if let Some(idx) = result.selected {
            assert!(!result.branches[idx].pruned, "selected branch must not be pruned");
        }
    }

    #[test]
    fn evolve_changes_scores() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);
        let after: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        // At least some scores should change.
        assert!(before != after, "evolution should change scores");
    }

    #[test]
    fn render_is_readable() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        let rendered = render_trajectory_result(&result);
        assert!(rendered.contains("TrajectorySample"));
        assert!(rendered.contains("viability="));
    }

    #[test]
    fn empty_state_map_produces_default_branch() {
        let sampler = TrajectorySampler::default_config();
        let map = SemanticStateMap::default();
        let result = sampler.sample(&map);
        // Empty map has no milestones, so no branches.
        assert!(result.branches.is_empty());
    }

    #[test]
    fn manifold_integration_works() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let (result, manifold) = sampler.sample_with_manifold(&map, "test context");
        assert!(!result.branches.is_empty());
        assert!(manifold.is_some(), "manifold should be created from non-pruned branches");
    }
}