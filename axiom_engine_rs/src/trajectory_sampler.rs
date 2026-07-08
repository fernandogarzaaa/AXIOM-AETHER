//! trajectory_sampler.rs - Branch-and-bound trajectory sampling over the
//! cognitive state manifold.  The sampler produces parallel trajectories,
//! scores them with a composite of novelty + coherence + budget-fit, prunes
//! low-scoring branches, and evolves survivors with Gaussian perturbation.
//!
//! Designed to be called by the MCP `axiom_sample_trajectories` tool.

use crate::state_predictor::SemanticStateMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default number of branches to explore.
pub const DEFAULT_NUM_BRANCHES: usize = 8;

/// Default prune threshold: branches with composite score below this are
/// discarded after the first round.
pub const DEFAULT_PRUNE_THRESHOLD: f32 = 0.10;

/// Std-dev of the Gaussian perturbation applied during evolution.
pub const EVOLVE_SIGMA: f32 = 0.15;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single trajectory branch with a composite score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryBranch {
    /// Human-readable label for the branch (e.g. "hypothesis -> decomposition").
    pub label: String,
    /// Projected state vector at the branch point.
    pub state_vector: Vec<f32>,
    /// Composite score in [0, 1] - higher is better.
    pub composite_score: f32,
    /// Novelty component [0, 1].
    pub novelty: f32,
    /// Coherence component [0, 1].
    pub coherence: f32,
    /// Budget-fit component [0, 1].
    pub budget_fit: f32,
    /// Depth in the trajectory tree (0 = root).
    pub depth: usize,
}

/// Result of a trajectory sampling round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectorySampleResult {
    /// All surviving branches after pruning.
    pub branches: Vec<TrajectoryBranch>,
    /// The session id used for prediction.
    pub session_id: String,
    /// Total number of branches explored (before pruning).
    pub total_explored: usize,
    /// Number of branches pruned.
    pub pruned: usize,
}

/// A cognitive manifold: a graph of state transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveManifold {
    /// Nodes (state vectors).
    pub nodes: Vec<Vec<f32>>,
    /// Edges as (from_idx, to_idx, weight).
    pub edges: Vec<(usize, usize, f32)>,
    /// Labels for each node.
    pub labels: Vec<String>,
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

/// Branch-and-bound trajectory sampler.
pub struct TrajectorySampler {
    num_branches: usize,
    prune_threshold: f32,
}

impl TrajectorySampler {
    /// Create a new sampler with custom config.
    pub fn new(num_branches: usize, prune_threshold: f32) -> Self {
        Self {
            num_branches,
            prune_threshold,
        }
    }

    /// Default configuration.
    pub fn default_config() -> Self {
        Self::new(DEFAULT_NUM_BRANCHES, DEFAULT_PRUNE_THRESHOLD)
    }

    /// Sample trajectories from a state map.
    pub fn sample(&self, map: &SemanticStateMap) -> TrajectorySampleResult {
        let mut branches = Vec::with_capacity(self.num_branches);

        // Generate branches by perturbing the context state
        for i in 0..self.num_branches {
            let perturbation = (i as f32) * 0.1;
            let state_vector = map
                .context_state
                .iter()
                .enumerate()
                .map(|(j, &v)| v + perturbation * ((j as f32).sin() * 0.05))
                .collect::<Vec<f32>>();

            // Score the branch
            let novelty = Self::compute_novelty(&state_vector, &map.context_state);
            let coherence = Self::compute_coherence(&state_vector, &map.milestones);
            let budget_fit = Self::compute_budget_fit(&map.milestones);
            let composite = 0.4 * novelty + 0.3 * coherence + 0.3 * budget_fit;

            let label = if map.milestones.is_empty() {
                format!("branch_{}", i)
            } else {
                let ms = &map.milestones[i % map.milestones.len()];
                format!("{} ({:.2})", ms.label, ms.confidence)
            };

            branches.push(TrajectoryBranch {
                label,
                state_vector,
                composite_score: composite,
                novelty,
                coherence,
                budget_fit,
                depth: 0,
            });
        }

        let total_explored = branches.len();

        // Prune low-scoring branches
        let before = branches.len();
        branches.retain(|b| b.composite_score >= self.prune_threshold);
        let pruned = before - branches.len();

        TrajectorySampleResult {
            branches,
            session_id: map.session_id.clone(),
            total_explored,
            pruned,
        }
    }

    /// Sample trajectories and also build a cognitive manifold.
    pub fn sample_with_manifold(
        &self,
        map: &SemanticStateMap,
        _context: &str,
    ) -> (TrajectorySampleResult, CognitiveManifold) {
        let result = self.sample(map);

        // Build manifold from non-pruned branches
        let nodes: Vec<Vec<f32>> = result
            .branches
            .iter()
            .map(|b| b.state_vector.clone())
            .collect();
        let labels: Vec<String> = result
            .branches
            .iter()
            .map(|b| b.label.clone())
            .collect();

        // Build edges based on cosine similarity
        let mut edges = Vec::new();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let sim = Self::cosine_sim(&nodes[i], &nodes[j]);
                if sim > 0.5 {
                    edges.push((i, j, sim));
                }
            }
        }

        let manifold = CognitiveManifold {
            nodes,
            edges,
            labels,
        };

        (result, manifold)
    }

    /// Evolve branches by applying Gaussian perturbation and re-scoring.
    pub fn evolve_branches(&self, result: &mut TrajectorySampleResult, sigma: f32) {
        for branch in &mut result.branches {
            // Perturb the state vector
            let perturbed: Vec<f32> = branch
                .state_vector
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let noise = ((i as f32 + 1.0).sin() * sigma).abs();
                    v + noise * 0.1
                })
                .collect();

            // Re-score
            let novelty = Self::compute_novelty(&perturbed, &branch.state_vector);
            let coherence = (branch.coherence + 0.05).min(1.0);
            let budget_fit = branch.budget_fit;
            let composite = 0.4 * novelty + 0.3 * coherence + 0.3 * budget_fit;

            branch.state_vector = perturbed;
            branch.composite_score = composite;
            branch.novelty = novelty;
            branch.coherence = coherence;
            branch.depth += 1;
        }
    }

    // -- Scoring helpers --------------------------------------------------

    fn compute_novelty(candidate: &[f32], reference: &[f32]) -> f32 {
        if candidate.is_empty() || reference.is_empty() {
            return 0.5;
        }
        let dist = Self::l2_dist(candidate, reference);
        let novelty = 1.0 / (1.0 + dist);
        novelty.clamp(0.0, 1.0)
    }

    fn compute_coherence(state: &[f32], milestones: &[crate::state_predictor::SemanticMilestone]) -> f32 {
        if milestones.is_empty() || state.is_empty() {
            return 0.5;
        }
        let avg_conf: f32 = milestones.iter().map(|m| m.confidence).sum::<f32>() / milestones.len() as f32;
        let norm: f32 = state.iter().map(|v| v * v).sum::<f32>().sqrt();
        (avg_conf * 0.7 + (norm / 10.0).min(1.0) * 0.3).clamp(0.0, 1.0)
    }

    fn compute_budget_fit(milestones: &[crate::state_predictor::SemanticMilestone]) -> f32 {
        if milestones.is_empty() {
            return 0.5;
        }
        let avg_budget: f32 = milestones.iter().map(|m| m.token_budget as f32).sum::<f32>()
            / milestones.len() as f32
            / 4096.0;
        avg_budget.clamp(0.0, 1.0)
    }

    fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = 0.0;
        for i in 0..len {
            let d = a[i] - b[i];
            sum += d * d;
        }
        sum.sqrt()
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        if len == 0 {
            return 0.0;
        }
        let dot: f32 = (0..len).map(|i| a[i] * b[i]).sum();
        let norm_a: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm_a < 1e-8 || norm_b < 1e-8 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render a trajectory sample result as a human-readable string.
pub fn render_trajectory_result(result: &TrajectorySampleResult) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "Trajectory Sample (session: {})\n",
        result.session_id
    ));
    s.push_str(&format!(
        "  Explored: {} | Pruned: {} | Surviving: {}\n\n",
        result.total_explored,
        result.pruned,
        result.branches.len()
    ));
    for (i, b) in result.branches.iter().enumerate() {
        s.push_str(&format!(
            "  [{}] {} (score={:.3}, novelty={:.3}, coherence={:.3}, budget_fit={:.3}, depth={})\n",
            i, b.label, b.composite_score, b.novelty, b.coherence, b.budget_fit, b.depth
        ));
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_predictor::{SemanticStateMap, SemanticMilestone, BetaBelief};

    fn make_state_map() -> SemanticStateMap {
        let milestones = vec![
            SemanticMilestone {
                label: "hypothesis".to_string(),
                confidence: 0.8,
                token_budget: 512,
            },
            SemanticMilestone {
                label: "decomposition".to_string(),
                confidence: 0.7,
                token_budget: 1024,
            },
            SemanticMilestone {
                label: "execution".to_string(),
                confidence: 0.6,
                token_budget: 2048,
            },
        ];
        SemanticStateMap {
            milestones,
            confidence: BetaBelief::default(),
            session_id: "test".to_string(),
            context_state: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    }

    #[test]
    fn sample_produces_branches() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        assert!(!result.branches.is_empty(), "should produce branches");
        assert!(result.total_explored > 0, "should have explored branches");
    }

    #[test]
    fn scores_are_in_valid_range() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        for b in &result.branches {
            assert!(b.composite_score >= 0.0 && b.composite_score <= 1.0);
            assert!(b.novelty >= 0.0 && b.novelty <= 1.0);
            assert!(b.coherence >= 0.0 && b.coherence <= 1.0);
            assert!(b.budget_fit >= 0.0 && b.budget_fit <= 1.0);
        }
    }

    #[test]
    fn pruning_removes_low_scoring_branches() {
        let sampler = TrajectorySampler::new(8, 0.99);
        let map = make_state_map();
        let result = sampler.sample(&map);
        assert!(result.pruned > 0, "should prune some branches");
    }

    #[test]
    fn selection_returns_best_branch() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        let best = result.branches.iter().max_by(|a, b| {
            a.composite_score
                .partial_cmp(&b.composite_score)
                .unwrap()
        });
        assert!(best.is_some(), "should have a best branch");
    }

    #[test]
    fn evolve_changes_scores() {
        let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);
        let after: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        assert_eq!(before.len(), after.len(), "evolve should not change branch count");
        let changed = before.iter().zip(after.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "evolution should change scores");
    }

    #[test]
    fn render_is_readable() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        let rendered = render_trajectory_result(&result);
        assert!(rendered.contains("Trajectory Sample"), "render should contain title");
        assert!(rendered.contains("session: test"), "render should contain session id");
    }

    #[test]
    fn empty_map_produces_valid_result() {
        let sampler = TrajectorySampler::default_config();
        let map = SemanticStateMap::default();
        let result = sampler.sample(&map);
        assert!(result.total_explored > 0, "should explore even with empty map");
    }

    #[test]
    fn manifold_integration_works() {
        let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let (result, manifold) = sampler.sample_with_manifold(&map, "test context");
        assert!(!manifold.nodes.is_empty(), "manifold should be created from non-pruned branches");
        assert_eq!(
            manifold.nodes.len(),
            result.branches.len(),
            "manifold nodes should match surviving branches"
        );
    }
}