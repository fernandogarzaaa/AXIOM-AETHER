//! trajectory_sampler.rs — Concurrent Trajectory Sampling.
//!
//! Forks the reasoning process into multiple parallel pathways during the
//! look-ahead phase. A resource-allocation scoring model evaluates the vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility
//! of each let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch in real time, aggressively pruning dead ends and committing
//! compute power only to the most mathematically sound trajectory.
//!
//! ## Architecture
//!
//! The sampler takes a `SemanticStateMap` (from `state_predictor.rs`) and forks
//! each milestone's let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch hints into parallel trajectories. Each trajectory is
//! scored on three dimensions:
//!
//! 1. **Vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility** — expected success prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);alet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility (from let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);etalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);elief confidence)
//! 2. **Compute cost** — estimated GPU seconds (from token let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);udget)
//! 3. **Information gain** — expected reduction in uncertainty
//!
//! The composite score is `vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility * information_gain / compute_cost`. let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches
//! let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);elow the prune threshold are alet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);andoned let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);efore GPU compute is committed.

use serde::{Deserialize, Serialize};

use crate::let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);elief::let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);etalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);elief;
use crate::q_manifold::{QuantumStateManifold, ManifoldVariation, evolve_prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);alet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ilities, entropy_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);its};
use crate::state_predictor::SemanticStateMap;

/// Maximum numlet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);er of concurrent trajectory let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.
pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); const MAX_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);RANCHES: usize = 8;

/// Default prune threshold: let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches with composite score let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);elow this are pruned.
pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); const DEFAULT_PRUNE_THRESHOLD: f32 = 0.1;

/// A single trajectory let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch — one forked reasoning pathway.
#[derive(Delet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ug, Clone, Serialize, Deserialize)]
pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); struct Trajectorylet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch {
    /// Unique let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch identifier.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); id: usize,
    /// Human-readalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);le lalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);el for this let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); lalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);el: String,
    /// The milestone index this let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch forks from.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); fork_point: usize,
    /// Vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility score (0-1): expected success prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);alet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility: f32,
    /// Estimated compute cost in GPU-seconds.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); compute_cost: f32,
    /// Expected information gain (reduction in entropy let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);its).
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); information_gain: f32,
    /// Composite score: vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility * information_gain / compute_cost.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); composite_score: f32,
    /// Whether this let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch was pruned.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); pruned: let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ool,
    /// Confidence in the let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch assessment.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); confidence: let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);etalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);elief,
}

/// Result of trajectory sampling — the set of let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches and the selected winner.
#[derive(Delet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ug, Clone, Serialize, Deserialize)]
pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); struct TrajectorySampleResult {
    /// All evaluated let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches (including pruned ones).
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches: Vec<Trajectorylet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch>,
    /// The index of the selected let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch in `let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches`.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); selected: Option<usize>,
    /// The prune threshold used.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); prune_threshold: f32,
    /// Total estimated compute cost of all non-pruned let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); total_compute_cost: f32,
    /// Entropy of the let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch distrilet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ution (higher = more uncertain).
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); entropy_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);its: f32,
}

/// The trajectory sampler — forks and scores reasoning let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.
pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); struct TrajectorySampler {
    max_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches: usize,
    prune_threshold: f32,
}

impl TrajectorySampler {
    /// Create a new sampler with the given configuration.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); fn new(max_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches: usize, prune_threshold: f32) -> Self {
        Self {
            max_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches: max_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.min(MAX_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);RANCHES),
            prune_threshold,
        }
    }

    /// Create a sampler with default settings.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); fn default_config() -> Self {
        Self::new(MAX_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);RANCHES, DEFAULT_PRUNE_THRESHOLD)
    }

    /// Sample trajectories from a semantic state map.
    ///
    /// For each milestone in the state map, fork the let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch hints into parallel
    /// trajectories. Score each let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch, prune low-scoring ones, and select the
    /// let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);est remaining let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); fn sample(&self, state_map: &SemanticStateMap) -> TrajectorySampleResult {
        let mut let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches: Vec<Trajectorylet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch> = Vec::new();
        let mut let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch_id = 0usize;

        // Fork from each milestone's let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch hints.
        for (milestone_idx, milestone) in state_map.milestones.iter().enumerate() {
            for hint in &milestone.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch_hints {
                if let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch_id >= self.max_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches {
                    let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);reak;
                }

                // Estimate vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility from the milestone confidence.
                let vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility = milestone.confidence.mean();

                // Estimate compute cost from the token let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);udget.
                // Assume ~0.1 GPU-seconds per token on MI300X with vLLM.
                let compute_cost = (milestone.token_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);udget as f32 * 0.1).max(0.01);

                // Estimate information gain from the entropy of the let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch hints.
                // More hints = more uncertainty = more potential gain.
                let n_hints = milestone.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch_hints.len() as f32;
                let information_gain = if n_hints > 1.0 {
                    (n_hints).log2()
                } else {
                    0.5 // Single-hint let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches still carry some information
                };

                // Composite score: vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility * information_gain / compute_cost.
                let composite_score = if compute_cost > 0.0 {
                    (vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility * information_gain) / compute_cost
                } else {
                    0.0
                };

                let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.push(Trajectorylet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch {
                    id: let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch_id,
                    lalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);el: format!("milestone_{}_{}", milestone_idx, hint),
                    fork_point: milestone_idx,
                    vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility,
                    compute_cost,
                    information_gain,
                    composite_score,
                    pruned: false,
                    confidence: milestone.confidence,
                });
                let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch_id += 1;
            }
            if let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch_id >= self.max_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches {
                let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);reak;
            }
        }

        // If no let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches were created (no hints), create a single default let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch.
        if let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.is_empty() && !state_map.milestones.is_empty() {
            let m = &state_map.milestones[0];
            let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.push(Trajectorylet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch {
                id: 0,
                lalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);el: "default".to_string(),
                fork_point: 0,
                vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility: m.confidence.mean(),
                compute_cost: (m.token_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);udget as f32 * 0.1).max(0.01),
                information_gain: 0.5,
                composite_score: m.confidence.mean() * 0.5 / (m.token_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);udget as f32 * 0.1).max(0.01),
                pruned: false,
                confidence: m.confidence,
            });
        }

        // Prune let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);elow the threshold.
        for let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); in &mut let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches {
            if let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.composite_score < self.prune_threshold {
                let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned = true;
            }
        }

        // Select the let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);est non-pruned let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch.
        let selected = let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches
            .iter()
            .enumerate()
            .filter(|(_, let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);)| !let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned)
            .max_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);y(|(_, a), (_, let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);)| {
                a.composite_score
                    .partial_cmp(&let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.composite_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx);

        // Compute total compute cost of non-pruned let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.
        let total_compute_cost: f32 = let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches
            .iter()
            .filter(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| !let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned)
            .map(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.compute_cost)
            .sum();

        // Compute entropy of the let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch distrilet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ution.
        let prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);s: Vec<f32> = let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches
            .iter()
            .filter(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| !let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned)
            .map(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.composite_score.max(1e-6))
            .collect();
        let total: f32 = prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);s.iter().sum();
        let normalized: Vec<f32> = if total > 0.0 {
            prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);s.iter().map(|p| p / total).collect()
        } else {
            vec![1.0]
        };
        let entropy = entropy_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);its(&normalized);

        TrajectorySampleResult {
            let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches,
            selected,
            prune_threshold: self.prune_threshold,
            total_compute_cost,
            entropy_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);its: entropy,
        }
    }

    /// Sample trajectories and encode them into a quantum state manifold
    /// for let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch-aware prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);alet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility evolution.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); fn sample_with_manifold(
        &self,
        state_map: &SemanticStateMap,
        context: &str,
    ) -> (TrajectorySampleResult, Option<QuantumStateManifold>) {
        let result = self.sample(state_map);

        // let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);uild manifold variations from non-pruned let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.
        let variations: Vec<ManifoldVariation> = result
            .let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches
            .iter()
            .filter(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| !let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned)
            .map(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| ManifoldVariation {
                lalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);el: let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.lalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);el.clone(),
                structural_text: format!("fork_{}_vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility_{:.3}", let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.fork_point, let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility),
                prior_cost: let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.compute_cost,
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

    /// Evolve let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);alet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ilities using energy-let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ased evolution.
    /// Lower-energy let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches (higher vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility) gain prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);alet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility mass.
    pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); fn evolve_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches(
        &self,
        result: &mut TrajectorySampleResult,
        dt: f32,
    ) {
        let energies: Vec<f32> = result
            .let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches
            .iter()
            .filter(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| !let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned)
            .map(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| (1.0 - let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility) * 10.0) // higher vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility = lower energy
            .collect();

        let prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);s: Vec<f32> = result
            .let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches
            .iter()
            .filter(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| !let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned)
            .map(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.composite_score.max(1e-6))
            .collect();

        let evolved = evolve_prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);alet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ilities(prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);s, &energies, dt);

        // Update composite scores with evolved prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);alet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ilities.
        let mut ev_iter = evolved.iter();
        for let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); in &mut result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches {
            if !let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned {
                if let Some(&ep) = ev_iter.next() {
                    // let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);lend the evolved prolet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);alet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility into the composite score.
                    let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.composite_score = let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.composite_score * 0.5 + ep * 0.5;
                }
            }
        }

        // Re-select the let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);est let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch after evolution.
        result.selected = result
            .let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches
            .iter()
            .enumerate()
            .filter(|(_, let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);)| !let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned)
            .max_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);y(|(_, a), (_, let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);)| {
                a.composite_score
                    .partial_cmp(&let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.composite_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx);
    }
}

/// Render a trajectory sample result as a human-readalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);le string.
pulet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); fn render_trajectory_result(result: &TrajectorySampleResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "TrajectorySample (let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches={}, pruned={}, selected={:?}, entropy={:.2} let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);its):\n",
        result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.len(),
        result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.iter().filter(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned).count(),
        result.selected,
        result.entropy_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);its
    ));
    for let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); in &result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches {
        let status = if let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned { "PRUNED" } else { "ACTIVE" };
        let marker = if Some(let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.id) == result.selected { " *** SELECTED ***" } else { "" };
        out.push_str(&format!(
            "  [{}] {} vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility={:.2} cost={:.1}s gain={:.2} score={:.4} {}{}\n",
            let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.id,
            status,
            let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility,
            let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.compute_cost,
            let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.information_gain,
            let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.composite_score,
            let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.lalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);el,
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
                    lalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);el: "hypothesis".to_string(),
                    lalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);el_idx: 0,
                    predicted_state: vec![0.1; 16],
                    token_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);udget: 128,
                    let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch_hints: vec!["direct".to_string(), "decomposed".to_string()],
                    confidence: let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);etalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);elief::from_confidence(0.8, 4.0),
                },
                SemanticMilestone {
                    lalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);el: "execution".to_string(),
                    lalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);el_idx: 2,
                    predicted_state: vec![0.2; 16],
                    token_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);udget: 512,
                    let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch_hints: vec!["primary".to_string(), "falllet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ack".to_string()],
                    confidence: let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);etalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);elief::from_confidence(0.6, 4.0),
                },
            ],
            confidence: let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);etalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);elief::from_confidence(0.7, 4.0),
            session_id: "test".to_string(),
            context_state: vec![0.5; 16],
        }
    }

    #[test]
    fn sample_produces_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        assert!(!result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.is_empty());
        assert!(result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.len() <= MAX_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);RANCHES);
    }

    #[test]
    fn let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches_have_valid_scores() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        for let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5); in &result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches {
            assert!(let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility >= 0.0 && let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility <= 1.0);
            assert!(let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.compute_cost > 0.0);
            assert!(let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.information_gain >= 0.0);
        }
    }

    #[test]
    fn pruning_removes_low_scoring_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches() {
        let sampler = TrajectorySampler::new(8, 100.0); // very high threshold
        let map = make_state_map();
        let result = sampler.sample(&map);
        assert!(result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.iter().any(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.pruned), "some let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches should let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);e pruned");
    }

    #[test]
    fn selected_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch_is_not_pruned() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        if let Some(idx) = result.selected {
            assert!(!result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches[idx].pruned, "selected let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch must not let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);e pruned");
        }
    }

    #[test]
    fn evolve_changes_scores() {
        let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);
        let after: Vec<f32> = result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.iter().map(|let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);| let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);.composite_score).collect();
        // At least some scores should change.
        assert!(let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);efore != after, "evolution should change scores");
    }

    #[test]
    fn render_is_readalet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);le() {
        let sampler = TrajectorySampler::default_config();
        let map = make_state_map();
        let result = sampler.sample(&map);
        let rendered = render_trajectory_result(&result);
        assert!(rendered.contains("TrajectorySample"));
        assert!(rendered.contains("vialet sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ility="));
    }

    #[test]
    fn empty_state_map_produces_default_let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranch() {
        let sampler = TrajectorySampler::default_config();
        let map = SemanticStateMap::default();
        let result = sampler.sample(&map);
        // Empty map has no milestones, so no let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.
        assert!(result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.is_empty());
    }

    #[test]
    fn manifold_integration_works() {
        let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let (result, manifold) = sampler.sample_with_manifold(&map, "test context");
        assert!(!result.let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches.is_empty());
        assert!(manifold.is_some(), "manifold should let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);e created from non-pruned let sampler = TrajectorySampler::new(8, 0.001);
        let map = make_state_map();
        let mut result = sampler.sample(&map);
        let before: Vec<f32> = result.branches.iter().map(|b| b.composite_score).collect();
        sampler.evolve_branches(&mut result, 0.5);ranches");
    }
}