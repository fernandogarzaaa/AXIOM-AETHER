//! The Kinetic Neural Mesh: sparse, dynamic prompt routing.
//!
//! The mesh is not a static layer — it is a set of worker nodes whose
//! affinity embeddings form a routing matrix that can be re-shaped at
//! runtime (nodes join, leave, or shift affinity as the residual evolves).
//!
//! Routing model, per forward pass:
//!
//! 1. **Gravitational field** — the payload embedding is projected against
//!    every node's affinity vector (`logits = A · p + bias`), optionally
//!    modulated by the current residual so nodes aligned with the *gap*
//!    pull harder than nodes aligned with what is already done.
//! 2. **Adhesion** — hard Gumbel-Softmax snaps the payload to a discrete
//!    winner per routing slot (top-k slots for multi-node fan-out).
//! 3. **Sparse activation** — only the snapped nodes are returned; nothing
//!    else in the mesh runs.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use thiserror::Error;

use crate::gumbel::gumbel_softmax;
use crate::node::{NodeId, WorkerNode};
use crate::residual::Residual;

#[derive(Debug, Error)]
pub enum MeshError {
    #[error("mesh has no nodes")]
    Empty,
    #[error("payload dim {payload} != mesh dim {mesh}")]
    DimMismatch { payload: usize, mesh: usize },
    #[error("node affinity dim {node} != mesh dim {mesh}")]
    NodeDimMismatch { node: usize, mesh: usize },
}

/// The outcome of one forward pass: which nodes the payload snapped to.
#[derive(Debug, Clone)]
pub struct Adhesion {
    /// Adhesion weight per node (n_nodes). With hard routing this is a
    /// multi-hot vector: 1.0 for each activated node, 0.0 elsewhere.
    pub weights: Array1<f32>,
    /// The activated nodes, strongest pull first. Sparse activation means
    /// dispatch iterates exactly this list and nothing else.
    pub active: Vec<NodeId>,
    /// Pre-adhesion gravitational field (the raw logits), for telemetry.
    pub field: Array1<f32>,
}

/// Result of an expert-choice batch dispatch (see
/// [`KineticNeuralMesh::forward_batch`]).
#[derive(Debug, Clone)]
pub struct BatchAdhesion {
    /// `(node, payload indices)` for every node that claimed at least one
    /// payload. A payload index appears in at most one node's list.
    pub assignments: Vec<(NodeId, Vec<usize>)>,
    /// Payload indices no node had capacity left for.
    pub dropped: Vec<usize>,
}

/// Configuration for the mesh's routing behavior.
#[derive(Debug, Clone)]
pub struct MeshConfig {
    /// Embedding dimension shared by payloads and node affinities.
    pub dim: usize,
    /// Gumbel-Softmax temperature used when `tau_gain == 0.0` (annealing
    /// disabled) or no residual is available for a call.
    pub tau: f32,
    /// Hard (discrete one-hot) routing. The KNM spec requires the hard
    /// variant; soft is kept for diagnostics.
    pub hard: bool,
    /// How many nodes a single payload may snap to (top-k fan-out).
    pub fan_out: usize,
    /// Gain applied to residual alignment when a residual is provided.
    /// 0.0 disables residual modulation.
    pub residual_gain: f32,
    /// Residual-adaptive temperature: when a residual is supplied and this
    /// gain is nonzero, `tau_effective = clamp(tau_gain * |residual|,
    /// tau_min, tau_max)` — routing explores more while far from the goal
    /// and sharpens toward argmax as the controller converges, mirroring
    /// the explore-then-exploit anneal used in simulated annealing and in
    /// entropy-regularized routing (ReinMax). 0.0 (default) disables this
    /// and keeps the fixed `tau` above, so existing callers are unaffected.
    pub tau_gain: f32,
    /// Lower clamp for the annealed temperature.
    pub tau_min: f32,
    /// Upper clamp for the annealed temperature.
    pub tau_max: f32,
    /// Backpressure: logit penalty per in-flight dispatch already active on
    /// a node (see [`KineticNeuralMesh::mark_active`]), analogous to
    /// per-expert capacity constraints in sparse MoE routing. 0.0 (default)
    /// disables this.
    pub capacity_penalty: f32,
    /// Weight applied to each node's online-learned routing quality — a
    /// running mean of the reward reported through
    /// [`KineticNeuralMesh::record_outcome`] — in the gravitational field.
    /// 0.0 (default) disables bandit-learned routing and leaves `field()`
    /// driven purely by affinity/bias/residual/capacity.
    pub bandit_gain: f32,
    /// UCB1 exploration coefficient: added on top of the quality term as
    /// `bandit_exploration * sqrt(ln(total_visits) / node_visits)`, biasing
    /// routing toward under-sampled nodes. Independent of `bandit_gain` —
    /// set this alone for pure exploration before any quality signal
    /// exists. 0.0 (default) disables the bonus.
    pub bandit_exploration: f32,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            tau: 0.5,
            hard: true,
            fan_out: 1,
            residual_gain: 1.0,
            tau_gain: 0.0,
            tau_min: 0.05,
            tau_max: 1.0,
            capacity_penalty: 0.0,
            bandit_gain: 0.0,
            bandit_exploration: 0.0,
        }
    }
}

/// The Kinetic Neural Mesh.
pub struct KineticNeuralMesh {
    config: MeshConfig,
    nodes: Vec<WorkerNode>,
    /// Row-stacked affinity matrix (n_nodes × dim), rebuilt on topology
    /// change so the forward pass is a single matrix-vector product.
    affinity: Array2<f32>,
    bias: Array1<f32>,
    /// In-flight dispatch count per node, index-aligned with `nodes`. Reset
    /// on topology change (see `rebuild`); tracked via `mark_active` /
    /// `mark_idle` and fed into `field()` as backpressure.
    in_flight: Vec<u32>,
    /// Running-mean observed reward per node (see `record_outcome`), reset
    /// on topology change.
    reward_mean: Vec<f32>,
    /// Observation count per node, backing both the running mean above and
    /// the UCB1 exploration bonus.
    visits: Vec<u32>,
}

impl KineticNeuralMesh {
    pub fn new(config: MeshConfig) -> Self {
        Self {
            affinity: Array2::zeros((0, config.dim)),
            bias: Array1::zeros(0),
            nodes: Vec::new(),
            in_flight: Vec::new(),
            reward_mean: Vec::new(),
            visits: Vec::new(),
            config,
        }
    }

    /// Record a dispatch starting on `id`, so subsequent routing decisions
    /// apply capacity backpressure against it (see `MeshConfig::capacity_penalty`).
    /// A no-op if `id` isn't currently in the mesh.
    pub fn mark_active(&mut self, id: NodeId) {
        if let Some(idx) = self.nodes.iter().position(|n| n.id == id) {
            self.in_flight[idx] += 1;
        }
    }

    /// Record a dispatch completing (success or failure) on `id`.
    pub fn mark_idle(&mut self, id: NodeId) {
        if let Some(idx) = self.nodes.iter().position(|n| n.id == id) {
            self.in_flight[idx] = self.in_flight[idx].saturating_sub(1);
        }
    }

    /// Feed back the outcome of a completed dispatch: `reward` should be
    /// the residual-norm improvement attributable to this node's action
    /// (positive = it helped shrink the gap toward the goal; negative or
    /// zero = it didn't). Updates a running-mean quality estimate that
    /// `field()` consumes when `MeshConfig::bandit_gain != 0.0`.
    ///
    /// Because reward is measured against the state *at the moment this
    /// dispatch resolves* (not at the moment it was sent), this remains a
    /// reasonable marginal-credit signal even with `fan_out > 1` and
    /// overlapping in-flight dispatches — each completion is scored
    /// against whatever the environment looks like when it lands, not
    /// against a stale snapshot from dispatch time.
    ///
    /// A no-op if `id` isn't currently in the mesh (e.g. it was quarantined
    /// while its dispatch was still in flight).
    pub fn record_outcome(&mut self, id: NodeId, reward: f32) {
        if let Some(idx) = self.nodes.iter().position(|n| n.id == id) {
            self.visits[idx] += 1;
            self.reward_mean[idx] += (reward - self.reward_mean[idx]) / self.visits[idx] as f32;
        }
    }

    /// Add a node to the mesh (topology reconfiguration is cheap and
    /// expected at runtime).
    pub fn add_node(&mut self, node: WorkerNode) -> Result<NodeId, MeshError> {
        if node.affinity.len() != self.config.dim {
            return Err(MeshError::NodeDimMismatch { node: node.affinity.len(), mesh: self.config.dim });
        }
        self.nodes.push(node);
        self.rebuild();
        Ok(self.nodes.last().unwrap().id)
    }

    /// Remove a node by id. Returns true if a node was removed.
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        let before = self.nodes.len();
        self.nodes.retain(|n| n.id != id);
        let removed = self.nodes.len() != before;
        if removed {
            self.rebuild();
        }
        removed
    }

    pub fn nodes(&self) -> &[WorkerNode] {
        &self.nodes
    }

    pub fn node(&self, id: NodeId) -> Option<&WorkerNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    fn rebuild(&mut self) {
        let n = self.nodes.len();
        let mut affinity = Array2::zeros((n, self.config.dim));
        let mut bias = Array1::zeros(n);
        for (i, node) in self.nodes.iter().enumerate() {
            affinity.row_mut(i).assign(&Array1::from_vec(node.affinity.clone()));
            bias[i] = node.bias;
        }
        self.affinity = affinity;
        self.bias = bias;
        // Topology changed the index<->node mapping, so any prior in-flight
        // counts and learned reward estimates are no longer attributable to
        // the right node; resetting is the safe default (a removed node's
        // calls will still `mark_idle`/`record_outcome` into a no-op once
        // it's gone, see above).
        self.in_flight = vec![0; n];
        self.reward_mean = vec![0.0; n];
        self.visits = vec![0; n];
    }

    /// Compute the gravitational field a payload projects over the mesh:
    /// affinity dot-products plus per-node bias, plus (optionally) a term
    /// rewarding nodes aligned with the current residual direction, minus
    /// capacity backpressure for nodes already busy, plus (optionally) an
    /// online-learned quality/exploration term from `record_outcome`.
    fn field(&self, payload: &Array1<f32>, residual: Option<&Residual>) -> Array1<f32> {
        let mut logits = self.affinity.dot(payload) + &self.bias;
        if let Some(r) = residual {
            if self.config.residual_gain != 0.0 && r.norm() > 0.0 {
                let direction = &r.vector / r.norm();
                logits = logits + self.config.residual_gain * self.affinity.dot(&direction);
            }
        }
        if self.config.capacity_penalty != 0.0 {
            for (i, &n) in self.in_flight.iter().enumerate() {
                logits[i] -= self.config.capacity_penalty * n as f32;
            }
        }
        if self.config.bandit_gain != 0.0 || self.config.bandit_exploration != 0.0 {
            // UCB1-style: total_visits/node_visits are floored at 1 rather
            // than special-cased at 0, an "optimistic initialization"
            // simplification that avoids infinities (which would poison
            // the softmax in gumbel_softmax with NaNs) while still giving
            // never-visited nodes a strong, bounded exploration bonus.
            let total_visits = (self.visits.iter().sum::<u32>().max(1)) as f32;
            for i in 0..logits.len() {
                if self.config.bandit_gain != 0.0 {
                    logits[i] += self.config.bandit_gain * self.reward_mean[i];
                }
                if self.config.bandit_exploration != 0.0 {
                    let n_i = self.visits[i].max(1) as f32;
                    logits[i] += self.config.bandit_exploration * (total_visits.ln().max(0.0) / n_i).sqrt();
                }
            }
        }
        logits
    }

    /// Effective Gumbel-Softmax temperature for this call: the residual-
    /// adaptive anneal when enabled and a residual is available, else the
    /// static `config.tau`.
    fn effective_tau(&self, residual: Option<&Residual>) -> f32 {
        match residual {
            Some(r) if self.config.tau_gain > 0.0 => {
                (self.config.tau_gain * r.norm()).clamp(self.config.tau_min, self.config.tau_max)
            }
            _ => self.config.tau,
        }
    }

    /// Forward pass: snap a payload embedding onto the mesh.
    ///
    /// `residual` — the current IDC residual; when provided, nodes whose
    /// affinity aligns with the remaining gap pull harder. Pass `None` for
    /// residual-agnostic routing (e.g. the very first dispatch).
    pub fn forward(
        &self,
        payload: &Array1<f32>,
        residual: Option<&Residual>,
        rng: &mut impl Rng,
    ) -> Result<Adhesion, MeshError> {
        if self.nodes.is_empty() {
            return Err(MeshError::Empty);
        }
        if payload.len() != self.config.dim {
            return Err(MeshError::DimMismatch { payload: payload.len(), mesh: self.config.dim });
        }

        let field = self.field(payload, residual);
        let fan_out = self.config.fan_out.min(self.nodes.len()).max(1);
        let tau = self.effective_tau(residual);

        // Top-k discrete routing: draw a hard Gumbel-Softmax winner, mask
        // it out, and redraw — each slot is an independent discrete snap,
        // so fan_out=1 is exactly the classic hard Gumbel-Softmax.
        let mut weights: Array1<f32> = Array1::zeros(self.nodes.len());
        let mut active: Vec<NodeId> = Vec::with_capacity(fan_out);
        let mut masked = field.clone();
        for _ in 0..fan_out {
            let sample = gumbel_softmax(&masked, tau, self.config.hard, rng);
            let w = sample.winner;
            weights[w] = if self.config.hard { 1.0 } else { sample.adhesion[w] };
            active.push(self.nodes[w].id);
            masked[w] = f32::NEG_INFINITY;
        }

        Ok(Adhesion { weights, active, field })
    }

    /// Expert-choice batch dispatch: instead of each payload picking its
    /// best node (`forward`'s model — one payload choosing among nodes),
    /// every node independently ranks *all* payloads and claims its best
    /// ones up to `capacity`. That flip — nodes choosing payloads rather
    /// than the reverse — is what actually solves load imbalance in
    /// sparse MoE routing (see docs/RESEARCH_BLUEPRINT.md): a popular
    /// node can't be swamped, because it's capacity-bounded regardless of
    /// how many payloads want it.
    ///
    /// Assignment is a deterministic greedy global match: every
    /// (payload, node) pair is scored via the same gravitational field
    /// `forward` uses, sorted best-first (ties broken by payload then node
    /// index so the result never depends on sort-stability quirks), and
    /// walked once — a payload is claimed by the first node in that order
    /// that still has room. There's no Gumbel-Softmax draw here: with
    /// `capacity` doing the real work of preventing pileup, a
    /// deterministic best-match walk is simpler than a stochastic one and
    /// no less principled — this runtime has no autograd tape either way,
    /// so nothing about the discrete decision needs to stay differentiable.
    ///
    /// A payload no node has room left for by the time its turn comes is
    /// dropped, matching real capacity-constrained MoE routing (which
    /// drops or pads overflow tokens rather than forcing an imbalanced
    /// assignment).
    pub fn forward_batch(
        &self,
        payloads: &[Array1<f32>],
        capacity: usize,
        residual: Option<&Residual>,
    ) -> Result<BatchAdhesion, MeshError> {
        if self.nodes.is_empty() {
            return Err(MeshError::Empty);
        }
        for p in payloads {
            if p.len() != self.config.dim {
                return Err(MeshError::DimMismatch { payload: p.len(), mesh: self.config.dim });
            }
        }
        let capacity = capacity.max(1);

        let mut scored: Vec<(f32, usize, usize)> =
            Vec::with_capacity(payloads.len() * self.nodes.len());
        for (p_idx, payload) in payloads.iter().enumerate() {
            let field = self.field(payload, residual);
            for (n_idx, &score) in field.iter().enumerate() {
                scored.push((score, p_idx, n_idx));
            }
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

        let mut node_slots = vec![0usize; self.nodes.len()];
        let mut payload_claimed = vec![false; payloads.len()];
        let mut per_node: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];

        for (_, p_idx, n_idx) in scored {
            if payload_claimed[p_idx] || node_slots[n_idx] >= capacity {
                continue;
            }
            payload_claimed[p_idx] = true;
            node_slots[n_idx] += 1;
            per_node[n_idx].push(p_idx);
        }

        let assignments = per_node
            .into_iter()
            .enumerate()
            .filter(|(_, ps)| !ps.is_empty())
            .map(|(n_idx, ps)| (self.nodes[n_idx].id, ps))
            .collect();
        let dropped = (0..payloads.len()).filter(|&p| !payload_claimed[p]).collect();

        Ok(BatchAdhesion { assignments, dropped })
    }

    /// Mean row of the affinity matrix — a cheap mesh "center of mass",
    /// useful for drift telemetry.
    pub fn center_of_mass(&self) -> Option<Array1<f32>> {
        if self.nodes.is_empty() {
            return None;
        }
        self.affinity.mean_axis(Axis(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeKind;
    use crate::residual::StateVector;
    use ndarray::array;
    use rand::{rngs::StdRng, SeedableRng};

    fn axis_node(id: usize, name: &str, dim: usize, axis: usize) -> WorkerNode {
        let mut affinity = vec![0.0; dim];
        affinity[axis] = 8.0; // strong pull along one axis
        WorkerNode::new(id, name, NodeKind::Llm(name.to_string()), affinity)
    }

    fn mesh3() -> KineticNeuralMesh {
        let mut mesh = KineticNeuralMesh::new(MeshConfig { dim: 3, tau: 0.1, ..Default::default() });
        mesh.add_node(axis_node(0, "codex", 3, 0)).unwrap();
        mesh.add_node(axis_node(1, "claude", 3, 1)).unwrap();
        mesh.add_node(axis_node(2, "gemini", 3, 2)).unwrap();
        mesh
    }

    #[test]
    fn payload_snaps_to_highest_affinity_node() {
        let mesh = mesh3();
        let mut rng = StdRng::seed_from_u64(3);
        // Payload lives entirely on axis 1 → must snap to "claude".
        let adhesion = mesh.forward(&array![0.0, 1.0, 0.0], None, &mut rng).unwrap();
        assert_eq!(adhesion.active, vec![NodeId(1)]);
        assert_eq!(adhesion.weights, array![0.0, 1.0, 0.0]);
    }

    #[test]
    fn sparse_activation_respects_fan_out() {
        let mut mesh = mesh3();
        mesh.config.fan_out = 2;
        let mut rng = StdRng::seed_from_u64(3);
        let adhesion = mesh.forward(&array![1.0, 1.0, 0.0], None, &mut rng).unwrap();
        assert_eq!(adhesion.active.len(), 2);
        assert_eq!(adhesion.weights.iter().filter(|&&w| w == 1.0).count(), 2);
        // Axis-2 node has no pull; it must stay dark.
        assert!(!adhesion.active.contains(&NodeId(2)));
    }

    #[test]
    fn residual_steers_routing() {
        let mesh = mesh3();
        let mut rng = StdRng::seed_from_u64(9);
        // Neutral payload, but the remaining gap is on axis 2 → gemini.
        let goal = StateVector(array![0.0, 0.0, 10.0]);
        let current = StateVector::zeros(3);
        let residual = Residual::between(&goal, &current);
        let adhesion = mesh.forward(&array![0.1, 0.1, 0.1], Some(&residual), &mut rng).unwrap();
        assert_eq!(adhesion.active, vec![NodeId(2)]);
    }

    #[test]
    fn topology_reconfigures_at_runtime() {
        let mut mesh = mesh3();
        assert!(mesh.remove_node(NodeId(1)));
        assert_eq!(mesh.nodes().len(), 2);
        let mut rng = StdRng::seed_from_u64(3);
        // Axis-1 payload now falls to whichever remaining node wins.
        let adhesion = mesh.forward(&array![0.0, 1.0, 0.0], None, &mut rng).unwrap();
        assert_eq!(adhesion.active.len(), 1);
        assert_ne!(adhesion.active[0], NodeId(1));
    }

    #[test]
    fn dimension_mismatch_is_an_error() {
        let mesh = mesh3();
        let mut rng = StdRng::seed_from_u64(0);
        assert!(matches!(
            mesh.forward(&array![1.0, 2.0], None, &mut rng),
            Err(MeshError::DimMismatch { .. })
        ));
    }

    #[test]
    fn empty_mesh_is_an_error() {
        let mesh = KineticNeuralMesh::new(MeshConfig { dim: 2, ..Default::default() });
        let mut rng = StdRng::seed_from_u64(0);
        assert!(matches!(mesh.forward(&array![1.0, 0.0], None, &mut rng), Err(MeshError::Empty)));
    }

    #[test]
    fn tau_annealing_is_disabled_by_default() {
        let mesh = mesh3();
        assert_eq!(mesh.effective_tau(None), mesh.config.tau);
        let goal = StateVector(array![0.0, 0.0, 10.0]);
        let residual = Residual::between(&goal, &StateVector::zeros(3));
        // tau_gain defaults to 0.0, so a residual must not change tau.
        assert_eq!(mesh.effective_tau(Some(&residual)), mesh.config.tau);
    }

    #[test]
    fn tau_annealing_scales_with_residual_and_clamps() {
        let mut mesh = mesh3();
        mesh.config.tau_gain = 1.0;
        mesh.config.tau_min = 0.05;
        mesh.config.tau_max = 2.0;

        let far = Residual::between(&StateVector(array![0.0, 0.0, 100.0]), &StateVector::zeros(3));
        assert_eq!(mesh.effective_tau(Some(&far)), 2.0, "large residual must clamp to tau_max");

        let near = Residual::between(&StateVector(array![0.0, 0.0, 0.1]), &StateVector::zeros(3));
        assert!(
            (mesh.effective_tau(Some(&near)) - 0.1).abs() < 1e-5,
            "small residual should scale down, not clamp"
        );
    }

    #[test]
    fn capacity_backpressure_routes_around_a_busy_node() {
        // Two nodes with identical, tied affinity toward the payload.
        let mut mesh = KineticNeuralMesh::new(MeshConfig {
            dim: 2,
            tau: 0.05,
            capacity_penalty: 100.0, // large enough to dominate the tie
            ..Default::default()
        });
        mesh.add_node(WorkerNode::new(0, "a", NodeKind::Llm("a".into()), vec![1.0, 0.0])).unwrap();
        mesh.add_node(WorkerNode::new(1, "b", NodeKind::Llm("b".into()), vec![1.0, 0.0])).unwrap();

        mesh.mark_active(NodeId(0));
        mesh.mark_active(NodeId(0));

        let mut rng = StdRng::seed_from_u64(1);
        let adhesion = mesh.forward(&array![1.0, 0.0], None, &mut rng).unwrap();
        assert_eq!(adhesion.active, vec![NodeId(1)], "busy node 0 must lose the tie to idle node 1");

        mesh.mark_idle(NodeId(0));
        mesh.mark_idle(NodeId(0));
        let adhesion = mesh.forward(&array![1.0, 0.0], None, &mut rng).unwrap();
        // Once idle again, node 0 is back on equal footing (no assertion on
        // which wins — just that the penalty no longer forces node 1).
        assert!(adhesion.active == vec![NodeId(0)] || adhesion.active == vec![NodeId(1)]);
    }

    #[test]
    fn topology_change_resets_in_flight_counts() {
        let mut mesh = mesh3();
        mesh.mark_active(NodeId(1));
        mesh.add_node(axis_node(3, "extra", 3, 0)).unwrap();
        // Rebuild must not panic on stale indices and must reset counts.
        mesh.mark_idle(NodeId(1)); // no-op after reset, must not underflow/panic
        assert_eq!(mesh.in_flight, vec![0, 0, 0, 0]);
    }

    fn tied_two_node_mesh(bandit_gain: f32, bandit_exploration: f32) -> KineticNeuralMesh {
        let mut mesh = KineticNeuralMesh::new(MeshConfig {
            dim: 2,
            tau: 0.05,
            bandit_gain,
            bandit_exploration,
            ..Default::default()
        });
        mesh.add_node(WorkerNode::new(0, "a", NodeKind::Llm("a".into()), vec![1.0, 0.0])).unwrap();
        mesh.add_node(WorkerNode::new(1, "b", NodeKind::Llm("b".into()), vec![1.0, 0.0])).unwrap();
        mesh
    }

    #[test]
    fn record_outcome_tracks_a_running_mean() {
        let mut mesh = tied_two_node_mesh(0.0, 0.0);
        mesh.record_outcome(NodeId(0), 1.0);
        mesh.record_outcome(NodeId(0), 0.0);
        assert!((mesh.reward_mean[0] - 0.5).abs() < 1e-6);
        mesh.record_outcome(NodeId(0), -1.5);
        assert!((mesh.reward_mean[0] - (1.0 + 0.0 - 1.5) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn record_outcome_on_unknown_node_is_a_no_op() {
        let mut mesh = tied_two_node_mesh(0.0, 0.0);
        mesh.record_outcome(NodeId(99), 1.0); // must not panic
        assert_eq!(mesh.visits, vec![0, 0]);
    }

    #[test]
    fn bandit_gain_disabled_by_default_leaves_a_tie_a_tie() {
        // With no learned signal and identical affinity, only Gumbel noise
        // decides — confirm the disabled path doesn't inject a bias either
        // way by checking the field itself is untouched by past outcomes.
        let mut mesh = tied_two_node_mesh(0.0, 0.0);
        let field_before = mesh.field(&array![1.0, 0.0], None);
        mesh.record_outcome(NodeId(0), 100.0); // would matter a lot if enabled
        let field_after = mesh.field(&array![1.0, 0.0], None);
        assert_eq!(field_before, field_after);
    }

    #[test]
    fn bandit_gain_learns_toward_the_higher_reward_node() {
        let mut mesh = tied_two_node_mesh(10.0, 0.0);
        for _ in 0..5 {
            mesh.record_outcome(NodeId(0), 1.0);
        }
        mesh.record_outcome(NodeId(1), -1.0);

        let mut rng = StdRng::seed_from_u64(1);
        let adhesion = mesh.forward(&array![1.0, 0.0], None, &mut rng).unwrap();
        assert_eq!(adhesion.active, vec![NodeId(0)], "higher observed reward must win the tie");
    }

    #[test]
    fn bandit_exploration_prioritizes_an_under_sampled_node() {
        let mut mesh = tied_two_node_mesh(0.0, 5.0);
        // Node 0 heavily sampled with a neutral reward; node 1 untouched.
        for _ in 0..100 {
            mesh.record_outcome(NodeId(0), 0.0);
        }

        let mut rng = StdRng::seed_from_u64(1);
        let adhesion = mesh.forward(&array![1.0, 0.0], None, &mut rng).unwrap();
        assert_eq!(
            adhesion.active,
            vec![NodeId(1)],
            "the never-sampled node must win on the exploration bonus alone"
        );
    }

    #[test]
    fn batch_dispatch_assigns_each_payload_to_its_clear_best_node() {
        let mesh = mesh3(); // axis-aligned nodes: codex=axis0, claude=axis1, gemini=axis2
        let payloads = vec![array![1.0, 0.0, 0.0], array![0.0, 1.0, 0.0], array![0.0, 0.0, 1.0]];
        let result = mesh.forward_batch(&payloads, 5, None).unwrap();
        assert!(result.dropped.is_empty());
        let mut by_node: Vec<_> = result.assignments;
        by_node.sort_by_key(|(id, _)| id.0);
        assert_eq!(by_node, vec![(NodeId(0), vec![0]), (NodeId(1), vec![1]), (NodeId(2), vec![2])]);
    }

    #[test]
    fn batch_dispatch_respects_capacity_and_drops_overflow() {
        // Two nodes tied toward the same direction; three payloads all
        // pulling that way, capacity 1 each — only 2 of 3 can be served.
        let mut mesh = KineticNeuralMesh::new(MeshConfig { dim: 2, ..Default::default() });
        mesh.add_node(WorkerNode::new(0, "a", NodeKind::Llm("a".into()), vec![1.0, 0.0])).unwrap();
        mesh.add_node(WorkerNode::new(1, "b", NodeKind::Llm("b".into()), vec![1.0, 0.0])).unwrap();

        let payloads = vec![array![1.0, 0.0], array![1.0, 0.0], array![1.0, 0.0]];
        let result = mesh.forward_batch(&payloads, 1, None).unwrap();

        let assigned_total: usize = result.assignments.iter().map(|(_, ps)| ps.len()).sum();
        assert_eq!(assigned_total, 2, "capacity 1 x 2 nodes must serve exactly 2 of 3 payloads");
        assert_eq!(result.dropped.len(), 1);
        for (_, ps) in &result.assignments {
            assert_eq!(ps.len(), 1, "no node may exceed its capacity");
        }
    }

    #[test]
    fn batch_dispatch_is_deterministic() {
        let mesh = mesh3();
        let payloads = vec![array![0.5, 0.5, 0.0], array![0.0, 0.5, 0.5], array![0.5, 0.0, 0.5]];
        let a = mesh.forward_batch(&payloads, 1, None).unwrap();
        let b = mesh.forward_batch(&payloads, 1, None).unwrap();
        let sorted = |r: &BatchAdhesion| {
            let mut v = r.assignments.clone();
            v.sort_by_key(|(id, _)| id.0);
            v
        };
        assert_eq!(sorted(&a), sorted(&b));
        assert_eq!(a.dropped, b.dropped);
    }

    #[test]
    fn batch_dispatch_dimension_mismatch_is_an_error() {
        let mesh = mesh3();
        assert!(matches!(
            mesh.forward_batch(&[array![1.0, 2.0]], 1, None),
            Err(MeshError::DimMismatch { .. })
        ));
    }

    #[test]
    fn batch_dispatch_on_empty_mesh_is_an_error() {
        let mesh = KineticNeuralMesh::new(MeshConfig { dim: 2, ..Default::default() });
        assert!(matches!(mesh.forward_batch(&[array![1.0, 0.0]], 1, None), Err(MeshError::Empty)));
    }

    #[test]
    fn batch_dispatch_with_no_payloads_assigns_nothing() {
        let mesh = mesh3();
        let result = mesh.forward_batch(&[], 3, None).unwrap();
        assert!(result.assignments.is_empty());
        assert!(result.dropped.is_empty());
    }

    // --- Property-based tests -------------------------------------------
    //
    // The hand-written tests above pin down specific scenarios; these sweep
    // the combinatorial config surface (residual/capacity/annealing/bandit
    // knobs interact in `field()`) that hand-picked examples can't fully
    // cover, per the counter-review step in docs/RESEARCH_BLUEPRINT.md.

    mod properties {
        use super::*;
        use proptest::prelude::*;
        use std::collections::HashSet;

        const PDIM: usize = 4;

        fn affinity_strategy() -> impl Strategy<Value = Vec<f32>> {
            prop::collection::vec(-5.0f32..5.0, PDIM)
        }

        fn build(affinities: &[Vec<f32>], config: MeshConfig) -> KineticNeuralMesh {
            let mut mesh = KineticNeuralMesh::new(config);
            for (i, affinity) in affinities.iter().enumerate() {
                mesh.add_node(WorkerNode::new(
                    i,
                    format!("n{i}"),
                    NodeKind::Llm("test".into()),
                    affinity.clone(),
                ))
                .expect("dims match by construction");
            }
            mesh
        }

        proptest! {
            #[test]
            fn forward_always_activates_exactly_fan_out_distinct_nodes(
                affinities in prop::collection::vec(affinity_strategy(), 1..=6),
                payload in affinity_strategy(),
                fan_out_raw in 1usize..=8,
                tau in 0.01f32..2.0,
                tau_gain in 0.0f32..2.0,
                capacity_penalty in 0.0f32..5.0,
                bandit_gain in -2.0f32..2.0,
                bandit_exploration in 0.0f32..2.0,
                use_residual in any::<bool>(),
                goal_raw in affinity_strategy(),
                seed in any::<u64>(),
            ) {
                let n_nodes = affinities.len();
                let config = MeshConfig {
                    dim: PDIM,
                    tau,
                    hard: true,
                    fan_out: fan_out_raw,
                    residual_gain: 1.0,
                    tau_gain,
                    tau_min: 0.05,
                    tau_max: 2.0,
                    capacity_penalty,
                    bandit_gain,
                    bandit_exploration,
                };
                let mesh = build(&affinities, config);
                let payload_arr = Array1::from_vec(payload);
                let residual = if use_residual {
                    let goal = StateVector(Array1::from_vec(goal_raw));
                    Some(Residual::between(&goal, &StateVector::zeros(PDIM)))
                } else {
                    None
                };
                let mut rng = StdRng::seed_from_u64(seed);
                let adhesion = mesh.forward(&payload_arr, residual.as_ref(), &mut rng).unwrap();

                let expected_active = fan_out_raw.min(n_nodes).max(1);
                prop_assert_eq!(adhesion.active.len(), expected_active);

                // No duplicate winners across fan-out slots.
                let mut seen = HashSet::new();
                for id in &adhesion.active {
                    prop_assert!(seen.insert(*id), "duplicate winner {:?}", id);
                }

                // Hard routing: exactly `expected_active` weights are 1.0,
                // every other weight is exactly 0.0.
                let ones = adhesion.weights.iter().filter(|&&w| w == 1.0).count();
                let zeros = adhesion.weights.iter().filter(|&&w| w == 0.0).count();
                prop_assert_eq!(ones, expected_active);
                prop_assert_eq!(ones + zeros, n_nodes);
            }

            #[test]
            fn mark_and_record_sequences_never_panic_or_underflow(
                affinities in prop::collection::vec(affinity_strategy(), 1..=4),
                ops in prop::collection::vec((0usize..4, any::<bool>(), -3.0f32..3.0), 0..50),
            ) {
                let n_nodes = affinities.len();
                let mut mesh = build(&affinities, MeshConfig { dim: PDIM, ..Default::default() });
                for (idx, is_active, reward) in ops {
                    let id = NodeId(idx % n_nodes);
                    if is_active {
                        mesh.mark_active(id);
                    } else {
                        mesh.mark_idle(id); // must be a no-op, never panic, even if not active
                    }
                    mesh.record_outcome(id, reward);
                }
                let payload = Array1::zeros(PDIM);
                let mut rng = StdRng::seed_from_u64(0);
                prop_assert!(mesh.forward(&payload, None, &mut rng).is_ok());
            }

            #[test]
            fn batch_dispatch_never_exceeds_capacity_and_partitions_payloads(
                affinities in prop::collection::vec(affinity_strategy(), 1..=5),
                payloads in prop::collection::vec(affinity_strategy(), 0..8),
                capacity in 1usize..5,
            ) {
                let mesh = build(&affinities, MeshConfig { dim: PDIM, ..Default::default() });
                let payload_arrs: Vec<Array1<f32>> = payloads.into_iter().map(Array1::from_vec).collect();
                let result = mesh.forward_batch(&payload_arrs, capacity, None).unwrap();

                for (_, ps) in &result.assignments {
                    prop_assert!(!ps.is_empty());
                    prop_assert!(ps.len() <= capacity);
                }

                // Every payload index appears in exactly one place: some
                // node's assignment list, or dropped — never both, never
                // neither, never twice.
                let mut seen = vec![0u8; payload_arrs.len()];
                for (_, ps) in &result.assignments {
                    for &p in ps {
                        seen[p] += 1;
                    }
                }
                for &p in &result.dropped {
                    seen[p] += 1;
                }
                prop_assert!(seen.iter().all(|&c| c == 1), "every payload must be covered exactly once");
            }
        }
    }
}
