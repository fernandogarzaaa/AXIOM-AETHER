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

/// Configuration for the mesh's routing behavior.
#[derive(Debug, Clone)]
pub struct MeshConfig {
    /// Embedding dimension shared by payloads and node affinities.
    pub dim: usize,
    /// Gumbel-Softmax temperature.
    pub tau: f32,
    /// Hard (discrete one-hot) routing. The KNM spec requires the hard
    /// variant; soft is kept for diagnostics.
    pub hard: bool,
    /// How many nodes a single payload may snap to (top-k fan-out).
    pub fan_out: usize,
    /// Gain applied to residual alignment when a residual is provided.
    /// 0.0 disables residual modulation.
    pub residual_gain: f32,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self { dim: 64, tau: 0.5, hard: true, fan_out: 1, residual_gain: 1.0 }
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
}

impl KineticNeuralMesh {
    pub fn new(config: MeshConfig) -> Self {
        Self {
            affinity: Array2::zeros((0, config.dim)),
            bias: Array1::zeros(0),
            nodes: Vec::new(),
            config,
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
    }

    /// Compute the gravitational field a payload projects over the mesh:
    /// affinity dot-products plus per-node bias, plus (optionally) a term
    /// rewarding nodes aligned with the current residual direction.
    fn field(&self, payload: &Array1<f32>, residual: Option<&Residual>) -> Array1<f32> {
        let mut logits = self.affinity.dot(payload) + &self.bias;
        if let Some(r) = residual {
            if self.config.residual_gain != 0.0 && r.norm() > 0.0 {
                let direction = &r.vector / r.norm();
                logits = logits + self.config.residual_gain * self.affinity.dot(&direction);
            }
        }
        logits
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

        // Top-k discrete routing: draw a hard Gumbel-Softmax winner, mask
        // it out, and redraw — each slot is an independent discrete snap,
        // so fan_out=1 is exactly the classic hard Gumbel-Softmax.
        let mut weights: Array1<f32> = Array1::zeros(self.nodes.len());
        let mut active: Vec<NodeId> = Vec::with_capacity(fan_out);
        let mut masked = field.clone();
        for _ in 0..fan_out {
            let sample = gumbel_softmax(&masked, self.config.tau, self.config.hard, rng);
            let w = sample.winner;
            weights[w] = if self.config.hard { 1.0 } else { sample.adhesion[w] };
            active.push(self.nodes[w].id);
            masked[w] = f32::NEG_INFINITY;
        }

        Ok(Adhesion { weights, active, field })
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
}
