//! Hierarchical topology: a supervisor mesh over regions, each an
//! independently-addressable `KineticNeuralMesh` — and, when driven the
//! same way the flat demo drives its own mesh, its own `SwarmFsm` loop.
//! This is the "hierarchical" leg of the centralized/decentralized/
//! hierarchical (+ dynamic-adaptive) taxonomy from the 2025-2026 multi-
//! agent orchestration survey cited in docs/RESEARCH_BLUEPRINT.md.
//!
//! `SwarmSupervisor` deliberately owns only the region-*selection* layer:
//! given a payload and residual, which region's Axiom Prime should own
//! this tick. What a region does once selected — its own mesh routing,
//! its own `SwarmFsm` loop, its own workers — is exactly the machinery
//! `main.rs` already runs for the flat, single-region case. A region is
//! not a new abstraction bolted onto `axiom_core`; it's a whole
//! `KineticNeuralMesh`, the same type the flat demo uses, deployed as one
//! node of the supervisor's own mesh. Hierarchy falls out of composing
//! the existing mesh with itself one level up, rather than making
//! `WorkerNode`/`NodeKind` recursive — that would have required new
//! derives (`Clone`/serde) on a mesh containing `ndarray` buffers and a
//! guard against a region containing itself, for no benefit this
//! composition doesn't already give.

use axiom_core::mesh::{Adhesion, KineticNeuralMesh, MeshConfig, MeshError};
use axiom_core::node::{NodeId, NodeKind, WorkerNode};
use axiom_core::residual::Residual;
use ndarray::Array1;
use rand::Rng;

/// One region: a name plus its own leaf-worker mesh. Opaque to the
/// supervisor beyond routing — a region is free to run its own `SwarmFsm`
/// over its own mesh exactly like the top-level demo does.
pub struct Region {
    pub name: String,
    pub mesh: KineticNeuralMesh,
}

/// Routes a payload to a region first, then leaves per-region routing to
/// that region's own mesh. Regions are appended, never removed — there's
/// no `remove_region`, so `regions()` indices stay aligned with the
/// underlying region-selection mesh's `NodeId`s for the supervisor's
/// lifetime (removal would need to keep the two in sync, a real design
/// question left for when a caller actually needs to retire a region).
pub struct SwarmSupervisor {
    /// A mesh over regions: each "node" here is a whole region, its
    /// affinity summarizing what kind of work that region is good at.
    region_mesh: KineticNeuralMesh,
    regions: Vec<Region>,
}

impl SwarmSupervisor {
    pub fn new(config: MeshConfig) -> Self {
        Self { region_mesh: KineticNeuralMesh::new(config), regions: Vec::new() }
    }

    /// Register a region with a routing affinity in the supervisor's own
    /// space (which may differ in dimension from any region's internal
    /// leaf-worker mesh — the supervisor routes on "what kind of work",
    /// each region routes on "which worker").
    pub fn add_region(
        &mut self,
        name: impl Into<String>,
        affinity: Vec<f32>,
        mesh: KineticNeuralMesh,
    ) -> Result<NodeId, MeshError> {
        let name = name.into();
        let id = self.regions.len();
        let region_id =
            self.region_mesh.add_node(WorkerNode::new(id, name.clone(), NodeKind::Tool("region".into()), affinity))?;
        self.regions.push(Region { name, mesh });
        Ok(region_id)
    }

    /// Pick the primary region for this payload/residual — the same
    /// gravitational-field + Gumbel-Softmax adhesion a leaf mesh uses to
    /// pick a worker, one level up. If the supervisor's own `fan_out > 1`,
    /// `adhesion.active` carries additional regions beyond the primary
    /// (e.g. for dispatching the same payload class to two regions in
    /// parallel); this always returns the top-ranked one.
    pub fn select_region(
        &self,
        payload: &Array1<f32>,
        residual: Option<&Residual>,
        rng: &mut impl Rng,
    ) -> Result<(&Region, Adhesion), MeshError> {
        let adhesion = self.region_mesh.forward(payload, residual, rng)?;
        let region = &self.regions[adhesion.active[0].0];
        Ok((region, adhesion))
    }

    pub fn region(&self, name: &str) -> Option<&Region> {
        self.regions.iter().find(|r| r.name == name)
    }

    pub fn region_mut(&mut self, name: &str) -> Option<&mut Region> {
        self.regions.iter_mut().find(|r| r.name == name)
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsm::{SwarmCommand, SwarmEvent, SwarmFsm};
    use axiom_core::residual::StateVector;
    use ndarray::array;
    use rand::{rngs::StdRng, SeedableRng};

    fn leaf_mesh(dim: usize, names: &[&str]) -> KineticNeuralMesh {
        let mut mesh = KineticNeuralMesh::new(MeshConfig { dim, tau: 0.1, ..Default::default() });
        for (i, name) in names.iter().enumerate() {
            let mut affinity = vec![0.0; dim];
            affinity[i % dim] = 5.0;
            mesh.add_node(WorkerNode::new(i, *name, NodeKind::Llm(name.to_string()), affinity)).unwrap();
        }
        mesh
    }

    #[test]
    fn select_region_picks_the_best_affinity_region() {
        let mut supervisor = SwarmSupervisor::new(MeshConfig { dim: 2, tau: 0.05, ..Default::default() });
        supervisor.add_region("codegen", vec![1.0, 0.0], leaf_mesh(3, &["codex-1", "codex-2"])).unwrap();
        supervisor.add_region("research", vec![0.0, 1.0], leaf_mesh(3, &["gemini-1"])).unwrap();

        let mut rng = StdRng::seed_from_u64(3);
        let (region, _) = supervisor.select_region(&array![0.0, 1.0], None, &mut rng).unwrap();
        assert_eq!(region.name, "research");
    }

    #[test]
    fn each_region_keeps_its_own_independently_addressable_mesh() {
        let mut supervisor = SwarmSupervisor::new(MeshConfig { dim: 2, ..Default::default() });
        supervisor.add_region("a", vec![1.0, 0.0], leaf_mesh(3, &["w1", "w2"])).unwrap();
        supervisor.add_region("b", vec![0.0, 1.0], leaf_mesh(3, &["w3"])).unwrap();

        assert_eq!(supervisor.region("a").unwrap().mesh.nodes().len(), 2);
        assert_eq!(supervisor.region("b").unwrap().mesh.nodes().len(), 1);

        // Regions are genuinely independent meshes: reconfiguring one's
        // topology must not affect the other.
        let region_a = supervisor.region_mut("a").unwrap();
        let w1 = region_a.mesh.nodes()[0].id;
        assert!(region_a.mesh.remove_node(w1));
        assert_eq!(supervisor.region("a").unwrap().mesh.nodes().len(), 1);
        assert_eq!(supervisor.region("b").unwrap().mesh.nodes().len(), 1);
    }

    #[test]
    fn empty_supervisor_is_an_error() {
        let supervisor = SwarmSupervisor::new(MeshConfig { dim: 2, ..Default::default() });
        let mut rng = StdRng::seed_from_u64(0);
        assert!(matches!(
            supervisor.select_region(&array![1.0, 0.0], None, &mut rng),
            Err(MeshError::Empty)
        ));
    }

    /// Demonstrates "sub-swarms with their own Axiom Prime": two regions,
    /// each driven by its own independent `SwarmFsm` over its own mesh,
    /// with the supervisor only deciding which region owns an incoming
    /// tick — exactly the hierarchical composition this module targets,
    /// built from the unmodified flat-demo machinery run twice.
    #[test]
    fn regions_run_independent_swarm_fsm_loops() {
        let mut supervisor = SwarmSupervisor::new(MeshConfig { dim: 2, tau: 0.1, ..Default::default() });
        supervisor.add_region("region-a", vec![1.0, 0.0], leaf_mesh(2, &["a-worker"])).unwrap();
        supervisor.add_region("region-b", vec![0.0, 1.0], leaf_mesh(2, &["b-worker"])).unwrap();

        let mut rng = StdRng::seed_from_u64(5);
        let (region, _) = supervisor.select_region(&array![1.0, 0.0], None, &mut rng).unwrap();
        assert_eq!(region.name, "region-a");

        // The selected region now runs its own Axiom Prime loop, entirely
        // independent of the supervisor and of the other region's FSM.
        let mut region_fsm = SwarmFsm::new();
        let mut cmds = region_fsm.step(SwarmEvent::GoalLoaded);
        assert_eq!(cmds, vec![SwarmCommand::FuseSensors]);

        cmds = region_fsm.step(SwarmEvent::Sensed { converged: false });
        assert_eq!(cmds, vec![SwarmCommand::RouteResidual]);

        let worker_mesh = &supervisor.region("region-a").unwrap().mesh;
        let goal = StateVector(array![0.0, 0.0]);
        let residual = Residual::between(&goal, &StateVector::zeros(2));
        let adhesion = worker_mesh.forward(&array![1.0, 0.0], Some(&residual), &mut rng).unwrap();

        cmds = region_fsm.step(SwarmEvent::Routed { nodes: adhesion.active });
        assert!(matches!(cmds[0], SwarmCommand::Dispatch { .. }));

        cmds = region_fsm.step(SwarmEvent::WorkerDone);
        assert_eq!(cmds, vec![SwarmCommand::FuseSensors]);
        cmds = region_fsm.step(SwarmEvent::Sensed { converged: true });
        assert_eq!(cmds, vec![SwarmCommand::AnnounceConverged]);
        assert!(region_fsm.is_terminal());

        // A second, entirely separate SwarmFsm for region-b never touched
        // any of the above state — independence, not shared mutable state.
        let region_b_fsm = SwarmFsm::new();
        assert!(!region_b_fsm.is_terminal());
    }

    mod properties {
        use super::*;
        use proptest::prelude::*;

        fn affinity(dim: usize) -> impl Strategy<Value = Vec<f32>> {
            prop::collection::vec(-5.0f32..5.0, dim)
        }

        proptest! {
            #[test]
            fn select_region_always_returns_a_registered_region(
                region_affinities in prop::collection::vec(affinity(3), 1..=6),
                payload in affinity(3),
                seed in any::<u64>(),
            ) {
                let mut supervisor = SwarmSupervisor::new(MeshConfig { dim: 3, tau: 0.2, ..Default::default() });
                let names: Vec<String> = (0..region_affinities.len()).map(|i| format!("region-{i}")).collect();
                for (name, aff) in names.iter().zip(region_affinities) {
                    supervisor.add_region(name.clone(), aff, leaf_mesh(2, &["leaf"])).unwrap();
                }
                let mut rng = StdRng::seed_from_u64(seed);
                let (region, adhesion) = supervisor
                    .select_region(&Array1::from_vec(payload), None, &mut rng)
                    .unwrap();
                prop_assert!(names.contains(&region.name));
                prop_assert_eq!(adhesion.active.len(), 1);
            }
        }
    }
}
