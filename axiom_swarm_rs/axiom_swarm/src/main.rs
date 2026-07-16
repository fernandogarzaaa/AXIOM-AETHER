//! Axiom Prime — the swarm orchestrator binary.
//!
//! Wires the three layers together and runs a closed-loop demo tick:
//!
//! * `axiom_core` — Kinetic Neural Mesh routing + IDC control law
//! * `axiom_mcp`  — Mini Aether sidecars conditioning worker payloads
//! * this binary  — the non-blocking FSM runner (async worker calls are
//!   tokio tasks whose completions come back as FSM events)
//!
//! Workers here are simulated: each "LLM call" is an async task that moves
//! the environment state a step toward the goal, standing in for a real
//! backend behind its sidecar. The control loop, routing, and payload
//! conditioning are the real implementations.

mod fsm;

use std::sync::Arc;

use ndarray::Array1;
use rand::{rngs::StdRng, SeedableRng};
use tokio::sync::mpsc;

use axiom_core::idc::IdcController;
use axiom_core::mesh::{KineticNeuralMesh, MeshConfig};
use axiom_core::node::{NodeId, NodeKind, WorkerNode};
use axiom_core::residual::StateVector;
use axiom_mcp::sidecar::MiniAetherSidecar;

use fsm::{SwarmCommand, SwarmEvent, SwarmFsm};

const DIM: usize = 8;
const EPSILON: f32 = 0.05;

/// Build a demo mesh with three LLM workers, each with a distinct affinity
/// profile over the state space.
fn build_mesh(rng: &mut StdRng) -> KineticNeuralMesh {
    use rand::Rng;
    let mut mesh = KineticNeuralMesh::new(MeshConfig { dim: DIM, tau: 0.3, ..Default::default() });
    for (i, name) in ["codex", "claude", "gemini"].iter().enumerate() {
        let affinity: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
        mesh.add_node(WorkerNode::new(i, *name, NodeKind::Llm(name.to_string()), affinity))
            .expect("affinity dim matches mesh dim");
    }
    mesh
}

/// Simulated environment: the mutable "current state" that worker actions
/// push toward the goal.
struct Environment {
    current: StateVector,
}

/// A simulated worker call: an async task that closes 40% of the gap along
/// the residual, then reports completion. In production this is a real
/// backend call routed through the node's Mini Aether sidecar.
async fn run_worker(
    node: NodeId,
    name: String,
    payload: Arc<str>,
    residual: Array1<f32>,
    tx: mpsc::Sender<(NodeId, Array1<f32>)>,
) {
    tokio::task::yield_now().await; // stand-in for network latency
    println!("    [worker:{name}] received {} bytes of conditioned payload", payload.len());
    let delta = residual * 0.4;
    let _ = tx.send((node, delta)).await;
}

#[tokio::main]
async fn main() {
    let mut rng = StdRng::seed_from_u64(2026);

    let mesh = build_mesh(&mut rng);
    let sidecars: Vec<MiniAetherSidecar> =
        mesh.nodes().iter().map(|n| MiniAetherSidecar::standard(n.name.clone())).collect();

    // Goal: an arbitrary target point in state space. In production this is
    // the fused encoding of "tests green, diff applied, exit code 0".
    let goal = StateVector(Array1::from_vec(vec![1.0, -0.5, 0.8, 0.0, 0.3, -0.2, 0.6, 0.1]));
    let controller = IdcController::new(goal, EPSILON);
    let mut env = Environment { current: StateVector::zeros(DIM) };

    let mut machine = SwarmFsm::new();
    let (tx, mut rx) = mpsc::channel::<(NodeId, Array1<f32>)>(16);

    println!("axiom_swarm: IDC control loop starting (dim={DIM}, epsilon={EPSILON})");
    let mut queue = machine.step(SwarmEvent::GoalLoaded);
    let mut tick = 0usize;

    while !machine.is_terminal() {
        let Some(command) = queue.pop() else {
            // No pending commands: we're awaiting workers. Their
            // completions are the only thing that can advance the FSM.
            let Some((node, delta)) = rx.recv().await else { break };
            println!("    [prime] worker node {} landed its correction", node.0);
            env.current = StateVector(&env.current.0 + &delta);
            queue.extend(machine.step(SwarmEvent::WorkerDone));
            continue;
        };

        match command {
            SwarmCommand::FuseSensors => {
                tick += 1;
                let residual = controller.residual(&env.current);
                println!("[tick {tick}] residual norm = {:.4}", residual.norm());
                queue.extend(
                    machine.step(SwarmEvent::Sensed { converged: residual.converged(EPSILON) }),
                );
            }
            SwarmCommand::RouteResidual => {
                let residual = controller.residual(&env.current);
                // Payload embedding: the residual direction itself — route
                // toward whoever best matches the remaining gap.
                let payload_embedding = residual.vector.clone() / residual.norm().max(1e-6);
                let adhesion = match mesh.forward(&payload_embedding, Some(&residual), &mut rng) {
                    Ok(a) => a,
                    Err(e) => {
                        queue.extend(machine.step(SwarmEvent::Fault { reason: e.to_string() }));
                        continue;
                    }
                };
                let names: Vec<&str> = adhesion
                    .active
                    .iter()
                    .filter_map(|id| mesh.node(*id).map(|n| n.name.as_str()))
                    .collect();
                println!("    [mesh] payload snapped to {names:?}");
                queue.extend(machine.step(SwarmEvent::Routed { nodes: adhesion.active }));
            }
            SwarmCommand::Dispatch { nodes } => {
                let residual = controller.residual(&env.current);
                let correction = controller.actuate(&residual, Some("cargo test"));
                let raw = format!(
                    "Sure, here's some context!\n<axiom-internal> tick={tick}\nintent: {:?}\nresidual norm: {:.4}",
                    correction.actions[0], correction.residual_norm
                );
                for id in nodes {
                    let node = mesh.node(id).expect("routed node exists");
                    let sidecar = &sidecars[id.0];
                    let payload = sidecar.condition(&raw);
                    tokio::spawn(run_worker(
                        id,
                        node.name.clone(),
                        Arc::clone(&payload.content),
                        residual.vector.clone(),
                        tx.clone(),
                    ));
                }
            }
            SwarmCommand::AnnounceConverged => {
                println!("[done] converged in {tick} ticks — residual within epsilon");
            }
            SwarmCommand::AnnounceHalt { reason } => {
                eprintln!("[halt] {reason}");
            }
        }
    }
    println!("axiom_swarm: terminal state = {:?}", machine.state());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_mesh_has_three_workers() {
        let mut rng = StdRng::seed_from_u64(2026);
        let mesh = build_mesh(&mut rng);
        assert_eq!(mesh.nodes().len(), 3);
    }

    #[test]
    fn residual_shrinks_under_worker_delta() {
        let goal = StateVector(Array1::from_elem(DIM, 1.0));
        let controller = IdcController::new(goal, EPSILON);
        let mut current = StateVector::zeros(DIM);
        let before = controller.residual(&current).norm();
        let delta = controller.residual(&current).vector * 0.4;
        current = StateVector(&current.0 + &delta);
        assert!(controller.residual(&current).norm() < before);
    }
}
