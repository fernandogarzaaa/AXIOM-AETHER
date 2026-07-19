//! Axiom Prime — the swarm orchestrator binary.
//!
//! Wires the three layers together and runs a closed-loop demo tick:
//!
//! * `axiom_core` — Kinetic Neural Mesh routing + IDC control law
//! * `axiom_mcp`  — Mini Aether sidecars conditioning worker payloads, plus
//!   the real JSON-RPC-over-stdio transport to worker processes
//! * this binary  — the non-blocking FSM runner (async worker calls are
//!   tokio tasks whose completions come back as FSM events)
//!
//! Two of the three demo workers ("codex", "claude") are dispatched to a
//! real child process (`aether_worker`) over the stdio wire protocol. The
//! third ("gemini") uses a transport that always fails, to exercise the
//! timeout/quarantine path: after enough consecutive failures, Axiom Prime
//! removes it from the mesh and keeps converging on the remaining nodes —
//! sparse activation and fault tolerance working together. The environment
//! dynamics that move the state toward the goal are still simulated (a
//! real backend would report actual command/test output instead).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ndarray::Array1;
use rand::{rngs::StdRng, SeedableRng};
use tokio::sync::mpsc;

use axiom_core::idc::IdcController;
use axiom_core::mesh::{KineticNeuralMesh, MeshConfig};
use axiom_core::node::{NodeId, NodeKind, WorkerNode};
use axiom_core::residual::StateVector;
use axiom_mcp::sidecar::MiniAetherSidecar;
use axiom_mcp::{DispatchParams, MockTransport, StdioTransport, WorkerTransport};

use axiom_swarm::fsm::{SwarmCommand, SwarmEvent, SwarmFsm};
use axiom_swarm::health::NodeHealth;

const DIM: usize = 8;
const EPSILON: f32 = 0.05;
/// Consecutive dispatch failures before a node is quarantined (removed
/// from the mesh). See `health::NodeHealth`.
const QUARANTINE_THRESHOLD: u32 = 2;
/// Per-dispatch timeout for the real stdio transport.
const WORKER_TIMEOUT: Duration = Duration::from_secs(2);

/// Build a demo mesh with three LLM workers, with fixed (not randomly
/// drawn) affinities chosen to make the demo's narrative deterministic
/// rather than a function of which way Gumbel noise happens to break ties
/// this run. Randomly-drawn affinities were tried first; they made
/// whether the quarantine path even fired a matter of luck — a coincidence
/// this demo shouldn't depend on to demonstrate a feature it specifically
/// exists to exercise.
///
/// `gemini`'s affinity is set to closely track `goal`'s own direction, so
/// it wins the very first routing decision before any annealing/bandit
/// learning has had a chance to shift things — and, since its dispatch
/// always fails without changing the environment, the residual direction
/// (and therefore the routing decision) stays essentially unchanged until
/// it's quarantined. `claude`'s affinity tracks the same direction at
/// lower magnitude, so it becomes the clear winner immediately afterward;
/// `codex`'s affinity points at a state axis with zero weight in `goal`,
/// so it's a legitimate option the mesh consistently has no reason to
/// pick — not a mistake, just realistic "not every worker fits every
/// task."
fn build_mesh(goal: &Array1<f32>) -> KineticNeuralMesh {
    let mut mesh = KineticNeuralMesh::new(MeshConfig {
        dim: DIM,
        tau: 0.3,
        // Anneal temperature with the residual: explore more early,
        // sharpen as the controller nears the goal (see MeshConfig docs).
        tau_gain: 0.25,
        tau_min: 0.05,
        tau_max: 1.2,
        // Soft backpressure: a node with dispatches already in flight
        // becomes less attractive to new ones (mirrors MoE expert-capacity
        // constraints).
        capacity_penalty: 0.5,
        // Online-learned routing: nodes that empirically shrink the
        // residual more get preferred over time, tempered by a UCB1
        // exploration bonus so a quarantined-then-recovered or newly
        // added node still gets tried.
        bandit_gain: 0.6,
        bandit_exploration: 0.3,
        ..Default::default()
    });

    let mut codex_affinity = vec![0.0; DIM];
    codex_affinity[3] = 3.0; // goal[3] == 0.0: a real axis, deliberately not this task's

    let claude_affinity: Vec<f32> = goal.iter().map(|g| g * 2.5).collect();
    let gemini_affinity: Vec<f32> = goal.iter().map(|g| g * 3.0).collect();

    let workers = [("codex", codex_affinity), ("claude", claude_affinity), ("gemini", gemini_affinity)];
    for (i, (name, affinity)) in workers.into_iter().enumerate() {
        mesh.add_node(WorkerNode::new(i, name, NodeKind::Llm(name.to_string()), affinity))
            .expect("affinity dim matches mesh dim");
    }
    mesh
}

/// Resolve the `aether_worker` binary as a sibling of the running
/// `axiom_swarm` executable (both live in the same workspace `target/`
/// directory), building it on demand if this is a fresh checkout.
fn worker_binary_path() -> PathBuf {
    let mut path = std::env::current_exe().expect("resolve current executable path");
    path.pop();
    path.push(if cfg!(windows) { "aether_worker.exe" } else { "aether_worker" });
    if !path.exists() {
        println!("axiom_swarm: building aether_worker (first run)...");
        let status = std::process::Command::new(env!("CARGO"))
            .args(["build", "--bin", "aether_worker"])
            .status()
            .expect("invoke cargo to build aether_worker");
        assert!(status.success(), "cargo build --bin aether_worker failed");
    }
    path
}

/// Simulated environment: the mutable "current state" that worker actions
/// push toward the goal.
struct Environment {
    current: StateVector,
}

/// What a dispatch resolved to, fed back into the FSM's event queue.
enum DispatchOutcome {
    Success { output: String },
    Failure { reason: String },
}

/// Run one dispatch against a worker's transport and report the outcome.
/// In production the transport reaches a real backend; here two nodes
/// reach the reference `aether_worker` process and one is wired to always
/// fail, to exercise quarantine.
async fn run_worker(
    node: NodeId,
    transport: Arc<dyn WorkerTransport>,
    params: DispatchParams,
    tx: mpsc::Sender<(NodeId, DispatchOutcome)>,
) {
    let outcome = match transport.dispatch(params).await {
        Ok(result) => DispatchOutcome::Success { output: result.output },
        Err(e) => DispatchOutcome::Failure { reason: e.to_string() },
    };
    let _ = tx.send((node, outcome)).await;
}

#[tokio::main]
async fn main() {
    let mut rng = StdRng::seed_from_u64(2026);

    // Goal: an arbitrary target point in state space. In production this is
    // the fused encoding of "tests green, diff applied, exit code 0".
    let goal_vec = Array1::from_vec(vec![1.0, -0.5, 0.8, 0.0, 0.3, -0.2, 0.6, 0.1]);
    let mut mesh = build_mesh(&goal_vec);
    let sidecars: Vec<MiniAetherSidecar> =
        mesh.nodes().iter().map(|n| MiniAetherSidecar::standard(n.name.clone())).collect();

    let worker_bin = worker_binary_path();
    let transports: Vec<Arc<dyn WorkerTransport>> = mesh
        .nodes()
        .iter()
        .map(|n| -> Arc<dyn WorkerTransport> {
            if n.name == "gemini" {
                // Simulates a backend outage: every dispatch fails, so the
                // demo exercises NodeHealth quarantine end to end.
                Arc::new(MockTransport { fail_with: Some("simulated backend outage".into()) })
            } else {
                Arc::new(
                    StdioTransport::spawn_with_timeout(&worker_bin, &[], WORKER_TIMEOUT)
                        .expect("spawn aether_worker"),
                )
            }
        })
        .collect();

    let controller = IdcController::new(StateVector(goal_vec), EPSILON);
    let mut env = Environment { current: StateVector::zeros(DIM) };
    let mut health = NodeHealth::new(QUARANTINE_THRESHOLD);

    let mut machine = SwarmFsm::new();
    let (tx, mut rx) = mpsc::channel::<(NodeId, DispatchOutcome)>(16);

    println!("axiom_swarm: IDC control loop starting (dim={DIM}, epsilon={EPSILON})");
    let mut queue = machine.step(SwarmEvent::GoalLoaded);
    let mut tick = 0usize;

    while !machine.is_terminal() {
        let Some(command) = queue.pop() else {
            // No pending commands: we're awaiting workers. Their
            // completions are the only thing that can advance the FSM.
            let Some((node, outcome)) = rx.recv().await else { break };
            mesh.mark_idle(node);
            match outcome {
                DispatchOutcome::Success { output } => {
                    println!("    [prime] worker {} -> {output}", node.0);
                    health.record_success(node);
                    let before_norm = controller.residual(&env.current).norm();
                    let delta = &controller.residual(&env.current).vector * 0.4;
                    env.current = StateVector(&env.current.0 + &delta);
                    let after_norm = controller.residual(&env.current).norm();
                    // Marginal credit: reward is the shrinkage attributable
                    // to this dispatch, measured at the moment it landed —
                    // see KineticNeuralMesh::record_outcome for why this
                    // stays a fair signal even under concurrent fan-out.
                    mesh.record_outcome(node, before_norm - after_norm);
                }
                DispatchOutcome::Failure { reason } => {
                    eprintln!("    [prime] worker {} failed: {reason}", node.0);
                    if health.record_failure(node) {
                        eprintln!(
                            "    [prime] quarantining worker {} after {QUARANTINE_THRESHOLD} consecutive failures",
                            node.0
                        );
                        mesh.remove_node(node);
                    }
                }
            }
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
                    let name = mesh.node(id).expect("routed node exists").name.clone();
                    let sidecar = &sidecars[id.0];
                    let payload = sidecar.condition(&raw);
                    let params = DispatchParams {
                        worker: name,
                        payload: payload.content.to_string(),
                        residual_norm: residual.norm(),
                    };
                    mesh.mark_active(id);
                    tokio::spawn(run_worker(id, Arc::clone(&transports[id.0]), params, tx.clone()));
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
        let goal = Array1::from_elem(DIM, 1.0);
        let mesh = build_mesh(&goal);
        assert_eq!(mesh.nodes().len(), 3);
    }

    #[test]
    fn demo_mesh_routes_the_initial_residual_to_gemini_first() {
        // Pins the deterministic-by-design narrative build_mesh documents:
        // gemini's affinity tracks `goal` most closely, so it must win the
        // very first routing decision, before any learning has occurred.
        let goal = Array1::from_vec(vec![1.0, -0.5, 0.8, 0.0, 0.3, -0.2, 0.6, 0.1]);
        let mesh = build_mesh(&goal);
        let normalized = &goal / goal.mapv(|x: f32| x * x).sum().sqrt();
        let mut rng = StdRng::seed_from_u64(2026);
        let adhesion = mesh.forward(&normalized, None, &mut rng).unwrap();
        let winner = mesh.node(adhesion.active[0]).unwrap();
        assert_eq!(winner.name, "gemini");
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
