//! Demonstrates `KineticNeuralMesh::forward_batch` (expert-choice batch
//! dispatch) against the naive per-payload alternative, to make the load-
//! balancing win from docs/RESEARCH_BLUEPRINT.md concrete rather than just
//! asserted in a unit test.
//!
//! Run with: `cargo run -p axiom_core --example batch_dispatch`

use axiom_core::mesh::{KineticNeuralMesh, MeshConfig};
use axiom_core::node::{NodeId, NodeKind, WorkerNode};
use ndarray::{array, Array1};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;

fn main() {
    const DIM: usize = 2;

    // Worker A is the clearly better fit for this whole batch — the
    // textbook MoE motivation: naive per-payload routing doesn't get this
    // "wrong" (worker A genuinely is the best pick for each payload taken
    // alone), it's *right* every time, which is exactly the problem — a
    // popular node gets flooded precisely because it's the correct choice
    // for many payloads at once, with nothing to notice or care that it's
    // already full.
    let mut mesh = KineticNeuralMesh::new(MeshConfig { dim: DIM, tau: 0.1, ..Default::default() });
    mesh.add_node(WorkerNode::new(0, "worker-a", NodeKind::Llm("a".into()), vec![1.0, 0.0])).unwrap();
    mesh.add_node(WorkerNode::new(1, "worker-b", NodeKind::Llm("b".into()), vec![0.3, 0.0])).unwrap();

    // Six payloads all pulling toward worker A's specialty.
    let payloads: Vec<Array1<f32>> = vec![
        array![1.0, 0.0],
        array![0.9, 0.1],
        array![1.0, 0.05],
        array![0.85, 0.0],
        array![0.95, 0.05],
        array![1.0, 0.1],
    ];

    println!("== naive per-payload routing (forward), no capacity awareness ==");
    let mut rng = StdRng::seed_from_u64(7);
    let mut naive_counts: HashMap<NodeId, usize> = HashMap::new();
    for (i, payload) in payloads.iter().enumerate() {
        let adhesion = mesh.forward(payload, None, &mut rng).unwrap();
        let winner = adhesion.active[0];
        *naive_counts.entry(winner).or_default() += 1;
        println!("  payload {i} -> {winner:?}");
    }
    println!("  totals: {naive_counts:?}\n");

    println!("== expert-choice batch dispatch (forward_batch), capacity = 3 ==");
    let result = mesh.forward_batch(&payloads, 3, None).unwrap();
    let mut batch_counts: HashMap<NodeId, usize> = HashMap::new();
    for (node, indices) in &result.assignments {
        batch_counts.insert(*node, indices.len());
        println!("  {node:?} claimed payloads {indices:?}");
    }
    if !result.dropped.is_empty() {
        println!("  dropped (no node had capacity): {:?}", result.dropped);
    }
    println!("  totals: {batch_counts:?}\n");

    println!(
        "Naive per-payload routing isn't wrong about any single payload —\n\
         worker-a really is the better fit each time — but it has no\n\
         mechanism to notice it's already full, so it floods worker-a and\n\
         leaves worker-b idle. Capacity-bounded batch dispatch enforces\n\
         fairness directly: once worker-a hits its cap, its remaining best-\n\
         fit payloads spill to worker-b instead of piling up regardless."
    );
}
