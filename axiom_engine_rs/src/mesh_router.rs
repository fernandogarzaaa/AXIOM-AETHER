//! Mesh-backed local-SLM model selection for [`crate::swarm_router`].
//!
//! `swarm_router::select_model` picks the first candidate that happens to
//! be present in Ollama's `/api/tags` — a static priority order with no
//! notion of whether a technically-present model is actually healthy or
//! slow in practice. `MeshModelSelector` replaces that decision with
//! `axiom_core::mesh::KineticNeuralMesh`: each candidate model is a worker
//! node (bias-ordered to match the original priority list, so behavior is
//! identical until real outcomes are recorded), and
//! [`MeshModelSelector::record_outcome`] feeds actual success/latency back
//! in, so the mesh learns to route around a model that's present but
//! degraded without needing an operator to edit the candidate list.
//!
//! Opt-in via `AXIOM_MESH_ROUTING=1` (see `SwarmRouterConfig::from_env`) —
//! the original static selector stays the default, since this path has
//! only been verified against a mocked Ollama server in tests, not a live
//! one (see `tests/mesh_router_proxy.rs`).
//!
//! There's no meaningful embedding of "what a chat payload needs" today
//! (that would require a real feature extractor, not a guess), so the
//! payload the mesh routes on is a constant — the routing signal is
//! entirely the static priority bias plus learned quality, which is an
//! honest scope for what this integration can actually justify.

use std::sync::Mutex;

use axiom_core::mesh::{KineticNeuralMesh, MeshConfig};
use axiom_core::node::{NodeId, NodeKind, WorkerNode};
use ndarray::Array1;
use rand::{rngs::StdRng, SeedableRng};

/// Constant payload/dim: see the module docs on why there's no real
/// per-request embedding here.
const DIM: usize = 1;

pub struct MeshModelSelector {
    mesh: Mutex<KineticNeuralMesh>,
    rng: Mutex<StdRng>,
    /// Model name -> its NodeId, in the caller's original priority order.
    nodes: Vec<(String, NodeId)>,
}

impl MeshModelSelector {
    /// One node per candidate, in priority order. Earlier candidates get
    /// a higher static bias so that, with no learned signal yet, this
    /// selects identically to the original `select_model` (prefer the
    /// first available candidate) — bandit learning only pulls behavior
    /// away from that prior as real outcomes accumulate.
    pub fn new(candidates: &[String]) -> Self {
        let mut mesh = KineticNeuralMesh::new(MeshConfig {
            dim: DIM,
            // Low on purpose: `axiom_core::gumbel_softmax` computes
            // `logits/tau + noise` (unscaled noise — see its own module
            // docs on why), so signal-to-noise for any logit gap scales
            // as gap/tau. With the small `bias` gap below (0.3, chosen so
            // one failure can overturn it quickly), tau=0.3 gave a ratio
            // of only 1.0 — barely better than a coin flip against Gumbel
            // noise, so even the very first, nothing-learned-yet call
            // couldn't reliably match the naive selector's "prefer the
            // first candidate" default. Lowering tau restores a high
            // ratio (0.3/0.05 = 6, matching what a bigger bias gap used
            // to give at the old tau) without needing the bias gap itself
            // back to a size that takes many failures to overturn.
            tau: 0.05,
            bandit_gain: 1.0,
            bandit_exploration: 0.3,
            ..Default::default()
        });
        let mut nodes = Vec::with_capacity(candidates.len());
        for (i, name) in candidates.iter().enumerate() {
            // Small on purpose: this only has to establish which candidate
            // wins when nothing has been learned yet (any positive gap
            // does that). A large gap would need many recorded failures
            // to overturn — see record_outcome's doc comment for why that
            // was a real, measured problem at a bigger step size.
            let bias = (candidates.len() - i) as f32 * 0.3;
            let node =
                WorkerNode::new(i, name.clone(), NodeKind::Llm(name.clone()), vec![1.0; DIM]).with_bias(bias);
            let id = mesh.add_node(node).expect("constant dim always matches");
            nodes.push((name.clone(), id));
        }
        Self { mesh: Mutex::new(mesh), rng: Mutex::new(StdRng::from_entropy()), nodes }
    }

    /// Pick a model from `available`, restricted to the subset the mesh
    /// actually knows about and that's currently reachable. `None` means
    /// no configured candidate is available right now — same fallback
    /// signal `select_model` gives, so callers don't need to branch on
    /// which selector produced it.
    pub fn select(&self, available: &[String]) -> Option<String> {
        let eligible: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(name, _)| available.iter().any(|a| a == name))
            .map(|(_, id)| *id)
            .collect();
        if eligible.is_empty() {
            return None;
        }
        let payload = Array1::<f32>::ones(DIM);
        let mesh = self.mesh.lock().unwrap();
        let mut rng = self.rng.lock().unwrap();
        let adhesion = mesh.forward_restricted(&payload, None, &eligible, &mut *rng).ok()?;
        let winner = adhesion.active.first()?;
        self.nodes.iter().find(|(_, id)| id == winner).map(|(name, _)| name.clone())
    }

    /// Feed back whether `model`'s call succeeded and how long it took.
    ///
    /// Bounded to `(0, 1]` on success and a flat `-1.0` on failure — an
    /// earlier version used an unbounded `1000.0 / latency_ms`, which
    /// looked reasonable for real LLM inference latency (hundreds to
    /// thousands of ms) but, tested end-to-end against a fast mock
    /// server, produced rewards in the hundreds from a sub-millisecond
    /// response. That swamped everything else in the field computation
    /// and, combined with a `bias` gap sized for a *static* priority
    /// order (unbounded reward vs. a fixed -1.0 failure penalty), meant a
    /// repeatedly-failing top-priority candidate could still win a
    /// two-digit number of consecutive routing draws before an
    /// alternative got a fair look — the opposite of "learn to route
    /// around a degraded model quickly." Bounding both sides to a
    /// comparable, fixed range fixes both problems at once: latency still
    /// discriminates fast-vs-slow successes, but no longer at a scale that
    /// depends on absolute wall-clock units or can dwarf a real failure
    /// signal.
    pub fn record_outcome(&self, model: &str, success: bool, latency_ms: u64) {
        let Some((_, id)) = self.nodes.iter().find(|(name, _)| name == model) else { return };
        let reward = if success { 200.0 / (latency_ms.max(1) as f32 + 200.0) } else { -1.0 };
        self.mesh.lock().unwrap().record_outcome(*id, reward);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn selects_the_top_priority_candidate_with_no_learned_signal() {
        let selector = MeshModelSelector::new(&names(&["phi4:3.8b", "deepseek-r1:8b", "llama3.3:8b"]));
        // All three available, tau low enough (0.3) with a clear bias gap
        // (2.0 per rank) that the top-priority candidate should win
        // essentially every draw — sweep seeds to confirm it's not luck.
        let mut top_wins = 0;
        for _ in 0..50 {
            if selector.select(&names(&["phi4:3.8b", "deepseek-r1:8b", "llama3.3:8b"]))
                == Some("phi4:3.8b".to_string())
            {
                top_wins += 1;
            }
        }
        assert!(top_wins >= 48, "expected the top-priority candidate to dominate, got {top_wins}/50");
    }

    #[test]
    fn only_selects_among_available_candidates() {
        let selector = MeshModelSelector::new(&names(&["phi4:3.8b", "deepseek-r1:8b"]));
        for _ in 0..20 {
            let picked = selector.select(&names(&["deepseek-r1:8b"]));
            assert_eq!(picked, Some("deepseek-r1:8b".to_string()));
        }
    }

    #[test]
    fn no_overlap_returns_none() {
        let selector = MeshModelSelector::new(&names(&["phi4:3.8b"]));
        assert_eq!(selector.select(&names(&["llama3.1:8b"])), None);
    }

    #[test]
    fn learns_to_route_around_a_failing_top_priority_model() {
        let selector = MeshModelSelector::new(&names(&["phi4:3.8b", "deepseek-r1:8b"]));
        // phi4 is top priority but keeps failing; deepseek keeps succeeding fast.
        for _ in 0..15 {
            selector.record_outcome("phi4:3.8b", false, 5000);
            selector.record_outcome("deepseek-r1:8b", true, 50);
        }
        let mut deepseek_wins = 0;
        for _ in 0..50 {
            if selector.select(&names(&["phi4:3.8b", "deepseek-r1:8b"])) == Some("deepseek-r1:8b".to_string()) {
                deepseek_wins += 1;
            }
        }
        assert!(
            deepseek_wins >= 40,
            "expected routing to have learned away from the failing top-priority model, got {deepseek_wins}/50"
        );
    }

    #[test]
    fn record_outcome_for_unknown_model_is_a_no_op() {
        let selector = MeshModelSelector::new(&names(&["phi4:3.8b"]));
        selector.record_outcome("not-a-configured-model", true, 10); // must not panic
        assert_eq!(selector.select(&names(&["phi4:3.8b"])), Some("phi4:3.8b".to_string()));
    }
}
