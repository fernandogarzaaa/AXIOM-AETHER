//! Node health tracking: quarantines workers after repeated dispatch
//! failures.
//!
//! This is "one-for-one" supervision in spirit (the restart/isolate
//! strategy from Erlang-style actor supervision trees), scoped down to
//! what the swarm actually needs: a node's failure streak never affects
//! its siblings' routing eligibility, and crossing the threshold is the
//! caller's cue to call `mesh.remove_node` rather than keep routing to a
//! backend that keeps failing.

use std::collections::HashMap;

use axiom_core::node::NodeId;

/// Tracks consecutive dispatch failures per node and flags when a node
/// should be quarantined (removed from the mesh).
#[derive(Debug)]
pub struct NodeHealth {
    consecutive_failures: HashMap<NodeId, u32>,
    threshold: u32,
}

impl NodeHealth {
    /// `threshold` — consecutive failures (with no intervening success)
    /// before `record_failure` reports quarantine.
    pub fn new(threshold: u32) -> Self {
        Self { consecutive_failures: HashMap::new(), threshold }
    }

    /// Record a successful dispatch: clears the node's failure streak.
    pub fn record_success(&mut self, id: NodeId) {
        self.consecutive_failures.remove(&id);
    }

    /// Record a failed dispatch. Returns `true` the moment this node
    /// crosses the quarantine threshold — the caller should remove it from
    /// the mesh when this fires.
    pub fn record_failure(&mut self, id: NodeId) -> bool {
        let count = self.consecutive_failures.entry(id).or_insert(0);
        *count += 1;
        *count >= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarantines_only_after_threshold_consecutive_failures() {
        let mut health = NodeHealth::new(3);
        assert!(!health.record_failure(NodeId(0)));
        assert!(!health.record_failure(NodeId(0)));
        assert!(health.record_failure(NodeId(0)), "third consecutive failure must quarantine");
    }

    #[test]
    fn success_resets_the_streak() {
        let mut health = NodeHealth::new(2);
        assert!(!health.record_failure(NodeId(0)));
        health.record_success(NodeId(0));
        assert!(!health.record_failure(NodeId(0)), "streak must have reset after success");
    }

    #[test]
    fn nodes_are_tracked_independently() {
        let mut health = NodeHealth::new(1);
        assert!(health.record_failure(NodeId(0)));
        assert!(health.record_failure(NodeId(1)), "node 1's own first failure must quarantine too");
    }
}
