//! Axiom Prime: a non-blocking finite state machine.
//!
//! The FSM never blocks and never performs I/O. [`SwarmFsm::step`] is a
//! pure transition function `(state, event) -> commands`; the async runner
//! in `main.rs` executes commands (spawning LLM calls and sensor reads as
//! tokio tasks) and feeds their completions back in as events. This keeps
//! Axiom Prime responsive regardless of how slow a worker is.

use axiom_core::node::NodeId;

/// The orchestrator's control states.
#[derive(Debug, Clone, PartialEq)]
pub enum SwarmState {
    /// No goal loaded.
    Idle,
    /// Waiting for sensor fusion of the environment.
    Sensing,
    /// Residual computed; waiting for the mesh routing decision.
    Routing,
    /// Payloads dispatched; waiting for `pending` workers to complete.
    AwaitingWorkers { pending: usize },
    /// Residual within epsilon — the goal state is reached.
    Converged,
    /// Unrecoverable fault.
    Halted { reason: String },
}

/// Everything that can happen to the FSM. Async task completions arrive
/// here — an in-flight LLM call is just a future `WorkerDone` event.
#[derive(Debug, Clone)]
pub enum SwarmEvent {
    /// A goal state was loaded into the controller.
    GoalLoaded,
    /// Sensor fusion finished and the residual was measured. The fused
    /// state itself stays with the runner/controller — the FSM only needs
    /// the convergence verdict.
    Sensed { converged: bool },
    /// The mesh snapped the payload to these nodes.
    Routed { nodes: Vec<NodeId> },
    /// One worker finished (LLM call resolved, tool exited). Which worker
    /// it was stays with the runner — the FSM only counts completions.
    WorkerDone,
    /// Something unrecoverable happened.
    Fault { reason: String },
}

/// Side effects the runner must perform. The FSM only *requests* work.
#[derive(Debug, Clone, PartialEq)]
pub enum SwarmCommand {
    /// Read sensors (terminal, diffs, tests) and fuse them.
    FuseSensors,
    /// Ask the mesh for a routing decision against the current residual.
    RouteResidual,
    /// Condition payloads through sidecars and dispatch to these nodes.
    Dispatch { nodes: Vec<NodeId> },
    /// Report convergence upstream.
    AnnounceConverged,
    /// Report a halt upstream.
    AnnounceHalt { reason: String },
}

/// The non-blocking FSM. Holds only control state — never sensor data,
/// never payloads (those flow through the runner and the mesh).
#[derive(Debug)]
pub struct SwarmFsm {
    state: SwarmState,
}

impl Default for SwarmFsm {
    fn default() -> Self {
        Self::new()
    }
}

impl SwarmFsm {
    pub fn new() -> Self {
        Self { state: SwarmState::Idle }
    }

    pub fn state(&self) -> &SwarmState {
        &self.state
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.state, SwarmState::Converged | SwarmState::Halted { .. })
    }

    /// Pure transition: consume an event, emit commands. Unexpected events
    /// in a given state are ignored (returns no commands) rather than
    /// faulting — late worker completions after convergence are normal.
    pub fn step(&mut self, event: SwarmEvent) -> Vec<SwarmCommand> {
        use SwarmEvent as E;
        use SwarmState as S;

        match (&self.state, event) {
            (_, E::Fault { reason }) => {
                self.state = S::Halted { reason: reason.clone() };
                vec![SwarmCommand::AnnounceHalt { reason }]
            }
            (S::Idle, E::GoalLoaded) => {
                self.state = S::Sensing;
                vec![SwarmCommand::FuseSensors]
            }
            (S::Sensing, E::Sensed { converged, .. }) => {
                if converged {
                    self.state = S::Converged;
                    vec![SwarmCommand::AnnounceConverged]
                } else {
                    self.state = S::Routing;
                    vec![SwarmCommand::RouteResidual]
                }
            }
            (S::Routing, E::Routed { nodes }) => {
                self.state = S::AwaitingWorkers { pending: nodes.len() };
                vec![SwarmCommand::Dispatch { nodes }]
            }
            (S::AwaitingWorkers { pending }, E::WorkerDone) => {
                let remaining = pending.saturating_sub(1);
                if remaining == 0 {
                    // All workers landed: close the loop by re-sensing.
                    self.state = S::Sensing;
                    vec![SwarmCommand::FuseSensors]
                } else {
                    self.state = S::AwaitingWorkers { pending: remaining };
                    vec![]
                }
            }
            // Anything else (late completions, duplicate events) is a no-op.
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sensed(converged: bool) -> SwarmEvent {
        SwarmEvent::Sensed { converged }
    }

    #[test]
    fn full_control_loop_reaches_convergence() {
        let mut fsm = SwarmFsm::new();
        assert_eq!(fsm.step(SwarmEvent::GoalLoaded), vec![SwarmCommand::FuseSensors]);
        assert_eq!(fsm.step(sensed(false)), vec![SwarmCommand::RouteResidual]);

        let nodes = vec![NodeId(0), NodeId(2)];
        assert_eq!(
            fsm.step(SwarmEvent::Routed { nodes: nodes.clone() }),
            vec![SwarmCommand::Dispatch { nodes }]
        );
        // First worker done: still waiting, no commands.
        assert!(fsm.step(SwarmEvent::WorkerDone).is_empty());
        // Second worker done: loop closes with a re-sense.
        assert_eq!(
            fsm.step(SwarmEvent::WorkerDone),
            vec![SwarmCommand::FuseSensors]
        );
        // Residual now within epsilon.
        assert_eq!(fsm.step(sensed(true)), vec![SwarmCommand::AnnounceConverged]);
        assert!(fsm.is_terminal());
    }

    #[test]
    fn fault_halts_from_any_state() {
        let mut fsm = SwarmFsm::new();
        fsm.step(SwarmEvent::GoalLoaded);
        let cmds = fsm.step(SwarmEvent::Fault { reason: "mesh empty".into() });
        assert_eq!(cmds, vec![SwarmCommand::AnnounceHalt { reason: "mesh empty".into() }]);
        assert!(fsm.is_terminal());
    }

    #[test]
    fn late_events_after_convergence_are_ignored() {
        let mut fsm = SwarmFsm::new();
        fsm.step(SwarmEvent::GoalLoaded);
        fsm.step(sensed(true));
        assert!(fsm.step(SwarmEvent::WorkerDone).is_empty());
        assert_eq!(*fsm.state(), SwarmState::Converged);
    }
}
