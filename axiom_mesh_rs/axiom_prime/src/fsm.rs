//! Axiom Prime: a non-blocking finite state machine.
//!
//! The FSM never blocks and never performs I/O. [`PrimeFsm::step`] is a
//! pure transition function `(state, event) -> commands`; the async runner
//! in `main.rs` executes commands (spawning LLM calls and sensor reads as
//! tokio tasks) and feeds their completions back in as events. This keeps
//! Axiom Prime responsive regardless of how slow a worker is.

use axiom_core::node::NodeId;

/// The orchestrator's control states.
#[derive(Debug, Clone, PartialEq)]
pub enum PrimeState {
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
pub enum PrimeEvent {
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
pub enum PrimeCommand {
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
pub struct PrimeFsm {
    state: PrimeState,
}

impl Default for PrimeFsm {
    fn default() -> Self {
        Self::new()
    }
}

impl PrimeFsm {
    pub fn new() -> Self {
        Self { state: PrimeState::Idle }
    }

    pub fn state(&self) -> &PrimeState {
        &self.state
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.state, PrimeState::Converged | PrimeState::Halted { .. })
    }

    /// Pure transition: consume an event, emit commands. Unexpected events
    /// in a given state are ignored (returns no commands) rather than
    /// faulting — late worker completions after convergence are normal.
    pub fn step(&mut self, event: PrimeEvent) -> Vec<PrimeCommand> {
        use PrimeEvent as E;
        use PrimeState as S;

        match (&self.state, event) {
            (_, E::Fault { reason }) => {
                self.state = S::Halted { reason: reason.clone() };
                vec![PrimeCommand::AnnounceHalt { reason }]
            }
            (S::Idle, E::GoalLoaded) => {
                self.state = S::Sensing;
                vec![PrimeCommand::FuseSensors]
            }
            (S::Sensing, E::Sensed { converged, .. }) => {
                if converged {
                    self.state = S::Converged;
                    vec![PrimeCommand::AnnounceConverged]
                } else {
                    self.state = S::Routing;
                    vec![PrimeCommand::RouteResidual]
                }
            }
            (S::Routing, E::Routed { nodes }) => {
                self.state = S::AwaitingWorkers { pending: nodes.len() };
                vec![PrimeCommand::Dispatch { nodes }]
            }
            (S::AwaitingWorkers { pending }, E::WorkerDone) => {
                let remaining = pending.saturating_sub(1);
                if remaining == 0 {
                    // All workers landed: close the loop by re-sensing.
                    self.state = S::Sensing;
                    vec![PrimeCommand::FuseSensors]
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

    fn sensed(converged: bool) -> PrimeEvent {
        PrimeEvent::Sensed { converged }
    }

    #[test]
    fn full_control_loop_reaches_convergence() {
        let mut fsm = PrimeFsm::new();
        assert_eq!(fsm.step(PrimeEvent::GoalLoaded), vec![PrimeCommand::FuseSensors]);
        assert_eq!(fsm.step(sensed(false)), vec![PrimeCommand::RouteResidual]);

        let nodes = vec![NodeId(0), NodeId(2)];
        assert_eq!(
            fsm.step(PrimeEvent::Routed { nodes: nodes.clone() }),
            vec![PrimeCommand::Dispatch { nodes }]
        );
        // First worker done: still waiting, no commands.
        assert!(fsm.step(PrimeEvent::WorkerDone).is_empty());
        // Second worker done: loop closes with a re-sense.
        assert_eq!(
            fsm.step(PrimeEvent::WorkerDone),
            vec![PrimeCommand::FuseSensors]
        );
        // Residual now within epsilon.
        assert_eq!(fsm.step(sensed(true)), vec![PrimeCommand::AnnounceConverged]);
        assert!(fsm.is_terminal());
    }

    #[test]
    fn fault_halts_from_any_state() {
        let mut fsm = PrimeFsm::new();
        fsm.step(PrimeEvent::GoalLoaded);
        let cmds = fsm.step(PrimeEvent::Fault { reason: "mesh empty".into() });
        assert_eq!(cmds, vec![PrimeCommand::AnnounceHalt { reason: "mesh empty".into() }]);
        assert!(fsm.is_terminal());
    }

    #[test]
    fn late_events_after_convergence_are_ignored() {
        let mut fsm = PrimeFsm::new();
        fsm.step(PrimeEvent::GoalLoaded);
        fsm.step(sensed(true));
        assert!(fsm.step(PrimeEvent::WorkerDone).is_empty());
        assert_eq!(*fsm.state(), PrimeState::Converged);
    }

    // --- Property-based test ---------------------------------------------
    //
    // `fault_halts_from_any_state` and `late_events_after_convergence_are_
    // ignored` pin down two specific interleavings by hand; this sweeps
    // arbitrary event sequences to check the invariant they're both
    // instances of: once Axiom Prime reaches a terminal state, no sequence
    // of further events — however it's shuffled — can pull it back into a
    // non-terminal one. (`Fault` can still turn `Converged` into `Halted`;
    // that's a terminal-to-terminal transition, not a violation.)

    mod properties {
        use super::*;
        use proptest::prelude::*;

        fn arbitrary_event() -> impl Strategy<Value = PrimeEvent> {
            prop_oneof![
                Just(PrimeEvent::GoalLoaded),
                any::<bool>().prop_map(|converged| PrimeEvent::Sensed { converged }),
                prop::collection::vec(0usize..4, 0..3)
                    .prop_map(|ids| PrimeEvent::Routed { nodes: ids.into_iter().map(NodeId).collect() }),
                Just(PrimeEvent::WorkerDone),
                "[a-z]{0,8}".prop_map(|reason| PrimeEvent::Fault { reason }),
            ]
        }

        proptest! {
            #[test]
            fn is_terminal_never_reverts_once_true(
                events in prop::collection::vec(arbitrary_event(), 0..30)
            ) {
                let mut fsm = PrimeFsm::new();
                let mut was_terminal = false;
                for event in events {
                    fsm.step(event);
                    if was_terminal {
                        prop_assert!(fsm.is_terminal(), "left a terminal state: {:?}", fsm.state());
                    }
                    was_terminal |= fsm.is_terminal();
                }
            }
        }
    }
}
