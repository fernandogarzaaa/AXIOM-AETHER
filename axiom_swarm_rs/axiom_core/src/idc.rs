//! Intent-Driven Convergence: sensor fusion and the actuator.
//!
//! The loop, each control tick:
//!
//! ```text
//!   sensors ──fuse──▶ StateVector ──(goal − state)──▶ Residual
//!        ▲                                               │
//!        │                                               ▼
//!   environment ◀──execute── CorrectionVector ◀──actuate─┘
//! ```
//!
//! Sensor fusion here is deliberately deterministic: readings are hashed
//! into a fixed-dimension feature space so the controller is testable with
//! no model in the loop. A learned encoder can replace [`fuse`] later
//! without touching the control law.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::residual::{Residual, StateVector};

/// Raw sensor data flowing into the controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorReading {
    /// Terminal output from a command the swarm ran.
    Terminal { command: String, stdout: String, stderr: String, exit_code: i32 },
    /// A file changed on disk.
    FileDiff { path: String, lines_added: usize, lines_removed: usize },
    /// A test run finished.
    TestLog { passed: usize, failed: usize, failures: Vec<String> },
}

/// A concrete action command — never conversational text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Actuation {
    /// Dispatch a payload to the mesh for worker execution.
    Dispatch { intent: String },
    /// Run a deterministic command (build, test, lint) to refresh sensors.
    RunCommand { command: String },
    /// Residual within epsilon: stop.
    Halt,
}

/// The controller's output for one tick: the action(s) chosen to shrink
/// the residual, plus the residual magnitude they were chosen against.
#[derive(Debug, Clone)]
pub struct CorrectionVector {
    pub residual_norm: f32,
    pub actions: Vec<Actuation>,
}

/// FNV-1a — stable, dependency-free feature hashing.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Hash a labeled feature into the state space with a signed contribution.
fn hash_feature(state: &mut Array1<f32>, label: &str, value: f32) {
    let h = fnv1a(label.as_bytes());
    let idx = (h % state.len() as u64) as usize;
    // Second hash bit decides sign so colliding features don't only add.
    let sign = if (h >> 32) & 1 == 0 { 1.0 } else { -1.0 };
    state[idx] += sign * value;
}

/// Fuse a batch of sensor readings into a single state vector of dimension
/// `dim`. Deterministic: identical readings always produce identical state.
pub fn fuse(readings: &[SensorReading], dim: usize) -> StateVector {
    let mut state = Array1::zeros(dim);
    for r in readings {
        match r {
            SensorReading::Terminal { command, stderr, exit_code, .. } => {
                hash_feature(&mut state, &format!("term:{command}"), 1.0);
                hash_feature(&mut state, "term:exit_ok", if *exit_code == 0 { 1.0 } else { -1.0 });
                if !stderr.is_empty() {
                    hash_feature(&mut state, "term:stderr", -1.0);
                }
            }
            SensorReading::FileDiff { path, lines_added, lines_removed } => {
                hash_feature(&mut state, &format!("diff:{path}"), 1.0);
                hash_feature(&mut state, "diff:churn", (*lines_added + *lines_removed) as f32 / 100.0);
            }
            SensorReading::TestLog { passed, failed, failures } => {
                let total = (passed + failed).max(1) as f32;
                hash_feature(&mut state, "tests:pass_rate", *passed as f32 / total);
                hash_feature(&mut state, "tests:failing", *failed as f32);
                for f in failures {
                    hash_feature(&mut state, &format!("tests:fail:{f}"), 1.0);
                }
            }
        }
    }
    StateVector(state)
}

/// The IDC controller: holds the goal state and turns residuals into
/// correction vectors.
pub struct IdcController {
    goal: StateVector,
    /// Convergence threshold on the residual norm.
    pub epsilon: f32,
}

impl IdcController {
    pub fn new(goal: StateVector, epsilon: f32) -> Self {
        Self { goal, epsilon }
    }

    pub fn goal(&self) -> &StateVector {
        &self.goal
    }

    /// Measure the residual against fused sensor state.
    pub fn residual(&self, current: &StateVector) -> Residual {
        Residual::between(&self.goal, current)
    }

    /// Actuator logic: map the residual to concrete action commands.
    ///
    /// The base policy is intentionally simple and total:
    /// * converged → `Halt`
    /// * otherwise → dispatch an intent describing the gap, then re-sense
    ///   via `verify_command` if one is configured.
    pub fn actuate(&self, residual: &Residual, verify_command: Option<&str>) -> CorrectionVector {
        let norm = residual.norm();
        if residual.converged(self.epsilon) {
            return CorrectionVector { residual_norm: norm, actions: vec![Actuation::Halt] };
        }
        let mut actions = vec![Actuation::Dispatch {
            intent: format!("reduce residual (norm {norm:.4}) toward goal state"),
        }];
        if let Some(cmd) = verify_command {
            actions.push(Actuation::RunCommand { command: cmd.to_string() });
        }
        CorrectionVector { residual_norm: norm, actions }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passing_tests() -> SensorReading {
        SensorReading::TestLog { passed: 10, failed: 0, failures: vec![] }
    }

    #[test]
    fn fusion_is_deterministic() {
        let readings = vec![
            SensorReading::Terminal {
                command: "cargo test".into(),
                stdout: "ok".into(),
                stderr: String::new(),
                exit_code: 0,
            },
            passing_tests(),
        ];
        assert_eq!(fuse(&readings, 32), fuse(&readings, 32));
    }

    #[test]
    fn failing_tests_move_state_away_from_goal() {
        let goal = fuse(&[passing_tests()], 32);
        let bad = fuse(
            &[SensorReading::TestLog { passed: 5, failed: 5, failures: vec!["t::a".into()] }],
            32,
        );
        let ctl = IdcController::new(goal, 1e-3);
        let r_bad = ctl.residual(&bad);
        assert!(!r_bad.converged(ctl.epsilon));
        // Reaching the goal state itself converges.
        let r_good = ctl.residual(&fuse(&[passing_tests()], 32));
        assert!(r_good.converged(ctl.epsilon));
    }

    #[test]
    fn actuator_halts_on_convergence_and_acts_otherwise() {
        let goal = fuse(&[passing_tests()], 16);
        let ctl = IdcController::new(goal.clone(), 1e-3);

        let done = ctl.actuate(&ctl.residual(&goal), Some("cargo test"));
        assert_eq!(done.actions, vec![Actuation::Halt]);

        let gap = ctl.residual(&StateVector::zeros(16));
        let cv = ctl.actuate(&gap, Some("cargo test"));
        assert!(cv.residual_norm > 0.0);
        assert!(matches!(cv.actions[0], Actuation::Dispatch { .. }));
        assert_eq!(cv.actions[1], Actuation::RunCommand { command: "cargo test".into() });
    }
}
