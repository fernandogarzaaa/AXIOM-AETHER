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
    /// Terminal output from a command the system ran.
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

/// Exponential moving average smoother for fused sensor state.
///
/// Raw per-tick `fuse()` output can chatter tick to tick under noisy
/// sensors (a flaky test that passes/fails intermittently, jittery
/// terminal timing) — feeding that straight into routing causes the mesh
/// to flip its pick on noise rather than signal, a classic control-theory
/// concern. `StateSmoother` is a composable pre-filter: call `update` with
/// each raw fused state before computing the residual, in place of using
/// the raw state directly.
///
/// This is deliberately a plain EMA rather than a full Kalman filter — no
/// process/measurement noise covariance to tune, at the cost of not
/// separately modeling sensor vs. process uncertainty. See
/// `docs/RESEARCH_BLUEPRINT.md` for why the fuller filter stayed queued.
pub struct StateSmoother {
    /// Smoothing factor in `(0.0, 1.0]`. `1.0` means no smoothing (each
    /// update replaces the estimate outright); smaller values weight
    /// history more heavily and damp noise harder at the cost of lag.
    alpha: f32,
    estimate: Option<StateVector>,
}

impl StateSmoother {
    /// # Panics
    /// Panics if `alpha` isn't in `(0.0, 1.0]`.
    pub fn new(alpha: f32) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "StateSmoother: alpha must be in (0.0, 1.0], got {alpha}");
        Self { alpha, estimate: None }
    }

    /// A smoother that performs no smoothing — every `update` returns the
    /// raw reading unchanged. Useful as the default when a caller wants the
    /// `StateSmoother` API without opting into the behavior yet.
    pub fn disabled() -> Self {
        Self::new(1.0)
    }

    /// Fold in a new raw reading and return the smoothed estimate. The
    /// first call seeds the estimate with the raw reading verbatim (no
    /// prior history to blend with).
    pub fn update(&mut self, raw: StateVector) -> StateVector {
        let smoothed = match &self.estimate {
            None => raw,
            Some(prev) => StateVector(&prev.0 * (1.0 - self.alpha) + &raw.0 * self.alpha),
        };
        self.estimate = Some(smoothed.clone());
        smoothed
    }
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

    #[test]
    fn disabled_smoother_passes_readings_through_unchanged() {
        let mut smoother = StateSmoother::disabled();
        let a = StateVector(Array1::from_vec(vec![1.0, -2.0]));
        let b = StateVector(Array1::from_vec(vec![5.0, 5.0]));
        assert_eq!(smoother.update(a.clone()), a);
        assert_eq!(smoother.update(b.clone()), b);
    }

    #[test]
    fn smoother_matches_hand_computed_ema() {
        let mut smoother = StateSmoother::new(0.25);
        let first = StateVector(Array1::from_vec(vec![0.0]));
        let second = StateVector(Array1::from_vec(vec![4.0]));

        // First call seeds the estimate verbatim.
        assert_eq!(smoother.update(first), StateVector(Array1::from_vec(vec![0.0])));
        // Second: 0.75*0.0 + 0.25*4.0 = 1.0.
        let out = smoother.update(second);
        assert!((out.0[0] - 1.0).abs() < 1e-6, "got {}", out.0[0]);
    }

    #[test]
    fn smoother_reduces_variance_of_an_oscillating_signal() {
        // A signal alternating far above/below its true mean (0.0) — the
        // kind of tick-to-tick chatter a flaky sensor produces.
        let raw: Vec<f32> = (0..40).map(|i| if i % 2 == 0 { 10.0 } else { -10.0 }).collect();

        let mut smoother = StateSmoother::new(0.1);
        let smoothed: Vec<f32> =
            raw.iter().map(|&x| smoother.update(StateVector(Array1::from_vec(vec![x]))).0[0]).collect();

        let variance = |xs: &[f32]| {
            let mean = xs.iter().sum::<f32>() / xs.len() as f32;
            xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / xs.len() as f32
        };

        assert!(
            variance(&smoothed) < variance(&raw) * 0.1,
            "smoothed variance {} should be far below raw variance {}",
            variance(&smoothed),
            variance(&raw)
        );
    }

    #[test]
    #[should_panic(expected = "alpha must be in")]
    fn smoother_rejects_out_of_range_alpha() {
        StateSmoother::new(0.0);
    }
}
