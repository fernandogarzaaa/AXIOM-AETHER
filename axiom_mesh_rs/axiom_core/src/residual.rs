//! State and residual vectors — the control-theoretic core of IDC.
//!
//! The system never asks "what should I say next"; it asks "how far is the
//! system from the goal state, and along which axes". `Residual` is that
//! answer: `Goal − Current`, in a fixed-dimension state space.

use ndarray::Array1;

/// A point in the system's state space. Produced by sensor fusion
/// ([`crate::idc`]) and by goal encoding.
#[derive(Debug, Clone, PartialEq)]
pub struct StateVector(pub Array1<f32>);

impl StateVector {
    pub fn zeros(dim: usize) -> Self {
        Self(Array1::zeros(dim))
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// L2 norm.
    pub fn norm(&self) -> f32 {
        self.0.mapv(|x| x * x).sum().sqrt()
    }
}

/// The gap between where the system is and where it must be:
/// `residual = goal − current`.
#[derive(Debug, Clone)]
pub struct Residual {
    pub vector: Array1<f32>,
}

impl Residual {
    /// Compute the residual between a goal and the current fused state.
    ///
    /// # Panics
    /// Panics if the two vectors have different dimensions — a dimension
    /// mismatch is a programming error, not a runtime condition.
    pub fn between(goal: &StateVector, current: &StateVector) -> Self {
        assert_eq!(
            goal.dim(),
            current.dim(),
            "residual: goal dim {} != current dim {}",
            goal.dim(),
            current.dim()
        );
        Self { vector: &goal.0 - &current.0 }
    }

    /// Magnitude of the remaining gap.
    pub fn norm(&self) -> f32 {
        self.vector.mapv(|x| x * x).sum().sqrt()
    }

    /// True when the residual is within `epsilon` of zero — the system has
    /// converged on the goal state and the FSM may terminate.
    pub fn converged(&self, epsilon: f32) -> bool {
        self.norm() <= epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn residual_is_goal_minus_current() {
        let goal = StateVector(array![1.0, 1.0, 0.0]);
        let current = StateVector(array![0.0, 1.0, 0.5]);
        let r = Residual::between(&goal, &current);
        assert_eq!(r.vector, array![1.0, 0.0, -0.5]);
    }

    #[test]
    fn zero_residual_converges() {
        let s = StateVector(array![0.3, 0.7]);
        let r = Residual::between(&s, &s);
        assert!(r.converged(1e-6));
        assert!(!Residual::between(&StateVector(array![1.0, 0.0]), &s).converged(0.1));
    }
}
