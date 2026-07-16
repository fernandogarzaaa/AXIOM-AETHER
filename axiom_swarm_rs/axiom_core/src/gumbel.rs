//! Hard Gumbel-Softmax sampling for discrete topology routing.
//!
//! The mesh needs a *differentiable-in-spirit* but *discrete-in-effect*
//! routing decision: soft probabilities would smear a prompt payload across
//! every worker node, defeating sparse activation. The hard variant samples
//! a categorical draw via Gumbel perturbation, then snaps to one-hot.
//!
//! There is no autograd tape in this runtime, so the classic
//! straight-through estimator (`y_hard - y.detach() + y`) degenerates to
//! returning the one-hot directly; the soft distribution is still exposed
//! for telemetry and for future gradient-based topology learning.

use ndarray::Array1;
use rand::Rng;

/// Numerically-stable softmax over a 1-D logit vector.
pub fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp = logits.mapv(|x| (x - max).exp());
    let sum = exp.sum();
    exp / sum
}

/// Draw one Gumbel(0, 1) sample per logit: `g = -ln(-ln(u))`, `u ~ U(0,1)`.
fn sample_gumbel(n: usize, rng: &mut impl Rng) -> Array1<f32> {
    Array1::from_iter((0..n).map(|_| {
        // Clamp away from 0 and 1 so both logs stay finite.
        let u: f32 = rng.gen_range(1e-7..1.0 - 1e-7);
        -(-u.ln()).ln()
    }))
}

/// The result of a Gumbel-Softmax draw.
#[derive(Debug, Clone)]
pub struct GumbelSample {
    /// Relaxed (soft) probabilities — telemetry / future gradient use.
    pub soft: Array1<f32>,
    /// The routing output. One-hot when `hard`, equal to `soft` otherwise.
    pub adhesion: Array1<f32>,
    /// Index of the winning node (argmax of the perturbed logits).
    pub winner: usize,
}

/// Gumbel-Softmax over routing logits.
///
/// * `tau` — temperature. Low values (≈0.1–0.5) sharpen toward argmax;
///   high values explore. Must be > 0.
/// * `hard` — when true, returns a one-hot adhesion vector (the discrete
///   topology snap the KNM requires); when false, returns the relaxed
///   distribution.
pub fn gumbel_softmax(logits: &Array1<f32>, tau: f32, hard: bool, rng: &mut impl Rng) -> GumbelSample {
    assert!(tau > 0.0, "gumbel_softmax: tau must be positive, got {tau}");
    assert!(!logits.is_empty(), "gumbel_softmax: empty logits");

    let g = sample_gumbel(logits.len(), rng);
    let perturbed = (logits + &g) / tau;
    let soft = softmax(&perturbed);

    let winner = soft
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("non-empty logits");

    let adhesion = if hard {
        let mut one_hot = Array1::zeros(soft.len());
        one_hot[winner] = 1.0;
        one_hot
    } else {
        soft.clone()
    };

    GumbelSample { soft, adhesion, winner }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn hard_sample_is_one_hot() {
        let mut rng = StdRng::seed_from_u64(7);
        let s = gumbel_softmax(&array![0.1, 2.0, -1.0], 0.5, true, &mut rng);
        assert_eq!(s.adhesion.iter().filter(|&&x| x == 1.0).count(), 1);
        assert!((s.adhesion.sum() - 1.0).abs() < 1e-6);
        assert_eq!(s.adhesion[s.winner], 1.0);
    }

    #[test]
    fn soft_sample_sums_to_one() {
        let mut rng = StdRng::seed_from_u64(7);
        let s = gumbel_softmax(&array![0.0, 0.0, 0.0, 0.0], 1.0, false, &mut rng);
        assert!((s.adhesion.sum() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn seeded_draws_are_deterministic() {
        let logits = array![1.0, 0.5, 0.2];
        let a = gumbel_softmax(&logits, 0.3, true, &mut StdRng::seed_from_u64(42));
        let b = gumbel_softmax(&logits, 0.3, true, &mut StdRng::seed_from_u64(42));
        assert_eq!(a.winner, b.winner);
        assert_eq!(a.soft, b.soft);
    }

    #[test]
    fn low_temperature_tracks_dominant_logit() {
        // With a huge margin and tiny tau, the dominant logit must win
        // essentially always.
        let logits = array![10.0, 0.0, 0.0];
        let mut rng = StdRng::seed_from_u64(1);
        let wins = (0..200)
            .filter(|_| gumbel_softmax(&logits, 0.1, true, &mut rng).winner == 0)
            .count();
        assert!(wins >= 198, "dominant logit won only {wins}/200 draws");
    }
}
