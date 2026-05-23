use candle_core::{bail, Result, Tensor};

/// Logarithmic associative scan utility for sequence state compression.
pub struct LogosAssociativeScanner;

impl LogosAssociativeScanner {
    /// Reduce `[B, T, C]` token states to `[B, 1, C]` in O(log T) reduction depth.
    ///
    /// Uses an associative merge operator:
    /// `a ⊕ b = a + b + a*b`, equivalent to `(1+a)(1+b)-1`.
    pub fn parallel_prefix_reduce(states: &Tensor) -> Result<Tensor> {
        let (b, t, c) = states.dims3()?;
        if t == 0 {
            bail!("parallel_prefix_reduce requires non-empty sequence length")
        }

        let target_len = t.next_power_of_two();
        // For `a ⊕ b = a + b + a*b`, the identity element is 0.
        let mut reduced = if target_len == t {
            states.clone()
        } else {
            let pad = Tensor::zeros((b, target_len - t, c), states.dtype(), states.device())?;
            Tensor::cat(&[states, &pad], 1)?
        };

        let mut current_len = target_len;
        while current_len > 1 {
            let paired = reduced.reshape((b, current_len / 2, 2, c))?;
            let even = paired.narrow(2, 0, 1)?.squeeze(2)?;
            let odd = paired.narrow(2, 1, 1)?.squeeze(2)?;
            let merged = even.add(&odd)?.add(&even.mul(&odd)?)?;
            reduced = merged;
            current_len /= 2;
        }

        reduced.narrow(1, 0, 1)
    }
}
