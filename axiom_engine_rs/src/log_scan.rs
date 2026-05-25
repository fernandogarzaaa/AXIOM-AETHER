use candle_core::{bail, Result, Tensor};

/// Logarithmic associative scan utility for sequence state compression.
///
/// Implements the Blelloch work-efficient parallel prefix scan algorithm in
/// O(log T) parallel steps using an associative operator.
pub struct LogosAssociativeScanner;

impl LogosAssociativeScanner {
    /// Full inclusive parallel prefix scan over `[B, T, C]` token states.
    ///
    /// Implements the Blelloch work-efficient parallel prefix scan algorithm:
    /// - **Up-sweep** (reduce phase): builds partial reductions in a binary
    ///   tree upward in log₂(T) passes, each touching T/2 pairs.
    /// - **Down-sweep** (distribute phase): sets the root to the identity
    ///   element (0) and propagates prefix sums back down the tree in log₂(T)
    ///   passes, yielding the exclusive scan.
    /// - **Inclusive conversion**: each exclusive position i is combined with
    ///   the original `states[i]` to yield the inclusive result.
    ///
    /// Associative operator: `a ⊕ b = a + b + a·b` (identity element: **0**).
    /// Geometric interpretation: `(1 + a)(1 + b) − 1`.
    ///
    /// # Arguments
    /// * `states` – `[B, T, C]` token state tensor.
    ///
    /// # Returns
    /// `[B, T, C]` tensor where position `i` holds the inclusive combination
    /// `states[0] ⊕ states[1] ⊕ … ⊕ states[i]`.
    ///
    /// Intermediate tensors are dropped immediately after each reduction step
    /// to bound peak VRAM usage.
    pub fn parallel_prefix_reduce(states: &Tensor) -> Result<Tensor> {
        let (b, t, c) = states.dims3()?;
        if t == 0 {
            bail!("parallel_prefix_reduce requires non-empty sequence length");
        }
        // Single element: inclusive scan is the element itself.
        if t == 1 {
            return Ok(states.clone());
        }

        let padded_len = t.next_power_of_two();
        let device = states.device();
        let dtype = states.dtype();

        // ------------------------------------------------------------------
        // Build working array: Vec of [B, C] slices, zero-padded to next
        // power of two.  Zero is the identity for a ⊕ b = a + b + a·b.
        // ------------------------------------------------------------------
        let identity = Tensor::zeros((b, c), dtype, device)?;
        let mut arr: Vec<Tensor> = Vec::with_capacity(padded_len);
        for i in 0..t {
            arr.push(states.narrow(1, i, 1)?.squeeze(1)?);
        }
        for _ in t..padded_len {
            // Cheap clone: candle Tensor shares the underlying buffer via Arc.
            arr.push(identity.clone());
        }

        // ------------------------------------------------------------------
        // Up-sweep (reduce): fold pairs up the binary tree.
        //
        // After pass d, every element at an index that is a multiple of
        // 2^(d+1) − 1 holds the partial reduction of 2^(d+1) consecutive
        // elements below it.
        // ------------------------------------------------------------------
        let log_n = padded_len.trailing_zeros() as usize;
        for d in 0..log_n {
            let half_stride = 1usize << d;
            let stride = half_stride << 1;
            let mut k = stride - 1;
            while k < padded_len {
                // a[k] ← a[k - half_stride] ⊕ a[k]
                let left = arr[k - half_stride].clone();
                let right = &arr[k];
                let product = left.mul(right)?;
                let merged = left.add(right)?.add(&product)?;
                drop(product);
                arr[k] = merged;
                k += stride;
            }
        }

        // ------------------------------------------------------------------
        // Down-sweep (distribute): set root to identity, then propagate.
        //
        // For each level (top → bottom), apply the classic Blelloch swap:
        //   t              = arr[left]
        //   arr[left]      = arr[right]          (pass running sum down-left)
        //   arr[right]     = t ⊕ arr[right]      (combine old-left with sum)
        // This produces an exclusive prefix scan in arr[0..padded_len].
        // ------------------------------------------------------------------
        arr[padded_len - 1] = Tensor::zeros((b, c), dtype, device)?;
        for d in (0..log_n).rev() {
            let half_stride = 1usize << d;
            let stride = half_stride << 1;
            let mut k = stride - 1;
            while k < padded_len {
                let left_val = arr[k - half_stride].clone();
                let right_val = arr[k].clone();
                // Pass running sum to left child (unchanged running sum).
                arr[k - half_stride] = right_val.clone();
                // Combine old-left with running sum for right child.
                let product = left_val.mul(&right_val)?;
                arr[k] = left_val.add(&right_val)?.add(&product)?;
                drop(product);
                k += stride;
            }
        }

        // ------------------------------------------------------------------
        // Convert exclusive scan → inclusive scan.
        //
        // exclusive[i] ⊕ original[i] = inclusive[i]
        // We re-read directly from the original `states` tensor (unmodified).
        // ------------------------------------------------------------------
        let mut inclusive: Vec<Tensor> = Vec::with_capacity(t);
        for (i, excl) in arr.iter().enumerate().take(t) {
            let orig = states.narrow(1, i, 1)?.squeeze(1)?;
            let product = excl.mul(&orig)?;
            let incl = excl.add(&orig)?.add(&product)?.unsqueeze(1)?;
            drop(product);
            inclusive.push(incl);
        }

        // Free working array immediately to reclaim VRAM.
        drop(arr);

        Tensor::cat(&inclusive, 1)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// Helper: round-trip a 1-element sequence — scan(a) = a.
    #[test]
    fn test_single_element_is_identity() {
        let device = Device::Cpu;
        let data = vec![3.0f32, 1.0, 4.0]; // B=1, T=1, C=3
        let states = Tensor::from_vec(data.clone(), (1usize, 1usize, 3usize), &device).unwrap();
        let result = LogosAssociativeScanner::parallel_prefix_reduce(&states).unwrap();
        assert_eq!(result.dims(), &[1, 1, 3]);
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in vals.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
        }
    }

    /// Two-element scan: pos 0 = a, pos 1 = a ⊕ b.
    ///
    /// a = [1,1,1], b = [2,2,2]:  a⊕b = 1+2+1*2 = 5 per channel.
    #[test]
    fn test_two_element_scan() {
        let device = Device::Cpu;
        let data = vec![1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0]; // B=1, T=2, C=3
        let states = Tensor::from_vec(data, (1usize, 2usize, 3usize), &device).unwrap();
        let result = LogosAssociativeScanner::parallel_prefix_reduce(&states).unwrap();
        assert_eq!(result.dims(), &[1, 2, 3]);
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let expected = [1.0f32, 1.0, 1.0, 5.0, 5.0, 5.0];
        for (a, b) in vals.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
        }
    }

    /// Four-element numerical verification (C=1 for easy manual check).
    ///
    /// Input:   [1, 2, 3, 4]
    /// Operator a ⊕ b = a + b + a*b:
    ///   pos 0: 1
    ///   pos 1: 1⊕2 = 5
    ///   pos 2: 5⊕3 = 5+3+15 = 23
    ///   pos 3: 23⊕4 = 23+4+92 = 119
    #[test]
    fn test_four_element_numerical() {
        let device = Device::Cpu;
        let data = vec![1.0f32, 2.0, 3.0, 4.0]; // B=1, T=4, C=1
        let states = Tensor::from_vec(data, (1usize, 4usize, 1usize), &device).unwrap();
        let result = LogosAssociativeScanner::parallel_prefix_reduce(&states).unwrap();
        assert_eq!(result.dims(), &[1, 4, 1]);
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let expected = [1.0f32, 5.0, 23.0, 119.0];
        for (a, b) in vals.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "expected {b}, got {a}");
        }
    }

    /// Non-power-of-two sequence (T=3, padded to 4).
    ///
    /// Input:   [2, 3, 4]
    ///   pos 0: 2
    ///   pos 1: 2⊕3 = 2+3+6 = 11
    ///   pos 2: 11⊕4 = 11+4+44 = 59
    #[test]
    fn test_non_power_of_two_t3() {
        let device = Device::Cpu;
        let data = vec![2.0f32, 3.0, 4.0]; // B=1, T=3, C=1
        let states = Tensor::from_vec(data, (1usize, 3usize, 1usize), &device).unwrap();
        let result = LogosAssociativeScanner::parallel_prefix_reduce(&states).unwrap();
        assert_eq!(result.dims(), &[1, 3, 1]);
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let expected = [2.0f32, 11.0, 59.0];
        for (a, b) in vals.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-2, "expected {b}, got {a}");
        }
    }

    /// All-zero input yields all-zero output (identity ⊕ identity = identity).
    #[test]
    fn test_all_zeros() {
        let device = Device::Cpu;
        let states = Tensor::zeros((2usize, 5usize, 8usize), DType::F32, &device).unwrap();
        let result = LogosAssociativeScanner::parallel_prefix_reduce(&states).unwrap();
        assert_eq!(result.dims(), &[2, 5, 8]);
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|&v| v == 0.0), "expected all zeros");
    }
}
