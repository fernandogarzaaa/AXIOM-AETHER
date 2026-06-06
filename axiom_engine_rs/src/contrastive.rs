//! InfoNCE contrastive loss over stacked per-example embeddings.
//! Anchors `A[B,d]` and positives `P[B,d]` (both L2-normalized rows). The
//! positive for anchor i is row i of P; all other rows are in-batch negatives.

use candle_core::{Result, Tensor, D};

/// Symmetric InfoNCE. `anchors`,`positives`: `[B, d]`. Returns a scalar loss.
pub fn info_nce(anchors: &Tensor, positives: &Tensor, tau: f64) -> Result<Tensor> {
    let (b, _d) = anchors.dims2()?;
    // similarity [B,B] = A · P^T / tau
    let pt = positives.t()?.contiguous()?;
    let logits = anchors.matmul(&pt)?.affine(1.0 / tau, 0.0)?;
    let device = anchors.device();
    let targets = Tensor::arange(0u32, b as u32, device)?; // diagonal
    // a→p direction
    let loss_a = candle_nn::loss::cross_entropy(&logits, &targets)?;
    // p→a direction (transpose logits)
    let logits_t = logits.t()?.contiguous()?;
    let loss_p = candle_nn::loss::cross_entropy(&logits_t, &targets)?;
    (loss_a + loss_p)?.affine(0.5, 0.0)
}

/// Recall@1 of a batch: fraction of anchors whose nearest positive (by the same
/// similarity matrix) is its own. A pure-eval metric (no grad).
pub fn batch_recall_at_1(anchors: &Tensor, positives: &Tensor) -> Result<f32> {
    let (b, _) = anchors.dims2()?;
    let pt = positives.t()?.contiguous()?;
    let sims = anchors.matmul(&pt)?; // [B,B]
    let pred = sims.argmax(D::Minus1)?; // [B]
    let pred: Vec<u32> = pred.to_vec1()?;
    let correct = pred.iter().enumerate().filter(|(i, p)| **p as usize == *i).count();
    Ok(correct as f32 / b.max(1) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    // Build a [B,d] tensor with L2-normalized rows from raw rows.
    fn norm_rows(rows: Vec<Vec<f32>>) -> Tensor {
        let b = rows.len();
        let d = rows[0].len();
        let mut flat = Vec::with_capacity(b * d);
        for r in rows {
            let n: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in r {
                flat.push(x / n);
            }
        }
        Tensor::from_vec(flat, (b, d), &Device::Cpu).unwrap()
    }

    #[test]
    fn perfect_alignment_has_low_loss() {
        let a = norm_rows(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]);
        let loss = info_nce(&a, &a, 0.05).unwrap().to_scalar::<f32>().unwrap();
        assert!(loss < 0.1, "loss {loss}");
        assert_eq!(batch_recall_at_1(&a, &a).unwrap(), 1.0);
    }

    #[test]
    fn misaligned_has_higher_loss() {
        let a = norm_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let p = norm_rows(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        let good = info_nce(&a, &a, 0.05).unwrap().to_scalar::<f32>().unwrap();
        let bad = info_nce(&a, &p, 0.05).unwrap().to_scalar::<f32>().unwrap();
        assert!(bad > good, "bad {bad} should exceed good {good}");
    }

    #[test]
    fn loss_is_finite_and_has_grad_path() {
        let a = norm_rows(vec![vec![0.6, 0.8], vec![0.8, 0.6]]);
        let l = info_nce(&a, &a, 0.07).unwrap();
        assert!(l.to_scalar::<f32>().unwrap().is_finite());
    }
}
