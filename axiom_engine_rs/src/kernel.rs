use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

/// Root Mean Square Layer Normalization.
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32, vs: VarBuilder) -> Result<Self> {
        let weight = vs.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let eps = Tensor::new(self.eps, x.device())?;
        let norm = x.broadcast_div(&variance.broadcast_add(&eps)?.sqrt()?)?;
        norm.broadcast_mul(&self.weight)
    }
}
