mod config;
mod kernel;
mod ttt_layer;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use config::AxiomConfig;
use kernel::AxiomTTTEngine;

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;

    // Use small dimensions so the demo runs quickly on CPU.
    let config = AxiomConfig {
        d_model: 64,
        n_layers: 2,
        num_heads: 4,
        head_dim: 16, // 64 / 4
        vocab_size: 256,
        lr_inner: 1e-3,
        rms_norm_eps: 1e-6,
    };

    let vm = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &device);
    let engine = AxiomTTTEngine::new(vs, config.clone())?;

    // -----------------------------------------------------------------------
    // Prefill
    // -----------------------------------------------------------------------
    let prompt_tokens = Tensor::zeros((1, 8), DType::U32, &device)?;
    let (prefill_logits, _) = engine.forward(&prompt_tokens, None, false)?;
    println!(
        "Prefill  logits shape : {:?}",
        prefill_logits.shape().dims()
    );

    // -----------------------------------------------------------------------
    // Decode (autoregressive)
    // -----------------------------------------------------------------------
    let mut states = engine.init_states(1, &device)?;
    let mut last_token = Tensor::zeros((1, 1), DType::U32, &device)?;

    for step in 0..4 {
        let (logits, next_states) = engine.forward(&last_token, Some(states), true)?;
        states = next_states.expect("decode must return states");
        let next_id = logits
            .squeeze(1)?
            .argmax(candle_core::D::Minus1)?
            .squeeze(0)?
            .to_scalar::<u32>()?;
        println!("Decode step {step}: next_token_id = {next_id}");
        last_token = Tensor::new(&[next_id], &device)?.unsqueeze(0)?;
    }

    Ok(())
}
