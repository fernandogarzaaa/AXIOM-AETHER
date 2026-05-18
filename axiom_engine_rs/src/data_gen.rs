use candle_core::{Device, Result, Tensor};
use rand::Rng;

use crate::config::DEFAULT_EOS_TOKEN;

/// Synthetic procedural dataset generator for structure-only training.
pub struct ProceduralDataset {
    vocab_size: u32,
    eos_token: u32,
}

impl ProceduralDataset {
    pub fn new(vocab_size: usize) -> Self {
        let vocab_size = vocab_size.max(8) as u32;
        Self {
            vocab_size,
            eos_token: DEFAULT_EOS_TOKEN,
        }
    }

    fn map_token(&self, token: u32) -> u32 {
        token % self.vocab_size
    }

    fn generate_variable_trace_sequence<R: Rng + ?Sized>(
        &self,
        seq_len: usize,
        rng: &mut R,
    ) -> Vec<u32> {
        if seq_len == 0 {
            return Vec::new();
        }

        let mut seq = Vec::with_capacity(seq_len);
        let mut var_a: u32 = rng.gen_range(0..32);
        let mut var_b: u32 = rng.gen_range(0..32);

        for step in 0..seq_len {
            if step == 0 {
                seq.push(self.map_token(1));
                continue;
            }

            match step % 4 {
                0 => {
                    let delta = rng.gen_range(1..8);
                    var_a = (var_a + delta) % 64;
                }
                1 => {
                    let delta = rng.gen_range(1..8);
                    var_b = (var_b + delta) % 64;
                }
                2 => {
                    var_a = (var_a * 2 + 3) % 64;
                }
                _ => {
                    std::mem::swap(&mut var_a, &mut var_b);
                }
            }

            let state_id = 32 + ((var_a * 64 + var_b) % 1024);
            seq.push(self.map_token(state_id));
        }

        seq
    }

    fn generate_logic_tree_sequence<R: Rng + ?Sized>(
        &self,
        seq_len: usize,
        rng: &mut R,
    ) -> Vec<u32> {
        if seq_len == 0 {
            return Vec::new();
        }

        // Token bands: symbols [128..191], operators [192..223].
        let symbol_base: u32 = 128;
        let op_base: u32 = 192;

        let mut seq = Vec::with_capacity(seq_len);
        for idx in 0..seq_len {
            let tok = match idx % 6 {
                0 => 4,                                  // logic-stream marker
                1 => symbol_base + rng.gen_range(0..32), // X
                2 => op_base,                            // ->
                3 => symbol_base + rng.gen_range(0..32), // Y
                4 => op_base + 1,                        // and
                _ => {
                    let x = symbol_base + rng.gen_range(0..32);
                    if rng.gen_bool(0.5) {
                        x
                    } else {
                        op_base + 2 // not
                    }
                }
            };
            seq.push(self.map_token(tok));
        }

        seq
    }

    pub fn generate_batch(
        &self,
        batch_size: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let mut rng = rand::thread_rng();
        let mut inputs = Vec::with_capacity(batch_size * seq_len);
        let mut targets = Vec::with_capacity(batch_size * seq_len);

        for _ in 0..batch_size {
            let sample = if rng.gen_bool(0.5) {
                self.generate_variable_trace_sequence(seq_len, &mut rng)
            } else {
                self.generate_logic_tree_sequence(seq_len, &mut rng)
            };

            let mut target = if seq_len > 1 {
                sample[1..].to_vec()
            } else {
                Vec::new()
            };
            target.push(self.eos_token);

            inputs.extend_from_slice(&sample);
            targets.extend_from_slice(&target);
        }

        let input_tensor = Tensor::from_vec(inputs, (batch_size, seq_len), device)?;
        let target_tensor = Tensor::from_vec(targets, (batch_size, seq_len), device)?;
        Ok((input_tensor, target_tensor))
    }
}
