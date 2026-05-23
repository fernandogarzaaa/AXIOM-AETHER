use std::collections::HashMap;

use candle_core::{Device, IndexOp, Result, Tensor};

use crate::config::MEM_TOKEN_ID;

/// Compresses large contexts into a single portable `[1, d_model]` memory vector.
///
/// # Concept – Context Singularity
///
/// The `<MEM>` token (ID `MEM_TOKEN_ID`) acts as a vocabulary sink.  When it
/// appears in a sequence, the forward pass accumulates the full causal context
/// into its activation via the TTT attention mechanism.  `MemoryCompressor`
/// extracts that activation as a discrete latent vector that can be serialised
/// to disk and later injected into any fresh inference session, effectively
/// compressing an O(N) or O(log N) context into a single O(1) tensor.
pub struct MemoryCompressor {
    pub d_model: usize,
}

impl MemoryCompressor {
    pub fn new(d_model: usize) -> Self {
        Self { d_model }
    }

    /// Extract the `<MEM>` token hidden-state from the final layer activations.
    ///
    /// # Arguments
    /// * `final_hidden_states` – `[B, T, d_model]` tensor (output of the last
    ///   transformer block, before the LM head).
    /// * `sequence_tokens`     – Flat slice of token IDs corresponding to the
    ///   `T` positions in `final_hidden_states`.  Must contain exactly one
    ///   occurrence of `MEM_TOKEN_ID`.
    ///
    /// # Returns
    /// `[1, d_model]` tensor – the memory vector for this context.
    pub fn extract_memory_vector(
        &self,
        final_hidden_states: &Tensor,
        sequence_tokens: &[u32],
    ) -> Result<Tensor> {
        let pos = sequence_tokens
            .iter()
            .position(|&id| id == MEM_TOKEN_ID)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "MEM_TOKEN_ID ({MEM_TOKEN_ID}) not found in sequence_tokens"
                ))
            })?;

        let (_, t, d) = final_hidden_states.dims3()?;
        if pos >= t {
            return Err(candle_core::Error::Msg(format!(
                "MEM token position {pos} out of bounds for sequence length {t}"
            )));
        }
        if d != self.d_model {
            return Err(candle_core::Error::Msg(format!(
                "d_model mismatch: compressor expects {}, tensor has {d}",
                self.d_model
            )));
        }

        // Batch index 0, time step `pos` → [d_model]; unsqueeze to [1, d_model].
        final_hidden_states.i((0, pos))?.unsqueeze(0)
    }

    /// Serialise a memory vector to disk using the safetensors format.
    ///
    /// The tensor is stored under the key `"memory"`.  The resulting file can
    /// be loaded by any safetensors-compatible reader (Python, Rust, etc.) and
    /// hot-swapped into a running engine in milliseconds.
    ///
    /// # Arguments
    /// * `tensor`   – `[1, d_model]` memory vector.
    /// * `filename` – Destination path, e.g. `"linux_kernel.mem"`.
    pub fn save_memory_token(tensor: &Tensor, filename: &str) -> Result<()> {
        let tensors: HashMap<String, Tensor> =
            HashMap::from([("memory".to_string(), tensor.clone())]);
        candle_core::safetensors::save(&tensors, filename)
    }

    /// Load a memory vector that was previously saved with `save_memory_token`.
    ///
    /// # Arguments
    /// * `filename` – Path to the `.mem` safetensors file.
    /// * `device`   – Target device for the loaded tensor.
    ///
    /// # Returns
    /// `[1, d_model]` tensor.
    pub fn load_memory_token(filename: &str, device: &Device) -> Result<Tensor> {
        let map = candle_core::safetensors::load(filename, device)?;
        map.get("memory")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("key 'memory' not found in .mem file".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn make_hidden(b: usize, t: usize, d: usize) -> Tensor {
        Tensor::arange(0f32, (b * t * d) as f32, &Device::Cpu)
            .unwrap()
            .reshape((b, t, d))
            .unwrap()
    }

    #[test]
    fn extract_memory_vector_selects_mem_position() {
        let d_model = 8;
        let compressor = MemoryCompressor::new(d_model);
        let hidden = make_hidden(1, 4, d_model);

        // Place MEM_TOKEN_ID at position 2.
        let tokens: Vec<u32> = vec![10, 20, MEM_TOKEN_ID, 30];
        let mem_vec = compressor
            .extract_memory_vector(&hidden, &tokens)
            .expect("extraction failed");

        assert_eq!(mem_vec.dims(), &[1, d_model]);

        // The extracted slice must match position-2 of the input tensor.
        let expected = hidden.i((0, 2)).unwrap().unsqueeze(0).unwrap();
        let diff = mem_vec
            .sub(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "extracted vector mismatch, diff={diff}");
    }

    #[test]
    fn extract_memory_vector_errors_when_no_mem_token() {
        let compressor = MemoryCompressor::new(4);
        let hidden = make_hidden(1, 3, 4);
        let tokens: Vec<u32> = vec![1, 2, 3];
        assert!(
            compressor.extract_memory_vector(&hidden, &tokens).is_err(),
            "should error when MEM_TOKEN_ID is absent"
        );
    }

    #[test]
    fn roundtrip_save_load_memory_token() {
        let d_model = 16usize;
        let data: Vec<f32> = (0..d_model).map(|i| i as f32 * 0.5).collect();
        let tensor = Tensor::from_vec(data.clone(), (1, d_model), &Device::Cpu).unwrap();

        let path = "/tmp/test_mem_token.mem";
        MemoryCompressor::save_memory_token(&tensor, path).expect("save failed");

        let loaded = MemoryCompressor::load_memory_token(path, &Device::Cpu).expect("load failed");
        assert_eq!(loaded.dims(), &[1, d_model]);

        let loaded_data = loaded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in data.iter().zip(loaded_data.iter()) {
            assert!((a - b).abs() < 1e-6, "round-trip mismatch: {a} vs {b}");
        }
    }
}
