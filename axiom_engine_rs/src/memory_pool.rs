//! Memory-token extraction and safetensors I/O utilities.
//!
//! This module provides:
//! - [`extract_memory_vector`]: slice the hidden-state vector for a specific
//!   memory token out of a sequence tensor.
//! - [`save_to_disk`] / [`load_from_disk`]: persist and restore a single
//!   tensor using the SafeTensors on-disk format.

use std::collections::HashMap;
use std::path::Path;

use crate::config::MEM_TOKEN_ID;
use candle_core::{Device, Result, Tensor};

// ---------------------------------------------------------------------------
// Memory token extraction
// ---------------------------------------------------------------------------

/// Extract the hidden-state vector for a specific memory token from a sequence.
///
/// # Arguments
/// * `hidden_states` – `[B, T, d_model]` tensor produced by a forward pass.
/// * `seq_tokens`    – Slice of token IDs corresponding to the T sequence
///                     positions (must have length T).
/// * `mem_token_id`  – The token ID whose hidden state should be extracted.
///
/// # Returns
/// `[B, 1, d_model]` tensor sliced from the **first** occurrence of
/// `mem_token_id` in `seq_tokens`.
///
/// # Errors
/// * `seq_tokens.len() != T`.
/// * `mem_token_id` not found in `seq_tokens`.
pub fn extract_memory_vector(
    hidden_states: &Tensor,
    seq_tokens: &[u32],
    mem_token_id: u32,
) -> Result<Tensor> {
    let (_, t, _) = hidden_states.dims3()?;
    if seq_tokens.len() != t {
        candle_core::bail!(
            "seq_tokens length {} does not match hidden_states sequence dimension {}",
            seq_tokens.len(),
            t
        );
    }

    let idx = seq_tokens
        .iter()
        .position(|&tok| tok == mem_token_id)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "mem_token_id {mem_token_id} not found in seq_tokens"
            ))
        })?;

    // Narrow the sequence dimension to a single slice: [B, 1, d_model].
    hidden_states.narrow(1, idx, 1)
}

// ---------------------------------------------------------------------------
// SafeTensors I/O
// ---------------------------------------------------------------------------

/// The canonical key used when serialising a single tensor to disk.
const TENSOR_KEY: &str = "tensor";

/// Persist a single tensor to a SafeTensors file.
///
/// The tensor is stored under the key `"tensor"`.  If the tensor lives on an
/// accelerator it is first moved to the CPU and cast to `f32` for maximum
/// compatibility with downstream tooling.
///
/// # Arguments
/// * `tensor` – Tensor to save (any dtype, any device).
/// * `path`   – Destination file path.  Created or overwritten atomically.
///
/// # Errors
/// Propagates any I/O or serialisation error from candle.
pub fn save_to_disk(tensor: &Tensor, path: &str) -> Result<()> {
    // Normalise to CPU f32 before serialising.
    let cpu_f32 = tensor
        .to_device(&Device::Cpu)?
        .to_dtype(candle_core::DType::F32)?;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert(TENSOR_KEY.to_string(), cpu_f32);

    candle_core::safetensors::save(&tensors, path)
}

/// Load a tensor from a SafeTensors file previously written by [`save_to_disk`].
///
/// # Arguments
/// * `path`   – Source file path.
/// * `device` – Target device for the loaded tensor.
///
/// # Returns
/// The tensor stored under the `"tensor"` key.
///
/// # Errors
/// * File does not exist.
/// * File does not contain a `"tensor"` key.
/// * Any I/O or deserialisation error from candle.
pub fn load_from_disk(path: &str, device: &Device) -> Result<Tensor> {
    if !Path::new(path).exists() {
        candle_core::bail!("checkpoint file not found: {path}");
    }

    let mut tensors = candle_core::safetensors::load(path, device)?;
    tensors.remove(TENSOR_KEY).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "safetensors file '{path}' does not contain a '{TENSOR_KEY}' key"
        ))
    })
}

/// Backwards-compatible helper for context-singularity workflows on `MEM_TOKEN_ID`.
pub struct MemoryCompressor {
    pub d_model: usize,
}

impl MemoryCompressor {
    pub fn new(d_model: usize) -> Self {
        Self { d_model }
    }

    /// Extract `[1, d_model]` memory vector from the first batch element.
    pub fn extract_memory_vector(
        &self,
        final_hidden_states: &Tensor,
        sequence_tokens: &[u32],
    ) -> Result<Tensor> {
        let (_, _, d) = final_hidden_states.dims3()?;
        if d != self.d_model {
            candle_core::bail!(
                "d_model mismatch: compressor expects {}, tensor has {d}",
                self.d_model
            );
        }
        let mem = extract_memory_vector(final_hidden_states, sequence_tokens, MEM_TOKEN_ID)?;
        mem.narrow(0, 0, 1)?.reshape((1usize, d))
    }

    /// Save memory vector under `"memory"` key for compatibility with older files.
    pub fn save_memory_token(tensor: &Tensor, filename: &str) -> Result<()> {
        let tensors: HashMap<String, Tensor> =
            HashMap::from([("memory".to_string(), tensor.clone())]);
        candle_core::safetensors::save(&tensors, filename)
    }

    /// Load memory vector stored under `"memory"` key.
    pub fn load_memory_token(filename: &str, device: &Device) -> Result<Tensor> {
        let map = candle_core::safetensors::load(filename, device)?;
        map.get("memory")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("key 'memory' not found in .mem file".into()))
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    // ----- extract_memory_vector -------------------------------------------

    #[test]
    fn test_extract_correct_slice() {
        let device = Device::Cpu;
        // hidden_states: B=1, T=4, d_model=8  (values 0..32)
        let data: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let hs = Tensor::from_vec(data, (1usize, 4usize, 8usize), &device).unwrap();
        let tokens = [10u32, 20, 30, 40];

        // Extract token 30, which is at index 2 → values 16..24.
        let mem_vec = extract_memory_vector(&hs, &tokens, 30).unwrap();
        assert_eq!(mem_vec.dims(), &[1, 1, 8]);

        let vals: Vec<f32> = mem_vec.flatten_all().unwrap().to_vec1().unwrap();
        let expected: Vec<f32> = (16..24).map(|x| x as f32).collect();
        assert_eq!(vals, expected);
    }

    #[test]
    fn test_extract_first_token() {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let hs = Tensor::from_vec(data, (1usize, 4usize, 4usize), &device).unwrap();
        let tokens = [99u32, 1, 2, 3];

        let mem_vec = extract_memory_vector(&hs, &tokens, 99).unwrap();
        assert_eq!(mem_vec.dims(), &[1, 1, 4]);

        let vals: Vec<f32> = mem_vec.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals, vec![0.0f32, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_extract_missing_token_errors() {
        let device = Device::Cpu;
        let hs = Tensor::zeros((1usize, 3usize, 4usize), DType::F32, &device).unwrap();
        let tokens = [1u32, 2, 3];
        assert!(extract_memory_vector(&hs, &tokens, 99).is_err());
    }

    #[test]
    fn test_extract_length_mismatch_errors() {
        let device = Device::Cpu;
        let hs = Tensor::zeros((1usize, 3usize, 4usize), DType::F32, &device).unwrap();
        // Provide 4 tokens but T=3.
        let tokens = [1u32, 2, 3, 4];
        assert!(extract_memory_vector(&hs, &tokens, 1).is_err());
    }

    // ----- save / load roundtrip -------------------------------------------

    #[test]
    fn test_save_load_f32_roundtrip() {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data.clone(), (3usize, 4usize), &device).unwrap();

        let path = std::env::temp_dir().join("axiom_pool_test.safetensors");
        let path_str = path.to_str().unwrap();

        save_to_disk(&tensor, path_str).unwrap();
        let loaded = load_from_disk(path_str, &device).unwrap();

        assert_eq!(loaded.dims(), &[3, 4]);
        let loaded_data: Vec<f32> = loaded.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(loaded_data, data);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_nonexistent_file_errors() {
        assert!(load_from_disk("/nonexistent/path/axiom.safetensors", &Device::Cpu).is_err());
    }

    #[test]
    fn test_save_preserves_shape_after_reload() {
        let device = Device::Cpu;
        // 3-D tensor: [2, 3, 4]
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(data, (2usize, 3usize, 4usize), &device).unwrap();

        let path = std::env::temp_dir().join("axiom_pool_3d_test.safetensors");
        let path_str = path.to_str().unwrap();

        save_to_disk(&tensor, path_str).unwrap();
        let loaded = load_from_disk(path_str, &device).unwrap();

        assert_eq!(loaded.dims(), &[2, 3, 4]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_memory_compressor_extract_and_roundtrip() {
        let device = Device::Cpu;
        let d_model = 6usize;
        let data: Vec<f32> = (0..18).map(|x| x as f32).collect();
        let hidden = Tensor::from_vec(data, (1usize, 3usize, d_model), &device).unwrap();
        let tokens = [1u32, MEM_TOKEN_ID, 3u32];

        let compressor = MemoryCompressor::new(d_model);
        let mem_vec = compressor.extract_memory_vector(&hidden, &tokens).unwrap();
        assert_eq!(mem_vec.dims(), &[1, d_model]);

        let path = std::env::temp_dir().join("axiom_mem_compat_test.mem");
        let path_str = path.to_str().unwrap();
        MemoryCompressor::save_memory_token(&mem_vec, path_str).unwrap();
        let loaded = MemoryCompressor::load_memory_token(path_str, &device).unwrap();
        assert_eq!(loaded.dims(), &[1, d_model]);

        let _ = std::fs::remove_file(&path);
    }
}
