use std::path::Path;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use sha2::Digest;
use tokenizers::Tokenizer;

use crate::config::{AxiomConfig, DEFAULT_CHECKPOINT_PATH};
use crate::jit_streamer::JitContextStreamer;
use crate::model::AxiomTTTLM;

#[derive(Clone, Debug, Default)]
pub struct InferenceRuntimeOptions {
    pub tokenizer_path: Option<String>,
    pub context_api_url: Option<String>,
    pub context_api_key: Option<String>,
    pub max_context_tokens: usize,
}

enum TokenizerBackend {
    Hf(Box<Tokenizer>),
    HashFallback,
}

pub struct InferencePipeline {
    model: AxiomTTTLM,
    _varmap: VarMap,
    device: Device,
    vocab_size: u32,
    tokenizer: TokenizerBackend,
    streamer: JitContextStreamer,
}

impl InferencePipeline {
    pub fn new(config: AxiomConfig, device: Device) -> Result<Self> {
        Self::with_checkpoint_and_options(
            config,
            device,
            DEFAULT_CHECKPOINT_PATH,
            InferenceRuntimeOptions::default(),
        )
    }

    pub fn with_checkpoint(
        config: AxiomConfig,
        device: Device,
        checkpoint_path: impl AsRef<str>,
    ) -> Result<Self> {
        Self::with_checkpoint_and_options(
            config,
            device,
            checkpoint_path,
            InferenceRuntimeOptions::default(),
        )
    }

    pub fn with_checkpoint_and_options(
        config: AxiomConfig,
        device: Device,
        checkpoint_path: impl AsRef<str>,
        runtime: InferenceRuntimeOptions,
    ) -> Result<Self> {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = AxiomTTTLM::new(vb, config.clone())?;

        let checkpoint = checkpoint_path.as_ref();
        if Path::new(checkpoint).exists() {
            varmap.load(checkpoint)?;
            println!("[+] Loaded checkpoint from {checkpoint}");
        } else {
            println!(
                "[!] Warning: No pre-trained checkpoint found. Initializing with baseline random weights."
            );
        }

        let tokenizer = match runtime
            .tokenizer_path
            .as_ref()
            .filter(|path| !path.trim().is_empty())
            .filter(|path| Path::new(path).exists())
        {
            Some(path) => match Tokenizer::from_file(path) {
                Ok(tok) => {
                    println!("[+] Loaded tokenizer from {path}");
                    TokenizerBackend::Hf(Box::new(tok))
                }
                Err(err) => {
                    println!("[!] Warning: failed to load tokenizer at {path}: {err}");
                    TokenizerBackend::HashFallback
                }
            },
            None => TokenizerBackend::HashFallback,
        };

        let streamer = JitContextStreamer::new(
            config.vocab_size as u32,
            runtime.max_context_tokens.max(1),
            runtime.context_api_url,
            runtime.context_api_key,
        );

        Ok(Self {
            model,
            _varmap: varmap,
            device,
            vocab_size: config.vocab_size as u32,
            tokenizer,
            streamer,
        })
    }

    /// Return a reference to the device this pipeline runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Allocate identity-matrix W_tilde states for one batch element across all layers.
    ///
    /// Each state is a `[d_model, d_model]` identity matrix — the neutral
    /// starting point before any test-time training has occurred.
    pub fn init_session_states(&self) -> Result<Vec<Tensor>> {
        self.model.init_states(&self.device)
    }

    /// Stateful generation: run inference while carrying an external W_tilde session.
    ///
    /// Tokens are processed directly through `AxiomTTTLM::forward_lm`, driving
    /// true linear-time autoregressive text generation without any attention
    /// caching overhead.  The session states are updated continuously as new
    /// tokens are processed.
    ///
    /// # Arguments
    /// * `prompt`          – Input text.
    /// * `max_new_tokens`  – Maximum tokens to produce.
    /// * `states`          – Per-layer `[d_model, d_model]` W_tilde tensors from
    ///   the current session.
    ///
    /// # Returns
    /// `(generated_text, updated_states)`.
    pub fn generate_with_session(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        mut states: Vec<Tensor>,
    ) -> Result<(String, Vec<Tensor>)> {
        // Process the prompt through the native TTT backbone; this updates the
        // fast-weight matrices so they encode the prompt context.
        let prompt_ids = self.encode(prompt);
        if !prompt_ids.is_empty() {
            let prompt_tensor =
                Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &self.device)?;
            let _ = self.model.forward_lm(&prompt_tensor, &mut states[..])?;
        }

        let mut last_token = *prompt_ids.last().unwrap_or(&0);
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let token_tensor = Tensor::from_vec(vec![last_token], (1, 1), &self.device)?;
            // Snapshot states so a non-finite update can be discarded.
            let states_snapshot = states.clone();
            let logits = self.model.forward_lm(&token_tensor, &mut states[..])?;
            if !session_states_are_finite(&states)? {
                eprintln!(
                    "[emergency] non-finite state detected during generate_with_session; \
                     discarding update and restoring prior snapshot"
                );
                states = states_snapshot;
            }

            // logits: [1, 1, vocab_size] → squeeze → [1, vocab_size] → argmax → scalar
            let next_id = logits
                .squeeze(1)?
                .argmax(D::Minus1)?
                .squeeze(0)?
                .to_scalar::<u32>()?;
            generated.push(next_id);
            last_token = next_id;
        }

        Ok((self.decode(&generated), states))
    }

    /// Adaptation with a configurable inner-loop step count.
    ///
    /// The `inner_loop_steps` parameter is retained for API backward compatibility
    /// (the `/v1/adapt` endpoint accepts a `steps` field).  In the native TTT
    /// architecture the optimal gradient step is computed automatically per token
    /// by `forward_lm`; the value of this parameter does not alter the output.
    pub fn adapt_on_corpus_with_steps(
        &self,
        corpus: &[String],
        mut states: Vec<Tensor>,
        _inner_loop_steps: usize,
    ) -> Result<Vec<Tensor>> {
        for text in corpus {
            let token_ids = self.encode(text);
            if !token_ids.is_empty() {
                let len = token_ids.len();
                let tensor = Tensor::from_vec(token_ids, (1, len), &self.device)?;
                // Snapshot for non-finite guard.
                let snapshot = states.clone();
                let _ = self.model.forward_lm(&tensor, &mut states[..])?;
                if !session_states_are_finite(&states)? {
                    eprintln!(
                        "[emergency] non-finite state detected during corpus adaptation; \
                         discarding update and restoring prior snapshot"
                    );
                    states = snapshot;
                }
            }
        }
        Ok(states)
    }

    fn encode(&self, prompt: &str) -> Vec<u32> {
        match &self.tokenizer {
            TokenizerBackend::Hf(tokenizer) => match tokenizer.encode(prompt, true) {
                Ok(encoding) => {
                    let mut ids: Vec<u32> = encoding
                        .get_ids()
                        .iter()
                        .map(|id| id % self.vocab_size)
                        .collect();
                    if ids.is_empty() {
                        ids.push(0);
                    }
                    ids
                }
                Err(_) => vec![0],
            },
            TokenizerBackend::HashFallback => {
                let mut ids = Vec::new();
                for tok in prompt.split_whitespace() {
                    let digest = sha2::Sha256::digest(tok.as_bytes());
                    let id = u64::from_le_bytes([
                        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5],
                        digest[6], digest[7],
                    ]) % self.vocab_size as u64;
                    ids.push(id as u32);
                }
                if ids.is_empty() {
                    ids.push(0);
                }
                ids
            }
        }
    }

    /// Public tokenizer access — encodes text into vocabulary token IDs.
    pub fn encode_text(&self, prompt: &str) -> Vec<u32> {
        self.encode(prompt)
    }

    /// Public tokenizer access — decodes token IDs back to text.
    pub fn decode_tokens(&self, token_ids: &[u32]) -> String {
        self.decode(token_ids)
    }

    /// Borrow the underlying TTT language model. Used by the context
    /// compressor to drive forward_lm with externally-held session states.
    pub fn model(&self) -> &AxiomTTTLM {
        &self.model
    }

    /// Whether the active tokenizer is a real HuggingFace model rather than
    /// the SHA-256 hash fallback. Distillation fingerprints use this flag
    /// to decide whether to decode the top-k recall indices as text.
    pub fn has_real_tokenizer(&self) -> bool {
        matches!(self.tokenizer, TokenizerBackend::Hf(_))
    }

    fn decode(&self, token_ids: &[u32]) -> String {
        match &self.tokenizer {
            TokenizerBackend::Hf(tokenizer) => tokenizer
                .decode(token_ids, true)
                .unwrap_or_else(|_| fallback_decode(token_ids)),
            TokenizerBackend::HashFallback => fallback_decode(token_ids),
        }
    }

    pub fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        self.generate_with_memory(prompt, max_new_tokens, None)
    }

    pub fn token_count(&self, text: &str) -> usize {
        self.encode(text).len()
    }

    /// Generation pipeline with optional memory-token priming.
    ///
    /// Fresh identity-matrix session states are allocated, optionally primed
    /// with JIT context (and a pre-computed memory vector when provided), and
    /// then used for autoregressive generation through `AxiomTTTLM::forward_lm`.
    ///
    /// # Arguments
    /// * `prompt`           – User text prompt.
    /// * `max_new_tokens`   – Maximum tokens to generate.
    /// * `loaded_mem_token` – Optional `[1, d_model]` memory vector; when
    ///   provided it is processed through the model before
    ///   the prompt to prime the fast-weight states.
    pub fn generate_with_memory(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        loaded_mem_token: Option<Tensor>,
    ) -> Result<String> {
        let mut states = self.model.init_states(&self.device)?;

        // Optional JIT context streamer priming.
        let context_ids = self.streamer.fetch_and_pack_context(prompt);
        if !context_ids.is_empty() {
            let len = context_ids.len();
            let ctx = Tensor::from_vec(context_ids, (1, len), &self.device)?;
            let _ = self.model.forward_lm(&ctx, &mut states[..])?;
        }

        // Optional memory-vector injection: directly condition the initial fast-weight
        // states using the pre-computed memory vector before processing the prompt.
        // The memory vector [1, d_model] is formed into a [d_model, d_model] outer-product
        // residual and added to every layer's fast-weight matrix as a scaled perturbation,
        // priming the model with the compressed context it encodes.
        if let Some(mem) = loaded_mem_token {
            if !states.is_empty() {
                let d = self.model.config.d_model;
                let mem_f32 = mem.to_dtype(candle_core::DType::F32)?;
                // Reshape [1, d_model] → [d_model, 1] and outer-product with an
                // all-ones row [1, d_model] to form a [d_model, d_model] residual.
                let col = mem_f32.t()?.contiguous()?; // [d_model, 1]
                let ones = Tensor::ones((1, d), candle_core::DType::F32, &self.device)?;
                let residual = col.matmul(&ones)?; // [d_model, d_model]
                let scale = Tensor::new(1e-3f32, &self.device)?;
                let scaled_residual = residual.broadcast_mul(&scale)?;
                for layer_state in states.iter_mut() {
                    let updated = layer_state.add(&scaled_residual)?;
                    if tensor_is_finite(&updated)? {
                        *layer_state = updated;
                    }
                }
            }
        }

        // Process the prompt, updating states.
        let prompt_ids = self.encode(prompt);
        if !prompt_ids.is_empty() {
            let prompt_tensor =
                Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &self.device)?;
            let _ = self.model.forward_lm(&prompt_tensor, &mut states[..])?;
        }

        let mut last_token = *prompt_ids.last().unwrap_or(&0);
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let token_tensor = Tensor::from_vec(vec![last_token], (1, 1), &self.device)?;
            let states_snapshot = states.clone();
            let logits = self.model.forward_lm(&token_tensor, &mut states[..])?;
            if !session_states_are_finite(&states)? {
                eprintln!(
                    "[emergency] non-finite state detected during generate_with_memory; \
                     discarding update and restoring prior snapshot"
                );
                states = states_snapshot;
            }

            let next_id = logits
                .squeeze(1)?
                .argmax(D::Minus1)?
                .squeeze(0)?
                .to_scalar::<u32>()?;
            generated.push(next_id);
            last_token = next_id;
        }

        Ok(self.decode(&generated))
    }
}

fn tensor_is_finite(tensor: &Tensor) -> Result<bool> {
    let values = tensor
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(values.into_iter().all(f32::is_finite))
}

fn session_states_are_finite(states: &[Tensor]) -> Result<bool> {
    for state in states {
        if !tensor_is_finite(state)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn fallback_decode(token_ids: &[u32]) -> String {
    token_ids
        .iter()
        .map(|id| format!("tok_{id}"))
        .collect::<Vec<_>>()
        .join(" ")
}
