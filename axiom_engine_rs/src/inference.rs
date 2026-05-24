use std::path::Path;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use sha2::Digest;
use tokenizers::Tokenizer;

use crate::chunk_kernel::ChunkFusedTTT;
use crate::config::{AxiomConfig, DEFAULT_CHECKPOINT_PATH};
use crate::jit_streamer::JitContextStreamer;
use crate::kernel::AxiomTTTEngine;

#[derive(Clone, Debug, Default)]
pub struct InferenceRuntimeOptions {
    pub tokenizer_path: Option<String>,
    pub context_api_url: Option<String>,
    pub context_api_key: Option<String>,
    pub max_context_tokens: usize,
}

enum TokenizerBackend {
    Hf(Tokenizer),
    HashFallback,
}

pub struct InferencePipeline {
    engine: AxiomTTTEngine,
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
        let engine = AxiomTTTEngine::new(vb, config.clone())?;

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
                    TokenizerBackend::Hf(tok)
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
            engine,
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

    /// Allocate zeroed W_tilde states for one batch element across all layers.
    ///
    /// These states are used as the initial condition for a new TTT session.
    pub fn init_session_states(&self) -> Result<Vec<Tensor>> {
        self.engine.init_states(1, &self.device)
    }

    /// Stateful generation: run inference while carrying an external W_tilde session.
    ///
    /// Unlike [`generate`], this method accepts the caller-owned TTT weight states and
    /// returns updated states alongside the generated text.  Call this for every turn in
    /// a persistent session so the model continuously learns from the conversation.
    ///
    /// # Arguments
    /// * `prompt`          – Input text.
    /// * `max_new_tokens`  – Maximum tokens to produce.
    /// * `states`          – Per-layer W_tilde tensors from the current session.
    ///
    /// # Returns
    /// `(generated_text, updated_states)`.
    pub fn generate_with_session(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        states: Vec<Tensor>,
    ) -> Result<(String, Vec<Tensor>)> {
        let context_ids = self.streamer.fetch_and_pack_context(prompt);
        let context_tensor =
            Tensor::from_vec(context_ids.clone(), (1, context_ids.len()), &self.device)?;
        // Prime context (stateless prefill to warm model internals).
        let _ = self.engine.forward(&context_tensor, None, false)?;

        let prompt_ids = self.encode(prompt);
        let prompt_len = prompt_ids.len();
        let prompt_tensor = Tensor::from_vec(prompt_ids.clone(), (1, prompt_len), &self.device)?;
        // Prefill prompt without updating the session state.
        let _ = self.engine.forward(&prompt_tensor, None, false)?;

        let mut last_token = *prompt_ids.last().unwrap_or(&0);
        let mut current_states = states;
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let token_tensor = Tensor::from_vec(vec![last_token], (1, 1), &self.device)?;
            let (logits, next_states) =
                self.engine
                    .forward(&token_tensor, Some(current_states), true)?;
            current_states = next_states.expect("decode must return states");

            let next_id = logits
                .squeeze(1)?
                .argmax(D::Minus1)?
                .squeeze(0)?
                .to_scalar::<u32>()?;
            generated.push(next_id);
            last_token = next_id;
        }

        Ok((self.decode(&generated), current_states))
    }

    /// In-place TTT adaptation over a text corpus.
    ///
    /// Runs the decode loop on every token of every corpus document, updating the
    /// per-layer W_tilde states via the TTT gradient rule — without touching the
    /// shared model weights.  The adapted states can then be used for generation
    /// via [`generate_with_session`] to produce personalised output.
    ///
    /// # Arguments
    /// * `corpus` – Text examples to adapt on.
    /// * `states` – Current session W_tilde tensors.
    ///
    /// # Returns
    /// Updated W_tilde tensors after processing all corpus tokens.
    pub fn adapt_on_corpus(&self, corpus: &[String], states: Vec<Tensor>) -> Result<Vec<Tensor>> {
        self.adapt_on_corpus_with_steps(corpus, states, 4)
    }

    pub fn adapt_on_corpus_with_steps(
        &self,
        corpus: &[String],
        states: Vec<Tensor>,
        inner_loop_steps: usize,
    ) -> Result<Vec<Tensor>> {
        let mut current_states = states;
        let clamped_steps = inner_loop_steps.clamp(1, 4);

        for text in corpus {
            let token_ids = self.encode(text);
            for &token_id in &token_ids {
                let token_tensor = Tensor::from_vec(vec![token_id], (1, 1), &self.device)?;
                let (_, next_states) = self.engine.forward_with_inner_loop_steps(
                    &token_tensor,
                    Some(current_states),
                    true,
                    Some(clamped_steps),
                )?;
                current_states = next_states.expect("decode must return states");
            }
        }

        Ok(current_states)
    }

    /// Adapt on corpus while routing long sequences through chunk-wise fused prefill.
    ///
    /// Any document with tokenized length strictly greater than `token_threshold`
    /// uses [`ChunkFusedTTT::forward_chunk_fused`] to compute a dense block update.
    pub fn adapt_on_corpus_with_chunk_fusion(
        &self,
        corpus: &[String],
        states: Vec<Tensor>,
        token_threshold: usize,
        block_size: usize,
    ) -> Result<Vec<Tensor>> {
        let mut current_states = states;

        for text in corpus {
            let token_ids = self.encode(text);
            if token_ids.len() > token_threshold {
                current_states =
                    self.adapt_long_sequence_chunk_fused(&token_ids, current_states, block_size)?;
                continue;
            }

            for &token_id in &token_ids {
                let token_tensor = Tensor::from_vec(vec![token_id], (1, 1), &self.device)?;
                let (_, next_states) =
                    self.engine
                        .forward(&token_tensor, Some(current_states), true)?;
                current_states = next_states.expect("decode must return states");
            }
        }

        Ok(current_states)
    }

    fn adapt_long_sequence_chunk_fused(
        &self,
        token_ids: &[u32],
        mut states: Vec<Tensor>,
        block_size: usize,
    ) -> Result<Vec<Tensor>> {
        if states.is_empty() {
            return Ok(states);
        }

        let (batch, heads, head_dim, _) = states[0].dims4()?;
        if batch != 1 {
            candle_core::bail!("chunk-fused adaptation expects batch=1, got {batch}");
        }
        let seq_len = token_ids.len();
        let inv_vocab = 1f32 / self.vocab_size as f32;

        let mut q_data = Vec::with_capacity(batch * heads * seq_len * head_dim);
        let mut k_data = Vec::with_capacity(batch * heads * seq_len * head_dim);
        let mut v_data = Vec::with_capacity(batch * heads * seq_len * head_dim);
        for _b in 0..batch {
            for h in 0..heads {
                for (t, tok) in token_ids.iter().enumerate() {
                    let token_value = *tok as f32 * inv_vocab;
                    for d in 0..head_dim {
                        let dim_factor = (d + 1) as f32 / head_dim as f32;
                        let head_factor = 1f32 + h as f32 / heads as f32;
                        let pos_factor = 1f32 + t as f32 / seq_len as f32;
                        q_data.push(token_value * dim_factor * head_factor);
                        k_data.push(token_value * dim_factor * pos_factor);
                        v_data.push(token_value * dim_factor);
                    }
                }
            }
        }

        let q = Tensor::from_vec(q_data, (batch, heads, seq_len, head_dim), &self.device)?;
        let k = Tensor::from_vec(k_data, (batch, heads, seq_len, head_dim), &self.device)?;
        let v = Tensor::from_vec(v_data, (batch, heads, seq_len, head_dim), &self.device)?;
        let (_, global_update) = ChunkFusedTTT::forward_chunk_fused(&q, &k, &v, block_size)?;

        for layer_state in states.iter_mut() {
            *layer_state = layer_state.add(&global_update)?;
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

    /// Generation pipeline with optional memory-token injection.
    ///
    /// When `loaded_mem_token` is `Some`, the engine bypasses the standard
    /// just-in-time (JIT) context-streamer prefill and instead prepends the supplied `[1, d_model]`
    /// vector directly into the first layer's embedding sequence.  This lets the
    /// model draw on a compressed, pre-computed context without re-tokenizing or
    /// re-processing the original document.
    ///
    /// When `loaded_mem_token` is `None` the behaviour is identical to
    /// [`generate`].
    ///
    /// # Arguments
    /// * `prompt`           – User text prompt (always tokenised normally).
    /// * `max_new_tokens`   – Maximum tokens to generate.
    /// * `loaded_mem_token` – Optional `[1, d_model]` memory vector.
    pub fn generate_with_memory(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        loaded_mem_token: Option<Tensor>,
    ) -> Result<String> {
        let prompt_ids = self.encode(prompt);
        let prompt_len = prompt_ids.len();
        let prompt_tensor = Tensor::from_vec(prompt_ids.clone(), (1, prompt_len), &self.device)?;

        let (_, mut states) = if let Some(ref mem) = loaded_mem_token {
            // Memory-injection path: bypass context prefill and inject the
            // memory vector as the foundational state modifier.
            self.engine
                .prefill_with_state_init_and_memory(&prompt_tensor, mem)?
        } else {
            // Standard path: prime the model with JIT context streamer output.
            let context_ids = self.streamer.fetch_and_pack_context(prompt);
            let context_tensor =
                Tensor::from_vec(context_ids.clone(), (1, context_ids.len()), &self.device)?;
            let _ = self.engine.forward(&context_tensor, None, false)?;
            self.engine.prefill_with_state_init(&prompt_tensor)?
        };

        let mut last_token = *prompt_ids.last().unwrap_or(&0);
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let token_tensor = Tensor::from_vec(vec![last_token], (1, 1), &self.device)?;
            let (logits, next_states) = self.engine.forward(&token_tensor, Some(states), true)?;
            states = next_states.expect("decode must return states");

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

fn fallback_decode(token_ids: &[u32]) -> String {
    token_ids
        .iter()
        .map(|id| format!("tok_{id}"))
        .collect::<Vec<_>>()
        .join(" ")
}
