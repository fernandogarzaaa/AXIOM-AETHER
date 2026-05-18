use std::path::Path;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use sha2::Digest;
use tokenizers::Tokenizer;

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
        let context_ids = self.streamer.fetch_and_pack_context(prompt);
        let context_tensor =
            Tensor::from_vec(context_ids.clone(), (1, context_ids.len()), &self.device)?;
        let _ = self.engine.forward(&context_tensor, None, false)?;

        let prompt_ids = self.encode(prompt);
        let prompt_len = prompt_ids.len();

        let prompt_tensor = Tensor::from_vec(prompt_ids.clone(), (1, prompt_len), &self.device)?;
        let _ = self.engine.forward(&prompt_tensor, None, false)?;

        let mut states = self.engine.init_states(1, &self.device)?;
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
