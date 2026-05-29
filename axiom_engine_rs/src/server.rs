//! OpenAI-compatible HTTP API server for the Axiom-TTT engine.
//!
//! Exposes the following endpoints:
//!
//! | Method | Path                                 | Description                              |
//! |--------|--------------------------------------|------------------------------------------|
//! | GET    | `/metrics`                           | Prometheus exposition endpoint           |
//! | GET    | `/v1/models`                         | List available models                    |
//! | POST   | `/v1/completions`                    | Text completion (stateless or session)   |
//! | POST   | `/v1/chat/completions`               | Chat completion (stateless or session)   |
//! | POST   | `/v1/messages`                       | Anthropic Messages API (Claude clients)  |
//! | POST   | `/v1/cluster/sync`                   | Delta state replication merge hook       |
//! | POST   | `/v1/sessions`                       | Create a new persistent TTT session      |
//! | DELETE | `/v1/sessions/{id}`                  | Delete a session                         |
//! | POST   | `/v1/adapt`                          | In-place TTT adaptation on a corpus      |
//! | GET    | `/v1/sessions/{id}/checkpoint`       | Export session state as JSON             |
//! | PUT    | `/v1/sessions/{id}/checkpoint`       | Restore session state from JSON          |

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use candle_core::{DType, Device, Tensor};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::task::spawn_blocking;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::anthropic_forwarder::{
    build_compressed_payload, partition_messages, whitespace_token_count, AnthropicForwarder,
    ClientAuth, ForwarderError,
};
use crate::claude_backend::{ChatTurn, ClaudeBackend};
use crate::cluster::StateDeltaUpdate;
use crate::config::AxiomConfig;
use crate::context_compressor::{
    adapt_session_blocking, extract_memory_vector_blocking, CompressorConfig, MemoryFingerprint,
    TttSessionStore,
};
use crate::inference::InferencePipeline;
use crate::metrics;
use crate::quantization::{NF4QuantizedDescriptor, NF4Quantizer};

const MAX_ACTIVE_VRAM_SESSIONS: usize = 32;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Shared application state
// ---------------------------------------------------------------------------

enum SessionResidency {
    Active(Vec<Tensor>),
    Quantized(Vec<NF4QuantizedDescriptor>),
}

#[derive(Clone, Copy)]
struct SequenceState {
    version: u64,
    timestamp: i64,
}

struct SessionData {
    residency: SessionResidency,
    created_at: u64,
    last_used: u64,
}

impl SessionData {
    fn new_active(states: Vec<Tensor>, created_at: u64) -> Self {
        Self {
            residency: SessionResidency::Active(states),
            created_at,
            last_used: created_at,
        }
    }

    fn replace_states(&mut self, states: Vec<Tensor>) {
        self.residency = SessionResidency::Active(states);
    }

    fn is_quantized(&self) -> bool {
        matches!(self.residency, SessionResidency::Quantized(_))
    }

    fn states_clone(&self) -> candle_core::Result<Vec<Tensor>> {
        match &self.residency {
            SessionResidency::Active(states) => Ok(states.clone()),
            SessionResidency::Quantized(_) => {
                candle_core::bail!("session is parked in compressed form")
            }
        }
    }

    fn ensure_active(&mut self, device: &Device) -> candle_core::Result<Vec<Tensor>> {
        if let SessionResidency::Quantized(descriptors) = &self.residency {
            let mut states = Vec::with_capacity(descriptors.len());
            for (descriptor_idx, descriptor) in descriptors.iter().enumerate() {
                if descriptor.packed_width == 0 {
                    candle_core::bail!("descriptor {descriptor_idx}: packed width must be non-zero")
                }
                if descriptor.scales.is_empty() {
                    candle_core::bail!("descriptor {descriptor_idx}: scale list cannot be empty")
                }
                let num_blocks = descriptor.scales.len();
                let packed = Tensor::from_vec(
                    descriptor.packed_indices.clone(),
                    (num_blocks, descriptor.packed_width),
                    device,
                )?;
                let scales = Tensor::from_vec(descriptor.scales.clone(), (num_blocks,), device)?;
                let mut data = NF4Quantizer::dequantize_state(&packed, &scales)?
                    .to_dtype(DType::F32)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let total: usize = descriptor.shape.iter().product();
                if data.len() < total {
                    candle_core::bail!(
                        "dequantized data too small: expected at least {}, got {}",
                        total,
                        data.len()
                    );
                }
                data.truncate(total);
                let tensor = Tensor::from_vec(data, (total,), device)?
                    .reshape(descriptor.shape.as_slice())?;
                states.push(tensor);
            }
            self.residency = SessionResidency::Active(states);
        }
        self.states_clone()
    }

    fn merge_delta(
        &mut self,
        layer_index: usize,
        delta: &Tensor,
        device: &Device,
    ) -> candle_core::Result<()> {
        let mut states = self.ensure_active(device)?;
        if layer_index >= states.len() {
            candle_core::bail!(
                "layer index {layer_index} out of range for {} session layers",
                states.len()
            );
        }
        let previous = states[layer_index].clone();
        let merged = states[layer_index].add(&delta.to_dtype(DType::F32)?)?;
        if !tensor_is_finite(&merged)? {
            eprintln!(
                "[emergency] non-finite tensor detected in merge_delta; session update discarded"
            );
            states[layer_index] = previous;
            candle_core::bail!("delta merge produced non-finite values");
        }
        states[layer_index] = merged;
        self.replace_states(states);
        Ok(())
    }
}

/// Global server state shared across all request handlers.
///
/// * `pipeline` — inference pipeline; wrapped in `Mutex` because
///   generation and adaptation mutate fast-weight state in-place and must remain
///   serialized. Operations do not hold this lock across `.await` points.
/// * `sessions` — active TTT sessions keyed by UUID.  `RwLock` allows
///   multiple simultaneous GET-style reads while mutations (create, adapt,
///   checkpoint write) acquire an exclusive write lock.
#[derive(Clone)]
pub struct AppState {
    pub pipeline: Arc<Mutex<InferencePipeline>>,
    pub device: Device,
    /// Active TTT sessions keyed by UUID string.
    /// `RwLock` enables concurrent reads; mutations take an exclusive write.
    sessions: Arc<RwLock<HashMap<String, SessionData>>>,
    /// Session-layer replication sequencing guard:
    /// key = "{session_id}:{layer_index}".
    sequence_versions: Arc<RwLock<HashMap<String, SequenceState>>>,
    /// Canonical model identifier returned in API responses.
    pub model_id: String,
    /// Optional Anthropic Claude backend. When `Some`, generation is
    /// routed through Claude instead of the local Axiom-TTT pipeline.
    pub claude_backend: Arc<Option<ClaudeBackend>>,
    /// Active-compression session store: per-tenant adapted fast-weight
    /// tensors held in a lock-free DashMap. Distinct from `sessions`
    /// above (which serves the legacy `/v1/sessions` API); this store
    /// is used exclusively by the `/v1/messages` compression path.
    pub ttt_sessions: Arc<TttSessionStore>,
    /// Outbound bridge to the real Anthropic API used by the compression
    /// path. `None` when no `ANTHROPIC_API_KEY` is configured.
    pub anthropic_forwarder: Arc<Option<AnthropicForwarder>>,
    /// Static compression configuration (threshold, top-k, enabled flag).
    pub compressor_config: Arc<CompressorConfig>,
}

impl AppState {
    pub fn new(pipeline: InferencePipeline, model_id: String) -> Self {
        let device = pipeline.device().clone();
        Self {
            pipeline: Arc::new(Mutex::new(pipeline)),
            device,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            sequence_versions: Arc::new(RwLock::new(HashMap::new())),
            model_id,
            claude_backend: Arc::new(None),
            ttt_sessions: Arc::new(TttSessionStore::new()),
            anthropic_forwarder: Arc::new(None),
            compressor_config: Arc::new(CompressorConfig::default()),
        }
    }

    /// Install a Claude backend on this app state, replacing any existing one.
    pub fn with_claude_backend(mut self, backend: Option<ClaudeBackend>) -> Self {
        self.claude_backend = Arc::new(backend);
        self
    }

    /// Install an Anthropic forwarder for the compression pipeline.
    pub fn with_anthropic_forwarder(mut self, forwarder: Option<AnthropicForwarder>) -> Self {
        self.anthropic_forwarder = Arc::new(forwarder);
        self
    }

    /// Override the compressor configuration.
    pub fn with_compressor_config(mut self, config: CompressorConfig) -> Self {
        self.compressor_config = Arc::new(config);
        self
    }

    /// True iff every component the compression path needs is configured:
    /// the feature flag, the forwarder, and a key.
    pub fn compression_active(&self) -> bool {
        self.compressor_config.enabled && self.anthropic_forwarder.is_some()
    }

    fn refresh_session_metrics(&self) -> Result<(), ApiError> {
        let sessions = self
            .sessions
            .read()
            .map_err(|_| ApiError::Internal("session lock poisoned".into()))?;
        let active = sessions
            .values()
            .filter(|session| !session.is_quantized())
            .count();
        let quantized = sessions
            .values()
            .filter(|session| session.is_quantized())
            .count();
        metrics::set_active_sessions(active);
        metrics::set_quantized_sessions(quantized);
        Ok(())
    }

    fn trigger_lru_vram_budget(&self) {
        let sessions = self.sessions.clone();
        tokio::spawn(async move {
            if let Err(err) = enforce_lru_vram_budget_async(sessions).await {
                eprintln!("[emergency] LRU budget enforcement failed: {err}");
            }
        });
    }
}

async fn enforce_lru_vram_budget_async(
    sessions: Arc<RwLock<HashMap<String, SessionData>>>,
) -> std::result::Result<(), String> {
    loop {
        let candidate = {
            let sessions_guard = sessions
                .read()
                .map_err(|_| "session lock poisoned".to_string())?;
            let active = sessions_guard
                .iter()
                .filter_map(|(id, session)| {
                    if session.is_quantized() {
                        None
                    } else {
                        Some((id.clone(), session.last_used))
                    }
                })
                .collect::<Vec<_>>();
            if active.len() <= MAX_ACTIVE_VRAM_SESSIONS {
                None
            } else {
                active.into_iter().min_by_key(|(_, last_used)| *last_used)
            }
        };

        let Some((evict_session_id, baseline_last_used)) = candidate else {
            refresh_session_metrics_from_sessions(&sessions)
                .map_err(|e| format!("session metrics refresh failed: {e}"))?;
            return Ok(());
        };

        let raw_layers = {
            let sessions_guard = sessions
                .read()
                .map_err(|_| "session lock poisoned".to_string())?;
            let Some(session) = sessions_guard.get(&evict_session_id) else {
                continue;
            };
            match &session.residency {
                SessionResidency::Quantized(_) => continue,
                SessionResidency::Active(states) => states
                    .iter()
                    .map(|state| {
                        let cpu_f32 = state.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                        let shape = cpu_f32.dims().to_vec();
                        let data = cpu_f32.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
                        Ok((shape, data))
                    })
                    .collect::<candle_core::Result<Vec<_>>>()
                    .map_err(|e| format!("state staging failed: {e}"))?,
            }
        };

        let quantized_layers = spawn_blocking(move || {
            raw_layers
                .into_iter()
                .map(|(shape, data)| {
                    let total: usize = shape.iter().product();
                    let staged = Tensor::from_vec(data, (total,), &Device::Cpu)
                        .and_then(|t| t.reshape(shape.as_slice()))
                        .map_err(|e| e.to_string())?;
                    let (packed_indices, scale) =
                        NF4Quantizer::quantize_state(&staged).map_err(|e| e.to_string())?;
                    let (num_blocks, packed_width) =
                        packed_indices.dims2().map_err(|e| e.to_string())?;
                    let packed_indices = packed_indices
                        .to_dtype(DType::U8)
                        .and_then(|t| t.contiguous())
                        .and_then(|t| t.flatten_all())
                        .and_then(|t| t.to_vec1::<u8>())
                        .map_err(|e| e.to_string())?;
                    let scales = scale
                        .to_dtype(DType::F32)
                        .and_then(|t| t.contiguous())
                        .and_then(|t| t.flatten_all())
                        .and_then(|t| t.to_vec1::<f32>())
                        .map_err(|e| e.to_string())?;
                    if scales.len() != num_blocks {
                        return Err(format!(
                            "invalid scale length for shape {:?}: expected {num_blocks}, got {}",
                            shape,
                            scales.len(),
                        ));
                    }
                    Ok(NF4QuantizedDescriptor {
                        shape,
                        packed_indices,
                        scales,
                        packed_width,
                    })
                })
                .collect::<std::result::Result<Vec<NF4QuantizedDescriptor>, String>>()
        })
        .await
        .map_err(|e| format!("state offload task join failed: {e}"))?
        .map_err(|e| format!("state offload quantization failed: {e}"))?;

        let mut sessions_guard = sessions
            .write()
            .map_err(|_| "session lock poisoned".to_string())?;
        if let Some(session) = sessions_guard.get_mut(&evict_session_id) {
            let still_active = !session.is_quantized();
            let unchanged_clock = session.last_used == baseline_last_used;
            if still_active && unchanged_clock {
                session.residency = SessionResidency::Quantized(quantized_layers);
                metrics::mark_session_quantized(&evict_session_id, true);
            }
        }
    }
}

fn refresh_session_metrics_from_sessions(
    sessions: &Arc<RwLock<HashMap<String, SessionData>>>,
) -> std::result::Result<(), String> {
    let sessions_guard = sessions
        .read()
        .map_err(|_| "session lock poisoned".to_string())?;
    let active = sessions_guard
        .values()
        .filter(|session| !session.is_quantized())
        .count();
    let quantized = sessions_guard
        .values()
        .filter(|session| session.is_quantized())
        .count();
    metrics::set_active_sessions(active);
    metrics::set_quantized_sessions(quantized);
    Ok(())
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum ApiError {
    Internal(String),
    NotFound(String),
    BadRequest(String),
    Conflict(String),
    /// Upstream (Anthropic) failure. Carries the upstream status code so the
    /// client can distinguish auth/rate-limit/server errors and the message
    /// body for diagnostics.
    Upstream { status: u16, message: String },
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg).into_response(),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg).into_response(),
            ApiError::Conflict(msg) => (StatusCode::CONFLICT, msg).into_response(),
            ApiError::Upstream { status, message } => {
                // Pass the upstream status through when it's a valid client/
                // server code; otherwise surface a 502 Bad Gateway.
                let code = StatusCode::from_u16(status)
                    .unwrap_or(StatusCode::BAD_GATEWAY);
                (code, message).into_response()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible request / response types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ListModelsResponse {
    object: String,
    data: Vec<ModelInfo>,
}

// -- /v1/completions --

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    /// If provided, generation uses and updates this TTT session's W_tilde states.
    pub session_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
}

// -- /v1/chat/completions --

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    /// See [`CompletionRequest::session_id`].
    pub session_id: Option<String>,
    /// When `true`, the response is an SSE stream of `chat.completion.chunk`
    /// objects terminated by `data: [DONE]`.
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
}

// -- /v1/sessions --

#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
struct CreateSessionResponse {
    session_id: String,
    object: String,
    created: u64,
    model: String,
}

#[derive(Debug, Serialize)]
struct DeleteSessionResponse {
    session_id: String,
    deleted: bool,
}

// -- /v1/adapt --

#[derive(Debug, Deserialize)]
pub struct AdaptRequest {
    /// Text examples to adapt on.
    pub corpus: Vec<String>,
    /// Maximum number of additional inner-loop steps per token (1–4).  Defaults to 4.
    pub steps: Option<usize>,
    /// Session to adapt; creates a new session if omitted.
    pub session_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct AdaptResponse {
    session_id: String,
    object: String,
    steps_per_token: usize,
    corpus_documents: usize,
}

// -- /v1/messages (Anthropic Messages API) --

/// Anthropic accepts message ``content`` either as a bare string or as a
/// list of typed blocks. We deserialise into [`AnthropicContent::Blocks`]
/// or [`AnthropicContent::Text`] and flatten with [`AnthropicContent::to_text`].
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicInputBlock>),
}

#[derive(Debug, Deserialize)]
pub struct AnthropicInputBlock {
    #[serde(rename = "type", default)]
    block_type: String,
    #[serde(default)]
    text: String,
}

impl AnthropicContent {
    fn to_text(&self) -> String {
        match self {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => blocks
                .iter()
                .filter(|b| b.block_type == "text" || b.block_type.is_empty())
                .map(|b| b.text.clone())
                .collect::<Vec<_>>()
                .concat(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_anthropic_max_tokens")]
    pub max_tokens: usize,
    pub messages: Vec<AnthropicInputMessage>,
    #[serde(default)]
    pub system: Option<AnthropicContent>,
    #[serde(default)]
    pub session_id: Option<String>,
}

fn default_anthropic_max_tokens() -> usize {
    1024
}

#[derive(Debug, Deserialize)]
pub struct AnthropicInputMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Serialize)]
struct AnthropicOutputBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[derive(Debug, Serialize)]
struct AnthropicMessagesResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    content: Vec<AnthropicOutputBlock>,
    model: String,
    stop_reason: String,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

// -- /v1/sessions/{id}/checkpoint --

/// Serialisable representation of one W_tilde layer tensor.
#[derive(Debug, Serialize, Deserialize)]
pub struct LayerCheckpoint {
    /// Tensor shape, e.g. `[1, 4, 16, 16]`.
    pub shape: Vec<usize>,
    /// Flattened f32 values (row-major).
    pub data: Vec<f32>,
}

/// Full serialisable session checkpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionCheckpoint {
    pub session_id: String,
    pub version: u32,
    pub created_at: u64,
    pub layers: Vec<LayerCheckpoint>,
}

// ---------------------------------------------------------------------------
// Checkpoint helpers
// ---------------------------------------------------------------------------

fn tensor_to_layer_checkpoint(t: &Tensor) -> candle_core::Result<LayerCheckpoint> {
    let shape = t.dims().to_vec();
    let data = t
        .to_dtype(candle_core::DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(LayerCheckpoint { shape, data })
}

fn layer_checkpoint_to_tensor(
    lc: &LayerCheckpoint,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let total: usize = lc.shape.iter().product();
    if total != lc.data.len() {
        candle_core::bail!(
            "checkpoint shape {:?} implies {} elements but data has {}",
            lc.shape,
            total,
            lc.data.len()
        );
    }
    Tensor::from_vec(lc.data.clone(), (total,), device)?.reshape(lc.shape.as_slice())
}

fn tensor_is_finite(tensor: &Tensor) -> candle_core::Result<bool> {
    let values = tensor
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(values.into_iter().all(f32::is_finite))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /v1/models` — list available models.
async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let resp = ListModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "axiom-ttt".to_string(),
        }],
    };
    Json(resp)
}

async fn export_metrics(State(state): State<AppState>) -> Result<Response, ApiError> {
    state.refresh_session_metrics()?;
    Ok((
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        metrics::render_metrics(),
    )
        .into_response())
}

/// `POST /v1/completions` — text completion (stateless or session-aware).
async fn create_completion(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let max_tokens = req.max_tokens.unwrap_or(32);
    let model = req.model.as_deref().unwrap_or(&state.model_id).to_string();

    let text = run_generation(&state, &req.prompt, max_tokens, req.session_id.as_deref())?;
    state.trigger_lru_vram_budget();

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: unix_now(),
        model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
    }))
}

/// `POST /v1/chat/completions` — chat completion (stateless or session-aware).
///
/// When `stream: true` is set in the request body, the response is an SSE
/// stream of `chat.completion.chunk` objects (OpenAI streaming format) terminated
/// by the sentinel `data: [DONE]\n\n`.  Clients such as Open WebUI, LangChain,
/// and curl --no-buffer work without any code change.
///
/// When `stream: false` (or absent), a single JSON object is returned.
async fn create_chat_completion(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.stream.unwrap_or(false) {
        let sse = chat_completion_sse(state.clone(), req);
        state.trigger_lru_vram_budget();
        sse.into_response()
    } else {
        let json = chat_completion_json(state.clone(), req);
        state.trigger_lru_vram_budget();
        json.into_response()
    }
}

// -- non-streaming JSON path ------------------------------------------------

fn chat_completion_json(
    state: AppState,
    req: ChatCompletionRequest,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let max_tokens = req.max_tokens.unwrap_or(32);
    let model = req.model.as_deref().unwrap_or(&state.model_id).to_string();
    let prompt = messages_to_prompt(&req.messages);
    let prompt_tokens = count_prompt_tokens(&state, &prompt)?;
    let started_at = Instant::now();
    let text = run_generation(&state, &prompt, max_tokens, req.session_id.as_deref())?;
    metrics::add_prefilled_tokens(prompt_tokens);
    metrics::observe_prefill_latency(started_at.elapsed().as_secs_f64());
    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: unix_now(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: "stop".to_string(),
        }],
    }))
}

// -- SSE streaming path -----------------------------------------------------

/// Build an SSE response from a pre-generated text, streaming one word-piece
/// per event to give clients the incremental token experience.
///
/// All generation is synchronous (the inference pipeline is CPU/GPU blocking);
/// we generate the full text first, then stream the result as SSE chunks.
/// This is fully OpenAI-wire-compatible: clients that open an SSE connection
/// will see tokens arrive progressively.
fn chat_completion_sse(
    state: AppState,
    req: ChatCompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let max_tokens = req.max_tokens.unwrap_or(32);
    let model = req.model.as_deref().unwrap_or(&state.model_id).to_string();
    let prompt = messages_to_prompt(&req.messages);
    let prompt_tokens = count_prompt_tokens(&state, &prompt).unwrap_or(0);

    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = unix_now();

    let started_at = Instant::now();
    let generation_result = run_generation(&state, &prompt, max_tokens, req.session_id.as_deref());
    metrics::add_prefilled_tokens(prompt_tokens);
    metrics::observe_prefill_latency(started_at.elapsed().as_secs_f64());

    // Build the event sequence.  On error, emit a single error event.
    let events: Vec<Result<Event, Infallible>> = match generation_result {
        Err(api_err) => {
            let body = match api_err {
                ApiError::Internal(m)
                | ApiError::NotFound(m)
                | ApiError::BadRequest(m)
                | ApiError::Conflict(m) => m,
                ApiError::Upstream { status, message } => {
                    format!("upstream {status}: {message}")
                }
            };
            vec![Ok(Event::default().data(format!("error: {body}")))]
        }
        Ok(text) => {
            // Split into word-pieces lazily; split_inclusive yields &str slices
            // into `text` — no extra String allocation per piece.
            let pieces: Vec<&str> = text.split_inclusive(' ').collect();

            let mut events: Vec<Result<Event, Infallible>> = Vec::with_capacity(pieces.len() + 2);

            for piece in pieces {
                let chunk = serde_json::json!({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": piece},
                        "finish_reason": null
                    }]
                });
                events.push(Ok(Event::default().data(chunk.to_string())));
            }

            // Final chunk: stop signal with empty delta.
            let stop_chunk = serde_json::json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            });
            events.push(Ok(Event::default().data(stop_chunk.to_string())));
            // OpenAI termination sentinel.
            events.push(Ok(Event::default().data("[DONE]")));
            events
        }
    };

    Sse::new(stream::iter(events)).keep_alive(KeepAlive::default())
}

/// `POST /v1/sessions` — create a new persistent TTT session.
async fn create_session(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let model = req.model.as_deref().unwrap_or(&state.model_id).to_string();

    // Initialise zeroed W_tilde states.
    let states = {
        let pipeline = state
            .pipeline
            .lock()
            .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
        pipeline
            .init_session_states()
            .map_err(|e| ApiError::Internal(format!("state init failed: {e}")))?
    };

    let session_id = Uuid::new_v4().to_string();
    let now = unix_now();
    {
        let mut sessions = state
            .sessions
            .write()
            .map_err(|_| ApiError::Internal("session lock poisoned".into()))?;
        sessions.insert(session_id.clone(), SessionData::new_active(states, now));
    }
    metrics::register_session(&session_id);
    state.refresh_session_metrics()?;
    state.trigger_lru_vram_budget();

    Ok(Json(CreateSessionResponse {
        session_id,
        object: "session".to_string(),
        created: now,
        model,
    }))
}

/// `DELETE /v1/sessions/{id}` — delete a session and free its W_tilde memory.
async fn delete_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let mut sessions = state
        .sessions
        .write()
        .map_err(|_| ApiError::Internal("session lock poisoned".into()))?;

    let deleted = sessions.remove(&session_id).is_some();
    drop(sessions);
    if deleted {
        metrics::remove_session(&session_id);
        let mut sequence_versions = state
            .sequence_versions
            .write()
            .map_err(|_| ApiError::Internal("sequence lock poisoned".into()))?;
        let prefix = format!("{session_id}:");
        sequence_versions.retain(|key, _| !key.starts_with(&prefix));
    }
    state.refresh_session_metrics()?;
    Ok(Json(DeleteSessionResponse {
        session_id,
        deleted,
    }))
}

/// `POST /v1/adapt` — TTT adaptation over a text corpus.
async fn adapt(
    State(state): State<AppState>,
    Json(req): Json<AdaptRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if req.corpus.is_empty() {
        return Err(ApiError::BadRequest(
            "corpus must contain at least one document".into(),
        ));
    }

    let corpus_len = req.corpus.len();
    let steps_per_token = req.steps.unwrap_or(4).clamp(1, 4);
    let corpus_tokens = count_corpus_tokens(&state, &req.corpus)?;

    // Resolve or create a session.
    let (session_id, initial_states) =
        resolve_or_create_session(&state, req.session_id.as_deref())?;

    let started_at = Instant::now();
    let updated_states = {
        let pipeline = state
            .pipeline
            .lock()
            .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
        pipeline
            .adapt_on_corpus_with_steps(&req.corpus, initial_states, steps_per_token)
            .map_err(|e| ApiError::Internal(format!("adapt failed: {e}")))?
    };
    metrics::add_prefilled_tokens(corpus_tokens);
    metrics::observe_prefill_latency(started_at.elapsed().as_secs_f64());

    // Persist updated states back into the session (exclusive write lock).
    {
        let mut sessions = state
            .sessions
            .write()
            .map_err(|_| ApiError::Internal("session lock poisoned".into()))?;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.replace_states(updated_states);
            session.last_used = unix_now();
            metrics::mark_session_quantized(&session_id, false);
        }
    }
    state.refresh_session_metrics()?;
    state.trigger_lru_vram_budget();

    Ok(Json(AdaptResponse {
        session_id,
        object: "adapt".to_string(),
        steps_per_token,
        corpus_documents: corpus_len,
    }))
}

/// `GET /v1/sessions/{id}/checkpoint` — export session W_tilde as JSON.
async fn get_checkpoint(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let mut sessions = state
        .sessions
        .write()
        .map_err(|_| ApiError::Internal("session lock poisoned".into()))?;

    let session = sessions
        .get_mut(&session_id)
        .ok_or_else(|| ApiError::NotFound(format!("session '{session_id}' not found")))?;
    let created_at = session.created_at;

    let layers = session
        .ensure_active(&state.device)
        .map_err(|e| ApiError::Internal(format!("session export failed: {e}")))?
        .iter()
        .map(tensor_to_layer_checkpoint)
        .collect::<candle_core::Result<Vec<_>>>()
        .map_err(|e| ApiError::Internal(format!("serialisation failed: {e}")))?;
    metrics::mark_session_quantized(&session_id, false);
    drop(sessions);
    state.refresh_session_metrics()?;

    Ok(Json(SessionCheckpoint {
        session_id: session_id.clone(),
        version: 1,
        created_at,
        layers,
    }))
}

/// `PUT /v1/sessions/{id}/checkpoint` — restore session W_tilde from JSON.
async fn put_checkpoint(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(checkpoint): Json<SessionCheckpoint>,
) -> Result<impl IntoResponse, ApiError> {
    if checkpoint.version != 1 {
        return Err(ApiError::BadRequest(format!(
            "unsupported checkpoint version {}",
            checkpoint.version
        )));
    }

    let states = checkpoint
        .layers
        .iter()
        .map(|lc| layer_checkpoint_to_tensor(lc, &state.device))
        .collect::<candle_core::Result<Vec<_>>>()
        .map_err(|e| ApiError::Internal(format!("deserialisation failed: {e}")))?;

    let now = unix_now();
    let mut sessions = state
        .sessions
        .write()
        .map_err(|_| ApiError::Internal("session lock poisoned".into()))?;

    sessions
        .entry(session_id.clone())
        .and_modify(|s| {
            s.replace_states(states.clone());
            s.last_used = now;
        })
        .or_insert_with(|| SessionData::new_active(states, now));
    drop(sessions);
    metrics::register_session(&session_id);
    metrics::mark_session_quantized(&session_id, false);
    state.refresh_session_metrics()?;
    state.trigger_lru_vram_budget();

    Ok(Json(CreateSessionResponse {
        session_id,
        object: "session".to_string(),
        created: now,
        model: state.model_id.clone(),
    }))
}

async fn cluster_sync(
    State(state): State<AppState>,
    Json(payload): Json<StateDeltaUpdate>,
) -> Result<impl IntoResponse, ApiError> {
    let delta = {
        let mut tensors =
            candle_core::safetensors::load_buffer(&payload.delta_bytes, &state.device)
                .map_err(|e| ApiError::BadRequest(format!("invalid delta payload: {e}")))?;
        tensors
            .remove("tensor")
            .ok_or_else(|| ApiError::BadRequest("delta payload missing 'tensor' key".into()))?
    };

    let sequence_key = format!("{}:{}", payload.session_id, payload.layer_index);
    let mut sequence_versions = state
        .sequence_versions
        .write()
        .map_err(|_| ApiError::Internal("sequence lock poisoned".into()))?;
    if let Some(existing) = sequence_versions.get(&sequence_key) {
        if payload.sequence_version <= existing.version {
            return Err(ApiError::Conflict(format!(
                "stale delta rejected: incoming sequence_version={} current={} timestamp={} current_timestamp={}",
                payload.sequence_version,
                existing.version,
                payload.timestamp,
                existing.timestamp
            )));
        }
    }

    let mut sessions = state
        .sessions
        .write()
        .map_err(|_| ApiError::Internal("session lock poisoned".into()))?;
    let session = sessions
        .get_mut(&payload.session_id)
        .ok_or_else(|| ApiError::NotFound(format!("session '{}' not found", payload.session_id)))?;
    session
        .merge_delta(payload.layer_index, &delta, &state.device)
        .map_err(|e| ApiError::BadRequest(format!("delta merge failed: {e}")))?;
    session.last_used = unix_now();
    metrics::mark_session_quantized(&payload.session_id, false);
    sequence_versions.insert(
        sequence_key,
        SequenceState {
            version: payload.sequence_version,
            timestamp: payload.timestamp,
        },
    );
    drop(sequence_versions);
    drop(sessions);
    state.refresh_session_metrics()?;
    state.trigger_lru_vram_budget();

    Ok(StatusCode::ACCEPTED)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Concatenate chat messages into a single prompt string.
fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn count_prompt_tokens(state: &AppState, prompt: &str) -> Result<usize, ApiError> {
    let pipeline = state
        .pipeline
        .lock()
        .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
    Ok(pipeline.token_count(prompt))
}

fn count_corpus_tokens(state: &AppState, corpus: &[String]) -> Result<usize, ApiError> {
    let pipeline = state
        .pipeline
        .lock()
        .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
    Ok(corpus.iter().map(|text| pipeline.token_count(text)).sum())
}

/// Run generation, optionally using and updating a named session.
///
/// When a Claude backend is installed on [`AppState`], generation is
/// routed to Anthropic and the local TTT lifecycle is skipped. Sessions
/// still exist so the wire contract holds, but `/v1/adapt` cannot
/// influence the remote model.
fn run_generation(
    state: &AppState,
    prompt: &str,
    max_tokens: usize,
    session_id: Option<&str>,
) -> Result<String, ApiError> {
    if let Some(backend) = state.claude_backend.as_ref() {
        return backend
            .generate(prompt, max_tokens)
            .map_err(ApiError::Internal);
    }

    match session_id {
        None => {
            // Stateless generation.
            let pipeline = state
                .pipeline
                .lock()
                .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
            pipeline
                .generate(prompt, max_tokens)
                .map_err(|e| ApiError::Internal(format!("generation failed: {e}")))
        }
        Some(sid) => {
            // Stateful generation — load states (write lock to allow dequantization), generate, write back.
            let initial_states = {
                let mut sessions = state
                    .sessions
                    .write()
                    .map_err(|_| ApiError::Internal("session lock poisoned".into()))?;
                let session = sessions
                    .get_mut(sid)
                    .ok_or_else(|| ApiError::NotFound(format!("session '{sid}' not found")))?;
                let initial_states = session.ensure_active(&state.device).map_err(|e| {
                    ApiError::Internal(format!("session dequantization failed: {e}"))
                })?;
                session.last_used = unix_now();
                metrics::mark_session_quantized(sid, false);
                initial_states
            };
            state.refresh_session_metrics()?;

            let (text, updated_states) = {
                let pipeline = state
                    .pipeline
                    .lock()
                    .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
                pipeline
                    .generate_with_session(prompt, max_tokens, initial_states)
                    .map_err(|e| ApiError::Internal(format!("generation failed: {e}")))?
            };

            {
                let mut sessions = state
                    .sessions
                    .write()
                    .map_err(|_| ApiError::Internal("session lock poisoned".into()))?;
                if let Some(session) = sessions.get_mut(sid) {
                    session.replace_states(updated_states);
                    session.last_used = unix_now();
                    metrics::mark_session_quantized(sid, false);
                }
            }
            state.refresh_session_metrics()?;

            Ok(text)
        }
    }
}

/// Resolve an existing session or create a fresh one, returning `(session_id, states)`.
fn resolve_or_create_session(
    state: &AppState,
    session_id: Option<&str>,
) -> Result<(String, Vec<Tensor>), ApiError> {
    if let Some(sid) = session_id {
        let mut sessions = state
            .sessions
            .write()
            .map_err(|_| ApiError::Internal("session lock poisoned".into()))?;
        let session = sessions
            .get_mut(sid)
            .ok_or_else(|| ApiError::NotFound(format!("session '{sid}' not found")))?;
        let states = session
            .ensure_active(&state.device)
            .map_err(|e| ApiError::Internal(format!("session dequantization failed: {e}")))?;
        session.last_used = unix_now();
        metrics::mark_session_quantized(sid, false);
        drop(sessions);
        state.refresh_session_metrics()?;
        Ok((sid.to_string(), states))
    } else {
        // Auto-create a transient session.
        let states = {
            let pipeline = state
                .pipeline
                .lock()
                .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
            pipeline
                .init_session_states()
                .map_err(|e| ApiError::Internal(format!("state init failed: {e}")))?
        };
        let new_id = Uuid::new_v4().to_string();
        let now = unix_now();
        let mut sessions = state
            .sessions
            .write()
            .map_err(|_| ApiError::Internal("session lock poisoned".into()))?;
        sessions.insert(new_id.clone(), SessionData::new_active(states.clone(), now));
        drop(sessions);
        metrics::register_session(&new_id);
        state.refresh_session_metrics()?;
        Ok((new_id, states))
    }
}

/// `POST /v1/messages` — Anthropic Messages API endpoint.
///
/// Drop-in target for the Anthropic SDK and Claude Code: clients that
/// point `ANTHROPIC_BASE_URL` at this server receive responses in the
/// native Messages format regardless of whether the local Axiom-TTT
/// pipeline, a Claude backend, or the active-compression upstream
/// produced them.
///
/// When `state.compression_active()` is true, the handler:
/// 1. Partitions the inbound messages into heavy context (above the
///    configured token threshold) and the surviving user query.
/// 2. Spawns a blocking task that tokenises the heavy context, runs it
///    through the TTT layer stack to mutate W̃ in-place, and extracts
///    a [`MemoryFingerprint`] via an associative recall pass.
/// 3. Rebuilds the outbound JSON payload with the heavy context stripped
///    and the fingerprint prepended to the surviving user turn.
/// 4. POSTs the lean payload to the real Anthropic API and returns the
///    response verbatim.
async fn create_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    // A client (e.g. the Claude CLI) can pin a deterministic TTT session by
    // sending `X-Axiom-Session-Id`. This takes precedence over any body
    // `session_id`, since real Anthropic clients never put session_id in the
    // body. Both fall back to a transient UUID when absent.
    let session_override = headers
        .get("x-axiom-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    if state.compression_active() {
        // Capture the client's own credentials so we can relay them upstream.
        // This is what makes the proxy work for a Claude *subscription*
        // (OAuth bearer) as well as for raw API-key clients — the proxy never
        // needs to hold a key of its own.
        let header_str = |name: &str| {
            headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        };
        let client_auth = ClientAuth {
            authorization: header_str("authorization"),
            x_api_key: header_str("x-api-key"),
            anthropic_version: header_str("anthropic-version"),
            anthropic_beta: header_str("anthropic-beta"),
        };
        match compressed_messages_path(&state, &body, session_override.as_deref(), &client_auth)
            .await
        {
            Ok(value) => return Json(value).into_response(),
            Err(err) => return err.into_response(),
        }
    }

    let req: AnthropicMessagesRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return ApiError::BadRequest(format!("invalid /v1/messages body: {e}")).into_response(),
    };
    local_messages_path(state, req).map_or_else(
        |e| e.into_response(),
        |json| json.into_response(),
    )
}

fn local_messages_path(
    state: AppState,
    req: AnthropicMessagesRequest,
) -> Result<Json<AnthropicMessagesResponse>, ApiError> {
    let model = req.model.clone().unwrap_or_else(|| state.model_id.clone());
    let system_text = req.system.as_ref().map(|c| c.to_text());

    let text = match state.claude_backend.as_ref() {
        Some(backend) => {
            let turns: Vec<ChatTurn> = req
                .messages
                .iter()
                .map(|m| ChatTurn {
                    role: m.role.clone(),
                    content: m.content.to_text(),
                })
                .collect();
            backend
                .generate_chat(&turns, req.max_tokens, system_text.clone())
                .map_err(ApiError::Internal)?
        }
        None => {
            let mut prompt_parts: Vec<String> = Vec::new();
            if let Some(ref sys) = system_text {
                prompt_parts.push(format!("system: {sys}"));
            }
            for msg in &req.messages {
                prompt_parts.push(format!("{}: {}", msg.role, msg.content.to_text()));
            }
            let prompt = prompt_parts.join("\n");
            run_generation(&state, &prompt, req.max_tokens, req.session_id.as_deref())?
        }
    };

    let input_tokens: usize = req
        .messages
        .iter()
        .map(|m| m.content.to_text().split_whitespace().count())
        .sum();
    let output_tokens = text.split_whitespace().count();

    Ok(Json(AnthropicMessagesResponse {
        id: format!("msg_{}", Uuid::new_v4().simple()),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![AnthropicOutputBlock {
            block_type: "text".to_string(),
            text,
        }],
        model,
        stop_reason: "end_turn".to_string(),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens,
            output_tokens,
        },
    }))
}

/// Active-compression code path: partition → adapt → recall → forward.
async fn compressed_messages_path(
    state: &AppState,
    body: &Value,
    session_override: Option<&str>,
    client_auth: &ClientAuth,
) -> Result<Value, ApiError> {
    let forwarder = state
        .anthropic_forwarder
        .as_ref()
        .as_ref()
        .ok_or_else(|| ApiError::Internal("compression active but no forwarder".into()))?
        .clone();

    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| ApiError::BadRequest("messages[] required".into()))?;

    let cfg = state.compressor_config.clone();
    let threshold = cfg.heavy_message_threshold_tokens;
    let top_k = cfg.recall_top_k;

    let partitioned = partition_messages(&messages, threshold, whitespace_token_count);

    // Resolve / create the TTT session. Precedence: the X-Axiom-Session-Id
    // header (passed in as session_override), then a body `session_id`, then
    // a minted transient UUID. Persistent compression benefits accrue only
    // when the caller pins a stable id via one of the first two.
    let session_id = session_override
        .map(str::to_string)
        .or_else(|| {
            body.get("session_id")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| format!("transient-{}", Uuid::new_v4()));

    let started = Instant::now();
    let log_heavy_count = partitioned.heavy_context.len();
    let log_heavy_tokens: usize = partitioned
        .heavy_context
        .iter()
        .map(|c| c.token_count)
        .sum();

    // Tokenise the surviving user query (for the associative recall pass)
    // alongside the heavy context — both happen inside spawn_blocking so
    // we don't stall the async runtime on the gradient loop.
    let user_query_text = partitioned
        .target_user_index
        .and_then(|idx| partitioned.surviving.get(idx))
        .and_then(|m| m.get("content"))
        .map(content_to_text)
        .unwrap_or_default();
    let heavy_combined = partitioned
        .heavy_context
        .iter()
        .map(|c| c.text.clone())
        .collect::<Vec<_>>()
        .join("\n\n");

    let fingerprint = if partitioned.heavy_context.is_empty() {
        // Nothing to ingest — emit a zero-context fingerprint so the
        // outbound payload still carries the schema marker.
        empty_fingerprint(state, &session_id, started)?
    } else {
        let pipeline_arc = state.pipeline.clone();
        let store = state.ttt_sessions.clone();
        let session_id_clone = session_id.clone();
        let heavy_clone = heavy_combined.clone();
        let query_clone = user_query_text.clone();

        // Spawn the compute-heavy loop on a blocking thread so the Tokio
        // runtime keeps serving other requests during the gradient steps.
        let fp_result: Result<MemoryFingerprint, ApiError> = tokio::task::spawn_blocking(move || {
            let pipeline = pipeline_arc
                .lock()
                .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
            let session = store.get_or_create(&session_id_clone, &pipeline).map_err(|e| {
                ApiError::Internal(format!("session allocation failed: {e}"))
            })?;
            // tokio::sync::Mutex::blocking_lock is safe in spawn_blocking.
            let mut session_states = session.blocking_lock();

            let context_tokens: Vec<u32> = pipeline.encode_text(&heavy_clone);
            adapt_session_blocking(&pipeline, &mut session_states, &context_tokens).map_err(
                |e| ApiError::Internal(format!("TTT adapt failed: {e}")),
            )?;

            let query_tokens: Vec<u32> = pipeline.encode_text(&query_clone);
            let fingerprint = extract_memory_vector_blocking(
                &pipeline,
                &mut session_states,
                &query_tokens,
                &session_id_clone,
                context_tokens.len(),
                started,
                top_k,
            )
            .map_err(|e| ApiError::Internal(format!("memory extraction failed: {e}")))?;
            Ok(fingerprint)
        })
        .await
        .map_err(|e| ApiError::Internal(format!("blocking task join failed: {e}")))?;
        fp_result?
    };

    eprintln!(
        "[axiom-ttt] compressed session={} heavy_msgs={} heavy_tokens~{} recall_norm={:.3} elapsed_ms={}",
        fingerprint.session_id,
        log_heavy_count,
        log_heavy_tokens,
        fingerprint.recall_norm,
        fingerprint.elapsed_ms,
    );

    let outbound = build_compressed_payload(body, &fingerprint, &partitioned);
    // Strip session_id (and any other Axiom extensions) from the upstream payload.
    let mut outbound = outbound;
    if let Some(obj) = outbound.as_object_mut() {
        obj.remove("session_id");
    }

    forwarder
        .forward_messages_json(&outbound, client_auth)
        .await
        .map_err(|e| match e {
            // Surface the real upstream status (401/429/5xx) to the client.
            ForwarderError::Upstream { status, body } => ApiError::Upstream {
                status,
                message: format!("anthropic upstream {status}: {body}"),
            },
            // No credential at all → 401 so the client knows to authenticate.
            ForwarderError::MissingAuth => ApiError::Upstream {
                status: StatusCode::UNAUTHORIZED.as_u16(),
                message: format!("{e}"),
            },
            // Network/decode failures mean we never got a usable response →
            // 502 Bad Gateway rather than a misleading 500.
            other => ApiError::Upstream {
                status: StatusCode::BAD_GATEWAY.as_u16(),
                message: format!("anthropic upstream call failed: {other}"),
            },
        })
}

fn empty_fingerprint(
    state: &AppState,
    session_id: &str,
    started: Instant,
) -> Result<MemoryFingerprint, ApiError> {
    let pipeline = state
        .pipeline
        .lock()
        .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
    let n_layers = pipeline.model().config.n_layers;
    let d_model = pipeline.model().config.d_model;
    Ok(MemoryFingerprint {
        schema: "axiom-ttt-context-fingerprint/v2".to_string(),
        session_id: session_id.to_string(),
        context_tokens_processed: 0,
        n_layers,
        d_model,
        state_hash: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
            .to_string(),
        layer_frobenius_norms: vec![0.0; n_layers],
        recall_norm: 0.0,
        recall_l1: 0.0,
        recall_top_k_indices: Vec::new(),
        recall_top_k_decoded: String::new(),
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn content_to_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b: &Value| {
                b.get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// `GET /v1/ttt/sessions` — count of active TTT compression sessions.
async fn ttt_sessions_stats(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "active_sessions": state.ttt_sessions.len(),
        "compression_active": state.compression_active(),
        "threshold_tokens": state.compressor_config.heavy_message_threshold_tokens,
        "recall_top_k": state.compressor_config.recall_top_k,
    }))
}

/// `DELETE /v1/ttt/sessions/:id` — drop the W̃ tensors for one session.
async fn ttt_session_drop(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let removed = state.ttt_sessions.drop_session(&id);
    Json(serde_json::json!({ "session_id": id, "removed": removed }))
}

/// `DELETE /v1/ttt/sessions` — drop every TTT session.
async fn ttt_sessions_clear(State(state): State<AppState>) -> impl IntoResponse {
    state.ttt_sessions.clear();
    Json(serde_json::json!({ "cleared": true }))
}

// ---------------------------------------------------------------------------
// Router construction
// ---------------------------------------------------------------------------

/// Build the axum Router with all API routes attached.
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/metrics", get(export_metrics))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(create_completion))
        .route("/v1/chat/completions", post(create_chat_completion))
        .route("/v1/messages", post(create_message))
        .route("/v1/cluster/sync", post(cluster_sync))
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions/:id", delete(delete_session))
        .route("/v1/adapt", post(adapt))
        .route("/v1/sessions/:id/checkpoint", get(get_checkpoint))
        .route("/v1/sessions/:id/checkpoint", put(put_checkpoint))
        .route("/v1/ttt/sessions", get(ttt_sessions_stats))
        .route("/v1/ttt/sessions", delete(ttt_sessions_clear))
        .route("/v1/ttt/sessions/:id", delete(ttt_session_drop))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Start the HTTP server and block until it is stopped.
pub async fn run_server(
    host: &str,
    port: u16,
    config: AxiomConfig,
    checkpoint_path: &str,
    device: Device,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use crate::config::DEFAULT_CHECKPOINT_PATH;

    println!("[*] Initializing system sanity check prior to binding network sockets...");

    let pipeline = if checkpoint_path == DEFAULT_CHECKPOINT_PATH {
        InferencePipeline::new(config.clone(), device)
    } else {
        InferencePipeline::with_checkpoint(config.clone(), device, checkpoint_path)
    }
    .map_err(|e| format!("failed to assemble inference pipeline: {e}"))?;

    println!(
        "[+] Sanity check passed. safetensors matrix dimensions align perfectly with {} layers.",
        config.n_layers
    );

    let model_id = "axiom-ttt-v1".to_string();
    let claude_backend = ClaudeBackend::from_env();
    if let Some(ref backend) = claude_backend {
        println!(
            "[+] Claude backend active — generation routed to model={} \
             (TTT adapt is a no-op in this mode)",
            backend.model()
        );
    }
    let compressor_config = CompressorConfig::from_env();
    let anthropic_forwarder = if compressor_config.enabled {
        AnthropicForwarder::from_env()
    } else {
        None
    };
    if compressor_config.enabled {
        match anthropic_forwarder.as_ref() {
            Some(fwd) => {
                println!(
                    "[+] Active-compression mode ON — heavy messages (>={} tokens) \
                     will be absorbed locally via TTT, then forwarded with a dense \
                     fingerprint to Anthropic (top_k={})",
                    compressor_config.heavy_message_threshold_tokens,
                    compressor_config.recall_top_k,
                );
                if fwd.has_own_key() {
                    println!(
                        "[+] Upstream auth: proxy-owned ANTHROPIC_API_KEY (injected as \
                         x-api-key when the client sends none)."
                    );
                } else {
                    println!(
                        "[+] Upstream auth: PASSTHROUGH — no proxy key set; the client's \
                         own Authorization/x-api-key headers are relayed upstream. This is \
                         the correct mode for a Claude subscription (OAuth via Claude Code)."
                    );
                }
            }
            None => println!(
                "[!] Active-compression enabled but the forwarder failed to construct — \
                 the compression path will be skipped and requests will fall back \
                 to the local pipeline"
            ),
        }
    }
    let state = AppState::new(pipeline, model_id)
        .with_claude_backend(claude_backend)
        .with_anthropic_forwarder(anthropic_forwarder)
        .with_compressor_config(compressor_config);
    let app = create_router(state);

    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    println!("[+] Axiom-TTT server listening on http://{host}:{port}");
    println!("[+] OpenAI- and Anthropic-compatible API endpoints:");
    println!("      GET  /metrics");
    println!("      GET  /v1/models");
    println!("      POST /v1/completions");
    println!("      POST /v1/chat/completions         (stream:true for SSE)");
    println!("      POST /v1/messages                 (Anthropic Messages API)");
    println!("      POST /v1/cluster/sync            (distributed delta merge)");
    println!("      POST /v1/sessions                 (create TTT session)");
    println!("      POST /v1/adapt                    (in-place TTT adaptation)");
    println!("      GET  /v1/sessions/{{id}}/checkpoint");
    println!("      PUT  /v1/sessions/{{id}}/checkpoint");

    axum::serve(listener, app).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use candle_core::{DType, Device};
    use tower::ServiceExt;

    fn build_pipeline() -> InferencePipeline {
        use crate::config::AxiomConfig;
        use crate::inference::InferencePipeline;

        let config = AxiomConfig {
            d_model: 16,
            n_layers: 1,
            vocab_size: 64,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        };
        InferencePipeline::new(config, Device::Cpu).expect("pipeline init")
    }

    /// Build AppState outside the async executor.
    ///
    /// `reqwest::blocking::Client` (used inside `JitContextStreamer`) creates a
    /// temporary tokio runtime during `build()` and drops it before returning.
    /// Dropping a runtime while already inside a tokio runtime panics.
    /// `spawn_blocking` moves that work to a thread-pool thread where blocking
    /// operations are allowed.
    async fn make_test_state() -> AppState {
        let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
        AppState::new(pipeline, "axiom-ttt-test".to_string())
    }

    /// Drop the pipeline `Arc` on a blocking thread for the same reason.
    async fn safe_drop(arc: std::sync::Arc<std::sync::Mutex<crate::inference::InferencePipeline>>) {
        tokio::task::spawn_blocking(move || drop(arc))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_list_models_returns_200() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let req = Request::builder()
            .method(Method::GET)
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_create_session_returns_session_id() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/sessions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["session_id"].is_string());
        assert_eq!(json["object"], "session");
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_delete_unknown_session_returns_deleted_false() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let req = Request::builder()
            .method(Method::DELETE)
            .uri("/v1/sessions/nonexistent-id")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["deleted"], false);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_adapt_requires_nonempty_corpus() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/adapt")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"corpus":[]}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_adapt_reuses_session_and_updates_metrics() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state.clone());
        let session_id = "adapt-session".to_string();
        let now = unix_now();
        let initial_states = {
            let pipeline = state.pipeline.lock().unwrap();
            pipeline
                .init_session_states()
                .unwrap()
                .into_iter()
                .map(|tensor| Tensor::ones(tensor.dims(), DType::F32, &state.device).unwrap())
                .collect::<Vec<_>>()
        };
        {
            let mut sessions = state.sessions.write().unwrap();
            sessions.insert(
                session_id.clone(),
                SessionData::new_active(initial_states.clone(), now),
            );
        }
        metrics::register_session(&session_id);
        state.refresh_session_metrics().unwrap();
        let tokens_before = crate::metrics::COUNTER_TOTAL_TOKENS_PREFILLED
            .load(std::sync::atomic::Ordering::Relaxed);

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/adapt")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "corpus": ["hello adaptation"],
                    "session_id": session_id,
                })
                .to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["session_id"], session_id);

        {
            let sessions = state.sessions.read().unwrap();
            let session = sessions.get("adapt-session").unwrap();
            let session_state_count = match &session.residency {
                SessionResidency::Active(states) => states.len(),
                SessionResidency::Quantized(states) => states.len(),
            };
            assert_eq!(session_state_count, initial_states.len());
            assert!(session.last_used >= now);
        }
        let tokens_after = crate::metrics::COUNTER_TOTAL_TOKENS_PREFILLED
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(tokens_after > tokens_before);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_chat_completion_returns_200() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let body = r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":4}"#;
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        safe_drop(pipeline_arc).await;
    }

    /// Verify that `stream: true` produces an SSE response (text/event-stream
    /// Content-Type header and body containing `[DONE]`).
    #[tokio::test]
    async fn test_chat_completion_stream_returns_sse() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let body = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"stream":true}"#;
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.contains("text/event-stream"),
            "expected SSE content-type, got: {ct}"
        );

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = std::str::from_utf8(&bytes).unwrap();
        assert!(
            body_str.contains("[DONE]"),
            "SSE body must contain [DONE] sentinel"
        );
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_metrics_endpoint_renders_prometheus_text() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let req = Request::builder()
            .method(Method::GET)
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("axiom_total_tokens_prefilled"));
        assert!(body.contains("axiom_prefill_latency_seconds_bucket"));
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_cluster_sync_merges_delta_into_layer_state() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state.clone());
        let session_id = "cluster-session".to_string();
        let now = unix_now();
        let states = {
            let pipeline = state.pipeline.lock().unwrap();
            pipeline.init_session_states().unwrap()
        };
        {
            let mut sessions = state.sessions.write().unwrap();
            sessions.insert(session_id.clone(), SessionData::new_active(states, now));
        }
        metrics::register_session(&session_id);
        state.refresh_session_metrics().unwrap();

        let delta = Tensor::ones((16usize, 16usize), DType::F32, &state.device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let delta_bytes = safetensors::serialize([("tensor", &delta)], &None).unwrap();
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/cluster/sync")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&StateDeltaUpdate {
                    session_id: session_id.clone(),
                    layer_index: 0,
                    sequence_version: 1,
                    timestamp: unix_now() as i64,
                    delta_bytes,
                })
                .unwrap(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);

        let layer_sum = {
            let sessions = state.sessions.read().unwrap();
            let session = sessions.get(&session_id).unwrap();
            let mut layers = session.states_clone().unwrap();
            let layer = layers.remove(0);
            layer.sum_all().unwrap().to_scalar::<f32>().unwrap()
        };
        assert!(layer_sum > 0.0);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_cluster_sync_rejects_out_of_order_delta() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state.clone());
        let session_id = "cluster-order-session".to_string();
        let now = unix_now();
        let states = {
            let pipeline = state.pipeline.lock().unwrap();
            pipeline.init_session_states().unwrap()
        };
        {
            let mut sessions = state.sessions.write().unwrap();
            sessions.insert(session_id.clone(), SessionData::new_active(states, now));
        }
        metrics::register_session(&session_id);
        state.refresh_session_metrics().unwrap();

        let delta = Tensor::ones((16usize, 16usize), DType::F32, &state.device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let delta_bytes = safetensors::serialize([("tensor", &delta)], &None).unwrap();
        let first_req = Request::builder()
            .method(Method::POST)
            .uri("/v1/cluster/sync")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&StateDeltaUpdate {
                    session_id: session_id.clone(),
                    layer_index: 0,
                    sequence_version: 2,
                    timestamp: unix_now() as i64,
                    delta_bytes: delta_bytes.clone(),
                })
                .unwrap(),
            ))
            .unwrap();
        let first_resp = app.clone().oneshot(first_req).await.unwrap();
        assert_eq!(first_resp.status(), StatusCode::ACCEPTED);

        let stale_req = Request::builder()
            .method(Method::POST)
            .uri("/v1/cluster/sync")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&StateDeltaUpdate {
                    session_id: session_id.clone(),
                    layer_index: 0,
                    sequence_version: 1,
                    timestamp: unix_now() as i64 - 1,
                    delta_bytes,
                })
                .unwrap(),
            ))
            .unwrap();
        let stale_resp = app.oneshot(stale_req).await.unwrap();
        assert_eq!(stale_resp.status(), StatusCode::CONFLICT);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_quantized_session_dequantizes_on_chat_path() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let session_id = "quantized-session".to_string();
        let now = unix_now();
        let states = {
            let pipeline = state.pipeline.lock().unwrap();
            pipeline.init_session_states().unwrap()
        };
        {
            let mut sessions = state.sessions.write().unwrap();
            let mut session = SessionData::new_active(states, now);
            let active = match &session.residency {
                SessionResidency::Active(active) => active.clone(),
                SessionResidency::Quantized(_) => Vec::new(),
            };
            let descriptors = active
                .iter()
                .map(|tensor| {
                    let shape = tensor.dims().to_vec();
                    let (packed, scale) = NF4Quantizer::quantize_state(tensor).unwrap();
                    let (num_blocks, packed_width) = packed.dims2().unwrap();
                    let packed_indices = packed
                        .to_dtype(DType::U8)
                        .unwrap()
                        .contiguous()
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<u8>()
                        .unwrap();
                    let scales = scale
                        .to_dtype(DType::F32)
                        .unwrap()
                        .contiguous()
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    assert_eq!(scales.len(), num_blocks);
                    NF4QuantizedDescriptor {
                        shape,
                        packed_indices,
                        scales,
                        packed_width,
                    }
                })
                .collect::<Vec<_>>();
            session.residency = SessionResidency::Quantized(descriptors);
            sessions.insert(session_id.clone(), session);
        }
        metrics::register_session(&session_id);
        metrics::mark_session_quantized(&session_id, true);
        state.refresh_session_metrics().unwrap();

        let app = create_router(state.clone());
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "session_id": session_id,
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        {
            let sessions = state.sessions.read().unwrap();
            let session = sessions.get("quantized-session").unwrap();
            assert!(!session.is_quantized());
        }
        safe_drop(pipeline_arc).await;
    }
}
