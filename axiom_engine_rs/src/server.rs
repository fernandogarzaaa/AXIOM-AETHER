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
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use candle_core::{DType, Device, Tensor};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::cluster::StateDeltaUpdate;
use crate::config::AxiomConfig;
use crate::inference::InferencePipeline;
use crate::metrics;

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
    Quantized(Vec<Vec<u8>>),
}

struct SessionData {
    residency: SessionResidency,
    created_at: u64,
    last_used: u64,
    model: String,
}

impl SessionData {
    fn new_active(states: Vec<Tensor>, created_at: u64, model: String) -> Self {
        Self {
            residency: SessionResidency::Active(states),
            created_at,
            last_used: created_at,
            model,
        }
    }

    fn replace_states(&mut self, states: Vec<Tensor>) {
        self.residency = SessionResidency::Active(states);
    }

    fn is_quantized(&self) -> bool {
        matches!(self.residency, SessionResidency::Quantized(_))
    }

    fn state_count(&self) -> usize {
        match &self.residency {
            SessionResidency::Active(states) => states.len(),
            SessionResidency::Quantized(states) => states.len(),
        }
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
        if let SessionResidency::Quantized(buffers) = &self.residency {
            let mut states = Vec::with_capacity(buffers.len());
            for buffer in buffers {
                let mut tensors = candle_core::safetensors::load_buffer(buffer, device)?;
                let tensor = tensors.remove("tensor").ok_or_else(|| {
                    candle_core::Error::Msg("compressed session state missing 'tensor' key".into())
                })?;
                states.push(tensor.to_dtype(DType::F32)?);
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
        states[layer_index] = states[layer_index].add(&delta.to_dtype(DType::F32)?)?;
        self.replace_states(states);
        Ok(())
    }

    fn park_quantized(&mut self) -> candle_core::Result<()> {
        let states = match &self.residency {
            SessionResidency::Active(states) => states,
            SessionResidency::Quantized(_) => return Ok(()),
        };
        let mut buffers = Vec::with_capacity(states.len());
        for state in states {
            let fp16 = state.to_device(&Device::Cpu)?.to_dtype(DType::F16)?;
            let bytes = safetensors::serialize([("tensor", &fp16)], &None).map_err(|err| {
                candle_core::Error::Msg(format!("session serialization failed: {err}"))
            })?;
            buffers.push(bytes);
        }
        self.residency = SessionResidency::Quantized(buffers);
        Ok(())
    }
}

/// Global server state shared across all request handlers.
///
/// * `pipeline` — inference pipeline; wrapped in `Mutex` because
///   `InferencePipeline` contains `RefCell` (via `TTTLinearLayer`) which is
///   `!Send`.  Generation itself is synchronous and does not hold the lock
///   across `.await` points.
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
    /// Canonical model identifier returned in API responses.
    pub model_id: String,
}

impl AppState {
    pub fn new(pipeline: InferencePipeline, model_id: String) -> Self {
        let device = pipeline.device().clone();
        Self {
            pipeline: Arc::new(Mutex::new(pipeline)),
            device,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            model_id,
        }
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
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum ApiError {
    Internal(String),
    NotFound(String),
    BadRequest(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg).into_response(),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg).into_response(),
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
        chat_completion_sse(state, req).into_response()
    } else {
        chat_completion_json(state, req).into_response()
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
                ApiError::Internal(m) | ApiError::NotFound(m) | ApiError::BadRequest(m) => m,
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
        sessions.insert(
            session_id.clone(),
            SessionData::new_active(states, now, model.clone()),
        );
    }
    metrics::register_session(&session_id);
    state.refresh_session_metrics()?;

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
            .adapt_on_corpus(&req.corpus, initial_states)
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
        .or_insert_with(|| SessionData::new_active(states, now, state.model_id.clone()));
    drop(sessions);
    metrics::register_session(&session_id);
    metrics::mark_session_quantized(&session_id, false);
    state.refresh_session_metrics()?;

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
    drop(sessions);
    state.refresh_session_metrics()?;

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
fn run_generation(
    state: &AppState,
    prompt: &str,
    max_tokens: usize,
    session_id: Option<&str>,
) -> Result<String, ApiError> {
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
        sessions.insert(
            new_id.clone(),
            SessionData::new_active(states.clone(), now, state.model_id.clone()),
        );
        drop(sessions);
        metrics::register_session(&new_id);
        state.refresh_session_metrics()?;
        Ok((new_id, states))
    }
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
        .route("/v1/cluster/sync", post(cluster_sync))
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions/:id", delete(delete_session))
        .route("/v1/adapt", post(adapt))
        .route("/v1/sessions/:id/checkpoint", get(get_checkpoint))
        .route("/v1/sessions/:id/checkpoint", put(put_checkpoint))
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
    let state = AppState::new(pipeline, model_id);
    let app = create_router(state);

    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    println!("[+] Axiom-TTT server listening on http://{host}:{port}");
    println!("[+] OpenAI-compatible API endpoints:");
    println!("      GET  /metrics");
    println!("      GET  /v1/models");
    println!("      POST /v1/completions");
    println!("      POST /v1/chat/completions         (stream:true for SSE)");
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
            num_heads: 2,
            head_dim: 8,
            vocab_size: 64,
            lr_inner: 1e-3,
            rms_norm_eps: 1e-6,
            use_log_scan: false,
            log_scan_auto_threshold: 100_000,
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
                SessionData::new_active(initial_states.clone(), now, state.model_id.clone()),
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

        let sessions = state.sessions.read().unwrap();
        let session = sessions.get("adapt-session").unwrap();
        assert_eq!(session.state_count(), initial_states.len());
        assert!(session.last_used >= now);
        drop(sessions);
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
            sessions.insert(
                session_id.clone(),
                SessionData::new_active(states, now, state.model_id.clone()),
            );
        }
        metrics::register_session(&session_id);
        state.refresh_session_metrics().unwrap();

        let delta = Tensor::ones((1usize, 2usize, 8usize, 8usize), DType::F32, &state.device)
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
                    delta_bytes,
                })
                .unwrap(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);

        let sessions = state.sessions.read().unwrap();
        let session = sessions.get(&session_id).unwrap();
        let mut layers = session.states_clone().unwrap();
        let layer = layers.remove(0);
        let layer_sum = layer.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(layer_sum > 0.0);
        drop(sessions);
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
            let mut session = SessionData::new_active(states, now, state.model_id.clone());
            session.park_quantized().unwrap();
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

        let sessions = state.sessions.read().unwrap();
        let session = sessions.get("quantized-session").unwrap();
        assert!(!session.is_quantized());
        drop(sessions);
        safe_drop(pipeline_arc).await;
    }
}
