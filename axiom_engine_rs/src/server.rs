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
use std::fs;
use std::path::{Path as FsPath, PathBuf};
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
use sha2::{Digest, Sha256};
use tokio::task::spawn_blocking;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::anthropic_forwarder::{
    build_compressed_payload, partition_messages, AnthropicForwarder, ClientAuth, ForwarderError,
};
use crate::claude_backend::{ChatTurn, ClaudeBackend};
use crate::cluster::StateDeltaUpdate;
use crate::config::AxiomConfig;
use crate::context_compressor::{
    adapt_session_blocking, extract_memory_vector_blocking, feedback_adaptation_text,
    should_retry_uncompressed, CompressionControls, CompressorConfig, MemoryFingerprint,
    SessionStates, TttSessionStore,
};
use crate::dwe::{extract_delta_fragment, DweBus, DweTelemetry};
use crate::hamiltonian::QuantumRuntimeStatus;
use crate::inference::InferencePipeline;
use crate::metrics;
use crate::openai_forwarder::{OpenAiClientAuth, OpenAiForwarder, OpenAiForwarderError};
use crate::poly_jit::{PolyJitEngine, PolyJitStatus};
use crate::quantization::{NF4QuantizedDescriptor, NF4Quantizer};
use crate::sandbox::{SandboxController, SandboxDiagnostic};
use crate::surprisal::{ExactAttentionResidualCache, ExactResidualTelemetry};
use crate::swarm_route::{LocalSwarmRouteMatrix, SwarmMatrixState};
use crate::swarm_router::{SwarmChatResult, SwarmRouter};
use crate::vfs::{NeuralVfs, VfsMountReport, VfsReadReport, VfsStats};
use crate::vibe_memory::MasterVibe;
use crate::weight_merge::{merge_checkpoint_files, MergeSummary};

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

fn context_hash(source: &str) -> String {
    let digest = Sha256::digest(source.as_bytes());
    format!("{digest:x}")
}

fn compression_cache_path() -> PathBuf {
    std::env::var("AXIOM_COMPRESSION_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("checkpoints/memory/axiom_compression_cache.bin"))
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
    /// Outbound bridge to the real Anthropic API used by the Claude compression
    /// path. `None` when Anthropic forwarding is disabled.
    pub anthropic_forwarder: Arc<Option<AnthropicForwarder>>,
    /// Outbound bridge to OpenAI-compatible chat completions used by the Codex
    /// compression path. `None` when OpenAI forwarding is disabled.
    pub openai_forwarder: Arc<Option<OpenAiForwarder>>,
    /// Optional local SLM router. When present, compressed outbound payloads are
    /// attempted against Ollama before falling back to cloud forwarders.
    pub swarm_router: Arc<Option<SwarmRouter>>,
    /// Static compression configuration (threshold, top-k, enabled flag).
    pub compressor_config: Arc<CompressorConfig>,
    /// Optional persistent "vibe memory". When `Some`, adapted session W̃
    /// states are EMA-merged into the master and serialised on session
    /// drop / clear / graceful shutdown (the automatic merge trigger).
    pub master_vibe: Arc<Mutex<Option<MasterVibe>>>,
    /// Original heavy-context source per session, kept so the skeleton
    /// round-trip can expand a dropped symbol body on demand (`POST /v1/expand`).
    /// Soft-bounded (cleared past a cap) since sessions are transient.
    pub source_store: Arc<RwLock<HashMap<String, String>>>,
    /// Last heavy-context digest successfully adapted per compression session.
    /// Repeated full-context prompts from Claude/Codex can then reuse the
    /// already-mutated fast-weights instead of redoing the same TTT prefill.
    adapted_context_hashes: Arc<RwLock<HashMap<String, String>>>,
    /// Runtime-mutable compression controls + live counters. Lets a dashboard
    /// retune the threshold / on-off without a restart (`/v1/config`).
    pub controls: Arc<CompressionControls>,
    /// Safe user-mode VFS loopback that feeds file reads into TTT prefill.
    pub neural_vfs: Arc<NeuralVfs>,
    /// User-mode polymorphic runtime status and execution wrapper.
    pub poly_jit: Arc<PolyJitEngine>,
    /// SR-TTT exact residual path for high-surprisal identifiers.
    pub exact_residual_cache: Arc<ExactAttentionResidualCache>,
    /// DWE binary tensor-delta broadcast bus.
    pub dwe_bus: Arc<DweBus>,
    /// Localized high-plasticity model allocation tracker.
    pub swarm_matrix: Arc<LocalSwarmRouteMatrix>,
    /// Path to the persistent heal memory served/merged by the swarm-immunity
    /// endpoints (`/v1/immunity`). `None` disables those routes.
    pub heal_memory_path: Arc<Option<std::path::PathBuf>>,
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
            openai_forwarder: Arc::new(None),
            swarm_router: Arc::new(None),
            compressor_config: Arc::new(CompressorConfig::default()),
            master_vibe: Arc::new(Mutex::new(None)),
            source_store: Arc::new(RwLock::new(HashMap::new())),
            adapted_context_hashes: Arc::new(RwLock::new(HashMap::new())),
            controls: Arc::new(CompressionControls::from_config(
                &CompressorConfig::default(),
            )),
            neural_vfs: Arc::new(NeuralVfs::new()),
            poly_jit: Arc::new(PolyJitEngine::default()),
            exact_residual_cache: Arc::new(ExactAttentionResidualCache::default()),
            dwe_bus: Arc::new(DweBus::from_env()),
            swarm_matrix: Arc::new(LocalSwarmRouteMatrix::new()),
            heal_memory_path: Arc::new(None),
        }
    }

    /// Enable the swarm-immunity endpoints against this heal-memory file.
    pub fn with_heal_memory_path(mut self, path: Option<std::path::PathBuf>) -> Self {
        self.heal_memory_path = Arc::new(path);
        self
    }

    /// Store the original heavy source for a session so `/v1/expand` can later
    /// retrieve a dropped symbol body. Soft-bounded: clears past 256 entries
    /// (transient sessions; approximate eviction is acceptable here).
    pub fn store_source(&self, session_id: &str, source: String) {
        if let Ok(mut map) = self.source_store.write() {
            if map.len() >= 256 {
                map.clear();
            }
            map.insert(session_id.to_string(), source);
        }
    }

    fn should_adapt_heavy_context(&self, session_id: &str, source: &str) -> bool {
        let hash = context_hash(source);
        self.adapted_context_hashes
            .read()
            .map(|map| map.get(session_id) != Some(&hash))
            .unwrap_or(true)
    }

    fn mark_heavy_context_adapted(&self, session_id: &str, source: &str) {
        if let Ok(mut map) = self.adapted_context_hashes.write() {
            if map.len() >= 256 {
                map.clear();
            }
            map.insert(session_id.to_string(), context_hash(source));
        }
    }

    async fn adapt_feedback_to_cache(
        &self,
        req: TttFeedbackRequest,
        cache_path: &FsPath,
    ) -> Result<TttFeedbackResponse, ApiError> {
        let session_id = req.session_id.trim().to_string();
        let message = req.message.trim().to_string();
        if session_id.is_empty() || message.is_empty() {
            return Err(ApiError::BadRequest(
                "session_id and message are required".into(),
            ));
        }

        let kind = req.feedback_type.as_deref().unwrap_or("execution_feedback");
        let feedback_text = feedback_adaptation_text(kind, &message, req.trace.as_deref());
        let pipeline_arc = self.pipeline.clone();
        let store = self.ttt_sessions.clone();
        let session_id_for_task = session_id.clone();
        let feedback_for_task = feedback_text.clone();
        let dwe_sequence = unix_now();

        let adapt_result = spawn_blocking(move || {
            let pipeline = pipeline_arc
                .lock()
                .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
            let handle = store
                .get_or_create(&session_id_for_task, &pipeline)
                .map_err(|e| ApiError::Internal(format!("session allocation failed: {e}")))?;
            let mut states = handle.blocking_lock();
            let baseline = states.clone();
            let tokens = pipeline.encode_text(&feedback_for_task);
            adapt_session_blocking(&pipeline, &mut states, &tokens)
                .map_err(|e| ApiError::Internal(format!("feedback TTT adapt failed: {e}")))?;
            let fragment =
                extract_delta_fragment(&session_id_for_task, dwe_sequence, &states, &baseline).ok();
            Ok::<_, ApiError>((tokens.len(), fragment))
        })
        .await
        .map_err(|e| ApiError::Internal(format!("blocking task join failed: {e}")))??;
        let (token_count, fragment) = adapt_result;
        if let Some(fragment) = fragment {
            self.dwe_bus.broadcast(fragment);
        }

        self.mark_heavy_context_adapted(&session_id, &feedback_text);
        self.persist_compression_cache_to(cache_path)
            .await
            .map_err(ApiError::Internal)?;

        Ok(TttFeedbackResponse {
            session_id,
            feedback_tokens: token_count,
            persisted: true,
            cache_path: cache_path.display().to_string(),
        })
    }

    async fn sandbox_local_synthesis(&self, session_id: &str, content: &str) {
        if SandboxController::rust_code_blocks(content).is_empty() {
            return;
        }
        let Some(sandbox) = SandboxController::from_env() else {
            return;
        };
        let state = self.clone();
        let cache_path = compression_cache_path();
        let result = sandbox
            .verify_rust_code_blocks_with_feedback(session_id, content, move |diag| {
                let state = state.clone();
                let cache_path = cache_path.clone();
                async move { state.adapt_sandbox_diagnostic(diag, &cache_path).await }
            })
            .await;
        match result {
            Ok(report) if report.passed => eprintln!(
                "[sandbox] session={} rust_blocks={} passed attempts={}",
                report.session_id, report.blocks_checked, report.attempts
            ),
            Ok(report) => eprintln!(
                "[sandbox] session={} rust_blocks={} failed diagnostics={} attempts={}",
                report.session_id,
                report.blocks_checked,
                report.diagnostics.len(),
                report.attempts
            ),
            Err(e) => eprintln!("[sandbox] verification skipped: {e}"),
        }
    }

    async fn adapt_sandbox_diagnostic(
        &self,
        diag: SandboxDiagnostic,
        cache_path: &FsPath,
    ) -> Result<(), String> {
        let message = format!(
            "sandbox compiler check failed at step {} with status {:?}",
            diag.step, diag.status_code
        );
        let trace = diag.feedback_trace();
        self.adapt_feedback_to_cache(
            TttFeedbackRequest {
                session_id: diag.session_id,
                message,
                feedback_type: Some("sandbox_compilation_error".to_string()),
                trace: Some(trace),
            },
            cache_path,
        )
        .await
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
    }

    async fn persist_compression_cache(&self) -> Result<(), String> {
        self.persist_compression_cache_to(&compression_cache_path())
            .await
    }

    async fn hydrate_compression_cache(&self) -> Result<usize, String> {
        self.hydrate_compression_cache_from(&compression_cache_path())
            .await
    }

    async fn persist_compression_cache_to(&self, path: &FsPath) -> Result<(), String> {
        let hashes = self
            .adapted_context_hashes
            .read()
            .map_err(|_| "adapted context hash lock poisoned".to_string())?
            .clone();
        if hashes.is_empty() {
            return Ok(());
        }

        let mut entries = Vec::new();
        for (session_id, handle) in self.ttt_sessions.snapshot_handles() {
            let Some(context_hash) = hashes.get(&session_id).cloned() else {
                continue;
            };
            let states = handle.lock().await;
            let layers = states
                .iter()
                .map(tensor_to_layer_checkpoint)
                .collect::<candle_core::Result<Vec<_>>>()
                .map_err(|e| format!("checkpoint encode failed: {e}"))?;
            entries.push(PersistedCompressionEntry {
                session_id: session_id.clone(),
                context_hash,
                checkpoint: SessionCheckpoint {
                    session_id,
                    version: 1,
                    created_at: unix_now(),
                    layers,
                },
            });
        }

        let payload = PersistedCompressionCache {
            version: 1,
            entries,
        };
        let bytes = bincode::serialize(&payload)
            .map_err(|e| format!("compression cache serialize failed: {e}"))?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("compression cache mkdir failed: {e}"))?;
        }
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, bytes).map_err(|e| format!("compression cache write failed: {e}"))?;
        fs::rename(&tmp, path).map_err(|e| format!("compression cache rename failed: {e}"))?;
        Ok(())
    }

    async fn hydrate_compression_cache_from(&self, path: &FsPath) -> Result<usize, String> {
        if !path.exists() {
            return Ok(0);
        }
        let bytes = fs::read(path).map_err(|e| format!("compression cache read failed: {e}"))?;
        let payload: PersistedCompressionCache = bincode::deserialize(&bytes)
            .map_err(|e| format!("compression cache decode failed: {e}"))?;
        if payload.version != 1 {
            return Err(format!(
                "unsupported compression cache version {}",
                payload.version
            ));
        }

        let mut restored = 0usize;
        for entry in payload.entries {
            if entry.checkpoint.version != 1 {
                continue;
            }
            let states = entry
                .checkpoint
                .layers
                .iter()
                .map(|lc| layer_checkpoint_to_tensor(lc, &self.device))
                .collect::<candle_core::Result<Vec<_>>>()
                .map_err(|e| format!("compression cache tensor restore failed: {e}"))?;
            self.ttt_sessions
                .insert_states(entry.session_id.clone(), states);
            if let Ok(mut hashes) = self.adapted_context_hashes.write() {
                hashes.insert(entry.session_id, entry.context_hash);
            }
            restored += 1;
        }
        Ok(restored)
    }

    /// Install (or clear) the persistent vibe-memory master.
    pub fn with_master_vibe(mut self, vibe: Option<MasterVibe>) -> Self {
        self.master_vibe = Arc::new(Mutex::new(vibe));
        self
    }

    /// EMA-merge one session handle's adapted W̃ into the master vibe and
    /// persist it. No-op when vibe memory is disabled. Reads the session
    /// states asynchronously, then performs the (sync, tiny) commit without
    /// holding the std mutex across an `.await`.
    async fn flush_session_to_vibe(&self, handle: &SessionStates) {
        // Cheap clone of the per-layer tensors out from under the async lock.
        let states = { handle.lock().await.clone() };
        if let Ok(mut guard) = self.master_vibe.lock() {
            if let Some(vibe) = guard.as_mut() {
                if let Err(e) = vibe.commit_and_save(&states) {
                    eprintln!("[vibe] auto-commit skipped: {e}");
                }
            }
        }
    }

    /// Flush every live session into the master vibe (used on graceful
    /// shutdown). Each session is committed in turn; the EMA naturally
    /// accumulates them.
    pub async fn flush_all_sessions_to_vibe(&self) {
        let enabled = self
            .master_vibe
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false);
        if !enabled {
            return;
        }
        let handles = self.ttt_sessions.snapshot_handles();
        if handles.is_empty() {
            return;
        }
        eprintln!(
            "[vibe] flushing {} live session(s) into master vibe",
            handles.len()
        );
        for (_, handle) in handles {
            self.flush_session_to_vibe(&handle).await;
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

    /// Install an OpenAI-compatible forwarder for the Codex compression path.
    pub fn with_openai_forwarder(mut self, forwarder: Option<OpenAiForwarder>) -> Self {
        self.openai_forwarder = Arc::new(forwarder);
        self
    }

    /// Install an optional local SLM/Ollama router.
    pub fn with_swarm_router(mut self, router: Option<SwarmRouter>) -> Self {
        self.swarm_router = Arc::new(router);
        self
    }

    /// Override the compressor configuration. Also seeds the runtime controls so
    /// the live threshold / on-off state starts from the configured values.
    pub fn with_compressor_config(mut self, config: CompressorConfig) -> Self {
        self.controls = Arc::new(CompressionControls::from_config(&config));
        self.compressor_config = Arc::new(config);
        self
    }

    /// True iff every component the compression path needs is configured:
    /// the (runtime) feature flag and the forwarder. The enabled flag is read
    /// from the live controls so a dashboard can toggle it without a restart.
    pub fn compression_active(&self) -> bool {
        self.controls.enabled() && self.anthropic_forwarder.is_some()
    }

    /// True iff OpenAI-compatible chat compression has both the runtime feature
    /// flag and an upstream forwarder.
    pub fn openai_compression_active(&self) -> bool {
        self.controls.enabled() && self.openai_forwarder.is_some()
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
    Upstream {
        status: u16,
        message: String,
    },
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
                let code = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
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

#[derive(Debug, Serialize, Deserialize)]
struct PersistedCompressionCache {
    version: u32,
    entries: Vec<PersistedCompressionEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedCompressionEntry {
    session_id: String,
    context_hash: String,
    checkpoint: SessionCheckpoint,
}

#[derive(Debug, Deserialize)]
struct TttFeedbackRequest {
    session_id: String,
    message: String,
    #[serde(default)]
    feedback_type: Option<String>,
    #[serde(default)]
    trace: Option<String>,
}

#[derive(Debug, Serialize)]
struct TttFeedbackResponse {
    session_id: String,
    feedback_tokens: usize,
    persisted: bool,
    cache_path: String,
}

#[derive(Debug, Deserialize)]
struct ClusterMergeRequest {
    inputs: Vec<String>,
    output: String,
    #[serde(default)]
    alpha: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct HypervisorMountRequest {
    root: String,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    warm_paths: Vec<String>,
}

#[derive(Debug, Serialize)]
struct HypervisorMountResponse {
    mount: VfsMountReport,
    warmed: Vec<VfsReadReport>,
    vfs: VfsStats,
}

#[derive(Debug, Serialize)]
struct HypervisorJitStatusResponse {
    jit: PolyJitStatus,
    vfs: VfsStats,
}

#[derive(Debug, Serialize)]
struct HypervisorQuantumStateResponse {
    quantum: QuantumRuntimeStatus,
}

#[derive(Debug, Serialize)]
struct SwarmMatrixStateResponse {
    matrix: SwarmMatrixState,
    dwe: DweTelemetry,
    exact_residual: ExactResidualTelemetry,
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
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let session_override = headers
        .get("x-axiom-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    if state.openai_compression_active() {
        let client_auth = OpenAiClientAuth {
            authorization: headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string),
        };
        match compressed_openai_chat_path(&state, &body, session_override.as_deref(), &client_auth)
            .await
        {
            Ok(resp) => return resp,
            Err(err) => return err.into_response(),
        }
    }

    let req: ChatCompletionRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => {
            return ApiError::BadRequest(format!("invalid /v1/chat/completions body: {e}"))
                .into_response()
        }
    };

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
                match openai_stream_delta(&completion_id, created, &model, piece) {
                    Ok(chunk) => events.push(Ok(Event::default().data(chunk))),
                    Err(e) => {
                        events.push(Ok(Event::default().data(format!("error: {e:?}"))));
                        return Sse::new(stream::iter(events)).keep_alive(KeepAlive::default());
                    }
                }
            }

            // Final chunk: stop signal with empty delta.
            match openai_stream_stop(&completion_id, created, &model) {
                Ok(stop_chunk) => events.push(Ok(Event::default().data(stop_chunk))),
                Err(e) => {
                    events.push(Ok(Event::default().data(format!("error: {e:?}"))));
                    return Sse::new(stream::iter(events)).keep_alive(KeepAlive::default());
                }
            }
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

/// `POST /v1/cluster/merge` — merge persisted W_tilde cache files into a fresh
/// bincode checkpoint via task-vector interpolation.
async fn cluster_merge(
    Json(req): Json<ClusterMergeRequest>,
) -> Result<Json<MergeSummary>, ApiError> {
    let alpha = req.alpha.unwrap_or(0.5);
    let inputs = req
        .inputs
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let output = PathBuf::from(req.output);
    let summary = spawn_blocking(move || merge_checkpoint_files(&inputs, &output, alpha))
        .await
        .map_err(|e| ApiError::Internal(format!("merge task join failed: {e}")))?
        .map_err(ApiError::BadRequest)?;
    Ok(Json(summary))
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

fn partition_messages_for_state(
    state: &AppState,
    messages: &[Value],
    threshold: usize,
) -> Result<crate::anthropic_forwarder::PartitionedMessages, ApiError> {
    let pipeline = state
        .pipeline
        .lock()
        .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
    Ok(partition_messages(messages, threshold, |text| {
        pipeline.token_count(text)
    }))
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
        Err(e) => {
            return ApiError::BadRequest(format!("invalid /v1/messages body: {e}")).into_response()
        }
    };
    local_messages_path(state, req).map_or_else(|e| e.into_response(), |json| json.into_response())
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
    let forwarder = state.anthropic_forwarder.as_ref().as_ref().cloned();
    if forwarder.is_none() && state.swarm_router.as_ref().is_none() {
        return Err(ApiError::Internal(
            "compression active but no Anthropic forwarder or local swarm router".into(),
        ));
    }

    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| ApiError::BadRequest("messages[] required".into()))?;

    let cfg = state.compressor_config.clone();
    // Threshold is read live from the runtime controls so a dashboard can retune
    // it without a restart; top_k stays a startup constant.
    let threshold = state.controls.threshold();
    let top_k = cfg.recall_top_k;

    let partitioned = partition_messages_for_state(state, &messages, threshold)?;

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
        let should_adapt = state.should_adapt_heavy_context(&session_id, &heavy_combined);
        let exact_cache = state.exact_residual_cache.clone();
        let dwe_sequence = unix_now();

        // Spawn the compute-heavy loop on a blocking thread so the Tokio
        // runtime keeps serving other requests during the gradient steps.
        let fp_result: Result<_, ApiError> = tokio::task::spawn_blocking(move || {
            let pipeline = pipeline_arc
                .lock()
                .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
            let session = store
                .get_or_create(&session_id_clone, &pipeline)
                .map_err(|e| ApiError::Internal(format!("session allocation failed: {e}")))?;
            // tokio::sync::Mutex::blocking_lock is safe in spawn_blocking.
            let mut session_states = session.blocking_lock();

            let mut dwe_fragment = None;
            let context_tokens_processed = if should_adapt {
                let baseline = session_states.clone();
                let context_tokens: Vec<u32> = pipeline.encode_text(&heavy_clone);
                let (fast_tokens, _sr_report) =
                    exact_cache.route_tokens(&session_id_clone, &context_tokens, &heavy_clone);
                adapt_session_blocking(&pipeline, &mut session_states, &fast_tokens)
                    .map_err(|e| ApiError::Internal(format!("TTT adapt failed: {e}")))?;
                dwe_fragment = extract_delta_fragment(
                    &session_id_clone,
                    dwe_sequence,
                    &session_states,
                    &baseline,
                )
                .ok();
                context_tokens.len()
            } else {
                pipeline.token_count(&heavy_clone)
            };

            let residual_prompt = exact_cache.residual_prompt(&session_id_clone, 96);
            let recall_query = if residual_prompt.is_empty() {
                query_clone
            } else {
                format!("{query_clone}\n\n{residual_prompt}")
            };
            let query_tokens: Vec<u32> = pipeline.encode_text(&recall_query);
            let fingerprint = extract_memory_vector_blocking(
                &pipeline,
                &mut session_states,
                &query_tokens,
                &session_id_clone,
                context_tokens_processed,
                started,
                top_k,
            )
            .map_err(|e| ApiError::Internal(format!("memory extraction failed: {e}")))?;
            Ok((fingerprint, dwe_fragment))
        })
        .await
        .map_err(|e| ApiError::Internal(format!("blocking task join failed: {e}")))?;
        let (fingerprint, dwe_fragment) = fp_result?;
        if should_adapt {
            if let Some(fragment) = dwe_fragment {
                state.dwe_bus.broadcast(fragment);
            }
            state.mark_heavy_context_adapted(&session_id, &heavy_combined);
            if let Err(e) = state.persist_compression_cache().await {
                eprintln!("[axiom-ttt] compression cache persist skipped: {e}");
            }
        }
        fingerprint
    };

    eprintln!(
        "[axiom-ttt] compressed session={} heavy_msgs={} heavy_tokens~{} recall_norm={:.3} elapsed_ms={}",
        fingerprint.session_id,
        log_heavy_count,
        log_heavy_tokens,
        fingerprint.recall_norm,
        fingerprint.elapsed_ms,
    );

    // Keep the original heavy source so a dropped symbol body can be expanded
    // on demand via POST /v1/expand (the skeleton round-trip).
    if !heavy_combined.trim().is_empty() {
        state.store_source(&session_id, heavy_combined.clone());
    }

    let outbound = build_compressed_payload(body, &fingerprint, &partitioned);
    // Strip session_id (and any other Axiom extensions) from the upstream payload.
    let mut outbound = outbound;
    if let Some(obj) = outbound.as_object_mut() {
        obj.remove("session_id");
    }

    // Active immunity: if the conversation references a command Axiom has
    // already learned to heal, inject a short advisory so Claude gets the fix
    // without anyone asking — the self-healing loop running autonomously. Only
    // fires on a precise command-signature match with a concrete learned heal.
    inject_immunity_advisory(&state, &mut outbound, &user_query_text, &heavy_combined);

    // Record live compression stats for the dashboard: original vs forwarded
    // payload size and how many heavy messages were absorbed this request.
    let bytes_in = serde_json::to_string(body)
        .map(|s| s.len() as u64)
        .unwrap_or(0);
    let bytes_out = serde_json::to_string(&outbound)
        .map(|s| s.len() as u64)
        .unwrap_or(0);
    state
        .controls
        .record(log_heavy_count as u64, bytes_in, bytes_out);

    if let Some(router) = state.swarm_router.as_ref().as_ref() {
        match router.route_chat_payload(&outbound).await {
            Ok(local) => {
                state
                    .sandbox_local_synthesis(&session_id, &local.content)
                    .await;
                return Ok(local_anthropic_message_response(&outbound, local));
            }
            Err(e) => {
                eprintln!("[swarm-router] local Anthropic route unavailable; falling back: {e}")
            }
        }
    }

    let forwarder = forwarder.ok_or_else(|| ApiError::Upstream {
        status: StatusCode::BAD_GATEWAY.as_u16(),
        message: "local swarm route failed and no Anthropic cloud forwarder is configured".into(),
    })?;

    // First attempt: forward the lean, compressed payload.
    match forwarder
        .forward_messages_json(&outbound, client_auth)
        .await
    {
        Ok(value) => Ok(value),
        Err(err) => {
            // Graceful degradation: a compression-side fault (or a transient
            // upstream/network hiccup) must never cost the client their turn. If
            // we actually altered the payload and the failure is one the original
            // payload could fix, retry ONCE with the uncompressed body. The TTT
            // session already absorbed the heavy context above, so this only
            // forgoes the token *savings* for this one request — never the answer.
            let did_compress = log_heavy_count > 0;
            if did_compress && should_retry_uncompressed(&err) {
                eprintln!(
                    "[axiom-ttt] compressed forward failed ({err}); retrying once with original \
                     uncompressed payload (session={session_id})"
                );
                state.controls.record_degraded_fallback();
                // The untouched client body, minus Axiom-only extensions.
                let mut fallback = body.clone();
                if let Some(obj) = fallback.as_object_mut() {
                    obj.remove("session_id");
                }
                return forwarder
                    .forward_messages_json(&fallback, client_auth)
                    .await
                    .map_err(map_anthropic_forwarder_error);
            }
            Err(map_anthropic_forwarder_error(err))
        }
    }
}

/// Map an Anthropic [`ForwarderError`] onto the client-facing [`ApiError`].
fn map_anthropic_forwarder_error(err: ForwarderError) -> ApiError {
    match err {
        // Surface the real upstream status (401/429/5xx) to the client.
        ForwarderError::Upstream { status, body } => ApiError::Upstream {
            status,
            message: format!("anthropic upstream {status}: {body}"),
        },
        // No credential at all → 401 so the client knows to authenticate.
        ForwarderError::MissingAuth => ApiError::Upstream {
            status: StatusCode::UNAUTHORIZED.as_u16(),
            message: format!("{err}"),
        },
        // Network/decode failures mean we never got a usable response →
        // 502 Bad Gateway rather than a misleading 500.
        other => ApiError::Upstream {
            status: StatusCode::BAD_GATEWAY.as_u16(),
            message: format!("anthropic upstream call failed: {other}"),
        },
    }
}

/// Map an OpenAI [`OpenAiForwarderError`] onto the client-facing [`ApiError`].
fn map_openai_forwarder_error(err: OpenAiForwarderError) -> ApiError {
    match err {
        OpenAiForwarderError::MissingAuth => ApiError::Upstream {
            status: StatusCode::UNAUTHORIZED.as_u16(),
            message: format!("{err}"),
        },
        other => ApiError::Upstream {
            status: StatusCode::BAD_GATEWAY.as_u16(),
            message: format!("OpenAI upstream call failed: {other}"),
        },
    }
}

/// OpenAI-compatible active-compression path: partition -> adapt -> recall ->
/// forward to `/v1/chat/completions`.
async fn compressed_openai_chat_path(
    state: &AppState,
    body: &Value,
    session_override: Option<&str>,
    client_auth: &OpenAiClientAuth,
) -> Result<Response, ApiError> {
    let forwarder = state.openai_forwarder.as_ref().as_ref().cloned();
    if forwarder.is_none() && state.swarm_router.as_ref().is_none() {
        return Err(ApiError::Internal(
            "OpenAI compression active but no OpenAI forwarder or local swarm router".into(),
        ));
    }

    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| ApiError::BadRequest("messages[] required".into()))?;

    let cfg = state.compressor_config.clone();
    let threshold = state.controls.threshold();
    let top_k = cfg.recall_top_k;
    let partitioned = partition_messages_for_state(state, &messages, threshold)?;

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
        empty_fingerprint(state, &session_id, started)?
    } else {
        let pipeline_arc = state.pipeline.clone();
        let store = state.ttt_sessions.clone();
        let session_id_clone = session_id.clone();
        let heavy_clone = heavy_combined.clone();
        let query_clone = user_query_text.clone();
        let should_adapt = state.should_adapt_heavy_context(&session_id, &heavy_combined);
        let exact_cache = state.exact_residual_cache.clone();
        let dwe_sequence = unix_now();

        let fp_result: Result<_, ApiError> = spawn_blocking(move || {
            let pipeline = pipeline_arc
                .lock()
                .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
            let session = store
                .get_or_create(&session_id_clone, &pipeline)
                .map_err(|e| ApiError::Internal(format!("session allocation failed: {e}")))?;
            let mut session_states = session.blocking_lock();

            let mut dwe_fragment = None;
            let context_tokens_processed = if should_adapt {
                let baseline = session_states.clone();
                let context_tokens: Vec<u32> = pipeline.encode_text(&heavy_clone);
                let (fast_tokens, _sr_report) =
                    exact_cache.route_tokens(&session_id_clone, &context_tokens, &heavy_clone);
                adapt_session_blocking(&pipeline, &mut session_states, &fast_tokens)
                    .map_err(|e| ApiError::Internal(format!("TTT adapt failed: {e}")))?;
                dwe_fragment = extract_delta_fragment(
                    &session_id_clone,
                    dwe_sequence,
                    &session_states,
                    &baseline,
                )
                .ok();
                context_tokens.len()
            } else {
                pipeline.token_count(&heavy_clone)
            };

            let residual_prompt = exact_cache.residual_prompt(&session_id_clone, 96);
            let recall_query = if residual_prompt.is_empty() {
                query_clone
            } else {
                format!("{query_clone}\n\n{residual_prompt}")
            };
            let query_tokens: Vec<u32> = pipeline.encode_text(&recall_query);
            let fingerprint = extract_memory_vector_blocking(
                &pipeline,
                &mut session_states,
                &query_tokens,
                &session_id_clone,
                context_tokens_processed,
                started,
                top_k,
            )
            .map_err(|e| ApiError::Internal(format!("memory extraction failed: {e}")))?;
            Ok((fingerprint, dwe_fragment))
        })
        .await
        .map_err(|e| ApiError::Internal(format!("blocking task join failed: {e}")))?;
        let (fingerprint, dwe_fragment) = fp_result?;
        if should_adapt {
            if let Some(fragment) = dwe_fragment {
                state.dwe_bus.broadcast(fragment);
            }
            state.mark_heavy_context_adapted(&session_id, &heavy_combined);
            if let Err(e) = state.persist_compression_cache().await {
                eprintln!("[axiom-ttt] compression cache persist skipped: {e}");
            }
        }
        fingerprint
    };

    eprintln!(
        "[axiom-ttt] openai compressed session={} heavy_msgs={} heavy_tokens~{} recall_norm={:.3} elapsed_ms={}",
        fingerprint.session_id,
        log_heavy_count,
        log_heavy_tokens,
        fingerprint.recall_norm,
        fingerprint.elapsed_ms,
    );

    if !heavy_combined.trim().is_empty() {
        state.store_source(&session_id, heavy_combined.clone());
    }

    let mut outbound = build_compressed_payload(body, &fingerprint, &partitioned);
    if let Some(obj) = outbound.as_object_mut() {
        obj.remove("session_id");
    }

    let bytes_in = serde_json::to_string(body)
        .map(|s| s.len() as u64)
        .unwrap_or(0);
    let bytes_out = serde_json::to_string(&outbound)
        .map(|s| s.len() as u64)
        .unwrap_or(0);
    state
        .controls
        .record(log_heavy_count as u64, bytes_in, bytes_out);

    if let Some(router) = state.swarm_router.as_ref().as_ref() {
        match router.route_chat_payload(&outbound).await {
            Ok(local) => {
                state
                    .sandbox_local_synthesis(&session_id, &local.content)
                    .await;
                return local_openai_chat_response(&outbound, local);
            }
            Err(e) => {
                eprintln!("[swarm-router] local OpenAI route unavailable; falling back: {e}")
            }
        }
    }

    let forwarder = forwarder.ok_or_else(|| ApiError::Upstream {
        status: StatusCode::BAD_GATEWAY.as_u16(),
        message: "local swarm route failed and no OpenAI cloud forwarder is configured".into(),
    })?;

    // Graceful degradation (mirrors the Anthropic path): a compression-side fault
    // must never cost the client their turn. We retry ONCE with the original
    // uncompressed body — for a transient network fault, or for a recoverable
    // non-2xx status (5xx / 400) that our injected fingerprint may have caused.
    // Unlike the Anthropic forwarder, this one returns Ok even for non-2xx (the
    // status rides in the response), so we inspect both the Err and the status.
    let did_compress = log_heavy_count > 0;
    let fallback_body = || {
        let mut fallback = body.clone();
        if let Some(obj) = fallback.as_object_mut() {
            obj.remove("session_id");
        }
        fallback
    };

    let mut fell_back = false;
    let mut upstream = match forwarder
        .forward_chat_completions_text(&outbound, client_auth)
        .await
    {
        Ok(resp) => resp,
        Err(OpenAiForwarderError::Network(msg)) if did_compress => {
            eprintln!(
                "[axiom-ttt] compressed OpenAI forward failed (network error: {msg}); retrying \
                 once with original uncompressed payload (session={session_id})"
            );
            state.controls.record_degraded_fallback();
            fell_back = true;
            forwarder
                .forward_chat_completions_text(&fallback_body(), client_auth)
                .await
                .map_err(map_openai_forwarder_error)?
        }
        Err(other) => return Err(map_openai_forwarder_error(other)),
    };

    if !fell_back && did_compress && (upstream.status >= 500 || upstream.status == 400) {
        eprintln!(
            "[axiom-ttt] compressed OpenAI forward returned {}; retrying once with original \
             uncompressed payload (session={session_id})",
            upstream.status
        );
        state.controls.record_degraded_fallback();
        if let Ok(retry) = forwarder
            .forward_chat_completions_text(&fallback_body(), client_auth)
            .await
        {
            upstream = retry;
        }
    }

    let status = StatusCode::from_u16(upstream.status).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);
    if let Some(content_type) = upstream.content_type {
        builder = builder.header(header::CONTENT_TYPE, content_type);
    } else {
        builder = builder.header(header::CONTENT_TYPE, "application/json");
    }
    builder
        .body(axum::body::Body::from(upstream.body))
        .map_err(|e| ApiError::Internal(format!("response build failed: {e}")))
}

fn local_openai_chat_response(
    outbound: &Value,
    local: SwarmChatResult,
) -> Result<Response, ApiError> {
    let completion_id = format!("chatcmpl-local-{}", Uuid::new_v4());
    let created = unix_now();
    let model = local.model;
    let content = local.content;
    let stream = outbound
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if stream {
        let mut body = String::new();
        for piece in content.split_inclusive(' ') {
            body.push_str("data: ");
            body.push_str(&openai_stream_delta(
                &completion_id,
                created,
                &model,
                piece,
            )?);
            body.push_str("\n\n");
        }
        body.push_str("data: ");
        body.push_str(&openai_stream_stop(&completion_id, created, &model)?);
        body.push_str("\n\n");
        body.push_str("data: [DONE]\n\n");
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(axum::body::Body::from(body))
            .map_err(|e| ApiError::Internal(format!("local stream response build failed: {e}")));
    }

    let body = openai_completion_body(&completion_id, created, &model, &content)?;
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(body))
        .map_err(|e| ApiError::Internal(format!("local response build failed: {e}")))
}

fn openai_stream_delta(
    completion_id: &str,
    created: u64,
    model: &str,
    piece: &str,
) -> Result<String, ApiError> {
    let id = json_string(completion_id)?;
    let model = json_string(model)?;
    let piece = json_string(piece)?;
    Ok(format!(
        r#"{{"id":{id},"object":"chat.completion.chunk","created":{created},"model":{model},"choices":[{{"index":0,"delta":{{"role":"assistant","content":{piece}}},"finish_reason":null}}]}}"#
    ))
}

fn openai_stream_stop(completion_id: &str, created: u64, model: &str) -> Result<String, ApiError> {
    let id = json_string(completion_id)?;
    let model = json_string(model)?;
    Ok(format!(
        r#"{{"id":{id},"object":"chat.completion.chunk","created":{created},"model":{model},"choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}"#
    ))
}

fn openai_completion_body(
    completion_id: &str,
    created: u64,
    model: &str,
    content: &str,
) -> Result<String, ApiError> {
    let id = json_string(completion_id)?;
    let model = json_string(model)?;
    let content = json_string(content)?;
    Ok(format!(
        r#"{{"id":{id},"object":"chat.completion","created":{created},"model":{model},"choices":[{{"index":0,"message":{{"role":"assistant","content":{content}}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}}}"#
    ))
}

fn json_string(value: &str) -> Result<String, ApiError> {
    serde_json::to_string(value).map_err(|e| ApiError::Internal(format!("json encode failed: {e}")))
}

fn local_anthropic_message_response(outbound: &Value, local: SwarmChatResult) -> Value {
    let input_tokens = outbound
        .get("messages")
        .and_then(Value::as_array)
        .map(|messages| {
            messages
                .iter()
                .map(|m| m.get("content").map(content_to_text).unwrap_or_default())
                .map(|text| text.split_whitespace().count())
                .sum::<usize>()
        })
        .unwrap_or(0);
    let id = json_string(&format!("msg_local_{}", Uuid::new_v4().simple()))
        .unwrap_or_else(|_| "\"msg_local\"".to_string());
    let model = json_string(&local.model).unwrap_or_else(|_| "\"local\"".to_string());
    let content = json_string(&local.content).unwrap_or_else(|_| "\"\"".to_string());
    let body = format!(
        r#"{{"id":{id},"type":"message","role":"assistant","content":[{{"type":"text","text":{content}}}],"model":{model},"stop_reason":"end_turn","stop_sequence":null,"usage":{{"input_tokens":{input_tokens},"output_tokens":0}}}}"#
    );
    serde_json::from_str(&body).unwrap_or(Value::Null)
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
            .filter_map(|b: &Value| b.get("text").and_then(Value::as_str).map(str::to_string))
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
        "openai_compression_active": state.openai_compression_active(),
        "threshold_tokens": state.compressor_config.heavy_message_threshold_tokens,
        "recall_top_k": state.compressor_config.recall_top_k,
    }))
}

/// `DELETE /v1/ttt/sessions/:id` — drop the W̃ tensors for one session.
async fn ttt_session_drop(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // Automatic merge trigger: fold the adapted W̃ into the master vibe before
    // the session's tensors are freed.
    let removed = match state.ttt_sessions.take_session(&id) {
        Some(handle) => {
            state.flush_session_to_vibe(&handle).await;
            true
        }
        None => false,
    };
    Json(serde_json::json!({ "session_id": id, "removed": removed }))
}

/// `DELETE /v1/ttt/sessions` — drop every TTT session.
async fn ttt_sessions_clear(State(state): State<AppState>) -> impl IntoResponse {
    // Flush all live sessions into the master vibe before clearing.
    state.flush_all_sessions_to_vibe().await;
    state.ttt_sessions.clear();
    Json(serde_json::json!({ "cleared": true }))
}

/// `POST /v1/ttt/feedback` — adapt an execution/compiler failure into a
/// session's W_tilde state and persist the updated compression cache.
async fn ttt_feedback(
    State(state): State<AppState>,
    Json(req): Json<TttFeedbackRequest>,
) -> Result<Json<TttFeedbackResponse>, ApiError> {
    let response = state
        .adapt_feedback_to_cache(req, &compression_cache_path())
        .await?;
    Ok(Json(response))
}

/// `POST /v1/hypervisor/mount` — install a safe user-mode VFS loopback mount.
/// Optional `warm_paths` are immediately read through the VFS and prefetched
/// into the session fast-weights.
async fn hypervisor_mount(
    State(state): State<AppState>,
    Json(req): Json<HypervisorMountRequest>,
) -> Result<Json<HypervisorMountResponse>, ApiError> {
    if req.root.trim().is_empty() {
        return Err(ApiError::BadRequest("root is required".into()));
    }
    let mount = state
        .neural_vfs
        .mount(req.root.trim())
        .map_err(ApiError::BadRequest)?;
    state
        .swarm_matrix
        .observe_vfs_target(&mount.mounted_root)
        .await;
    let session_id = req
        .session_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("hypervisor-vfs");
    let mut warmed = Vec::new();
    for path in req.warm_paths {
        let report = state
            .neural_vfs
            .read_file_and_prefill(
                path,
                session_id,
                state.pipeline.clone(),
                state.ttt_sessions.clone(),
            )
            .await
            .map_err(ApiError::BadRequest)?;
        warmed.push(report);
    }
    Ok(Json(HypervisorMountResponse {
        mount,
        warmed,
        vfs: state.neural_vfs.status(),
    }))
}

/// `GET /v1/hypervisor/jit_status` — report current user-mode JIT/VFS state.
async fn hypervisor_jit_status(State(state): State<AppState>) -> Json<HypervisorJitStatusResponse> {
    Json(HypervisorJitStatusResponse {
        jit: state.poly_jit.status(),
        vfs: state.neural_vfs.status(),
    })
}

/// `GET /v1/hypervisor/quantum_coherent_state` — report Q-TTT manifold state.
async fn hypervisor_quantum_coherent_state(
    State(state): State<AppState>,
) -> Json<HypervisorQuantumStateResponse> {
    Json(HypervisorQuantumStateResponse {
        quantum: state.poly_jit.quantum_status(),
    })
}

/// `GET /v1/swarm/matrix_state` — report localized model/DWE/SR-TTT state.
async fn swarm_matrix_state(State(state): State<AppState>) -> Json<SwarmMatrixStateResponse> {
    let dwe = state.dwe_bus.telemetry();
    let exact_residual = state.exact_residual_cache.telemetry();
    let matrix = state
        .swarm_matrix
        .state(dwe.clone(), exact_residual.clone());
    Json(SwarmMatrixStateResponse {
        matrix,
        dwe,
        exact_residual,
    })
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
        .route("/v1/cluster/merge", post(cluster_merge))
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions/:id", delete(delete_session))
        .route("/v1/adapt", post(adapt))
        .route("/v1/sessions/:id/checkpoint", get(get_checkpoint))
        .route("/v1/sessions/:id/checkpoint", put(put_checkpoint))
        .route("/v1/ttt/sessions", get(ttt_sessions_stats))
        .route("/v1/ttt/sessions", delete(ttt_sessions_clear))
        .route("/v1/ttt/sessions/:id", delete(ttt_session_drop))
        .route("/v1/ttt/feedback", post(ttt_feedback))
        .route("/v1/hypervisor/mount", post(hypervisor_mount))
        .route("/v1/hypervisor/jit_status", get(hypervisor_jit_status))
        .route(
            "/v1/hypervisor/quantum_coherent_state",
            get(hypervisor_quantum_coherent_state),
        )
        .route("/v1/swarm/matrix_state", get(swarm_matrix_state))
        .route("/v1/expand", post(expand_symbol_handler))
        .route("/v1/immunity", get(get_immunity))
        .route("/v1/immunity/merge", post(post_immunity_merge))
        .route("/v1/config", get(get_config).post(post_config))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// `POST /v1/expand` — the retrieval half of the skeleton round-trip. Given a
/// `session_id` and a `symbol` name, return the full declaration + body that the
/// digest dropped. Body: `{"session_id": "...", "symbol": "..."}`.
async fn expand_symbol_handler(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    let session_id = body.get("session_id").and_then(Value::as_str).unwrap_or("");
    let symbol = body.get("symbol").and_then(Value::as_str).unwrap_or("");
    if session_id.is_empty() || symbol.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "session_id and symbol are required"})),
        )
            .into_response();
    }

    let source = state
        .source_store
        .read()
        .ok()
        .and_then(|m| m.get(session_id).cloned());

    let Some(source) = source else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "no stored source for session_id (expired or never compressed)",
                "session_id": session_id,
            })),
        )
            .into_response();
    };

    match crate::skeleton::expand_symbol(&source, symbol) {
        Some(block) => Json(serde_json::json!({
            "session_id": session_id,
            "symbol": symbol,
            "found": true,
            "body": block,
        }))
        .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "session_id": session_id,
                "symbol": symbol,
                "found": false,
                "error": "symbol not found in stored source",
            })),
        )
            .into_response(),
    }
}

/// Prepend an `<axiom_immunity>` advisory to the outbound payload's last user
/// turn when the conversation references a command Axiom has learned to heal.
/// Disabled by `AXIOM_IMMUNITY_INJECT=0`; a no-op when heal memory is
/// unconfigured or nothing matches.
fn inject_immunity_advisory(
    state: &AppState,
    outbound: &mut Value,
    user_query: &str,
    heavy_context: &str,
) {
    if std::env::var("AXIOM_IMMUNITY_INJECT").as_deref() == Ok("0") {
        return;
    }
    let Some(path) = state.heal_memory_path.as_ref() else {
        return;
    };
    let text = format!("{user_query}\n{heavy_context}");
    let advisories = crate::heal_memory::HealMemory::load(path).advisories_for_text(&text);
    if advisories.is_empty() {
        return;
    }
    let mut block = String::from("<axiom_immunity>\nAxiom has prior self-healing experience with commands referenced here:\n");
    for a in &advisories {
        block.push_str("- ");
        block.push_str(a);
        block.push('\n');
    }
    block.push_str("</axiom_immunity>");
    eprintln!("[axiom-ttt] injected immunity advisory ({} command(s))", advisories.len());
    crate::anthropic_forwarder::prepend_block_to_last_user_turn(outbound, &block);
}

/// `GET /v1/immunity` — export this node's heal memory for swarm peers.
async fn get_immunity(State(state): State<AppState>) -> Response {
    let Some(path) = state.heal_memory_path.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "heal memory not configured on this node"})),
        )
            .into_response();
    };
    let memory = crate::heal_memory::HealMemory::load(path);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        memory.to_json(),
    )
        .into_response()
}

/// `POST /v1/immunity/merge` — fold a peer's exported heal memory into this
/// node's (swarm immunity). Body: the peer's `GET /v1/immunity` payload.
/// Returns the merge report. Local learning is never weakened: dirs are
/// unioned and tension histories are count-weighted.
async fn post_immunity_merge(State(state): State<AppState>, body: String) -> Response {
    let Some(path) = state.heal_memory_path.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "heal memory not configured on this node"})),
        )
            .into_response();
    };
    let mut memory = crate::heal_memory::HealMemory::load(path);
    match memory.merge_json(&body) {
        Ok(report) => {
            if let Err(e) = memory.save() {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": format!("merge succeeded but save failed: {e}")})),
                )
                    .into_response();
            }
            Json(serde_json::json!({
                "merged": true,
                "programs_added": report.programs_added,
                "programs_merged": report.programs_merged,
                "dirs_added": report.dirs_added,
            }))
            .into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

/// Map a friendly compression "level" to a per-message threshold (whitespace
/// words). Higher compression = lower threshold (more messages qualify).
fn level_to_threshold(level: &str) -> Option<usize> {
    match level.to_ascii_lowercase().as_str() {
        "high" => Some(80),    // aggressive — even medium pastes compress
        "medium" => Some(200), // conservative default — large pastes
        "low" => Some(400),    // only very large pastes
        _ => None,
    }
}

/// Derive the closest friendly level name from the current threshold (for GET).
fn threshold_to_level(t: usize) -> &'static str {
    if t <= 120 {
        "high"
    } else if t <= 300 {
        "medium"
    } else {
        "low"
    }
}

/// `GET /v1/config` — live compression state + counters for the dashboard.
async fn get_config(State(state): State<AppState>) -> impl IntoResponse {
    let (requests, msgs, bytes_in, bytes_out) = state.controls.counters();
    let enabled = state.controls.enabled();
    let threshold = state.controls.threshold();
    let savings_pct = if bytes_in > 0 {
        (1.0 - (bytes_out as f64 / bytes_in as f64)) * 100.0
    } else {
        0.0
    };
    Json(serde_json::json!({
        "enabled": enabled,
        "level": if enabled { threshold_to_level(threshold) } else { "off" },
        "threshold_tokens": threshold,
        "recall_top_k": state.compressor_config.recall_top_k,
        "forwarder_ready": state.anthropic_forwarder.is_some(),
        "openai_forwarder_ready": state.openai_forwarder.is_some(),
        "compression_active": state.compression_active(),
        "openai_compression_active": state.openai_compression_active(),
        "counters": {
            "requests": requests,
            "messages_compressed": msgs,
            "bytes_in": bytes_in,
            "bytes_out": bytes_out,
            "savings_pct": (savings_pct * 10.0).round() / 10.0,
            "degraded_fallbacks": state.controls.degraded_fallbacks(),
        }
    }))
}

/// `POST /v1/config` — retune compression live (no restart). Accepts any of:
/// `{"level":"off|low|medium|high"}`, `{"enabled":bool}`, `{"threshold_tokens":N}`.
async fn post_config(State(state): State<AppState>, Json(body): Json<Value>) -> impl IntoResponse {
    if let Some(level) = body.get("level").and_then(Value::as_str) {
        if level.eq_ignore_ascii_case("off") {
            state.controls.set_enabled(false);
        } else if let Some(t) = level_to_threshold(level) {
            state.controls.set_threshold(t);
            state.controls.set_enabled(true);
        }
    }
    if let Some(enabled) = body.get("enabled").and_then(Value::as_bool) {
        state.controls.set_enabled(enabled);
    }
    if let Some(t) = body.get("threshold_tokens").and_then(Value::as_u64) {
        state.controls.set_threshold(t as usize);
    }
    // Echo back the resulting live state.
    let enabled = state.controls.enabled();
    let threshold = state.controls.threshold();
    Json(serde_json::json!({
        "ok": true,
        "enabled": enabled,
        "level": if enabled { threshold_to_level(threshold) } else { "off" },
        "threshold_tokens": threshold,
        "compression_active": state.compression_active(),
        "openai_compression_active": state.openai_compression_active(),
    }))
}

/// Start the HTTP server and block until it is stopped.
pub async fn run_server(
    host: &str,
    port: u16,
    config: AxiomConfig,
    checkpoint_path: &str,
    device: Device,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("[*] Initializing system sanity check prior to binding network sockets...");

    // BPE tokenizer wiring: when AXIOM_TOKENIZER points at a tokenizer.json the
    // pipeline encodes with the real BPE vocab; otherwise it falls back to the
    // legacy hash tokenizer. with_checkpoint_and_options tolerates a missing
    // checkpoint (warns + random init), so the DEFAULT_CHECKPOINT_PATH special
    // case is no longer needed.
    let runtime = crate::inference::InferenceRuntimeOptions {
        tokenizer_path: std::env::var("AXIOM_TOKENIZER")
            .ok()
            .filter(|p| !p.trim().is_empty()),
        ..Default::default()
    };
    let pipeline_config = config.clone();
    let pipeline_device = device.clone();
    let pipeline_checkpoint = checkpoint_path.to_string();
    let pipeline = spawn_blocking(move || {
        InferencePipeline::with_checkpoint_and_options(
            pipeline_config,
            pipeline_device,
            &pipeline_checkpoint,
            runtime,
        )
    })
    .await
    .map_err(|e| format!("pipeline assembly task failed: {e}"))?
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
    let openai_forwarder = if compressor_config.enabled {
        OpenAiForwarder::from_env()
    } else {
        None
    };
    let swarm_router = if compressor_config.enabled {
        SwarmRouter::from_env()
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
        match openai_forwarder.as_ref() {
            Some(fwd) => {
                println!(
                    "[+] OpenAI/Codex compression bridge ON — /v1/chat/completions \
                     can absorb heavy messages locally, then forward a lean payload upstream."
                );
                if fwd.has_own_key() {
                    println!(
                        "[+] OpenAI upstream auth: proxy-owned OPENAI_API_KEY injected as \
                         bearer auth when the client sends none."
                    );
                } else {
                    println!(
                        "[+] OpenAI upstream auth: PASSTHROUGH — no proxy key set; the client's \
                         own Authorization header is relayed upstream."
                    );
                }
            }
            None => println!(
                "[!] Active-compression enabled but the OpenAI forwarder failed to construct — \
                 /v1/chat/completions will fall back to the local pipeline"
            ),
        }
        if let Some(router) = swarm_router.as_ref() {
            println!(
                "[+] Local swarm router ON — Ollama base={} candidates={:?} num_ctx={} timeout_ms={}",
                router.config().base_url,
                router.config().model_candidates,
                router.config().num_ctx,
                router.config().timeout_ms
            );
        } else {
            println!(
                "[+] Local swarm router OFF — set AXIOM_SWARM_LOCAL=1 to route compressed payloads to Ollama before cloud fallback."
            );
        }
    }
    // Persistent vibe memory (automatic EMA merge on session drop/clear/
    // shutdown). Enabled by default; set AXIOM_VIBE=0 to disable all
    // master-vibe persistence in the proxy.
    let vibe_enabled = std::env::var("AXIOM_VIBE")
        .map(|v| v != "0")
        .unwrap_or(true);
    let master_vibe = if vibe_enabled {
        let v = MasterVibe::from_env(config.n_layers, config.d_model, &device);
        println!(
            "[+] Persistent vibe memory ON — sessions EMA-merge into {} on drop/clear/shutdown \
             (decay={}). Set AXIOM_VIBE=0 to disable.",
            v.path().display(),
            v.decay()
        );
        Some(v)
    } else {
        println!("[+] Persistent vibe memory OFF (AXIOM_VIBE=0)");
        None
    };

    // Swarm immunity: serve/merge the same heal memory `axiom run` learns into
    // (AXIOM_HEAL_MEMORY overrides the path, 0/off disables the endpoints).
    let heal_memory_path = crate::heal_memory::HealMemory::default_path();
    if let Some(p) = heal_memory_path.as_ref() {
        println!("[+] Swarm immunity ON — /v1/immunity serves and merges {}", p.display());
    }

    let state = AppState::new(pipeline, model_id)
        .with_claude_backend(claude_backend)
        .with_anthropic_forwarder(anthropic_forwarder)
        .with_openai_forwarder(openai_forwarder)
        .with_swarm_router(swarm_router)
        .with_compressor_config(compressor_config)
        .with_master_vibe(master_vibe)
        .with_heal_memory_path(heal_memory_path);
    if state.compression_active() {
        match state.hydrate_compression_cache().await {
            Ok(0) => println!(
                "[+] Compression cache: no persisted adapted sessions found at {}",
                compression_cache_path().display()
            ),
            Ok(n) => println!(
                "[+] Compression cache: hydrated {n} adapted session(s) from {}",
                compression_cache_path().display()
            ),
            Err(e) => eprintln!("[!] Compression cache hydration skipped: {e}"),
        }
    }
    // Keep a handle for the graceful-shutdown flush (AppState is cheaply Clone).
    let shutdown_state = state.clone();
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

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            // Wait for Ctrl-C / SIGTERM, then flush all live sessions into the
            // persistent master vibe so accumulated structure survives restart.
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("[+] shutdown signal received — flushing vibe memory");
            shutdown_state.flush_all_sessions_to_vibe().await;
        })
        .await?;

    // The pipeline owns a `reqwest::blocking::Client` whose internal runtime
    // must not be dropped from within this async context. We are intentionally
    // terminating, so exit the process directly and let the OS reclaim memory
    // rather than running destructors on the async runtime.
    std::process::exit(0);
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

    fn seed_active_test_session(state: &AppState, session_id: &str) {
        let now = unix_now();
        let states = state
            .pipeline
            .lock()
            .unwrap()
            .init_session_states()
            .unwrap();
        let mut sessions = state.sessions.write().unwrap();
        sessions.insert(session_id.to_string(), SessionData::new_active(states, now));
        drop(sessions);
        metrics::register_session(session_id);
        state.refresh_session_metrics().unwrap();
    }

    async fn hydrate_test_cache(state: &AppState, path: &FsPath) -> usize {
        state.hydrate_compression_cache_from(path).await.unwrap()
    }

    #[tokio::test]
    async fn context_adaptation_cache_skips_identical_context_per_session() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();

        assert!(state.should_adapt_heavy_context("s1", "fn a() {}"));
        state.mark_heavy_context_adapted("s1", "fn a() {}");
        assert!(!state.should_adapt_heavy_context("s1", "fn a() {}"));
        assert!(state.should_adapt_heavy_context("s1", "fn b() {}"));
        assert!(state.should_adapt_heavy_context("s2", "fn a() {}"));

        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn compression_partition_uses_active_bpe_token_count() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let tokenizer = root.join("checkpoints/axiom_bpe.json");
        if !tokenizer.exists() {
            return;
        }

        let state = tokio::task::spawn_blocking(move || {
            use crate::config::AxiomConfig;
            use crate::inference::{InferencePipeline, InferenceRuntimeOptions};

            let config = AxiomConfig {
                d_model: 16,
                n_layers: 1,
                vocab_size: 16000,
                lr_inner: 1e-3,
                norm_eps: 1e-6,
            };
            let runtime = InferenceRuntimeOptions {
                tokenizer_path: Some(tokenizer.to_string_lossy().into_owned()),
                ..Default::default()
            };
            let pipeline =
                InferencePipeline::with_checkpoint_and_options(config, Device::Cpu, "", runtime)
                    .expect("pipeline init");
            AppState::new(pipeline, "axiom-ttt-test".to_string())
        })
        .await
        .unwrap();
        let pipeline_arc = state.pipeline.clone();

        let compact_code = "fn calculate_invoice_total(customer_id:&str)->Result<Money>{lookup_contract_discount(customer_id)?;Ok(Money::zero(\"USD\"))}";
        assert!(crate::anthropic_forwarder::whitespace_token_count(compact_code) < 20);

        let messages = vec![serde_json::json!({
            "role": "developer",
            "content": compact_code
        })];
        let partitioned = partition_messages_for_state(&state, &messages, 20).unwrap();

        assert_eq!(partitioned.heavy_context.len(), 1);
        assert!(partitioned.heavy_context[0].token_count >= 20);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn compression_cache_persists_hash_and_session_checkpoint() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let temp_path = std::env::temp_dir().join(format!(
            "axiom_compression_cache_test_{}_{}.bin",
            unix_now(),
            std::process::id()
        ));

        {
            let pipeline = state.pipeline.lock().unwrap();
            let handle = state
                .ttt_sessions
                .get_or_create("persist-s1", &pipeline)
                .unwrap();
            let mut states = handle.lock().await;
            let tokens = pipeline.encode_text("fn persisted() { cache(); }");
            adapt_session_blocking(&pipeline, &mut states, &tokens).unwrap();
        }
        state.mark_heavy_context_adapted("persist-s1", "fn persisted() { cache(); }");
        state
            .persist_compression_cache_to(&temp_path)
            .await
            .unwrap();

        let fresh = make_test_state().await;
        let fresh_pipeline_arc = fresh.pipeline.clone();
        fresh
            .hydrate_compression_cache_from(&temp_path)
            .await
            .unwrap();

        assert!(!fresh.should_adapt_heavy_context("persist-s1", "fn persisted() { cache(); }"));
        assert_eq!(fresh.ttt_sessions.len(), 1);

        let _ = std::fs::remove_file(&temp_path);
        safe_drop(pipeline_arc).await;
        safe_drop(fresh_pipeline_arc).await;
    }

    #[tokio::test]
    async fn feedback_loop_adapts_and_persists_session_checkpoint() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let temp_path = std::env::temp_dir().join(format!(
            "axiom_feedback_cache_test_{}_{}.bin",
            unix_now(),
            std::process::id()
        ));

        let response = state
            .adapt_feedback_to_cache(
                TttFeedbackRequest {
                    session_id: "feedback-s1".to_string(),
                    message: "compile error: expected struct LayerState field weight".to_string(),
                    feedback_type: Some("compilation_error".to_string()),
                    trace: Some("error[E0609]: no field `weights` on type LayerState".to_string()),
                },
                &temp_path,
            )
            .await
            .unwrap();

        assert_eq!(response.session_id, "feedback-s1");
        assert!(response.feedback_tokens > 0);
        assert!(temp_path.exists());
        assert_eq!(state.ttt_sessions.len(), 1);

        let fresh = make_test_state().await;
        let fresh_pipeline_arc = fresh.pipeline.clone();
        let restored = hydrate_test_cache(&fresh, &temp_path).await;
        assert_eq!(restored, 1);
        assert_eq!(fresh.ttt_sessions.len(), 1);

        let _ = std::fs::remove_file(&temp_path);
        safe_drop(pipeline_arc).await;
        safe_drop(fresh_pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_cluster_sync_merges_delta_into_layer_state() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state.clone());
        let session_id = "cluster-session".to_string();
        seed_active_test_session(&state, &session_id);

        let delta = Tensor::ones((16usize, 16usize), DType::F32, &state.device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let delta_bytes = safetensors::serialize([("tensor", &delta)], None).unwrap();
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
        seed_active_test_session(&state, &session_id);

        let delta = Tensor::ones((16usize, 16usize), DType::F32, &state.device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let delta_bytes = safetensors::serialize([("tensor", &delta)], None).unwrap();
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
