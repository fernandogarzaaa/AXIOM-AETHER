// OpenAI-compatible HTTP API server for the Axiom-TTT engine.
//
// Exposes the following endpoints:
//
// | Method | Path                                 | Description                              |
// |--------|--------------------------------------|------------------------------------------|
// | GET    | `/metrics`                           | Prometheus exposition endpoint           |
// | GET    | `/v1/models`                         | List available models                    |
// | POST   | `/v1/completions`                    | Text completion (stateless or session)   |
// | POST   | `/v1/chat/completions`               | Chat completion (stateless or session)   |
// | POST   | `/v1/responses`                      | Native OpenAI Responses API passthrough  |
// | POST   | `/v1/messages`                       | Anthropic Messages API (Claude clients)  |
// | POST   | `/v1/cluster/sync`                   | Delta state replication merge hook       |
// | GET    | `/v1/patches`                        | Signed verified-patch export (fleet)     |
// | POST   | `/v1/patches/merge`                  | Merge a peer's signed patch export       |
// | POST   | `/v1/sessions`                       | Create a new persistent TTT session      |
// | DELETE | `/v1/sessions/{id}`                  | Delete a session                         |
// | POST   | `/v1/adapt`                          | In-place TTT adaptation on a corpus      |
// | GET    | `/v1/sessions/{id}/checkpoint`       | Export session state as JSON             |
// | PUT    | `/v1/sessions/{id}/checkpoint`       | Restore session state from JSON          |

use std::collections::HashMap;
use std::convert::Infallible;
use std::fs;
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use axum::middleware::{self, Next};
use candle_core::{DType, Device, Tensor};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::task::{spawn_blocking, JoinSet};
use tower_http::cors::CorsLayer;
use tower_http::decompression::RequestDecompressionLayer;
use uuid::Uuid;

use crate::anthropic_forwarder::{
    build_compressed_payload, partition_messages, whitespace_token_count, AnthropicForwarder,
    ClientAuth, ForwarderError,
};
use crate::backend_router::{Router as BackendRouter, TaskKind};
use crate::claude_backend::{ChatTurn, ClaudeBackend};
use crate::cluster::StateDeltaUpdate;
use crate::config::AxiomConfig;
use crate::cvm_store::CvmStore;
use crate::context_compressor::{
    adapt_session_blocking, extract_memory_vector_blocking, feedback_adaptation_text,
    should_retry_uncompressed, CompressionControls, CompressorConfig, MemoryFingerprint,
    SessionStates, TttSessionStore,
};
use crate::digest::Digestor;
use crate::dwe::{
    extract_delta_fragment, record_applied_fragment, record_rejected_fragment, DweBus, DweTelemetry,
};
use crate::hamiltonian::QuantumRuntimeStatus;
use crate::inference::InferencePipeline;
use crate::metrics;
use crate::openai_forwarder::{OpenAiClientAuth, OpenAiForwarder, OpenAiForwarderError};
use crate::poly_jit::{PolyJitEngine, PolyJitReport, PolyJitRunRequest, PolyJitStatus};
use crate::quantization::{NF4QuantizedDescriptor, NF4Quantizer};
use crate::responses_compressor::{apply_plan, plan_compression};
use crate::sandbox::{SandboxController, SandboxDiagnostic};
use crate::session_awareness::AwarenessStore;
use crate::surprisal::{ExactAttentionResidualCache, ExactResidualTelemetry};
use crate::swarm_route::{LocalSwarmRouteMatrix, SwarmMatrixState};
use crate::swarm_router::{SwarmChatResult, SwarmRouter};
use crate::vfs::{NeuralVfs, VfsMountReport, VfsReadReport, VfsStats};
use crate::vibe_memory::MasterVibe;
use crate::weight_merge::{
    fleet_dare_ties, merge_checkpoint_files_with, MergeMethod, MergeSummary,
};

const MAX_ACTIVE_VRAM_SESSIONS: usize = 32;
const MAX_RESPONSES_RUN_CONCURRENCY: usize = 4;

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
    /// The base model + tokenizer. Immutable once loaded — every
    /// `InferencePipeline` method used on the hot path takes `&self`; the only
    /// mutable state involved in generation/adaptation is the per-session
    /// fast-weight `Vec<Tensor>` threaded through explicitly (see
    /// `context_compressor::TttSessionStore`), never `self`. `RwLock` (not
    /// `Mutex`) so concurrent requests — e.g. the JoinSet fan-out in
    /// `responses_run_fingerprint` — can hold the read lock simultaneously
    /// instead of serializing on a single writer-only mutex.
    pub pipeline: Arc<RwLock<InferencePipeline>>,
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
    /// Multi-provider router (`AXIOM_BACKEND=router`): when present, generation is
    /// routed across GPT/Claude/local with failover/consensus. `None` ⇒ the
    /// default single-backend path.
    pub router: Arc<Option<BackendRouter>>,
    /// Remote MCP context (`AXIOM_MCP_HTTP=1`): when present, the `/mcp` route
    /// exposes Axiom's MCP tools over HTTP (for the ChatGPT connector / Claude
    /// remote connectors), sharing the same dispatch as the stdio server.
    pub mcp: Arc<Option<crate::mcp_stdio::McpContext>>,
    /// Optional bearer token guarding the remote `/mcp` transport
    /// (`AXIOM_MCP_TOKEN`). When `Some`, `/mcp` requests must present
    /// `Authorization: Bearer <token>`; when `None`, `/mcp` is unauthenticated
    /// (intended only for trusted/local networks).
    pub mcp_token: Arc<Option<String>>,
    /// Optional pre-shared key guarding the data-plane routes (`AXIOM_API_KEY`).
    /// When `Some`, every route except the ops endpoints (`/healthz`, `/readyz`,
    /// `/metrics`) and `/mcp` (which has its own `AXIOM_MCP_TOKEN`) requires the
    /// header `X-Axiom-Key: <key>`. A dedicated header is used — not
    /// `Authorization`/`x-api-key` — so it never collides with the client
    /// credentials the `/v1/messages` proxy relays upstream. When `None`, the
    /// data plane is open (the local-first default).
    pub api_key: Arc<Option<String>>,
    /// Capability gate for `POST /v1/hypervisor/jit_run` (`AXIOM_ENABLE_JIT_EXEC`).
    /// That route executes an arbitrary caller-supplied `command`/`args` as a
    /// child process and can overwrite `source_path` with a synthesized patch —
    /// it is a `process.execute` + `filesystem.write` capability, not merely
    /// data-plane access. `AXIOM_API_KEY` controls *who* may call it; this flag
    /// controls whether the capability exists at all. Defaults to `false`
    /// (disabled) so a server started with no configuration cannot be used for
    /// remote command execution; an operator must opt in explicitly, ideally
    /// alongside `AXIOM_API_KEY`. See docs/SECURITY-AUDIT.md.
    pub jit_exec_enabled: bool,
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
    /// S1 (CVM cost stack) cache-safety determinism memo: key =
    /// "{session_id}:{sha256(mutable-tail raw JSON)}", value = the exact
    /// `messages` JSON array previously sent upstream for that identical
    /// mutable-tail content. The TTT fingerprint pipeline is not naturally
    /// deterministic across repeated calls (each call mutates live session
    /// state), so identical input is guaranteed identical WIRE output by
    /// overwriting the freshly-computed messages with this memoized copy
    /// rather than by making the pipeline itself pure. See
    /// docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S1.
    cache_safety_memo: Arc<RwLock<HashMap<String, String>>>,
    /// S4 (CVM cost stack) prefix-diet: the last request's dedup report per
    /// session, served by `GET /v1/prefix-diet/report/:session_id`.
    prefix_diet_last: Arc<RwLock<HashMap<String, crate::prefix_diet::DietReport>>>,
    /// S3 (CVM cost stack) digest admission control: a monotonically
    /// increasing per-session turn counter, incremented once per real
    /// `/v1/messages` request that reaches the digestion hook. Recorded
    /// alongside each digested page (out-of-band, not inside `CvmPage`
    /// itself, to avoid touching S2's already-shipped `CvmStore::put`
    /// signature) so a later `/v1/expand` fault can compute
    /// `turns_since_digest` -- S7's training signal. See
    /// docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S3.
    digest_turn: Arc<RwLock<HashMap<String, u64>>>,
    /// S3: turn number at which each `(session_id, page_id)` was digested,
    /// for the `turns_since_digest` fault computation above.
    digest_page_turn: Arc<RwLock<HashMap<(String, String), u64>>>,
    /// P2 (PSS) R2 break detection: last frozen-prefix fingerprint
    /// `(message count, hash)` per session. A GENUINE break is a non-append
    /// change to the prefix (compaction / session restructure) -- the free
    /// window in which `rebase_transcript` restructures old turns at zero
    /// marginal cache cost. Append-only growth (Anthropic's moving automatic
    /// breakpoint) is normal cached operation and must NOT count (the
    /// 2026-07-12 live-eval FAIL). See
    /// docs/superpowers/plans/2026-07-11-prolonged-session-stack.md, step P2.
    pss_prefix_hash: Arc<RwLock<HashMap<String, (usize, String)>>>,
    /// P2 (PSS) R3 adaptive TTL: `(last_turn_unix, long_gap_count)` per
    /// session. A gap longer than the 5-minute cache TTL increments the count;
    /// once it crosses a threshold, `rebase::choose_ttl` elects the 1-hour
    /// cache TTL (a one-time write premium beats repeated full re-writes).
    pss_gap: Arc<RwLock<HashMap<String, (u64, u32)>>>,
    /// P3 (PSS) L-B retry guard: `(last_local_inbound_hash, unix_ts)` per
    /// session. If the identical inbound arrives again within the cooldown, the
    /// client retried -- it did not accept our local ack -- so that turn is
    /// forwarded upstream instead of short-circuited again.
    pss_local_last: Arc<RwLock<HashMap<String, (String, u64)>>>,
    /// P4 (PSS) R1 routing sticky escalation: remaining cooldown turns per
    /// session. An error signature on a tool_result sets it to 3; each
    /// subsequent turn decrements it. While it is non-zero, routing is
    /// suppressed -- the session stays on its strong tier through the rough
    /// patch rather than being downgraded mid-debug.
    pss_route_cooldown: Arc<RwLock<HashMap<String, u32>>>,
    /// Runtime-mutable compression controls + live counters. Lets a dashboard
    /// retune the threshold / on-off without a restart (`/v1/config`).
    pub controls: Arc<CompressionControls>,
    /// Safe user-mode VFS loopback that feeds file reads into TTT prefill.
    pub neural_vfs: Arc<NeuralVfs>,
    /// Per-session compression savings ledger: session_id → (bytes_in,
    /// bytes_forwarded). Fed at each compression record site; drained into a
    /// one-line receipt when the session drops; summed for /metrics.
    pub savings: Arc<Mutex<HashMap<String, (u64, u64)>>>,
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
    /// Per-session token-awareness and self-awareness state. Tracks budget,
    /// token costs of Axiom responses, and compression metrics so Axiom can
    /// adapt autonomously. Populated via `POST /v1/budget`.
    pub awareness: AwarenessStore,
    /// S2 (CVM cost stack) L2 store: content-addressed, session-scoped
    /// full-fidelity text recoverable via `POST /v1/expand` and the
    /// `axiom_expand` MCP tool. Root overridable via `AXIOM_CVM_DIR`
    /// (default `checkpoints/cvm`). See
    /// docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S2.
    pub cvm_store: Arc<CvmStore>,
    /// S6 (CVM cost stack) actuarial keepalive. Defaults to
    /// `KeepaliveManager::disabled()` (spawns nothing, stores nothing);
    /// `run_server` opts in explicitly via `with_keepalive_manager` when
    /// `AXIOM_KEEPALIVE=1`. See
    /// docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S6.
    pub keepalive: crate::keepalive::KeepaliveManager,
}

impl AppState {
    pub fn new(pipeline: InferencePipeline, model_id: String) -> Self {
        let device = pipeline.device().clone();
        Self {
            pipeline: Arc::new(RwLock::new(pipeline)),
            device,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            sequence_versions: Arc::new(RwLock::new(HashMap::new())),
            model_id,
            claude_backend: Arc::new(None),
            router: Arc::new(None),
            mcp: Arc::new(None),
            mcp_token: Arc::new(None),
            api_key: Arc::new(None),
            jit_exec_enabled: false,
            ttt_sessions: Arc::new(TttSessionStore::new()),
            anthropic_forwarder: Arc::new(None),
            openai_forwarder: Arc::new(None),
            swarm_router: Arc::new(None),
            compressor_config: Arc::new(CompressorConfig::default()),
            master_vibe: Arc::new(Mutex::new(None)),
            source_store: Arc::new(RwLock::new(HashMap::new())),
            adapted_context_hashes: Arc::new(RwLock::new(HashMap::new())),
            cache_safety_memo: Arc::new(RwLock::new(HashMap::new())),
            prefix_diet_last: Arc::new(RwLock::new(HashMap::new())),
            digest_turn: Arc::new(RwLock::new(HashMap::new())),
            digest_page_turn: Arc::new(RwLock::new(HashMap::new())),
            pss_prefix_hash: Arc::new(RwLock::new(HashMap::new())),
            pss_gap: Arc::new(RwLock::new(HashMap::new())),
            pss_local_last: Arc::new(RwLock::new(HashMap::new())),
            pss_route_cooldown: Arc::new(RwLock::new(HashMap::new())),
            controls: Arc::new(CompressionControls::from_config(
                &CompressorConfig::default(),
            )),
            neural_vfs: Arc::new(NeuralVfs::new()),
            savings: Arc::new(Mutex::new(HashMap::new())),
            poly_jit: Arc::new(PolyJitEngine::default()),
            exact_residual_cache: Arc::new(ExactAttentionResidualCache::default()),
            dwe_bus: Arc::new(DweBus::from_env()),
            swarm_matrix: Arc::new(LocalSwarmRouteMatrix::new()),
            heal_memory_path: Arc::new(None),
            awareness: AwarenessStore::new(),
            cvm_store: Arc::new(Self::open_cvm_store()),
            keepalive: crate::keepalive::KeepaliveManager::disabled(),
        }
    }

    /// Open the S2 CVM store, falling back to a scratch temp directory (and
    /// logging why) if the configured/default root can't be created --
    /// never panics the server over a missing/unwritable disk path.
    fn open_cvm_store() -> CvmStore {
        let root =
            std::env::var("AXIOM_CVM_DIR").unwrap_or_else(|_| "checkpoints/cvm".to_string());
        CvmStore::open(&root).unwrap_or_else(|e| {
            eprintln!(
                "[axiom-cvm] failed to open CVM store at {root}: {e} -- falling back to a scratch temp dir"
            );
            let fallback = std::env::temp_dir().join("axiom-cvm-fallback");
            CvmStore::open(fallback).expect("temp dir CVM store open must succeed")
        })
    }

    /// Opt into S6 actuarial keepalive (see [`crate::keepalive`]). Callers
    /// should pass a manager built via `KeepaliveManager::from_env` (or
    /// `::disabled()` to explicitly no-op).
    pub fn with_keepalive_manager(mut self, manager: crate::keepalive::KeepaliveManager) -> Self {
        self.keepalive = manager;
        self
    }

    /// Enable the swarm-immunity endpoints against this heal-memory file.
    pub fn with_heal_memory_path(mut self, path: Option<std::path::PathBuf>) -> Self {
        self.heal_memory_path = Arc::new(path);
        self
    }

    /// Override the S2 CVM store (tests: inject an isolated temp-dir-backed
    /// store instead of the process-wide `AXIOM_CVM_DIR`/default path, which
    /// would otherwise race across tests running in parallel in the same
    /// binary -- the same hazard class as the S1 `AXIOM_CACHE_SAFE` race).
    pub fn with_cvm_store(mut self, store: CvmStore) -> Self {
        self.cvm_store = Arc::new(store);
        self
    }

    pub fn with_dwe_bus(mut self, bus: DweBus) -> Self {
        self.dwe_bus = Arc::new(bus);
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

    /// S1 cache-safety: look up a previously-memoized mutable-tail messages
    /// array for `key` ("{session_id}:{sha256(mutable-tail raw JSON)}").
    pub(crate) fn cache_safety_memo_get(&self, key: &str) -> Option<String> {
        self.cache_safety_memo
            .read()
            .ok()
            .and_then(|map| map.get(key).cloned())
    }

    /// S1 cache-safety: store the messages array actually sent upstream for
    /// `key`, so a later request with byte-identical mutable-tail content
    /// reuses it verbatim instead of re-deriving a fresh (and possibly
    /// different) fingerprint.
    pub(crate) fn cache_safety_memo_set(&self, key: String, messages_json: String) {
        if let Ok(mut map) = self.cache_safety_memo.write() {
            if map.len() >= 512 {
                map.clear();
            }
            map.insert(key, messages_json);
        }
    }

    /// S4 prefix-diet: the last request's dedup report for `session_id`, if
    /// any (`GET /v1/prefix-diet/report/:session_id`).
    pub(crate) fn prefix_diet_last_get(
        &self,
        session_id: &str,
    ) -> Option<crate::prefix_diet::DietReport> {
        self.prefix_diet_last
            .read()
            .ok()
            .and_then(|map| map.get(session_id).copied())
    }

    /// S4 prefix-diet: record this request's dedup report as the session's
    /// latest.
    pub(crate) fn prefix_diet_last_set(
        &self,
        session_id: String,
        report: crate::prefix_diet::DietReport,
    ) {
        if let Ok(mut map) = self.prefix_diet_last.write() {
            if map.len() >= 512 {
                map.clear();
            }
            map.insert(session_id, report);
        }
    }

    /// S3: advance and return `session_id`'s digest-turn counter. Call once
    /// per real request that reaches the digestion hook (not once per page
    /// digested -- multiple heavy tool_results in one turn share a turn
    /// number).
    pub(crate) fn digest_turn_next(&self, session_id: &str) -> u64 {
        let Ok(mut map) = self.digest_turn.write() else {
            return 0;
        };
        let entry = map.entry(session_id.to_string()).or_insert(0);
        *entry += 1;
        *entry
    }

    /// S3: record the turn at which `page_id` was digested for `session_id`.
    pub(crate) fn digest_page_turn_set(&self, session_id: &str, page_id: &str, turn: u64) {
        if let Ok(mut map) = self.digest_page_turn.write() {
            if map.len() >= 4096 {
                map.clear();
            }
            map.insert((session_id.to_string(), page_id.to_string()), turn);
        }
    }

    /// S3: `turns_since_digest` for a fault -- the session's current turn
    /// minus the turn `page_id` was created at, or `0` if unknown (e.g. the
    /// process restarted since digestion).
    pub(crate) fn digest_turns_since(&self, session_id: &str, page_id: &str) -> u64 {
        let created_turn = self
            .digest_page_turn
            .read()
            .ok()
            .and_then(|m| m.get(&(session_id.to_string(), page_id.to_string())).copied())
            .unwrap_or(0);
        let current_turn = self
            .digest_turn
            .read()
            .ok()
            .and_then(|m| m.get(session_id).copied())
            .unwrap_or(0);
        current_turn.saturating_sub(created_turn)
    }

    /// P2 (PSS) R2: returns `true` iff a GENUINE cache break is detected for
    /// `session_id` -- the frozen prefix changed in a NON-APPEND way vs the
    /// previous turn (compaction / session restructure), meaning the client's
    /// prompt cache was already invalidated this turn. Append-only growth
    /// (Anthropic's moving automatic breakpoint) is normal cached operation
    /// and returns `false` (see `rebase::is_genuine_break`). Always records
    /// this turn's fingerprint as the session's latest. The first turn of a
    /// session is never a break (nothing older to rebase).
    pub(crate) fn pss_detect_break(&self, session_id: &str, frozen: &[Value]) -> bool {
        let Ok(mut map) = self.pss_prefix_hash.write() else {
            return false;
        };
        if map.len() >= 512 {
            map.clear();
        }
        let fingerprint = crate::rebase::frozen_fingerprint(frozen);
        let prev = map.insert(session_id.to_string(), fingerprint);
        match prev {
            Some((prev_len, prev_hash)) => {
                crate::rebase::is_genuine_break(prev_len, &prev_hash, frozen)
            }
            None => false,
        }
    }

    /// P2 (PSS) R2: overwrite the session's stored prefix fingerprint with
    /// `frozen` as forwarded. Called after a break-window rebase mutates the
    /// frozen prefix, so the next turn's `pss_detect_break` compares against
    /// the transformed (stubbed) prefix that was actually sent upstream --
    /// otherwise the deterministic stub re-application would register as a
    /// spurious second break every following turn.
    pub(crate) fn pss_note_prefix(&self, session_id: &str, frozen: &[Value]) {
        if let Ok(mut map) = self.pss_prefix_hash.write() {
            map.insert(
                session_id.to_string(),
                crate::rebase::frozen_fingerprint(frozen),
            );
        }
    }

    /// P2 (PSS) R3: record this turn's timestamp for `session_id` and return
    /// the running long-gap count. A gap longer than `gap_threshold_secs` since
    /// the previous turn (the 5-minute cache TTL window) increments the count.
    /// The first turn seeds the timestamp without incrementing.
    pub(crate) fn pss_gap_tick(
        &self,
        session_id: &str,
        now_unix: u64,
        gap_threshold_secs: u64,
    ) -> u32 {
        let Ok(mut map) = self.pss_gap.write() else {
            return 0;
        };
        if map.len() >= 512 {
            map.clear();
        }
        let entry = map.entry(session_id.to_string()).or_insert((now_unix, 0));
        let (last_ts, count) = *entry;
        let new_count =
            crate::rebase::next_long_gap_count(last_ts, now_unix, count, gap_threshold_secs);
        *entry = (now_unix, new_count);
        new_count
    }

    /// P3 (PSS) L-B retry guard. Given this turn's `inbound_hash` and the
    /// current time, decide whether the trivial turn may be answered locally.
    /// Returns `false` (→ forward upstream) when the identical inbound was
    /// already answered locally within `cooldown_secs` -- the client retried,
    /// so it did not accept our local ack. Otherwise records this hash+time as
    /// the session's last local answer and returns `true`. The timestamp is
    /// refreshed on a retry too, so a burst of retries keeps forwarding.
    pub(crate) fn pss_local_admit(
        &self,
        session_id: &str,
        inbound_hash: &str,
        now: u64,
        cooldown_secs: u64,
    ) -> bool {
        let Ok(mut map) = self.pss_local_last.write() else {
            return false;
        };
        let prior = map.get(session_id).cloned();
        if let Some((h, ts)) = prior {
            if h == inbound_hash && now.saturating_sub(ts) < cooldown_secs {
                map.insert(session_id.to_string(), (inbound_hash.to_string(), now));
                return false;
            }
        }
        if map.len() >= 512 {
            map.clear();
        }
        map.insert(session_id.to_string(), (inbound_hash.to_string(), now));
        true
    }

    /// P4 (PSS) R1 routing sticky escalation. Advance `session_id`'s cooldown
    /// for this turn and return the remaining count. An error signature this
    /// turn (`had_error`) escalates to a fresh 3-turn cooldown; otherwise the
    /// counter decays by one. While the returned value is non-zero, `route`
    /// suppresses downgrades.
    pub(crate) fn pss_route_cooldown_tick(&self, session_id: &str, had_error: bool) -> u32 {
        let Ok(mut map) = self.pss_route_cooldown.write() else {
            return 0;
        };
        if map.len() >= 512 {
            map.clear();
        }
        let entry = map.entry(session_id.to_string()).or_insert(0);
        if had_error {
            *entry = 3;
        } else if *entry > 0 {
            *entry -= 1;
        }
        *entry
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
                .read()
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

    /// Install the multi-provider router (`AXIOM_BACKEND=router`). `None` keeps
    /// the default single-backend generation path.
    pub fn with_router(mut self, router: Option<BackendRouter>) -> Self {
        self.router = Arc::new(router);
        self
    }

    /// Install the remote MCP context, enabling the `/mcp` HTTP route.
    pub fn with_mcp(mut self, mcp: Option<crate::mcp_stdio::McpContext>) -> Self {
        self.mcp = Arc::new(mcp);
        self
    }

    /// Set the optional bearer token guarding the `/mcp` route.
    pub fn with_mcp_token(mut self, token: Option<String>) -> Self {
        self.mcp_token = Arc::new(token);
        self
    }

    /// Set the optional pre-shared key guarding the data-plane routes
    /// (`X-Axiom-Key` header). `None` leaves the data plane open.
    pub fn with_api_key(mut self, key: Option<String>) -> Self {
        self.api_key = Arc::new(key);
        self
    }

    /// Enable the `process.execute` capability behind `POST
    /// /v1/hypervisor/jit_run` (`AXIOM_ENABLE_JIT_EXEC`). Off by default; see
    /// the field doc on [`AppState::jit_exec_enabled`].
    pub fn with_jit_exec_enabled(mut self, enabled: bool) -> Self {
        self.jit_exec_enabled = enabled;
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
    /// A capability that exists in the binary but is not enabled on this
    /// server (e.g. `AXIOM_ENABLE_JIT_EXEC` unset). Distinct from `BadRequest`:
    /// the request is well-formed, the operator has simply not opted in.
    Forbidden(String),
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
            ApiError::Forbidden(msg) => (StatusCode::FORBIDDEN, msg).into_response(),
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
    /// Merge strategy: "dare_ties" (default; sign-elected, sparsified) or
    /// "alpha_blend" (uniform task-vector interpolation).
    #[serde(default)]
    method: Option<String>,
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

#[derive(Debug, Deserialize)]
struct HypervisorReadRequest {
    /// Path under the mounted root to read and absorb into the session's W̃.
    path: String,
    #[serde(default)]
    session_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct HypervisorReadResponse {
    read: VfsReadReport,
    vfs: VfsStats,
}

#[derive(Debug, Deserialize)]
struct HypervisorJitRunRequest {
    #[serde(default)]
    session_id: Option<String>,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    working_dir: Option<String>,
    /// Source artifact the Poly JIT may patch. When set, it is backed up before
    /// the run and restored if the repair does not ultimately pass.
    #[serde(default)]
    source_path: Option<String>,
}

/// `POST /v1/budget` request: agent reports remaining token budget.
#[derive(Debug, Deserialize)]
struct BudgetRequest {
    /// Remaining tokens in the agent's context window.
    remaining_tokens: usize,
    /// Optional model identifier, e.g. "claude-sonnet-4-6".
    model: Option<String>,
    /// Optional session ID to scope the budget (defaults to "global").
    session_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct HypervisorJitRunResponse {
    report: PolyJitReport,
    /// True when a failed repair left the source restored to its original bytes.
    source_restored: bool,
    jit: PolyJitStatus,
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
