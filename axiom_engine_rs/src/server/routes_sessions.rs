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
            .read()
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
        // Sessions are only ever freed by an explicit DELETE; without a cap a
        // client can allocate W_tilde tensors until the process dies.
        let cap = state.max_sessions;
        if sessions.len() >= cap {
            return Err(ApiError::Conflict(format!(
                "session limit reached ({cap} live sessions; DELETE /v1/sessions/{{id}} to free one, \
                 or raise AXIOM_MAX_SESSIONS)"
            )));
        }
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
        emit_savings_receipt(&state, &session_id);
        // S2 (CVM cost stack): drop this session's L2 store file too, unless
        // the operator opted into retention for post-mortem inspection.
        if std::env::var("AXIOM_CVM_RETAIN").as_deref() != Ok("1") {
            if let Err(e) = state.cvm_store.delete_session(&session_id) {
                eprintln!("[axiom-cvm] failed to delete CVM store for session={session_id}: {e}");
            }
        }
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
    // The inner TTT loop is pure CPU/GPU work; run it on the blocking pool so
    // a long corpus does not stall every other request on this worker.
    let updated_states = {
        let pipeline = state.pipeline.clone();
        let corpus = req.corpus;
        spawn_blocking(move || {
            let pipeline = pipeline
                .read()
                .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
            pipeline
                .adapt_on_corpus_with_steps(&corpus, initial_states, steps_per_token)
                .map_err(|e| ApiError::Internal(format!("adapt failed: {e}")))
        })
        .await
        .map_err(|e| ApiError::Internal(format!("adapt task join failed: {e}")))??
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
                payload.sequence_version, existing.version, payload.timestamp, existing.timestamp
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
    // The request names files on this server's disk. Confine every path to
    // the merge roots so a data-plane caller cannot read arbitrary files
    // (error messages echo the path) or overwrite arbitrary ones.
    let roots = merge_allowed_roots();
    let inputs = req
        .inputs
        .iter()
        .map(|p| confine_merge_path(FsPath::new(p), &roots, true))
        .collect::<Result<Vec<_>, _>>()
        .map_err(ApiError::BadRequest)?;
    let output = confine_merge_path(FsPath::new(&req.output), &roots, false)
        .map_err(ApiError::BadRequest)?;
    // Fleet merges default to DARE-TIES (sign-elected, sparsified) so agreeing
    // peers compound and conflicting deltas don't cancel; "alpha_blend" selects
    // the uniform task-vector interpolation instead.
    let method = match req.method.as_deref().map(str::trim) {
        Some("alpha_blend") => MergeMethod::AlphaBlend { alpha },
        Some("dare_ties") | None => fleet_dare_ties(alpha),
        Some(other) => {
            return Err(ApiError::BadRequest(format!(
                "unknown merge method '{other}' (expected \"dare_ties\" or \"alpha_blend\")"
            )));
        }
    };
    let summary = spawn_blocking(move || merge_checkpoint_files_with(&inputs, &output, method))
        .await
        .map_err(|e| ApiError::Internal(format!("merge task join failed: {e}")))?
        .map_err(ApiError::BadRequest)?;
    Ok(Json(summary))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Directories `POST /v1/cluster/merge` may read inputs from and write its
/// output into. `AXIOM_MERGE_ROOT` (one or more directories separated by `;`)
/// sets them explicitly; otherwise the current working directory and
/// `~/.axiom` are allowed, which covers the compression-cache files the route
/// exists to merge. Roots that do not exist are skipped.
fn merge_allowed_roots() -> Vec<PathBuf> {
    let configured = std::env::var("AXIOM_MERGE_ROOT").ok();
    let candidates: Vec<PathBuf> = match configured.as_deref().map(str::trim) {
        Some(v) if !v.is_empty() => v
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect(),
        _ => {
            let mut v = Vec::new();
            if let Ok(cwd) = std::env::current_dir() {
                v.push(cwd);
            }
            if let Some(home) = dirs::home_dir() {
                v.push(home.join(".axiom"));
            }
            v
        }
    };
    candidates
        .into_iter()
        .filter_map(|p| p.canonicalize().ok())
        .collect()
}

/// Resolve `path` and check it lies under one of `roots`.
///
/// For `must_exist` (inputs) the whole path is canonicalized. For an output
/// that may not exist yet, the nearest existing ancestor is canonicalized and
/// checked instead, and `..` components are rejected outright so the un-
/// canonicalized tail cannot climb back out. Returns the resolved path.
fn confine_merge_path(path: &FsPath, roots: &[PathBuf], must_exist: bool) -> Result<PathBuf, String> {
    use std::path::Component;
    if path.as_os_str().is_empty() {
        return Err("path must not be empty".into());
    }
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!("path '{}' must not contain '..'", path.display()));
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| format!("cannot resolve current directory: {e}"))?
            .join(path)
    };
    let resolved = if must_exist {
        absolute
            .canonicalize()
            .map_err(|e| format!("input '{}' is not readable: {e}", path.display()))?
    } else {
        // Walk up to the nearest existing ancestor, canonicalize it, then
        // re-attach the (already '..'-free) remainder.
        let mut existing = absolute.clone();
        let mut tail: Vec<std::ffi::OsString> = Vec::new();
        while !existing.exists() {
            let Some(name) = existing.file_name() else {
                return Err(format!("output '{}' has no existing ancestor", path.display()));
            };
            tail.push(name.to_os_string());
            existing = existing
                .parent()
                .map(FsPath::to_path_buf)
                .ok_or_else(|| format!("output '{}' has no existing ancestor", path.display()))?;
        }
        let mut resolved = existing
            .canonicalize()
            .map_err(|e| format!("output '{}' is not resolvable: {e}", path.display()))?;
        for name in tail.iter().rev() {
            resolved.push(name);
        }
        resolved
    };
    if roots.iter().any(|root| resolved.starts_with(root)) {
        Ok(resolved)
    } else {
        Err(format!(
            "path '{}' is outside the merge roots (set AXIOM_MERGE_ROOT to allow other directories)",
            path.display()
        ))
    }
}

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
        .read()
        .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
    Ok(pipeline.token_count(prompt))
}

fn count_corpus_tokens(state: &AppState, corpus: &[String]) -> Result<usize, ApiError> {
    let pipeline = state
        .pipeline
        .read()
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
        .read()
        .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
    Ok(partition_messages(messages, threshold, |text| {
        pipeline.token_count(text)
    }))
}

/// Default for `AXIOM_MAX_SESSIONS`: cap on `/v1/sessions` W_tilde sessions.
const MAX_LEGACY_SESSIONS: usize = 1024;

/// A positive integer from an env value, or `default` for unset / empty /
/// zero / unparsable. Pure so tests never mutate the process environment.
fn parse_positive_usize(raw: Option<&str>, default: usize) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// `AXIOM_MAX_SESSIONS` or [`MAX_LEGACY_SESSIONS`]; read once into
/// `AppState::max_sessions` at construction.
fn max_legacy_sessions_from_env() -> usize {
    parse_positive_usize(
        std::env::var("AXIOM_MAX_SESSIONS").ok().as_deref(),
        MAX_LEGACY_SESSIONS,
    )
}

/// Hard ceiling on `max_tokens` for any generation served by this process
/// (`/v1/completions`, `/v1/chat/completions`, local `/v1/messages`). Override
/// with `AXIOM_MAX_GENERATION_TOKENS`. The request value is clamped, not
/// rejected, so OpenAI-style clients that send a large default keep working.
const MAX_GENERATION_TOKENS: usize = 16_384;

fn clamp_max_tokens(requested: usize) -> usize {
    clamp_max_tokens_with(
        requested,
        std::env::var("AXIOM_MAX_GENERATION_TOKENS").ok().as_deref(),
    )
}

/// [`clamp_max_tokens`] with the env value passed in.
fn clamp_max_tokens_with(requested: usize, ceiling_env: Option<&str>) -> usize {
    requested.min(parse_positive_usize(ceiling_env, MAX_GENERATION_TOKENS))
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
    // `max_tokens` is attacker-controlled: an enormous value used to size a
    // pre-allocation (or spin the decode loop) for as long as the client
    // likes. Clamp it to the server ceiling before any backend sees it.
    let max_tokens = clamp_max_tokens(max_tokens);
    // Router mode (AXIOM_BACKEND=router): GPT + Claude + local together, with
    // failover. Generation is delegated to external providers, so session TTT
    // state does not apply here.
    if let Some(router) = state.router.as_ref() {
        return router
            .generate(TaskKind::General, prompt, max_tokens)
            .map(|a| a.text)
            .map_err(|e| ApiError::Internal(format!("router generation failed: {e}")));
    }

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
                .read()
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
                    .read()
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
                .read()
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
