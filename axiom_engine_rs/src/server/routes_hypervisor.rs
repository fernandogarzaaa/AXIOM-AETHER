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
    emit_savings_receipt(&state, &id);
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
/// `POST /v1/hypervisor/read` — absorb a single mounted file into a session's
/// fast-weights incrementally (after `mount`), returning the read report and
/// live VFS stats. The path is confined to the mounted root.
async fn hypervisor_read(
    State(state): State<AppState>,
    Json(req): Json<HypervisorReadRequest>,
) -> Result<Json<HypervisorReadResponse>, ApiError> {
    if req.path.trim().is_empty() {
        return Err(ApiError::BadRequest("path is required".into()));
    }
    let session_id = req
        .session_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("hypervisor-vfs");
    let read = state
        .neural_vfs
        .read_file_and_prefill(
            req.path.trim(),
            session_id,
            state.pipeline.clone(),
            state.ttt_sessions.clone(),
        )
        .await
        .map_err(ApiError::BadRequest)?;
    Ok(Json(HypervisorReadResponse {
        read,
        vfs: state.neural_vfs.status(),
    }))
}

/// `POST /v1/hypervisor/list` — readdir over the mounted VFS. Body:
/// `{"path": "relative/or/absolute"}` (default `.` = the mount root).
/// Paths are canonicalized and confined to the mounted root.
async fn hypervisor_list(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let path = body.get("path").and_then(Value::as_str).unwrap_or(".");
    let entries = state
        .neural_vfs
        .readdir(path)
        .map_err(ApiError::BadRequest)?;
    Ok(Json(serde_json::json!({
        "entries": entries,
        "vfs": state.neural_vfs.status(),
    })))
}

/// `POST /v1/hypervisor/stat` — getattr over the mounted VFS. Body:
/// `{"path": "relative/or/absolute"}`. Same root confinement as `list`.
async fn hypervisor_stat(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let Some(path) = body.get("path").and_then(Value::as_str) else {
        return Err(ApiError::BadRequest("path is required".into()));
    };
    let attr = state
        .neural_vfs
        .getattr(path)
        .map_err(ApiError::BadRequest)?;
    Ok(Json(serde_json::json!({
        "attr": attr,
        "vfs": state.neural_vfs.status(),
    })))
}

/// `POST /v1/hypervisor/jit_run` — drive the Poly JIT closed loop: run a
/// command; on failure feed the fault trace into the session's W̃ (TTT) and
/// apply a bounded, deterministic source patch (Q-TTT-ranked candidates), then
/// retry — up to the engine's step cap. SAFETY: when a `source_path` is given
/// it is backed up first and **restored** if the repair does not pass, so a
/// failed attempt never leaves the artifact corrupted.
async fn hypervisor_jit_run(
    State(state): State<AppState>,
    Json(req): Json<HypervisorJitRunRequest>,
) -> Result<Json<HypervisorJitRunResponse>, ApiError> {
    if req.command.trim().is_empty() {
        return Err(ApiError::BadRequest("command is required".into()));
    }
    let session_id = req
        .session_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("hypervisor-jit")
        .to_string();
    let source_path = req.source_path.clone();
    // Back up the source so a failed repair is always reversible.
    let backup = match source_path.as_deref() {
        Some(p) => Some(
            tokio::fs::read_to_string(p)
                .await
                .map_err(|e| ApiError::BadRequest(format!("cannot read source_path: {e}")))?,
        ),
        None => None,
    };

    let run_req = PolyJitRunRequest {
        session_id: session_id.clone(),
        command: req.command,
        args: req.args,
        working_dir: req.working_dir,
        source_path: source_path.clone(),
    };

    let cache_path = compression_cache_path();
    let report = state
        .poly_jit
        .run_with_feedback(run_req, |diag| {
            // Feed each fault trace into the session's fast-weights (TTT).
            let state = state.clone();
            let cache_path = cache_path.clone();
            async move {
                let message = format!(
                    "poly-jit fault at step {} (exit {:?})",
                    diag.step, diag.status_code
                );
                let trace = format!("{}\n{}", diag.stdout, diag.stderr);
                // TTT feedback is auxiliary: a cache hiccup must never abort the
                // repair loop, so failures are logged and swallowed.
                if let Err(e) = state
                    .adapt_feedback_to_cache(
                        TttFeedbackRequest {
                            session_id: diag.session_id.clone(),
                            message,
                            feedback_type: Some("poly_jit_fault".to_string()),
                            trace: Some(trace),
                        },
                        &cache_path,
                    )
                    .await
                {
                    eprintln!("[poly-jit] TTT feedback skipped: {e:?}");
                }
                Ok(())
            }
        })
        .await
        .map_err(|e| ApiError::Internal(format!("poly-jit run failed: {e}")))?;

    // SAFETY: restore the original source if the repair did not pass.
    let mut source_restored = false;
    if !report.passed {
        if let (Some(p), Some(b)) = (source_path.as_deref(), backup.as_ref()) {
            if tokio::fs::write(p, b).await.is_ok() {
                source_restored = true;
            }
        }
    }

    Ok(Json(HypervisorJitRunResponse {
        report,
        source_restored,
        jit: state.poly_jit.status(),
    }))
}

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
// Liveness / readiness probes (Kubernetes, Cloud Run, ALB health checks)
// ---------------------------------------------------------------------------
