fn patch_store_path(state: &AppState) -> Option<PathBuf> {
    (*state.heal_memory_path)
        .as_ref()
        .map(|p| p.with_file_name("axiom_patch_memory.json"))
}

/// `GET /v1/patches` — this node's verified-patch store, provenance-signed
/// (SHA-256 + optional HMAC via `AXIOM_FLEET_KEY`) so a peer can verify before
/// trusting. The payload only carries candidates; a peer never applies them
/// without re-verifying locally (see `POST /v1/patches/merge`).
async fn get_patches(State(state): State<AppState>) -> Response {
    let Some(path) = patch_store_path(&state) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "patch memory not configured on this node"})),
        )
            .into_response();
    };
    let memory = crate::patch_memory::PatchMemory::load(&path);
    let signed = memory.export_signed(fleet_key().as_deref());
    Json(signed).into_response()
}

/// `POST /v1/patches/merge` — fold a peer's signed patch export (its
/// `GET /v1/patches` payload) into this node's store. Provenance is enforced:
/// `merge_signed` verifies the signature/hash and recomputes content hashes, and
/// a fleet key (if configured) is required. SAFETY: merging only *records*
/// candidates — a peer's fix is never executed or written until the autonomous
/// repair loop re-verifies it green locally, so an incorrect or malicious peer
/// patch is inert.
async fn post_patches_merge(State(state): State<AppState>, body: String) -> Response {
    let Some(path) = patch_store_path(&state) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "patch memory not configured on this node"})),
        )
            .into_response();
    };
    let export: crate::provenance::SignedExport = match serde_json::from_str(&body) {
        Ok(e) => e,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "expected a signed patch export (the GET /v1/patches payload)"
                })),
            )
                .into_response();
        }
    };
    let mut memory = crate::patch_memory::PatchMemory::load(&path);
    // Byzantine-robust: a peer holding the fleet key still cannot inflate trust
    // or flood the store — the robust policy bounds per-peer contribution. Local
    // re-verification still gates any application.
    match memory.merge_signed_guarded(
        &export,
        fleet_key().as_deref(),
        crate::patch_memory::FleetTrustPolicy::robust(),
    ) {
        Ok(report) => {
            if let Err(e) = memory.save(&path) {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("merge succeeded but save failed: {e}")
                    })),
                )
                    .into_response();
            }
            Json(serde_json::json!({
                "merged": true,
                "new_candidates": report.new_candidates,
                "reinforced": report.reinforced,
                "byzantine_rejected": report.byzantine_rejected,
            }))
            .into_response()
        }
        Err(e) => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": format!("provenance rejected: {e}")})),
        )
            .into_response(),
    }
}

/// `POST /v1/chimera/run` — execute a ChimeraLang program with AXIOM's in-tree
/// implementation ([`crate::chimera`]). Body: `{"source": "..."}`. Returns the
/// emitted values, belief means, guard violations, and trace. When a generation
/// router is configured (`AXIOM_BACKEND=router/openai/opendrop`), `inquire`
/// beliefs are grounded in it via [`crate::chimera::RouterAdapter`] (with agent
/// pins like `[claude]`/`[gpt]` honored); otherwise the offline mock adapter is
/// used so programs still run.
///
/// Experimental: compiled only with `--features experimental` (see
/// `docs/EXPERIMENTAL.md`); the `/v1/chimera/run` route is registered only then.
#[cfg(feature = "experimental")]
async fn post_chimera_run(State(state): State<AppState>, body: String) -> axum::response::Response {
    use axum::response::IntoResponse;
    let source = match serde_json::from_str::<serde_json::Value>(&body) {
        Ok(v) => v
            .get("source")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string()),
        Err(_) => None,
    };
    let Some(source) = source else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "expected JSON body {\"source\": \"...\"}"})),
        )
            .into_response();
    };
    // Ground beliefs in the live router when available. The router's HTTP
    // backends are blocking, so run on a blocking thread (cloning the Arc to keep
    // the adapter's borrow `'static`).
    let result = if state.router.is_some() {
        let router_arc = state.router.clone();
        let src = source.clone();
        tokio::task::spawn_blocking(move || {
            // Safe: router_arc.is_some() checked above and the Arc keeps it alive.
            let router = router_arc.as_ref().as_ref().unwrap();
            let adapter = crate::chimera::RouterAdapter { router };
            crate::chimera::run_source(&src, Some(&adapter))
        })
        .await
        .unwrap_or_else(|e| Err(format!("chimera execution task panicked: {e}")))
    } else {
        crate::chimera::run_source(&source, None)
    };
    match result {
        Ok(res) => Json(serde_json::json!({
            "emitted": res.emitted,
            "beliefs": res.beliefs,
            "guard_violations": res.guard_violations,
            "trace": res.trace,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

/// Check the `Authorization: Bearer <token>` header against the configured
/// `AXIOM_MCP_TOKEN`. Returns `None` when access is allowed (no token configured,
/// or the presented token matches) and `Some(401 response)` when it must be
/// rejected. The token is a pre-shared secret carried over TLS, so a plain
/// equality check (not constant-time) is acceptable here.
fn mcp_unauthorized(state: &AppState, headers: &axum::http::HeaderMap) -> Option<Response> {
    use axum::response::IntoResponse;
    let expected = state.mcp_token.as_ref().as_ref()?; // None ⇒ open access
    let presented = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.strip_prefix("Bearer ")
                .or_else(|| v.strip_prefix("bearer "))
        })
        .map(str::trim);
    if presented == Some(expected.as_str()) {
        None
    } else {
        Some(
            (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({
                    "error": "missing or invalid bearer token for /mcp (set Authorization: Bearer <AXIOM_MCP_TOKEN>)"
                })),
            )
                .into_response(),
        )
    }
}

/// `POST /mcp` — remote MCP transport. Dispatches a JSON-RPC 2.0 request against
/// Axiom's MCP tools (the same dispatch the stdio server uses) and returns the
/// JSON-RPC response. Enables ChatGPT connectors and Claude remote connectors to
/// drive Axiom over HTTP. Returns 503 unless started with `AXIOM_MCP_HTTP=1`,
/// and 401 if `AXIOM_MCP_TOKEN` is set and the bearer token is missing/wrong.
async fn post_mcp(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: String,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = mcp_unauthorized(&state, &headers) {
        return resp;
    }
    let ctx = match state.mcp.as_ref() {
        Some(c) => c,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "MCP HTTP transport disabled; start the server with AXIOM_MCP_HTTP=1"
                })),
            )
                .into_response();
        }
    };
    match crate::mcp_stdio::dispatch(&body, ctx).await {
        // Request → JSON-RPC response.
        Some(resp) => Json(resp).into_response(),
        // Notification (no id) → accepted, no body.
        None => StatusCode::ACCEPTED.into_response(),
    }
}

/// `GET /mcp` — SSE stream for Streamable-HTTP MCP clients that open a stream for
/// server-initiated messages. Axiom's tool calls are request/response (handled by
/// `POST /mcp`), so this stream just stays open with keep-alive pings.
async fn mcp_http_sse(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = mcp_unauthorized(&state, &headers) {
        return resp;
    }
    if state.mcp.is_none() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "MCP HTTP transport disabled; start the server with AXIOM_MCP_HTTP=1",
        )
            .into_response();
    }
    let keepalive = stream::pending::<Result<Event, std::convert::Infallible>>();
    Sse::new(keepalive)
        .keep_alive(KeepAlive::default())
        .into_response()
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

/// `POST /v1/budget` — agent reports remaining token budget.
///
/// Stores the budget in the awareness store. When a budget is set, the
/// compression threshold is automatically capped to 60 % of remaining
/// tokens so Axiom doesn't overshoot the context window.
async fn post_budget(
    State(state): State<AppState>,
    Json(body): Json<BudgetRequest>,
) -> impl IntoResponse {
    let id = body.session_id.as_deref().unwrap_or("global");
    let awareness = state.awareness.get_or_create(id);
    awareness.set_budget(body.remaining_tokens, body.model.clone());

    // Auto-tune compression threshold: cap at 60 % of remaining budget.
    if let Some(target) = awareness.compression_target_tokens() {
        let current = state.controls.threshold();
        if target < current {
            state.controls.set_threshold(target);
        }
    }

    Json(serde_json::json!({
        "ok": true,
        "session_id": id,
        "remaining_tokens": body.remaining_tokens,
        "compression_threshold_tokens": state.controls.threshold(),
        "model": body.model,
    }))
}

/// `GET /v1/awareness/{id}` — return the current awareness state for a session.
async fn get_awareness(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match state.awareness.get(&id) {
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "no awareness state for session", "id": id })),
        )
            .into_response(),
        Some(a) => {
            let (requests, msgs, bytes_in, bytes_out) = state.controls.counters();
            let cost = a.cost_summary();
            Json(serde_json::json!({
                "session_id": id,
                "budget_remaining": a.budget(),
                "target_model": a.target_model.lock().ok().and_then(|g| g.clone()),
                "tokens_spent_on_axiom": a.tokens_spent.load(std::sync::atomic::Ordering::Relaxed),
                "tool_calls_total": a.tool_calls_total.load(std::sync::atomic::Ordering::Relaxed),
                "expansion_calls": a.expansion_calls.load(std::sync::atomic::Ordering::Relaxed),
                "compression_ratio": a.compression_ratio(),
                "compression_target_tokens": a.compression_target_tokens(),
                "is_tight": a.is_tight(),
                "recommendation": a.recommendation(),
                "cost": {
                    "usd_total": cost.usd_total,
                    "usd_uncached_equivalent": cost.usd_uncached_equivalent,
                    "uncached_input_tokens": cost.uncached_input_tokens,
                    "cache_write_tokens": cost.cache_write_tokens,
                    "cache_read_tokens": cost.cache_read_tokens,
                    "output_tokens": cost.output_tokens,
                    "cache_hit_rate": cost.cache_hit_rate(),
                    "estimated": cost.estimated,
                },
                "global_counters": {
                    "requests": requests,
                    "messages_compressed": msgs,
                    "bytes_in": bytes_in,
                    "bytes_out": bytes_out,
                },
            }))
            .into_response()
        }
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
