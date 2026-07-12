/// Responses compression is ON by default (opt-out). It still requires general
/// compression (`state.controls.enabled()` / AXIOM_TTT_COMPRESS) to be active;
/// this gate only lets an operator disable the Responses path specifically via
/// AXIOM_RESPONSES_COMPRESS in {0,false,no,off}.
fn is_falsy_flag(value: &str) -> bool {
    matches!(value.to_lowercase().as_str(), "0" | "false" | "no" | "off")
}

fn responses_compression_enabled() -> bool {
    match std::env::var("AXIOM_RESPONSES_COMPRESS") {
        Ok(value) => !is_falsy_flag(&value),
        Err(_) => true,
    }
}

fn responses_compression_header_enabled(headers: &HeaderMap) -> bool {
    headers
        .get("x-axiom-responses-compress")
        .and_then(|value| value.to_str().ok())
        .map(|value| !is_falsy_flag(value))
        .unwrap_or(true)
}

fn responses_run_concurrency(run_count: usize) -> usize {
    run_count.clamp(1, MAX_RESPONSES_RUN_CONCURRENCY)
}

fn cleanup_responses_run_subsessions(
    state: &AppState,
    session_id: &str,
    run_count: usize,
    is_transient: bool,
) {
    if !is_transient {
        return;
    }
    for ordinal in 0..run_count {
        let _ = state
            .ttt_sessions
            .take_session(&format!("{session_id}#r{ordinal}"));
    }
}

async fn compressed_responses_payload(
    state: &AppState,
    body: &Value,
    session_override: Option<&str>,
    request_compression_enabled: bool,
) -> Result<Option<Value>, ApiError> {
    if !request_compression_enabled || !state.controls.enabled() || !responses_compression_enabled()
    {
        return Ok(None);
    }
    let Some(plan) = plan_compression(body) else {
        return Ok(None);
    };
    let threshold = state.controls.threshold();
    let context_tokens = state
        .pipeline
        .lock()
        .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?
        .token_count(&plan.total_context());
    if context_tokens < threshold {
        return Ok(None);
    }

    let session_id = session_override
        .map(str::to_string)
        .unwrap_or_else(|| format!("responses-{}", Uuid::new_v4()));
    let is_transient_session = session_override.is_none();

    // One fingerprint per contiguous run, each adapted in its own sub-session
    // so a run's recall vector reflects only that run's context. Runs are
    // independent, so fan them out with a small cap and restore plan order
    // before applying the transform.
    let concurrency = responses_run_concurrency(plan.runs.len());
    let mut fingerprints = vec![String::new(); plan.runs.len()];
    let mut tasks = JoinSet::new();
    let mut next_run = 0;
    while next_run < plan.runs.len() || !tasks.is_empty() {
        while next_run < plan.runs.len() && tasks.len() < concurrency {
            let ordinal = next_run;
            let run = &plan.runs[ordinal];
            let state = state.clone();
            let run_session = format!("{session_id}#r{ordinal}");
            let context = run.context.clone();
            let query = plan.query.clone();
            tasks.spawn(async move {
                let fingerprint =
                    responses_run_fingerprint(&state, &run_session, &context, &query).await?;
                Ok::<_, ApiError>((ordinal, fingerprint))
            });
            next_run += 1;
        }
        if let Some(result) = tasks.join_next().await {
            let (ordinal, fingerprint) = match result {
                Ok(Ok(value)) => value,
                Ok(Err(error)) => {
                    cleanup_responses_run_subsessions(
                        state,
                        &session_id,
                        plan.runs.len(),
                        is_transient_session,
                    );
                    return Err(error);
                }
                Err(error) => {
                    cleanup_responses_run_subsessions(
                        state,
                        &session_id,
                        plan.runs.len(),
                        is_transient_session,
                    );
                    return Err(ApiError::Internal(format!(
                        "Responses run task failed: {error}"
                    )));
                }
            };
            fingerprints[ordinal] = fingerprint;
        }
    }

    // Transient sessions (no client x-axiom-session-id) must not leak their
    // per-run adaptation sub-sessions — nothing will ever drop them. Runs on
    // BOTH the success and error branches of apply_plan and task fan-in.
    let compressed = match apply_plan(body, &plan, &fingerprints) {
        Some(c) => c,
        None => {
            cleanup_responses_run_subsessions(
                state,
                &session_id,
                plan.runs.len(),
                is_transient_session,
            );
            return Err(ApiError::Internal(
                "Responses compression transform failed".into(),
            ));
        }
    };
    cleanup_responses_run_subsessions(state, &session_id, plan.runs.len(), is_transient_session);

    let bytes_in = serde_json::to_vec(body)
        .map(|value| value.len())
        .unwrap_or(0) as u64;
    let bytes_out = serde_json::to_vec(&compressed)
        .map(|value| value.len())
        .unwrap_or(0) as u64;
    // Fingerprint + manifest overhead can exceed the replaced text on short
    // multi-run transcripts — compression must never expand the payload.
    if bytes_out >= bytes_in {
        eprintln!(
            "[axiom-ttt] responses compression skipped (would expand {bytes_in}=>{bytes_out} bytes); forwarding original"
        );
        return Ok(None);
    }
    state
        .controls
        .record(plan.item_indices().len() as u64, bytes_in, bytes_out);
    record_savings(state, &session_id, bytes_in, bytes_out);
    eprintln!(
        "[axiom-ttt] responses compressed session={} runs={} assistant_items={} tokens={} bytes={}=>{}",
        session_id,
        plan.runs.len(),
        plan.item_indices().len(),
        context_tokens,
        bytes_in,
        bytes_out
    );
    Ok(Some(compressed))
}

/// Adapt one run's context into a fresh TTT sub-session and return the recall
/// fingerprint block. Extracted from the former inline body of
/// `compressed_responses_payload` so each run gets its own adaptation.
async fn responses_run_fingerprint(
    state: &AppState,
    session_id: &str,
    context: &str,
    query: &str,
) -> Result<String, ApiError> {
    let pipeline_arc = state.pipeline.clone();
    let store = state.ttt_sessions.clone();
    let context = context.to_string();
    let query = query.to_string();
    let top_k = state.compressor_config.recall_top_k;
    let session_for_task = session_id.to_string();
    let started = Instant::now();
    let fingerprint = spawn_blocking(move || -> Result<_, ApiError> {
        let pipeline = pipeline_arc
            .lock()
            .map_err(|_| ApiError::Internal("pipeline lock poisoned".into()))?;
        let session = store
            .get_or_create(&session_for_task, &pipeline)
            .map_err(|error| ApiError::Internal(format!("session allocation failed: {error}")))?;
        let mut states = session.blocking_lock();
        let context_ids = pipeline.encode_text(&context);
        adapt_session_blocking(&pipeline, &mut states, &context_ids)
            .map_err(|error| ApiError::Internal(format!("TTT adapt failed: {error}")))?;
        let query_ids = pipeline.encode_text(&query);
        extract_memory_vector_blocking(
            &pipeline,
            &mut states,
            &query_ids,
            &session_for_task,
            context_ids.len(),
            started,
            top_k,
        )
        .map_err(|error| ApiError::Internal(format!("memory extraction failed: {error}")))
    })
    .await
    .map_err(|error| ApiError::Internal(format!("blocking task join failed: {error}")))??;
    Ok(fingerprint.to_prompt_block())
}

/// One-line human receipt: thousands of bytes with one decimal + saved ratio.
fn savings_receipt(bytes_in: u64, bytes_out: u64) -> String {
    let saved_pct = if bytes_in > 0 && bytes_out < bytes_in {
        ((bytes_in - bytes_out) * 100 / bytes_in) as u32
    } else {
        0
    };
    format!(
        "{:.1}k in, {:.1}k forwarded, {}% saved",
        bytes_in as f64 / 1000.0,
        bytes_out as f64 / 1000.0,
        saved_pct
    )
}

/// Lifetime savings counters: monotone across the process, independent of the
/// per-session ledger (whose entries are removed when sessions drop).
static LIFETIME_SAVINGS_IN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LIFETIME_SAVINGS_OUT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Lifetime dollar-true cost counters (S0, CVM cost stack). Same monotone
/// rationale as the savings counters above: per-session `AwarenessState` is
/// dropped when a session ends, so these are the only durable totals.
/// USD stored in micro-dollars (1e-6 USD) so accumulation stays exact.
pub(crate) static LIFETIME_COST_USD_MICROS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LIFETIME_COST_UNCACHED_EQUIVALENT_USD_MICROS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LIFETIME_CACHE_READ_TOKENS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LIFETIME_CACHE_WRITE_TOKENS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LIFETIME_UNCACHED_INPUT_TOKENS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// P0 (Prolonged-Session Stack): lifetime subscription quota units consumed,
/// stored as units x 1e6 so accumulation stays exact.
pub(crate) static LIFETIME_QUOTA_UNITS_MICROS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// P0: lifetime counterfactual quota units with zero caching (the denominator
/// for the stack's quota-savings ratio). Units x 1e6.
pub(crate) static LIFETIME_QUOTA_UNITS_UNCACHED_MICROS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// P3 (Prolonged-Session Stack): lifetime count of turns answered locally by
/// the L-B trivial-turn short-circuit (turns that never reached the network).
/// The headline signal for L-B's hit rate in the P5 live eval.
pub(crate) static LIFETIME_LOCAL_ANSWERED_TURNS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// P4 (Prolonged-Session Stack): lifetime R1 routing counters -- turns
/// downgraded to Haiku, routed turns that fell back after a 4xx, and the
/// subscription quota units saved by the downgrades (x 1e6). R1's live-eval
/// attribution.
pub(crate) static LIFETIME_ROUTED_TURNS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LIFETIME_ROUTE_FALLBACKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LIFETIME_ROUTED_QUOTA_SAVED_MICROS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Record one priced API turn into the lifetime cost counters. Called
/// alongside `AwarenessState::record_turn_cost` (the per-session view) so
/// `/metrics` totals survive individual sessions dropping.
pub(crate) fn record_lifetime_cost(
    tc: &crate::cost_ledger::TurnCost,
    prices: &crate::cost_ledger::PriceTable,
) {
    use std::sync::atomic::Ordering;
    let micros = (tc.usd * 1_000_000.0).round() as u64;
    let uncached_micros = (tc.uncached_equivalent_usd(prices) * 1_000_000.0).round() as u64;
    let quota_micros =
        (crate::cost_ledger::quota_units(tc, prices) * 1_000_000.0).round() as u64;
    let quota_uncached_micros =
        (crate::cost_ledger::quota_units_uncached(tc, prices) * 1_000_000.0).round() as u64;
    LIFETIME_COST_USD_MICROS.fetch_add(micros, Ordering::Relaxed);
    LIFETIME_COST_UNCACHED_EQUIVALENT_USD_MICROS.fetch_add(uncached_micros, Ordering::Relaxed);
    LIFETIME_CACHE_READ_TOKENS.fetch_add(tc.cache_read, Ordering::Relaxed);
    LIFETIME_CACHE_WRITE_TOKENS.fetch_add(tc.cache_write, Ordering::Relaxed);
    LIFETIME_UNCACHED_INPUT_TOKENS.fetch_add(tc.uncached_in, Ordering::Relaxed);
    LIFETIME_QUOTA_UNITS_MICROS.fetch_add(quota_micros, Ordering::Relaxed);
    LIFETIME_QUOTA_UNITS_UNCACHED_MICROS.fetch_add(quota_uncached_micros, Ordering::Relaxed);
}

/// Accumulate a compression event into the per-session savings ledger and the
/// lifetime totals.
fn record_savings(state: &AppState, session_id: &str, bytes_in: u64, bytes_out: u64) {
    use std::sync::atomic::Ordering;
    LIFETIME_SAVINGS_IN.fetch_add(bytes_in, Ordering::Relaxed);
    LIFETIME_SAVINGS_OUT.fetch_add(bytes_out, Ordering::Relaxed);
    if let Ok(mut ledger) = state.savings.lock() {
        let entry = ledger.entry(session_id.to_string()).or_insert((0, 0));
        entry.0 += bytes_in;
        entry.1 += bytes_out;
    }
}

/// Drop a session's ledger entry, printing its receipt when non-trivial.
fn emit_savings_receipt(state: &AppState, session_id: &str) {
    let entry = state
        .savings
        .lock()
        .ok()
        .and_then(|mut ledger| ledger.remove(session_id));
    if let Some((bytes_in, bytes_out)) = entry {
        if bytes_in > 0 {
            eprintln!(
                "[receipt] session {session_id}: {}",
                savings_receipt(bytes_in, bytes_out)
            );
        }
    }
}

/// Fire-and-forget session recording (AXIOM_SESSION_RECORD=1). `response` is
/// the full JSON body when available, else `{"streamed":true,"status":N}`.
/// File I/O runs on the blocking pool; failures never block the request path.
fn record_proxy_exchange(
    endpoint: &str,
    session_id: &str,
    request: &Value,
    response: Value,
    compressed: bool,
) {
    if !crate::session_recorder::recording_enabled() {
        return;
    }
    let record = crate::session_recorder::ExchangeRecord {
        ts: unix_now(),
        endpoint: endpoint.to_string(),
        session_id: session_id.to_string(),
        request: request.clone(),
        response,
        compressed,
    };
    tokio::task::spawn_blocking(move || crate::session_recorder::record_exchange(record));
}

/// Client headers relayed verbatim to the Responses upstream, selected by
/// allowlist (review hardening: a denylist silently forwards anything a
/// client invents — cookies, x-forwarded-*, internal headers). The list
/// covers what Codex and OpenAI SDKs actually send; extend via
/// `AXIOM_RESPONSES_RELAY_EXTRA` (comma-separated lowercase names).
fn relayable_response_headers(headers: &HeaderMap) -> Vec<(String, String)> {
    const ALLOW: &[&str] = &[
        "accept",
        "accept-language",
        "user-agent",
        "chatgpt-account-id",
        "openai-beta",
        "openai-organization",
        "openai-project",
        "session_id",
        "conversation_id",
        "originator",
        "x-request-id",
        "idempotency-key",
    ];
    let extra = std::env::var("AXIOM_RESPONSES_RELAY_EXTRA").unwrap_or_default();
    let extra: Vec<String> = extra
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    headers
        .iter()
        .filter_map(|(name, value)| {
            let name = name.as_str().to_ascii_lowercase();
            let allowed = ALLOW.contains(&name.as_str())
                || name.starts_with("openai-")
                || name.starts_with("x-stainless-") // official SDK telemetry headers
                || extra.iter().any(|e| e == &name);
            if !allowed {
                return None;
            }
            value.to_str().ok().map(|v| (name, v.to_string()))
        })
        .collect()
}

/// Upstream WebSocket URL for the Responses passthrough, mirroring the HTTP
/// upstream selection: explicit env override, ChatGPT backend when the client
/// is a ChatGPT-subscription Codex (chatgpt-account-id header), else platform.
fn responses_ws_upstream(headers: &HeaderMap) -> String {
    if let Ok(explicit) = std::env::var("AXIOM_OPENAI_RESPONSES_WS_UPSTREAM") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if headers.contains_key("chatgpt-account-id") {
        "wss://chatgpt.com/backend-api/codex/responses".to_string()
    } else {
        "wss://api.openai.com/v1/responses".to_string()
    }
}

/// `GET /v1/responses` (WebSocket upgrade) — transparent frame relay to the
/// upstream Responses WebSocket. Codex prefers this transport and previously
/// burned five 405-retries (~10s) before falling back to HTTPS. The proxy
/// does not interpret the protocol: client auth and relay headers are
/// forwarded on the CONNECT, then frames are pumped verbatim both ways until
/// either side closes. Compression does not apply on this path.
///
/// A GET that is *not* a complete WebSocket handshake (`ws` extracts to
/// `None`) gets a structured JSON diagnostic instead of axum's plain-text
/// extractor rejection, so misconfigured local Codex/OpenAI clients that try
/// to parse the error body as JSON see an actionable message (PR #84).
async fn responses_websocket(
    ws: Option<axum::extract::ws::WebSocketUpgrade>,
    headers: HeaderMap,
) -> Response {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    let Some(ws) = ws else {
        return (
            StatusCode::UPGRADE_REQUIRED,
            Json(serde_json::json!({
                "error": {
                    "message": "GET /v1/responses requires a complete WebSocket handshake (Codex relays frames upstream); for plain HTTP use POST with a JSON body — streaming Responses are delivered as SSE over HTTP.",
                    "type": "invalid_request_error",
                    "code": "responses_requires_post_or_websocket"
                }
            })),
        )
            .into_response();
    };

    let upstream_url = responses_ws_upstream(&headers);
    let mut request = match upstream_url.as_str().into_client_request() {
        Ok(r) => r,
        Err(e) => {
            return ApiError::Internal(format!("invalid upstream ws url: {e}")).into_response();
        }
    };
    // Relay client auth + the same header set as the HTTP path. Skip
    // WebSocket handshake headers — tungstenite generates its own.
    const WS_HANDSHAKE: &[&str] = &[
        "sec-websocket-key",
        "sec-websocket-version",
        "sec-websocket-extensions",
        "sec-websocket-protocol",
        "upgrade",
        "connection",
    ];
    if let Some(auth) = headers.get(header::AUTHORIZATION) {
        request
            .headers_mut()
            .insert(header::AUTHORIZATION, auth.clone());
    }
    for (name, value) in relayable_response_headers(&headers) {
        if WS_HANDSHAKE.contains(&name.as_str()) {
            continue;
        }
        if let (Ok(n), Ok(v)) = (
            reqwest::header::HeaderName::from_bytes(name.as_bytes()),
            reqwest::header::HeaderValue::from_str(&value),
        ) {
            request.headers_mut().insert(n, v);
        }
    }

    ws.on_upgrade(move |client_socket| async move {
        use axum::extract::ws::Message as AxMsg;
        use futures::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message as TgMsg;

        let upstream = match tokio_tungstenite::connect_async(request).await {
            Ok((socket, _resp)) => socket,
            Err(e) => {
                eprintln!("[responses-ws] upstream connect failed ({upstream_url}): {e}");
                let mut client = client_socket;
                let _ = client
                    .send(AxMsg::Close(Some(axum::extract::ws::CloseFrame {
                        code: 1011,
                        reason: format!("upstream connect failed: {e}").into(),
                    })))
                    .await;
                return;
            }
        };

        let (mut up_tx, mut up_rx) = upstream.split();
        let (mut cl_tx, mut cl_rx) = client_socket.split();

        // Client → upstream.
        let c2u = async {
            while let Some(Ok(msg)) = cl_rx.next().await {
                let out = match msg {
                    AxMsg::Text(t) => TgMsg::Text(t),
                    AxMsg::Binary(b) => TgMsg::Binary(b),
                    AxMsg::Ping(p) => TgMsg::Ping(p),
                    AxMsg::Pong(p) => TgMsg::Pong(p),
                    AxMsg::Close(_) => break,
                };
                if up_tx.send(out).await.is_err() {
                    break;
                }
            }
            let _ = up_tx.send(TgMsg::Close(None)).await;
        };

        // Upstream → client.
        let u2c = async {
            while let Some(Ok(msg)) = up_rx.next().await {
                let out = match msg {
                    TgMsg::Text(t) => AxMsg::Text(t),
                    TgMsg::Binary(b) => AxMsg::Binary(b),
                    TgMsg::Ping(p) => AxMsg::Ping(p),
                    TgMsg::Pong(p) => AxMsg::Pong(p),
                    TgMsg::Close(_) => break,
                    TgMsg::Frame(_) => continue, // raw frames never surface from read
                };
                if cl_tx.send(out).await.is_err() {
                    break;
                }
            }
            let _ = cl_tx.send(AxMsg::Close(None)).await;
        };

        // Run both pumps; when one side closes, the other unwinds.
        tokio::select! {
            _ = c2u => {}
            _ = u2c => {}
        }
    })
}

/// `POST /v1/responses` - native passthrough with optional semantic compression.
///
/// Structural items stay on their native wire protocol. When explicitly
/// enabled, only safe historical assistant text is replaced by a fingerprint.
async fn create_response(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let Some(forwarder) = state.openai_forwarder.as_ref().as_ref() else {
        return ApiError::Upstream {
            status: StatusCode::BAD_GATEWAY.as_u16(),
            message: "OpenAI Responses forwarder is not configured".into(),
        }
        .into_response();
    };
    let client_auth = OpenAiClientAuth {
        authorization: headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string),
        extra_headers: relayable_response_headers(&headers),
    };
    let session_override = headers
        .get("x-axiom-session-id")
        .and_then(|value| value.to_str().ok());
    let compressed = match compressed_responses_payload(
        &state,
        &body,
        session_override,
        responses_compression_header_enabled(&headers),
    )
    .await
    {
        Ok(payload) => payload,
        Err(error) => {
            eprintln!("[axiom-ttt] Responses compression skipped after local failure: {error:?}");
            state.controls.record_degraded_fallback();
            None
        }
    };
    let outbound = compressed.as_ref().unwrap_or(&body);
    let mut upstream = match forwarder.forward_responses(outbound, &client_auth).await {
        Ok(response) => response,
        Err(OpenAiForwarderError::Network(error)) if compressed.is_some() => {
            eprintln!(
                "[axiom-ttt] compressed Responses network failure ({error}); retrying original payload"
            );
            state.controls.record_degraded_fallback();
            match forwarder.forward_responses(&body, &client_auth).await {
                Ok(response) => response,
                Err(error) => return map_openai_forwarder_error(error).into_response(),
            }
        }
        Err(error) => return map_openai_forwarder_error(error).into_response(),
    };
    if compressed.is_some()
        && (upstream.status() == StatusCode::BAD_REQUEST || upstream.status().is_server_error())
    {
        eprintln!(
            "[axiom-ttt] compressed Responses returned {}; retrying original payload",
            upstream.status()
        );
        state.controls.record_degraded_fallback();
        if let Ok(response) = forwarder.forward_responses(&body, &client_auth).await {
            upstream = response;
        }
    }
    let status = upstream.status();
    // The Responses body always streams through untouched, so the record
    // carries the request plus a streamed marker rather than a buffered body.
    record_proxy_exchange(
        "/v1/responses",
        session_override.unwrap_or("anonymous"),
        &body,
        serde_json::json!({"streamed": true, "status": status.as_u16()}),
        compressed.is_some(),
    );
    let content_type = upstream
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("application/json"));
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, content_type)
        .body(axum::body::Body::from_stream(upstream.bytes_stream()))
        .unwrap_or_else(|error| {
            ApiError::Internal(format!("response build failed: {error}")).into_response()
        })
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
