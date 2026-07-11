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
            Ok(resp) => return resp,
            Err(err) => return err.into_response(),
        }
    }

    let req: AnthropicMessagesRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => {
            return ApiError::BadRequest(format!("invalid /v1/messages body: {e}")).into_response();
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
) -> Result<Response, ApiError> {
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

    // Resolve / create the TTT session. Precedence: the X-Axiom-Session-Id
    // header (passed in as session_override), then a body `session_id`, then
    // a minted transient UUID. Persistent compression benefits accrue only
    // when the caller pins a stable id via one of the first two. Resolved
    // here (rather than after partitioning, as before S1) because the
    // cache-safety determinism memo below is keyed by session_id.
    let session_id = session_override
        .map(str::to_string)
        .or_else(|| {
            body.get("session_id")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| format!("transient-{}", Uuid::new_v4()));

    // S1 (CVM cost stack): cache-safety. Anthropic prompt caching is a
    // byte-exact prefix match; any change at or before a `cache_control`
    // breakpoint invalidates the client's cache for everything after it,
    // and simulation showed compression then costs MORE than it saves.
    // Split messages into a frozen prefix (never touched by compression)
    // and a mutable tail (the only part eligible for extraction). See
    // docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S1.
    let cache_safe_enabled = std::env::var("AXIOM_CACHE_SAFE").as_deref() != Ok("0");
    let uses_cache = cache_safe_enabled && crate::cache_safety::request_uses_cache(body);
    let frozen_len = if uses_cache {
        crate::cache_safety::frozen_prefix_len(body, &messages)
    } else {
        0
    };
    let (frozen_messages, mut mutable_messages): (Vec<Value>, Vec<Value>) = if frozen_len > 0 {
        (messages[..frozen_len].to_vec(), messages[frozen_len..].to_vec())
    } else {
        (Vec::new(), messages.clone())
    };
    // P2 (PSS) R2 free-window rebasing. When the client's prompt cache is
    // already broken this turn -- detected as a change in the frozen prefix vs
    // the session's previous turn (compaction / session restructure) -- the
    // whole prefix is re-written at the premium rate regardless. In exactly
    // that window we restructure every OLD heavy `tool_result` into a stub + L2
    // page at zero *marginal* cache cost, shrinking every FUTURE turn's re-read.
    // Never proxy-initiated: it only piggybacks on a break the client caused.
    // Default off (AXIOM_REBASE_ON_BREAK=on) until the live eval passes.
    if std::env::var("AXIOM_REBASE_ON_BREAK").as_deref() == Ok("on") {
        let frozen_hash = {
            use sha2::{Digest, Sha256};
            let bytes = serde_json::to_vec(&frozen_messages).unwrap_or_default();
            format!("{:x}", Sha256::digest(&bytes))
        };
        if state.pss_detect_break(&session_id, &frozen_hash) && mutable_messages.len() > 1 {
            let old_turns = mutable_messages.len() - 1;
            // rebase_transcript does synchronous L2-store writes + digest work
            // per old heavy turn, unbounded by transcript length -- offload it
            // to a blocking thread so a break turn can't pin a Tokio worker.
            let store = state.cvm_store.clone();
            let sid = session_id.clone();
            let msgs = std::mem::take(&mut mutable_messages);
            mutable_messages = tokio::task::spawn_blocking(move || {
                crate::rebase::rebase_transcript(&msgs, &store, &sid)
            })
            .await
            .map_err(|e| ApiError::Internal(format!("rebase task join failed: {e}")))?;
            eprintln!(
                "[axiom-pss] R2 rebase-on-break: session {session_id} restructured {old_turns} old turn(s)"
            );
        }
    }
    // Freeze-on-first-send determinism: the TTT fingerprint pipeline is not
    // naturally deterministic across repeated calls (each call mutates live
    // session state -- confirmed empirically: identical input produces a
    // different state_hash per call). Memoize the exact outbound messages
    // produced for a given mutable-tail content so identical input always
    // yields identical WIRE output, matching the abort criteria's required
    // mechanism rather than relying on the pipeline itself being pure.
    let mutable_hash = {
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(&mutable_messages).unwrap_or_default();
        let digest = Sha256::digest(&bytes);
        format!("{digest:x}")
    };
    let memo_key = format!("{session_id}:{mutable_hash}");
    let memoized_mutable_messages: Option<Vec<Value>> = uses_cache
        .then(|| state.cache_safety_memo_get(&memo_key))
        .flatten()
        .and_then(|s| serde_json::from_str::<Vec<Value>>(&s).ok());

    let cfg = state.compressor_config.clone();
    // Threshold is read live from the runtime controls so a dashboard can retune
    // it without a restart; top_k stays a startup constant.
    let base_threshold = state.controls.threshold();
    let top_k = cfg.recall_top_k;

    let mut threshold = base_threshold;
    let mut partitioned = partition_messages_for_state(state, &mutable_messages, threshold)?;

    // Confidence-gated adaptive compression (opt-in, AXIOM_ADAPTIVE_COMPRESS=1):
    // if the heavy context is surprising to the model (CE above the drift gate
    // → novel/high-information, unsafe to skeletonize), raise the threshold so
    // more is forwarded verbatim. The drift signal that flags hallucination on
    // the response path here gates the compression budget on the request path.
    if std::env::var("AXIOM_ADAPTIVE_COMPRESS").as_deref() == Ok("1")
        && !partitioned.heavy_context.is_empty()
    {
        let heavy_text = partitioned
            .heavy_context
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if let Some(ce) = mean_surprisal(state, &heavy_text) {
            let gate = std::env::var("AXIOM_DRIFT_THRESHOLD")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(7.03);
            let eff = crate::adaptive::adaptive_threshold(base_threshold, ce, gate);
            if eff != threshold {
                eprintln!(
                    "[axiom-ttt] adaptive compression: heavy surprisal {ce:.2} vs gate {gate:.2} → threshold {threshold} -> {eff}"
                );
                threshold = eff;
                partitioned = partition_messages_for_state(state, &mutable_messages, threshold)?;
            }
        }
    }

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
    inject_immunity_advisory(state, &mut outbound, &user_query_text, &heavy_combined);

    // S1 (CVM cost stack) cache-safety, continued: `outbound["messages"]` at
    // this point is entirely derived from `mutable_messages` (the frozen
    // prefix was excluded from partitioning above). Enforce determinism --
    // reuse a prior identical-input result verbatim if one exists, else
    // memoize this one -- then splice the untouched frozen prefix back onto
    // the front. See the memo/frozen-len computation near the top of this
    // function for the rationale.
    if uses_cache {
        if let Some(mutable_out) = memoized_mutable_messages {
            outbound["messages"] = Value::Array(mutable_out);
        } else if let Some(arr) = outbound.get("messages").and_then(Value::as_array) {
            if let Ok(serialized) = serde_json::to_string(arr) {
                state.cache_safety_memo_set(memo_key.clone(), serialized);
            }
        }
    }
    if !frozen_messages.is_empty() {
        let mutable_out = outbound
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut spliced = frozen_messages.clone();
        spliced.extend(mutable_out);
        outbound["messages"] = Value::Array(spliced);
    }
    eprintln!(
        "[axiom-ttt] cache_safe: frozen_blocks={} mutable_blocks={} compressed={}",
        frozen_messages.len(),
        mutable_messages.len(),
        log_heavy_count,
    );

    // S3 (CVM cost stack) digest admission control: replace heavy
    // tool_result blocks in the newest turn (by definition after every
    // cache breakpoint, not yet cached) with a digest + stub; the full
    // text goes to the L2 store. Default flipped to "skeleton" after S5's
    // live eval passed on 2026-07-11 (12/12 -> 11/12 correctness, 0%
    // fault rate, cost strictly lower) -- see
    // docs/superpowers/plans/2026-07-10-cvm-cost-stack.md and
    // bench/cvm/RESULTS-2026-07-11.md.
    let digest_mode =
        std::env::var("AXIOM_CVM_DIGEST").unwrap_or_else(|_| "skeleton".to_string());
    if digest_mode != "off" {
        let threshold = std::env::var("AXIOM_CVM_DIGEST_THRESHOLD_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(crate::digest::DEFAULT_DIGEST_THRESHOLD_TOKENS);
        apply_digest_admission(
            state,
            &session_id,
            &mut outbound,
            &digest_mode,
            threshold,
            forwarder.as_ref(),
            client_auth,
        )
        .await;
    }

    // S4 (CVM cost stack) prefix diet: lossless dedup of the fixed system
    // prefix. Gated the same way as S1's compression -- only when the
    // client actually caches (dedup output is a pure function of input
    // bytes, so it stays byte-stable turn over turn, which is what makes it
    // cache-safe without S1's determinism-memo mechanism). Stays default
    // off (AXIOM_PREFIX_DEDUP=1 opt-in) even after S5 passed (2026-07-11):
    // S4 has its own, separate, already-measured abort criteria (0% real
    // dedup gain on this machine's actual rule files, marked
    // DONE-BUT-WEAK) that S5's generic correctness/safety pass does not
    // override -- S5 is necessary but not sufficient for this specific
    // flag. See docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, S4's
    // status annotation.
    if std::env::var("AXIOM_PREFIX_DEDUP").as_deref() == Ok("1") && uses_cache {
        if let Some(system) = outbound.get("system").cloned() {
            let (dieted, report) = crate::prefix_diet::diet_system_field(&system);
            outbound["system"] = dieted;
            state
                .awareness
                .get_or_create(&session_id)
                .record_prefix_diet(report.dedup_tokens);
            state.prefix_diet_last_set(session_id.clone(), report);
            eprintln!(
                "[axiom-ttt] prefix_diet: original_tokens={} dedup_tokens={} blocks_deduped={}",
                report.original_tokens, report.dedup_tokens, report.blocks_deduped,
            );
        }
    }

    // P1 (Prolonged-Session Stack) L-A: tool deferral. Mark tools outside the
    // recent working set with `defer_loading: true` so they drop out of the
    // cached prefix (Anthropic loads them on demand as `tool_reference` blocks
    // without breaking the cache). Only ever ADDS the flag -- names/order/count
    // are unchanged -- so the tools array stays byte-stable turn-over-turn.
    // Default off (AXIOM_TOOL_DEFER=on) until the live eval passes. See
    // docs/superpowers/plans/2026-07-11-prolonged-session-stack.md, step P1.
    if std::env::var("AXIOM_TOOL_DEFER").as_deref() == Ok("on") {
        if let Some(tools) = outbound.get("tools").and_then(Value::as_array).cloned() {
            let keep = crate::tool_defer::working_set(&messages, 8);
            let (deferred_tools, deferred_count) = crate::tool_defer::mark_deferred(&tools, &keep);
            if deferred_count > 0 {
                outbound["tools"] = Value::Array(deferred_tools);
                eprintln!(
                    "[axiom-ttt] tool_defer: tools={} deferred={} kept={}",
                    tools.len(),
                    deferred_count,
                    tools.len() - deferred_count,
                );
            }
        }
    }

    // P2 (PSS) R3 adaptive cache TTL. Sessions with long thinking gaps keep
    // losing the 5-minute prompt cache and re-paying the full write. Once a
    // session's long-gap count crosses the threshold, elect Anthropic's 1-hour
    // TTL (a one-time 2x write premium beats repeated 1.25x full re-writes) by
    // annotating the newest `cache_control` breakpoint with `"ttl":"1h"`. This
    // only ever ADDS a field to the final breakpoint block; it never reorders
    // or removes content, so the cached prefix stays byte-stable. Default off
    // (AXIOM_ADAPTIVE_TTL=on) until the live eval passes.
    if std::env::var("AXIOM_ADAPTIVE_TTL").as_deref() == Ok("on") {
        let count = state.pss_gap_tick(&session_id, unix_now(), 240);
        if let Some(ttl) = crate::rebase::choose_ttl(count, 3) {
            if let Some(msgs) = outbound.get_mut("messages").and_then(Value::as_array_mut) {
                if crate::rebase::set_newest_cache_ttl(msgs, ttl) {
                    eprintln!(
                        "[axiom-pss] R3 adaptive-ttl: session {session_id} long_gaps={count} -> ttl={ttl}"
                    );
                }
            }
        }
    }

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
    record_savings(state, &session_id, bytes_in, bytes_out);

    if let Some(router) = state.swarm_router.as_ref().as_ref() {
        match router.route_chat_payload(&outbound).await {
            Ok(local) => {
                state
                    .sandbox_local_synthesis(&session_id, &local.content)
                    .await;
                return Ok(Json(local_anthropic_message_response(&outbound, local)).into_response());
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

    // Streaming clients (Claude Code sends `stream:true`) get a
    // `text/event-stream` upstream body. Relay `bytes_stream()` straight into the
    // client response: never JSON-parse it (that caused the `502 decode error:
    // expected value at line 1 column 1`) and never buffer it (which withholds
    // token chunks and lets long generations hit the request timeout). The
    // epistemic/ground-correct passes need a parsed body and so do not apply to a
    // live stream.
    let client_wants_stream = body
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if client_wants_stream {
        let mut upstream = forwarder
            .forward_messages_stream(&outbound, client_auth)
            .await
            .map_err(map_anthropic_forwarder_error)?;
        // Status/headers arrive before the body, so we can still retry ONCE with
        // the uncompressed payload on a 4xx/5xx that compression may have caused,
        // before streaming anything to the client.
        let did_compress = log_heavy_count > 0;
        if did_compress
            && (upstream.status() == StatusCode::BAD_REQUEST
                || upstream.status().is_server_error())
        {
            eprintln!(
                "[axiom-ttt] compressed stream forward returned {}; retrying once with \
                 original uncompressed payload (session={session_id})",
                upstream.status()
            );
            state.controls.record_degraded_fallback();
            let mut fallback = body.clone();
            if let Some(obj) = fallback.as_object_mut() {
                obj.remove("session_id");
            }
            if let Ok(retry) = forwarder.forward_messages_stream(&fallback, client_auth).await {
                upstream = retry;
            }
        }
        let status = upstream.status();
        // S6 (CVM cost stack): a real (streaming) request completed --
        // record activity for the keepalive timer. No-op when disabled.
        if status.is_success() && state.keepalive.is_enabled() {
            state.keepalive.record_activity(
                &session_id,
                crate::keepalive::HeldHeaders::from_client_auth(client_auth),
                outbound.clone(),
                forwarder.clone(),
            );
        }
        // The body streams through untouched, so record the request plus a
        // streamed marker rather than a buffered body.
        record_proxy_exchange(
            "/v1/messages",
            &session_id,
            body,
            serde_json::json!({"streamed": true, "status": status.as_u16()}),
            did_compress,
        );
        let content_type = upstream
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .unwrap_or_else(|| "text/event-stream".to_string());

        // S0 (CVM cost stack): scan the SSE bytes for `message_start` (input
        // side usage) and `message_delta` (output side usage) events as they
        // pass through, without buffering, delaying, or reordering a single
        // byte of the actual response -- `inspect` observes each chunk and
        // forwards it downstream unmodified. `message_start.message.usage`
        // and `message_delta.usage` are merged into one usage object and
        // priced when `message_delta` (the terminal usage event) arrives. A
        // chunk boundary that splits a multi-byte UTF-8 character can
        // corrupt this best-effort internal parse (never the bytes actually
        // sent to the client); worst case a turn's cost telemetry is missed.
        use futures::StreamExt;
        let awareness = state.awareness.get_or_create(&session_id);
        // Seeded from the request; overwritten by message_start's
        // message.model (the upstream-resolved model) the moment it arrives,
        // matching the non-streaming path's precedent of preferring the
        // response-declared model over the request's.
        let mut model_for_usage = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("claude-sonnet-4-6")
            .to_string();
        let mut sse_buffer = String::new();
        let mut collected_usage = serde_json::Map::new();
        let scanned = upstream.bytes_stream().inspect(move |chunk_result| {
            let Ok(bytes) = chunk_result else { return };
            sse_buffer.push_str(&String::from_utf8_lossy(bytes));
            while let Some(nl) = sse_buffer.find('\n') {
                let line: String = sse_buffer.drain(..=nl).collect();
                let line = line.trim_end_matches(['\r', '\n']);
                let Some(data) = line
                    .strip_prefix("data: ")
                    .or_else(|| line.strip_prefix("data:"))
                else {
                    continue;
                };
                let Ok(event) = serde_json::from_str::<Value>(data.trim()) else {
                    continue;
                };
                match event.get("type").and_then(Value::as_str) {
                    Some("message_start") => {
                        if let Some(m) = event
                            .get("message")
                            .and_then(|m| m.get("model"))
                            .and_then(Value::as_str)
                        {
                            model_for_usage = m.to_string();
                        }
                        if let Some(usage) = event
                            .get("message")
                            .and_then(|m| m.get("usage"))
                            .and_then(Value::as_object)
                        {
                            for (k, v) in usage {
                                collected_usage.insert(k.clone(), v.clone());
                            }
                        }
                    }
                    Some("message_delta") => {
                        if let Some(usage) = event.get("usage").and_then(Value::as_object) {
                            for (k, v) in usage {
                                collected_usage.insert(k.clone(), v.clone());
                            }
                        }
                        let usage_val = Value::Object(collected_usage.clone());
                        if let Some(tc) = crate::cost_ledger::turn_cost(&model_for_usage, &usage_val)
                        {
                            let (prices, _) =
                                crate::cost_ledger::PriceTable::for_model(&model_for_usage);
                            awareness.record_turn_cost(&tc, &prices);
                            awareness.record_turn_quota(crate::cost_ledger::quota_units(&tc, &prices));
                            record_lifetime_cost(&tc, &prices);
                        }
                        collected_usage.clear();
                    }
                    _ => {}
                }
            }
        });
        return Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, content_type)
            .body(axum::body::Body::from_stream(scanned))
            .map_err(|e| ApiError::Internal(format!("stream response build failed: {e}")));
    }

    // First attempt: forward the lean, compressed payload.
    let mut forwarded = match forwarder
        .forward_messages_json(&outbound, client_auth)
        .await
    {
        Ok(value) => value,
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
                forwarder
                    .forward_messages_json(&fallback, client_auth)
                    .await
                    .map_err(map_anthropic_forwarder_error)?
            } else {
                return Err(map_anthropic_forwarder_error(err));
            }
        }
    };

    // S0 (CVM cost stack): record dollar-true cost from the real usage
    // Anthropic just returned, before any opt-in follow-up call might
    // replace `forwarded`. See docs/superpowers/plans/2026-07-10-cvm-cost-stack.md.
    if let Some(usage) = forwarded.get("usage") {
        let model = forwarded
            .get("model")
            .and_then(Value::as_str)
            .or_else(|| body.get("model").and_then(Value::as_str))
            .unwrap_or("claude-sonnet-4-6");
        if let Some(tc) = crate::cost_ledger::turn_cost(model, usage) {
            let (prices, _) = crate::cost_ledger::PriceTable::for_model(model);
            let aware = state.awareness.get_or_create(&session_id);
            aware.record_turn_cost(&tc, &prices);
            aware.record_turn_quota(crate::cost_ledger::quota_units(&tc, &prices));
            record_lifetime_cost(&tc, &prices);
        }
    }

    // S6 (CVM cost stack): a real request completed -- record activity so
    // the keepalive timer (if AXIOM_KEEPALIVE=1) knows this session is
    // alive and holds its auth headers for a possible future ping. No-op
    // when keepalive is disabled (the default).
    if state.keepalive.is_enabled() {
        state.keepalive.record_activity(
            &session_id,
            crate::keepalive::HeldHeaders::from_client_auth(client_auth),
            outbound.clone(),
            forwarder.clone(),
        );
    }

    // Self-correction (opt-in, AXIOM_GROUND_CORRECT=1): if the answer makes
    // claims unsupported by the absorbed context, send ONE bounded follow-up
    // asking the model to revise grounded in that context, and return the
    // revision. Moves from *flagging* hallucinations to *reducing* them. Costs
    // one extra upstream call only when claims are actually flagged.
    if std::env::var("AXIOM_GROUND_CORRECT").as_deref() == Ok("1") {
        if let Some(revised) = ground_correct_round(
            &forwarder,
            client_auth,
            &outbound,
            &forwarded,
            &heavy_combined,
        )
        .await
        {
            forwarded = revised;
        }
    }

    // Automatic epistemic monitoring (opt-in, AXIOM_EPISTEMIC_AUTO=1): fire a
    // fire-and-forget semantic-judge validation of the answer against the
    // absorbed context. No-op unless explicitly configured; never blocks or
    // mutates the response.
    crate::epistemic_drift::spawn_automatic_validation(
        user_query_text,
        assistant_text(&forwarded),
        heavy_combined.clone(),
        body.get("model")
            .and_then(Value::as_str)
            .unwrap_or("anthropic-upstream")
            .to_string(),
    );

    // Auto grounding-verification (opt-in): annotate the response in place with
    // any claims not grounded in the context we just absorbed — hallucination
    // flagging with nobody asking. Never alters the answer text itself; appends
    // a clearly-marked advisory only when claims are flagged.
    annotate_response_grounding(&mut forwarded, &heavy_combined);
    record_proxy_exchange(
        "/v1/messages",
        &session_id,
        body,
        forwarded.clone(),
        log_heavy_count > 0,
    );
    Ok(Json(forwarded).into_response())
}

/// Extract the concatenated assistant text from an Anthropic `/v1/messages`
/// response (`content[].text`).
fn assistant_text(value: &Value) -> String {
    value
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// One bounded self-correction round. Returns the revised response only if the
/// answer had unsupported claims AND the corrective re-ask succeeded; otherwise
/// `None` (caller keeps the original). The follow-up reuses the *compressed*
/// outbound context, so correction stays token-efficient.
async fn ground_correct_round(
    forwarder: &AnthropicForwarder,
    client_auth: &ClientAuth,
    outbound: &Value,
    forwarded: &Value,
    evidence: &str,
) -> Option<Value> {
    if evidence.trim().is_empty() {
        return None;
    }
    let answer = assistant_text(forwarded);
    let report = crate::hallucination::verify(&answer, evidence);
    let flagged = report.flagged();
    if flagged.is_empty() {
        return None;
    }
    let claim_list = flagged
        .iter()
        .map(|c| format!("- {}", c.claim))
        .collect::<Vec<_>>()
        .join("\n");
    let correction = format!(
        "Your previous answer contained claims that are NOT supported by the provided context:\n{claim_list}\n\n\
         Revise your answer so every factual claim is grounded in the provided context. \
         Remove or explicitly mark as uncertain anything the context does not support. \
         Return only the corrected answer."
    );

    // Build the follow-up: the (compressed) turns we sent + the assistant's
    // answer + the correction request.
    let mut payload = outbound.clone();
    let messages = payload.get_mut("messages").and_then(Value::as_array_mut)?;
    messages.push(serde_json::json!({"role": "assistant", "content": answer}));
    messages.push(serde_json::json!({"role": "user", "content": correction}));

    match forwarder.forward_messages_json(&payload, client_auth).await {
        Ok(revised) => {
            // External-grounding guardrail: 2024 evidence (Huang et al. ICLR;
            // TACL critical survey) shows intrinsic self-correction can *degrade*
            // output. Keep the revision only if re-verifying it against the same
            // evidence shows it is strictly *more* grounded than the original —
            // i.e. gated on an external signal, not the model's self-critique.
            let revised_answer = assistant_text(&revised);
            if revised_answer.trim().is_empty() {
                eprintln!("[axiom-ttt] self-correction rejected (revision was empty)");
                return None;
            }
            let revised_report = crate::hallucination::verify(&revised_answer, evidence);
            if revised_report.grounded_fraction > report.grounded_fraction {
                eprintln!(
                    "[axiom-ttt] self-correction accepted: grounded {:.2} → {:.2} ({} flagged claim(s) addressed)",
                    report.grounded_fraction,
                    revised_report.grounded_fraction,
                    flagged.len()
                );
                Some(revised)
            } else {
                eprintln!(
                    "[axiom-ttt] self-correction rejected: revision not better grounded ({:.2} → {:.2}); keeping original",
                    report.grounded_fraction, revised_report.grounded_fraction
                );
                None
            }
        }
        Err(e) => {
            eprintln!("[axiom-ttt] self-correction skipped (re-ask failed: {e})");
            None
        }
    }
}

/// Pure: the grounding advisory block for an assistant `text` against
/// `evidence`, or `None` when nothing is flagged.
fn grounding_advisory_block(text: &str, evidence: &str) -> Option<String> {
    if text.trim().is_empty() || evidence.trim().is_empty() {
        return None;
    }
    let report = crate::hallucination::verify(text, evidence);
    let flagged = report.flagged();
    if flagged.is_empty() {
        return None;
    }
    let mut block = format!(
        "\n\n<axiom_grounding grounded_fraction=\"{:.2}\">\nThe following claims are not grounded in the provided context — verify before relying on them:",
        report.grounded_fraction
    );
    for c in &flagged {
        block.push_str(&format!("\n  - {}", c.claim));
    }
    block.push_str("\n</axiom_grounding>");
    Some(block)
}

/// Append a grounding advisory to the last `text` content block of an Anthropic
/// `/v1/messages` response, in place. Opt-in via `AXIOM_VERIFY_RESPONSES=1`.
fn annotate_response_grounding(value: &mut Value, evidence: &str) {
    if std::env::var("AXIOM_VERIFY_RESPONSES").as_deref() != Ok("1") {
        return;
    }
    // Concatenate assistant text blocks for verification.
    let text: String = value
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    let Some(block) = grounding_advisory_block(&text, evidence) else {
        return;
    };
    // Append to the last text block (or push one if none exist).
    if let Some(blocks) = value.get_mut("content").and_then(Value::as_array_mut) {
        if let Some(last) = blocks
            .iter_mut()
            .rev()
            .find(|b| b.get("type").and_then(Value::as_str) == Some("text"))
        {
            if let Some(t) = last.get("text").and_then(Value::as_str) {
                last["text"] = Value::String(format!("{t}{block}"));
            }
        } else {
            blocks.push(serde_json::json!({"type": "text", "text": block}));
        }
    }
    eprintln!("[axiom-ttt] grounding: appended advisory for ungrounded response claims");
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
    record_savings(state, &session_id, bytes_in, bytes_out);

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

    // Automatic epistemic monitoring (opt-in, AXIOM_EPISTEMIC_AUTO=1): mirror the
    // Anthropic forwarder path for OpenAI-compatible responses. Handles both a
    // single JSON completion and a streamed SSE body. Fire-and-forget; no-op
    // unless a judge is configured; never blocks or mutates the response. The
    // enabled-check gates body parsing so the hot path does no work when off.
    if upstream.status < 300 && crate::epistemic_drift::automatic_validation_enabled() {
        let answer = openai_assistant_answer(&upstream.body);
        if !answer.is_empty() {
            crate::epistemic_drift::spawn_automatic_validation(
                user_query_text.clone(),
                answer,
                heavy_combined.clone(),
                body.get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("openai-upstream")
                    .to_string(),
            );
        }
    }

    let status = StatusCode::from_u16(upstream.status).unwrap_or(StatusCode::BAD_GATEWAY);
    // The forwarder buffers this body in full, so record it verbatim (JSON
    // when parseable, else as a raw-string wrapper for SSE bodies).
    let recorded_body = serde_json::from_str::<Value>(&upstream.body)
        .unwrap_or_else(|_| serde_json::json!({"raw": upstream.body, "status": upstream.status}));
    record_proxy_exchange(
        "/v1/chat/completions",
        &session_id,
        body,
        recorded_body,
        log_heavy_count > 0,
    );
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

/// Extract the assistant answer text from an OpenAI-compatible chat-completion
/// response body, handling both a single JSON object (non-streamed:
/// `choices[0].message.content`) and a streamed SSE body (concatenating
/// `choices[].delta.content` across `data:` frames, ignoring the `[DONE]`
/// sentinel). Returns an empty string when no answer can be recovered.
fn openai_assistant_answer(body: &str) -> String {
    // Non-streamed: the whole body is one JSON completion object.
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        if let Some(text) = value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
        {
            return text.to_string();
        }
    }
    // Streamed (SSE): concatenate the incremental delta content from each frame.
    let mut answer = String::new();
    for line in body.lines() {
        let Some(payload) = line.trim_start().strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        if let Ok(chunk) = serde_json::from_str::<Value>(payload) {
            if let Some(piece) = chunk
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
            {
                answer.push_str(piece);
            }
        }
    }
    answer
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

/// Extract the flattened text of a `tool_result` content block's own
/// `content` field (which is independently either a string or an array of
/// text blocks, per the Anthropic Messages API -- distinct from
/// `content_to_text`, which reads a *message's* top-level `content`).
fn tool_result_text(block: &Value) -> Option<String> {
    if block.get("type").and_then(Value::as_str) != Some("tool_result") {
        return None;
    }
    let inner = block.get("content")?;
    let text = content_to_text(inner);
    (!text.is_empty()).then_some(text)
}

/// Overwrite a `tool_result` block's `content` with plain replacement text
/// (a valid, simpler-than-original shape the Anthropic API accepts).
fn set_tool_result_text(block: &mut Value, new_text: &str) {
    if let Some(obj) = block.as_object_mut() {
        obj.insert("content".to_string(), Value::String(new_text.to_string()));
    }
}

/// S3 (CVM cost stack) digest admission control: for each `tool_result`
/// block in the newest turn (`outbound["messages"]`'s last message -- by
/// construction always part of the mutable tail, since a real newest turn
/// comes after any `cache_control` breakpoint) whose token estimate is at
/// or above `threshold_tokens`, replace it with a stub + digest, storing
/// the full original text in the L2 store (`cvm_store`).
#[allow(clippy::too_many_arguments)]
async fn apply_digest_admission(
    state: &AppState,
    session_id: &str,
    outbound: &mut Value,
    digest_mode: &str,
    threshold_tokens: usize,
    forwarder: Option<&AnthropicForwarder>,
    client_auth: &ClientAuth,
) {
    let Some(messages) = outbound.get("messages").and_then(Value::as_array) else {
        return;
    };
    let Some(newest_idx) = messages.len().checked_sub(1) else {
        return;
    };
    let Some(content) = messages[newest_idx].get("content").and_then(Value::as_array) else {
        return;
    };

    let candidates: Vec<(usize, String)> = content
        .iter()
        .enumerate()
        .filter_map(|(i, block)| {
            let text = tool_result_text(block)?;
            (whitespace_token_count(&text) >= threshold_tokens).then_some((i, text))
        })
        .collect();
    if candidates.is_empty() {
        return;
    }

    let turn = state.digest_turn_next(session_id);
    let mut replacements: Vec<(usize, String)> = Vec::with_capacity(candidates.len());
    let mut bytes_in = 0usize;
    let mut bytes_out = 0usize;

    for (i, text) in candidates {
        let orig_tokens = whitespace_token_count(&text);
        let budget = ((orig_tokens as f64) * 0.15).round() as usize;

        let digest_text = if digest_mode == "haiku" {
            match forwarder {
                Some(fwd) => {
                    match crate::digest::haiku_digest(fwd, client_auth, &text, budget).await {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!(
                                "[axiom-cvm] haiku digest failed ({e}); falling back to skeleton"
                            );
                            crate::digest::SkeletonDigestor.digest(&text, budget)
                        }
                    }
                }
                None => crate::digest::SkeletonDigestor.digest(&text, budget),
            }
        } else {
            crate::digest::SkeletonDigestor.digest(&text, budget)
        };

        let page_id = match state.cvm_store.put(session_id, "tool_result", &text) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("[axiom-cvm] failed to store digested page: {e}");
                continue;
            }
        };
        state.digest_page_turn_set(session_id, &page_id, turn);
        let stub = crate::cvm_store::build_stub(&page_id, orig_tokens, "tool_result", &text);
        let replacement =
            format!("{stub}\n{digest_text}\n[AXIOM-PAGE-END expand with axiom_expand(\"{page_id}\")]");
        bytes_in += text.len();
        bytes_out += replacement.len();
        replacements.push((i, replacement));
    }
    if replacements.is_empty() {
        return;
    }

    if let Some(content) = outbound
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|m| m.get_mut(newest_idx))
        .and_then(|m| m.get_mut("content"))
        .and_then(Value::as_array_mut)
    {
        for (i, replacement) in &replacements {
            if let Some(block) = content.get_mut(*i) {
                set_tool_result_text(block, replacement);
            }
        }
    }

    state
        .awareness
        .get_or_create(session_id)
        .record_digest(replacements.len(), bytes_in, bytes_out);
    eprintln!(
        "[axiom-cvm] digest: mode={digest_mode} blocks={} bytes_in={bytes_in} bytes_out={bytes_out}",
        replacements.len(),
    );
}
