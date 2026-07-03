/// `GET /healthz` — liveness. Cheap and unconditional: if the process can
/// answer, it is alive. Never touches the pipeline lock so a long in-flight
/// generation can't make the orchestrator think the pod is dead and kill it.
async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// `GET /readyz` — readiness. The server only binds its socket *after* the
/// inference pipeline is assembled (see `run_server`), so reaching this
/// handler already implies the model loaded. We additionally confirm the
/// pipeline lock is reachable (not poisoned) and report the live model id so
/// the probe doubles as a smoke check. Returns 503 if the lock is poisoned.
async fn readyz(State(state): State<AppState>) -> Response {
    match state.pipeline.try_lock() {
        Ok(_) | Err(std::sync::TryLockError::WouldBlock) => (
            StatusCode::OK,
            Json(serde_json::json!({ "status": "ready", "model": state.model_id })),
        )
            .into_response(),
        Err(std::sync::TryLockError::Poisoned(_)) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(
                serde_json::json!({ "status": "unavailable", "reason": "pipeline lock poisoned" }),
            ),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Router construction
// ---------------------------------------------------------------------------

/// Build the axum Router with all API routes attached.
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(export_metrics))
        .route("/v1/fleet/status", get(fleet_status))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(create_completion))
        .route("/v1/chat/completions", post(create_chat_completion))
        .route(
            "/v1/responses",
            post(create_response).get(responses_websocket),
        )
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
        .route("/v1/hypervisor/read", post(hypervisor_read))
        .route("/v1/hypervisor/list", post(hypervisor_list))
        .route("/v1/hypervisor/stat", post(hypervisor_stat))
        .route("/v1/hypervisor/jit_run", post(hypervisor_jit_run))
        .route("/v1/hypervisor/jit_status", get(hypervisor_jit_status))
        .route(
            "/v1/hypervisor/quantum_coherent_state",
            get(hypervisor_quantum_coherent_state),
        )
        .route("/v1/swarm/matrix_state", get(swarm_matrix_state))
        .route("/v1/expand", post(expand_symbol_handler))
        .route("/v1/verify", post(verify_grounding))
        .route("/v1/epistemic/validate", post(validate_epistemic_drift))
        .route("/v1/immunity", get(get_immunity))
        .route("/v1/immunity/merge", post(post_immunity_merge))
        .route("/v1/patches", get(get_patches))
        .route(
            "/v1/patches/merge",
            // Patch candidates carry full file contents, so a signed export of a
            // real fix can exceed axum's 2 MB default body limit. Raise it for
            // this route so larger-but-valid fixes can still gossip.
            post(post_patches_merge).layer(DefaultBodyLimit::max(32 * 1024 * 1024)),
        )
        .route("/v1/chimera/run", post(post_chimera_run))
        .route("/mcp", get(mcp_http_sse).post(post_mcp))
        .route("/v1/budget", post(post_budget))
        .route("/v1/awareness/:id", get(get_awareness))
        .route("/v1/config", get(get_config).post(post_config))
        .layer(CorsLayer::permissive())
        // Codex CLI compresses HTTP request bodies (gzip/zstd); decompress
        // before the Json extractors so /v1/responses accepts them.
        .layer(RequestDecompressionLayer::new())
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
    let memory = crate::heal_memory::HealMemory::load(path);
    let advisories = memory.advisories_for_text(&text);
    // Cross-reactive (analogical) hints from sibling commands — advisory only.
    let hints = memory.cross_reactive_hints(&text);
    if advisories.is_empty() && hints.is_empty() {
        return;
    }
    let mut block = String::from(
        "<axiom_immunity>\nAxiom has prior self-healing experience with commands referenced here:\n",
    );
    for a in &advisories {
        block.push_str("- ");
        block.push_str(a);
        block.push('\n');
    }
    for h in &hints {
        block.push_str("- ");
        block.push_str(h);
        block.push('\n');
    }
    block.push_str("</axiom_immunity>");
    eprintln!(
        "[axiom-ttt] injected immunity advisory ({} command(s), {} cross-reactive hint(s))",
        advisories.len(),
        hints.len()
    );
    crate::anthropic_forwarder::prepend_block_to_last_user_turn(outbound, &block);
}

/// Mean next-token cross-entropy of `ids` through a clone of `states` (the
/// states are not mutated). Mirrors `eval_model`/`self_heal` CE scoring.
fn claim_ce(
    pipeline: &InferencePipeline,
    states: &[candle_core::Tensor],
    ids: &[u32],
) -> Option<f32> {
    if ids.len() < 2 {
        return None;
    }
    let dev = pipeline.device();
    let mut probe = states.to_vec();
    let input =
        candle_core::Tensor::from_vec(ids[..ids.len() - 1].to_vec(), (1, ids.len() - 1), dev)
            .ok()?;
    let logits = pipeline.model().forward_lm(&input, &mut probe).ok()?;
    let (_, t, v) = logits.dims3().ok()?;
    let l2d = logits.squeeze(0).ok()?.reshape((t, v)).ok()?;
    let tgt = candle_core::Tensor::from_vec(ids[1..].to_vec(), (ids.len() - 1,), dev).ok()?;
    candle_nn::loss::cross_entropy(&l2d, &tgt)
        .ok()?
        .to_scalar::<f32>()
        .ok()
}

/// Mean next-token cross-entropy of `text` under a fresh (unadapted) model —
/// "how surprising is this to the model." Bounded to a token budget so the
/// adaptive-compression gate stays cheap. Locks the pipeline briefly.
fn mean_surprisal(state: &AppState, text: &str) -> Option<f32> {
    let pipeline = state.pipeline.lock().ok()?;
    let ids: Vec<u32> = pipeline.encode_text(text).into_iter().take(256).collect();
    let states = pipeline.init_session_states().ok()?;
    claim_ce(&pipeline, &states, &ids)
}

/// Neural-surprisal grounding lifts per claim: `CE_base − CE_context`, where the
/// context is absorbed into W̃ via TTT once and reused. Positive ⇒ the absorbed
/// evidence predicts the claim (grounded). Returns a map claim→lift.
fn neural_lifts(
    pipeline: &InferencePipeline,
    evidence: &str,
    claims: &[String],
) -> std::collections::HashMap<String, f32> {
    let mut out = std::collections::HashMap::new();
    let base = match pipeline.init_session_states() {
        Ok(s) => s,
        Err(_) => return out,
    };
    let mut ctx = base.clone();
    let ev_ids = pipeline.encode_text(evidence);
    if adapt_session_blocking(pipeline, &mut ctx, &ev_ids).is_err() {
        return out;
    }
    for claim in claims {
        let ids = pipeline.encode_text(claim);
        if let (Some(ce_base), Some(ce_ctx)) = (
            claim_ce(pipeline, &base, &ids),
            claim_ce(pipeline, &ctx, &ids),
        ) {
            out.insert(claim.clone(), ce_base - ce_ctx);
        }
    }
    out
}

/// `POST /v1/verify` — grounding verification. Body:
/// `{"response": "...", "evidence": "..."}`. Returns per-claim verdicts
/// (SUPPORTED/UNSUPPORTED/UNVERIFIED) plus the grounded fraction and the
/// flagged (unsupported) claims — flags hallucinations *relative to the
/// supplied evidence*, not universal fact-checking.
async fn verify_grounding(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    // Conformal calibration mode: `{"calibrate": [[score, truly_supported],
    // ...], "delta": 0.1}` computes a coverage-guaranteed threshold τ from a
    // held-out sample. Export the result as AXIOM_CONFORMAL_THRESHOLD to gate
    // future verdicts at (1-δ) coverage of genuinely supported claims.
    // Checked before `response` validation — calibration needs no response.
    if let Some(cal) = body.get("calibrate").and_then(Value::as_array) {
        // Strict parsing: a malformed sample silently dropped would shift the
        // computed quantile — reject the request instead.
        let mut pairs: Vec<(f32, bool)> = Vec::with_capacity(cal.len());
        for (i, entry) in cal.iter().enumerate() {
            let parsed = entry.as_array().and_then(|pair| {
                let score = pair.first()?.as_f64()? as f32;
                let supported = pair.get(1)?.as_bool()?;
                (pair.len() == 2 && score.is_finite()).then_some((score, supported))
            });
            match parsed {
                Some(p) => pairs.push(p),
                None => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "error": format!(
                                "calibrate[{i}] is not a [finite_score, truly_supported_bool] pair"
                            ),
                        })),
                    )
                        .into_response();
                }
            }
        }
        if pairs.is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "calibrate must be a non-empty array of [score, truly_supported] pairs",
                })),
            )
                .into_response();
        }
        let delta = body
            .get("delta")
            .and_then(Value::as_f64)
            .map(|d| d as f32)
            .unwrap_or(0.10)
            .clamp(0.0, 1.0);
        let threshold = crate::hallucination::calibrate_conformal_threshold(&pairs, delta);
        return Json(serde_json::json!({
            "mode": "conformal_calibration",
            "threshold": threshold,
            "delta": delta,
            "samples": pairs.len(),
            "positives": pairs.iter().filter(|(_, y)| *y).count(),
            "note": "set AXIOM_CONFORMAL_THRESHOLD to this value (and optionally AXIOM_CONFORMAL_DELTA) to activate the calibrated gate for all future /v1/verify verdicts",
        }))
        .into_response();
    }

    let response = body.get("response").and_then(Value::as_str).unwrap_or("");
    let evidence = body.get("evidence").and_then(Value::as_str).unwrap_or("");
    if response.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "response is required"})),
        )
            .into_response();
    }

    let claim_json = |c: &crate::hallucination::ClaimVerdict| {
        serde_json::json!({
            "claim": c.claim,
            "verdict": c.verdict.to_string(),
            "support": c.support,
            "confidence": c.confidence.mean(),
            "uncertainty": c.confidence.variance().sqrt(),
            "grounding_lift": c.lift,
        })
    };

    // Grounding-gated expansion: when `expand` is set with a `session_id`, the
    // skeleton is the lean evidence and any claim it cannot ground triggers an
    // expansion of ONLY that claim's referenced symbols (from the stored
    // source), then a re-verify. Tokens are spent back surgically.
    let expand = body.get("expand").and_then(Value::as_bool).unwrap_or(false);
    let session_id = body.get("session_id").and_then(Value::as_str).unwrap_or("");
    if expand && !session_id.is_empty() {
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
        let report = crate::hallucination::verify_with_gated_expansion(response, evidence, |sym| {
            crate::skeleton::expand_symbol(&source, sym)
        });
        return Json(serde_json::json!({
            "mode": "grounding_gated_expansion",
            "grounded_fraction_before": report.before.grounded_fraction,
            "grounded_fraction_after": report.after.grounded_fraction,
            "expanded_symbols": report.expanded_symbols,
            "expansion_bytes": report.expansion_bytes,
            "resolved_claims": report.resolved_claims,
            "flagged": report.after.flagged().iter().map(|c| &c.claim).collect::<Vec<_>>(),
            "claims": report.after.claims.iter().map(&claim_json).collect::<Vec<_>>(),
            "note": "tokens were spent only on claims the skeleton could not ground; expansion pulled only those claims' referenced symbols",
        }))
        .into_response();
    }

    // Neural-surprisal tier: when `neural:true`, compute per-claim grounding
    // lift against the context-adapted W̃ and demote lexically-supported claims
    // the model finds surprising (the contradiction-catcher). Model-dependent:
    // sharp with the trained checkpoint, near-flat on the bootstrap model.
    let neural = body.get("neural").and_then(Value::as_bool).unwrap_or(false);
    if neural {
        let claims = crate::hallucination::extract_claims(response);
        let pipeline = state.pipeline.clone();
        let evidence_s = evidence.to_string();
        let lifts = tokio::task::spawn_blocking(move || {
            let p = pipeline.lock().ok()?;
            Some(neural_lifts(&p, &evidence_s, &claims))
        })
        .await
        .ok()
        .flatten()
        .unwrap_or_default();
        let report = crate::hallucination::verify_with_signals(response, evidence, |claim| {
            lifts.get(claim).copied()
        });
        return Json(serde_json::json!({
            "mode": "lexical+neural",
            "grounded_fraction": report.grounded_fraction,
            "supported": report.supported,
            "unsupported": report.unsupported,
            "unverified": report.unverified,
            "flagged": report.flagged().iter().map(|c| &c.claim).collect::<Vec<_>>(),
            "claims": report.claims.iter().map(&claim_json).collect::<Vec<_>>(),
            "note": "neural tier demotes lexically-supported claims with non-positive grounding lift (surprisal under the context-adapted W̃); signal sharpness scales with the trained model",
        }))
        .into_response();
    }

    let report = crate::hallucination::verify(response, evidence);
    Json(serde_json::json!({
        "grounded_fraction": report.grounded_fraction,
        "supported": report.supported,
        "unsupported": report.unsupported,
        "unverified": report.unverified,
        "flagged": report.flagged().iter().map(|c| &c.claim).collect::<Vec<_>>(),
        "claims": report.claims.iter().map(&claim_json).collect::<Vec<_>>(),
        "note": "grounding verification against supplied evidence; lexical tier does not catch vocabulary-sharing contradictions",
    }))
    .into_response()
}

/// `POST /v1/epistemic/validate` — combine grounding verification with an
/// explicitly configured semantic LLM judge and append local JSONL telemetry.
async fn validate_epistemic_drift(Json(body): Json<Value>) -> Response {
    let prompt = body.get("prompt").and_then(Value::as_str).unwrap_or("");
    let response = body.get("response").and_then(Value::as_str).unwrap_or("");
    let evidence = body.get("evidence").and_then(Value::as_str).unwrap_or("");
    if prompt.trim().is_empty() || response.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "prompt and response are required"})),
        )
            .into_response();
    }
    let config = match crate::epistemic_drift::EpistemicJudgeConfig::from_env() {
        Ok(Some(config)) => config,
        Ok(None) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "epistemic judge is not configured"})),
            )
                .into_response();
        }
        Err(error) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": error})),
            )
                .into_response();
        }
    };
    let judge = match crate::epistemic_drift::OpenAiSemanticJudge::new(config) {
        Ok(judge) => judge,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": error})),
            )
                .into_response();
        }
    };
    let request_id = body
        .get("request_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let target_model = body
        .get("target_model")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    match crate::epistemic_drift::evaluate_with_judge(
        &judge,
        request_id,
        prompt,
        response,
        evidence,
        target_model,
    )
    .await
    {
        Ok(evaluation) => Json(evaluation).into_response(),
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": error})),
        )
            .into_response(),
    }
}
