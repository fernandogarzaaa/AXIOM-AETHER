// Integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use candle_core::{DType, Device};
    use std::sync::Mutex;
    use tower::ServiceExt;

    static ENV_GUARD: Mutex<()> = Mutex::new(());

    #[test]
    fn savings_receipt_formats_bytes_and_ratio() {
        assert_eq!(
            savings_receipt(41_200, 17_300),
            "41.2k in, 17.3k forwarded, 58% saved"
        );
        assert_eq!(savings_receipt(0, 0), "0.0k in, 0.0k forwarded, 0% saved");
        // Never negative even if out > in (fingerprint overhead on tiny bodies).
        assert_eq!(
            savings_receipt(100, 250),
            "0.1k in, 0.2k forwarded, 0% saved"
        );
    }

    #[test]
    fn responses_compression_defaults_on_and_opts_out() {
        let _guard = ENV_GUARD.lock().expect("env guard poisoned");
        std::env::remove_var("AXIOM_RESPONSES_COMPRESS");
        assert!(responses_compression_enabled(), "on by default when unset");

        std::env::set_var("AXIOM_RESPONSES_COMPRESS", "0");
        assert!(!responses_compression_enabled(), "explicit 0 disables");
        std::env::set_var("AXIOM_RESPONSES_COMPRESS", "off");
        assert!(!responses_compression_enabled(), "explicit off disables");

        std::env::set_var("AXIOM_RESPONSES_COMPRESS", "1");
        assert!(responses_compression_enabled(), "explicit 1 enables");
        std::env::remove_var("AXIOM_RESPONSES_COMPRESS");
    }

    #[test]
    fn responses_compression_header_can_disable_one_request() {
        let mut headers = HeaderMap::new();
        assert!(responses_compression_header_enabled(&headers));

        headers.insert("x-axiom-responses-compress", "0".parse().unwrap());
        assert!(!responses_compression_header_enabled(&headers));

        headers.insert("x-axiom-responses-compress", "1".parse().unwrap());
        assert!(responses_compression_header_enabled(&headers));
    }

    #[test]
    fn relayable_response_headers_skips_hop_by_hop_and_managed() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer tok".parse().unwrap());
        headers.insert("host", "127.0.0.1:3000".parse().unwrap());
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("content-encoding", "gzip".parse().unwrap());
        headers.insert("accept-encoding", "gzip, br".parse().unwrap());
        headers.insert("x-axiom-session-id", "sess".parse().unwrap());
        headers.insert("x-axiom-responses-compress", "0".parse().unwrap());
        headers.insert("chatgpt-account-id", "acct-123".parse().unwrap());
        headers.insert("openai-beta", "responses=experimental".parse().unwrap());
        headers.insert("session_id", "abc".parse().unwrap());
        headers.insert("originator", "codex_cli_rs".parse().unwrap());
        headers.insert("accept", "text/event-stream".parse().unwrap());

        let relayed = relayable_response_headers(&headers);
        let names: Vec<&str> = relayed.iter().map(|(n, _)| n.as_str()).collect();

        for skipped in [
            "authorization",
            "host",
            "content-type",
            "content-encoding",
            "accept-encoding",
            "x-axiom-session-id",
            "x-axiom-responses-compress",
        ] {
            assert!(!names.contains(&skipped), "{skipped} must not be relayed");
        }
        for kept in [
            "chatgpt-account-id",
            "openai-beta",
            "session_id",
            "originator",
            "accept",
        ] {
            assert!(names.contains(&kept), "{kept} must be relayed");
        }
    }

    #[test]
    fn openai_assistant_answer_reads_non_streamed_completion() {
        let body = serde_json::json!({
            "choices": [{ "message": { "role": "assistant", "content": "hello world" } }]
        })
        .to_string();
        assert_eq!(openai_assistant_answer(&body), "hello world");
    }

    #[test]
    fn openai_assistant_answer_concatenates_sse_stream() {
        let body = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\
                    data: [DONE]\n";
        assert_eq!(openai_assistant_answer(body), "hello");
    }

    #[test]
    fn openai_assistant_answer_empty_when_unparseable() {
        assert_eq!(openai_assistant_answer("not json, not sse"), "");
    }

    #[test]
    fn grounding_advisory_only_when_claims_flagged() {
        let evidence = "Axiom is an inference engine with online test-time training.";
        // Fully grounded → no advisory.
        assert!(
            grounding_advisory_block("Axiom uses online test-time training.", evidence).is_none()
        );
        // Ungrounded claim → advisory naming it.
        let block = grounding_advisory_block(
            "Axiom was written in COBOL and launched on the moon.",
            evidence,
        )
        .expect("ungrounded claim must produce an advisory");
        assert!(block.contains("<axiom_grounding"));
        assert!(block.contains("COBOL") || block.contains("moon"));
    }

    async fn start_mock_upstream(
        reply_text: &'static str,
    ) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let app = axum::Router::new().route(
            "/v1/messages",
            axum::routing::post(move || {
                let c = c.clone();
                async move {
                    c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    axum::Json(serde_json::json!({
                        "content": [{"type": "text", "text": reply_text}]
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        (format!("http://{addr}"), calls)
    }

    #[tokio::test]
    async fn self_correction_reasks_and_returns_grounded_revision() {
        use std::sync::atomic::Ordering;
        // Mock upstream returns a grounded revision on the corrective re-ask.
        let (base, calls) = start_mock_upstream("Axiom uses online test-time training.").await;
        let forwarder =
            crate::anthropic_forwarder::AnthropicForwarder::new(Some("k".into()), Some(base));
        let outbound = serde_json::json!({"model":"m","max_tokens":64,
            "messages":[{"role":"user","content":"summarize the doc"}]});
        let ungrounded = serde_json::json!({"content":[{"type":"text",
            "text":"Axiom was funded by NASA in 1972."}]});
        let evidence = "Axiom is an inference engine with online test-time training.";
        let revised = ground_correct_round(
            &forwarder,
            &ClientAuth::default(),
            &outbound,
            &ungrounded,
            evidence,
        )
        .await;
        assert!(
            revised.is_some(),
            "flagged claim must trigger a corrective re-ask"
        );
        assert!(assistant_text(&revised.unwrap()).contains("test-time training"));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "exactly one corrective re-ask"
        );
    }

    #[tokio::test]
    async fn self_correction_is_noop_when_answer_is_grounded() {
        use std::sync::atomic::Ordering;
        let (base, calls) = start_mock_upstream("unused").await;
        let forwarder =
            crate::anthropic_forwarder::AnthropicForwarder::new(Some("k".into()), Some(base));
        let outbound = serde_json::json!({"model":"m","max_tokens":64,
            "messages":[{"role":"user","content":"summarize"}]});
        let grounded = serde_json::json!({"content":[{"type":"text",
            "text":"Axiom uses online test-time training."}]});
        let evidence = "Axiom is an inference engine with online test-time training.";
        let revised = ground_correct_round(
            &forwarder,
            &ClientAuth::default(),
            &outbound,
            &grounded,
            evidence,
        )
        .await;
        assert!(revised.is_none(), "grounded answer → no re-ask");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "no upstream call when nothing is flagged"
        );
    }

    #[tokio::test]
    async fn self_correction_rejects_revision_that_is_not_more_grounded() {
        use std::sync::atomic::Ordering;
        // The re-ask returns an answer that is still ungrounded (shares nothing
        // with the evidence), so the guardrail must reject it and keep None.
        let (base, calls) = start_mock_upstream("Bananas ripen faster in zero gravity.").await;
        let forwarder =
            crate::anthropic_forwarder::AnthropicForwarder::new(Some("k".into()), Some(base));
        let outbound = serde_json::json!({"model":"m","max_tokens":64,
            "messages":[{"role":"user","content":"summarize the doc"}]});
        let ungrounded = serde_json::json!({"content":[{"type":"text",
            "text":"Axiom was funded by NASA in 1972."}]});
        let evidence = "Axiom is an inference engine with online test-time training.";
        let revised = ground_correct_round(
            &forwarder,
            &ClientAuth::default(),
            &outbound,
            &ungrounded,
            evidence,
        )
        .await;
        assert!(
            revised.is_none(),
            "a revision that is not better grounded must be rejected"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the re-ask still happened once"
        );
    }

    #[test]
    fn annotate_response_grounding_respects_env_flag() {
        let _guard = ENV_GUARD.lock().expect("env guard poisoned");
        // Single test (sequential) to avoid racing on the process-global env var.
        let evidence = "Axiom uses online test-time training.";
        let make = || {
            serde_json::json!({
                "content": [{"type": "text", "text": "Axiom was funded by NASA in 1972."}]
            })
        };

        // Disabled → response untouched.
        std::env::remove_var("AXIOM_VERIFY_RESPONSES");
        let mut off = make();
        let before = off.clone();
        annotate_response_grounding(&mut off, evidence);
        assert_eq!(off, before, "disabled → response untouched");

        // Enabled → advisory appended in place, original answer preserved.
        std::env::set_var("AXIOM_VERIFY_RESPONSES", "1");
        let mut on = make();
        annotate_response_grounding(&mut on, evidence);
        std::env::remove_var("AXIOM_VERIFY_RESPONSES");
        let text = on["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("<axiom_grounding"),
            "advisory must be appended"
        );
        assert!(
            text.starts_with("Axiom was funded by NASA"),
            "original answer preserved"
        );
    }

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
    async fn health_and_readiness_probes_report_up() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state.clone());

        let health = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);

        let ready = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ready.status(), StatusCode::OK);
        let body = axum::body::to_bytes(ready.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ready");
        assert_eq!(json["model"], "axiom-ttt-test");

        safe_drop(pipeline_arc).await;
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
    async fn mcp_http_route_returns_503_when_disabled() {
        // Without AXIOM_MCP_HTTP the context is None, so /mcp must refuse cleanly
        // (not 404, not 500) so clients get an actionable error.
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["error"].as_str().unwrap().contains("AXIOM_MCP_HTTP"));
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn router_mode_routes_generation_through_the_router() {
        use crate::backend_router::{
            BackendError, ChatBackend, Provider, RoutePolicy, Router as BR,
        };
        struct Mock;
        impl ChatBackend for Mock {
            fn complete(&self, _p: &str, _max_tokens: usize) -> Result<String, BackendError> {
                Ok("routed-answer".into())
            }
        }
        // General task routes to OpenAi by default; register the mock there.
        let router = BR::new(RoutePolicy::default()).with(Provider::OpenAi, Box::new(Mock));
        let state = make_test_state().await.with_router(Some(router));
        let pipeline_arc = state.pipeline.clone();
        // run_generation must short-circuit to the router, not the local pipeline.
        let out = run_generation(&state, "hello", 8, None).unwrap();
        assert_eq!(out, "routed-answer");
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_chimera_run_endpoint_executes_a_program() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state.clone());

        let body = serde_json::json!({
            "source": "val xs = [1, 2, 3]\nfor x in xs\n  emit x\nend"
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chimera/run")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["emitted"], serde_json::json!(["1", "2", "3"]));
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn mcp_route_rejects_missing_token_when_configured() {
        // With a token configured, an unauthenticated /mcp request is 401 — and
        // the auth check runs *before* the 503 disabled check, so a missing token
        // is rejected even when the MCP context itself is disabled.
        let state = make_test_state()
            .await
            .with_mcp_token(Some("s3cret".into()));
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state.clone());

        // No Authorization header → 401.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Wrong token → 401.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer wrong")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Correct token passes the auth gate; MCP is disabled here so it falls
        // through to 503 (proving auth succeeded, not 401).
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer s3cret")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn data_plane_open_by_default_when_no_api_key() {
        // Default (no AXIOM_API_KEY): a guarded route is reachable with no
        // X-Axiom-Key header — the local-first behavior is unchanged.
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn data_plane_requires_api_key_when_configured() {
        let state = make_test_state().await.with_api_key(Some("k3y".into()));
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);

        // Missing header → 401.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Wrong key → 401.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/config")
                    .header("x-axiom-key", "wrong")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Correct key passes the gate (any non-401 status proves auth succeeded).
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/config")
                    .header("x-axiom-key", "k3y")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn ops_endpoints_exempt_from_api_key() {
        // /healthz and /metrics must stay reachable for probes/scrapers even
        // when the data plane is locked down.
        let state = make_test_state().await.with_api_key(Some("k3y".into()));
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        for uri in ["/healthz", "/metrics"] {
            let resp = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::GET)
                        .uri(uri)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{uri} should be exempt");
        }
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_patches_export_and_merge_roundtrip() {
        let dir = std::env::temp_dir().join(format!("axiom_srv_patches_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let heal_path = dir.join("axiom_heal_memory.json");
        let patch_path = dir.join("axiom_patch_memory.json");

        // Seed this node's verified-patch store with one candidate.
        {
            let mut pm = crate::patch_memory::PatchMemory::new();
            pm.record_verified("fp1", "src/a.rs", "FIXED");
            pm.save(&patch_path).unwrap();
        }

        let state = make_test_state()
            .await
            .with_heal_memory_path(Some(heal_path));
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state.clone());

        // GET /v1/patches → provenance-signed export of the store.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/patches")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let export = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();

        // POST it back to /v1/patches/merge → the identical candidate reinforces,
        // exercising provenance verification + content-hash recompute over HTTP.
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/patches/merge")
                    .header("content-type", "application/json")
                    .body(Body::from(export.to_vec()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["merged"], true);
        assert_eq!(v["reinforced"], 1);
        assert_eq!(v["new_candidates"], 0);

        // The store now records the candidate as verified by two nodes.
        let pm = crate::patch_memory::PatchMemory::load(&patch_path);
        assert_eq!(pm.candidates("fp1")[0].verified_count, 2);

        let _ = std::fs::remove_dir_all(&dir);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn test_patches_merge_accepts_bodies_over_2mb() {
        // A patch candidate carries full file contents; a real fix can push the
        // signed export past axum's 2 MB default body limit. The /v1/patches/merge
        // route raises that limit, so a large-but-valid export must still merge.
        let dir = std::env::temp_dir().join(format!("axiom_srv_bigpatch_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let heal_path = dir.join("axiom_heal_memory.json");
        let patch_path = dir.join("axiom_patch_memory.json");
        {
            let big = "X".repeat(3 * 1024 * 1024); // ~3 MB > the 2 MB default
            let mut pm = crate::patch_memory::PatchMemory::new();
            pm.record_verified("fp-big", "src/big.rs", &big);
            pm.save(&patch_path).unwrap();
        }

        let state = make_test_state()
            .await
            .with_heal_memory_path(Some(heal_path));
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state.clone());

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/patches")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let export = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(
            export.len() > 2 * 1024 * 1024,
            "export should exceed the default limit"
        );

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/patches/merge")
                    .header("content-type", "application/json")
                    .body(Body::from(export.to_vec()))
                    .unwrap(),
            )
            .await
            .unwrap();
        // Without the raised limit this would be 413 PAYLOAD_TOO_LARGE.
        assert_eq!(resp.status(), StatusCode::OK);

        let _ = std::fs::remove_dir_all(&dir);
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
