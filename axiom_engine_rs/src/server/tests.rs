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
    async fn safe_drop(arc: std::sync::Arc<std::sync::RwLock<crate::inference::InferencePipeline>>) {
        tokio::task::spawn_blocking(move || drop(arc))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn local_openai_stream_response_emits_deltas_then_done() {
        use axum::body::to_bytes;
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let stream = crate::swarm_router::SwarmStream {
            model: "phi4:3.8b".to_string(),
            receiver: rx,
        };
        tx.send(Ok("Hel".to_string())).await.unwrap();
        tx.send(Ok("lo world".to_string())).await.unwrap();
        drop(tx);
        let state = make_test_state().await;
        let resp = local_openai_chat_stream_response(&state, "sess-stream-test", stream)
            .expect("stream response must build");
        assert_eq!(resp.status(), StatusCode::OK);
        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            content_type.contains("text/event-stream"),
            "stream response must be SSE, got {content_type}"
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        let frames: Vec<&str> = text.split("\n\n").filter(|f| !f.is_empty()).collect();
        assert_eq!(frames.len(), 4, "two deltas, stop chunk, [DONE]:\n{text}");
        assert!(
            frames[0].contains(r#""content":"Hel""#),
            "first delta frame: {}",
            frames[0]
        );
        assert!(
            frames[1].contains(r#""content":"lo world""#),
            "second delta frame: {}",
            frames[1]
        );
        assert!(
            frames[2].contains(r#""finish_reason":"stop""#),
            "stop frame: {}",
            frames[2]
        );
        assert_eq!(frames[3], "data: [DONE]");
        safe_drop(state.pipeline.clone()).await;
    }

    #[tokio::test]
    async fn local_openai_stream_response_ends_without_done_on_upstream_error() {
        use axum::body::to_bytes;
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let stream = crate::swarm_router::SwarmStream {
            model: "phi4:3.8b".to_string(),
            receiver: rx,
        };
        tx.send(Ok("Hel".to_string())).await.unwrap();
        tx.send(Err(crate::swarm_router::SwarmRouteError::Upstream(
            "ollama died".to_string(),
        )))
        .await
        .unwrap();
        let state = make_test_state().await;
        let resp = local_openai_chat_stream_response(&state, "sess-stream-err", stream)
            .expect("stream response must build");
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            text.contains(r#""content":"Hel""#),
            "delta sent before the error must arrive:\n{text}"
        );
        assert!(
            !text.contains("[DONE]"),
            "an errored stream must not claim a clean stop:\n{text}"
        );
        safe_drop(state.pipeline.clone()).await;
    }

    #[tokio::test]
    async fn local_openai_chat_response_returns_single_json_object() {
        use axum::body::to_bytes;
        let local = SwarmChatResult {
            model: "phi4:3.8b".to_string(),
            content: "Hello world".to_string(),
        };
        let resp = local_openai_chat_response(local).expect("response must build");
        assert_eq!(resp.status(), StatusCode::OK);
        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            content_type.contains("application/json"),
            "non-stream response must be JSON, got {content_type}"
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).expect("body must be one JSON object");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["model"], "phi4:3.8b");
    }

    /// `AppState.pipeline` must be an `RwLock`, not a `Mutex`: the whole point
    /// of switching (see the field doc) is that concurrent readers — e.g. the
    /// `responses_run_fingerprint` JoinSet fan-out — can hold the lock
    /// simultaneously instead of serializing on a single writer-only mutex. A
    /// `Mutex` would make the second `try_read()` below fail with
    /// `WouldBlock` while the first guard is alive; this regresses if the
    /// field is ever changed back.
    #[tokio::test]
    async fn pipeline_lock_allows_concurrent_readers() {
        let state = make_test_state().await;
        {
            let first = state.pipeline.read().unwrap();
            let second = state
                .pipeline
                .try_read()
                .expect("a second concurrent reader must not block behind the first");
            assert_eq!(first.token_count("same pipeline"), second.token_count("same pipeline"));
        }
        safe_drop(state.pipeline.clone()).await;
    }

    /// Cancellation mid-generation must not leak transient sub-sessions.
    /// Audit section 6 / issue #93: dropping `compressed_responses_payload`
    /// while fingerprints are generating must reach the same cleanup the
    /// error branches use, and no detached fingerprint task may recreate a
    /// sub-session after that cleanup ran. The session ids are minted
    /// (`responses-<uuid>#r<ordinal>`) so the test identifies them by the
    /// `#r` suffix in this state's own store.
    //
    // The env guard is held across the short poll loop below on purpose: it
    // serializes access to the process-global AXIOM_RESPONSES_COMPRESS var
    // while the payload reads it. The payload task never takes the guard and
    // the only other holders are short synchronous tests, so no deadlock is
    // possible; the guard is dropped before the long settle awaits.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn cancelled_responses_compression_leaves_no_transient_subsessions() {
        let _env_guard = ENV_GUARD.lock().expect("env guard poisoned");
        std::env::remove_var("AXIOM_RESPONSES_COMPRESS");
        let state = make_test_state().await;
        state.controls.set_enabled(true);
        state.controls.set_threshold(1);

        // Eight runs (assistant messages separated by user messages, then a
        // final user query) with long contexts, so generation is still in
        // flight when the test cancels it. Concurrency is capped at 4, so
        // later runs are still queued behind the first four.
        let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(400);
        let mut input = Vec::new();
        for ordinal in 0..8usize {
            input.push(serde_json::json!({
                "type": "message",
                "role": "assistant",
                "content": format!("Run {ordinal} context. {long_text}"),
            }));
            input.push(serde_json::json!({
                "type": "message",
                "role": "user",
                "content": format!("interlude {ordinal}"),
            }));
        }
        input.push(serde_json::json!({
            "type": "message",
            "role": "user",
            "content": "latest query",
        }));
        let body = serde_json::json!({ "input": input });

        fn transient_subsessions(state: &AppState) -> Vec<String> {
            state
                .ttt_sessions
                .snapshot_handles()
                .into_iter()
                .map(|(id, _)| id)
                .filter(|id| id.contains("#r"))
                .collect()
        }

        // Drive the payload and cancel it mid-generation. Bounded retry on
        // the observation step only: under heavy parallel load this test's
        // poll loop can be starved long enough for the whole generation to
        // finish before it wakes, which proves nothing either way. A retry
        // re-drives from scratch; the leak assertions below never retry, so
        // a real leak still fails deterministically.
        let mut attempt = 0;
        loop {
            attempt += 1;
            let run_state = state.clone();
            let run_body = body.clone();
            let handle = tokio::spawn(async move {
                compressed_responses_payload(&run_state, &run_body, None, true).await
            });

            // Wait until generation is observably in flight: at least one
            // fingerprint task has registered its transient sub-session.
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(60);
            let mut seen_in_flight = false;
            while tokio::time::Instant::now() < deadline {
                if !transient_subsessions(&state).is_empty() {
                    seen_in_flight = true;
                    break;
                }
                if handle.is_finished() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
            if !seen_in_flight {
                // Generation finished before this attempt could observe it.
                // Join the completed run (its normal path cleans up) and
                // retry the observation; never assert the leak from here.
                let _ = handle.await;
                let leftover = transient_subsessions(&state);
                assert!(
                    leftover.is_empty(),
                    "completed run left transient sub-sessions behind: {leftover:?}"
                );
                assert!(
                    attempt < 3,
                    "generation finished before cancellation could land mid-flight \
                     on 3 attempts; lengthen the run contexts so fingerprints are \
                     still generating"
                );
                continue;
            }
            // The payload reads AXIOM_RESPONSES_COMPRESS once at its start, which
            // has passed by now (fingerprint tasks are running). Release the env
            // guard before any further await: holding a std MutexGuard across an
            // await trips clippy::await_holding_lock.
            drop(_env_guard);
            let in_flight = transient_subsessions(&state);
            assert!(
                !in_flight.is_empty(),
                "expected at least one transient sub-session at cancel time"
            );

            // Cancel mid-generation: abort the task (drops the payload future).
            handle.abort();
            let aborted = handle.await;
            assert!(aborted.is_err(), "aborted task must report cancellation");
            assert!(
                aborted.unwrap_err().is_cancelled(),
                "aborted task must report a cancellation, not a panic"
            );
            break;
        }

        // Settle: detached blocking fingerprint work runs to completion
        // even after the abort, so wait until no transient sub-session
        // remains, then keep watching through a quiet period — a session
        // recreated after cleanup must fail this test, not slip past it.
        let settle_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        while !transient_subsessions(&state).is_empty()
            && tokio::time::Instant::now() < settle_deadline
        {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let remaining = transient_subsessions(&state);
        assert!(
            remaining.is_empty(),
            "cancelled compression leaked transient sub-sessions: {remaining:?}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let recreated = transient_subsessions(&state);
        assert!(
            recreated.is_empty(),
            "fingerprint tasks recreated transient sub-sessions after cleanup: {recreated:?}"
        );

        safe_drop(state.pipeline.clone()).await;
    }

    fn seed_active_test_session(state: &AppState, session_id: &str) {
        let now = unix_now();
        let states = state
            .pipeline
            .read()
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

    /// Helper: GET /readyz against a fresh router built from `state`.
    async fn readyz_response(state: AppState) -> (StatusCode, serde_json::Value) {
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        safe_drop(pipeline_arc).await;
        (status, json)
    }

    #[tokio::test]
    async fn readyz_503_in_production_mode_with_no_checkpoint() {
        let state = make_test_state()
            .await
            .with_checkpoint_gate(true, CheckpointStatus::Absent);
        let (status, json) = readyz_response(state).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(json["status"], "unavailable");
        assert!(
            json["reason"].as_str().unwrap().contains("absent"),
            "reason must name the absent weights: {json}"
        );
    }

    #[tokio::test]
    async fn readyz_503_in_production_mode_with_unverified_checkpoint() {
        let state = make_test_state()
            .await
            .with_checkpoint_gate(true, CheckpointStatus::Unverified);
        let (status, json) = readyz_response(state).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(json["status"], "unavailable");
        assert!(
            json["reason"].as_str().unwrap().contains("unverified"),
            "reason must name the unverified weights: {json}"
        );
    }

    #[tokio::test]
    async fn readyz_503_in_production_mode_with_mismatched_checkpoint() {
        let state = make_test_state()
            .await
            .with_checkpoint_gate(true, CheckpointStatus::Mismatched);
        let (status, json) = readyz_response(state).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(json["status"], "unavailable");
        assert!(
            json["reason"].as_str().unwrap().contains("mismatch"),
            "reason must name the mismatch: {json}"
        );
    }

    #[tokio::test]
    async fn readyz_200_in_production_mode_with_verified_checkpoint() {
        let state = make_test_state()
            .await
            .with_checkpoint_gate(true, CheckpointStatus::Verified);
        let (status, json) = readyz_response(state).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["status"], "ready");
        assert_eq!(json["checkpoint"], "verified");
    }

    #[tokio::test]
    async fn readyz_200_in_non_production_mode_with_legacy_bootstrap() {
        // The gate is off and no verified checkpoint exists (the legacy
        // bootstrap model): readiness is unchanged from before this change.
        let state = make_test_state()
            .await
            .with_checkpoint_gate(false, CheckpointStatus::Absent);
        let (status, json) = readyz_response(state).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["status"], "ready");
        let state = make_test_state()
            .await
            .with_checkpoint_gate(false, CheckpointStatus::Unverified);
        let (status, json) = readyz_response(state).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["status"], "ready");
    }

    #[test]
    fn resolve_checkpoint_status_absent_when_file_missing() {
        let missing = std::path::Path::new("/definitely/not/a/checkpoint.bin");
        assert_eq!(
            resolve_checkpoint_status(missing, None, None, None),
            CheckpointStatus::Absent
        );
    }

    #[test]
    fn resolve_checkpoint_status_unverified_without_pinned_hash() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("ckpt.bin");
        std::fs::write(&ckpt, b"weights").unwrap();
        assert_eq!(
            resolve_checkpoint_status(&ckpt, None, None, None),
            CheckpointStatus::Unverified
        );
        assert_eq!(
            resolve_checkpoint_status(&ckpt, None, Some(""), None),
            CheckpointStatus::Unverified,
            "an empty pinned hash is the same as no pinned hash"
        );
    }

    #[test]
    fn resolve_checkpoint_status_verified_on_matching_pinned_hash() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("ckpt.bin");
        std::fs::write(&ckpt, b"weights").unwrap();
        let digest = format!("{:x}", Sha256::digest(b"weights"));
        assert_eq!(
            resolve_checkpoint_status(&ckpt, None, Some(&digest), None),
            CheckpointStatus::Verified
        );
    }

    #[test]
    fn resolve_checkpoint_status_mismatched_on_wrong_pinned_hash() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("ckpt.bin");
        std::fs::write(&ckpt, b"weights").unwrap();
        assert_eq!(
            resolve_checkpoint_status(&ckpt, None, Some("deadbeef"), None),
            CheckpointStatus::Mismatched
        );
    }

    #[test]
    fn resolve_checkpoint_status_mismatched_when_tokenizer_missing() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("ckpt.bin");
        std::fs::write(&ckpt, b"weights").unwrap();
        let digest = format!("{:x}", Sha256::digest(b"weights"));
        let missing_tok = dir.path().join("no-tokenizer.json");
        assert_eq!(
            resolve_checkpoint_status(&ckpt, Some(&missing_tok), Some(&digest), None),
            CheckpointStatus::Mismatched,
            "a verified checkpoint without its tokenizer is not servable"
        );
    }

    #[test]
    fn resolve_checkpoint_status_verified_when_tokenizer_hash_matches() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("ckpt.bin");
        std::fs::write(&ckpt, b"weights").unwrap();
        let tok = dir.path().join("tokenizer.json");
        std::fs::write(&tok, b"tokenizer-bytes").unwrap();
        let ckpt_digest = format!("{:x}", Sha256::digest(b"weights"));
        let tok_digest = format!("{:x}", Sha256::digest(b"tokenizer-bytes"));
        assert_eq!(
            resolve_checkpoint_status(&ckpt, Some(&tok), Some(&ckpt_digest), Some(&tok_digest)),
            CheckpointStatus::Verified
        );
    }

    #[test]
    fn resolve_checkpoint_status_mismatched_when_tokenizer_hash_mismatches() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("ckpt.bin");
        std::fs::write(&ckpt, b"weights").unwrap();
        let tok = dir.path().join("tokenizer.json");
        std::fs::write(&tok, b"substituted-tokenizer").unwrap();
        let ckpt_digest = format!("{:x}", Sha256::digest(b"weights"));
        let pinned_tok_digest = format!("{:x}", Sha256::digest(b"tokenizer-bytes"));
        assert_eq!(
            resolve_checkpoint_status(
                &ckpt,
                Some(&tok),
                Some(&ckpt_digest),
                Some(&pinned_tok_digest)
            ),
            CheckpointStatus::Mismatched,
            "a substituted tokenizer must not pass on a valid checkpoint hash alone"
        );
    }

    #[test]
    fn resolve_checkpoint_status_presence_only_without_pinned_tokenizer_hash() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("ckpt.bin");
        std::fs::write(&ckpt, b"weights").unwrap();
        let tok = dir.path().join("tokenizer.json");
        std::fs::write(&tok, b"any-tokenizer").unwrap();
        let ckpt_digest = format!("{:x}", Sha256::digest(b"weights"));
        assert_eq!(
            resolve_checkpoint_status(&ckpt, Some(&tok), Some(&ckpt_digest), None),
            CheckpointStatus::Verified,
            "no pinned tokenizer hash keeps presence-only behavior"
        );
        assert_eq!(
            resolve_checkpoint_status(&ckpt, Some(&tok), Some(&ckpt_digest), Some("")),
            CheckpointStatus::Verified,
            "an empty pinned tokenizer hash is the same as no pinned hash"
        );
    }

    #[test]
    fn resolve_checkpoint_status_mismatched_when_tokenizer_hash_pinned_but_no_tokenizer_path() {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("ckpt.bin");
        std::fs::write(&ckpt, b"weights").unwrap();
        let ckpt_digest = format!("{:x}", Sha256::digest(b"weights"));
        let tok_digest = format!("{:x}", Sha256::digest(b"tokenizer-bytes"));
        assert_eq!(
            resolve_checkpoint_status(&ckpt, None, Some(&ckpt_digest), Some(&tok_digest)),
            CheckpointStatus::Mismatched,
            "a pinned tokenizer hash with no tokenizer file to check proves nothing"
        );
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
            // Take the async session lock first so the std pipeline read guard
            // is never held across an `.await`.
            let handle = {
                let pipeline = state.pipeline.read().unwrap();
                state
                    .ttt_sessions
                    .get_or_create("persist-s1", &pipeline)
                    .unwrap()
            };
            let mut states = handle.lock().await;
            let pipeline = state.pipeline.read().unwrap();
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

    #[cfg(feature = "experimental")]
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
    async fn metrics_requires_api_key_when_configured() {
        // /metrics is gated by require_api_key: without the key it 401s, with
        // the key it serves the exposition body.
        let state = make_test_state().await.with_api_key(Some("k3y".into()));
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);

        // Missing header → 401.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/metrics")
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
                    .uri("/metrics")
                    .header("x-axiom-key", "wrong")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Correct key → 200 with the exposition body.
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/metrics")
                    .header("x-axiom-key", "k3y")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("axiom_total_tokens_prefilled"));
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn metrics_open_by_default_when_no_api_key() {
        // With AXIOM_API_KEY unset the guard is a passthrough, so /metrics
        // stays open with no header, as before.
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn ops_endpoints_exempt_from_api_key() {
        // /healthz and /readyz must stay reachable for probes even when the
        // data plane is locked down. /metrics is NOT exempt: it is gated by
        // require_api_key (see metrics_requires_api_key_when_configured).
        let state = make_test_state().await.with_api_key(Some("k3y".into()));
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        for uri in ["/healthz", "/readyz"] {
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
            let pipeline = state.pipeline.read().unwrap();
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

    // -----------------------------------------------------------------------
    // Hardening regressions (Oct 2026 audit): request-reachable panics,
    // unbounded allocation, filesystem confinement, /mcp auth parity.
    // -----------------------------------------------------------------------

    #[test]
    fn constant_time_eq_matches_only_equal_strings() {
        assert!(constant_time_eq("k3y", "k3y"));
        assert!(constant_time_eq("", ""));
        assert!(!constant_time_eq("k3y", "k3Y"));
        assert!(!constant_time_eq("k3y", "k3y "));
        assert!(!constant_time_eq("k3y", ""));
        assert!(!constant_time_eq("", "k3y"));
        assert!(!constant_time_eq("abc", "abcd"));
    }

    #[test]
    fn checked_shape_product_detects_overflow() {
        assert_eq!(checked_shape_product(&[2, 3, 4]), Some(24));
        assert_eq!(checked_shape_product(&[]), Some(1));
        assert_eq!(checked_shape_product(&[0, 7]), Some(0));
        assert_eq!(checked_shape_product(&[1 << 40, 1 << 40]), None);
        assert_eq!(checked_shape_product(&[usize::MAX, 2]), None);
    }

    // The hardening tests below never mutate the process environment: tests
    // run in parallel threads, and an `AXIOM_*` value set here would leak into
    // unrelated tests running at the same time.

    #[test]
    fn clamp_max_tokens_respects_default_and_configured_ceiling() {
        assert_eq!(clamp_max_tokens_with(8, None), 8);
        assert_eq!(clamp_max_tokens_with(usize::MAX, None), MAX_GENERATION_TOKENS);
        assert_eq!(
            clamp_max_tokens_with(1_000_000_000_000, None),
            MAX_GENERATION_TOKENS
        );
        assert_eq!(clamp_max_tokens_with(1000, Some("64")), 64);
        assert_eq!(clamp_max_tokens_with(10, Some("64")), 10);
        // Nonsense values fall back to the default rather than disabling the cap.
        assert_eq!(
            clamp_max_tokens_with(usize::MAX, Some("0")),
            MAX_GENERATION_TOKENS
        );
        assert_eq!(
            clamp_max_tokens_with(usize::MAX, Some("lots")),
            MAX_GENERATION_TOKENS
        );
    }

    #[test]
    fn parse_positive_usize_falls_back_on_bad_input() {
        assert_eq!(parse_positive_usize(None, 7), 7);
        assert_eq!(parse_positive_usize(Some(""), 7), 7);
        assert_eq!(parse_positive_usize(Some("0"), 7), 7);
        assert_eq!(parse_positive_usize(Some("-3"), 7), 7);
        assert_eq!(parse_positive_usize(Some(" 12 "), 7), 12);
    }

    #[test]
    fn confine_merge_path_keeps_inputs_and_outputs_inside_roots() {
        let root = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let inside = root.path().join("cache.bin");
        std::fs::write(&inside, b"x").unwrap();
        let outside = other.path().join("cache.bin");
        std::fs::write(&outside, b"x").unwrap();
        let roots = vec![root.path().canonicalize().unwrap()];

        // Existing input under the root: accepted, canonicalized.
        let ok = confine_merge_path(&inside, &roots, true).unwrap();
        assert!(ok.starts_with(&roots[0]));
        // Existing input outside every root: refused, and the message names
        // the override so an operator knows how to allow it deliberately.
        let err = confine_merge_path(&outside, &roots, true).unwrap_err();
        assert!(err.contains("outside the merge roots"), "{err}");
        assert!(err.contains("AXIOM_MERGE_ROOT"), "{err}");
        // Missing input: refused (not silently accepted as a future file).
        assert!(confine_merge_path(&root.path().join("nope.bin"), &roots, true).is_err());

        // Output that does not exist yet, under the root, nested in a
        // directory that does not exist yet either: accepted.
        let out = confine_merge_path(&root.path().join("new/dir/merged.bin"), &roots, false).unwrap();
        assert!(out.starts_with(&roots[0]), "{}", out.display());
        assert!(out.ends_with("new/dir/merged.bin"), "{}", out.display());
        // Output outside the root: refused.
        assert!(confine_merge_path(&other.path().join("merged.bin"), &roots, false).is_err());
        // `..` is refused outright even when it would resolve back inside.
        let dotdot = root.path().join("sub/../cache.bin");
        let err = confine_merge_path(&dotdot, &roots, true).unwrap_err();
        assert!(err.contains(".."), "{err}");
        let dotdot_out = root.path().join("../escape.bin");
        assert!(confine_merge_path(&dotdot_out, &roots, false).is_err());
        // Empty path is an error, not the current directory.
        assert!(confine_merge_path(FsPath::new(""), &roots, false).is_err());
    }

    #[tokio::test]
    async fn cluster_merge_refuses_paths_outside_merge_roots() {
        // Default roots are the test's cwd (the crate dir) and ~/.axiom; a
        // fresh temp dir is under neither, so both paths must be refused.
        let root = tempfile::tempdir().unwrap();
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let outside_in = tempfile::NamedTempFile::new().unwrap();
        let body = serde_json::json!({
            "inputs": [outside_in.path()],
            "output": root.path().join("merged.bin"),
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/cluster/merge")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("outside the merge roots"), "{text}");
        assert!(!root.path().join("merged.bin").exists());
        safe_drop(pipeline_arc).await;
    }

    async fn mcp_status(app: Router, headers: &[(&str, &str)]) -> StatusCode {
        let mut req = Request::builder()
            .method(Method::POST)
            .uri("/mcp")
            .header("content-type", "application/json");
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#;
        app.oneshot(req.body(Body::from(body)).unwrap())
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn mcp_requires_data_plane_key_when_no_mcp_token_is_set() {
        // /mcp is outside the require_api_key layer (so AXIOM_MCP_TOKEN can
        // guard it alone), but with only AXIOM_API_KEY configured it must not
        // be more open than the data plane it reaches.
        let state = make_test_state().await.with_api_key(Some("k3y".into()));
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        assert_eq!(mcp_status(app.clone(), &[]).await, StatusCode::UNAUTHORIZED);
        assert_eq!(
            mcp_status(app.clone(), &[("x-axiom-key", "wrong")]).await,
            StatusCode::UNAUTHORIZED
        );
        // The data-plane key is accepted in either header form; auth passing
        // is proven by any non-401 (the MCP transport itself is off → 503).
        assert_ne!(
            mcp_status(app.clone(), &[("x-axiom-key", "k3y")]).await,
            StatusCode::UNAUTHORIZED
        );
        assert_ne!(
            mcp_status(app, &[("authorization", "Bearer k3y")]).await,
            StatusCode::UNAUTHORIZED
        );
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn mcp_token_takes_precedence_over_data_plane_key() {
        let state = make_test_state()
            .await
            .with_api_key(Some("k3y".into()))
            .with_mcp_token(Some("mcp-t0k".into()));
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        // The data-plane key alone is not enough once a dedicated MCP token exists.
        assert_eq!(
            mcp_status(app.clone(), &[("x-axiom-key", "k3y")]).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            mcp_status(app.clone(), &[("authorization", "Bearer k3y")]).await,
            StatusCode::UNAUTHORIZED
        );
        assert_ne!(
            mcp_status(app, &[("authorization", "Bearer mcp-t0k")]).await,
            StatusCode::UNAUTHORIZED
        );
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn mcp_stays_open_when_nothing_is_configured() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        assert_ne!(mcp_status(app, &[]).await, StatusCode::UNAUTHORIZED);
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn session_creation_is_capped() {
        let state = make_test_state().await.with_max_sessions(1);
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let post = || {
            Request::builder()
                .method(Method::POST)
                .uri("/v1/sessions")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap()
        };
        let first = app.clone().oneshot(post()).await.unwrap();
        assert!(first.status().is_success(), "{}", first.status());
        let second = app.oneshot(post()).await.unwrap();
        assert_eq!(second.status(), StatusCode::CONFLICT);
        let bytes = axum::body::to_bytes(second.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("AXIOM_MAX_SESSIONS"), "{text}");
        safe_drop(pipeline_arc).await;
    }

    #[tokio::test]
    async fn checkpoint_upload_with_overflowing_shape_is_rejected_not_panicking() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let create = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/sessions")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(create.into_body(), usize::MAX).await.unwrap();
        let sid = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["session_id"]
            .as_str()
            .unwrap()
            .to_string();
        let body = serde_json::json!({
            "session_id": sid,
            "version": 1,
            "created_at": 0,
            "layers": [{ "shape": [1u64 << 40, 1u64 << 40], "data": [0.0] }],
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::PUT)
                    .uri(format!("/v1/sessions/{sid}/checkpoint"))
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            resp.status().is_client_error() || resp.status().is_server_error(),
            "{}",
            resp.status()
        );
        safe_drop(pipeline_arc).await;
    }

    /// Build a JIT-enabled state whose Poly-JIT engine times out fast, so
    /// timeout tests run in seconds instead of the 20 s default.
    async fn make_jit_test_state(timeout_secs: u64) -> AppState {
        let mut state = make_test_state().await.with_jit_exec_enabled(true);
        state.poly_jit = std::sync::Arc::new(crate::poly_jit::PolyJitEngine::new(
            std::time::Duration::from_secs(timeout_secs),
        ));
        state
    }

    fn jit_tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("axiom_srv_jit_{tag}_{}", std::process::id()))
    }

    async fn post_jit_run(
        app: axum::Router,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/hypervisor/jit_run")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        // Error responses from this API are plain text (ApiError), not
        // JSON: parse leniently and let the caller assert on the status.
        // Success responses carry the JSON report.
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, value)
    }

    #[tokio::test]
    async fn hypervisor_jit_run_restores_source_when_run_times_out_after_patch() {
        // Repro for audit P1-2: the run writes a candidate patch on step 1,
        // then times out on step 2. The old handler returned the run error
        // before its restore branch, leaving the patch in place.
        let dir = jit_tmp("timeout");
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("src.txt");
        let original = "AXIOM_POLYJIT_FIXTURE_FAIL\n";
        std::fs::write(&source, original).unwrap();
        let script = dir.join("run.sh");
        let counter = dir.join("ran_once");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nif [ -f \"{}\" ]; then sleep 300; else touch \"{}\"; exit 1; fi\n",
                counter.display(),
                counter.display()
            ),
        )
        .unwrap();

        let state = make_jit_test_state(2).await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let body = serde_json::json!({
            "command": "sh",
            "args": [script.to_string_lossy(), counter.to_string_lossy()],
            "source_path": source.to_string_lossy(),
            "session_id": "jit-timeout-test",
        });
        let (status, _) = post_jit_run(app, body).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            std::fs::read_to_string(&source).unwrap(),
            original,
            "a timed-out run must not leave the candidate patch behind"
        );
        safe_drop(pipeline_arc).await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn hypervisor_jit_run_restores_source_when_report_fails() {
        // The pre-existing restore path: every step fails, the report says
        // passed=false, the source must come back byte-for-byte.
        let dir = jit_tmp("fail");
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("src.txt");
        let original = "AXIOM_POLYJIT_FIXTURE_FAIL\n";
        std::fs::write(&source, original).unwrap();
        let script = dir.join("run.sh");
        std::fs::write(&script, "#!/bin/sh\nexit 1\n").unwrap();

        let state = make_jit_test_state(5).await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let body = serde_json::json!({
            "command": "sh",
            "args": [script.to_string_lossy()],
            "source_path": source.to_string_lossy(),
            "session_id": "jit-fail-test",
        });
        let (status, value) = post_jit_run(app, body).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(value["source_restored"], true);
        assert_eq!(std::fs::read_to_string(&source).unwrap(), original);
        safe_drop(pipeline_arc).await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn hypervisor_jit_run_keeps_patch_when_report_passes() {
        // First run fails, the patch is written, the second run passes:
        // the passing patch stays and source_restored is false.
        let dir = jit_tmp("pass");
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("src.txt");
        std::fs::write(&source, "AXIOM_POLYJIT_FIXTURE_FAIL\n").unwrap();
        let script = dir.join("run.sh");
        let counter = dir.join("ran_once");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nif [ -f \"{}\" ]; then exit 0; else touch \"{}\"; exit 1; fi\n",
                counter.display(),
                counter.display()
            ),
        )
        .unwrap();

        let state = make_jit_test_state(5).await;
        let pipeline_arc = state.pipeline.clone();
        let app = create_router(state);
        let body = serde_json::json!({
            "command": "sh",
            "args": [script.to_string_lossy(), counter.to_string_lossy()],
            "source_path": source.to_string_lossy(),
            "session_id": "jit-pass-test",
        });
        let (status, value) = post_jit_run(app, body).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(value["source_restored"], false);
        assert_eq!(
            std::fs::read_to_string(&source).unwrap(),
            "AXIOM_POLYJIT_FIXTURE_PASS\n",
            "the passing patch stays"
        );
        safe_drop(pipeline_arc).await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn source_restore_guard_restores_on_drop_when_armed() {
        let dir = jit_tmp("guard_drop");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("src.txt");
        std::fs::write(&f, "patched").unwrap();
        {
            let _guard = SourceRestoreGuard::new(
                Some(f.to_string_lossy().into_owned()),
                Some("original".to_string()),
            );
        }
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "original");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn source_restore_guard_keeps_file_when_disarmed() {
        let dir = jit_tmp("guard_disarm");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("src.txt");
        std::fs::write(&f, "patched").unwrap();
        {
            let mut guard = SourceRestoreGuard::new(
                Some(f.to_string_lossy().into_owned()),
                Some("original".to_string()),
            );
            guard.disarm();
        }
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "patched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn source_restore_guard_explicit_restore_disarms() {
        let dir = jit_tmp("guard_restore");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("src.txt");
        std::fs::write(&f, "patched").unwrap();
        let mut guard = SourceRestoreGuard::new(
            Some(f.to_string_lossy().into_owned()),
            Some("original".to_string()),
        );
        guard.restore().await.unwrap();
        assert!(guard.restored());
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "original");
        drop(guard);
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "original",
            "an explicit restore must not run twice on drop"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // CORS (audit Finding 7, P2): default deny cross-origin, opt-in
    // allowlist via AXIOM_CORS_ORIGINS. A disallowed origin must get no
    // Access-Control-Allow-Origin header, so a browser refuses to expose
    // the response to that origin's scripts. The request itself is still
    // served (CORS gates the browser read, not the server); only the
    // header is the assertion.
    //
    // The env guard is held only around the env set + router build:
    // `create_router` reads AXIOM_CORS_ORIGINS once at build time, so the
    // var is unset again before any request runs. No assertion runs while
    // the guard is held, so a failing assertion can never poison the
    // guard for the other tests.
    async fn cors_get_healthz(
        allowlist: Option<&str>,
        origin: Option<&str>,
    ) -> axum::response::Response {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = {
            let _guard = ENV_GUARD.lock().expect("env guard poisoned");
            match allowlist {
                Some(list) => std::env::set_var("AXIOM_CORS_ORIGINS", list),
                None => std::env::remove_var("AXIOM_CORS_ORIGINS"),
            }
            let app = create_router(state);
            std::env::remove_var("AXIOM_CORS_ORIGINS");
            app
        };
        let mut builder = Request::builder().method(Method::GET).uri("/healthz");
        if let Some(origin) = origin {
            builder = builder.header("origin", origin);
        }
        let resp = app.oneshot(builder.body(Body::empty()).unwrap()).await.unwrap();
        safe_drop(pipeline_arc).await;
        resp
    }

    #[tokio::test]
    async fn cors_default_denies_unlisted_origin() {
        let resp = cors_get_healthz(None, Some("https://evil.example")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "unlisted origin must get no Access-Control-Allow-Origin header, got {:?}",
            resp.headers().get("access-control-allow-origin")
        );
    }

    #[tokio::test]
    async fn cors_allows_listed_origin() {
        let resp =
            cors_get_healthz(Some("https://app.example.com"), Some("https://app.example.com"))
                .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("https://app.example.com"),
            "listed origin must be echoed back as the allowed origin"
        );
    }

    #[tokio::test]
    async fn cors_disallowed_origin_when_allowlist_set() {
        let resp =
            cors_get_healthz(Some("https://app.example.com"), Some("https://other.example")).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "origin outside the allowlist must get no Access-Control-Allow-Origin header"
        );
    }

    #[tokio::test]
    async fn cors_request_without_origin_gets_no_cors_header() {
        let resp = cors_get_healthz(None, None).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "non-browser requests without an Origin header must get no CORS header"
        );
    }

    #[tokio::test]
    async fn cors_preflight_for_listed_origin_is_allowed() {
        let state = make_test_state().await;
        let pipeline_arc = state.pipeline.clone();
        let app = {
            let _guard = ENV_GUARD.lock().expect("env guard poisoned");
            std::env::set_var("AXIOM_CORS_ORIGINS", "https://app.example.com");
            let app = create_router(state);
            std::env::remove_var("AXIOM_CORS_ORIGINS");
            app
        };
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/healthz")
                    .header("origin", "https://app.example.com")
                    .header("access-control-request-method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("https://app.example.com"),
            "preflight from a listed origin must carry the allowed origin"
        );
        safe_drop(pipeline_arc).await;
    }

    #[test]
    fn parse_cors_origins_trims_and_drops_wildcard_and_empties() {
        let origins = parse_cors_origins(" https://a.example , ,*, http://localhost:3000 ");
        let values: Vec<&str> = origins.iter().map(|v| v.to_str().unwrap()).collect();
        assert_eq!(values, ["https://a.example", "http://localhost:3000"]);
    }

    /// Minimal raw-TCP fake for the route-level streaming-fallback test:
    /// serves one fixed response per request line. `hit` records whether
    /// the expected path was requested at all.
    async fn serve_fixed_responses(
        listener: tokio::net::TcpListener,
        routes: Vec<(&'static str, &'static str, String)>,
        hit: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let routes = routes.clone();
            let hit = hit.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                loop {
                    let Ok(n) = socket.read(&mut tmp).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                    if buf.len() > 65_536 {
                        return;
                    }
                }
                let head = String::from_utf8_lossy(&buf).to_string();
                // Drain the declared request body before responding: the
                // client may still be sending it, and closing the socket
                // early turns into a connection reset on the client side.
                // The drain is capped so a lying Content-Length cannot hang
                // the fake.
                let header_end = buf
                    .windows(4)
                    .position(|w| w == b"\r\n\r\n")
                    .map(|i| i + 4)
                    .unwrap_or(buf.len());
                let content_length: usize = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.trim().eq_ignore_ascii_case("content-length") {
                            value.trim().parse().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let buffered_body = buf.len().saturating_sub(header_end);
                let mut remaining = content_length.saturating_sub(buffered_body).min(65_536);
                while remaining > 0 {
                    let chunk = remaining.min(tmp.len());
                    match socket.read(&mut tmp[..chunk]).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => remaining -= n,
                    }
                }
                let request_line = head.lines().next().unwrap_or("").to_string();
                let mut parts = request_line.split_whitespace();
                let method = parts.next().unwrap_or("");
                let path = parts.next().unwrap_or("");
                for (route_method, route_path, response) in &routes {
                    if method == *route_method && path == *route_path {
                        hit.store(true, std::sync::atomic::Ordering::SeqCst);
                        let _ = socket.write_all(response.as_bytes()).await;
                        return;
                    }
                }
                let _ = socket
                    .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await;
            });
        }
    }

    #[tokio::test]
    async fn streaming_startup_failure_falls_back_to_cloud_forwarder() {
        use axum::body::to_bytes;
        use std::sync::atomic::{AtomicBool, Ordering};

        // Fake Ollama: /api/tags succeeds (model available) but /api/chat
        // fails, modelling an upstream that dies between selection and
        // generation.
        let ollama_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ollama_addr = ollama_listener.local_addr().unwrap();
        let ollama_hit = std::sync::Arc::new(AtomicBool::new(false));
        let ollama_hit_clone = ollama_hit.clone();
        tokio::spawn(serve_fixed_responses(
            ollama_listener,
            vec![
                (
                    "GET",
                    "/api/tags",
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 33\r\nConnection: close\r\n\r\n{\"models\":[{\"name\":\"phi4:3.8b\"}]}"
                        .to_string(),
                ),
                (
                    "POST",
                    "/api/chat",
                    "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: 15\r\nConnection: close\r\n\r\nollama exploded"
                        .to_string(),
                ),
            ],
            ollama_hit_clone,
        ));

        // Fake cloud: answers chat completions with a fixed completion.
        let cloud_body = r#"{"id":"chatcmpl-cloud","object":"chat.completion","created":1,"model":"cloud","choices":[{"index":0,"message":{"role":"assistant","content":"cloud answer"},"finish_reason":"stop"}]}"#;
        let cloud_response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            cloud_body.len(),
            cloud_body
        );
        let cloud_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let cloud_addr = cloud_listener.local_addr().unwrap();
        let cloud_hit = std::sync::Arc::new(AtomicBool::new(false));
        let cloud_hit_clone = cloud_hit.clone();
        tokio::spawn(serve_fixed_responses(
            cloud_listener,
            vec![("POST", "/v1/chat/completions", cloud_response)],
            cloud_hit_clone,
        ));

        let router = crate::swarm_router::SwarmRouter::new(crate::swarm_router::SwarmRouterConfig {
            enabled: true,
            base_url: format!("http://{ollama_addr}"),
            model_candidates: vec!["phi4:3.8b".to_string()],
            num_ctx: 4096,
            timeout_ms: 5_000,
            mesh_routing: false,
        });
        let forwarder = crate::openai_forwarder::OpenAiForwarder::new(
            None,
            Some(format!("http://{cloud_addr}")),
        );
        let state = make_test_state()
            .await
            .with_swarm_router(Some(router))
            .with_openai_forwarder(Some(forwarder));
        let auth = crate::openai_forwarder::OpenAiClientAuth {
            authorization: Some("Bearer test-key".to_string()),
            extra_headers: vec![],
        };
        let body = serde_json::json!({
            "model": "phi4:3.8b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
        });
        let resp = compressed_openai_chat_path(&state, &body, None, &auth)
            .await
            .expect("route must answer via the cloud fallback");
        assert_eq!(resp.status(), StatusCode::OK);
        let text = String::from_utf8(to_bytes(resp.into_body(), usize::MAX).await.unwrap().to_vec())
            .unwrap();
        assert!(
            ollama_hit.load(Ordering::SeqCst),
            "the local dispatch must have been attempted first"
        );
        assert!(
            cloud_hit.load(Ordering::SeqCst),
            "the startup failure must reach the cloud forwarder"
        );
        assert!(
            text.contains("cloud answer"),
            "the response must carry the cloud answer, got:\n{text}"
        );
        assert!(
            !text.contains("data: [DONE]"),
            "the fallback response must not be an empty SSE stream, got:\n{text}"
        );
        safe_drop(state.pipeline.clone()).await;
    }
}
