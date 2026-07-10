//! Integration coverage for S0: dollar-true cache-aware cost telemetry.
//!
//! Verifies the proxy parses Anthropic `usage` from a real (mocked) upstream
//! response and exposes it through `/metrics` and `GET /v1/awareness/:id`.

use axiom_engine::anthropic_forwarder::AnthropicForwarder;
use axiom_engine::config::AxiomConfig;
use axiom_engine::context_compressor::CompressorConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::http::{header, Method, Request, StatusCode};
use axum::response::Response;
use axum::routing::post;
use axum::{Json, Router};
use candle_core::Device;
use serde_json::{json, Value};
use tower::ServiceExt;

fn tiny_pipeline() -> InferencePipeline {
    let config = AxiomConfig {
        d_model: 16,
        n_layers: 1,
        vocab_size: 64,
        lr_inner: 1e-3,
        norm_eps: 1e-6,
    };
    InferencePipeline::new(config, Device::Cpu).expect("pipeline init")
}

/// Mock Anthropic `/v1/messages` upstream returning a fixed `usage` block.
async fn start_mock_anthropic_upstream() -> (String, tokio::task::JoinHandle<()>) {
    async fn handler() -> Json<Value> {
        Json(json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 1000,
                "cache_creation_input_tokens": 2000,
                "cache_read_input_tokens": 77_000,
                "output_tokens": 500,
            }
        }))
    }
    let app = Router::new().route("/v1/messages", post(handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), task)
}

/// Mock Anthropic `/v1/messages` upstream returning a real SSE event stream,
/// with usage split across `message_start` (input side) and `message_delta`
/// (output side) -- the same split the S0 streaming scanner must reassemble.
async fn start_mock_anthropic_streaming_upstream() -> (String, tokio::task::JoinHandle<()>) {
    async fn handler() -> Response {
        let events = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_test\",\"type\":\"message\",\
             \"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"content\":[],\
             \"stop_reason\":null,\"usage\":{\"input_tokens\":1000,\
             \"cache_creation_input_tokens\":2000,\"cache_read_input_tokens\":77000,\
             \"output_tokens\":1}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\
             \"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\
             \"usage\":{\"output_tokens\":500}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from(events))
            .unwrap()
    }
    let app = Router::new().route("/v1/messages", post(handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), task)
}

async fn post_json(app: &Router, path: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header("content-type", "application/json")
        .header("x-api-key", "sk-ant-test-key")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, parsed)
}

async fn get_json(app: &Router, path: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(Method::GET)
        .uri(path)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, parsed)
}

#[tokio::test]
async fn non_streaming_messages_turn_updates_awareness_cost_summary() {
    let (upstream, _task) = start_mock_anthropic_upstream().await;
    let pipeline = tokio::task::spawn_blocking(tiny_pipeline).await.unwrap();
    let state = AppState::new(pipeline, "axiom-ttt-test".to_string())
        .with_anthropic_forwarder(Some(AnthropicForwarder::new(None, Some(upstream))))
        .with_compressor_config(CompressorConfig {
            heavy_message_threshold_tokens: 512,
            recall_top_k: 32,
            enabled: true,
        });
    let app = create_router(state);

    let session_id = "cost-test-session";
    let (status, _body) = post_json(
        &app,
        "/v1/messages",
        json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "session_id": session_id,
            "messages": [{"role": "user", "content": "hello"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (aw_status, awareness) =
        get_json(&app, &format!("/v1/awareness/{session_id}")).await;
    assert_eq!(aw_status, StatusCode::OK);

    let cost = &awareness["cost"];
    assert_eq!(cost["cache_read_tokens"], 77_000);
    assert_eq!(cost["cache_write_tokens"], 2000);
    assert_eq!(cost["uncached_input_tokens"], 1000);
    assert_eq!(cost["output_tokens"], 500);
    let expected_usd = 1000.0 / 1e6 * 3.00
        + 2000.0 / 1e6 * 3.75
        + 77_000.0 / 1e6 * 0.30
        + 500.0 / 1e6 * 15.00;
    let usd = cost["usd_total"].as_f64().unwrap();
    assert!((usd - expected_usd).abs() < 1e-6, "{usd} vs {expected_usd}");
    let uncached_equiv = cost["usd_uncached_equivalent"].as_f64().unwrap();
    assert!(
        uncached_equiv > usd,
        "uncached equivalent must exceed the cached actual cost"
    );
    assert_eq!(cost["estimated"], false);

    let req = Request::builder()
        .method(Method::GET)
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    // Lifetime counters are process-global (same pattern as the existing
    // axiom_savings_* counters), so other tests in this binary contribute to
    // the same totals -- assert presence and a floor, not an exact value.
    assert!(text.contains("axiom_cost_usd_total"));
    assert!(text.contains("axiom_cost_uncached_usd_total"));
    let extract = |metric: &str| -> u64 {
        text.lines()
            .find(|l| l.starts_with(metric) && l.chars().nth(metric.len()) == Some(' '))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("metric {metric} not found in /metrics output"))
    };
    assert!(extract("axiom_cache_read_tokens_total") >= 77_000);
    assert!(extract("axiom_cache_write_tokens_total") >= 2_000);
    assert!(extract("axiom_uncached_input_tokens_total") >= 1_000);
}

#[tokio::test]
async fn streaming_messages_turn_scans_sse_usage_without_altering_the_stream() {
    let (upstream, _task) = start_mock_anthropic_streaming_upstream().await;
    let pipeline = tokio::task::spawn_blocking(tiny_pipeline).await.unwrap();
    let state = AppState::new(pipeline, "axiom-ttt-test".to_string())
        .with_anthropic_forwarder(Some(AnthropicForwarder::new(None, Some(upstream))))
        .with_compressor_config(CompressorConfig {
            heavy_message_threshold_tokens: 512,
            recall_top_k: 32,
            enabled: true,
        });
    let app = create_router(state);

    let session_id = "cost-stream-session";
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .header("x-api-key", "sk-ant-test-key")
        .body(Body::from(
            json!({
                "model": "claude-sonnet-4-6",
                "max_tokens": 16,
                "session_id": session_id,
                "stream": true,
                "messages": [{"role": "user", "content": "hello"}],
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // The response body must reach the client byte-for-byte unchanged --
    // the scanner observes, it never rewrites.
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.contains("event: message_start"));
    assert!(text.contains("event: message_delta"));
    assert!(text.contains("\"text\":\"hi\""));

    let (aw_status, awareness) =
        get_json(&app, &format!("/v1/awareness/{session_id}")).await;
    assert_eq!(aw_status, StatusCode::OK);
    let cost = &awareness["cost"];
    assert_eq!(cost["cache_read_tokens"], 77_000);
    assert_eq!(cost["cache_write_tokens"], 2000);
    assert_eq!(cost["uncached_input_tokens"], 1000);
    // output_tokens must come from message_delta (500), not message_start's
    // placeholder (1) -- proves the two events were merged, not just the
    // first one recorded.
    assert_eq!(cost["output_tokens"], 500);
    assert!(cost["usd_total"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn streaming_prices_with_upstream_resolved_model_not_the_request_model() {
    // The request declares "claude-sonnet-5" but the mock's message_start
    // resolves to "claude-sonnet-4-6" (the two now have genuinely different
    // rates: $2 vs $3 input/MTok). Pricing must follow the upstream-resolved
    // model, matching the non-streaming path's precedent (forwarded.model
    // preferred over body.model), not silently price at the request's
    // declared model.
    let (upstream, _task) = start_mock_anthropic_streaming_upstream().await;
    let pipeline = tokio::task::spawn_blocking(tiny_pipeline).await.unwrap();
    let state = AppState::new(pipeline, "axiom-ttt-test".to_string())
        .with_anthropic_forwarder(Some(AnthropicForwarder::new(None, Some(upstream))))
        .with_compressor_config(CompressorConfig {
            heavy_message_threshold_tokens: 512,
            recall_top_k: 32,
            enabled: true,
        });
    let app = create_router(state);

    let session_id = "cost-stream-model-mismatch";
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .header("x-api-key", "sk-ant-test-key")
        .body(Body::from(
            json!({
                "model": "claude-sonnet-5",
                "max_tokens": 16,
                "session_id": session_id,
                "stream": true,
                "messages": [{"role": "user", "content": "hello"}],
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let _ = to_bytes(resp.into_body(), usize::MAX).await.unwrap();

    let (_, awareness) = get_json(&app, &format!("/v1/awareness/{session_id}")).await;
    let usd = awareness["cost"]["usd_total"].as_f64().unwrap();
    // Legacy Sonnet 4.x rate ($3/$3.75/$0.30/$15), matching message_start's
    // declared model -- NOT the request's "claude-sonnet-5" ($2/.../$10).
    let expected_legacy = 1000.0 / 1e6 * 3.00
        + 2000.0 / 1e6 * 3.75
        + 77_000.0 / 1e6 * 0.30
        + 500.0 / 1e6 * 15.00;
    assert!(
        (usd - expected_legacy).abs() < 1e-6,
        "expected upstream-resolved (legacy Sonnet 4.6) pricing {expected_legacy}, got {usd}"
    );
}
