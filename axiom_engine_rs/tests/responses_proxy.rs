use std::sync::{Arc, Mutex};

use axiom_engine::config::AxiomConfig;
use axiom_engine::inference::InferencePipeline;
use axiom_engine::openai_forwarder::OpenAiForwarder;
use axiom_engine::server::{create_router, AppState};
use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::{header, HeaderMap, Method, Request, Response, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use candle_core::Device;
use serde_json::{json, Value};
use tower::ServiceExt;

#[derive(Clone, Default)]
struct Capture {
    requests: Arc<Mutex<Vec<(Option<String>, Value)>>>,
}

fn build_pipeline() -> InferencePipeline {
    InferencePipeline::new(
        AxiomConfig {
            d_model: 16,
            n_layers: 1,
            vocab_size: 64,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        },
        Device::Cpu,
    )
    .expect("tiny pipeline must build")
}

async fn start_mock_responses_upstream(
    streaming: bool,
) -> (String, Capture, tokio::task::JoinHandle<()>) {
    async fn handler(
        State((capture, streaming)): State<(Capture, bool)>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Response<Body> {
        let auth = headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        capture.requests.lock().unwrap().push((auth, body));

        if streaming {
            let events = concat!(
                "data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_test\",\"object\":\"response\",\"status\":\"in_progress\"}}\n\n",
                "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"item_id\":\"msg_test\",\"output_index\":0,\"content_index\":0,\"delta\":\"hello\"}\n\n",
                "data: {\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{\"id\":\"resp_test\",\"object\":\"response\",\"status\":\"completed\"}}\n\n"
            );
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from(events))
                .unwrap()
        } else {
            Response::builder()
                .status(StatusCode::CREATED)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "id": "resp_test",
                        "object": "response",
                        "status": "completed",
                        "output": [{
                            "id": "msg_test",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "hello", "annotations": []}]
                        }]
                    })
                    .to_string(),
                ))
                .unwrap()
        }
    }

    let capture = Capture::default();
    let app = Router::new()
        .route("/v1/responses", post(handler))
        .with_state((capture.clone(), streaming));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), capture, task)
}

async fn proxy_app(upstream: String) -> Router {
    let pipeline = tokio::task::spawn_blocking(build_pipeline).await.unwrap();
    let state = AppState::new(pipeline, "responses-proxy-test".to_string())
        .with_openai_forwarder(Some(OpenAiForwarder::new(None, Some(upstream))));
    create_router(state)
}

#[tokio::test]
async fn responses_json_relays_auth_request_and_upstream_response() {
    let (upstream, capture, task) = start_mock_responses_upstream(false).await;
    let app = proxy_app(upstream).await;
    let request_body = json!({
        "model": "gpt-5.5",
        "instructions": "Be concise.",
        "input": "hello",
        "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
        "stream": false,
        "store": false
    });
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer client-secret")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/json"
    );
    let response_body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: Value = serde_json::from_slice(&response_body).unwrap();
    assert_eq!(response_json["object"], "response");
    assert_eq!(response_json["output"][0]["content"][0]["text"], "hello");

    let captured = capture.requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].0.as_deref(), Some("Bearer client-secret"));
    assert_eq!(captured[0].1, request_body);
    task.abort();
}

#[tokio::test]
async fn responses_stream_relays_semantic_sse_events_unchanged() {
    let (upstream, capture, task) = start_mock_responses_upstream(true).await;
    let app = proxy_app(upstream).await;
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer stream-secret")
                .body(Body::from(
                    json!({"model": "gpt-5.5", "input": "hello", "stream": true}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(response
        .headers()
        .get(header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("response.output_text.delta"));
    assert!(text.contains("response.completed"));
    assert!(!text.contains("chat.completion.chunk"));
    assert_eq!(capture.requests.lock().unwrap().len(), 1);
    task.abort();
}

#[tokio::test]
async fn responses_plain_get_returns_json_diagnostic_instead_of_bare_rejection() {
    // A GET without a complete WebSocket handshake (e.g. a misconfigured
    // client probing the endpoint, or a bare `Upgrade: websocket` header with
    // no Sec-WebSocket-Key) must get a structured JSON diagnostic instead of
    // axum's plain-text extractor rejection. A *valid* WebSocket handshake on
    // this route is relayed upstream and is not what this test exercises.
    let (upstream, _capture, task) = start_mock_responses_upstream(false).await;
    let app = proxy_app(upstream).await;
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/responses")
                .header(header::UPGRADE, "websocket")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UPGRADE_REQUIRED);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/json"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        response_json["error"]["code"],
        "responses_requires_post_or_websocket"
    );
    let message = response_json["error"]["message"].as_str().unwrap();
    assert!(message.contains("POST"), "message should mention the HTTP POST path");
    assert!(
        message.contains("WebSocket"),
        "message should mention the supported WebSocket upgrade path"
    );
    task.abort();
}
