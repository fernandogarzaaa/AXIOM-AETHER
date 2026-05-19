use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tower_http::cors::{Any, CorsLayer};

use crate::inference::InferencePipeline;

#[derive(Clone)]
struct AppState {
    pipeline: Arc<InferencePipeline>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    prompt: String,
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: usize,
    #[serde(default)]
    runtime_configs: Option<Value>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    response: String,
}

#[derive(Debug, Deserialize)]
struct StreamQuery {
    prompt: String,
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: usize,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({ "error": self.message }));
        (self.status, body).into_response()
    }
}

fn default_max_new_tokens() -> usize {
    32
}

pub async fn start_server(
    host: &str,
    port: u16,
    pipeline: Arc<InferencePipeline>,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState { pipeline };
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/stream", get(chat_stream))
        .layer(cors)
        .with_state(state);

    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    println!("[+] Axiom-TTT Engine API active and listening on http://{bind_addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    if payload.prompt.trim().is_empty() {
        return Err(ApiError::bad_request("prompt must not be empty"));
    }
    let _ = payload.runtime_configs;
    let prompt = payload.prompt;
    let max_new_tokens = payload.max_new_tokens.max(1);
    let pipeline = state.pipeline.clone();

    let generated = tokio::task::spawn_blocking(move || pipeline.generate(&prompt, max_new_tokens))
        .await
        .map_err(|err| ApiError::internal(format!("generation task failed: {err}")))?
        .map_err(|err| ApiError::internal(format!("generation failed: {err}")))?;

    Ok(Json(ChatCompletionResponse {
        response: generated,
    }))
}

async fn chat_stream(
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
) -> Result<Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    if query.prompt.trim().is_empty() {
        return Err(ApiError::bad_request("prompt must not be empty"));
    }
    let prompt = query.prompt;
    let max_new_tokens = query.max_new_tokens.max(1);
    let pipeline = state.pipeline.clone();

    let generated = tokio::task::spawn_blocking(move || pipeline.generate(&prompt, max_new_tokens))
        .await
        .map_err(|err| ApiError::internal(format!("stream generation task failed: {err}")))?
        .map_err(|err| ApiError::internal(format!("stream generation failed: {err}")))?;

    let mut chunks = generated
        .split_whitespace()
        .map(|token| token.to_string())
        .collect::<Vec<_>>();
    if chunks.is_empty() {
        chunks.push(generated);
    }
    chunks.push(String::from("[DONE]"));

    let token_stream = stream::unfold((0usize, chunks), |(idx, chunks)| async move {
        if idx >= chunks.len() {
            return None;
        }
        let chunk = chunks[idx].clone();
        tokio::time::sleep(Duration::from_millis(8)).await;
        let event = Event::default().event("token").data(chunk);
        Some((Ok(event), (idx + 1, chunks)))
    });

    Ok(Sse::new(token_stream).keep_alive(KeepAlive::default()))
}
