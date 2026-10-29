/// `POST /v1/completions` — text completion (stateless or session-aware).
async fn create_completion(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let max_tokens = req.max_tokens.unwrap_or(32);
    let model = req.model.as_deref().unwrap_or(&state.model_id).to_string();

    // Generation is synchronous (model decode loop, or a blocking HTTP call
    // to a remote backend); keep it off the async worker threads.
    let text = {
        let gen_state = state.clone();
        spawn_blocking(move || {
            run_generation(&gen_state, &req.prompt, max_tokens, req.session_id.as_deref())
        })
        .await
        .map_err(|e| ApiError::Internal(format!("generation task join failed: {e}")))??
    };
    state.trigger_lru_vram_budget();

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: unix_now(),
        model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
    }))
}

/// `POST /v1/chat/completions` — chat completion (stateless or session-aware).
///
/// When `stream: true` is set in the request body, the response is an SSE
/// stream of `chat.completion.chunk` objects (OpenAI streaming format) terminated
/// by the sentinel `data: [DONE]\n\n`.  Clients such as Open WebUI, LangChain,
/// and curl --no-buffer work without any code change.
///
/// When `stream: false` (or absent), a single JSON object is returned.
async fn create_chat_completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let session_override = headers
        .get("x-axiom-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    if state.openai_compression_active() {
        let client_auth = OpenAiClientAuth {
            authorization: headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string),
            extra_headers: Vec::new(),
        };
        match compressed_openai_chat_path(&state, &body, session_override.as_deref(), &client_auth)
            .await
        {
            Ok(resp) => return resp,
            Err(err) => return err.into_response(),
        }
    }

    let req: ChatCompletionRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => {
            return ApiError::BadRequest(format!("invalid /v1/chat/completions body: {e}"))
                .into_response();
        }
    };

    // Both paths generate synchronously (see `run_generation`), so run them
    // on the blocking pool rather than stalling an async worker.
    let gen_state = state.clone();
    let response = if req.stream.unwrap_or(false) {
        spawn_blocking(move || chat_completion_sse(gen_state, req).into_response()).await
    } else {
        spawn_blocking(move || chat_completion_json(gen_state, req).into_response()).await
    };
    state.trigger_lru_vram_budget();
    response.unwrap_or_else(|e| {
        ApiError::Internal(format!("generation task join failed: {e}")).into_response()
    })
}
