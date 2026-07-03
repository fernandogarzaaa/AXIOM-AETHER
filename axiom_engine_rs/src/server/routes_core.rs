// Handlers
// ---------------------------------------------------------------------------

/// `GET /v1/models` — list available models.
async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let resp = ListModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "axiom-ttt".to_string(),
        }],
    };
    Json(resp)
}

async fn export_metrics(State(state): State<AppState>) -> Result<Response, ApiError> {
    state.refresh_session_metrics()?;
    let mut rendered = metrics::render_metrics();
    let dwe = state.dwe_bus.telemetry();
    rendered.push_str(&format!(
        "# HELP axiom_dwe_sent Total DWE fragments sent to peers.\n\
         # TYPE axiom_dwe_sent counter\n\
         axiom_dwe_sent {}\n\
         # HELP axiom_dwe_received Total DWE fragments received from peers.\n\
         # TYPE axiom_dwe_received counter\n\
         axiom_dwe_received {}\n\
         # HELP axiom_dwe_applied Total DWE fragments applied to local sessions.\n\
         # TYPE axiom_dwe_applied counter\n\
         axiom_dwe_applied {}\n\
         # HELP axiom_dwe_rejected Total DWE fragments rejected before apply.\n\
         # TYPE axiom_dwe_rejected counter\n\
         axiom_dwe_rejected {}\n",
        dwe.sent_fragments, dwe.received_fragments, dwe.applied_fragments, dwe.rejected_fragments
    ));
    // Compression savings: lifetime monotone counters (session drops remove
    // ledger entries for receipts, so the ledger alone would undercount).
    let bytes_in = LIFETIME_SAVINGS_IN.load(std::sync::atomic::Ordering::Relaxed);
    let bytes_out = LIFETIME_SAVINGS_OUT.load(std::sync::atomic::Ordering::Relaxed);
    let ratio = if bytes_in > 0 && bytes_out < bytes_in {
        (bytes_in - bytes_out) as f64 / bytes_in as f64
    } else {
        0.0
    };
    rendered.push_str(&format!(
        "# TYPE axiom_savings_bytes_in_total counter\naxiom_savings_bytes_in_total {bytes_in}\n\
         # TYPE axiom_savings_bytes_forwarded_total counter\naxiom_savings_bytes_forwarded_total {bytes_out}\n\
         # TYPE axiom_savings_ratio gauge\naxiom_savings_ratio {ratio:.4}\n"
    ));
    Ok((
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        rendered,
    )
        .into_response())
}
