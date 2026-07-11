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
    // Dollar-true cache-aware cost telemetry (S0, CVM cost stack). Lifetime
    // monotone counters -- see docs/superpowers/plans/2026-07-10-cvm-cost-stack.md.
    {
        use std::sync::atomic::Ordering;
        let cost_usd = LIFETIME_COST_USD_MICROS.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let uncached_usd =
            LIFETIME_COST_UNCACHED_EQUIVALENT_USD_MICROS.load(Ordering::Relaxed) as f64
                / 1_000_000.0;
        let cache_read = LIFETIME_CACHE_READ_TOKENS.load(Ordering::Relaxed);
        let cache_write = LIFETIME_CACHE_WRITE_TOKENS.load(Ordering::Relaxed);
        let uncached_in = LIFETIME_UNCACHED_INPUT_TOKENS.load(Ordering::Relaxed);
        let quota_units = LIFETIME_QUOTA_UNITS_MICROS.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let quota_units_uncached =
            LIFETIME_QUOTA_UNITS_UNCACHED_MICROS.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        rendered.push_str(&format!(
            "# HELP axiom_cost_usd_total Lifetime dollar-true cost of proxied API turns.\n\
             # TYPE axiom_cost_usd_total counter\naxiom_cost_usd_total {cost_usd:.6}\n\
             # HELP axiom_cost_uncached_usd_total Counterfactual cost with zero caching.\n\
             # TYPE axiom_cost_uncached_usd_total counter\naxiom_cost_uncached_usd_total {uncached_usd:.6}\n\
             # TYPE axiom_cache_read_tokens_total counter\naxiom_cache_read_tokens_total {cache_read}\n\
             # TYPE axiom_cache_write_tokens_total counter\naxiom_cache_write_tokens_total {cache_write}\n\
             # TYPE axiom_uncached_input_tokens_total counter\naxiom_uncached_input_tokens_total {uncached_in}\n\
             # HELP axiom_quota_units_total Lifetime subscription quota units (1 unit = 1 Sonnet-5 intro-rate uncached input token).\n\
             # TYPE axiom_quota_units_total counter\naxiom_quota_units_total {quota_units:.6}\n\
             # HELP axiom_quota_units_uncached_total Counterfactual quota units with zero caching.\n\
             # TYPE axiom_quota_units_uncached_total counter\naxiom_quota_units_uncached_total {quota_units_uncached:.6}\n"
        ));
    }
    Ok((
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        rendered,
    )
        .into_response())
}
