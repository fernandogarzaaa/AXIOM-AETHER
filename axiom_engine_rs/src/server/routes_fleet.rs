#[derive(Debug, Serialize)]
struct FleetStatusResponse {
    dwe: DweTelemetry,
    peers: Vec<String>,
    listen: Option<String>,
    key_configured: bool,
    previous_key_configured: bool,
}

async fn fleet_status(State(state): State<AppState>) -> Json<FleetStatusResponse> {
    Json(FleetStatusResponse {
        dwe: state.dwe_bus.telemetry(),
        peers: configured_dwe_peers(),
        listen: nonempty_env("AXIOM_DWE_LISTEN"),
        key_configured: fleet_key().is_some(),
        previous_key_configured: fleet_key_prev().is_some(),
    })
}

/// `GET /v1/immunity` — export this node's heal memory for swarm peers.
async fn get_immunity(State(state): State<AppState>) -> Response {
    let Some(path) = state.heal_memory_path.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "heal memory not configured on this node"})),
        )
            .into_response();
    };
    let memory = crate::heal_memory::HealMemory::load(path);
    // Wrap the export with tamper-evident provenance (full SHA-256 + optional
    // HMAC when AXIOM_FLEET_KEY is set) so peers can verify before trusting.
    let fleet_key = fleet_key();
    let signed = crate::provenance::sign_export(&memory.to_json(), fleet_key.as_deref());
    Json(signed).into_response()
}

/// The shared fleet secret for swarm-immunity authentication, from
/// `AXIOM_FLEET_KEY`. `None` → exports are hashed but not signed.
fn fleet_key() -> Option<Vec<u8>> {
    nonempty_env("AXIOM_FLEET_KEY").map(String::into_bytes)
}

fn fleet_key_prev() -> Option<Vec<u8>> {
    nonempty_env("AXIOM_FLEET_KEY_PREV").map(String::into_bytes)
}

fn nonempty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn configured_dwe_peers() -> Vec<String> {
    std::env::var("AXIOM_DWE_PEERS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|peer| !peer.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// `POST /v1/immunity/merge` — fold a peer's exported heal memory into this
/// node's (swarm immunity). Body: the peer's `GET /v1/immunity` payload.
/// Returns the merge report. Local learning is never weakened: dirs are
/// unioned and tension histories are count-weighted.
async fn post_immunity_merge(State(state): State<AppState>, body: String) -> Response {
    let Some(path) = state.heal_memory_path.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "heal memory not configured on this node"})),
        )
            .into_response();
    };
    let mut memory = crate::heal_memory::HealMemory::load(path);
    // If the body is a signed export, verify provenance before trusting it.
    // A raw heal-memory JSON (no schema marker) is still accepted for
    // back-compat — but if a fleet key is configured, unsigned input is refused.
    let fleet_key = fleet_key();
    let payload: String = match serde_json::from_str::<crate::provenance::SignedExport>(&body) {
        Ok(export) => match crate::provenance::verify_export(&export, fleet_key.as_deref()) {
            Ok(verified) => verified.to_string(),
            Err(e) => {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": format!("provenance rejected: {e}")})),
                )
                    .into_response();
            }
        },
        Err(_) if fleet_key.is_some() => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({
                    "error": "fleet key configured but peer sent an unsigned (raw) payload"
                })),
            )
                .into_response();
        }
        Err(_) => body.clone(), // back-compat: unsigned raw payload, no key required
    };
    match memory.merge_json(&payload) {
        Ok(report) => {
            if let Err(e) = memory.save() {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": format!("merge succeeded but save failed: {e}")})),
                )
                    .into_response();
            }
            Json(serde_json::json!({
                "merged": true,
                "programs_added": report.programs_added,
                "programs_merged": report.programs_merged,
                "dirs_added": report.dirs_added,
                "belief_conflicts": report.belief_conflicts,
                "byzantine_rejected": report.byzantine_rejected,
            }))
            .into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

/// Locate this node's verified-patch store, co-located with the configured heal
/// memory (`axiom_patch_memory.json`), matching `PatchMemory::default_path()`.
pub fn start_dwe_apply_loop(
    apply_state: AppState,
    mut in_rx: tokio::sync::mpsc::Receiver<crate::dwe::DweFragment>,
    verify_secret: Vec<u8>,
    previous_secret: Option<Vec<u8>>,
    telemetry: Arc<Mutex<DweTelemetry>>,
) {
    tokio::spawn(async move {
        while let Some(fragment) = in_rx.recv().await {
            if let Err(e) = crate::dwe::verify_fragment_with_rotation(
                &fragment,
                &verify_secret,
                previous_secret.as_deref(),
            ) {
                eprintln!("[dwe] rejected fragment for '{}': {e}", fragment.session_id);
                record_rejected_fragment(&telemetry, &fragment.session_id, &e);
                continue;
            }
            {
                let seq_key = format!("dwe:{}", fragment.session_id);
                let Ok(mut seqs) = apply_state.sequence_versions.write() else {
                    record_rejected_fragment(
                        &telemetry,
                        &fragment.session_id,
                        "sequence lock poisoned",
                    );
                    continue;
                };
                if let Some(prev) = seqs.get(&seq_key) {
                    if fragment.sequence <= prev.version {
                        let error = format!(
                            "stale fragment seq {} <= {}",
                            fragment.sequence, prev.version
                        );
                        eprintln!("[dwe] {error} for '{}'; dropped", fragment.session_id);
                        record_rejected_fragment(&telemetry, &fragment.session_id, &error);
                        continue;
                    }
                }
                seqs.insert(
                    seq_key,
                    SequenceState {
                        version: fragment.sequence,
                        timestamp: unix_now() as i64,
                    },
                );
            }
            let Ok(mut sessions) = apply_state.sessions.write() else {
                record_rejected_fragment(&telemetry, &fragment.session_id, "session lock poisoned");
                continue;
            };
            let Some(session) = sessions.get_mut(&fragment.session_id) else {
                eprintln!(
                    "[dwe] fragment for unknown session '{}' dropped",
                    fragment.session_id
                );
                record_rejected_fragment(&telemetry, &fragment.session_id, "unknown session");
                continue;
            };
            let mut applied_any = false;
            for layer in &fragment.layers {
                let total: usize = layer.shape.iter().product();
                if total != layer.values.len() {
                    let error = format!("layer {} shape/value mismatch", layer.layer_index);
                    eprintln!("[dwe] {error}");
                    record_rejected_fragment(&telemetry, &fragment.session_id, &error);
                    continue;
                }
                let delta = Tensor::from_vec(layer.values.clone(), (total,), &apply_state.device)
                    .and_then(|t| t.reshape(layer.shape.as_slice()));
                match delta {
                    Ok(delta) => {
                        if let Err(e) =
                            session.merge_delta(layer.layer_index, &delta, &apply_state.device)
                        {
                            let error =
                                format!("merge failed for layer {}: {e}", layer.layer_index);
                            eprintln!(
                                "[dwe] merge failed for session '{}' layer {}: {e}",
                                fragment.session_id, layer.layer_index
                            );
                            record_rejected_fragment(&telemetry, &fragment.session_id, &error);
                        } else {
                            applied_any = true;
                        }
                    }
                    Err(e) => {
                        let error = format!(
                            "fragment tensor rebuild failed for layer {}: {e}",
                            layer.layer_index
                        );
                        eprintln!(
                            "[dwe] fragment tensor rebuild failed (layer {}): {e}",
                            layer.layer_index
                        );
                        record_rejected_fragment(&telemetry, &fragment.session_id, &error);
                    }
                }
            }
            if applied_any {
                session.last_used = unix_now();
                record_applied_fragment(&telemetry, &fragment.session_id);
            }
        }
    });
}
