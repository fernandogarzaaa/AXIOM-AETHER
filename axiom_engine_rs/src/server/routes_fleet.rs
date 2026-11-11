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
        key_configured: state.fleet_key.is_some(),
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
    let memory = crate::heal_memory::HealMemory::load_shared(path);
    // Wrap the export with tamper-evident provenance (full SHA-256 + an HMAC
    // when a fleet key is configured). A peer only merges an export whose HMAC
    // verifies under the shared key, so an export from a keyless node is
    // readable but will not merge anywhere.
    let signed =
        crate::provenance::sign_export(&memory.to_json(), state.fleet_key.as_deref());
    Json(signed).into_response()
}

/// The shared fleet secret from `AXIOM_FLEET_KEY`, read once at startup into
/// [`AppState::fleet_key`]. `None` → exports are hashed but not signed, and
/// the merge routes refuse everything.
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

/// Extra `Host` names for the fleet merge routes, from
/// `AXIOM_MERGE_ALLOWED_HOSTS` (comma-separated). Empty by default.
fn merge_allowed_hosts_from_env() -> Vec<String> {
    std::env::var("AXIOM_MERGE_ALLOWED_HOSTS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(normalize_host_name)
                .filter(|h| !h.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Lowercase a host name and drop surrounding whitespace and one trailing dot
/// (`Example.COM.` and `example.com` name the same host).
fn normalize_host_name(raw: &str) -> String {
    let h = raw.trim().to_ascii_lowercase();
    h.strip_suffix('.').unwrap_or(&h).to_string()
}

/// Split a `Host` header (or URI authority) into its host part: `[::1]:8080`
/// → `::1`, `10.0.0.5:8080` → `10.0.0.5`, `node-a` → `node-a`. `None` when the
/// value is not a well-formed host with an optional numeric port.
fn host_without_port(authority: &str) -> Option<String> {
    let authority = authority.trim();
    if authority.is_empty() || authority.contains('@') || authority.contains('/') {
        return None;
    }
    let (host, port) = if let Some(rest) = authority.strip_prefix('[') {
        // Bracketed IPv6 literal, optionally followed by `:port`.
        let (inner, after) = rest.split_once(']')?;
        let port = match after {
            "" => None,
            p => Some(p.strip_prefix(':')?),
        };
        (inner, port)
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (authority, None),
        }
    };
    if let Some(port) = port {
        if port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
    }
    let host = normalize_host_name(host);
    if host.is_empty() || host.contains(':') && host.parse::<std::net::Ipv6Addr>().is_err() {
        return None;
    }
    Some(host)
}

/// Is `authority` (a `Host` header value) acceptable on the fleet merge
/// routes?
///
/// The check exists to stop DNS rebinding: a web page on `evil.example` whose
/// name is re-pointed at this node becomes same-origin with it in the browser
/// and can then send any request it likes, but every such request still
/// carries `Host: evil.example`. So a name is accepted only when it cannot be
/// attacker-chosen DNS:
/// * `localhost`;
/// * an IP literal (v4, or bracketed v6): rebinding needs a DNS name, and
///   fleet peers address each other by pod or node IP;
/// * a name the operator listed in `AXIOM_MERGE_ALLOWED_HOSTS`.
fn merge_host_allowed(authority: &str, extra_hosts: &[String]) -> bool {
    let Some(host) = host_without_port(authority) else {
        return false;
    };
    host == "localhost"
        || host.parse::<std::net::IpAddr>().is_ok()
        || extra_hosts.contains(&host)
}

/// Is this `Origin` header acceptable on the fleet merge routes? Only an
/// exact entry of the operator's allowlist is. `null` (sandboxed frames,
/// `file:` pages, redirects) is never accepted, even if listed.
fn merge_origin_allowed(origin: &HeaderValue, allowed: &[HeaderValue]) -> bool {
    origin.as_bytes() != b"null" && allowed.iter().any(|a| a == origin)
}

fn merge_refusal(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

/// Guard for the fleet merge routes (`POST /v1/immunity/merge`,
/// `POST /v1/patches/merge`). What these routes store is later acted on with
/// the user's authority: a merged patch candidate is written into a source
/// tree and executed by `axiom solve`'s verify run, and a merged immunity
/// record is injected into upstream prompts. So, independently of
/// `AXIOM_API_KEY` (which is unset by default), a merge is refused unless:
///
/// 1. the request names this node by an allowlisted `Host`
///    ([`merge_host_allowed`]) — 403 otherwise;
/// 2. it carries no `Origin`, or an allowlisted one
///    ([`merge_origin_allowed`]). Browsers attach `Origin` to every
///    cross-origin POST, including the `text/plain` "simple" requests that
///    skip CORS preflight, so this is what stops a web page — 403 otherwise;
/// 3. this node has a fleet key — 401 otherwise. The handlers then require the
///    export's HMAC to verify under that key.
async fn require_fleet_merge_trust(
    State(state): State<AppState>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    let mut host_headers = req.headers().get_all(header::HOST).iter();
    let host = match (host_headers.next(), host_headers.next()) {
        // Exactly one Host header.
        (Some(value), None) => value.to_str().ok().map(str::to_string),
        // HTTP/2 carries the authority in the request target instead.
        (None, _) => req.uri().authority().map(|a| a.as_str().to_string()),
        // Two Host headers: ambiguous, refuse.
        (Some(_), Some(_)) => None,
    };
    if !host
        .as_deref()
        .is_some_and(|h| merge_host_allowed(h, &state.merge_allowed_hosts))
    {
        return merge_refusal(
            StatusCode::FORBIDDEN,
            "fleet merge refused: the Host header must be localhost, an IP literal, \
             or a name listed in AXIOM_MERGE_ALLOWED_HOSTS",
        );
    }
    let mut origins = req.headers().get_all(header::ORIGIN).iter();
    if let Some(origin) = origins.next() {
        if origins.next().is_some()
            || !merge_origin_allowed(origin, &state.merge_allowed_origins)
        {
            return merge_refusal(
                StatusCode::FORBIDDEN,
                "fleet merge refused: cross-origin requests are accepted only from an \
                 origin listed in AXIOM_CORS_ORIGINS",
            );
        }
    }
    if state.fleet_key.is_none() {
        return merge_refusal(
            StatusCode::UNAUTHORIZED,
            "fleet merge refused: this node has no AXIOM_FLEET_KEY, so no export can be \
             authenticated; set the same key on every fleet member",
        );
    }
    next.run(req).await
}

/// `POST /v1/immunity/merge` — fold a peer's exported heal memory into this
/// node's (swarm immunity). Body: the peer's `GET /v1/immunity` payload, a
/// [`crate::provenance::SignedExport`] whose HMAC must verify under this
/// node's fleet key. An unsigned export, a raw heal-memory JSON, and any
/// request to a node with no fleet key are refused with 401 (the route also
/// sits behind [`require_fleet_merge_trust`]). Returns the merge report.
/// Local learning is never weakened: dirs are unioned and tension histories
/// are count-weighted.
async fn post_immunity_merge(State(state): State<AppState>, body: String) -> Response {
    let Some(path) = state.heal_memory_path.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "heal memory not configured on this node"})),
        )
            .into_response();
    };
    // Authenticate before touching the store. Merged records end up in
    // upstream prompts as "established fixes", so nothing unauthenticated may
    // be folded in: no key → refuse; raw (unwrapped) payload → refuse; export
    // without a valid HMAC under the fleet key → refuse.
    let Ok(export) = serde_json::from_str::<crate::provenance::SignedExport>(&body) else {
        return merge_refusal(
            StatusCode::UNAUTHORIZED,
            "expected a signed immunity export (the GET /v1/immunity payload); \
             raw heal-memory payloads are not accepted",
        );
    };
    let payload = match crate::provenance::verify_export_authenticated(
        &export,
        state.fleet_key.as_deref(),
    ) {
        Ok(verified) => verified.to_string(),
        Err(e) => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": format!("provenance rejected: {e}")})),
            )
                .into_response();
        }
    };
    let mut memory = crate::heal_memory::HealMemory::load(path);
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

/// Spawn the background task that consumes inbound DWE fragments, verifies
/// HMAC with current/previous fleet-key rotation, enforces per-session sequence
/// ordering, and merges accepted layer deltas into session state.
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
                let total = checked_shape_product(&layer.shape);
                if total != Some(layer.values.len()) {
                    let error = format!("layer {} shape/value mismatch", layer.layer_index);
                    eprintln!("[dwe] {error}");
                    record_rejected_fragment(&telemetry, &fragment.session_id, &error);
                    continue;
                }
                let total = layer.values.len();
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
