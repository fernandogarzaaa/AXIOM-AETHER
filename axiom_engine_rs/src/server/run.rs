/// Start the HTTP server and block until it is stopped.
pub async fn run_server(
    host: &str,
    port: u16,
    config: AxiomConfig,
    checkpoint_path: &str,
    device: Device,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Fail closed before any heavy init: refuse to expose the unauthenticated
    // data plane (which forwards with the operator's upstream API key) on a
    // non-loopback address unless AXIOM_API_KEY is set or the operator opts
    // out explicitly with AXIOM_ALLOW_OPEN=1.
    let api_key_configured = std::env::var("AXIOM_API_KEY")
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);
    let allow_open = env_flag_enabled(std::env::var("AXIOM_ALLOW_OPEN").ok().as_deref());
    check_bind_policy(host, api_key_configured, allow_open)?;

    println!("[*] Initializing system sanity check prior to binding network sockets...");

    // BPE tokenizer wiring: when AXIOM_TOKENIZER points at a tokenizer.json the
    // pipeline encodes with the real BPE vocab; otherwise it falls back to the
    // legacy hash tokenizer. with_checkpoint_and_options tolerates a missing
    // checkpoint (warns + random init), so the DEFAULT_CHECKPOINT_PATH special
    // case is no longer needed.
    let runtime = crate::inference::InferenceRuntimeOptions {
        tokenizer_path: std::env::var("AXIOM_TOKENIZER")
            .ok()
            .filter(|p| !p.trim().is_empty()),
        ..Default::default()
    };
    let pipeline_config = config.clone();
    let pipeline_device = device.clone();
    let pipeline_checkpoint = checkpoint_path.to_string();
    let pipeline = spawn_blocking(move || {
        InferencePipeline::with_checkpoint_and_options(
            pipeline_config,
            pipeline_device,
            &pipeline_checkpoint,
            runtime,
        )
    })
    .await
    .map_err(|e| format!("pipeline assembly task failed: {e}"))?
    .map_err(|e| format!("failed to assemble inference pipeline: {e}"))?;

    println!(
        "[+] Sanity check passed. safetensors matrix dimensions align perfectly with {} layers.",
        config.n_layers
    );

    let model_id = "axiom-ttt-v1".to_string();
    let claude_backend = ClaudeBackend::from_env();
    if let Some(ref backend) = claude_backend {
        println!(
            "[+] Claude backend active — generation routed to model={} \
             (TTT adapt is a no-op in this mode)",
            backend.model()
        );
    }
    let compressor_config = CompressorConfig::from_env();
    let anthropic_forwarder = if compressor_config.enabled {
        AnthropicForwarder::from_env()
    } else {
        None
    };
    // Responses passthrough is useful independently of context compression.
    // `from_env` supports client-auth passthrough and holds no key by default.
    let openai_forwarder = OpenAiForwarder::from_env();
    let swarm_router = if compressor_config.enabled {
        SwarmRouter::from_env()
    } else {
        None
    };
    if compressor_config.enabled {
        match anthropic_forwarder.as_ref() {
            Some(fwd) => {
                println!(
                    "[+] Active-compression mode ON — heavy messages (>={} tokens) \
                     will be absorbed locally via TTT, then forwarded with a dense \
                     fingerprint to Anthropic (top_k={})",
                    compressor_config.heavy_message_threshold_tokens,
                    compressor_config.recall_top_k,
                );
                if fwd.has_own_key() {
                    println!(
                        "[+] Upstream auth: proxy-owned ANTHROPIC_API_KEY (injected as \
                         x-api-key when the client sends none)."
                    );
                } else {
                    println!(
                        "[+] Upstream auth: PASSTHROUGH — no proxy key set; the client's \
                         own Authorization/x-api-key headers are relayed upstream. This is \
                         the correct mode for a Claude subscription (OAuth via Claude Code)."
                    );
                }
            }
            None => println!(
                "[!] Active-compression enabled but the forwarder failed to construct — \
                 the compression path will be skipped and requests will fall back \
                 to the local pipeline"
            ),
        }
        match openai_forwarder.as_ref() {
            Some(fwd) => {
                println!(
                    "[+] OpenAI/Codex compression bridge ON — /v1/chat/completions \
                     can absorb heavy messages locally, then forward a lean payload upstream."
                );
                if fwd.has_own_key() {
                    println!(
                        "[+] OpenAI upstream auth: proxy-owned OPENAI_API_KEY injected as \
                         bearer auth when the client sends none."
                    );
                } else {
                    println!(
                        "[+] OpenAI upstream auth: PASSTHROUGH — no proxy key set; the client's \
                         own Authorization header is relayed upstream."
                    );
                }
            }
            None => println!(
                "[!] Active-compression enabled but the OpenAI forwarder failed to construct — \
                 /v1/chat/completions will fall back to the local pipeline"
            ),
        }
        if let Some(router) = swarm_router.as_ref() {
            println!(
                "[+] Local swarm router ON — Ollama base={} candidates={:?} num_ctx={} timeout_ms={}",
                router.config().base_url,
                router.config().model_candidates,
                router.config().num_ctx,
                router.config().timeout_ms
            );
        } else {
            println!(
                "[+] Local swarm router OFF — set AXIOM_SWARM_LOCAL=1 to route compressed payloads to Ollama before cloud fallback."
            );
        }
    }
    // Persistent vibe memory (automatic EMA merge on session drop/clear/
    // shutdown). Enabled by default; set AXIOM_VIBE=0 to disable all
    // master-vibe persistence in the proxy.
    let vibe_enabled = std::env::var("AXIOM_VIBE")
        .map(|v| v != "0")
        .unwrap_or(true);
    let master_vibe = if vibe_enabled {
        let v = MasterVibe::from_env(config.n_layers, config.d_model, &device);
        println!(
            "[+] Persistent vibe memory ON — sessions EMA-merge into {} on drop/clear/shutdown \
             (decay={}). Set AXIOM_VIBE=0 to disable.",
            v.path().display(),
            v.decay()
        );
        Some(v)
    } else {
        println!("[+] Persistent vibe memory OFF (AXIOM_VIBE=0)");
        None
    };

    // Swarm immunity: serve/merge the same heal memory `axiom run` learns into
    // (AXIOM_HEAL_MEMORY overrides the path, 0/off disables the endpoints).
    let heal_memory_path = crate::heal_memory::HealMemory::default_path();
    if let Some(p) = heal_memory_path.as_ref() {
        println!(
            "[+] Swarm immunity ON — /v1/immunity serves and merges {}",
            p.display()
        );
    }

    let state = AppState::new(pipeline, model_id)
        .with_claude_backend(claude_backend)
        .with_anthropic_forwarder(anthropic_forwarder)
        .with_openai_forwarder(openai_forwarder)
        .with_swarm_router(swarm_router)
        .with_compressor_config(compressor_config)
        .with_master_vibe(master_vibe)
        .with_heal_memory_path(heal_memory_path);
    // S6 (CVM cost stack): actuarial keepalive, AXIOM_KEEPALIVE=1 opt-in
    // only (KeepaliveManager::from_env prints the required security boot
    // banner when enabled; a no-op manager otherwise).
    let keepalive_awareness = state.awareness.clone();
    let state = state.with_keepalive_manager(crate::keepalive::KeepaliveManager::from_env(
        keepalive_awareness,
    ));
    // AXIOM_BACKEND=router/openai/opendrop: assemble the multi-provider router
    // from the live pipeline + Claude (ANTHROPIC_API_KEY) + OpenAI creds. None
    // unless one of those modes is set. Built on a blocking thread because it
    // constructs `reqwest::blocking` clients, which panic if created inside the
    // Tokio runtime.
    let pipeline_for_router = state.pipeline.clone();
    let router = tokio::task::spawn_blocking(move || {
        crate::backend_live::router_from_env(pipeline_for_router)
    })
    .await
    .expect("router_from_env build task panicked");
    if router.is_some() {
        println!(
            "[+] AXIOM_BACKEND router mode — generation routed across configured providers \
             (GPT/Claude/local), with failover"
        );
    }
    let state = state.with_router(router);

    // Remote MCP transport (AXIOM_MCP_HTTP=1): expose Axiom's MCP tools over HTTP
    // at /mcp so the ChatGPT connector and Claude remote connectors can attach.
    // Builds a dedicated MCP context (own pipeline/vibe/memory) like the stdio
    // server; a build failure disables /mcp rather than taking the server down.
    let mcp_token = std::env::var("AXIOM_MCP_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty());
    let mcp = if std::env::var("AXIOM_MCP_HTTP")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        match crate::mcp_stdio::build_context(
            config.clone(),
            device.clone(),
            checkpoint_path.to_string(),
        )
        .await
        {
            Ok(c) => {
                println!(
                    "[+] AXIOM_MCP_HTTP=1 — MCP tools served at POST /mcp (remote connectors)"
                );
                if mcp_token.is_some() {
                    println!("[+] /mcp requires a bearer token (AXIOM_MCP_TOKEN set)");
                } else {
                    eprintln!(
                        "[axiom] WARNING: /mcp is UNAUTHENTICATED. Anyone who can reach this \
                         endpoint can drive Axiom's tools. Set AXIOM_MCP_TOKEN before exposing \
                         it publicly (e.g. through a tunnel)."
                    );
                }
                Some(c)
            }
            Err(e) => {
                eprintln!("[axiom] /mcp disabled: failed to build MCP context: {e}");
                None
            }
        }
    } else {
        None
    };
    // Optional data-plane auth: when AXIOM_API_KEY is set, every route except
    // ops (/healthz,/readyz,/metrics) and /mcp requires `X-Axiom-Key: <key>`.
    // Trim before storing: a key sourced from a secret file often carries a
    // trailing newline, and `require_api_key` compares against the trimmed
    // presented header — so an untrimmed stored key would 401 every request.
    let api_key = std::env::var("AXIOM_API_KEY")
        .ok()
        .map(|k| k.trim().to_owned())
        .filter(|k| !k.is_empty());
    if api_key.is_some() {
        println!("[+] Data plane requires X-Axiom-Key header (AXIOM_API_KEY set)");
    } else if !bind_host_is_loopback(host) {
        // Only reachable with AXIOM_ALLOW_OPEN=1 (check_bind_policy above).
        eprintln!(
            "[axiom] WARNING: data-plane endpoints are UNAUTHENTICATED and bound to {host} \
             (AXIOM_ALLOW_OPEN=1). Anyone who can reach this host can drive Axiom and spend \
             the configured upstream API key. Set AXIOM_API_KEY before exposing it beyond a \
             trusted network."
        );
    }
    // Optional process-execution capability: POST /v1/hypervisor/jit_run runs
    // an arbitrary caller-supplied command and can overwrite a source file on
    // disk. Off by default — an operator must opt in explicitly. See
    // docs/SECURITY-AUDIT.md for the threat model before enabling this on a
    // network-reachable host.
    let jit_exec_enabled = std::env::var("AXIOM_ENABLE_JIT_EXEC")
        .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "on" | "ON"))
        .unwrap_or(false);
    if jit_exec_enabled {
        println!(
            "[+] Hypervisor jit_run enabled (AXIOM_ENABLE_JIT_EXEC=1): \
             POST /v1/hypervisor/jit_run will execute caller-supplied commands."
        );
        if api_key.is_none() {
            eprintln!(
                "[axiom] WARNING: process execution is enabled with NO data-plane auth \
                 (AXIOM_API_KEY unset) — anyone who can reach this host can run arbitrary \
                 commands as this process. Set AXIOM_API_KEY before exposing it."
            );
        }
    }
    let state = state
        .with_mcp(mcp)
        .with_mcp_token(mcp_token)
        .with_api_key(api_key)
        .with_jit_exec_enabled(jit_exec_enabled);
    if state.compression_active() {
        match state.hydrate_compression_cache().await {
            Ok(0) => println!(
                "[+] Compression cache: no persisted adapted sessions found at {}",
                compression_cache_path().display()
            ),
            Ok(n) => println!(
                "[+] Compression cache: hydrated {n} adapted session(s) from {}",
                compression_cache_path().display()
            ),
            Err(e) => eprintln!("[!] Compression cache hydration skipped: {e}"),
        }
    }
    // Differential Weight Exchange inbound listener: when AXIOM_DWE_LISTEN is
    // set (host:port), accept peer fragments over TCP and merge their layer
    // deltas into the named session — the network mirror of /v1/cluster/sync.
    if let Ok(listen_addr) = std::env::var("AXIOM_DWE_LISTEN") {
        let listen_addr = listen_addr.trim().to_string();
        // Fail closed: never accept peer weight deltas without a fleet key to
        // authenticate them. Configured-but-keyless ⇒ skip the listener.
        let verify_secret = if listen_addr.is_empty() {
            None
        } else {
            match fleet_key() {
                Some(secret) => Some(secret),
                None => {
                    eprintln!(
                        "[dwe] AXIOM_DWE_LISTEN is set but AXIOM_FLEET_KEY is not — refusing to \
                         start an unauthenticated weight-fragment listener"
                    );
                    None
                }
            }
        };
        if let Some(verify_secret) = verify_secret {
            let telemetry = state.dwe_bus.telemetry_handle();
            let (in_tx, in_rx) = tokio::sync::mpsc::channel::<crate::dwe::DweFragment>(64);
            start_dwe_apply_loop(
                state.clone(),
                in_rx,
                verify_secret,
                fleet_key_prev(),
                telemetry.clone(),
            );
            let listener_addr = listen_addr.clone();
            tokio::spawn(async move {
                if let Err(e) =
                    crate::dwe::start_dwe_listener(&listener_addr, in_tx, telemetry).await
                {
                    eprintln!("[dwe] listener on {listener_addr} exited: {e}");
                }
            });
            println!("[+] DWE inbound listener ON — accepting peer weight deltas at {listen_addr}");
        }
    }

    // Keep a handle for the graceful-shutdown flush (AppState is cheaply Clone).
    let shutdown_state = state.clone();
    let app = create_router(state);

    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    println!("[+] Axiom-TTT server listening on http://{host}:{port}");
    println!("[+] OpenAI- and Anthropic-compatible API endpoints:");
    println!("      GET  /metrics");
    println!("      GET  /v1/models");
    println!("      POST /v1/completions");
    println!("      POST /v1/chat/completions         (stream:true for SSE)");
    println!("      POST /v1/responses                (native OpenAI passthrough)");
    println!(
        "[+] Responses input compression: {} (opt out with AXIOM_RESPONSES_COMPRESS=0)",
        if responses_compression_enabled() {
            "ON (default)"
        } else {
            "OFF"
        }
    );
    println!("      POST /v1/messages                 (Anthropic Messages API)");
    println!("      POST /v1/cluster/sync            (distributed delta merge)");
    println!("      POST /v1/sessions                 (create TTT session)");
    println!("      POST /v1/adapt                    (in-place TTT adaptation)");
    println!("      GET  /v1/sessions/{{id}}/checkpoint");
    println!("      PUT  /v1/sessions/{{id}}/checkpoint");

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            // Wait for Ctrl-C / SIGTERM, then flush all live sessions into the
            // persistent master vibe so accumulated structure survives restart.
            wait_for_shutdown_signal().await;
            eprintln!("[+] shutdown signal received — flushing vibe memory");
            shutdown_state.flush_all_sessions_to_vibe().await;
        })
        .await?;

    // The pipeline owns a `reqwest::blocking::Client` whose internal runtime
    // must not be dropped from within this async context. We are intentionally
    // terminating, so exit the process directly and let the OS reclaim memory
    // rather than running destructors on the async runtime.
    std::process::exit(0);
}

#[cfg(unix)]
async fn wait_for_shutdown_signal() {
    match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(mut sigterm) => {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = sigterm.recv() => {}
            }
        }
        Err(e) => {
            eprintln!("[!] failed to install SIGTERM handler: {e}; waiting for Ctrl-C only");
            let _ = tokio::signal::ctrl_c().await;
        }
    }
}

#[cfg(not(unix))]
async fn wait_for_shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

// ---------------------------------------------------------------------------

/// True when `host` names a loopback interface (`localhost`, `127.0.0.0/8`,
/// `::1`, optionally in `[::1]` bracket form). Anything else, including
/// `0.0.0.0`, `::`, LAN addresses and other hostnames, counts as exposed.
fn bind_host_is_loopback(host: &str) -> bool {
    let h = host.trim();
    let h = h
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(h);
    if h.eq_ignore_ascii_case("localhost") {
        return true;
    }
    h.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// Parse a boolean opt-in env value (`1`, `true`, `yes`, `on`; case-insensitive).
fn env_flag_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Bind-safety policy for the HTTP server.
///
/// Loopback binds are always allowed. A non-loopback bind is allowed only when
/// `AXIOM_API_KEY` is configured (data-plane routes then require
/// `X-Axiom-Key`) or the operator sets `AXIOM_ALLOW_OPEN=1` explicitly (e.g.
/// inside a container whose port publishing is the real boundary).
fn check_bind_policy(
    host: &str,
    api_key_configured: bool,
    allow_open: bool,
) -> std::result::Result<(), String> {
    if bind_host_is_loopback(host) || api_key_configured || allow_open {
        return Ok(());
    }
    Err(format!(
        "refusing to bind the unauthenticated data plane to non-loopback address '{host}': \
         anyone who can reach it could use the configured upstream API key. Set AXIOM_API_KEY \
         (clients then send X-Axiom-Key), bind to 127.0.0.1, or set AXIOM_ALLOW_OPEN=1 to \
         accept the risk explicitly."
    ))
}

#[cfg(test)]
mod bind_policy_tests {
    use super::{bind_host_is_loopback, check_bind_policy, env_flag_enabled};

    #[test]
    fn loopback_hosts_are_recognised() {
        for host in [
            "127.0.0.1",
            "127.1.2.3",
            "localhost",
            "LOCALHOST",
            "::1",
            "[::1]",
            " 127.0.0.1 ",
        ] {
            assert!(bind_host_is_loopback(host), "{host} should be loopback");
        }
    }

    #[test]
    fn exposed_hosts_are_not_loopback() {
        for host in [
            "0.0.0.0",
            "::",
            "[::]",
            "192.168.1.10",
            "10.0.0.5",
            "example.com",
            "",
            "localhost.evil.com",
            "128.0.0.1",
        ] {
            assert!(
                !bind_host_is_loopback(host),
                "{host} should not be loopback"
            );
        }
    }

    #[test]
    fn loopback_bind_is_allowed_without_key() {
        assert!(check_bind_policy("127.0.0.1", false, false).is_ok());
        assert!(check_bind_policy("localhost", false, false).is_ok());
        assert!(check_bind_policy("::1", false, false).is_ok());
    }

    #[test]
    fn public_bind_without_key_is_refused() {
        let err = check_bind_policy("0.0.0.0", false, false).unwrap_err();
        assert!(err.contains("0.0.0.0"));
        assert!(err.contains("AXIOM_API_KEY"));
        assert!(err.contains("AXIOM_ALLOW_OPEN"));
        assert!(check_bind_policy("192.168.1.10", false, false).is_err());
        assert!(check_bind_policy("::", false, false).is_err());
    }

    #[test]
    fn public_bind_is_allowed_with_key_or_explicit_opt_out() {
        assert!(check_bind_policy("0.0.0.0", true, false).is_ok());
        assert!(check_bind_policy("0.0.0.0", false, true).is_ok());
        assert!(check_bind_policy("0.0.0.0", true, true).is_ok());
    }

    #[test]
    fn allow_open_flag_parsing() {
        for v in ["1", "true", "TRUE", "yes", "on", " On "] {
            assert!(env_flag_enabled(Some(v)), "{v} should enable");
        }
        for v in ["0", "false", "no", "off", "", "2", "enabled"] {
            assert!(!env_flag_enabled(Some(v)), "{v} should not enable");
        }
        assert!(!env_flag_enabled(None));
    }
}
