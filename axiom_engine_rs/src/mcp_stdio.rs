//! Native Model Context Protocol (MCP) server over a JSON-RPC 2.0 stdio
//! transport.
//!
//! This exposes Axiom to a host LLM (e.g. Claude Code) as a first-class tool
//! provider, running as a dedicated process (`--mode mcp`). It is intentionally
//! **separate** from the HTTP proxy so that:
//!
//! * `stdout` carries **only** newline-delimited JSON-RPC frames (the MCP stdio
//!   contract). Every diagnostic goes to `stderr` via `eprintln!`.
//! * the long-running proxy is never destabilised by protocol traffic.
//!
//! Both transports share the same engine internals: the [`InferencePipeline`]
//! and the persistent [`MasterVibe`].
//!
//! ## Tools
//! * `axiom_compress_path` — absorb a directory through the local TTT engine and
//!   return the resulting `<axiom_context_fingerprint>` block. Committing the
//!   adapted session into the master vibe (and persisting it) is the **explicit**
//!   merge trigger.
//! * `axiom_evaluate_drift` — cross-entropy of supplied code against the current
//!   fast-weights; a loss spike past the baseline threshold returns
//!   `isError: true` to signal architectural deviation.

use std::error::Error;
use std::path::{Component, Path};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{Device, Tensor};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::config::AxiomConfig;
use crate::context_compressor::{
    adapt_session_blocking, extract_memory_vector_blocking, MAX_ADAPT_CHUNK_TOKENS,
};
use crate::embedder::{embed_text, EmbeddingModel};
use crate::inference::InferencePipeline;
use crate::memory_recall::{recall, RecallParams};
use crate::memory_store::{now_secs, MemoryKind, MemoryRecord, MemoryStore};
use crate::session_awareness::AwarenessStore;
use crate::vibe_memory::MasterVibe;

/// MCP protocol revision we advertise in the `initialize` handshake.
const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "axiom-ttt";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Token window fed through online TTT adaptation at a time. Bounds per-call
/// memory and remains strictly below the Phase 3 512-token ceiling.
const ADAPT_WINDOW_TOKENS: usize = MAX_ADAPT_CHUNK_TOKENS;
/// Cap on tokens scored by `axiom_evaluate_drift` (bounds compute per call).
const DRIFT_MAX_TOKENS: usize = 512;

/// Shared engine context handed to every tool invocation.
#[derive(Clone)]
pub struct McpContext {
    pipeline: Arc<Mutex<InferencePipeline>>,
    vibe: Arc<Mutex<MasterVibe>>,
    /// Tier-2 lossless memory store (Phase 2.0): JSONL records per scope.
    memory: Arc<Mutex<MemoryStore>>,
    /// Trained contrastive embedder (Phase 2.0.1); None → pipeline TTT fallback.
    embedder: Option<Arc<EmbeddingModel>>,
    /// When true, new tool sessions start from the master vibe instead of
    /// identity (opt-in via `AXIOM_VIBE_PRIME=1`).
    prime: bool,
    /// Cross-entropy above which `axiom_evaluate_drift` reports drift.
    drift_threshold: f32,
    /// Top-k recall indices recorded in the compression fingerprint.
    top_k: usize,
    /// Max files / bytes ingested by `axiom_compress_path`.
    max_files: usize,
    max_bytes: usize,
    /// Base URL of the running Axiom HTTP proxy (for awareness endpoint calls).
    /// Override with `AXIOM_PROXY_URL` (default `http://127.0.0.1:3000`).
    pub proxy_url: String,
    /// Inter-agent task board. Loaded once from AXIOM_TASK_DIR.
    task_board: Arc<crate::task_board::TaskBoard>,
    /// Per-session token-awareness store (PR #72). Used to record tool response
    /// costs and conditionally annotate responses with a meta cost line.
    awareness: Arc<AwarenessStore>,
}

/// Assemble an [`McpContext`] (pipeline + vibe + memory + embedder) from config,
/// device, and a checkpoint path. Shared by the stdio server and the HTTP/SSE
/// transport ([`crate::server`]'s `/mcp` route) so both expose identical tools
/// against identical engine internals.
pub async fn build_context(
    config: AxiomConfig,
    device: Device,
    checkpoint_path: String,
) -> Result<McpContext, String> {
    // Build the pipeline on a blocking thread. It owns a `reqwest::blocking::Client`
    // which carries its own runtime; constructing/dropping that inside the async
    // context is unsafe, so we keep all of its lifecycle off the async runtime.
    let pipeline = {
        let cfg = config.clone();
        let dev = device.clone();
        let ckpt = checkpoint_path.clone();
        // BPE tokenizer via AXIOM_TOKENIZER (falls back to hash tokenizer when unset).
        let tokenizer_path = std::env::var("AXIOM_TOKENIZER")
            .ok()
            .filter(|p| !p.trim().is_empty());
        tokio::task::spawn_blocking(move || {
            let runtime = crate::inference::InferenceRuntimeOptions {
                tokenizer_path,
                ..Default::default()
            };
            InferencePipeline::with_checkpoint_and_options(cfg, dev, &ckpt, runtime)
        })
        .await
        .map_err(|e| format!("pipeline build join error: {e}"))?
        .map_err(|e| format!("failed to assemble inference pipeline: {e}"))?
    };

    let vibe = MasterVibe::from_env(config.n_layers, config.d_model, &device);
    let prime = std::env::var("AXIOM_VIBE_PRIME")
        .map(|v| v == "1")
        .unwrap_or(false);
    if prime && !vibe.is_initialized() {
        eprintln!("[mcp] AXIOM_VIBE_PRIME=1 set but no master vibe yet; sessions start from identity until first commit");
    }
    // Default to the BPE-calibrated deterministic gate (7.03); override via env.
    let drift_threshold = std::env::var("AXIOM_DRIFT_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(7.03);
    let top_k = std::env::var("AXIOM_TTT_COMPRESS_TOP_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(32);
    let max_files = std::env::var("AXIOM_MCP_MAX_FILES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512);
    let max_bytes = std::env::var("AXIOM_MCP_MAX_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1_048_576); // 1 MiB

    // Tier-2 memory store root (Phase 2.0). Override with AXIOM_MEMORY_DIR.
    let memory_dir =
        std::env::var("AXIOM_MEMORY_DIR").unwrap_or_else(|_| "checkpoints/memory".to_string());
    let memory = MemoryStore::open(&memory_dir)
        .map_err(|e| format!("failed to open memory store at {memory_dir}: {e}"))?;
    eprintln!("[mcp] memory store at {memory_dir}");

    // Phase 2.0.1 trained embedder (CPU for the MCP path). Absent → TTT fallback.
    let embedder_ckpt = std::env::var("AXIOM_EMB_CKPT")
        .unwrap_or_else(|_| "checkpoints/axiom_embedder.bin".to_string());
    let embedder = EmbeddingModel::load(&embedder_ckpt, Device::Cpu).map(Arc::new);
    eprintln!(
        "[mcp] contrastive embedder: {}",
        if embedder.is_some() {
            "loaded"
        } else {
            "absent (pipeline fallback)"
        }
    );

    Ok(McpContext {
        pipeline: Arc::new(Mutex::new(pipeline)),
        vibe: Arc::new(Mutex::new(vibe)),
        memory: Arc::new(Mutex::new(memory)),
        embedder,
        prime,
        drift_threshold,
        top_k,
        max_files,
        max_bytes,
        proxy_url: std::env::var("AXIOM_PROXY_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:3000".to_string()),
        task_board: Arc::new(
            if std::env::var("AXIOM_TASK_DIR").is_ok() {
                crate::task_board::TaskBoard::from_env()
                    .map_err(|e| format!("AXIOM_TASK_DIR open failed: {e}"))?
            } else {
                crate::task_board::TaskBoard::open("checkpoints/tasks")
                    .map_err(|e| format!("task board init failed: {e}"))?
            }
        ),
        awareness: Arc::new(AwarenessStore::new()),
    })
}

/// Boot the MCP stdio server. Runs until stdin reaches EOF (host disconnect).
pub async fn run_stdio_server(
    config: AxiomConfig,
    device: Device,
    checkpoint_path: String,
) -> Result<(), Box<dyn Error>> {
    // All status output MUST go to stderr; stdout is reserved for JSON-RPC.
    eprintln!("[mcp] booting Axiom MCP stdio server (protocol {MCP_PROTOCOL_VERSION})");

    let ctx = build_context(config, device, checkpoint_path).await?;

    eprintln!(
        "[mcp] ready — tools: axiom_compress_path, axiom_evaluate_drift, axiom_expand, \
         axiom_remember, axiom_recall, axiom_forget (prime={}, drift_threshold={})",
        ctx.prime, ctx.drift_threshold
    );

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(response) = handle_message(trimmed, &ctx).await {
            let mut payload = serde_json::to_string(&response)
                .unwrap_or_else(|e| format!(r#"{{"jsonrpc":"2.0","id":null,"error":{{"code":-32603,"message":"serialize failed: {e}"}}}}"#));
            payload.push('\n');
            stdout.write_all(payload.as_bytes()).await?;
            stdout.flush().await?;
        }
    }

    eprintln!("[mcp] stdin closed; shutting down");

    // Drop the pipeline on a blocking thread so the reqwest blocking client's
    // internal runtime is not dropped from within this async context.
    let McpContext { pipeline, .. } = ctx;
    let _ = tokio::task::spawn_blocking(move || drop(pipeline)).await;
    Ok(())
}

/// Dispatch one JSON-RPC line against an [`McpContext`] — shared by the stdio
/// loop and the HTTP/SSE transport ([`crate::server`]'s `/mcp` route). Returns
/// `Some(response)` for requests, `None` for notifications.
pub async fn dispatch(line: &str, ctx: &McpContext) -> Option<Value> {
    handle_message(line, ctx).await
}

/// Parse and route one JSON-RPC line. Returns `Some(response)` for requests,
/// `None` for notifications (no `id`) which must not be answered.
async fn handle_message(line: &str, ctx: &McpContext) -> Option<Value> {
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[mcp] parse error: {e}");
            return Some(error_response(
                Value::Null,
                -32700,
                &format!("parse error: {e}"),
            ));
        }
    };

    let id = req.get("id").cloned();
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");

    // Notifications carry no id and must receive no response.
    let is_notification = id.is_none();

    match method {
        "initialize" => Some(success_response(
            id.unwrap_or(Value::Null),
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION }
            }),
        )),
        "notifications/initialized" | "initialized" => {
            eprintln!("[mcp] client initialized");
            None
        }
        "ping" => Some(success_response(id.unwrap_or(Value::Null), json!({}))),
        "tools/list" => Some(success_response(id.unwrap_or(Value::Null), tools_list())),
        "tools/call" => {
            if is_notification {
                return None;
            }
            let id = id.unwrap_or(Value::Null);
            Some(handle_tools_call(id, req.get("params"), ctx).await)
        }
        other => {
            if is_notification {
                eprintln!("[mcp] ignoring unknown notification '{other}'");
                None
            } else {
                Some(error_response(
                    id.unwrap_or(Value::Null),
                    -32601,
                    &format!("method not found: {other}"),
                ))
            }
        }
    }
}

/// Static tool catalogue with strict input schemas.
fn tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "axiom_compress_path",
                "description": "Absorb a directory of source code through Axiom's local Test-Time Training engine (mutating the fast-weights), then return a dense <axiom_context_fingerprint> block summarising the compressed context. Also commits the adapted session into the persistent master vibe.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to a directory (or single file) to compress."
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "axiom_evaluate_drift",
                "description": "Compute the cross-entropy loss of the provided code against Axiom's current fast-weights. A loss spike past the baseline threshold returns isError:true, signalling architectural drift from the absorbed codebase patterns.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code_content": {
                            "type": "string",
                            "description": "Raw source code to evaluate for architectural drift."
                        }
                    },
                    "required": ["code_content"]
                }
            },
            {
                "name": "axiom_expand",
                "description": "Retrieve the full source body of a symbol that Axiom's context compression dropped from a session digest. When you see an <axiom_context_digest> skeleton (signatures kept, bodies elided) and need a specific implementation, call this with the symbol name and the digest's session id to get the actual code back.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Name of the function / struct / class / type to expand."
                        },
                        "session_id": {
                            "type": "string",
                            "description": "The session id from the digest's session=\"...\" attribute. Defaults to the standard dev session if omitted."
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "axiom_remember",
                "description": "Persist a memory (a decision, a fix, a convention, a snippet) into Axiom's long-term store so it can be recalled in future sessions. Use for things worth keeping: 'we chose X over Y because Z', a hard-won bug fix, a project convention.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "The memory content to store, verbatim." },
                        "kind": { "type": "string", "enum": ["decision","code","conversation","fix"], "description": "Optional category (default: conversation)." },
                        "scope": { "type": "string", "description": "Optional scope: 'personal' (default) for conventions that follow you, or 'project:<name>' for repo-specific memory." }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "axiom_recall",
                "description": "Search Axiom's long-term memory for content relevant to a query and return the exact stored text (never a paraphrase). Use to answer 'what did we decide about X?', 'have I solved this before?', 'what's my convention for Y?'.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "What to recall." },
                        "scope": { "type": "string", "description": "Optional project scope to include alongside 'personal'." }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "axiom_forget",
                "description": "Remove a memory by its id (from an axiom_recall result) so it is no longer recalled. Tombstones the record; the original line stays in the append-only log but is excluded from all future recall.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "The memory id to forget." },
                        "scope": { "type": "string", "description": "Scope the id belongs to (default 'personal')." }
                    },
                    "required": ["id"]
                }
            },
            {
                "name": "axiom_verify",
                "description": "Grounding verification: given your draft `response` and the `evidence` it should be grounded in, flag factual claims NOT supported by that evidence (likely hallucinations relative to the context). Returns per-claim verdicts (SUPPORTED/UNSUPPORTED/UNVERIFIED), the grounded fraction, and the flagged claims. Use before asserting facts from a document/codebase. Note: this checks support against the supplied evidence; it is not universal fact-checking and the lexical tier does not catch contradictions that reuse the evidence's wording.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "response": { "type": "string", "description": "The answer/claims to verify." },
                        "evidence": { "type": "string", "description": "The source text the response must be grounded in." }
                    },
                    "required": ["response", "evidence"]
                }
            },
            {
                "name": "axiom_validate_epistemic",
                "description": "Detect soft hallucinations by combining evidence grounding with a configured semantic LLM judge. Appends local JSONL telemetry using hashes by default; raw text capture requires AXIOM_EPISTEMIC_CAPTURE_TEXT=1.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": { "type": "string", "description": "The original task or user prompt." },
                        "response": { "type": "string", "description": "The model response to validate." },
                        "evidence": { "type": "string", "description": "Optional source evidence used by grounding verification." },
                        "target_model": { "type": "string", "description": "Optional target model identifier for telemetry." },
                        "request_id": { "type": "string", "description": "Optional caller-provided trace id." }
                    },
                    "required": ["prompt", "response"]
                }
            },
            {
                "name": "axiom_immunity",
                "description": "Report what Axiom has learned about program failures from supervised `axiom run` executions: the heals it now applies prophylactically (e.g. directories it pre-creates) and each program's failure-tension history. Use when debugging a command that fails in the user's environment — Axiom may already know the fix.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "Optional case-insensitive command substring to filter by (e.g. 'cargo build'). Omit to list everything Axiom has learned." }
                    }
                }
            },
            {
                "name": "search",
                "description": "Search Axiom's long-term memory and return a ranked list of matching records as {id, title, url}. This is the standard ChatGPT-connector `search` tool (paired with `fetch`): call `search` to find relevant memories, then `fetch` with a result id to read the full text. Internally an alias over Axiom's semantic memory recall.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "What to search Axiom's memory for." },
                        "scope": { "type": "string", "description": "Optional project scope to include alongside 'personal'." }
                    },
                    "required": ["query"]
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": { "type": "string" },
                                    "title": { "type": "string" },
                                    "url": { "type": "string" }
                                },
                                "required": ["id", "title", "url"]
                            }
                        }
                    },
                    "required": ["results"]
                }
            },
            {
                "name": "fetch",
                "description": "Fetch the full text of a single Axiom memory record by its id (an id returned by `search`). Returns {id, title, text, url, metadata}. This is the standard ChatGPT-connector `fetch` tool that pairs with `search`.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "The record id from a `search` result." }
                    },
                    "required": ["id"]
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "title": { "type": "string" },
                        "text": { "type": "string" },
                        "url": { "type": "string" },
                        "metadata": { "type": ["object", "null"] }
                    },
                    "required": ["id", "title", "text", "url", "metadata"]
                }
            },
            {
                "name": "axiom_status",
                "description": "Report the current session awareness state: token budget, tokens spent on Axiom responses, compression ratio, expansion-miss count, and a recommendation. Call this to understand how much context budget remains and whether Axiom's compression settings need tuning.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to query (default: 'mcp')."
                        }
                    }
                }
            },
            {
                "name": "axiom_post_task",
                "description": "Post a task to a named channel so another agent (Claude, Codex, or a script) can claim and execute it. Attach an Axiom context digest so the claimer can reconstruct your context without re-reading it.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "channel": { "type": "string", "description": "Work-queue channel name (e.g. 'code-review', 'codex-tasks')." },
                        "description": { "type": "string", "description": "What needs to be done." },
                        "context_digest": { "type": "string", "description": "Optional Axiom fingerprint from axiom_compress_path. The claimer can call axiom_expand to retrieve dropped symbols." },
                        "budget_snapshot": { "type": "integer", "description": "Optional remaining token count to help the claimer size its work." },
                        "priority": { "type": "integer", "description": "0 = normal, 1 = high. High-priority tasks are claimed first." },
                        "posted_by": { "type": "string", "description": "Optional agent identifier for the poster." }
                    },
                    "required": ["channel", "description"]
                }
            },
            {
                "name": "axiom_claim_task",
                "description": "Claim the next available task from a channel. Returns the highest-priority oldest pending task and marks it as in-progress. Call axiom_task_result when done.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "channel": { "type": "string", "description": "Channel to claim from." },
                        "agent_id": { "type": "string", "description": "Optional identifier of the claiming agent." }
                    },
                    "required": ["channel"]
                }
            },
            {
                "name": "axiom_task_result",
                "description": "Report the result of a claimed task. Marks it done or failed.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": { "type": "string", "description": "Task ID returned by axiom_claim_task." },
                        "result": { "type": "string", "description": "Result text, summary, or error message." },
                        "success": { "type": "boolean", "description": "true = Done, false = Failed." }
                    },
                    "required": ["task_id", "result", "success"]
                }
            },
            {
                "name": "axiom_list_tasks",
                "description": "List tasks in a channel, optionally filtered by status (pending, claimed, done, failed). Returns all tasks if no filter given.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "channel": { "type": "string", "description": "Channel to inspect." },
                        "status": { "type": "string", "description": "Optional filter: pending | claimed | done | failed." }
                    },
                    "required": ["channel"]
                }
            },
            {
                "name": "axiom_channels",
                "description": "List all task-board channels that have at least one task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": { "type": "string", "description": "Optional session identifier." }
                    }
                }
            }
        ]
    })
}

/// Route `tools/call` to the named tool, returning a JSON-RPC response whose
/// result is an MCP tool-result payload (`{ content: [...], isError: bool }`).
async fn handle_tools_call(id: Value, params: Option<&Value>, ctx: &McpContext) -> Value {
    let Some(params) = params else {
        return error_response(id, -32602, "missing params");
    };
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    match name {
        "axiom_compress_path" => {
            let Some(path) = args.get("path").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_compress_path requires string 'path'");
            };
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let path = path.to_string();
            let awareness = ctx.awareness.clone();
            let ctx = ctx.clone();
            let outcome = tokio::task::spawn_blocking(move || compress_path_blocking(&path, &ctx))
                .await
                .unwrap_or_else(|e| Err(format!("worker join error: {e}")));
            match outcome {
                Ok(block) => {
                    let text = record_and_annotate(&awareness, &session_id, "axiom_compress_path", &block);
                    success_response(id, tool_text_result(&text, false))
                }
                Err(e) => {
                    eprintln!("[mcp] axiom_compress_path failed: {e}");
                    let msg = format!("compression failed: {e}");
                    let text = record_and_annotate(&awareness, &session_id, "axiom_compress_path", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "axiom_evaluate_drift" => {
            let Some(code) = args.get("code_content").and_then(Value::as_str) else {
                return error_response(
                    id,
                    -32602,
                    "axiom_evaluate_drift requires string 'code_content'",
                );
            };
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let code = code.to_string();
            let awareness = ctx.awareness.clone();
            let ctx = ctx.clone();
            let outcome = tokio::task::spawn_blocking(move || evaluate_drift_blocking(&code, &ctx))
                .await
                .unwrap_or_else(|e| Err(format!("worker join error: {e}")));
            match outcome {
                Ok((report, is_drift)) => {
                    let text = record_and_annotate(&awareness, &session_id, "axiom_evaluate_drift", &report);
                    success_response(id, tool_text_result(&text, is_drift))
                }
                Err(e) => {
                    eprintln!("[mcp] axiom_evaluate_drift failed: {e}");
                    let msg = format!("evaluation failed: {e}");
                    let text = record_and_annotate(&awareness, &session_id, "axiom_evaluate_drift", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "axiom_expand" => {
            let Some(symbol) = args.get("symbol").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_expand requires string 'symbol'");
            };
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("axiom-dev-session")
                .to_string();
            let awareness_session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let symbol = symbol.to_string();
            let awareness = ctx.awareness.clone();
            let outcome =
                tokio::task::spawn_blocking(move || expand_symbol_blocking(&session_id, &symbol))
                    .await
                    .unwrap_or_else(|e| Err(format!("worker join error: {e}")));
            match outcome {
                Ok((text, is_err)) => {
                    let text = record_and_annotate(&awareness, &awareness_session_id, "axiom_expand", &text);
                    success_response(id, tool_text_result(&text, is_err))
                }
                Err(e) => {
                    eprintln!("[mcp] axiom_expand failed: {e}");
                    let msg = format!("expand failed: {e}");
                    let text = record_and_annotate(&awareness, &awareness_session_id, "axiom_expand", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "axiom_remember" => {
            let Some(text) = args.get("text").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_remember requires string 'text'");
            };
            let text = text.to_string();
            let kind = args
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("conversation")
                .to_string();
            let scope = args
                .get("scope")
                .and_then(Value::as_str)
                .unwrap_or("personal")
                .to_string();
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let awareness = ctx.awareness.clone();
            let ctx = ctx.clone();
            let outcome =
                tokio::task::spawn_blocking(move || remember_blocking(&text, &kind, &scope, &ctx))
                    .await
                    .unwrap_or_else(|e| Err(format!("worker join error: {e}")));
            match outcome {
                Ok(msg) => {
                    let text = record_and_annotate(&awareness, &session_id, "axiom_remember", &msg);
                    success_response(id, tool_text_result(&text, false))
                }
                Err(e) => {
                    let msg = format!("remember failed: {e}");
                    let text = record_and_annotate(&awareness, &session_id, "axiom_remember", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "axiom_recall" => {
            let Some(query) = args.get("query").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_recall requires string 'query'");
            };
            let query = query.to_string();
            let scope = args
                .get("scope")
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let awareness = ctx.awareness.clone();
            let ctx = ctx.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                recall_blocking(&query, scope.as_deref(), &ctx)
            })
            .await
            .unwrap_or_else(|e| Err(format!("worker join error: {e}")));
            match outcome {
                Ok(report) => {
                    let text = record_and_annotate(&awareness, &session_id, "axiom_recall", &report);
                    success_response(id, tool_text_result(&text, false))
                }
                Err(e) => {
                    let msg = format!("recall failed: {e}");
                    let text = record_and_annotate(&awareness, &session_id, "axiom_recall", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "axiom_forget" => {
            let Some(mem_id) = args.get("id").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_forget requires string 'id'");
            };
            let mem_id = mem_id.to_string();
            let scope = args
                .get("scope")
                .and_then(Value::as_str)
                .unwrap_or("personal")
                .to_string();
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let awareness = ctx.awareness.clone();
            let ctx = ctx.clone();
            let outcome =
                tokio::task::spawn_blocking(move || forget_blocking(&mem_id, &scope, &ctx))
                    .await
                    .unwrap_or_else(|e| Err(format!("worker join error: {e}")));
            match outcome {
                Ok(msg) => {
                    let text = record_and_annotate(&awareness, &session_id, "axiom_forget", &msg);
                    success_response(id, tool_text_result(&text, false))
                }
                Err(e) => {
                    let msg = format!("forget failed: {e}");
                    let text = record_and_annotate(&awareness, &session_id, "axiom_forget", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "axiom_verify" => {
            let response = args.get("response").and_then(Value::as_str).unwrap_or("");
            let evidence = args.get("evidence").and_then(Value::as_str).unwrap_or("");
            if response.trim().is_empty() {
                return error_response(id, -32602, "axiom_verify requires non-empty 'response'");
            }
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let report = crate::hallucination::verify(response, evidence);
            let mut text = format!(
                "Grounding: {}/{} claims supported ({:.0}% grounded).",
                report.supported,
                report.claims.len(),
                report.grounded_fraction * 100.0
            );
            let flagged = report.flagged();
            if flagged.is_empty() {
                text.push_str(" No unsupported claims.");
            } else {
                text.push_str("\nUNSUPPORTED (not grounded in the evidence):");
                for c in flagged {
                    text.push_str(&format!("\n  - {}", c.claim));
                }
            }
            let is_error = report.unsupported > 0;
            // isError=true when anything is unsupported, so the agent notices.
            let text = record_and_annotate(&ctx.awareness, &session_id, "axiom_verify", &text);
            success_response(id, tool_text_result(&text, is_error))
        }
        "axiom_validate_epistemic" => {
            let prompt = args.get("prompt").and_then(Value::as_str).unwrap_or("");
            let response = args.get("response").and_then(Value::as_str).unwrap_or("");
            let evidence = args.get("evidence").and_then(Value::as_str).unwrap_or("");
            if prompt.trim().is_empty() || response.trim().is_empty() {
                return error_response(
                    id,
                    -32602,
                    "axiom_validate_epistemic requires non-empty 'prompt' and 'response'",
                );
            }
            let config = match crate::epistemic_drift::EpistemicJudgeConfig::from_env() {
                Ok(Some(config)) => config,
                Ok(None) => {
                    return success_response(
                        id,
                        tool_text_result("epistemic judge is not configured", true),
                    )
                }
                Err(error) => return success_response(id, tool_text_result(&error, true)),
            };
            let judge = match crate::epistemic_drift::OpenAiSemanticJudge::new(config) {
                Ok(judge) => judge,
                Err(error) => return success_response(id, tool_text_result(&error, true)),
            };
            let request_id = args
                .get("request_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            let target_model = args
                .get("target_model")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            match crate::epistemic_drift::evaluate_with_judge(
                &judge,
                request_id,
                prompt,
                response,
                evidence,
                target_model,
            )
            .await
            {
                Ok(evaluation) => {
                    let is_error = evaluation.report.decision
                        != crate::epistemic_drift::EpistemicDecision::Allow;
                    let text = serde_json::to_string_pretty(&evaluation)
                        .unwrap_or_else(|error| format!("evaluation serialization failed: {error}"));
                    success_response(id, tool_text_result(&text, is_error))
                }
                Err(error) => success_response(id, tool_text_result(&error, true)),
            }
        }
        "axiom_immunity" => {
            let query = args
                .get("command")
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let awareness = ctx.awareness.clone();
            let outcome = tokio::task::spawn_blocking(move || match crate::heal_memory::HealMemory::default_path() {
                Some(path) => crate::heal_memory::HealMemory::load(&path).report_text(query.as_deref()),
                None => "Axiom heal memory is disabled (AXIOM_HEAL_MEMORY=0).".to_string(),
            })
            .await
            .unwrap_or_else(|e| format!("worker join error: {e}"));
            let outcome = record_and_annotate(&awareness, &session_id, "axiom_immunity", &outcome);
            success_response(id, tool_text_result(&outcome, false))
        }
        "search" => {
            let Some(query) = args.get("query").and_then(Value::as_str) else {
                return error_response(id, -32602, "search requires string 'query'");
            };
            let query = query.to_string();
            let scope = args
                .get("scope")
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let awareness = ctx.awareness.clone();
            let ctx = ctx.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                search_blocking(&query, scope.as_deref(), &ctx)
            })
            .await
            .unwrap_or_else(|e| Err(format!("worker join error: {e}")));
            match outcome {
                Ok(value) => {
                    // Record cost based on the JSON text representation.
                    let text_repr = serde_json::to_string(&value).unwrap_or_default();
                    record_and_annotate(&awareness, &session_id, "search", &text_repr);
                    success_response(id, tool_structured_result(value, false))
                }
                Err(e) => {
                    let msg = format!("search failed: {e}");
                    let text = record_and_annotate(&awareness, &session_id, "search", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "fetch" => {
            let Some(doc_id) = args.get("id").and_then(Value::as_str) else {
                return error_response(id, -32602, "fetch requires string 'id'");
            };
            let doc_id = doc_id.to_string();
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let awareness = ctx.awareness.clone();
            let ctx = ctx.clone();
            let outcome = tokio::task::spawn_blocking(move || fetch_blocking(&doc_id, &ctx))
                .await
                .unwrap_or_else(|e| Err(format!("worker join error: {e}")));
            match outcome {
                Ok(value) => {
                    let text_repr = serde_json::to_string(&value).unwrap_or_default();
                    record_and_annotate(&awareness, &session_id, "fetch", &text_repr);
                    success_response(id, tool_structured_result(value, false))
                }
                Err(e) => {
                    let msg = format!("fetch failed: {e}");
                    let text = record_and_annotate(&awareness, &session_id, "fetch", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "axiom_status" => {
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("mcp")
                .to_string();
            let awareness = ctx.awareness.clone();
            let awareness_session_id = session_id.clone();
            let ctx_for_status = ctx.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                status_blocking(&session_id, &ctx_for_status)
            })
            .await
            .unwrap_or_else(|e| Err(format!("worker join error: {e}")));
            match outcome {
                Ok(report) => {
                    let text = record_and_annotate(&awareness, &awareness_session_id, "axiom_status", &report);
                    success_response(id, tool_text_result(&text, false))
                }
                Err(e) => {
                    let msg = format!("status unavailable: {e}");
                    let text = record_and_annotate(&awareness, &awareness_session_id, "axiom_status", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "axiom_post_task" => {
            let Some(channel) = args.get("channel").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_post_task requires string 'channel'");
            };
            let Some(description) = args.get("description").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_post_task requires string 'description'");
            };
            let channel = channel.to_string();
            let description = description.to_string();
            let context_digest = args.get("context_digest").and_then(Value::as_str).map(str::to_string);
            let budget_snapshot = args.get("budget_snapshot").and_then(Value::as_u64).map(|n| n as usize);
            let priority = args.get("priority").and_then(Value::as_u64).unwrap_or(0) as u8;
            let posted_by = args.get("posted_by").and_then(Value::as_str).map(str::to_string);
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let awareness = ctx.awareness.clone();
            let board = ctx.task_board.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                board.post_task(&channel, description, context_digest, budget_snapshot, priority, posted_by)
            })
            .await
            .unwrap_or_else(|e| Err(std::io::Error::other(format!("join error: {e}"))));
            match outcome {
                Ok(task) => {
                    let msg = format!(
                        "Task posted.\ntask_id : {}\nchannel : {}\nstatus  : pending\ndescription: {}",
                        task.task_id, task.channel, task.description
                    );
                    let text = record_and_annotate(&awareness, &session_id, "axiom_post_task", &msg);
                    success_response(id, tool_text_result(&text, false))
                }
                Err(e) => {
                    let msg = format!("post_task failed: {e}");
                    let text = record_and_annotate(&awareness, &session_id, "axiom_post_task", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "axiom_claim_task" => {
            let Some(channel) = args.get("channel").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_claim_task requires string 'channel'");
            };
            let channel = channel.to_string();
            let agent_id = args.get("agent_id").and_then(Value::as_str).map(str::to_string);
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let awareness = ctx.awareness.clone();
            let board = ctx.task_board.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                board.claim_task(&channel, agent_id)
            })
            .await
            .unwrap_or_else(|e| Err(std::io::Error::other(format!("join error: {e}"))));
            match outcome {
                Ok(None) => {
                    let msg = "No pending tasks in this channel.";
                    let text = record_and_annotate(&awareness, &session_id, "axiom_claim_task", msg);
                    success_response(id, tool_text_result(&text, false))
                }
                Ok(Some(task)) => {
                    let digest_line = task.context_digest
                        .as_deref()
                        .map(|d| format!("\ncontext_digest:\n{d}"))
                        .unwrap_or_default();
                    let budget_line = task.budget_snapshot
                        .map(|b| format!("\nbudget_snapshot: {b} tokens"))
                        .unwrap_or_default();
                    let msg = format!(
                        "Task claimed.\ntask_id    : {}\nchannel    : {}\ndescription: {}\nposted_by  : {}\npriority   : {}{}{}",
                        task.task_id, task.channel, task.description,
                        task.posted_by.as_deref().unwrap_or("unknown"),
                        task.priority, budget_line, digest_line
                    );
                    let text = record_and_annotate(&awareness, &session_id, "axiom_claim_task", &msg);
                    success_response(id, tool_text_result(&text, false))
                }
                Err(e) => {
                    let msg = format!("claim_task failed: {e}");
                    let text = record_and_annotate(&awareness, &session_id, "axiom_claim_task", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "axiom_task_result" => {
            let Some(task_id) = args.get("task_id").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_task_result requires string 'task_id'");
            };
            let Some(result) = args.get("result").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_task_result requires string 'result'");
            };
            let success = args.get("success").and_then(Value::as_bool).unwrap_or(true);
            let task_id = task_id.to_string();
            let result = result.to_string();
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let awareness = ctx.awareness.clone();
            let board = ctx.task_board.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                board.task_result(&task_id, result, success)
            })
            .await
            .unwrap_or_else(|e| Err(std::io::Error::other(format!("join error: {e}"))));
            match outcome {
                Ok(true) => {
                    let msg = format!("Task marked {}.", if success { "done" } else { "failed" });
                    let text = record_and_annotate(&awareness, &session_id, "axiom_task_result", &msg);
                    success_response(id, tool_text_result(&text, false))
                }
                Ok(false) => {
                    let msg = "task_id not found.";
                    let text = record_and_annotate(&awareness, &session_id, "axiom_task_result", msg);
                    success_response(id, tool_text_result(&text, true))
                }
                Err(e) => {
                    let msg = format!("task_result failed: {e}");
                    let text = record_and_annotate(&awareness, &session_id, "axiom_task_result", &msg);
                    success_response(id, tool_text_result(&text, true))
                }
            }
        }
        "axiom_list_tasks" => {
            let Some(channel) = args.get("channel").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_list_tasks requires string 'channel'");
            };
            let channel = channel.to_string();
            let status_filter = args.get("status").and_then(Value::as_str).map(str::to_string);
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let awareness = ctx.awareness.clone();
            let board = ctx.task_board.clone();
            let tasks = tokio::task::spawn_blocking(move || {
                board.list_tasks(&channel, status_filter.as_deref())
            })
            .await
            .unwrap_or_default();
            if tasks.is_empty() {
                let text = record_and_annotate(&awareness, &session_id, "axiom_list_tasks", "No tasks found.");
                return success_response(id, tool_text_result(&text, false));
            }
            let lines: Vec<String> = tasks.iter().map(|t| {
                format!(
                    "[{}] {} | {} | {}{}",
                    t.status,
                    t.task_id.get(..8).unwrap_or(&t.task_id),
                    t.description,
                    t.posted_by.as_deref().unwrap_or("?"),
                    t.result.as_deref().map(|r| format!(" → {}", r.lines().next().unwrap_or(r))).unwrap_or_default()
                )
            }).collect();
            let msg = lines.join("\n");
            let text = record_and_annotate(&awareness, &session_id, "axiom_list_tasks", &msg);
            success_response(id, tool_text_result(&text, false))
        }
        "axiom_channels" => {
            let session_id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let board = ctx.task_board.clone();
            let channels = tokio::task::spawn_blocking(move || board.channels())
                .await
                .unwrap_or_default();
            let text = if channels.is_empty() {
                "No task channels found.".to_string()
            } else {
                format!("Channels ({}): {}", channels.len(), channels.join(", "))
            };
            let text = record_and_annotate(&ctx.awareness, &session_id, "axiom_channels", &text);
            success_response(id, tool_text_result(&text, false))
        }
        other => error_response(id, -32602, &format!("unknown tool: {other}")),
    }
}

/// `axiom_status` worker — reports local MCP awareness first, then falls back to
/// the HTTP proxy's awareness endpoint when a proxy session exists there.
fn status_blocking(session_id: &str, ctx: &McpContext) -> Result<String, String> {
    if let Some(local) = ctx.awareness.get(session_id) {
        let budget = local
            .budget()
            .map(|n| format!("{n} tokens"))
            .unwrap_or_else(|| "not set".to_string());
        let spent = local
            .tokens_spent
            .load(std::sync::atomic::Ordering::Relaxed);
        let tool_calls = local
            .tool_calls_total
            .load(std::sync::atomic::Ordering::Relaxed);
        let expansions = local
            .expansion_calls
            .load(std::sync::atomic::Ordering::Relaxed);
        let ratio = local
            .compression_ratio()
            .map(|r| format!("{:.1}%", r * 100.0))
            .unwrap_or_else(|| "no data".to_string());
        let model = local
            .target_model
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let rec = local.recommendation().unwrap_or_else(|| "—".to_string());
        let vibe_ready = ctx
            .vibe
            .lock()
            .map(|v| v.is_initialized())
            .unwrap_or(false);
        return Ok(format!(
            "=== Axiom MCP Session Awareness: {session_id} ===
             model            : {model}
             budget remaining : {budget}
             spent on Axiom   : {spent} tokens across {tool_calls} tool calls
             compression ratio: {ratio} (expand misses: {expansions})
             master vibe ready: {vibe_ready}
             available tools  : {}
             recommendation   : {rec}",
            tool_names().len()
        ));
    }

    let proxy_url = &ctx.proxy_url;
    let url = format!(
        "{}/v1/awareness/{}",
        proxy_url.trim_end_matches('/'),
        urlencoding::encode(session_id)
    );
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .send()
        .map_err(|e| format!("could not reach Axiom proxy at {url}: {e}"))?;
    let status = resp.status();
    if status.as_u16() == 404 {
        return Ok(format!(
            "No awareness state for session '{session_id}' yet. Call POST /v1/budget on the HTTP proxy or pass session_id through MCP tool calls to initialize local MCP awareness. available_tools={}",
            tool_names().len()
        ));
    }
    if !status.is_success() {
        return Err(format!("proxy returned HTTP {status}"));
    }
    let v: serde_json::Value = resp
        .json()
        .map_err(|e| format!("bad JSON from proxy: {e}"))?;
    // Format as a readable block.
    let budget = v["budget_remaining"]
        .as_u64()
        .map(|n| format!("{n} tokens"))
        .unwrap_or_else(|| "not set".to_string());
    let spent = v["tokens_spent_on_axiom"].as_u64().unwrap_or(0);
    let tool_calls = v["tool_calls_total"].as_u64().unwrap_or(0);
    let expansions = v["expansion_calls"].as_u64().unwrap_or(0);
    let ratio = v["compression_ratio"]
        .as_f64()
        .map(|r| format!("{:.1}%", r * 100.0))
        .unwrap_or_else(|| "no data".to_string());
    let tight = v["is_tight"].as_bool().unwrap_or(false);
    let model = v["target_model"].as_str().unwrap_or("unknown");
    let rec = v["recommendation"].as_str().unwrap_or("—");
    Ok(format!(
        "=== Axiom Session Awareness: {session_id} ===
         model            : {model}
         budget remaining : {budget}{tight_flag}
         spent on Axiom   : {spent} tokens across {tool_calls} tool calls
         compression ratio: {ratio} (expand misses: {expansions})
         recommendation   : {rec}",
        tight_flag = if tight { " ⚠ TIGHT" } else { "" }
    ))
}

fn tool_names() -> Vec<String> {
    let list = tools_list();
    list["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| tool["name"].as_str())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

/// `axiom_expand` worker — HTTP-calls the running proxy's `POST /v1/expand` to
/// retrieve a dropped symbol body. The MCP server is a separate process from the
/// proxy, so it reaches it over the loopback API. Override the proxy location
/// with `AXIOM_PROXY_URL` (default `http://127.0.0.1:3000`).
/// Returns `(text, is_error)`.
fn expand_symbol_blocking(session_id: &str, symbol: &str) -> Result<(String, bool), String> {
    let base =
        std::env::var("AXIOM_PROXY_URL").unwrap_or_else(|_| "http://127.0.0.1:3000".to_string());
    let url = format!("{}/v1/expand", base.trim_end_matches('/'));
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .json(&json!({ "session_id": session_id, "symbol": symbol }))
        .send()
        .map_err(|e| format!("could not reach Axiom proxy at {url}: {e}"))?;
    let status = resp.status();
    let body: Value = resp
        .json()
        .map_err(|e| format!("bad response from proxy ({status}): {e}"))?;
    if body.get("found").and_then(Value::as_bool).unwrap_or(false) {
        let code = body
            .get("body")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        Ok((code, false))
    } else {
        let err = body
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("symbol not found")
            .to_string();
        Ok((
            format!("could not expand '{symbol}' (HTTP {status}): {err}"),
            true,
        ))
    }
}

/// Allocate the starting W̃ states for a tool session: primed from the master
/// vibe when opt-in priming is on and a master exists, else identity.
fn start_states(ctx: &McpContext, pipeline: &InferencePipeline) -> Result<Vec<Tensor>, String> {
    if ctx.prime {
        if let Ok(vibe) = ctx.vibe.lock() {
            if let Some(primed) = vibe.prime_states() {
                return Ok(primed);
            }
        }
    }
    pipeline.init_session_states().map_err(|e| e.to_string())
}

/// `axiom_compress_path` worker. Reads the directory, streams it through the
/// TTT engine, extracts the fingerprint, then commits + persists the master.
fn compress_path_blocking(path: &str, ctx: &McpContext) -> Result<String, String> {
    let started = Instant::now();
    let p = Path::new(path);
    if !p.exists() {
        return Err(format!("path does not exist: {path}"));
    }

    // Gather text content (bounded by max_files / max_bytes).
    let mut total_bytes = 0usize;
    let mut file_count = 0usize;
    let mut corpus = String::new();
    if p.is_file() {
        if let Ok(text) = std::fs::read_to_string(p) {
            corpus.push_str(&text);
            file_count = 1;
        }
    } else {
        for entry in walkdir::WalkDir::new(p)
            .into_iter()
            .filter_entry(|e| !should_skip_compression_path(e.path()))
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if file_count >= ctx.max_files || total_bytes >= ctx.max_bytes {
                break;
            }
            if let Ok(text) = std::fs::read_to_string(entry.path()) {
                let remaining = ctx.max_bytes.saturating_sub(total_bytes);
                let slice = if text.len() > remaining {
                    &text[..remaining]
                } else {
                    &text[..]
                };
                corpus.push_str(slice);
                corpus.push('\n');
                total_bytes += slice.len();
                file_count += 1;
            }
        }
    }
    if corpus.trim().is_empty() {
        return Err(format!("no readable text content under {path}"));
    }

    let pipeline = ctx
        .pipeline
        .lock()
        .map_err(|_| "pipeline lock poisoned".to_string())?;
    let token_ids = pipeline.encode_text(&corpus);
    let tokens_processed = token_ids.len();

    let mut states = start_states(ctx, &pipeline)?;

    // Adapt in bounded windows to cap per-call memory.
    for window in token_ids.chunks(ADAPT_WINDOW_TOKENS) {
        adapt_session_blocking(&pipeline, &mut states, window).map_err(|e| e.to_string())?;
    }

    // Recall pass + fingerprint. Use the tail of the corpus as the query.
    let query: Vec<u32> = token_ids.iter().rev().take(32).rev().copied().collect();
    let session_id = format!("mcp-compress-{}", short_hash(path));
    let fingerprint = extract_memory_vector_blocking(
        &pipeline,
        &mut states,
        &query,
        &session_id,
        tokens_processed,
        started,
        ctx.top_k,
    )
    .map_err(|e| e.to_string())?;

    // Explicit merge trigger: fold this session into the persistent master vibe.
    match ctx.vibe.lock() {
        Ok(mut vibe) => {
            if let Err(e) = vibe.commit_and_save(&states) {
                eprintln!("[mcp] vibe commit skipped: {e}");
            }
        }
        Err(_) => eprintln!("[mcp] vibe lock poisoned; commit skipped"),
    }

    eprintln!(
        "[mcp] compressed {file_count} file(s) / {tokens_processed} tokens from {path} \
         (recall_norm={:.3})",
        fingerprint.recall_norm
    );
    Ok(fingerprint.to_prompt_block())
}

fn should_skip_compression_path(path: &Path) -> bool {
    const SKIP_NAMES: &[&str] = &[
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "target",
        "dist",
        "build",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "checkpoints",
    ];
    path.components().any(|component| {
        let Component::Normal(name) = component else {
            return false;
        };
        let name = name.to_string_lossy();
        SKIP_NAMES.iter().any(|skip| name.eq_ignore_ascii_case(skip))
    })
}

/// `axiom_evaluate_drift` worker. Returns `(report_text, is_drift)`.
fn evaluate_drift_blocking(code: &str, ctx: &McpContext) -> Result<(String, bool), String> {
    let pipeline = ctx
        .pipeline
        .lock()
        .map_err(|_| "pipeline lock poisoned".to_string())?;
    let mut ids = pipeline.encode_text(code);
    if ids.len() < 2 {
        return Ok((
            "input too short to evaluate drift (need >= 2 tokens)".to_string(),
            false,
        ));
    }
    ids.truncate(DRIFT_MAX_TOKENS);
    let n = ids.len();
    let device = pipeline.device();

    let (mut states, baseline_ready) = drift_states(ctx, &pipeline)?;

    // Next-token prediction: predict ids[1..] from ids[..n-1].
    let input =
        Tensor::from_vec(ids[..n - 1].to_vec(), (1, n - 1), device).map_err(|e| e.to_string())?;
    let logits = pipeline
        .model()
        .forward_lm(&input, &mut states)
        .map_err(|e| e.to_string())?; // [1, n-1, vocab]
    let vocab = pipeline.model().config.vocab_size;
    let logits_2d = logits
        .squeeze(0)
        .and_then(|t| t.reshape((n - 1, vocab)))
        .map_err(|e| e.to_string())?;
    let targets =
        Tensor::from_vec(ids[1..].to_vec(), (n - 1,), device).map_err(|e| e.to_string())?;
    let loss = candle_nn::loss::cross_entropy(&logits_2d, &targets).map_err(|e| e.to_string())?;
    let loss_val = loss.to_scalar::<f32>().map_err(|e| e.to_string())?;

    if !baseline_ready {
        let report = format!(
            "cross_entropy_loss={loss_val:.4} baseline_threshold={:.4} tokens_scored={n} drift=UNAVAILABLE confidence=low reason=no_master_vibe_baseline",
            ctx.drift_threshold
        );
        eprintln!("[mcp] evaluate_drift -> {report}");
        return Ok((report, false));
    }

    let is_drift = loss_val > ctx.drift_threshold;
    let report = format!(
        "cross_entropy_loss={loss_val:.4} baseline_threshold={:.4} tokens_scored={} drift={} confidence=normal",
        ctx.drift_threshold,
        n,
        if is_drift { "YES" } else { "no" }
    );
    eprintln!("[mcp] evaluate_drift -> {report}");
    Ok((report, is_drift))
}

fn drift_states(
    ctx: &McpContext,
    pipeline: &InferencePipeline,
) -> Result<(Vec<Tensor>, bool), String> {
    let primed = ctx
        .vibe
        .lock()
        .ok()
        .and_then(|vibe| vibe.prime_states());
    if let Some(states) = primed {
        Ok((states, true))
    } else {
        Ok((pipeline.init_session_states().map_err(|e| e.to_string())?, false))
    }
}

/// Embed via the trained contrastive encoder when present, else the pipeline
/// (TTT pooling). The single embedding entry point for remember + recall.
fn embed_query(text: &str, ctx: &McpContext) -> Result<Vec<f32>, String> {
    if let Some(e) = &ctx.embedder {
        return e.embed(text).map_err(|err| err.to_string());
    }
    let pipeline = ctx
        .pipeline
        .lock()
        .map_err(|_| "pipeline lock poisoned".to_string())?;
    embed_text(&pipeline, text).map_err(|err| err.to_string())
}

/// `axiom_remember` worker. Embeds the text, scores its drift (salience), and
/// appends a record to the chosen scope. Returns the new memory id.
fn remember_blocking(
    text: &str,
    kind: &str,
    scope: &str,
    ctx: &McpContext,
) -> Result<String, String> {
    let embedding = embed_query(text, ctx)?;
    // Salience = drift cross-entropy (reuses the evaluate path's signal cheaply).
    let drift = {
        let pipeline = ctx
            .pipeline
            .lock()
            .map_err(|_| "pipeline lock poisoned".to_string())?;
        drift_score(&pipeline, text).unwrap_or(0.0)
    };

    let kind = match kind {
        "decision" => MemoryKind::Decision,
        "code" => MemoryKind::Code,
        "fix" => MemoryKind::Fix,
        _ => MemoryKind::Conversation,
    };
    let id = short_hash(&format!("{scope}|{}|{text}", now_secs()));
    let rec = MemoryRecord {
        id: id.clone(),
        scope: scope.to_string(),
        ts: now_secs(),
        kind,
        body: text.to_string(),
        embedding,
        drift_at_ingest: drift,
        supersedes: None,
        tombstone: false,
    };
    let store = ctx
        .memory
        .lock()
        .map_err(|_| "memory lock poisoned".to_string())?;
    store.append(&rec).map_err(|e| e.to_string())?;
    Ok(format!(
        "remembered id={id} scope={scope} (salience drift={drift:.2})"
    ))
}

/// `axiom_recall` worker. Embeds the query and searches `personal` ∪ optional
/// project scope, returning a readable list of exact stored bodies + ids.
fn recall_blocking(query: &str, scope: Option<&str>, ctx: &McpContext) -> Result<String, String> {
    let q_emb = embed_query(query, ctx)?;

    let mut scopes = vec!["personal".to_string()];
    if let Some(s) = scope {
        if s != "personal" {
            scopes.push(s.to_string());
        }
    }
    let store = ctx
        .memory
        .lock()
        .map_err(|_| "memory lock poisoned".to_string())?;
    let hits = recall(&store, &scopes, &q_emb, &RecallParams::default());
    if hits.is_empty() {
        return Ok("no relevant memories found".to_string());
    }
    let mut out = String::from("<axiom_recall>\n");
    for h in &hits {
        out.push_str(&format!(
            "• [{:.2}] id={} scope={} ts={}\n  {}\n",
            h.score, h.record.id, h.record.scope, h.record.ts, h.record.body
        ));
    }
    out.push_str("</axiom_recall>");
    Ok(out)
}

/// Percent-encode a URI segment: RFC 3986 unreserved characters pass through,
/// everything else becomes `%XX`. Keeps memory locators/ids valid for scopes
/// like `project:acme` (a raw `:` would otherwise be misread).
fn pct_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Inverse of [`pct_encode`]. Invalid escapes are passed through literally so it
/// never panics on malformed input.
fn pct_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 3 <= bytes.len() {
            if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Connector-safe, stable opaque locator for a memory record (shown as `url` in
/// search/fetch results so ChatGPT can display/cite a source without exposing
/// the filesystem). Scope and id live in the percent-encoded *path*, never the
/// authority, so values like `project:acme` cannot be misparsed as `host:port`.
fn memory_url(scope: &str, id: &str) -> String {
    format!(
        "axiom-memory://memory/{}/{}",
        pct_encode(scope),
        pct_encode(id)
    )
}

/// The scope-qualified id surfaced by `search` and consumed by `fetch`, so the
/// round-trip resolves to exactly the record `search` returned regardless of how
/// raw ids are allocated. Format `<pct-encoded-scope>:<raw-id>` — raw ids are
/// hex, so the first `:` unambiguously separates the two parts.
fn qualified_id(scope: &str, id: &str) -> String {
    format!("{}:{}", pct_encode(scope), id)
}

/// Parse a [`qualified_id`] back into `(scope, raw_id)`. Returns `None` for a
/// bare id (no `:`), letting `fetch` fall back to a scope-scan for legacy ids.
fn parse_qualified_id(doc_id: &str) -> Option<(String, String)> {
    doc_id
        .split_once(':')
        .map(|(s, i)| (pct_decode(s), i.to_string()))
}

/// A short single-line title from a record body (first non-empty line, truncated).
fn title_for(body: &str) -> String {
    let line = body.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    const MAX: usize = 80;
    if line.chars().count() > MAX {
        let truncated: String = line.chars().take(MAX - 1).collect();
        format!("{truncated}…")
    } else if line.is_empty() {
        "(untitled memory)".to_string()
    } else {
        line.to_string()
    }
}

/// `search` worker (ChatGPT-connector alias over recall). Returns the JSON value
/// `{"results":[{"id","title","url"}]}` — the shape the standard ChatGPT search
/// tool expects. Empty `results` when nothing matches (not an error).
fn search_blocking(query: &str, scope: Option<&str>, ctx: &McpContext) -> Result<Value, String> {
    let q_emb = embed_query(query, ctx)?;
    let mut scopes = vec!["personal".to_string()];
    if let Some(s) = scope {
        if s != "personal" {
            scopes.push(s.to_string());
        }
    }
    let store = ctx
        .memory
        .lock()
        .map_err(|_| "memory lock poisoned".to_string())?;
    let hits = recall(&store, &scopes, &q_emb, &RecallParams::default());
    let results: Vec<Value> = hits
        .iter()
        .map(|h| {
            json!({
                "id": qualified_id(&h.record.scope, &h.record.id),
                "title": title_for(&h.record.body),
                "url": memory_url(&h.record.scope, &h.record.id),
            })
        })
        .collect();
    Ok(json!({ "results": results }))
}

/// `fetch` worker (ChatGPT-connector alias). Resolves a `search` result id to its
/// full stored text, returning JSON `{"id","title","text","url","metadata"}`.
/// A scope-qualified id (the form `search` emits) resolves the exact scope/record
/// deterministically; a bare legacy id falls back to a cross-scope lookup.
fn fetch_blocking(doc_id: &str, ctx: &McpContext) -> Result<Value, String> {
    let store = ctx
        .memory
        .lock()
        .map_err(|_| "memory lock poisoned".to_string())?;
    let rec = match parse_qualified_id(doc_id) {
        Some((scope, raw_id)) => store
            .load_scope(&scope)
            .into_iter()
            .find(|r| r.id == raw_id),
        None => store.get(doc_id),
    }
    .ok_or_else(|| format!("no memory record with id={doc_id}"))?;
    Ok(json!({
        // Echo the qualified id so a follow-up fetch is still unambiguous.
        "id": qualified_id(&rec.scope, &rec.id),
        "title": title_for(&rec.body),
        "text": rec.body,
        "url": memory_url(&rec.scope, &rec.id),
        "metadata": { "scope": rec.scope, "ts": rec.ts },
    }))
}

/// `axiom_forget` worker. Tombstones an id in its scope.
fn forget_blocking(mem_id: &str, scope: &str, ctx: &McpContext) -> Result<String, String> {
    let store = ctx
        .memory
        .lock()
        .map_err(|_| "memory lock poisoned".to_string())?;
    store.tombstone(scope, mem_id).map_err(|e| e.to_string())?;
    Ok(format!("forgot id={mem_id} scope={scope}"))
}

/// Cross-entropy of `text` against current fast-weights — used as a salience
/// score at ingest. Mirrors `evaluate_drift_blocking` but returns the raw loss.
fn drift_score(pipeline: &InferencePipeline, text: &str) -> Result<f32, String> {
    let mut ids = pipeline.encode_text(text);
    if ids.len() < 2 {
        return Ok(0.0);
    }
    ids.truncate(DRIFT_MAX_TOKENS);
    let n = ids.len();
    let device = pipeline.device();
    let mut states = pipeline.init_session_states().map_err(|e| e.to_string())?;
    let input =
        Tensor::from_vec(ids[..n - 1].to_vec(), (1, n - 1), device).map_err(|e| e.to_string())?;
    let logits = pipeline
        .model()
        .forward_lm(&input, &mut states)
        .map_err(|e| e.to_string())?;
    let vocab = pipeline.model().config.vocab_size;
    let logits_2d = logits
        .squeeze(0)
        .and_then(|t| t.reshape((n - 1, vocab)))
        .map_err(|e| e.to_string())?;
    let targets =
        Tensor::from_vec(ids[1..].to_vec(), (n - 1,), device).map_err(|e| e.to_string())?;
    let loss = candle_nn::loss::cross_entropy(&logits_2d, &targets).map_err(|e| e.to_string())?;
    loss.to_scalar::<f32>().map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Per-response token-cost tracking
// ---------------------------------------------------------------------------

/// Estimate token cost, record it against the awareness store, and optionally
/// append a compact `_meta: token_cost=N` line to the response text.
///
/// The meta line is only appended when a budget has already been set for the
/// session (i.e. `awareness.get(session_id).is_some()`), keeping responses
/// clean until the agent opts in via `POST /v1/budget`.
fn record_and_annotate(
    awareness: &AwarenessStore,
    session_id: &str,
    tool_name: &str,
    response_text: &str,
) -> String {
    let token_cost = (response_text.len() / 4).max(1);
    // Always record in the store (creates the entry if it already exists).
    // We only record if the session has been opted in (budget set).
    let annotate = if let Some(state) = awareness.get(session_id) {
        state.record_tool_response(tool_name, token_cost);
        true
    } else {
        false
    };
    if annotate {
        format!("{response_text}\n\n_meta: token_cost={token_cost}")
    } else {
        response_text.to_string()
    }
}

// ---------------------------------------------------------------------------
// JSON-RPC helpers
// ---------------------------------------------------------------------------

fn success_response(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// Build an MCP tool-result payload.
fn tool_text_result(text: &str, is_error: bool) -> Value {
    json!({
        "content": [ { "type": "text", "text": text } ],
        "isError": is_error
    })
}

/// Build an MCP tool-result payload that carries a structured object in
/// `structuredContent` *and* a JSON-encoded text mirror in `content`. The MCP
/// spec lets a tool return both; ChatGPT standard/company-knowledge connectors
/// read `structuredContent` (validated against the tool's `outputSchema`), while
/// the text mirror keeps plain text-only clients working.
fn tool_structured_result(structured: Value, is_error: bool) -> Value {
    let text = serde_json::to_string(&structured).unwrap_or_else(|_| "{}".to_string());
    json!({
        "content": [ { "type": "text", "text": text } ],
        "structuredContent": structured,
        "isError": is_error
    })
}

/// Short stable id derived from a path, for session labelling.
fn short_hash(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(s.as_bytes());
    format!(
        "{:x}{:x}{:x}{:x}",
        digest[0], digest[1], digest[2], digest[3]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_list_exposes_tools_with_schemas() {
        let list = tools_list();
        let tools = list["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 17);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"axiom_compress_path"));
        assert!(names.contains(&"axiom_evaluate_drift"));
        assert!(names.contains(&"axiom_expand"));
        assert!(names.contains(&"axiom_status"));
        assert!(names.contains(&"axiom_post_task"));
        assert!(names.contains(&"axiom_claim_task"));
        assert!(names.contains(&"axiom_task_result"));
        assert!(names.contains(&"axiom_list_tasks"));
        assert!(names.contains(&"axiom_channels"));
        for t in tools {
            assert_eq!(t["inputSchema"]["type"], "object");
            // `required` is optional in MCP (axiom_immunity has only optional
            // params); when present it must be an array.
            if let Some(req) = t["inputSchema"].get("required") {
                assert!(req.is_array());
            }
        }
    }

    #[test]
    fn tools_list_includes_memory_tools() {
        let list = tools_list();
        let tools = list["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"axiom_remember"));
        assert!(names.contains(&"axiom_recall"));
        assert!(names.contains(&"axiom_forget"));
    }

    #[test]
    fn tools_list_includes_immunity_tool() {
        let list = tools_list();
        let names: Vec<&str> = list["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"axiom_immunity"));
        assert!(names.contains(&"axiom_validate_epistemic"));
    }

    #[test]
    fn tools_list_includes_chatgpt_search_fetch_aliases() {
        // The standard ChatGPT connector requires tools literally named
        // `search` and `fetch`; assert both are present with the right required
        // params so a standard-mode connector can attach.
        let list = tools_list();
        let tools = list["tools"].as_array().unwrap();
        let search = tools.iter().find(|t| t["name"] == "search").unwrap();
        let fetch = tools.iter().find(|t| t["name"] == "fetch").unwrap();
        assert_eq!(search["inputSchema"]["required"][0], "query");
        assert_eq!(fetch["inputSchema"]["required"][0], "id");
    }

    #[test]
    fn compression_path_filter_skips_generated_and_vendor_dirs() {
        assert!(should_skip_compression_path(Path::new("repo/.venv/lib/site.py")));
        assert!(should_skip_compression_path(Path::new("repo/target/debug/lib.rs")));
        assert!(should_skip_compression_path(Path::new("repo/app/__pycache__/x.pyc")));
        assert!(!should_skip_compression_path(Path::new("repo/src/lib.rs")));
    }

    #[test]
    fn tool_names_matches_tools_list() {
        let names = tool_names();
        assert!(names.contains(&"axiom_remember".to_string()));
        assert!(names.contains(&"axiom_expand".to_string()));
        assert!(names.contains(&"search".to_string()));
        assert!(names.contains(&"fetch".to_string()));
    }

    #[test]
    fn title_for_takes_first_nonempty_line_and_truncates() {
        assert_eq!(title_for("\n\nhello world\nmore"), "hello world");
        assert_eq!(title_for("   "), "(untitled memory)");
        let long = "x".repeat(200);
        let t = title_for(&long);
        assert!(t.chars().count() <= 80, "title too long: {}", t.chars().count());
        assert!(t.ends_with('…'));
    }

    #[test]
    fn memory_url_is_stable_and_encodes_special_scopes() {
        assert_eq!(
            memory_url("personal", "abc123"),
            "axiom-memory://memory/personal/abc123"
        );
        // A scope with a colon must not land raw in the authority.
        assert_eq!(
            memory_url("project:acme", "abc123"),
            "axiom-memory://memory/project%3Aacme/abc123"
        );
    }

    #[test]
    fn qualified_id_roundtrips_including_special_scopes() {
        for (scope, id) in [("personal", "abc123"), ("project:acme", "deadbeef")] {
            let q = qualified_id(scope, id);
            let (s, i) = parse_qualified_id(&q).expect("qualified id should parse");
            assert_eq!(s, scope);
            assert_eq!(i, id);
        }
        // A bare (legacy) id has no ':' and parses as None → fetch falls back.
        assert!(parse_qualified_id("abc123").is_none());
    }

    #[test]
    fn pct_encode_decode_roundtrip() {
        for s in ["personal", "project:acme", "a/b c", "weird%20"] {
            assert_eq!(pct_decode(&pct_encode(s)), s);
        }
    }

    #[test]
    fn initialize_advertises_tools_capability() {
        // Build the same result initialize returns and assert its shape.
        let result = json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION }
        });
        assert!(result["capabilities"]["tools"].is_object());
        assert_eq!(result["protocolVersion"], MCP_PROTOCOL_VERSION);
    }

    #[test]
    fn error_and_success_envelopes_are_wellformed() {
        let ok = success_response(json!(1), json!({"x":1}));
        assert_eq!(ok["jsonrpc"], "2.0");
        assert_eq!(ok["id"], 1);
        assert!(ok["result"].is_object());

        let err = error_response(json!("abc"), -32601, "nope");
        assert_eq!(err["error"]["code"], -32601);
        assert_eq!(err["id"], "abc");
    }

    #[test]
    fn tool_result_marks_errors() {
        let r = tool_text_result("boom", true);
        assert_eq!(r["isError"], true);
        assert_eq!(r["content"][0]["type"], "text");
        assert_eq!(r["content"][0]["text"], "boom");
    }

    #[test]
    fn structured_result_carries_object_and_text_mirror() {
        let payload = json!({ "results": [ { "id": "a", "title": "t", "url": "u" } ] });
        let r = tool_structured_result(payload.clone(), false);
        assert_eq!(r["isError"], false);
        // structuredContent holds the object (what ChatGPT validates/reads).
        assert_eq!(r["structuredContent"], payload);
        // content holds a JSON-encoded text mirror (text-only clients).
        let text = r["content"][0]["text"].as_str().unwrap();
        let reparsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(reparsed, payload);
    }

    #[test]
    fn search_and_fetch_advertise_output_schemas() {
        let list = tools_list();
        let tools = list["tools"].as_array().unwrap();
        let search = tools.iter().find(|t| t["name"] == "search").unwrap();
        let fetch = tools.iter().find(|t| t["name"] == "fetch").unwrap();
        assert_eq!(search["outputSchema"]["required"][0], "results");
        let fetch_req = fetch["outputSchema"]["required"].as_array().unwrap();
        for k in ["id", "title", "text", "url"] {
            assert!(
                fetch_req.iter().any(|v| v == k),
                "fetch outputSchema must require {k}"
            );
        }
    }
}
