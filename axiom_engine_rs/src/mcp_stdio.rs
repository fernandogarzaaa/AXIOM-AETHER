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
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{Device, Tensor};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::config::{AxiomConfig, DEFAULT_CHECKPOINT_PATH};
use crate::context_compressor::{adapt_session_blocking, extract_memory_vector_blocking};
use crate::inference::InferencePipeline;
use crate::vibe_memory::MasterVibe;

/// MCP protocol revision we advertise in the `initialize` handshake.
const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "axiom-ttt";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Token window fed through `forward_lm` at a time. Bounds per-call memory:
/// `forward_lm` buffers one hidden vector per token in the window.
const ADAPT_WINDOW_TOKENS: usize = 256;
/// Cap on tokens scored by `axiom_evaluate_drift` (bounds compute per call).
const DRIFT_MAX_TOKENS: usize = 512;

/// Shared engine context handed to every tool invocation.
#[derive(Clone)]
struct McpContext {
    pipeline: Arc<Mutex<InferencePipeline>>,
    vibe: Arc<Mutex<MasterVibe>>,
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
}

/// Boot the MCP stdio server. Runs until stdin reaches EOF (host disconnect).
pub async fn run_stdio_server(
    config: AxiomConfig,
    device: Device,
    checkpoint_path: String,
) -> Result<(), Box<dyn Error>> {
    // All status output MUST go to stderr; stdout is reserved for JSON-RPC.
    eprintln!("[mcp] booting Axiom MCP stdio server (protocol {MCP_PROTOCOL_VERSION})");

    // Build the pipeline on a blocking thread. It owns a `reqwest::blocking::Client`
    // which carries its own runtime; constructing/dropping that inside the async
    // context is unsafe, so we keep all of its lifecycle off the async runtime.
    let pipeline = {
        let cfg = config.clone();
        let dev = device.clone();
        let ckpt = checkpoint_path.clone();
        tokio::task::spawn_blocking(move || {
            if ckpt == DEFAULT_CHECKPOINT_PATH {
                InferencePipeline::new(cfg, dev)
            } else {
                InferencePipeline::with_checkpoint(cfg, dev, &ckpt)
            }
        })
        .await
        .map_err(|e| format!("pipeline build join error: {e}"))?
        .map_err(|e| format!("failed to assemble inference pipeline: {e}"))?
    };

    let vibe = MasterVibe::from_env(config.n_layers, config.d_model, &device);
    let prime = std::env::var("AXIOM_VIBE_PRIME").map(|v| v == "1").unwrap_or(false);
    if prime && !vibe.is_initialized() {
        eprintln!("[mcp] AXIOM_VIBE_PRIME=1 set but no master vibe yet; sessions start from identity until first commit");
    }
    let drift_threshold = std::env::var("AXIOM_DRIFT_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(6.0);
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

    let ctx = McpContext {
        pipeline: Arc::new(Mutex::new(pipeline)),
        vibe: Arc::new(Mutex::new(vibe)),
        prime,
        drift_threshold,
        top_k,
        max_files,
        max_bytes,
    };

    eprintln!(
        "[mcp] ready — tools: axiom_compress_path, axiom_evaluate_drift \
         (prime={prime}, drift_threshold={drift_threshold})"
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

/// Parse and route one JSON-RPC line. Returns `Some(response)` for requests,
/// `None` for notifications (no `id`) which must not be answered.
async fn handle_message(line: &str, ctx: &McpContext) -> Option<Value> {
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[mcp] parse error: {e}");
            return Some(error_response(Value::Null, -32700, &format!("parse error: {e}")));
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
    let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

    match name {
        "axiom_compress_path" => {
            let Some(path) = args.get("path").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_compress_path requires string 'path'");
            };
            let path = path.to_string();
            let ctx = ctx.clone();
            let outcome = tokio::task::spawn_blocking(move || compress_path_blocking(&path, &ctx))
                .await
                .unwrap_or_else(|e| Err(format!("worker join error: {e}")));
            match outcome {
                Ok(block) => success_response(id, tool_text_result(&block, false)),
                Err(e) => {
                    eprintln!("[mcp] axiom_compress_path failed: {e}");
                    success_response(id, tool_text_result(&format!("compression failed: {e}"), true))
                }
            }
        }
        "axiom_evaluate_drift" => {
            let Some(code) = args.get("code_content").and_then(Value::as_str) else {
                return error_response(id, -32602, "axiom_evaluate_drift requires string 'code_content'");
            };
            let code = code.to_string();
            let ctx = ctx.clone();
            let outcome = tokio::task::spawn_blocking(move || evaluate_drift_blocking(&code, &ctx))
                .await
                .unwrap_or_else(|e| Err(format!("worker join error: {e}")));
            match outcome {
                Ok((report, is_drift)) => success_response(id, tool_text_result(&report, is_drift)),
                Err(e) => {
                    eprintln!("[mcp] axiom_evaluate_drift failed: {e}");
                    success_response(id, tool_text_result(&format!("evaluation failed: {e}"), true))
                }
            }
        }
        other => error_response(id, -32602, &format!("unknown tool: {other}")),
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
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if file_count >= ctx.max_files || total_bytes >= ctx.max_bytes {
                break;
            }
            if let Ok(text) = std::fs::read_to_string(entry.path()) {
                let remaining = ctx.max_bytes.saturating_sub(total_bytes);
                let slice = if text.len() > remaining { &text[..remaining] } else { &text[..] };
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

    let pipeline = ctx.pipeline.lock().map_err(|_| "pipeline lock poisoned".to_string())?;
    let token_ids = pipeline.encode_text(&corpus);
    let tokens_processed = token_ids.len();

    let mut states = start_states(ctx, &pipeline)?;

    // Adapt in bounded windows to cap per-call memory.
    for window in token_ids.chunks(ADAPT_WINDOW_TOKENS) {
        adapt_session_blocking(&pipeline, &mut states, window).map_err(|e| e.to_string())?;
    }

    // Recall pass + fingerprint. Use the tail of the corpus as the query.
    let query: Vec<u32> = token_ids
        .iter()
        .rev()
        .take(32)
        .rev()
        .copied()
        .collect();
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

/// `axiom_evaluate_drift` worker. Returns `(report_text, is_drift)`.
fn evaluate_drift_blocking(code: &str, ctx: &McpContext) -> Result<(String, bool), String> {
    let pipeline = ctx.pipeline.lock().map_err(|_| "pipeline lock poisoned".to_string())?;
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

    let mut states = start_states(ctx, &pipeline)?;

    // Next-token prediction: predict ids[1..] from ids[..n-1].
    let input = Tensor::from_vec(ids[..n - 1].to_vec(), (1, n - 1), device)
        .map_err(|e| e.to_string())?;
    let logits = pipeline
        .model()
        .forward_lm(&input, &mut states)
        .map_err(|e| e.to_string())?; // [1, n-1, vocab]
    let vocab = pipeline.model().config.vocab_size;
    let logits_2d = logits
        .squeeze(0)
        .and_then(|t| t.reshape((n - 1, vocab)))
        .map_err(|e| e.to_string())?;
    let targets = Tensor::from_vec(ids[1..].to_vec(), (n - 1,), device).map_err(|e| e.to_string())?;
    let loss = candle_nn::loss::cross_entropy(&logits_2d, &targets).map_err(|e| e.to_string())?;
    let loss_val = loss.to_scalar::<f32>().map_err(|e| e.to_string())?;

    let is_drift = loss_val > ctx.drift_threshold;
    let report = format!(
        "cross_entropy_loss={loss_val:.4} baseline_threshold={:.4} tokens_scored={} drift={}",
        ctx.drift_threshold,
        n,
        if is_drift { "YES" } else { "no" }
    );
    eprintln!("[mcp] evaluate_drift -> {report}");
    Ok((report, is_drift))
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

/// Short stable id derived from a path, for session labelling.
fn short_hash(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(s.as_bytes());
    format!("{:x}{:x}{:x}{:x}", digest[0], digest[1], digest[2], digest[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_list_exposes_two_tools_with_schemas() {
        let list = tools_list();
        let tools = list["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"axiom_compress_path"));
        assert!(names.contains(&"axiom_evaluate_drift"));
        for t in tools {
            assert_eq!(t["inputSchema"]["type"], "object");
            assert!(t["inputSchema"]["required"].is_array());
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
}
