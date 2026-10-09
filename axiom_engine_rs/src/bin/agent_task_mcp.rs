//! `axiom-agent-task` — agent-driven autonomous coding without an LLM API key.
//!
//! Exposes AXIOM's verifier-gated [`agentic`](axiom_engine::agentic) machinery
//! as tools over stdio. An external agent (which *is* a language
//! model) drives the loop: it proposes edit-sets, axiom verifies them with
//! all-or-nothing transactions and byte-for-byte rollback on failure.
//!
//! Two protocol surfaces are served on the same stdio stream:
//!
//! ## Model Context Protocol (MCP)
//!
//! Standard MCP JSON-RPC 2.0 over stdio. Handshake, then tools:
//!
//! ```json
//! // initialize
//! {"jsonrpc":"2.0","id":1,"method":"initialize",
//!  "params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"c","version":"1"}}}
//! // → {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",
//! //     "capabilities":{"tools":{}},"serverInfo":{"name":"axiom-agent-task","version":"..."}}}
//! // client then sends the notifications/initialized notification (no response)
//!
//! // tools/list
//! {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
//! // → {"jsonrpc":"2.0","id":2,"result":{"tools":[
//! //     {"name":"task_start","description":"...",
//! //      "inputSchema":{"type":"object","properties":{...},"required":[...]}},
//! //     ...]}}
//!
//! // tools/call
//! {"jsonrpc":"2.0","id":3,"method":"tools/call",
//!  "params":{"name":"task_start","arguments":{"goal":"...","verify_cmd":"...","files":[],"max_attempts":4}}}
//! // → {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"task_id\":\"...\"}"}]}}
//! ```
//!
//! ## Legacy direct methods (backward compatible)
//!
//! The original method names still work unchanged:
//!
//! ```json
//! // start
//! {"jsonrpc":"2.0","id":1,"method":"task_start",
//!  "params":{"goal":"...","verify_cmd":"...","files":["a.rs"],"max_attempts":4}}
//! // → {"jsonrpc":"2.0","id":1,"result":{"task_id":"..."}}
//!
//! // propose
//! {"jsonrpc":"2.0","id":2,"method":"task_propose",
//!  "params":{"task_id":"...","edits":[{"path":"a.rs","content":"..."}]}}
//! // → {"jsonrpc":"2.0","id":2,
//! //     "result":{"passed":false,"attempt":1,"output":"...","fingerprint":"..."}}
//!
//! // history
//! {"jsonrpc":"2.0","id":3,"method":"task_history","params":{"task_id":"..."}}
//! // → {"jsonrpc":"2.0","id":3,"result":{"attempts":[...]}}
//!
//! // finish
//! {"jsonrpc":"2.0","id":4,"method":"task_finish",
//!  "params":{"task_id":"...","commit":true}}
//! // → {"jsonrpc":"2.0","id":4,"result":{"committed":true}}
//! ```
//!
//! Errors follow JSON-RPC 2.0 `{"jsonrpc":"2.0","id":..,"error":{"code":..,"message":..}}`.
//! MCP `tools/call` results wrap the tool output as text content; tool-level
//! failures are returned inside the result (with `"isError":true`) rather than
//! as JSON-RPC errors, per the MCP specification.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axiom_engine::agent_task::{AgentTask, FileEdit, TaskRegistry};

/// MCP protocol version this server speaks.
const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Crate version, surfaced as the MCP server version.
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Deserialize)]
struct TaskStartParams {
    goal: String,
    verify_cmd: String,
    #[serde(default)]
    files: Vec<String>,
    #[serde(default = "default_max_attempts")]
    max_attempts: usize,
}

/// Use four as the stored attempt limit when omitted from start parameters.
fn default_max_attempts() -> usize {
    4
}

#[derive(Deserialize)]
struct TaskProposeParams {
    task_id: String,
    edits: Vec<EditParam>,
}

#[derive(Deserialize)]
struct EditParam {
    path: String,
    content: String,
}

#[derive(Deserialize)]
struct TaskHistoryParams {
    task_id: String,
}

#[derive(Deserialize)]
struct TaskFinishParams {
    task_id: String,
    #[serde(default = "default_commit")]
    commit: bool,
}

/// Keep edits when finish parameters omit `commit`.
fn default_commit() -> bool {
    true
}

#[derive(Serialize)]
struct RpcError {
    code: i32,
    message: String,
}

/// Serialize a JSON-RPC success response with the supplied request ID and result.
fn ok(id: &Value, result: Value) -> String {
    serde_json::to_string(&json!({"jsonrpc":"2.0","id":id,"result":result})).unwrap()
}

/// Serialize a JSON-RPC error response with the supplied request ID, code, and message.
fn err(id: &Value, code: i32, message: String) -> String {
    let e = RpcError { code, message };
    serde_json::to_string(&json!({"jsonrpc":"2.0","id":id,"error":e})).unwrap()
}

/// Build a unique task ID from nanoseconds since the Unix epoch plus a
/// process-wide sequence number. The counter guards against coarse clocks
/// (e.g. Windows) returning the same timestamp for rapid successive calls.
fn new_task_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("task-{nanos:x}-{seq}")
}

/// Dispatch nonblank stdin lines as JSON-RPC requests and write one response per line.
///
/// Parse errors produce code -32700; unknown methods produce -32601. Requests
/// without an "id" member are notifications and receive no response. EOF or a
/// read error ends the loop without
/// aborting active tasks; stdout write and flush errors are ignored.
///
/// # Panics
///
/// Panics from request handlers propagate and terminate the server.
fn main() {
    let registry = TaskRegistry::new();
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(stdout, "{}", err(&Value::Null, -32700, format!("parse error: {e}")));
                let _ = stdout.flush();
                continue;
            }
        };
        let has_id = msg.get("id").is_some();
        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        let response = match method {
            // MCP protocol methods.
            "initialize" => handle_initialize(&id, params),
            "tools/list" => handle_tools_list(&id),
            "tools/call" => handle_tools_call(&registry, &id, params),
            // MCP lifecycle notification; no response (handled by has_id check).
            "notifications/initialized" => String::new(),
            // Legacy direct methods (backward compatible).
            "task_start" => handle_start(&registry, &id, params),
            "task_propose" => handle_propose(&registry, &id, params),
            "task_history" => handle_history(&registry, &id, params),
            "task_finish" => handle_finish(&registry, &id, params),
            _ => err(&id, -32601, format!("unknown method: {method}")),
        };
        if has_id {
            let _ = writeln!(stdout, "{response}");
            let _ = stdout.flush();
        }
    }
}

/// Snapshot and register a task, returning its ID in a JSON-RPC response.
/// Invalid parameters produce code -32602; snapshot read failures are suppressed.
///
/// # Panics
///
/// Panics if the registry mutex is poisoned.
fn handle_start(registry: &TaskRegistry, id: &Value, params: Value) -> String {
    let p: TaskStartParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return err(id, -32602, format!("invalid params: {e}")),
    };
    let task_id = new_task_id();
    let files: Vec<PathBuf> = p.files.into_iter().map(PathBuf::from).collect();
    match AgentTask::start(task_id.clone(), p.goal, p.verify_cmd, files, p.max_attempts) {
        Ok(task) => {
            registry.insert(task);
            ok(id, json!({"task_id": task_id}))
        }
        Err(e) => err(id, -32000, format!("task_start failed: {e}")),
    }
}

/// Submit edits and return the proposal outcome in a JSON-RPC response.
/// Invalid parameters produce code -32602; an unknown task produces -32001.
/// Apply and verifier execution failures are returned as unsuccessful results.
///
/// # Panics
///
/// Panics if the registry mutex is poisoned or verifier output truncation splits
/// a UTF-8 character at byte 8000.
fn handle_propose(registry: &TaskRegistry, id: &Value, params: Value) -> String {
    let p: TaskProposeParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return err(id, -32602, format!("invalid params: {e}")),
    };
    let edits: Vec<FileEdit> = p
        .edits
        .into_iter()
        .map(|e| FileEdit {
            path: PathBuf::from(e.path),
            content: e.content,
        })
        .collect();
    match registry.with_task(&p.task_id, |t| t.propose(edits)) {
        Some(outcome) => ok(
            id,
            json!({
                "passed": outcome.passed,
                "attempt": outcome.attempt,
                "output": outcome.output,
                "fingerprint": outcome.fingerprint,
            }),
        ),
        None => err(id, -32001, format!("unknown task_id: {}", p.task_id)),
    }
}

/// Return recorded attempts in order in a JSON-RPC response.
/// Invalid parameters produce code -32602; an unknown task produces -32001.
///
/// # Panics
///
/// Panics if the registry mutex is poisoned.
fn handle_history(registry: &TaskRegistry, id: &Value, params: Value) -> String {
    let p: TaskHistoryParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return err(id, -32602, format!("invalid params: {e}")),
    };
    match registry.with_task(&p.task_id, |t| t.history().to_vec()) {
        Some(history) => ok(id, json!({"attempts": history})),
        None => err(id, -32001, format!("unknown task_id: {}", p.task_id)),
    }
}

/// Remove and finish a task, returning the requested `commit` value as `committed`.
/// Abort restoration is best effort; the response does not confirm restoration.
/// Invalid parameters produce code -32602; an unknown task produces -32001.
///
/// # Panics
///
/// Panics if the registry mutex is poisoned.
fn handle_finish(registry: &TaskRegistry, id: &Value, params: Value) -> String {
    let p: TaskFinishParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return err(id, -32602, format!("invalid params: {e}")),
    };
    let result = registry.with_task(&p.task_id, |task| task.finish(p.commit));
    match result {
        Some(Ok(())) => {
            registry.remove(&p.task_id);
            ok(id, json!({"committed": p.commit}))
        }
        Some(Err(e)) => err(id, -32002, format!("finish failed, task retained for retry: {e}")),
        None => err(id, -32001, format!("unknown task_id: {}", p.task_id)),
    }
}

/// Handle MCP `initialize`. Returns the protocol version this server speaks,
/// its tool capability, and server info. The client-declared protocol version
/// is accepted as-is; this server only implements 2024-11-05.
fn handle_initialize(id: &Value, _params: Value) -> String {
    ok(
        id,
        json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "axiom-agent-task", "version": SERVER_VERSION},
        }),
    )
}

/// Describe one MCP tool with its JSON input schema.
fn tool_def(name: &str, description: &str, schema: Value) -> Value {
    json!({"name": name, "description": description, "inputSchema": schema})
}

/// Handle MCP `tools/list`. Exposes the four task operations as MCP tools.
fn handle_tools_list(id: &Value) -> String {
    let tools = vec![
        tool_def(
            "task_start",
            "Start a verifier-gated agent task. Snapshots the listed files and returns a task_id.",
            json!({
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "What the task should achieve."},
                    "verify_cmd": {"type": "string", "description": "Shell command run to verify each proposal; exit 0 means pass."},
                    "files": {"type": "array", "items": {"type": "string"}, "description": "File paths the task may modify."},
                    "max_attempts": {"type": "integer", "description": "Maximum distinct proposals before the task refuses more.", "default": 4},
                },
                "required": ["goal", "verify_cmd"],
            }),
        ),
        tool_def(
            "task_propose",
            "Propose an edit-set for a task. The edits are applied transactionally and the verifier runs; failures roll back byte-for-byte.",
            json!({
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID from task_start."},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                        "description": "Edits to apply atomically.",
                    },
                },
                "required": ["task_id", "edits"],
            }),
        ),
        tool_def(
            "task_history",
            "Return all recorded attempts for a task in order.",
            json!({
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID from task_start."},
                },
                "required": ["task_id"],
            }),
        ),
        tool_def(
            "task_finish",
            "Finish a task. commit=true keeps the last passing edits; commit=false restores the pre-task snapshot.",
            json!({
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID from task_start."},
                    "commit": {"type": "boolean", "description": "Keep edits (true) or restore snapshot (false).", "default": true},
                },
                "required": ["task_id"],
            }),
        ),
    ];
    ok(id, json!({"tools": tools}))
}

/// Parameters for MCP `tools/call`: a tool name plus its arguments object.
#[derive(Deserialize)]
struct ToolsCallParams {
    name: String,
    #[serde(default)]
    arguments: Value,
}

/// Handle MCP `tools/call` by routing to the existing task operation
/// implementations. The inner result is wrapped as MCP text content.
/// Tool-level failures (bad params, unknown task) are returned inside the
/// result with `"isError": true`, per the MCP specification, rather than as
/// JSON-RPC errors.
fn handle_tools_call(registry: &TaskRegistry, id: &Value, params: Value) -> String {
    let p: ToolsCallParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return err(id, -32602, format!("invalid params: {e}")),
    };
    let arguments = if p.arguments.is_null() {
        json!({})
    } else {
        p.arguments
    };
    // Run the underlying operation and capture its result or error payload.
    let inner: Value = match p.name.as_str() {
        "task_start" => match serde_json::from_value::<TaskStartParams>(arguments) {
            Ok(sp) => {
                let task_id = new_task_id();
                let files: Vec<PathBuf> = sp.files.into_iter().map(PathBuf::from).collect();
                match AgentTask::start(
                    task_id.clone(),
                    sp.goal,
                    sp.verify_cmd,
                    files,
                    sp.max_attempts,
                ) {
                    Ok(task) => {
                        registry.insert(task);
                        json!({"task_id": task_id})
                    }
                    Err(e) => return tool_error(id, format!("task_start failed: {e}")),
                }
            }
            Err(e) => return tool_error(id, format!("invalid arguments: {e}")),
        },
        "task_propose" => match serde_json::from_value::<TaskProposeParams>(arguments) {
            Ok(pp) => {
                let edits: Vec<FileEdit> = pp
                    .edits
                    .into_iter()
                    .map(|e| FileEdit {
                        path: PathBuf::from(e.path),
                        content: e.content,
                    })
                    .collect();
                match registry.with_task(&pp.task_id, |t| t.propose(edits)) {
                    Some(outcome) => json!({
                        "passed": outcome.passed,
                        "attempt": outcome.attempt,
                        "output": outcome.output,
                        "fingerprint": outcome.fingerprint,
                    }),
                    None => return tool_error(id, format!("unknown task_id: {}", pp.task_id)),
                }
            }
            Err(e) => return tool_error(id, format!("invalid arguments: {e}")),
        },
        "task_history" => match serde_json::from_value::<TaskHistoryParams>(arguments) {
            Ok(hp) => match registry.with_task(&hp.task_id, |t| t.history().to_vec()) {
                Some(history) => json!({"attempts": history}),
                None => return tool_error(id, format!("unknown task_id: {}", hp.task_id)),
            },
            Err(e) => return tool_error(id, format!("invalid arguments: {e}")),
        },
        "task_finish" => match serde_json::from_value::<TaskFinishParams>(arguments) {
            Ok(fp) => {
                let result = registry.with_task(&fp.task_id, |task| task.finish(fp.commit));
                match result {
                    Some(Ok(())) => {
                        registry.remove(&fp.task_id);
                        json!({"committed": fp.commit})
                    }
                    Some(Err(e)) => {
                        return tool_error(
                            id,
                            format!("finish failed, task retained for retry: {e}"),
                        )
                    }
                    None => return tool_error(id, format!("unknown task_id: {}", fp.task_id)),
                }
            }
            Err(e) => return tool_error(id, format!("invalid arguments: {e}")),
        },
        other => return err(id, -32602, format!("unknown tool: {other}")),
    };
    ok(
        id,
        json!({"content": [{"type": "text", "text": inner.to_string()}]}),
    )
}

/// Serialize an MCP `tools/call` tool-level error: a normal result whose
/// content carries the message and `"isError": true`.
fn tool_error(id: &Value, message: String) -> String {
    ok(
        id,
        json!({
            "content": [{"type": "text", "text": message}],
            "isError": true,
        }),
    )
}
