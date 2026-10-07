//! `axiom-agent-task` — agent-driven autonomous coding without an LLM API key.
//!
//! Exposes AXIOM's verifier-gated [`agentic`](axiom_engine::agentic) machinery
//! as JSON-RPC 2.0 tools over stdio. An external agent (which *is* a language
//! model) drives the loop: it proposes edit-sets, axiom verifies them with
//! all-or-nothing transactions and byte-for-byte rollback on failure.
//!
//! Protocol (each line is one JSON-RPC message):
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

use std::io::{BufRead, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use axiom_engine::agent_task::{AgentTask, FileEdit, TaskRegistry};

#[derive(Deserialize)]
struct TaskStartParams {
    goal: String,
    verify_cmd: String,
    #[serde(default)]
    files: Vec<String>,
    #[serde(default = "default_max_attempts")]
    max_attempts: usize,
}

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

fn default_commit() -> bool {
    true
}

#[derive(Serialize)]
struct RpcError {
    code: i32,
    message: String,
}

fn ok(id: &Value, result: Value) -> String {
    serde_json::to_string(&json!({"jsonrpc":"2.0","id":id,"result":result})).unwrap()
}

fn err(id: &Value, code: i32, message: String) -> String {
    let e = RpcError { code, message };
    serde_json::to_string(&json!({"jsonrpc":"2.0","id":id,"error":e})).unwrap()
}

fn new_task_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("task-{nanos:x}")
}

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
        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        let response = match method {
            "task_start" => handle_start(&registry, &id, params),
            "task_propose" => handle_propose(&registry, &id, params),
            "task_history" => handle_history(&registry, &id, params),
            "task_finish" => handle_finish(&registry, &id, params),
            _ => err(&id, -32601, format!("unknown method: {method}")),
        };
        let _ = writeln!(stdout, "{response}");
        let _ = stdout.flush();
    }
}

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

fn handle_finish(registry: &TaskRegistry, id: &Value, params: Value) -> String {
    let p: TaskFinishParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return err(id, -32602, format!("invalid params: {e}")),
    };
    match registry.remove(&p.task_id) {
        Some(mut task) => {
            task.finish(p.commit);
            ok(id, json!({"committed": p.commit}))
        }
        None => err(id, -32001, format!("unknown task_id: {}", p.task_id)),
    }
}
