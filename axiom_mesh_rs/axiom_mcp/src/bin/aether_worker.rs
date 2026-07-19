//! Reference Mini Aether worker: line-delimited JSON-RPC 2.0 on stdio.
//!
//! Speaks the `aether/dispatch` protocol and acknowledges every payload.
//! This is the process a `StdioTransport` spawns in tests and demos; a real
//! backend adapter (claude / codex / gemini CLI) replaces the handler body
//! while keeping the framing identical.

use std::io::{BufRead, Write};

use axiom_mcp::protocol::{DispatchParams, DispatchResult, RpcRequest, RpcResponse, DISPATCH_METHOD};

fn handle(req: RpcRequest) -> RpcResponse {
    if req.method != DISPATCH_METHOD {
        return RpcResponse::error(req.id, -32601, format!("unknown method {}", req.method));
    }
    match serde_json::from_value::<DispatchParams>(req.params) {
        Ok(p) => RpcResponse::result(
            req.id,
            &DispatchResult {
                ok: true,
                output: format!(
                    "worker[{}] processed {} bytes at residual {:.4}",
                    p.worker,
                    p.payload.len(),
                    p.residual_norm
                ),
            },
        ),
        Err(e) => RpcResponse::error(req.id, -32602, format!("invalid params: {e}")),
    }
}

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<RpcRequest>(&line) {
            Ok(req) => handle(req),
            Err(e) => RpcResponse::error(0, -32700, format!("parse error: {e}")),
        };
        let mut out = stdout.lock();
        let _ = serde_json::to_writer(&mut out, &response);
        let _ = out.write_all(b"\n");
        let _ = out.flush();
    }
}
