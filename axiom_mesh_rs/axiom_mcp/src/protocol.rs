//! The Mini Aether wire protocol: line-delimited JSON-RPC 2.0 on stdio.
//!
//! One request per line, one response per line, correlated by `id`. This is
//! the same framing MCP stdio servers use, so a worker written against this
//! module can later graduate to a full MCP server without changing framing.

use serde::{Deserialize, Serialize};

/// The single method a Mini Aether worker must implement.
pub const DISPATCH_METHOD: &str = "aether/dispatch";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    pub params: serde_json::Value,
}

impl RpcRequest {
    pub fn dispatch(id: u64, params: &DispatchParams) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            method: DISPATCH_METHOD.into(),
            params: serde_json::to_value(params).expect("DispatchParams serializes"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse {
    pub jsonrpc: String,
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl RpcResponse {
    pub fn result(id: u64, result: &DispatchResult) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(serde_json::to_value(result).expect("DispatchResult serializes")),
            error: None,
        }
    }

    pub fn error(id: u64, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(RpcError { code, message: message.into() }),
        }
    }
}

/// Params for `aether/dispatch`: a payload already conditioned by the
/// sidecar, plus enough control context for the worker to prioritize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchParams {
    /// The worker backend name this dispatch targets (telemetry).
    pub worker: String,
    /// Scrubbed, compressed payload from the sidecar.
    pub payload: String,
    /// Residual magnitude at dispatch time.
    pub residual_norm: f32,
}

/// A worker's reply to `aether/dispatch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchResult {
    /// Whether the worker acted on the payload.
    pub ok: bool,
    /// Worker output — fed back into sensor fusion upstream.
    pub output: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips_through_json() {
        let params =
            DispatchParams { worker: "claude".into(), payload: "state: x".into(), residual_norm: 0.5 };
        let req = RpcRequest::dispatch(7, &params);
        let line = serde_json::to_string(&req).unwrap();
        let back: RpcRequest = serde_json::from_str(&line).unwrap();
        assert_eq!(back.id, 7);
        assert_eq!(back.method, DISPATCH_METHOD);
        let p: DispatchParams = serde_json::from_value(back.params).unwrap();
        assert_eq!(p.worker, "claude");
    }

    #[test]
    fn error_response_carries_code_and_message() {
        let resp = RpcResponse::error(3, -32601, "method not found");
        let line = serde_json::to_string(&resp).unwrap();
        let back: RpcResponse = serde_json::from_str(&line).unwrap();
        assert!(back.result.is_none());
        assert_eq!(back.error.unwrap().code, -32601);
    }
}
