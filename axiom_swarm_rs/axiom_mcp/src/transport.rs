//! Worker transports: how a dispatched payload physically reaches a worker.
//!
//! [`WorkerTransport`] is deliberately dyn-safe (it returns a boxed future)
//! so the orchestrator can hold heterogeneous transports — a stdio child
//! process for one node, an in-process mock for another — behind one
//! `Arc<dyn WorkerTransport>`.

use std::ffi::OsStr;
use std::future::Future;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::protocol::{DispatchParams, DispatchResult, RpcRequest, RpcResponse};

/// Default per-request timeout for [`StdioTransport::spawn`]. A worker that
/// wedges (crashes without exiting, deadlocks, gets stuck on its own I/O)
/// would otherwise hold the orchestrator's `AwaitingWorkers` state forever —
/// the FSM has no other way to notice a stalled call.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("codec: {0}")]
    Codec(#[from] serde_json::Error),
    #[error("worker closed the stream")]
    Closed,
    #[error("worker error {code}: {message}")]
    Worker { code: i64, message: String },
    #[error("worker dispatch timed out after {0:?}")]
    Timeout(Duration),
}

pub type DispatchFuture<'a> =
    Pin<Box<dyn Future<Output = Result<DispatchResult, TransportError>> + Send + 'a>>;

/// A dyn-safe async dispatch channel to one worker.
pub trait WorkerTransport: Send + Sync {
    fn dispatch(&self, params: DispatchParams) -> DispatchFuture<'_>;
}

/// JSON-RPC over a child process's stdio — the production transport for
/// Mini Aether workers. Requests are serialized one per line; responses
/// are correlated by id. The child is killed when the transport drops.
pub struct StdioTransport {
    next_id: AtomicU64,
    io: Mutex<(ChildStdin, Lines<BufReader<ChildStdout>>)>,
    _child: Child,
    request_timeout: Duration,
}

impl StdioTransport {
    /// Spawn a worker process with [`DEFAULT_REQUEST_TIMEOUT`] per call.
    pub fn spawn(program: impl AsRef<OsStr>, args: &[&str]) -> Result<Self, TransportError> {
        Self::spawn_with_timeout(program, args, DEFAULT_REQUEST_TIMEOUT)
    }

    /// Spawn a worker process and take ownership of its stdio, bounding every
    /// `dispatch` call to `request_timeout`.
    pub fn spawn_with_timeout(
        program: impl AsRef<OsStr>,
        args: &[&str],
        request_timeout: Duration,
    ) -> Result<Self, TransportError> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let stdin = child.stdin.take().ok_or(TransportError::Closed)?;
        let stdout = BufReader::new(child.stdout.take().ok_or(TransportError::Closed)?).lines();
        Ok(Self {
            next_id: AtomicU64::new(1),
            io: Mutex::new((stdin, stdout)),
            _child: child,
            request_timeout,
        })
    }

    async fn dispatch_inner(
        &self,
        params: DispatchParams,
    ) -> Result<DispatchResult, TransportError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let line = serde_json::to_string(&RpcRequest::dispatch(id, &params))?;

        // One in-flight request per worker at a time: the lock spans
        // write → matching response, which also serializes callers.
        let mut io = self.io.lock().await;
        io.0.write_all(line.as_bytes()).await?;
        io.0.write_all(b"\n").await?;
        io.0.flush().await?;

        loop {
            let Some(line) = io.1.next_line().await? else {
                return Err(TransportError::Closed);
            };
            if line.trim().is_empty() {
                continue;
            }
            // Non-protocol noise on stdout is skipped, not fatal.
            let Ok(resp) = serde_json::from_str::<RpcResponse>(&line) else { continue };
            if resp.id != id {
                continue; // stale response from a previous, timed-out call
            }
            if let Some(err) = resp.error {
                return Err(TransportError::Worker { code: err.code, message: err.message });
            }
            let result = resp.result.ok_or(TransportError::Closed)?;
            return Ok(serde_json::from_value(result)?);
        }
    }
}

impl WorkerTransport for StdioTransport {
    fn dispatch(&self, params: DispatchParams) -> DispatchFuture<'_> {
        Box::pin(async move {
            // On timeout the mutex guard inside `dispatch_inner` is dropped
            // (the future is cancelled), not held — a later call can still
            // acquire it. Its eventual, late response is simply skipped by
            // the `resp.id != id` check above, so no special recovery is
            // needed here beyond surfacing the timeout to the caller.
            match tokio::time::timeout(self.request_timeout, self.dispatch_inner(params)).await {
                Ok(result) => result,
                Err(_) => Err(TransportError::Timeout(self.request_timeout)),
            }
        })
    }
}

/// In-process transport for tests and demos: acknowledges every dispatch.
#[derive(Debug, Default)]
pub struct MockTransport {
    /// When set, every dispatch fails with this worker error message.
    pub fail_with: Option<String>,
}

impl WorkerTransport for MockTransport {
    fn dispatch(&self, params: DispatchParams) -> DispatchFuture<'_> {
        Box::pin(async move {
            if let Some(msg) = &self.fail_with {
                return Err(TransportError::Worker { code: -1, message: msg.clone() });
            }
            Ok(DispatchResult {
                ok: true,
                output: format!(
                    "mock[{}] ack {} bytes at residual {:.4}",
                    params.worker,
                    params.payload.len(),
                    params.residual_norm
                ),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_transport_acks() {
        let t = MockTransport::default();
        let res = t
            .dispatch(DispatchParams {
                worker: "codex".into(),
                payload: "abc".into(),
                residual_norm: 1.0,
            })
            .await
            .unwrap();
        assert!(res.ok);
        assert!(res.output.contains("3 bytes"));
    }

    #[tokio::test]
    async fn mock_transport_fails_when_told() {
        let t = MockTransport { fail_with: Some("down".into()) };
        let err = t
            .dispatch(DispatchParams { worker: "x".into(), payload: "".into(), residual_norm: 0.0 })
            .await
            .unwrap_err();
        assert!(matches!(err, TransportError::Worker { .. }));
    }

    #[tokio::test]
    async fn stdio_transport_times_out_on_a_wedged_worker() {
        // `sleep 5` never reads stdin or writes stdout — the closest
        // portable stand-in for a worker process that has hung. A short
        // request_timeout must surface Timeout rather than hang the test.
        let t = StdioTransport::spawn_with_timeout("sleep", &["5"], Duration::from_millis(200))
            .expect("spawn sleep");
        let err = t
            .dispatch(DispatchParams { worker: "wedged".into(), payload: "x".into(), residual_norm: 0.0 })
            .await
            .unwrap_err();
        assert!(matches!(err, TransportError::Timeout(_)), "got: {err:?}");
    }
}
