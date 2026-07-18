//! End-to-end test of the stdio transport against the reference worker
//! binary — a real child process, real pipes, real JSON-RPC framing.

use axiom_mcp::protocol::DispatchParams;
use axiom_mcp::transport::{StdioTransport, WorkerTransport};

fn params(payload: &str, norm: f32) -> DispatchParams {
    DispatchParams { worker: "claude".into(), payload: payload.into(), residual_norm: norm }
}

#[tokio::test]
async fn dispatch_round_trips_through_a_real_child_process() {
    let transport =
        StdioTransport::spawn(env!("CARGO_BIN_EXE_aether_worker"), &[]).expect("spawn worker");

    let res = transport.dispatch(params("residual norm: 0.9", 0.9)).await.expect("dispatch ok");
    assert!(res.ok);
    assert!(res.output.contains("18 bytes"), "got: {}", res.output);

    // Second dispatch on the same connection: id correlation must hold.
    let res2 = transport.dispatch(params("x", 0.1)).await.expect("second dispatch ok");
    assert!(res2.output.contains("1 bytes"), "got: {}", res2.output);
}

#[tokio::test]
async fn concurrent_dispatches_serialize_without_cross_talk() {
    let transport = std::sync::Arc::new(
        StdioTransport::spawn(env!("CARGO_BIN_EXE_aether_worker"), &[]).expect("spawn worker"),
    );

    let mut handles = Vec::new();
    for i in 0..8usize {
        let t = std::sync::Arc::clone(&transport);
        handles.push(tokio::spawn(async move {
            let payload = "p".repeat(i + 1);
            let res = t.dispatch(params(&payload, i as f32)).await.expect("dispatch ok");
            assert!(
                res.output.contains(&format!("{} bytes", i + 1)),
                "task {i} got wrong response: {}",
                res.output
            );
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
}
