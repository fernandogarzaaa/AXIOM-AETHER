//! Genuine local token streaming: the Ollama-backed chat path must emit
//! tokens as they arrive instead of buffering the whole answer.
//!
//! A fake Ollama HTTP server (raw TCP, no live Ollama in this sandbox)
//! emits NDJSON chunks with delays. Assertions:
//! 1. The first chunk reaches the client before the full body completes.
//! 2. A slow consumer bounds memory: the producer blocks on the bounded
//!    channel instead of accumulating the whole stream.
//! 3. Client disconnect aborts the upstream Ollama request.
//! 4. The non-stream path still returns one complete JSON object.
//!
//! Failing-test-first: these tests reference `SwarmRouter::route_chat_stream`,
//! which does not exist on the old buffered code, so they fail (compile
//! error) until the streaming dispatch lands.

use axiom_engine::swarm_router::{SwarmRouter, SwarmRouterConfig};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::{sleep, timeout};

fn router_for(base_url: &str) -> SwarmRouter {
    SwarmRouter::new(SwarmRouterConfig {
        enabled: true,
        base_url: base_url.to_string(),
        model_candidates: vec!["phi4:3.8b".to_string()],
        num_ctx: 4096,
        timeout_ms: 5_000,
        mesh_routing: false,
    })
}

fn chat_payload() -> Value {
    json!({"messages": [{"role": "user", "content": "hello"}]})
}

/// One NDJSON content line the fake Ollama emits, after `delay_before`.
#[derive(Clone)]
struct FakeChunk {
    content: String,
    delay_before: Duration,
}

#[derive(Default)]
struct FakeStats {
    /// Content lines fully written to the socket.
    chunks_sent: AtomicUsize,
    /// Bytes fully written to the socket (content lines + done line).
    bytes_written: AtomicUsize,
    /// A socket write failed, i.e. the peer went away mid-stream.
    aborted: AtomicBool,
    /// The fake saw `"stream": true` in a /api/chat request body.
    saw_stream_true: AtomicBool,
}

struct FakeOllama {
    base_url: String,
    stats: Arc<FakeStats>,
}

impl FakeOllama {
    async fn spawn(chunks: Vec<FakeChunk>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let stats = Arc::new(FakeStats::default());
        let serve_stats = stats.clone();
        tokio::spawn(async move {
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    return;
                };
                let chunks = chunks.clone();
                let stats = serve_stats.clone();
                tokio::spawn(async move {
                    handle_connection(socket, chunks, stats).await;
                });
            }
        });
        Self {
            base_url: format!("http://{addr}"),
            stats,
        }
    }
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> Option<(String, String, Vec<u8>)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = socket.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if buf.len() > 65_536 {
            return None;
        }
    }
    let header_end = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|pos| pos + 4)?;
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or("").to_string();
    let mut content_length = 0usize;
    for line in lines {
        if let Some(value) = line.strip_prefix("Content-Length:") {
            content_length = value.trim().parse().unwrap_or(0);
        } else if let Some(value) = line.strip_prefix("content-length:") {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        let n = socket.read(&mut tmp).await.ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(content_length);
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    Some((method, path, body))
}

async fn handle_connection(
    mut socket: tokio::net::TcpStream,
    chunks: Vec<FakeChunk>,
    stats: Arc<FakeStats>,
) {
    let Some((method, path, body)) = read_request(&mut socket).await else {
        return;
    };
    if method == "GET" && path == "/api/tags" {
        let body = r#"{"models":[{"name":"phi4:3.8b"}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = socket.write_all(response.as_bytes()).await;
        return;
    }
    if method != "POST" || path != "/api/chat" {
        let _ = socket
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .await;
        return;
    }
    let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let stream = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if stream {
        stats.saw_stream_true.store(true, Ordering::SeqCst);
        let head =
            "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n";
        if socket.write_all(head.as_bytes()).await.is_err() {
            stats.aborted.store(true, Ordering::SeqCst);
            return;
        }
        for chunk in &chunks {
            sleep(chunk.delay_before).await;
            let line = serde_json::to_string(&json!({
                "message": {"role": "assistant", "content": chunk.content},
                "done": false,
            }))
            .unwrap()
                + "\n";
            if socket.write_all(line.as_bytes()).await.is_err() {
                stats.aborted.store(true, Ordering::SeqCst);
                return;
            }
            if socket.flush().await.is_err() {
                stats.aborted.store(true, Ordering::SeqCst);
                return;
            }
            stats.chunks_sent.fetch_add(1, Ordering::SeqCst);
            stats.bytes_written.fetch_add(line.len(), Ordering::SeqCst);
        }
        let done = serde_json::to_string(&json!({
            "message": {"role": "assistant", "content": ""},
            "done": true,
        }))
        .unwrap()
            + "\n";
        if socket.write_all(done.as_bytes()).await.is_err() {
            stats.aborted.store(true, Ordering::SeqCst);
            return;
        }
        stats.bytes_written.fetch_add(done.len(), Ordering::SeqCst);
        return;
    }
    // Non-stream: wait out the chunk delays, then answer with one object.
    let total_delay: Duration = chunks.iter().map(|c| c.delay_before).sum();
    sleep(total_delay).await;
    let full: String = chunks.iter().map(|c| c.content.clone()).collect();
    let body = serde_json::to_string(&json!({
        "message": {"role": "assistant", "content": full},
        "done": true,
    }))
    .unwrap();
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = socket.write_all(response.as_bytes()).await;
}

#[tokio::test]
async fn first_chunk_arrives_before_full_body_completes() {
    let fake = FakeOllama::spawn(vec![
        FakeChunk {
            content: "Hel".to_string(),
            delay_before: Duration::from_millis(0),
        },
        FakeChunk {
            content: "lo ".to_string(),
            delay_before: Duration::from_millis(500),
        },
        FakeChunk {
            content: "world".to_string(),
            delay_before: Duration::from_millis(500),
        },
    ])
    .await;
    let router = router_for(&fake.base_url);
    let mut stream = router
        .route_chat_stream(&chat_payload())
        .await
        .expect("streaming dispatch must succeed");
    assert_eq!(stream.model, "phi4:3.8b");
    let first = timeout(Duration::from_secs(10), stream.receiver.recv())
        .await
        .expect("first chunk must arrive promptly")
        .expect("channel must stay open for the first chunk");
    let first = first.expect("first chunk must not be an upstream error");
    assert_eq!(first, "Hel");
    assert!(
        fake.stats.saw_stream_true.load(Ordering::SeqCst),
        "dispatch must POST /api/chat with \"stream\": true"
    );
    // The old buffered code could only deliver anything after the whole
    // upstream body arrived (about 1000 ms of chunk delays). Here the first
    // chunk arrives while the fake has sent exactly one content line.
    assert_eq!(
        fake.stats.chunks_sent.load(Ordering::SeqCst),
        1,
        "first chunk reached the client before the fake sent the rest"
    );
    // Drain the rest; the channel closes when the producer finishes.
    let mut rest = String::new();
    while let Some(item) = timeout(Duration::from_secs(10), stream.receiver.recv())
        .await
        .expect("later chunks must arrive")
    {
        rest.push_str(&item.expect("later chunk must not be an upstream error"));
    }
    assert_eq!(rest, "lo world");
}

#[tokio::test]
async fn slow_consumer_bounds_memory_via_backpressure() {
    // 512 chunks of 64 KiB = 32 MiB total. The bounded channel holds 16
    // deltas, so once it fills the producer must stop reading the socket and
    // the fake's writes must stall: bytes written while nobody drains stay
    // far below the 32 MiB total.
    let big = "x".repeat(64 * 1024);
    let chunks: Vec<FakeChunk> = (0..512)
        .map(|_| FakeChunk {
            content: big.clone(),
            delay_before: Duration::from_millis(0),
        })
        .collect();
    let fake = FakeOllama::spawn(chunks).await;
    let router = router_for(&fake.base_url);
    let mut stream = router
        .route_chat_stream(&chat_payload())
        .await
        .expect("streaming dispatch must succeed");
    // Do not poll the receiver at all for 800 ms.
    sleep(Duration::from_millis(800)).await;
    let written = fake.stats.bytes_written.load(Ordering::SeqCst);
    assert!(
        written < 4 * 1024 * 1024,
        "slow consumer must bound memory: fake wrote {written} bytes of 32 MiB while nobody drained"
    );
    assert!(
        written > 0,
        "producer must have started emitting before the slow-consumer check"
    );
    // Draining delivers every chunk, in order.
    let mut received = 0usize;
    while let Some(item) = timeout(Duration::from_secs(30), stream.receiver.recv())
        .await
        .expect("drain must not hang")
    {
        let delta = item.expect("chunk must not be an upstream error");
        assert_eq!(delta.len(), 64 * 1024);
        received += 1;
    }
    assert_eq!(received, 512, "all chunks must arrive once drained");
}

#[tokio::test]
async fn client_disconnect_aborts_upstream_request() {
    let fake = FakeOllama::spawn(
        (0..50)
            .map(|i| FakeChunk {
                content: format!("chunk-{i} "),
                delay_before: Duration::from_millis(200),
            })
            .collect(),
    )
    .await;
    let router = router_for(&fake.base_url);
    let mut stream = router
        .route_chat_stream(&chat_payload())
        .await
        .expect("streaming dispatch must succeed");
    let first = timeout(Duration::from_secs(10), stream.receiver.recv())
        .await
        .expect("first chunk must arrive")
        .expect("channel must stay open")
        .expect("first chunk must not be an upstream error");
    assert!(first.starts_with("chunk-0"));
    // Client disconnect: drop the receiver. The producer must abort the
    // upstream Ollama request instead of reading the remaining 49 chunks.
    drop(stream);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if fake.stats.aborted.load(Ordering::SeqCst) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "fake Ollama never observed the upstream abort after client disconnect"
        );
        sleep(Duration::from_millis(50)).await;
    }
    assert!(
        fake.stats.chunks_sent.load(Ordering::SeqCst) < 50,
        "upstream must stop emitting after the abort"
    );
}

#[tokio::test]
async fn non_stream_path_still_returns_one_json_object() {
    let fake = FakeOllama::spawn(vec![
        FakeChunk {
            content: "Hel".to_string(),
            delay_before: Duration::from_millis(300),
        },
        FakeChunk {
            content: "lo ".to_string(),
            delay_before: Duration::from_millis(300),
        },
        FakeChunk {
            content: "world".to_string(),
            delay_before: Duration::from_millis(300),
        },
    ])
    .await;
    let router = router_for(&fake.base_url);
    let result = router
        .route_chat_payload(&chat_payload())
        .await
        .expect("non-stream dispatch must succeed");
    assert_eq!(result.model, "phi4:3.8b");
    assert_eq!(result.content, "Hello world");
    assert!(
        !fake.stats.saw_stream_true.load(Ordering::SeqCst),
        "non-stream dispatch must not request a stream"
    );
}

#[tokio::test]
async fn first_token_latency_is_recorded() {
    let fake = FakeOllama::spawn(vec![FakeChunk {
        content: "hi".to_string(),
        delay_before: Duration::from_millis(0),
    }])
    .await;
    let router = router_for(&fake.base_url);
    let mut stream = router
        .route_chat_stream(&chat_payload())
        .await
        .expect("streaming dispatch must succeed");
    while let Some(item) = timeout(Duration::from_secs(10), stream.receiver.recv())
        .await
        .expect("chunks must arrive")
    {
        let _ = item.expect("chunk must not be an upstream error");
    }
    let rendered = axiom_engine::metrics::render_metrics();
    assert!(
        rendered.contains("axiom_first_token_latency_seconds_count"),
        "first-token latency histogram must be rendered, got:\n{rendered}"
    );
    let count_line = rendered
        .lines()
        .find(|line| line.starts_with("axiom_first_token_latency_seconds_count"))
        .expect("count line must exist");
    let count: u64 = count_line
        .split_whitespace()
        .nth(1)
        .expect("count value must exist")
        .parse()
        .expect("count must parse");
    assert!(
        count >= 1,
        "at least one first-token observation must be recorded"
    );
}
