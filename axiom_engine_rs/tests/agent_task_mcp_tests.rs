//! The RPC handlers are private to the binary, so exercise their public stdio
//! interface. Requests have bounded waits and every server is reaped on drop.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const RESPONSE_TIMEOUT: Duration = Duration::from_secs(3);

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    responses: Receiver<std::io::Result<String>>,
    reader: Option<JoinHandle<()>>,
    workspace: tempfile::TempDir,
}

impl Server {
    fn new() -> Self {
        let workspace = tempfile::tempdir().unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_axiom-agent-task"))
            .current_dir(workspace.path())
            .env_remove("ANTHROPIC_API_KEY")
            .env_remove("OPENAI_API_KEY")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start axiom-agent-task");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (sender, responses) = mpsc::channel();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Self {
            child,
            stdin: Some(stdin),
            responses,
            reader: Some(reader),
            workspace,
        }
    }

    fn send(&mut self, text: &str) {
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(stdin, "{text}").unwrap();
        stdin.flush().unwrap();
    }

    fn receive(&self) -> Value {
        let line = self
            .responses
            .recv_timeout(RESPONSE_TIMEOUT)
            .expect("server must respond within three seconds")
            .expect("read response line");
        let response: Value = serde_json::from_str(&line).expect("one JSON response per line");
        assert_eq!(response["jsonrpc"], "2.0");
        assert!(response.get("id").is_some());
        assert_ne!(response.get("result").is_some(), response.get("error").is_some());
        response
    }

    fn request(&mut self, id: Value, method: &str, params: Value) -> Value {
        self.send(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        );
        let response = self.receive();
        assert_eq!(response["id"], id);
        response
    }

    fn result(&mut self, method: &str, params: Value) -> Value {
        let response = self.request(json!(1), method, params);
        assert!(response.get("error").is_none(), "{response}");
        response["result"].clone()
    }

    fn start(&mut self, verifier: &str, files: &[&str]) -> String {
        let result = self.result(
            "task_start",
            json!({"goal": "repair fixture", "verify_cmd": verifier, "files": files}),
        );
        let task_id = result["task_id"].as_str().unwrap();
        assert!(!task_id.is_empty());
        task_id.to_owned()
    }

    fn finish_input(&mut self) {
        drop(self.stdin.take());
        let deadline = Instant::now() + RESPONSE_TIMEOUT;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success(), "server exited with {status}");
                return;
            }
            assert!(Instant::now() < deadline, "server did not exit on EOF");
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

fn assert_error(response: &Value, code: i64, message: &str) {
    assert!(response.get("result").is_none(), "{response}");
    assert_eq!(response["error"]["code"], code, "{response}");
    assert!(response["error"]["message"].as_str().unwrap().contains(message), "{response}");
}

#[test]
fn malformed_json_and_blank_lines_do_not_stop_subsequent_requests() {
    let mut server = Server::new();
    server.send("  \n\t\n{not json}");
    let malformed = server.receive();
    assert_eq!(malformed["id"], Value::Null);
    assert_error(&malformed, -32700, "parse error");

    for id in [json!(42), json!("request-\u{03b1}"), Value::Null] {
        let response = server.request(id, "does_not_exist", json!({}));
        assert_error(&response, -32601, "unknown method");
    }
    let task_id = server.start("exit 0", &[]);
    assert_eq!(
        server.result("task_history", json!({"task_id": task_id})),
        json!({"attempts": []})
    );
    server.finish_input();
}

#[test]
fn invalid_params_are_rejected_for_every_tool() {
    let mut server = Server::new();
    let cases = [
        ("task_start", json!({"verify_cmd": "exit 0"})),
        ("task_start", json!({"goal": "goal"})),
        ("task_start", json!({"goal": 12, "verify_cmd": "exit 0"})),
        ("task_start", json!({"goal": "g", "verify_cmd": "exit 0", "files": "a"})),
        ("task_start", json!({"goal": "g", "verify_cmd": "exit 0", "max_attempts": -1})),
        ("task_propose", json!({"edits": []})),
        ("task_propose", json!({"task_id": "missing"})),
        ("task_propose", json!({"task_id": "missing", "edits": [{"path": "a"}]})),
        ("task_propose", json!({"task_id": "missing", "edits": [{"path": 1, "content": "x"}]})),
        ("task_history", json!({})),
        ("task_history", json!({"task_id": 1})),
        ("task_finish", json!({})),
        ("task_finish", json!({"task_id": "missing", "commit": "false"})),
    ];
    for (index, (method, params)) in cases.into_iter().enumerate() {
        let response = server.request(json!(index), method, params);
        assert_error(&response, -32602, "invalid params");
    }
    // Optional start parameters really are optional, even after invalid calls.
    let started = server.result("task_start", json!({"goal": "goal", "verify_cmd": "exit 0"}));
    assert!(started["task_id"].is_string());
}

#[test]
fn missing_and_null_params_are_invalid() {
    let mut server = Server::new();
    for method in ["task_start", "task_propose", "task_history", "task_finish"] {
        server.send(&json!({"jsonrpc": "2.0", "id": "missing", "method": method}).to_string());
        let response = server.receive();
        assert_eq!(response["id"], "missing");
        assert_error(&response, -32602, "invalid params");
        let response = server.request(json!("null"), method, Value::Null);
        assert_error(&response, -32602, "invalid params");
    }
}

#[test]
fn unknown_task_ids_return_application_errors() {
    let mut server = Server::new();
    for method in ["task_propose", "task_history", "task_finish"] {
        let response = server.request(
            json!(method),
            method,
            json!({"task_id": "absent", "edits": [], "commit": false}),
        );
        assert_error(&response, -32001, "unknown task_id: absent");
    }
}

#[test]
fn successful_lifecycle_preserves_contents_and_defaults_to_commit() {
    let mut server = Server::new();
    let path = server.workspace.path().join("file.txt");
    fs::write(&path, "original").unwrap();
    let task_id = server.start("exit 0", &["file.txt"]);
    assert_eq!(server.result("task_history", json!({"task_id": task_id}))["attempts"], json!([]));

    let content = "updated\n\"quoted\" \\ \u{03bb}\n";
    let proposed = server.result(
        "task_propose",
        json!({"task_id": task_id, "edits": [{"path": "file.txt", "content": content}]}),
    );
    assert_eq!(proposed["passed"], true);
    assert_eq!(proposed["attempt"], 1);
    assert_eq!(proposed["output"], "");
    assert!(!proposed["fingerprint"].as_str().unwrap().is_empty());
    assert_eq!(fs::read_to_string(&path).unwrap(), content);
    assert_eq!(
        server.result("task_history", json!({"task_id": task_id})),
        json!({"attempts": [proposed]})
    );
    assert_eq!(
        server.result("task_finish", json!({"task_id": task_id})),
        json!({"committed": true})
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), content);
    for method in ["task_history", "task_propose", "task_finish"] {
        let response = server.request(json!(2), method, json!({"task_id": task_id, "edits": []}));
        assert_error(&response, -32001, "unknown task_id");
    }
}

#[test]
fn abort_restores_original_bytes_and_removes_new_files() {
    let mut server = Server::new();
    let path = server.workspace.path().join("original.dat");
    let created = server.workspace.path().join("created.txt");
    let original = [0xff, 0, 0x80];
    fs::write(&path, original).unwrap();
    let task_id = server.start("exit 0", &["original.dat", "created.txt"]);
    let proposed = server.result(
        "task_propose",
        json!({"task_id": task_id, "edits": [
            {"path": "original.dat", "content": "changed"},
            {"path": "created.txt", "content": "new"}
        ]}),
    );
    assert_eq!(proposed["passed"], true);
    assert_eq!(fs::read_to_string(&path).unwrap(), "changed");
    assert!(created.exists());
    assert_eq!(
        server.result("task_finish", json!({"task_id": task_id, "commit": false})),
        json!({"committed": false})
    );
    assert_eq!(fs::read(&path).unwrap(), original);
    assert!(!created.exists());
}

#[test]
fn rejected_and_empty_proposals_are_results_and_history_is_ordered() {
    let mut server = Server::new();
    let path = server.workspace.path().join("file.txt");
    fs::write(&path, "original").unwrap();
    let task_id = server.start("exit 1", &["file.txt"]);
    let params = json!({"task_id": task_id, "edits": [{"path": "file.txt", "content": "bad"}]});
    let first = server.result("task_propose", params.clone());
    assert_eq!(first["passed"], false);
    assert_eq!(first["attempt"], 1);
    assert_eq!(fs::read_to_string(&path).unwrap(), "original");
    let duplicate = server.result("task_propose", params);
    assert_eq!(duplicate["passed"], false);
    assert_eq!(duplicate["attempt"], 2);
    assert_eq!(duplicate["fingerprint"], first["fingerprint"]);
    assert!(duplicate["output"].as_str().unwrap().contains("already rejected"));
    let empty = server.result("task_propose", json!({"task_id": task_id, "edits": []}));
    assert_eq!(empty["passed"], false);
    assert_eq!(empty["attempt"], 3);
    assert_eq!(empty["fingerprint"], "");
    assert!(empty["output"].as_str().unwrap().contains("empty edit-set"));
    assert_eq!(
        server.result("task_history", json!({"task_id": task_id})),
        json!({"attempts": [first, duplicate, empty]})
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "original");
}

#[test]
fn finishing_one_task_does_not_remove_another() {
    let mut server = Server::new();
    let first = server.start("exit 0", &[]);
    let second = server.start("exit 0", &[]);
    assert_ne!(first, second);
    server.result("task_propose", json!({"task_id": first, "edits": []}));
    server.result("task_finish", json!({"task_id": first, "commit": true}));
    assert_eq!(
        server.result("task_history", json!({"task_id": second})),
        json!({"attempts": []})
    );
    server.result("task_finish", json!({"task_id": second, "commit": true}));
}
