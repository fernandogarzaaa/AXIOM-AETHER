//! The stdio MCP server keeps its state per user, not in the directory the
//! host starts it in.
//!
//! Claude Code (and every other MCP host) starts `axiom --mode mcp` in the
//! project that is open. With working-directory defaults, a remember call or a
//! task post wrote `checkpoints/memory/`, `checkpoints/tasks/` and
//! `axiom_master_vibe.bin` into that repo and gave each repo its own memory.
//! These tests drive the real `axiom` binary over stdio with `HOME` pointed at
//! a temporary directory and every `AXIOM_*` variable cleared.
//! Unix only: they point `HOME` (not `USERPROFILE`) at the fake home.
#![cfg(unix)]

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::{json, Value};

/// Generous: a debug build boots the model before it answers the first call.
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(180);

struct Sandbox {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    home: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();
        let home = root.join("home");
        std::fs::create_dir_all(&home).unwrap();
        Self {
            _tmp: tmp,
            root,
            home,
        }
    }

    /// A new, empty directory to start a server in.
    fn project(&self, name: &str) -> PathBuf {
        let dir = self.root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    responses: Receiver<String>,
    stdout_reader: Option<JoinHandle<()>>,
    stderr_reader: Option<JoinHandle<String>>,
    next_id: u64,
}

impl Server {
    /// `axiom --mode mcp` in `cwd` with only PATH, HOME and `env` set.
    fn start(cwd: &Path, home: &Path, env: &[(&str, &Path)]) -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_axiom"));
        cmd.args(["--mode", "mcp"])
            .current_dir(cwd)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", home)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in env {
            cmd.env(key, value);
        }
        let mut child = cmd.spawn().expect("start axiom --mode mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let (sender, responses) = mpsc::channel();
        let stdout_reader = thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        let stderr_reader = thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });
        Self {
            child,
            stdin: Some(stdin),
            responses,
            stdout_reader: Some(stdout_reader),
            stderr_reader: Some(stderr_reader),
            next_id: 1,
        }
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        let line = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(stdin, "{line}").unwrap();
        stdin.flush().unwrap();
        let reply = self
            .responses
            .recv_timeout(RESPONSE_TIMEOUT)
            .unwrap_or_else(|e| panic!("no response to {method}: {e}"));
        let reply: Value = serde_json::from_str(&reply).expect("one JSON response per line");
        assert_eq!(reply["id"], id, "response to {method}");
        assert!(reply.get("error").is_none(), "{method} failed: {reply}");
        reply["result"].clone()
    }

    /// Call a tool and return its text, failing the test on `isError`.
    fn call(&mut self, tool: &str, arguments: Value) -> String {
        let result = self.request("tools/call", json!({"name": tool, "arguments": arguments}));
        let text = result["content"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        assert_eq!(result["isError"], false, "{tool} reported an error: {text}");
        text
    }

    /// Close stdin, wait for a clean exit, and return everything on stderr.
    fn shut_down(mut self) -> String {
        drop(self.stdin.take());
        let status = self.child.wait().unwrap();
        self.stdout_reader.take().unwrap().join().unwrap();
        let stderr = self.stderr_reader.take().unwrap().join().unwrap();
        assert!(status.success(), "server exited with {status}:\n{stderr}");
        stderr
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Every path under `dir`, relative to it, sorted.
fn tree(dir: &Path) -> Vec<String> {
    fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            out.push(path.strip_prefix(base).unwrap().display().to_string());
            if path.is_dir() {
                walk(base, &path, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The jsonl files directly in `dir`, concatenated.
fn jsonl_in(dir: &Path) -> String {
    let mut text = String::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        if entry.path().extension().and_then(|e| e.to_str()) == Some("jsonl") {
            text.push_str(&read(&entry.path()));
        }
    }
    text
}

/// `axiom init --no-fetch` in a new, empty directory with a short offline
/// bootstrap; the directory must stay empty and the kernel land next to the
/// models. Returns the kernel path.
fn init_home(sandbox: &Sandbox) -> PathBuf {
    let cwd = sandbox.project("init-cwd");
    let output = Command::new(env!("CARGO_BIN_EXE_axiom"))
        .args(["init", "--no-fetch"])
        .current_dir(&cwd)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", &sandbox.home)
        .env("AXIOM_INIT_EPOCHS", "1")
        .env("AXIOM_INIT_STEPS", "3")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "axiom init failed:\n{log}");
    assert_eq!(
        tree(&cwd),
        Vec::<String>::new(),
        "init wrote into its cwd\n{log}"
    );
    let kernel = sandbox
        .home
        .join(".axiom")
        .join("models")
        .join("axiom_kernel_v1.safetensors");
    assert!(kernel.is_file(), "no kernel at {}\n{log}", kernel.display());
    let memory = sandbox.home.join(".axiom").join("memory");
    assert!(
        log.contains(&format!("memory: {}", memory.display())),
        "init does not say where memory goes\n{log}"
    );
    kernel
}

#[test]
fn init_writes_the_kernel_under_home_and_the_server_loads_it() {
    let sandbox = Sandbox::new();
    let kernel = init_home(&sandbox);

    let project = sandbox.project("project");
    let mut server = Server::start(&project, &sandbox.home, &[]);
    server.request("tools/list", json!({}));
    let stderr = server.shut_down();
    assert!(
        !stderr.contains("No pre-trained checkpoint found"),
        "the server did not load {}:\n{stderr}",
        kernel.display()
    );
    assert_eq!(tree(&project), Vec::<String>::new());
}

/// Remember something and post a task, then shut the server down.
fn remember_and_post(server: &mut Server, memory: &str, task: &str) {
    server.request(
        "initialize",
        json!({"protocolVersion": "2026-07-28", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}),
    );
    let remembered = server.call(
        "axiom_remember",
        json!({"text": memory, "kind": "decision"}),
    );
    assert!(remembered.contains("remembered"), "{remembered}");
    let posted = server.call(
        "axiom_post_task",
        json!({"channel": "review", "description": task}),
    );
    assert!(!posted.is_empty());
}

#[test]
fn stdio_server_with_variables_unset_keeps_state_under_home() {
    let sandbox = Sandbox::new();
    let project = sandbox.project("project");

    let mut server = Server::start(&project, &sandbox.home, &[]);
    remember_and_post(
        &mut server,
        "we chose sqlite over postgres for the cache",
        "review the cache layer",
    );
    let stderr = server.shut_down();

    assert_eq!(
        tree(&project),
        Vec::<String>::new(),
        "the server wrote into the directory it was started in\n{stderr}"
    );
    let axiom = sandbox.home.join(".axiom");
    assert!(
        read(&axiom.join("memory").join("personal.jsonl"))
            .contains("we chose sqlite over postgres for the cache"),
        "memory not under ~/.axiom/memory\n{stderr}"
    );
    assert!(
        jsonl_in(&axiom.join("tasks")).contains("review the cache layer"),
        "task not under ~/.axiom/tasks\n{stderr}"
    );
    assert!(
        stderr.contains(&format!(
            "master vibe at {}",
            axiom.join("axiom_master_vibe.bin").display()
        )),
        "{stderr}"
    );
}

#[test]
fn explicit_variables_still_win() {
    let sandbox = Sandbox::new();
    let project = sandbox.project("project");
    let memory = sandbox.root.join("custom").join("memory");
    let tasks = sandbox.root.join("custom").join("tasks");
    let vibe = sandbox.root.join("custom").join("vibe.bin");

    let mut server = Server::start(
        &project,
        &sandbox.home,
        &[
            ("AXIOM_MEMORY_DIR", &memory),
            ("AXIOM_TASK_DIR", &tasks),
            ("AXIOM_VIBE_PATH", &vibe),
        ],
    );
    remember_and_post(&mut server, "custom memory", "custom task");
    let stderr = server.shut_down();

    assert!(read(&memory.join("personal.jsonl")).contains("custom memory"));
    assert!(jsonl_in(&tasks).contains("custom task"));
    assert!(
        stderr.contains(&format!("master vibe at {}", vibe.display())),
        "{stderr}"
    );
    assert_eq!(tree(&project), Vec::<String>::new());
    let axiom = sandbox.home.join(".axiom");
    assert!(!axiom.join("memory").exists(), "{:?}", tree(&axiom));
    assert!(!axiom.join("tasks").exists(), "{:?}", tree(&axiom));
}

/// The shell block under "### Moving existing memory" in docs/MCP-CLIENTS.md.
fn documented_move_script() -> String {
    let doc = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/MCP-CLIENTS.md");
    let doc = read(&doc);
    let section = doc
        .split("### Moving existing memory")
        .nth(1)
        .expect("docs/MCP-CLIENTS.md has a \"Moving existing memory\" section");
    let block = section
        .split("```bash\n")
        .nth(1)
        .expect("the section has a bash block");
    block.split("```").next().unwrap().to_string()
}

#[test]
fn memory_from_the_old_project_default_moves_with_the_documented_commands() {
    let sandbox = Sandbox::new();
    // A trained kernel makes embeddings the same in every process, so a
    // record written by one server is recalled by the next.
    init_home(&sandbox);
    let old_repo = sandbox.project("old-repo");
    let text = "the old release kept this in the repo";

    // What releases up to 0.7.0 did: the relative default, inside the repo.
    let mut server = Server::start(
        &old_repo,
        &sandbox.home,
        &[
            ("AXIOM_MEMORY_DIR", Path::new("checkpoints/memory")),
            ("AXIOM_TASK_DIR", Path::new("checkpoints/tasks")),
        ],
    );
    remember_and_post(&mut server, text, "an old task");
    server.shut_down();
    assert!(read(&old_repo.join("checkpoints/memory/personal.jsonl")).contains(text));

    // Started there with the variables unset, the server points at the old
    // records instead of silently ignoring them.
    let mut server = Server::start(&old_repo, &sandbox.home, &[]);
    server.request("tools/list", json!({}));
    let stderr = server.shut_down();
    assert!(
        stderr.contains("memory from an older Axiom") && stderr.contains("Moving existing memory"),
        "no notice about the old memory:\n{stderr}"
    );

    // Run the documented commands, exactly as written, from the old repo.
    let status = Command::new("sh")
        .arg("-c")
        .arg(documented_move_script())
        .current_dir(&old_repo)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", &sandbox.home)
        .status()
        .unwrap();
    assert!(status.success(), "the documented commands failed");

    // A server started in a different project now recalls the old memory.
    let elsewhere = sandbox.project("another-repo");
    let mut server = Server::start(&elsewhere, &sandbox.home, &[]);
    server.request(
        "initialize",
        json!({"protocolVersion": "2026-07-28", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}),
    );
    let recalled = server.call("axiom_recall", json!({"query": text}));
    let tasks = server.call("axiom_list_tasks", json!({"channel": "review"}));
    let stderr = server.shut_down();
    assert!(
        recalled.contains(text),
        "not recalled: {recalled}\n{stderr}"
    );
    assert!(tasks.contains("an old task"), "task not moved: {tasks}");
    assert!(!stderr.contains("memory from an older Axiom"), "{stderr}");
    assert_eq!(tree(&elsewhere), Vec::<String>::new());
}
