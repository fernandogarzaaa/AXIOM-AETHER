//! Experimental `cli` generation backend: drive a vendor CLI as a subprocess.
//!
//! When Axiom's own paths need a model (autonomous repair in `axiom solve`,
//! the `axiom task` agent loop, server generation behind the backend router)
//! and the operator has **no API key** but does have a logged-in vendor CLI,
//! this backend spawns that CLI in non-interactive print mode and uses its
//! answer. The first (and currently only) supported CLI is Claude Code:
//!
//! ```text
//! claude -p --output-format json --tools "" --no-session-persistence ... < prompt
//! ```
//!
//! Selected with `AXIOM_BACKEND=cli` (+ `AXIOM_CLI_COMMAND`, default
//! `claude`). Compiled only with `--features experimental`
//! (docs/EXPERIMENTAL.md); the proxy pass-through path
//! (docs/CLI-INTEGRATION.md) is the recommended integration and needs none of
//! this.
//!
//! Safety properties, each covered by a unit test:
//!   * **Version pin.** `<command> --version` is run once at construction and
//!     must parse as `MAJOR.MINOR.PATCH (Claude Code)` with
//!     `MAJOR == SUPPORTED_MAJOR`; anything else refuses to construct.
//!   * **Fail closed.** The CLI's JSON result must be exactly the shape we
//!     know (`type == "result"`, `subtype == "success"`, `is_error == false`,
//!     string `result`); anything else is an error, never a guessed completion.
//!   * **Bounded.** Every subprocess runs under a wall-clock timeout
//!     (`AXIOM_CLI_TIMEOUT_SECS`, default 120) and is killed past it.
//!   * **No shell.** The CLI is spawned directly with an argv; the prompt goes
//!     in on stdin and never appears in argv or in any log line.
//!   * **No prompt/output logging.** Nothing here prints the prompt or the
//!     model output; error messages describe shapes and exit codes only.
//!
//! `max_tokens` is accepted for trait compatibility but the CLI exposes no
//! output cap, so it is advisory here.

use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::backend_router::{BackendError, ChatBackend};

/// Default CLI program (`AXIOM_CLI_COMMAND`).
pub const DEFAULT_CLI_COMMAND: &str = "claude";
/// Default per-call wall-clock timeout (`AXIOM_CLI_TIMEOUT_SECS`).
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// The Claude Code major version whose `-p --output-format json` contract
/// this module was written and tested against. A different major refuses to
/// run rather than guessing at a changed output shape.
pub const SUPPORTED_MAJOR: u64 = 2;
/// Product marker that must follow the version number (`2.1.318 (Claude Code)`).
const VERSION_PRODUCT_MARKER: &str = "Claude Code";
/// Version probes are cheap; give them a short, separate bound.
const VERSION_TIMEOUT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(20);
/// Cap on bytes read from the child's stdout/stderr (a runaway child must
/// not exhaust memory).
const MAX_CAPTURE_BYTES: u64 = 16 * 1024 * 1024;
/// Bytes of the child's stderr quoted in an exit-status error (enough for
/// "not logged in" style diagnostics, never the whole stream).
const STDERR_EXCERPT_BYTES: usize = 240;

/// Parsed `MAJOR.MINOR.PATCH` from `<command> --version`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CliVersion {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

impl CliVersion {
    /// Parse the output of `claude --version`, e.g. `2.1.318 (Claude Code)`.
    /// The first token must be three dot-separated unsigned integers and the
    /// remainder must name the product; everything else is unparsable.
    pub fn parse(raw: &str) -> Result<Self, CliBackendError> {
        let trimmed = raw.trim();
        let unparsable = || CliBackendError::VersionUnparsable(excerpt(trimmed, 80));
        let (first, rest) = trimmed
            .split_once(char::is_whitespace)
            .ok_or_else(unparsable)?;
        let mut parts = first.split('.');
        let mut next = || -> Result<u64, CliBackendError> {
            let p = parts.next().ok_or_else(unparsable)?;
            if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) {
                return Err(unparsable());
            }
            p.parse::<u64>().map_err(|_| unparsable())
        };
        let (major, minor, patch) = (next()?, next()?, next()?);
        if parts.next().is_some() || !rest.contains(VERSION_PRODUCT_MARKER) {
            return Err(unparsable());
        }
        Ok(Self {
            major,
            minor,
            patch,
        })
    }

    /// Whether this version is inside the pin ([`SUPPORTED_MAJOR`]).
    pub fn is_supported(&self) -> bool {
        self.major == SUPPORTED_MAJOR
    }
}

impl std::fmt::Display for CliVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Why the CLI backend could not be built or could not answer. None of the
/// variants ever carries the prompt or the model output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliBackendError {
    /// Bad `AXIOM_CLI_*` configuration.
    Config(String),
    /// The CLI program could not be spawned (not on PATH, not executable…).
    Spawn(String),
    /// `--version` printed something we do not recognise.
    VersionUnparsable(String),
    /// `--version` parsed but is outside the pin.
    VersionUnsupported {
        found: CliVersion,
        pinned_major: u64,
    },
    /// The subprocess exceeded its wall-clock bound and was killed.
    Timeout(Duration),
    /// The subprocess exited non-zero; carries an excerpt of its stderr.
    ExitStatus {
        code: Option<i32>,
        stderr_excerpt: String,
    },
    /// The subprocess exited zero but its output is not the shape we accept.
    MalformedOutput(String),
    /// Pipe / I/O failure while talking to the subprocess.
    Io(String),
}

impl std::fmt::Display for CliBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliBackendError::Config(m) => write!(f, "cli backend configuration error: {m}"),
            CliBackendError::Spawn(m) => write!(f, "cli backend could not spawn the CLI: {m}"),
            CliBackendError::VersionUnparsable(raw) => write!(
                f,
                "cli backend refused to run: unrecognised `--version` output {raw:?} \
                 (expected `MAJOR.MINOR.PATCH ({VERSION_PRODUCT_MARKER})`)"
            ),
            CliBackendError::VersionUnsupported {
                found,
                pinned_major,
            } => write!(
                f,
                "cli backend refused to run: CLI version {found} is outside the pin \
                 (major {pinned_major}.x)"
            ),
            CliBackendError::Timeout(d) => {
                write!(f, "cli backend timed out after {}s", d.as_secs_f64())
            }
            CliBackendError::ExitStatus {
                code,
                stderr_excerpt,
            } => match code {
                Some(c) => write!(f, "cli exited with status {c}: {stderr_excerpt}"),
                None => write!(f, "cli was killed by a signal: {stderr_excerpt}"),
            },
            CliBackendError::MalformedOutput(shape) => write!(
                f,
                "cli backend refused the CLI output (fail closed): {shape}"
            ),
            CliBackendError::Io(m) => write!(f, "cli backend I/O error: {m}"),
        }
    }
}

impl std::error::Error for CliBackendError {}

/// Construction parameters (see [`CliBackend::from_env`] for the env mapping).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliBackendConfig {
    /// Program to spawn: a bare name resolved on `PATH` or an absolute path.
    pub command: String,
    /// Wall-clock bound per completion.
    pub timeout: Duration,
    /// Optional `--model` passed through to the CLI.
    pub model: Option<String>,
}

impl Default for CliBackendConfig {
    fn default() -> Self {
        Self {
            command: DEFAULT_CLI_COMMAND.to_string(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            model: None,
        }
    }
}

/// A vendor CLI driven in print mode as a [`ChatBackend`].
#[derive(Debug, Clone)]
pub struct CliBackend {
    config: CliBackendConfig,
    version: CliVersion,
}

/// Captured result of one bounded subprocess run.
struct Captured {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl CliBackend {
    /// Probe `<command> --version`, enforce the pin, and return a ready
    /// backend. Refuses (returns `Err`) on an unrecognised or unsupported
    /// version string, or when the CLI cannot be spawned at all.
    pub fn new(config: CliBackendConfig) -> Result<Self, CliBackendError> {
        if config.command.trim().is_empty() {
            return Err(CliBackendError::Config(
                "AXIOM_CLI_COMMAND must not be empty".into(),
            ));
        }
        if config.timeout.is_zero() {
            return Err(CliBackendError::Config(
                "AXIOM_CLI_TIMEOUT_SECS must be a positive integer".into(),
            ));
        }
        let mut cmd = Command::new(&config.command);
        cmd.arg("--version");
        let out = run_bounded(cmd, None, VERSION_TIMEOUT)?;
        if !out.status.success() {
            return Err(CliBackendError::ExitStatus {
                code: out.status.code(),
                stderr_excerpt: excerpt(
                    &String::from_utf8_lossy(&out.stderr),
                    STDERR_EXCERPT_BYTES,
                ),
            });
        }
        let raw = String::from_utf8_lossy(&out.stdout);
        let version = CliVersion::parse(&raw)?;
        if !version.is_supported() {
            return Err(CliBackendError::VersionUnsupported {
                found: version,
                pinned_major: SUPPORTED_MAJOR,
            });
        }
        Ok(Self { config, version })
    }

    /// Env-driven construction. `None` unless `AXIOM_BACKEND=cli`; otherwise
    /// `Some(Err(..))` when the configuration or the version probe fails
    /// (callers must treat that as "refuse to run", not "fall back").
    ///
    /// * `AXIOM_CLI_COMMAND` — program to spawn (default `claude`).
    /// * `AXIOM_CLI_TIMEOUT_SECS` — per-call bound (default 120).
    /// * `AXIOM_CLI_MODEL` — optional `--model` value.
    pub fn from_env() -> Option<Result<Self, CliBackendError>> {
        Self::from_env_values(
            std::env::var("AXIOM_BACKEND").ok().as_deref(),
            std::env::var("AXIOM_CLI_COMMAND").ok().as_deref(),
            std::env::var("AXIOM_CLI_TIMEOUT_SECS").ok().as_deref(),
            std::env::var("AXIOM_CLI_MODEL").ok().as_deref(),
        )
    }

    /// Pure core of [`Self::from_env`] (env values passed in, so tests need
    /// not mutate process-global state).
    pub fn from_env_values(
        backend: Option<&str>,
        command: Option<&str>,
        timeout_secs: Option<&str>,
        model: Option<&str>,
    ) -> Option<Result<Self, CliBackendError>> {
        if !backend
            .map(|b| b.trim().eq_ignore_ascii_case("cli"))
            .unwrap_or(false)
        {
            return None;
        }
        let config = match Self::config_from_values(command, timeout_secs, model) {
            Ok(c) => c,
            Err(e) => return Some(Err(e)),
        };
        Some(Self::new(config))
    }

    /// Build a [`CliBackendConfig`] from the raw `AXIOM_CLI_*` values.
    pub fn config_from_values(
        command: Option<&str>,
        timeout_secs: Option<&str>,
        model: Option<&str>,
    ) -> Result<CliBackendConfig, CliBackendError> {
        let command = command
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .unwrap_or(DEFAULT_CLI_COMMAND)
            .to_string();
        let timeout = match timeout_secs.map(str::trim).filter(|t| !t.is_empty()) {
            None => Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            Some(t) => match t.parse::<u64>() {
                Ok(secs) if secs > 0 => Duration::from_secs(secs),
                _ => {
                    return Err(CliBackendError::Config(format!(
                        "AXIOM_CLI_TIMEOUT_SECS must be a positive integer, got {t:?}"
                    )))
                }
            },
        };
        let model = model
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .map(str::to_string);
        Ok(CliBackendConfig {
            command,
            timeout,
            model,
        })
    }

    /// The pinned-and-verified CLI version.
    pub fn version(&self) -> CliVersion {
        self.version
    }

    /// The program this backend spawns.
    pub fn command(&self) -> &str {
        &self.config.command
    }

    /// Argument vector for one print-mode completion. The prompt is **not**
    /// part of it (it travels on stdin); tools, session persistence and MCP
    /// servers are all disabled so the CLI answers as a plain completion and
    /// never acts on the operator's filesystem.
    pub fn print_argv(&self) -> Vec<String> {
        let mut argv: Vec<String> = [
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--mcp-config",
            "{\"mcpServers\":{}}",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        if let Some(model) = &self.config.model {
            argv.push("--model".into());
            argv.push(model.clone());
        }
        argv
    }

    /// One prompt → one completion, or a precise error. Never logs either.
    pub fn complete_prompt(&self, prompt: &str) -> Result<String, CliBackendError> {
        let mut cmd = Command::new(&self.config.command);
        cmd.args(self.print_argv());
        let out = run_bounded(cmd, Some(prompt.as_bytes()), self.config.timeout)?;
        if !out.status.success() {
            return Err(CliBackendError::ExitStatus {
                code: out.status.code(),
                stderr_excerpt: excerpt(
                    &String::from_utf8_lossy(&out.stderr),
                    STDERR_EXCERPT_BYTES,
                ),
            });
        }
        parse_print_json(&out.stdout)
    }
}

impl ChatBackend for CliBackend {
    fn complete(&self, prompt: &str, _max_tokens: usize) -> Result<String, BackendError> {
        self.complete_prompt(prompt)
            .map_err(|e| BackendError::Upstream(e.to_string()))
    }
}

/// Accept exactly the Claude Code `--output-format json` success shape and
/// nothing else. The error text describes the *shape* found (keys, types),
/// never the content.
pub fn parse_print_json(stdout: &[u8]) -> Result<String, CliBackendError> {
    let value: Value = serde_json::from_slice(stdout).map_err(|_| {
        CliBackendError::MalformedOutput(format!(
            "stdout is not a JSON document ({} bytes)",
            stdout.len()
        ))
    })?;
    let obj = value.as_object().ok_or_else(|| {
        CliBackendError::MalformedOutput("JSON root is not an object".to_string())
    })?;
    let field_str = |k: &str| obj.get(k).and_then(Value::as_str);
    match field_str("type") {
        Some("result") => {}
        other => {
            return Err(CliBackendError::MalformedOutput(format!(
                "`type` is {}, expected \"result\"",
                describe(other.map(Value::from))
            )))
        }
    }
    match field_str("subtype") {
        Some("success") => {}
        other => {
            return Err(CliBackendError::MalformedOutput(format!(
                "`subtype` is {}, expected \"success\"",
                describe(other.map(Value::from))
            )))
        }
    }
    match obj.get("is_error").and_then(Value::as_bool) {
        Some(false) => {}
        other => {
            return Err(CliBackendError::MalformedOutput(format!(
                "`is_error` is {}, expected false",
                describe(other.map(Value::from))
            )))
        }
    }
    match obj.get("result") {
        Some(Value::String(s)) => Ok(s.clone()),
        other => Err(CliBackendError::MalformedOutput(format!(
            "`result` is {}, expected a string",
            describe(other.cloned())
        ))),
    }
}

/// Shape-only description of a JSON field for error messages: the literal
/// value for the small enum-like fields (`type`, `subtype`, `is_error`), the
/// JSON type name for anything else, so free text never leaks into an error.
fn describe(v: Option<Value>) -> String {
    match v {
        None => "missing".to_string(),
        Some(Value::Null) => "null".to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Number(_)) => "a number".to_string(),
        Some(Value::String(s)) if s.len() <= 40 && s.bytes().all(|b| b.is_ascii_graphic()) => {
            format!("{s:?}")
        }
        Some(Value::String(_)) => "a string".to_string(),
        Some(Value::Array(_)) => "an array".to_string(),
        Some(Value::Object(_)) => "an object".to_string(),
    }
}

/// First `max` bytes of `s` on a char boundary, for diagnostics.
fn excerpt(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// Spawn `cmd` with piped stdio, feed `stdin_data` (if any) on a helper
/// thread, drain stdout/stderr on helper threads, and wait at most `timeout`.
/// Past the deadline the child is killed and [`CliBackendError::Timeout`] is
/// returned. No shell is involved: `cmd` is executed directly.
fn run_bounded(
    mut cmd: Command,
    stdin_data: Option<&[u8]>,
    timeout: Duration,
) -> Result<Captured, CliBackendError> {
    cmd.stdin(if stdin_data.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    })
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    let mut child: Child = cmd.spawn().map_err(|e| {
        CliBackendError::Spawn(format!("{}: {e}", cmd.get_program().to_string_lossy()))
    })?;

    // Writer: hand the prompt over on stdin and close it so the CLI knows the
    // input is complete. Done off-thread so a child that never reads cannot
    // block us past the deadline; a broken pipe (child exited early) is not
    // an error here — the exit status / output parse reports it.
    let writer = match (child.stdin.take(), stdin_data) {
        (Some(mut stdin), Some(data)) => {
            let data = data.to_vec();
            Some(thread::spawn(move || {
                let _ = stdin.write_all(&data);
                let _ = stdin.flush();
                drop(stdin);
            }))
        }
        _ => None,
    };
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| CliBackendError::Io("child stdout was not captured".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| CliBackendError::Io("child stderr was not captured".into()))?;
    let out_reader = thread::spawn(move || drain(stdout));
    let err_reader = thread::spawn(move || drain(stderr));

    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if started.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    // Do NOT join the reader threads here: a grandchild the
                    // CLI spawned may still hold the pipe open, and waiting
                    // for it would turn a bounded timeout into an unbounded
                    // one. The detached readers exit on their own once the
                    // pipe finally closes; their buffers are discarded.
                    drop(out_reader);
                    drop(err_reader);
                    drop(writer);
                    return Err(CliBackendError::Timeout(timeout));
                }
                thread::sleep(POLL_INTERVAL);
            }
            Err(e) => {
                let _ = child.kill();
                return Err(CliBackendError::Io(format!("wait failed: {e}")));
            }
        }
    };
    let stdout = out_reader
        .join()
        .map_err(|_| CliBackendError::Io("stdout reader panicked".into()))?;
    let stderr = err_reader
        .join()
        .map_err(|_| CliBackendError::Io("stderr reader panicked".into()))?;
    if let Some(w) = writer {
        let _ = w.join();
    }
    Ok(Captured {
        status,
        stdout,
        stderr,
    })
}

fn drain<R: Read>(reader: R) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = reader.take(MAX_CAPTURE_BYTES).read_to_end(&mut buf);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parse_accepts_the_real_claude_code_format() {
        let v = CliVersion::parse("2.1.318 (Claude Code)\n").unwrap();
        assert_eq!(
            v,
            CliVersion {
                major: 2,
                minor: 1,
                patch: 318
            }
        );
        assert!(v.is_supported());
        assert_eq!(v.to_string(), "2.1.318");
    }

    #[test]
    fn version_parse_rejects_unparsable_strings() {
        for raw in [
            "",
            "banana",
            "2.1 (Claude Code)",
            "2.1.318",
            "2.1.318.4 (Claude Code)",
            "v2.1.318 (Claude Code)",
            "2.1.318 (Some Other CLI)",
            "2.x.1 (Claude Code)",
        ] {
            match CliVersion::parse(raw) {
                Err(CliBackendError::VersionUnparsable(_)) => {}
                other => panic!("{raw:?} should be unparsable, got {other:?}"),
            }
        }
    }

    #[test]
    fn version_outside_the_pin_is_unsupported() {
        let v = CliVersion::parse("1.0.80 (Claude Code)").unwrap();
        assert!(!v.is_supported());
        let v = CliVersion::parse("3.0.0 (Claude Code)").unwrap();
        assert!(!v.is_supported());
    }

    #[test]
    fn parse_print_json_accepts_only_the_success_shape() {
        let ok = br#"{"type":"result","subtype":"success","is_error":false,"result":"pong","num_turns":1}"#;
        assert_eq!(parse_print_json(ok).unwrap(), "pong");

        let bad: &[(&[u8], &str)] = &[
            (b"pong", "not json"),
            (b"[]", "array root"),
            (br#"{"type":"assistant","subtype":"success","is_error":false,"result":"x"}"#, "wrong type"),
            (br#"{"type":"result","subtype":"error_during_execution","is_error":true,"result":"x"}"#, "error subtype"),
            (br#"{"type":"result","subtype":"success","is_error":true,"result":"x"}"#, "is_error true"),
            (br#"{"type":"result","subtype":"success","result":"x"}"#, "is_error missing"),
            (br#"{"type":"result","subtype":"success","is_error":false}"#, "result missing"),
            (br#"{"type":"result","subtype":"success","is_error":false,"result":42}"#, "result not a string"),
        ];
        for (input, why) in bad {
            match parse_print_json(input) {
                Err(CliBackendError::MalformedOutput(_)) => {}
                other => panic!("{why}: expected MalformedOutput, got {other:?}"),
            }
        }
    }

    #[test]
    fn malformed_output_errors_never_quote_free_text() {
        // The `result` field is the model's free text; a shape error about it
        // must say "a string"/"a number", never echo the content.
        let secret =
            br#"{"type":"result","subtype":"success","is_error":true,"result":"SECRET-OUTPUT"}"#;
        let err = parse_print_json(secret).unwrap_err().to_string();
        assert!(!err.contains("SECRET-OUTPUT"), "{err}");
        let long_type = br#"{"type":"THIS IS A VERY LONG FREE TEXT TYPE FIELD THAT MUST NOT BE QUOTED VERBATIM"}"#;
        let err = parse_print_json(long_type).unwrap_err().to_string();
        assert!(err.contains("a string"), "{err}");
        assert!(!err.contains("VERY LONG"), "{err}");
    }

    #[test]
    fn config_from_values_defaults_and_validation() {
        let c = CliBackend::config_from_values(None, None, None).unwrap();
        assert_eq!(c, CliBackendConfig::default());
        let c = CliBackend::config_from_values(Some(" /opt/bin/claude "), Some("7"), Some("haiku"))
            .unwrap();
        assert_eq!(c.command, "/opt/bin/claude");
        assert_eq!(c.timeout, Duration::from_secs(7));
        assert_eq!(c.model.as_deref(), Some("haiku"));
        for bad in ["0", "-1", "abc", "1.5"] {
            assert!(matches!(
                CliBackend::config_from_values(None, Some(bad), None),
                Err(CliBackendError::Config(_))
            ));
        }
    }

    #[test]
    fn from_env_values_is_none_unless_backend_is_cli() {
        assert!(CliBackend::from_env_values(None, None, None, None).is_none());
        assert!(CliBackend::from_env_values(Some("claude"), None, None, None).is_none());
        assert!(CliBackend::from_env_values(Some("router"), None, None, None).is_none());
    }

    #[test]
    fn from_env_values_surfaces_config_errors_without_spawning() {
        match CliBackend::from_env_values(
            Some("cli"),
            Some("definitely-not-a-cli"),
            Some("x"),
            None,
        ) {
            Some(Err(CliBackendError::Config(_))) => {}
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn missing_program_fails_to_spawn() {
        let cfg = CliBackendConfig {
            command: "/nonexistent/axiom-fake-cli-binary".into(),
            ..Default::default()
        };
        match CliBackend::new(cfg) {
            Err(CliBackendError::Spawn(_)) => {}
            other => panic!("expected Spawn error, got {other:?}"),
        }
    }

    /// Subprocess tests use small `sh` scripts as a fake CLI; unix only.
    #[cfg(unix)]
    mod with_fake_cli {
        use super::*;
        use std::path::{Path, PathBuf};

        const OK_JSON: &str = r#"{"type":"result","subtype":"success","is_error":false,"result":"canned answer","num_turns":1}"#;

        /// Write an executable `sh` script that answers `--version` with
        /// `version_line` and anything else with `body` (a shell snippet that
        /// runs after stdin has been drained to `$STDIN_FILE`).
        fn fake_cli(dir: &Path, name: &str, version_line: &str, body: &str) -> PathBuf {
            let path = dir.join(name);
            let argv_file = dir.join(format!("{name}.argv"));
            let stdin_file = dir.join(format!("{name}.stdin"));
            let script = format!(
                "#!/bin/sh\n\
                 if [ \"$1\" = \"--version\" ]; then printf '%s\\n' '{version_line}'; exit 0; fi\n\
                 printf '%s\\n' \"$@\" > '{argv}'\n\
                 STDIN_FILE='{stdin}'\n\
                 cat > \"$STDIN_FILE\"\n\
                 {body}\n",
                argv = argv_file.display(),
                stdin = stdin_file.display(),
            );
            write_executable(&path, &script);
            path
        }

        /// Write `script` to `path` and make it executable, without this
        /// process ever having the file open for writing.
        ///
        /// A file that any process has open for writing cannot be executed
        /// ("Text file busy"). Writing it here with `std::fs::write` holds it
        /// open only for a moment, but a process another thread starts in that
        /// moment begins as a copy of this one and keeps the file open until
        /// it has turned into its own program. Run before then, the script is
        /// refused. So a child writes the file: nothing this process starts
        /// can inherit a file this process never opened, and the child has
        /// exited, with everything it started, before this returns.
        fn write_executable(path: &Path, script: &str) {
            use std::io::Write;
            let mut writer = Command::new("sh")
                .args(["-c", r#"cat > "$1" && chmod 755 "$1""#, "sh"])
                .arg(path)
                .stdin(Stdio::piped())
                .spawn()
                .expect("start sh to write the script");
            writer
                .stdin
                .take()
                .expect("piped stdin")
                .write_all(script.as_bytes())
                .expect("send the script to sh");
            let status = writer.wait().expect("wait for sh");
            assert!(status.success(), "writing {} failed", path.display());
        }

        /// A script `fake_cli` returns can be run at once, whatever the rest
        /// of the test binary is doing. Tests run on many threads and most of
        /// them start processes; a script that this process had open for
        /// writing while another thread started one could not be run until
        /// that other process got going ("Text file busy"), which failed
        /// whichever test owned the script.
        #[test]
        fn a_fake_cli_can_be_run_while_other_threads_start_processes() {
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::Arc;

            /// Holds the process-starting threads. Dropping it (including while
            /// unwinding from a failed assertion) sets the stop flag and joins them, so a
            /// failure cannot leave them starting processes for the rest of the test run.
            struct StopOnDrop {
                stop: Arc<AtomicBool>,
                threads: Vec<std::thread::JoinHandle<()>>,
            }
            impl StopOnDrop {
                fn stop_and_join(mut self) {
                    self.stop.store(true, Ordering::Relaxed);
                    for thread in self.threads.drain(..) {
                        thread.join().unwrap();
                    }
                }
            }
            impl Drop for StopOnDrop {
                fn drop(&mut self) {
                    self.stop.store(true, Ordering::Relaxed);
                    for thread in self.threads.drain(..) {
                        let _ = thread.join();
                    }
                }
            }

            const SCRIPTS: usize = 150;
            let dir = tempfile::tempdir().unwrap();
            let stop = Arc::new(AtomicBool::new(false));
            let others = StopOnDrop {
                stop: stop.clone(),
                threads: (0..8)
                    .map(|_| {
                        let stop = stop.clone();
                        std::thread::spawn(move || {
                            while !stop.load(Ordering::Relaxed) {
                                let _ = Command::new("true").status();
                            }
                        })
                    })
                    .collect(),
            };

            let mut refused = Vec::new();
            for n in 0..SCRIPTS {
                let cli = fake_cli(dir.path(), &format!("cli{n}"), "9.9.9 (Fake)", "exit 0");
                match Command::new(&cli).arg("--version").output() {
                    Ok(out) => assert_eq!(out.stdout, b"9.9.9 (Fake)\n"),
                    Err(e) => refused.push(e.to_string()),
                }
            }
            others.stop_and_join();
            assert!(
                refused.is_empty(),
                "{} of {SCRIPTS} scripts could not be run; the first: {}",
                refused.len(),
                refused[0]
            );
        }

        fn backend(path: &Path, timeout: Duration) -> Result<CliBackend, CliBackendError> {
            CliBackend::new(CliBackendConfig {
                command: path.display().to_string(),
                timeout,
                model: None,
            })
        }

        #[test]
        fn version_pin_accepted() {
            let dir = tempfile::tempdir().unwrap();
            let cli = fake_cli(
                dir.path(),
                "fakeclaude",
                "2.1.318 (Claude Code)",
                &format!("printf '%s' '{OK_JSON}'"),
            );
            let b = backend(&cli, Duration::from_secs(10)).expect("pinned major accepted");
            assert_eq!(b.version().to_string(), "2.1.318");
        }

        #[test]
        fn version_pin_rejected_for_other_major_and_garbage() {
            let dir = tempfile::tempdir().unwrap();
            let old = fake_cli(dir.path(), "oldclaude", "1.0.80 (Claude Code)", "exit 0");
            match backend(&old, Duration::from_secs(10)) {
                Err(CliBackendError::VersionUnsupported {
                    found,
                    pinned_major,
                }) => {
                    assert_eq!(found.major, 1);
                    assert_eq!(pinned_major, SUPPORTED_MAJOR);
                }
                other => panic!("expected VersionUnsupported, got {other:?}"),
            }
            let garbage = fake_cli(dir.path(), "garbage", "banana", "exit 0");
            match backend(&garbage, Duration::from_secs(10)) {
                Err(CliBackendError::VersionUnparsable(_)) => {}
                other => panic!("expected VersionUnparsable, got {other:?}"),
            }
            let other_product = fake_cli(dir.path(), "notclaude", "2.1.0 (Other CLI)", "exit 0");
            assert!(matches!(
                backend(&other_product, Duration::from_secs(10)),
                Err(CliBackendError::VersionUnparsable(_))
            ));
        }

        #[test]
        fn well_formed_output_is_parsed_and_prompt_travels_on_stdin_only() {
            let dir = tempfile::tempdir().unwrap();
            let cli = fake_cli(
                dir.path(),
                "fakeclaude",
                "2.1.318 (Claude Code)",
                &format!("printf '%s' '{OK_JSON}'"),
            );
            let b = backend(&cli, Duration::from_secs(10)).unwrap();
            let prompt = "fix this file; it is a $(dangerous) `prompt` with \"quotes\" and newlines\nline two";
            let answer = b.complete(prompt, 512).unwrap();
            assert_eq!(answer, "canned answer");

            let argv = std::fs::read_to_string(dir.path().join("fakeclaude.argv")).unwrap();
            assert!(
                !argv.contains("fix this file"),
                "prompt must never be in argv: {argv}"
            );
            assert!(argv.contains("-p"), "{argv}");
            assert!(argv.contains("--output-format\njson"), "{argv}");
            assert!(
                argv.contains("--tools\n\n"),
                "tools must be disabled: {argv}"
            );
            assert!(argv.contains("--no-session-persistence"), "{argv}");
            let stdin = std::fs::read_to_string(dir.path().join("fakeclaude.stdin")).unwrap();
            assert_eq!(stdin, prompt, "prompt must arrive verbatim on stdin");
        }

        #[test]
        fn model_flag_is_threaded_into_argv() {
            let dir = tempfile::tempdir().unwrap();
            let cli = fake_cli(
                dir.path(),
                "fakeclaude",
                "2.1.318 (Claude Code)",
                &format!("printf '%s' '{OK_JSON}'"),
            );
            let b = CliBackend::new(CliBackendConfig {
                command: cli.display().to_string(),
                timeout: Duration::from_secs(10),
                model: Some("haiku".into()),
            })
            .unwrap();
            b.complete("q", 8).unwrap();
            let argv = std::fs::read_to_string(dir.path().join("fakeclaude.argv")).unwrap();
            assert!(argv.contains("--model\nhaiku"), "{argv}");
        }

        #[test]
        fn malformed_output_fails_closed() {
            let dir = tempfile::tempdir().unwrap();
            let cases: &[(&str, &str)] = &[
                ("plain", "printf 'pong'"),
                (
                    "is_error",
                    r#"printf '%s' '{"type":"result","subtype":"error_during_execution","is_error":true,"result":"boom"}'"#,
                ),
                (
                    "noresult",
                    r#"printf '%s' '{"type":"result","subtype":"success","is_error":false}'"#,
                ),
                ("empty", "true"),
            ];
            for (name, body) in cases {
                let cli = fake_cli(dir.path(), name, "2.1.318 (Claude Code)", body);
                let b = backend(&cli, Duration::from_secs(10)).unwrap();
                match b.complete_prompt("q") {
                    Err(CliBackendError::MalformedOutput(_)) => {}
                    other => panic!("{name}: expected MalformedOutput, got {other:?}"),
                }
            }
            // Via the ChatBackend trait the same failure surfaces as Upstream,
            // never as Ok(text).
            let cli = fake_cli(
                dir.path(),
                "plain2",
                "2.1.318 (Claude Code)",
                "printf 'pong'",
            );
            let b = backend(&cli, Duration::from_secs(10)).unwrap();
            assert!(matches!(b.complete("q", 8), Err(BackendError::Upstream(_))));
        }

        #[test]
        fn nonzero_exit_is_an_error_with_a_stderr_excerpt() {
            let dir = tempfile::tempdir().unwrap();
            let cli = fake_cli(
                dir.path(),
                "fakeclaude",
                "2.1.318 (Claude Code)",
                &format!("printf '%s' '{OK_JSON}'; echo 'Not logged in' >&2; exit 3"),
            );
            let b = backend(&cli, Duration::from_secs(10)).unwrap();
            match b.complete_prompt("q") {
                Err(CliBackendError::ExitStatus {
                    code: Some(3),
                    stderr_excerpt,
                }) => assert!(stderr_excerpt.contains("Not logged in")),
                other => panic!("expected ExitStatus(3), got {other:?}"),
            }
        }

        #[test]
        fn timeout_kills_the_child_and_fails() {
            let dir = tempfile::tempdir().unwrap();
            let cli = fake_cli(
                dir.path(),
                "slowclaude",
                "2.1.318 (Claude Code)",
                &format!("sleep 20; printf '%s' '{OK_JSON}'"),
            );
            let b = backend(&cli, Duration::from_millis(300)).unwrap();
            let started = Instant::now();
            match b.complete_prompt("q") {
                Err(CliBackendError::Timeout(d)) => assert_eq!(d, Duration::from_millis(300)),
                other => panic!("expected Timeout, got {other:?}"),
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "timeout must not wait for the child's own sleep"
            );
        }

        /// Resolving a bare command name on `PATH` is exercised in a child
        /// process whose `PATH` is set *before* it starts, because mutating
        /// `PATH` inside this multi-threaded test binary while other tests
        /// spawn processes is a data race. The parent writes the fake CLI,
        /// re-runs this very test binary filtered to the child test below with
        /// the adjusted `PATH`, and checks it passed.
        const CHILD_FLAG: &str = "AXIOM_CLI_BACKEND_TEST_CHILD";
        const CHILD_FAKE_NAME: &str = "axiom-fake-cli-on-path";

        #[test]
        fn bare_command_name_resolves_on_path() {
            if std::env::var_os(CHILD_FLAG).is_some() {
                return; // we are the child: the sibling test below does the work
            }
            let dir = tempfile::tempdir().unwrap();
            fake_cli(
                dir.path(),
                CHILD_FAKE_NAME,
                "2.1.318 (Claude Code)",
                &format!("printf '%s' '{OK_JSON}'"),
            );
            let mut new_path = dir.path().as_os_str().to_owned();
            new_path.push(":");
            new_path.push(std::env::var_os("PATH").unwrap_or_default());
            let exe = std::env::current_exe().unwrap();
            let out = Command::new(exe)
                .args([
                    "--exact",
                    "cli_backend::tests::with_fake_cli::bare_command_name_resolves_on_path_child",
                    "--nocapture",
                ])
                .env("PATH", new_path)
                .env(CHILD_FLAG, "1")
                .output()
                .expect("re-run the test binary");
            assert!(
                out.status.success(),
                "child failed:\n{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }

        #[test]
        fn bare_command_name_resolves_on_path_child() {
            if std::env::var_os(CHILD_FLAG).is_none() {
                return; // only meaningful inside the parent's re-run
            }
            let b =
                CliBackend::from_env_values(Some("cli"), Some(CHILD_FAKE_NAME), Some("10"), None)
                    .expect("backend=cli")
                    .expect("fake cli on PATH accepted");
            assert_eq!(b.command(), CHILD_FAKE_NAME);
            assert_eq!(b.complete("q", 8).unwrap(), "canned answer");
        }
    }
}
