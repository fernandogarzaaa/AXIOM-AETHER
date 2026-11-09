//! Time-bounded child processes for verifier runs.
//!
//! A verifier that never finishes (a fault that loops forever) must not hang
//! the repair loop, and killing only the direct child is not enough. A test
//! runner that starts subprocesses leaves them holding its stdout/stderr pipes,
//! so reading the output would still wait for them. Here the child runs in its
//! own process group (Unix), the whole group is killed on timeout, and pipe
//! draining has a bounded grace period after the deadline.

use std::io::Read;
use std::process::{Command, Output, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How long to keep reading output after the run ended or was killed.
pub(crate) const DRAIN_GRACE: Duration = Duration::from_secs(2);

/// Maximum bytes retained from one child stream (stdout or stderr).
/// A verifier that prints without limit must not grow process memory
/// without bound, so output past this cap is counted and discarded.
pub const MAX_OUTPUT_BYTES: usize = 1 << 20;

/// Bytes kept from the start of an over-cap stream. The other half of
/// [`MAX_OUTPUT_BYTES`] is kept from the end, so diagnostics show both
/// how the run began and how it ended.
pub const HEAD_BYTES: usize = MAX_OUTPUT_BYTES / 2;

/// A bounded sink for child-process output. Bytes are kept until
/// [`MAX_OUTPUT_BYTES`]; past the cap only the first [`HEAD_BYTES`] and
/// the last [`HEAD_BYTES`] are retained, joined by a truncation marker
/// that reports the total bytes seen. Memory use stays at roughly
/// [`MAX_OUTPUT_BYTES`] no matter how much the child prints.
pub struct CappedOutput {
    head: Vec<u8>,
    tail: Vec<u8>,
    total: usize,
    truncated: bool,
}

impl CappedOutput {
    pub fn new() -> Self {
        Self {
            head: Vec::new(),
            tail: Vec::new(),
            total: 0,
            truncated: false,
        }
    }

    /// Total bytes fed in, including bytes discarded past the cap.
    pub fn total_bytes(&self) -> usize {
        self.total
    }

    /// Whether any bytes were discarded past the cap.
    pub fn was_truncated(&self) -> bool {
        self.truncated
    }

    pub fn push(&mut self, chunk: &[u8]) {
        if chunk.is_empty() {
            return;
        }
        self.total = self.total.saturating_add(chunk.len());
        if !self.truncated {
            if self.head.len() + self.tail.len() + chunk.len() <= MAX_OUTPUT_BYTES {
                // Still within the cap: keep everything, split head/tail at
                // HEAD_BYTES so the head is frozen once it fills.
                if self.head.len() < HEAD_BYTES {
                    let take = (HEAD_BYTES - self.head.len()).min(chunk.len());
                    self.head.extend_from_slice(&chunk[..take]);
                    self.tail.extend_from_slice(&chunk[take..]);
                } else {
                    self.tail.extend_from_slice(chunk);
                }
                return;
            }
            // Cross the cap: freeze the head, keep the last HEAD_BYTES of
            // everything seen so far as the rolling tail.
            self.truncated = true;
            let mut combined = std::mem::take(&mut self.head);
            combined.extend_from_slice(&self.tail);
            combined.extend_from_slice(chunk);
            let head_len = HEAD_BYTES.min(combined.len());
            self.head = combined[..head_len].to_vec();
            let tail_start = combined.len().saturating_sub(HEAD_BYTES);
            self.tail = combined[tail_start..].to_vec();
            return;
        }
        self.tail.extend_from_slice(chunk);
        if self.tail.len() > HEAD_BYTES {
            let drop = self.tail.len() - HEAD_BYTES;
            self.tail.drain(..drop);
        }
    }

    /// The retained bytes: everything seen when under the cap, otherwise
    /// head, truncation marker, tail.
    pub fn finish(mut self) -> Vec<u8> {
        if !self.truncated {
            let mut out = std::mem::take(&mut self.head);
            out.extend_from_slice(&self.tail);
            return out;
        }
        let mut out = std::mem::take(&mut self.head);
        out.extend_from_slice(
            format!(
                "\n[output truncated: showing first {HEAD_BYTES} and last {HEAD_BYTES} of {} bytes]\n",
                self.total
            )
            .as_bytes(),
        );
        out.extend_from_slice(&self.tail);
        out
    }
}

impl Default for CappedOutput {
    fn default() -> Self {
        Self::new()
    }
}

/// Put `cmd` in a new process group, so [`kill_group`] reaches its
/// descendants. No-op off Unix.
pub fn isolate_group(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(not(unix))]
    let _ = cmd;
}

/// SIGKILL the process group led by `pid` (see [`isolate_group`]). Off Unix
/// there is no group to kill and this does nothing; callers also kill the
/// direct child.
pub fn kill_group(pid: u32) {
    #[cfg(unix)]
    {
        if let Ok(pid) = i32::try_from(pid) {
            if pid > 0 {
                // SAFETY: kill(2) with a negative pid signals that process
                // group; it reads no memory and fails harmlessly with ESRCH
                // if the group is already gone.
                unsafe {
                    libc::kill(-pid, libc::SIGKILL);
                }
            }
        }
    }
    #[cfg(not(unix))]
    let _ = pid;
}

/// `Command::output`, killed after `timeout` (when set). With `timeout =
/// None` this is plain `output()`. A killed run keeps whatever it printed,
/// gains a `[timed out after …]` line on stderr, and reports a non-success
/// status. Whether or not the run timed out, output that is still being held
/// open by an orphaned descendant is read for at most a short grace period,
/// after which the group is killed and the run returns.
pub fn output_bounded(cmd: &mut Command, timeout: Option<Duration>) -> std::io::Result<Output> {
    let Some(limit) = timeout else {
        return cmd.output();
    };
    isolate_group(cmd);
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let pid = child.id();
    let (out_tx, out_rx) = mpsc::channel::<Vec<u8>>();
    let (err_tx, err_rx) = mpsc::channel::<Vec<u8>>();
    if let Some(mut p) = child.stdout.take() {
        std::thread::spawn(move || {
            let mut capped = CappedOutput::new();
            let mut buf = [0u8; 8192];
            loop {
                match p.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => capped.push(&buf[..n]),
                    Err(_) => break,
                }
            }
            let _ = out_tx.send(capped.finish());
        });
    }
    if let Some(mut p) = child.stderr.take() {
        std::thread::spawn(move || {
            let mut capped = CappedOutput::new();
            let mut buf = [0u8; 8192];
            loop {
                match p.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => capped.push(&buf[..n]),
                    Err(_) => break,
                }
            }
            let _ = err_tx.send(capped.finish());
        });
    }

    let deadline = Instant::now() + limit;
    let (status, timed_out) = loop {
        if let Some(st) = child.try_wait()? {
            break (st, false);
        }
        if Instant::now() >= deadline {
            kill_group(pid);
            let _ = child.kill();
            break (child.wait()?, true);
        }
        std::thread::sleep(Duration::from_millis(20));
    };

    // A descendant that inherited the pipes can keep them open after the
    // child exits; give it a moment, then kill the group so the readers
    // finish, and never wait past the grace period.
    let collect = |rx: &mpsc::Receiver<Vec<u8>>| -> Vec<u8> {
        match rx.recv_timeout(DRAIN_GRACE) {
            Ok(b) => b,
            Err(_) => {
                kill_group(pid);
                rx.recv_timeout(DRAIN_GRACE).unwrap_or_default()
            }
        }
    };
    let stdout = collect(&out_rx);
    let mut stderr = collect(&err_rx);
    if timed_out {
        stderr.extend_from_slice(
            format!("\n[timed out after {:.0}s]\n", limit.as_secs_f32()).as_bytes(),
        );
    }
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn kills_a_hung_command_and_keeps_its_output() {
        let t0 = Instant::now();
        let out = output_bounded(
            Command::new("sh").args(["-c", "echo started; while :; do :; done"]),
            Some(Duration::from_millis(500)),
        )
        .unwrap();
        assert!(t0.elapsed() < Duration::from_secs(10), "killed promptly");
        assert!(!out.status.success(), "a killed run is a failed run");
        assert!(String::from_utf8_lossy(&out.stdout).contains("started"));
        assert!(String::from_utf8_lossy(&out.stderr).contains("timed out"));
        let out = output_bounded(
            Command::new("sh").args(["-c", "echo ok; exit 3"]),
            Some(Duration::from_secs(30)),
        )
        .unwrap();
        assert_eq!(out.status.code(), Some(3));
        assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
    }

    #[test]
    fn output_bounded_caps_a_spamming_child() {
        // 3 MiB on stdout and 2 MiB on stderr: both must come back capped
        // at roughly MAX_OUTPUT_BYTES, the run must still finish, and the
        // retained window must show the start and the end of the stream.
        let t0 = Instant::now();
        let out = output_bounded(
            Command::new("sh").args([
                "-c",
                "(yes abcdefghij | head -c 2999999; printf 'END-STDOUT'); (yes ZYXWVUTSRQ | head -c 1999999 >&2; printf 'END-STDERR' >&2)",
            ]),
            Some(Duration::from_secs(30)),
        )
        .unwrap();
        assert!(
            t0.elapsed() < Duration::from_secs(20),
            "returned in {:?}",
            t0.elapsed()
        );
        assert!(
            out.status.success(),
            "a spammer that exits is a passing run"
        );
        assert!(
            out.stdout.len() <= MAX_OUTPUT_BYTES + 128,
            "stdout capped: got {} bytes",
            out.stdout.len()
        );
        assert!(
            out.stderr.len() <= MAX_OUTPUT_BYTES + 128,
            "stderr capped: got {} bytes",
            out.stderr.len()
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.starts_with("abcdefghij\n"),
            "head kept the stream start"
        );
        assert!(stdout.contains("output truncated"), "truncation is marked");
        assert!(
            stdout.contains("3000009 bytes"),
            "marker reports the total seen"
        );
        assert!(stdout.ends_with("END-STDOUT"), "tail kept the stream end");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("2000009 bytes"),
            "stderr marker reports its total"
        );
        assert!(
            stderr.ends_with("END-STDERR"),
            "stderr tail kept the stream end"
        );
    }

    #[test]
    fn capped_output_keeps_small_streams_whole() {
        let mut capped = CappedOutput::new();
        capped.push(b"hello ");
        capped.push(b"world");
        assert!(!capped.was_truncated());
        assert_eq!(capped.total_bytes(), 11);
        assert_eq!(capped.finish(), b"hello world");
    }

    #[test]
    fn capped_output_stays_whole_below_the_cap() {
        // 600 KiB in 8 KiB chunks is well under the 1 MiB cap, so nothing
        // may be discarded. This pins the cap arithmetic: a wrong
        // retained-size sum truncates early and loses bytes here.
        let mut capped = CappedOutput::new();
        let chunk = vec![b'y'; 8192];
        for _ in 0..75 {
            capped.push(&chunk);
        }
        assert!(!capped.was_truncated());
        assert_eq!(capped.total_bytes(), 75 * 8192);
        assert_eq!(capped.finish().len(), 75 * 8192);
    }

    #[test]
    fn capped_output_keeps_head_and_tail_past_the_cap() {
        // Distinct head and tail markers, with filler in the middle that
        // must be dropped.
        let mut capped = CappedOutput::new();
        capped.push(b"HEAD");
        capped.push(&vec![b'x'; MAX_OUTPUT_BYTES]);
        capped.push(b"TAIL");
        assert!(capped.was_truncated());
        assert_eq!(capped.total_bytes(), MAX_OUTPUT_BYTES + 8);
        let out = capped.finish();
        assert!(
            out.len() <= MAX_OUTPUT_BYTES + 128,
            "retained {}",
            out.len()
        );
        assert!(out.starts_with(b"HEAD"), "head kept");
        assert!(out.ends_with(b"TAIL"), "tail kept");
        assert!(
            String::from_utf8_lossy(&out).contains("output truncated"),
            "marker present"
        );
    }

    #[test]
    fn a_descendant_holding_the_pipes_cannot_hang_the_caller() {
        // The child waits on a background grandchild that inherits stdout;
        // killing only the child would leave the pipe open for 60 s.
        let t0 = Instant::now();
        let out = output_bounded(
            Command::new("sh").args(["-c", "sleep 60 & wait"]),
            Some(Duration::from_millis(300)),
        )
        .unwrap();
        assert!(
            t0.elapsed() < Duration::from_secs(8),
            "returned in {:?}",
            t0.elapsed()
        );
        assert!(!out.status.success());

        // A child that exits at once but leaves an orphan holding the pipe:
        // bounded by the drain grace, not by the orphan.
        let t0 = Instant::now();
        let out = output_bounded(
            Command::new("sh").args(["-c", "sleep 60 & echo bye"]),
            Some(Duration::from_secs(30)),
        )
        .unwrap();
        assert!(t0.elapsed() < Duration::from_secs(8), "{:?}", t0.elapsed());
        assert!(out.status.success());
    }
}
