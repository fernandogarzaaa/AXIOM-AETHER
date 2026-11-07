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
const DRAIN_GRACE: Duration = Duration::from_secs(2);

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
            let mut b = Vec::new();
            let _ = p.read_to_end(&mut b);
            let _ = out_tx.send(b);
        });
    }
    if let Some(mut p) = child.stderr.take() {
        std::thread::spawn(move || {
            let mut b = Vec::new();
            let _ = p.read_to_end(&mut b);
            let _ = err_tx.send(b);
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
