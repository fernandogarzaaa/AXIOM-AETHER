use std::fs::{self, OpenOptions};
use std::io;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::json;

use crate::config::{load_or_init_user_config, AxiomPaths, UserConfig};

#[derive(Debug, Clone)]
pub struct DaemonStatus {
    pub pid: Option<u32>,
    pub running: bool,
    pub log_path: PathBuf,
    pub pid_file: PathBuf,
    pub endpoint: String,
}

pub fn start() -> io::Result<DaemonStatus> {
    let (paths, cfg, _) = load_or_init_user_config()?;
    if let Some(status) = status_from_paths(&paths, &cfg)? {
        if status.running {
            return Ok(status);
        }
    }
    paths.create_all()?;
    let exe = std::env::current_exe()?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.hypervisor_log)?;
    let err = log.try_clone()?;
    let mut command = Command::new(exe);
    command
        .arg("--mode")
        .arg("server")
        .arg("--host")
        .arg(&cfg.runtime.host)
        .arg("--port")
        .arg(cfg.runtime.port.to_string())
        .arg("--device")
        .arg(&cfg.runtime.device)
        .env(
            "AXIOM_MAX_CONTEXT_TOKENS",
            cfg.runtime.max_context_tokens.to_string(),
        )
        .env("AXIOM_DWE_PEERS", cfg.swarm.peers.join(","))
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err));
    detach_command(&mut command);
    let child = command.spawn()?;
    fs::write(&paths.pid_file, child.id().to_string())?;
    Ok(DaemonStatus {
        pid: Some(child.id()),
        running: true,
        log_path: paths.hypervisor_log,
        pid_file: paths.pid_file,
        endpoint: endpoint(&cfg),
    })
}

pub fn stop() -> io::Result<DaemonStatus> {
    let (paths, cfg, _) = load_or_init_user_config()?;
    let mut status = status_from_paths(&paths, &cfg)?.unwrap_or_else(|| DaemonStatus {
        pid: None,
        running: false,
        log_path: paths.hypervisor_log.clone(),
        pid_file: paths.pid_file.clone(),
        endpoint: endpoint(&cfg),
    });
    if let Some(pid) = status.pid {
        if status.running {
            terminate_pid(pid)?;
        }
    }
    let _ = fs::remove_file(&paths.pid_file);
    status.running = false;
    Ok(status)
}

pub fn status() -> io::Result<DaemonStatus> {
    let (paths, cfg, _) = load_or_init_user_config()?;
    Ok(
        status_from_paths(&paths, &cfg)?.unwrap_or_else(|| DaemonStatus {
            pid: None,
            running: false,
            log_path: paths.hypervisor_log,
            pid_file: paths.pid_file,
            endpoint: endpoint(&cfg),
        }),
    )
}

pub fn mount(path: PathBuf) -> Result<String, String> {
    let status = status().map_err(|e| format!("could not inspect daemon: {e}"))?;
    if !status.running {
        return Err(format!(
            "Axiom daemon is not running. Start it with `axiom daemon start` before mounting {}.",
            path.display()
        ));
    }
    let root = path
        .canonicalize()
        .map_err(|e| format!("mount path is not accessible: {e}"))?;
    let url = format!("{}/v1/hypervisor/mount", status.endpoint);
    let body = json!({
        "root": root.display().to_string(),
        "session_id": "hypervisor-vfs",
        "warm_paths": []
    });
    let response = reqwest::blocking::Client::new()
        .post(url)
        .json(&body)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("daemon mount request failed: {e}"))?;
    response
        .text()
        .map_err(|e| format!("daemon mount response failed: {e}"))
}

fn status_from_paths(paths: &AxiomPaths, cfg: &UserConfig) -> io::Result<Option<DaemonStatus>> {
    let pid = match fs::read_to_string(&paths.pid_file) {
        Ok(raw) => raw.trim().parse::<u32>().ok(),
        Err(err) if err.kind() == io::ErrorKind::NotFound => None,
        Err(err) => return Err(err),
    };
    let running = pid.map(process_exists).unwrap_or(false);
    Ok(Some(DaemonStatus {
        pid,
        running,
        log_path: paths.hypervisor_log.clone(),
        pid_file: paths.pid_file.clone(),
        endpoint: endpoint(cfg),
    }))
}

fn endpoint(cfg: &UserConfig) -> String {
    format!("http://{}:{}", cfg.runtime.host, cfg.runtime.port)
}

#[cfg(windows)]
fn detach_command(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn detach_command(_command: &mut Command) {}

#[cfg(windows)]
fn process_exists(pid: u32) -> bool {
    let filter = format!("PID eq {pid}");
    Command::new("tasklist")
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|out| out.contains(&pid.to_string()))
        .unwrap_or(false)
}

#[cfg(not(windows))]
fn process_exists(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(windows)]
fn terminate_pid(pid: u32) -> io::Result<()> {
    let status = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::Other, "taskkill failed"))
    }
}

#[cfg(not(windows))]
fn terminate_pid(pid: u32) -> io::Result<()> {
    let status = Command::new("kill").arg(pid.to_string()).status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other("kill failed"))
    }
}
