use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

pub const DEFAULT_CHECKPOINT_PATH: &str = "axiom_kernel_v1.safetensors";
pub const DEFAULT_EOS_TOKEN: u32 = 2;
pub const DEFAULT_BASE_MODEL_FILE: &str = "axiom-base-d256.safetensors";
pub const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/fernandogarzaaa/AXIOM-AETHER/resolve/main/axiom-base-d256.safetensors";

/// Static hyper-parameters for the Axiom-TTT inference engine.
#[derive(Debug, Clone)]
pub struct AxiomConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    /// Inner-loop learning rate for the TTT weight update.
    pub lr_inner: f32,
    pub norm_eps: f32,
}

impl AxiomConfig {
    /// Small CPU-friendly runtime dims — what the server, prime, bench, and the
    /// init bootstrap actually instantiate when no scaled checkpoint is present.
    pub fn runtime_small() -> Self {
        Self {
            d_model: 64,
            n_layers: 2,
            vocab_size: 256,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        }
    }

    /// The 7B-scale architectural blueprint. NEVER instantiate a runtime model
    /// from this on commodity hardware: materialising 32 layers of
    /// [4096 × 4096] fast-weights hangs/OOMs a CPU host (observed). It exists
    /// for documentation and sizing math only — which is why `Default` is the
    /// safe runtime config instead.
    pub fn blueprint_7b() -> Self {
        Self {
            d_model: 4096,
            n_layers: 32,
            vocab_size: 32000,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        }
    }
}

impl Default for AxiomConfig {
    fn default() -> Self {
        Self::runtime_small()
    }
}

#[derive(Debug, Clone)]
pub struct AxiomPaths {
    pub home: PathBuf,
    pub config: PathBuf,
    pub logs_dir: PathBuf,
    pub models_dir: PathBuf,
    pub run_dir: PathBuf,
    pub hypervisor_log: PathBuf,
    pub pid_file: PathBuf,
}

impl AxiomPaths {
    pub fn discover() -> io::Result<Self> {
        let home = dirs::home_dir()
            .or_else(|| std::env::current_dir().ok())
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "could not resolve home dir"))?
            .join(".axiom");
        Ok(Self::from_home(home))
    }

    pub fn from_home(home: PathBuf) -> Self {
        let logs_dir = home.join("logs");
        let models_dir = home.join("models");
        let run_dir = home.join("run");
        Self {
            config: home.join("config.toml"),
            hypervisor_log: logs_dir.join("hypervisor.log"),
            pid_file: run_dir.join("axiom.pid"),
            home,
            logs_dir,
            models_dir,
            run_dir,
        }
    }

    pub fn create_all(&self) -> io::Result<()> {
        fs::create_dir_all(&self.home)?;
        fs::create_dir_all(&self.logs_dir)?;
        fs::create_dir_all(&self.models_dir)?;
        fs::create_dir_all(&self.run_dir)?;
        Ok(())
    }

    pub fn base_model_path(&self, cfg: &UserConfig) -> PathBuf {
        self.models_dir.join(&cfg.models.base_model_file)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UserConfig {
    pub runtime: RuntimeConfig,
    pub model: ModelRuntimeConfig,
    pub vfs: VfsRuntimeConfig,
    pub surprisal: SurprisalConfig,
    pub swarm: SwarmConfig,
    pub models: ModelFetchConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeConfig {
    pub host: String,
    pub port: u16,
    pub device: String,
    pub vram_budget_mb: u32,
    pub max_context_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelRuntimeConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VfsRuntimeConfig {
    pub loopback_port: u16,
    pub default_session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurprisalConfig {
    pub threshold: f32,
    pub exact_residual_max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SwarmConfig {
    pub dwe_bind: String,
    pub peers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelFetchConfig {
    pub base_model_file: String,
    pub base_model_url: String,
    pub auto_fetch: bool,
}

impl Default for UserConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig {
                host: "127.0.0.1".into(),
                port: 8080,
                device: "auto".into(),
                vram_budget_mb: 5200,
                max_context_tokens: 4096,
            },
            model: ModelRuntimeConfig {
                d_model: 256,
                n_layers: 4,
                vocab_size: 16_000,
            },
            vfs: VfsRuntimeConfig {
                loopback_port: 8765,
                default_session_id: "hypervisor-vfs".into(),
            },
            surprisal: SurprisalConfig {
                threshold: 8.0,
                exact_residual_max_tokens: 4096,
            },
            swarm: SwarmConfig {
                dwe_bind: "127.0.0.1:9191".into(),
                peers: Vec::new(),
            },
            models: ModelFetchConfig {
                base_model_file: DEFAULT_BASE_MODEL_FILE.into(),
                base_model_url: DEFAULT_MODEL_URL.into(),
                auto_fetch: true,
            },
        }
    }
}

pub fn load_or_init_user_config() -> io::Result<(AxiomPaths, UserConfig, bool)> {
    let paths = AxiomPaths::discover()?;
    paths.create_all()?;
    if paths.config.exists() {
        let raw = fs::read_to_string(&paths.config)?;
        let cfg: UserConfig = toml::from_str(&raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        return Ok((paths, cfg, false));
    }
    let cfg = UserConfig::default();
    write_user_config(&paths, &cfg)?;
    Ok((paths, cfg, true))
}

pub fn write_user_config(paths: &AxiomPaths, cfg: &UserConfig) -> io::Result<()> {
    paths.create_all()?;
    let raw = toml::to_string_pretty(cfg)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    fs::write(&paths.config, raw)
}

pub fn add_swarm_peer(peer: &str) -> io::Result<(AxiomPaths, UserConfig, bool)> {
    let (paths, mut cfg, _) = load_or_init_user_config()?;
    let peer = peer.trim();
    if peer.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "peer address cannot be empty",
        ));
    }
    let inserted = if cfg.swarm.peers.iter().any(|p| p == peer) {
        false
    } else {
        cfg.swarm.peers.push(peer.to_string());
        true
    };
    write_user_config(&paths, &cfg)?;
    Ok((paths, cfg, inserted))
}

pub fn ensure_base_model(paths: &AxiomPaths, cfg: &UserConfig) -> io::Result<Option<PathBuf>> {
    let target = paths.base_model_path(cfg);
    if target.exists() {
        return Ok(Some(target));
    }
    if !cfg.models.auto_fetch {
        return Ok(None);
    }
    fetch_base_model(&cfg.models.base_model_url, &target)?;
    Ok(Some(target))
}

fn fetch_base_model(url: &str, target: &Path) -> io::Result<()> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut response = reqwest::blocking::get(url)
        .and_then(|r| r.error_for_status())
        .map_err(|e| io::Error::other(format!("model fetch failed: {e}")))?;
    let total = response.content_length().unwrap_or(0);
    let pb = if total > 0 {
        ProgressBar::new(total)
    } else {
        ProgressBar::new_spinner()
    };
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {bytes}/{total_bytes} {wide_bar}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    let tmp = target.with_extension("download");
    let mut file = fs::File::create(&tmp)?;
    let mut buf = [0_u8; 64 * 1024];
    let mut downloaded = 0_u64;
    loop {
        let n = response.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }
    file.flush()?;
    fs::rename(tmp, target)?;
    pb.finish_and_clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_config_round_trips_as_toml() {
        let cfg = UserConfig::default();
        let raw = toml::to_string(&cfg).unwrap();
        let parsed: UserConfig = toml::from_str(&raw).unwrap();
        assert_eq!(parsed, cfg);
        assert_eq!(parsed.runtime.vram_budget_mb, 5200);
    }
}
