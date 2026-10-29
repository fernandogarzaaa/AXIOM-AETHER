use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const DEFAULT_CHECKPOINT_PATH: &str = "axiom_kernel_v1.safetensors";
pub const DEFAULT_EOS_TOKEN: u32 = 2;
pub const DEFAULT_BASE_MODEL_FILE: &str = "axiom_production_bpe.bin";
pub const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/fggg23/axiom-multilang-v1/resolve/main/axiom_production_bpe.bin";
/// Tokenizer published next to the checkpoint on the Hub; fetched by `axiom
/// init` from the checkpoint URL's directory.
pub const DEFAULT_TOKENIZER_FILE: &str = "axiom_bpe.json";
/// Architecture sidecar published next to the checkpoint (see `model_meta`).
pub const DEFAULT_META_FILE: &str = "axiom_production_bpe.meta.json";

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

    pub fn tokenizer_path(&self, cfg: &UserConfig) -> PathBuf {
        self.models_dir.join(&cfg.models.tokenizer_file)
    }

    pub fn meta_path(&self, cfg: &UserConfig) -> PathBuf {
        self.models_dir.join(&cfg.models.meta_file)
    }

    /// The installed production model, if `axiom init` (or the user) put both
    /// the checkpoint and its tokenizer under `models/`.
    pub fn user_production_model(&self, cfg: &UserConfig) -> Option<UserProductionModel> {
        let checkpoint = self.base_model_path(cfg);
        let tokenizer = self.tokenizer_path(cfg);
        (checkpoint.is_file() && tokenizer.is_file()).then_some(UserProductionModel {
            checkpoint,
            tokenizer,
        })
    }
}

/// Look up the production model `axiom init` installed under `~/.axiom`,
/// reading `config.toml` when present (defaults otherwise) and never writing
/// anything. `None` when there is no `~/.axiom`, no config, or either file
/// is missing, so callers fall back to the legacy model exactly as before.
pub fn installed_production_model() -> Option<UserProductionModel> {
    let paths = AxiomPaths::discover().ok()?;
    let cfg = match fs::read_to_string(&paths.config) {
        Ok(raw) => toml::from_str::<UserConfig>(&raw).ok()?,
        Err(_) => UserConfig::default(),
    };
    paths.user_production_model(&cfg)
}

/// Which model the runtime should load, decided from the three inputs that
/// matter and nothing else (pure, so it is unit-testable):
///
/// * `flag` is `AXIOM_PRODUCTION_BPE`: `Some("1")` forces the production
///   path, any other `Some` forces the legacy model, `None` means "auto".
/// * `env_paths` is `(AXIOM_BPE_CKPT, AXIOM_TOKENIZER)` as set by the user.
/// * `installed` is what `axiom init` put under `~/.axiom/models/`.
///
/// Auto picks the installed model when neither env path is set and both
/// files exist; otherwise it stays on the legacy model. An explicit `1`
/// uses the env paths when given and the installed model as the fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductionSelection {
    Legacy,
    /// `(checkpoint, tokenizer, auto_selected)`.
    Production(PathBuf, PathBuf, bool),
}

pub fn select_production_model(
    flag: Option<&str>,
    env_paths: (Option<&str>, Option<&str>),
    installed: Option<&UserProductionModel>,
) -> ProductionSelection {
    let (env_ckpt, env_tok) = env_paths;
    match flag {
        Some("1") => {
            let ckpt = env_ckpt.map(PathBuf::from);
            let tok = env_tok.map(PathBuf::from);
            match (ckpt, tok, installed) {
                (Some(c), Some(t), _) => ProductionSelection::Production(c, t, false),
                (None, None, Some(m)) => ProductionSelection::Production(
                    m.checkpoint.clone(),
                    m.tokenizer.clone(),
                    false,
                ),
                (c, t, _) => ProductionSelection::Production(
                    c.unwrap_or_else(|| PathBuf::from("checkpoints/axiom_production_bpe.bin")),
                    t.unwrap_or_else(|| PathBuf::from("checkpoints/axiom_bpe.json")),
                    false,
                ),
            }
        }
        Some(_) => ProductionSelection::Legacy,
        None => match (env_ckpt, env_tok, installed) {
            (None, None, Some(m)) => {
                ProductionSelection::Production(m.checkpoint.clone(), m.tokenizer.clone(), true)
            }
            _ => ProductionSelection::Legacy,
        },
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
    #[serde(default)]
    pub features: FeatureConfig,
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
    /// Tokenizer file name, fetched from the same directory as
    /// `base_model_url` into `~/.axiom/models/`. Older `config.toml` files
    /// without this field get the default.
    #[serde(default = "default_tokenizer_file")]
    pub tokenizer_file: String,
    /// Architecture sidecar file name, fetched the same way. Optional at
    /// runtime (the engine falls back to the legacy dims without it) but the
    /// published checkpoints all ship one.
    #[serde(default = "default_meta_file")]
    pub meta_file: String,
}

fn default_tokenizer_file() -> String {
    DEFAULT_TOKENIZER_FILE.to_string()
}

fn default_meta_file() -> String {
    DEFAULT_META_FILE.to_string()
}

/// The production model `axiom init` installed under `~/.axiom/models/`, when
/// both halves the engine needs are present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserProductionModel {
    pub checkpoint: PathBuf,
    pub tokenizer: PathBuf,
}

/// Replace the last path segment of `url` with `file`
/// (`…/resolve/main/axiom_production_bpe.bin` + `axiom_bpe.json` →
/// `…/resolve/main/axiom_bpe.json`). A URL with no `/` is returned unchanged.
pub fn sibling_url(url: &str, file: &str) -> String {
    match url.rfind('/') {
        Some(i) => format!("{}/{file}", &url[..i]),
        None => url.to_string(),
    }
}

/// Opt-in server features that were previously configurable only via
/// environment variables read at server-process startup (`CompressorConfig`/
/// `SwarmRouterConfig`'s own `from_env`). Persisting them here lets `axiom
/// daemon start` carry a user's choices across restarts instead of requiring
/// them to be re-exported into the shell every time; `daemon::start` turns
/// these into the same env vars a manually-launched server already reads, so
/// behavior for either launch path stays identical.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureConfig {
    pub ttt_compress: bool,
    pub ttt_compress_threshold_tokens: usize,
    pub swarm_local: bool,
    pub mesh_routing: bool,
    pub ollama_url: String,
    pub ollama_models: Vec<String>,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            ttt_compress: false,
            ttt_compress_threshold_tokens: 512,
            swarm_local: false,
            mesh_routing: false,
            ollama_url: "http://127.0.0.1:11434".into(),
            ollama_models: vec![
                "phi4:3.8b".into(),
                "deepseek-r1:8b".into(),
                "llama3.3:8b".into(),
                "llama3.1:8b".into(),
            ],
        }
    }
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
                tokenizer_file: DEFAULT_TOKENIZER_FILE.into(),
                meta_file: DEFAULT_META_FILE.into(),
            },
            features: FeatureConfig::default(),
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
    if !target.exists() {
        if !cfg.models.auto_fetch {
            return Ok(None);
        }
        fetch_base_model(&cfg.models.base_model_url, &target, None)?;
    }
    // The checkpoint is useless to the engine without its tokenizer (and the
    // sidecar carries the dims). Fetch both from the same Hub directory; a
    // failure here is reported but does not undo the checkpoint download.
    if cfg.models.auto_fetch {
        for (file, dest, required) in [
            (
                cfg.models.tokenizer_file.as_str(),
                paths.tokenizer_path(cfg),
                true,
            ),
            (cfg.models.meta_file.as_str(), paths.meta_path(cfg), false),
        ] {
            if dest.exists() {
                continue;
            }
            let url = sibling_url(&cfg.models.base_model_url, file);
            if let Err(e) = fetch_base_model(&url, &dest, None) {
                if required {
                    eprintln!(
                        "[axiom] tokenizer fetch failed ({e}); the production model needs {} next to \
                         the checkpoint in {} — place it there or set AXIOM_TOKENIZER",
                        file,
                        paths.models_dir.display()
                    );
                } else {
                    eprintln!("[axiom] sidecar fetch skipped ({e}); legacy dims will be assumed");
                }
            }
        }
    }
    Ok(Some(target))
}

/// Download `url` into `target`, verifying integrity before the file is ever
/// visible under its final name.
///
/// When `expected_sha256` is `Some`, the download is hashed while streaming
/// and compared case-insensitively; a mismatch deletes the partial download
/// and returns an error instead of installing an unverified checkpoint —
/// a checkpoint fetched over the network and loaded straight into the
/// runtime is a supply-chain input, so a corrupted or substituted artifact
/// must fail loudly rather than silently become "the model". When
/// `expected_sha256` is `None` (the common case: no pinned hash configured),
/// the computed digest is still printed so an operator can capture and pin
/// it for next time; this is not a security check by itself, only a record.
fn fetch_base_model(url: &str, target: &Path, expected_sha256: Option<&str>) -> io::Result<()> {
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
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 64 * 1024];
    let mut downloaded = 0_u64;
    loop {
        let n = response.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        downloaded += n as u64;
        pb.set_position(downloaded);
    }
    file.flush()?;
    drop(file);
    pb.finish_and_clear();
    let digest = format!("{:x}", hasher.finalize());
    if let Err(e) = verify_checksum(expected_sha256, &digest) {
        let _ = fs::remove_file(&tmp);
        return Err(io::Error::other(format!(
            "checkpoint integrity check failed for {url}: {e}"
        )));
    }
    match expected_sha256 {
        Some(_) => println!("[axiom] checkpoint sha256 verified: {digest}"),
        None => println!(
            "[axiom] fetched {url} (sha256={digest}, unverified — no expected hash configured; \
             pin it to verify future fetches)"
        ),
    }
    fs::rename(tmp, target)?;
    Ok(())
}

/// Compare a computed digest against an optional pinned one. `None` always
/// passes (nothing was pinned to check against); a mismatch is reported with
/// both values so the failure is actionable from the message alone.
fn verify_checksum(expected_sha256: Option<&str>, actual_digest: &str) -> Result<(), String> {
    match expected_sha256 {
        None => Ok(()),
        Some(expected) if expected.eq_ignore_ascii_case(actual_digest) => Ok(()),
        Some(expected) => Err(format!(
            "expected sha256={expected}, got sha256={actual_digest}. The partial download was \
             deleted; refusing to install an unverified checkpoint."
        )),
    }
}

/// Fetch the production BPE checkpoint (+ tokenizer) from a release URL if
/// one is configured and the files aren't already present.
///
/// Mirrors the `AXIOM_CHECKPOINT_URL` / `AXIOM_TOKENIZER_URL` convention
/// `scripts/docker_entrypoint.sh` already uses, so the same release asset
/// works for both the container entrypoint and `axiom init`. Every prior
/// audit of this repo found that a fresh clone (and CI) never has a trained
/// checkpoint — `axiom init` only ever bootstrapped the small legacy model
/// (see `bootstrap.rs`), never the production BPE one. This closes that gap
/// when a URL is configured; unset (the default), `axiom init`'s behavior is
/// unchanged.
///
/// Best-effort: a fetch failure or unset URL returns `Ok(false)` rather than
/// erroring, so callers should fall back to the offline bootstrap. A checksum
/// *mismatch* is the one exception — that is a verification failure, not a
/// network hiccup, and propagates as `Err` so it cannot be silently treated
/// as "no checkpoint configured, fall back".
///
/// Set `AXIOM_CHECKPOINT_SHA256` / `AXIOM_TOKENIZER_SHA256` to pin the
/// expected digest of each download; unset (the default) fetches without
/// enforcement but still logs the computed hash. See docs/SECURITY-AUDIT.md.
pub fn ensure_production_checkpoint() -> io::Result<bool> {
    let Ok(ckpt_url) = std::env::var("AXIOM_CHECKPOINT_URL") else {
        return Ok(false);
    };
    let ckpt_path = std::env::var("AXIOM_BPE_CKPT")
        .unwrap_or_else(|_| "checkpoints/axiom_production_bpe.bin".to_string());
    let ckpt_target = Path::new(&ckpt_path);
    if ckpt_target.exists() {
        return Ok(false);
    }
    let ckpt_sha256 = std::env::var("AXIOM_CHECKPOINT_SHA256").ok();
    fetch_base_model(&ckpt_url, ckpt_target, ckpt_sha256.as_deref())?;

    if let Ok(tok_url) = std::env::var("AXIOM_TOKENIZER_URL") {
        let tok_path = std::env::var("AXIOM_TOKENIZER")
            .unwrap_or_else(|_| "checkpoints/axiom_bpe.json".to_string());
        let tok_target = Path::new(&tok_path);
        if !tok_target.exists() {
            let tok_sha256 = std::env::var("AXIOM_TOKENIZER_SHA256").ok();
            if let Err(e) = fetch_base_model(&tok_url, tok_target, tok_sha256.as_deref()) {
                eprintln!(
                    "[axiom] tokenizer fetch failed ({e}); checkpoint was fetched but the \
                     production model needs a matching tokenizer too."
                );
            }
        }
    }
    Ok(true)
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

    #[test]
    fn sibling_url_replaces_last_segment() {
        assert_eq!(
            sibling_url(DEFAULT_MODEL_URL, "axiom_bpe.json"),
            "https://huggingface.co/fggg23/axiom-multilang-v1/resolve/main/axiom_bpe.json"
        );
        assert_eq!(sibling_url("nofile", "x"), "nofile");
    }

    #[test]
    fn old_config_without_tokenizer_fields_still_loads() {
        // A config.toml written before tokenizer_file/meta_file existed.
        let mut raw = toml::to_string(&UserConfig::default()).unwrap();
        raw = raw
            .lines()
            .filter(|l| !l.starts_with("tokenizer_file") && !l.starts_with("meta_file"))
            .collect::<Vec<_>>()
            .join("\n");
        let parsed: UserConfig = toml::from_str(&raw).unwrap();
        assert_eq!(parsed.models.tokenizer_file, DEFAULT_TOKENIZER_FILE);
        assert_eq!(parsed.models.meta_file, DEFAULT_META_FILE);
    }

    #[test]
    fn user_production_model_needs_both_files() {
        let home = tempfile::tempdir().unwrap();
        let paths = AxiomPaths::from_home(home.path().join(".axiom"));
        paths.create_all().unwrap();
        let cfg = UserConfig::default();
        assert_eq!(paths.user_production_model(&cfg), None);
        fs::write(paths.base_model_path(&cfg), b"ckpt").unwrap();
        assert_eq!(paths.user_production_model(&cfg), None, "tokenizer missing");
        fs::write(paths.tokenizer_path(&cfg), b"{}").unwrap();
        let m = paths.user_production_model(&cfg).unwrap();
        assert_eq!(m.checkpoint, paths.base_model_path(&cfg));
        assert_eq!(m.tokenizer, paths.tokenizer_path(&cfg));
    }

    #[test]
    fn production_selection_rules() {
        use ProductionSelection::*;
        let installed = UserProductionModel {
            checkpoint: PathBuf::from("/h/.axiom/models/axiom_production_bpe.bin"),
            tokenizer: PathBuf::from("/h/.axiom/models/axiom_bpe.json"),
        };
        // Unset flag, nothing installed → legacy (the historical default).
        assert_eq!(select_production_model(None, (None, None), None), Legacy);
        // Unset flag, installed model → auto-selected.
        assert_eq!(
            select_production_model(None, (None, None), Some(&installed)),
            Production(
                installed.checkpoint.clone(),
                installed.tokenizer.clone(),
                true
            )
        );
        // Unset flag but an explicit env path → stays legacy (env paths only
        // mean something with AXIOM_PRODUCTION_BPE=1, as before).
        assert_eq!(
            select_production_model(None, (Some("x.bin"), None), Some(&installed)),
            Legacy
        );
        // Explicit 0 always wins.
        assert_eq!(
            select_production_model(Some("0"), (None, None), Some(&installed)),
            Legacy
        );
        // Explicit 1 with env paths uses them verbatim.
        assert_eq!(
            select_production_model(Some("1"), (Some("c.bin"), Some("t.json")), Some(&installed)),
            Production(PathBuf::from("c.bin"), PathBuf::from("t.json"), false)
        );
        // Explicit 1 without env paths falls back to the installed model...
        assert_eq!(
            select_production_model(Some("1"), (None, None), Some(&installed)),
            Production(
                installed.checkpoint.clone(),
                installed.tokenizer.clone(),
                false
            )
        );
        // ...and to the historical checkpoints/ defaults when nothing is installed.
        assert_eq!(
            select_production_model(Some("1"), (None, None), None),
            Production(
                PathBuf::from("checkpoints/axiom_production_bpe.bin"),
                PathBuf::from("checkpoints/axiom_bpe.json"),
                false
            )
        );
    }

    #[test]
    fn verify_checksum_passes_when_nothing_pinned() {
        assert!(verify_checksum(None, "anything").is_ok());
    }

    #[test]
    fn verify_checksum_passes_on_case_insensitive_match() {
        assert!(verify_checksum(Some("ABCDEF"), "abcdef").is_ok());
    }

    #[test]
    fn verify_checksum_fails_on_mismatch_with_both_hashes_in_message() {
        let err = verify_checksum(Some("expected123"), "actual456").unwrap_err();
        assert!(err.contains("expected123"));
        assert!(err.contains("actual456"));
    }

    #[test]
    fn a_config_toml_written_before_features_existed_still_loads() {
        let cfg = UserConfig::default();
        let mut raw = toml::to_string(&cfg).unwrap();
        let features_start = raw.find("[features]").expect("features section present");
        raw.truncate(features_start);
        let parsed: UserConfig = toml::from_str(&raw).unwrap();
        assert_eq!(parsed.features, FeatureConfig::default());
    }
}
