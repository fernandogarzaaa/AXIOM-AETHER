//! User-mode Neural VFS loopback.
//!
//! This is intentionally not a kernel driver. Windows kernel file-system
//! filters require admin installation and are not suitable for the default test
//! path. The module exposes the same read/getattr/readdir lifecycle in a safe
//! loopback layer and triggers Axiom's structural digest + TTT prefill whenever
//! a mounted source file is read through it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::Result as CandleResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::task::spawn_blocking;

use crate::context_compressor::{adapt_session_blocking, TttSessionStore};
use crate::inference::InferencePipeline;
use crate::skeleton::build_digest;

#[derive(Debug, Clone)]
pub struct NeuralVfs {
    root: Arc<RwLock<Option<PathBuf>>>,
    stats: Arc<Mutex<VfsStats>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VfsStats {
    pub mode: String,
    pub mounted_root: Option<String>,
    pub read_events: u64,
    pub readdir_events: u64,
    pub getattr_events: u64,
    pub last_path: Option<String>,
    pub last_session_id: Option<String>,
    pub last_digest_tokens: usize,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfsMountReport {
    pub mode: String,
    pub mounted_root: String,
    pub kernel_driver: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfsReadReport {
    pub path: String,
    pub bytes_read: usize,
    pub digest_tokens: usize,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfsAttr {
    pub path: String,
    pub is_dir: bool,
    pub len: u64,
    pub modified_unix_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfsDirEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
}

impl Default for NeuralVfs {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralVfs {
    pub fn new() -> Self {
        let stats = VfsStats {
            mode: "user_loopback".to_string(),
            ..VfsStats::default()
        };
        Self {
            root: Arc::new(RwLock::new(None)),
            stats: Arc::new(Mutex::new(stats)),
        }
    }

    pub fn mount(&self, root: impl AsRef<Path>) -> Result<VfsMountReport, String> {
        self.mount_with_allowed(root, vfs_allowed_roots().as_deref())
    }

    /// [`NeuralVfs::mount`] with the root allowlist passed in rather than read
    /// from `AXIOM_VFS_ALLOWED_ROOTS`. `None` allows any directory.
    pub fn mount_with_allowed(
        &self,
        root: impl AsRef<Path>,
        allowed_roots: Option<&[PathBuf]>,
    ) -> Result<VfsMountReport, String> {
        let root = root.as_ref();
        let canonical = root
            .canonicalize()
            .map_err(|e| format!("vfs mount root not accessible: {e}"))?;
        if !canonical.is_dir() {
            return Err("vfs mount root must be a directory".into());
        }
        if let Some(allowed) = allowed_roots {
            if !allowed.iter().any(|a| canonical.starts_with(a)) {
                return Err(format!(
                    "vfs mount root '{}' is outside AXIOM_VFS_ALLOWED_ROOTS",
                    canonical.display()
                ));
            }
        }
        *self
            .root
            .write()
            .map_err(|_| "vfs root lock poisoned".to_string())? = Some(canonical.clone());
        if let Ok(mut stats) = self.stats.lock() {
            stats.mode = "user_loopback".to_string();
            stats.mounted_root = Some(canonical.display().to_string());
            stats.last_error = None;
        }
        Ok(VfsMountReport {
            mode: "user_loopback".to_string(),
            mounted_root: canonical.display().to_string(),
            kernel_driver: false,
            message: "mounted as safe user-mode loopback; kernel driver not required".into(),
        })
    }

    pub fn status(&self) -> VfsStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn getattr(&self, path: impl AsRef<Path>) -> Result<VfsAttr, String> {
        let resolved = self.resolve(path.as_ref())?;
        let meta = std::fs::metadata(&resolved).map_err(|e| format!("getattr failed: {e}"))?;
        if let Ok(mut stats) = self.stats.lock() {
            stats.getattr_events += 1;
            stats.last_path = Some(resolved.display().to_string());
            stats.last_error = None;
        }
        Ok(VfsAttr {
            path: resolved.display().to_string(),
            is_dir: meta.is_dir(),
            len: meta.len(),
            modified_unix_secs: meta.modified().ok().and_then(system_time_secs),
        })
    }

    pub fn readdir(&self, path: impl AsRef<Path>) -> Result<Vec<VfsDirEntry>, String> {
        let resolved = self.resolve(path.as_ref())?;
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(&resolved).map_err(|e| format!("readdir failed: {e}"))? {
            let entry = entry.map_err(|e| format!("readdir entry failed: {e}"))?;
            let meta = entry
                .metadata()
                .map_err(|e| format!("readdir metadata failed: {e}"))?;
            entries.push(VfsDirEntry {
                name: entry.file_name().to_string_lossy().to_string(),
                path: entry.path().display().to_string(),
                is_dir: meta.is_dir(),
            });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        if let Ok(mut stats) = self.stats.lock() {
            stats.readdir_events += 1;
            stats.last_path = Some(resolved.display().to_string());
            stats.last_error = None;
        }
        Ok(entries)
    }

    pub async fn read_file_and_prefill(
        &self,
        path: impl AsRef<Path>,
        session_id: &str,
        pipeline: Arc<std::sync::RwLock<InferencePipeline>>,
        sessions: Arc<TttSessionStore>,
    ) -> Result<VfsReadReport, String> {
        let resolved = self.resolve(path.as_ref())?;
        // Only regular files, and only up to the read cap: the whole file is
        // read into memory and then tokenized, so a device node (`/dev/zero`)
        // or a multi-GB file would otherwise exhaust the server.
        let meta = tokio::fs::metadata(&resolved)
            .await
            .map_err(|e| format!("vfs read failed: {e}"))?;
        if !meta.is_file() {
            return Err(format!(
                "vfs read refused: '{}' is not a regular file",
                resolved.display()
            ));
        }
        let max_bytes = vfs_max_read_bytes();
        if meta.len() > max_bytes {
            return Err(format!(
                "vfs read refused: '{}' is {} bytes, above the {} byte cap (AXIOM_VFS_MAX_READ_BYTES)",
                resolved.display(),
                meta.len(),
                max_bytes
            ));
        }
        let text = tokio::fs::read_to_string(&resolved)
            .await
            .map_err(|e| format!("vfs read failed: {e}"))?;
        let digest = structural_digest_for_vfs(&resolved, &text);
        let session_id = session_id.trim();
        let session_id = if session_id.is_empty() {
            resolved.to_string_lossy().as_ref().to_string()
        } else {
            session_id.to_string()
        };
        let digest_for_task = digest.clone();
        let session_for_task = session_id.clone();
        let token_count: CandleResult<usize> = spawn_blocking(move || {
            let pipeline = pipeline
                .read()
                .map_err(|_| candle_core::Error::Msg("pipeline lock poisoned".into()))?;
            let handle = sessions.get_or_create(&session_for_task, &pipeline)?;
            let mut states = handle.blocking_lock();
            let tokens = pipeline.encode_text(&digest_for_task);
            adapt_session_blocking(&pipeline, &mut states, &tokens)?;
            Ok(tokens.len())
        })
        .await
        .map_err(|e| format!("vfs prefill join failed: {e}"))?;
        let token_count = token_count.map_err(|e| format!("vfs prefill failed: {e}"))?;
        let report = VfsReadReport {
            path: resolved.display().to_string(),
            bytes_read: text.len(),
            digest_tokens: token_count,
            session_id,
        };
        if let Ok(mut stats) = self.stats.lock() {
            stats.read_events += 1;
            stats.last_path = Some(report.path.clone());
            stats.last_session_id = Some(report.session_id.clone());
            stats.last_digest_tokens = report.digest_tokens;
            stats.last_error = None;
        }
        Ok(report)
    }

    fn resolve(&self, path: &Path) -> Result<PathBuf, String> {
        let root = self
            .root
            .read()
            .map_err(|_| "vfs root lock poisoned".to_string())?
            .clone()
            .ok_or_else(|| "vfs root is not mounted".to_string())?;
        let candidate = if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        };
        let canonical = candidate
            .canonicalize()
            .map_err(|e| format!("vfs path not accessible: {e}"))?;
        if !canonical.starts_with(&root) {
            return Err("vfs path escapes mounted root".into());
        }
        Ok(canonical)
    }
}

/// Default cap on a single `/v1/hypervisor/read` (16 MiB).
pub const DEFAULT_VFS_MAX_READ_BYTES: u64 = 16 * 1024 * 1024;

/// `AXIOM_VFS_MAX_READ_BYTES` or the default cap.
fn vfs_max_read_bytes() -> u64 {
    parse_max_read_bytes(std::env::var("AXIOM_VFS_MAX_READ_BYTES").ok().as_deref())
}

/// A positive byte count, or the default for unset / empty / zero / garbage.
fn parse_max_read_bytes(raw: Option<&str>) -> u64 {
    raw.and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_VFS_MAX_READ_BYTES)
}

/// `AXIOM_VFS_ALLOWED_ROOTS` (directories separated by `;`), canonicalized.
/// `None` when unset: any directory may be mounted, the local-first default
/// (the data plane is loopback-only or API-key guarded). Set it on any server
/// reachable by more than its operator so a caller cannot mount `/`.
fn vfs_allowed_roots() -> Option<Vec<PathBuf>> {
    parse_allowed_roots(std::env::var("AXIOM_VFS_ALLOWED_ROOTS").ok().as_deref())
}

/// Parse a `;`-separated root list. Unset or blank ⇒ `None` (no
/// restriction). Set but naming only directories that do not exist ⇒
/// `Some(empty)`, which refuses every mount: a misconfigured allowlist fails
/// closed rather than silently allowing everything.
fn parse_allowed_roots(raw: Option<&str>) -> Option<Vec<PathBuf>> {
    let raw = raw?;
    if raw.trim().is_empty() {
        return None;
    }
    Some(
        raw.split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter_map(|s| Path::new(s).canonicalize().ok())
            .collect(),
    )
}

pub fn structural_digest_for_vfs(path: &Path, text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(text.as_bytes());
    let state_hash = format!("sha256:{:x}", hasher.finalize());
    build_digest(
        text,
        &path.to_string_lossy(),
        text.split_whitespace().count(),
        0.0,
        &state_hash,
        8,
    )
}

fn system_time_secs(t: SystemTime) -> Option<u64> {
    t.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn vfs_digest_strips_rust_bodies() {
        let digest = structural_digest_for_vfs(
            Path::new("demo.rs"),
            "pub fn run() -> i32 { let secret = 41; secret + 1 }",
        );
        assert!(digest.contains("fn run"));
        assert!(!digest.contains("secret + 1"));
    }

    // These tests never touch the process environment: tests run in
    // parallel threads, so setting AXIOM_VFS_* here would leak into
    // `mount_readdir_getattr_roundtrip` and others running at the same time.

    #[test]
    fn mount_honours_allowed_roots() {
        let allowed = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(allowed.path().join("sub")).unwrap();
        let roots = parse_allowed_roots(Some(&allowed.path().to_string_lossy())).unwrap();
        let vfs = NeuralVfs::new();
        assert!(vfs
            .mount_with_allowed(allowed.path().join("sub"), Some(&roots))
            .is_ok());
        let err = vfs
            .mount_with_allowed(other.path(), Some(&roots))
            .unwrap_err();
        assert!(err.contains("AXIOM_VFS_ALLOWED_ROOTS"), "{err}");
        // No allowlist → any directory mounts (the local-first default).
        assert!(vfs.mount_with_allowed(other.path(), None).is_ok());
    }

    #[test]
    fn allowed_roots_parsing() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        assert_eq!(parse_allowed_roots(None), None);
        assert_eq!(parse_allowed_roots(Some("  ")), None);
        let both = format!("{} ; {};", a.path().display(), b.path().display());
        let parsed = parse_allowed_roots(Some(&both)).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!(parsed.iter().all(|p| p.is_absolute()));
        // Only nonexistent directories: fail closed (an empty allowlist).
        let missing = parse_allowed_roots(Some("/definitely/not/a/real/dir")).unwrap();
        assert!(missing.is_empty());
        let vfs = NeuralVfs::new();
        assert!(vfs.mount_with_allowed(a.path(), Some(&missing)).is_err());
    }

    #[test]
    fn read_cap_defaults_and_parses() {
        assert_eq!(DEFAULT_VFS_MAX_READ_BYTES, 16 * 1024 * 1024);
        assert_eq!(parse_max_read_bytes(Some("123")), 123);
        assert_eq!(parse_max_read_bytes(Some(" 456 ")), 456);
        assert_eq!(parse_max_read_bytes(Some("0")), DEFAULT_VFS_MAX_READ_BYTES);
        assert_eq!(
            parse_max_read_bytes(Some("lots")),
            DEFAULT_VFS_MAX_READ_BYTES
        );
        assert_eq!(parse_max_read_bytes(None), DEFAULT_VFS_MAX_READ_BYTES);
    }

    #[test]
    fn mount_readdir_getattr_roundtrip() {
        let root = std::env::temp_dir().join(format!("axiom-vfs-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("lib.rs"), "pub fn ok() {}").unwrap();
        let vfs = NeuralVfs::new();
        let mount = vfs.mount(&root).unwrap();
        assert!(!mount.kernel_driver);
        let attr = vfs.getattr("lib.rs").unwrap();
        assert!(!attr.is_dir);
        let entries = vfs.readdir(".").unwrap();
        assert_eq!(entries.len(), 1);
        let _ = std::fs::remove_dir_all(root);
    }
}
