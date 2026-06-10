//! Learned immunity: persistent heal memory for supervised programs.
//!
//! `axiom run` heals a program's environment reactively — it waits for the
//! crash, repairs, and restarts. This module adds the *acquired* layer: every
//! successful heal is remembered against a stable fingerprint of the command,
//! together with the tension (CE) profile of its past failures. The next time
//! the same program runs — even in a fresh environment — the supervisor
//! **immunizes** it first: remembered directories are re-created *before* the
//! first attempt, so the failure never happens at all.
//!
//! The memory also keeps a running mean of failure-trace CE per program, so a
//! new failure can be classified as a KNOWN failure mode (tension close to the
//! historical mean) or a NOVEL one (a genuine drift spike). The classification
//! is informational with an unbaked checkpoint and sharp with the trained
//! semantic model — the plumbing is identical.
//!
//! Storage is a small human-readable JSON file (default
//! `~/.axiom/heal_memory.json`). Only directory heals are remembered: transient
//! retries are situational and never replayed prophylactically.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// What the memory knows about one supervised program.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ProgramRecord {
    /// Human-readable command line (for inspection of the JSON; the map key is
    /// the fingerprint).
    pub command: String,
    /// Directories this program needed created in the past.
    pub dirs: Vec<PathBuf>,
    /// Running mean of failure-trace cross-entropy.
    pub ce_mean: f32,
    /// Number of failures folded into `ce_mean`.
    pub ce_count: u32,
}

/// Classification of a fresh failure against the program's tension history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Novelty {
    /// First observed failure — nothing to compare against.
    First,
    /// Tension within tolerance of the historical mean: a known failure mode.
    Known,
    /// Tension deviates from the historical mean: a new failure mode.
    Novel,
}

impl std::fmt::Display for Novelty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Novelty::First => write!(f, "FIRST"),
            Novelty::Known => write!(f, "KNOWN"),
            Novelty::Novel => write!(f, "NOVEL"),
        }
    }
}

/// Relative CE deviation below which a failure counts as a known mode.
const KNOWN_TOLERANCE: f32 = 0.05;

#[derive(Debug, Default, Serialize, Deserialize)]
struct MemoryFile {
    programs: HashMap<String, ProgramRecord>,
}

/// Persistent heal memory, loaded eagerly and saved explicitly.
#[derive(Debug)]
pub struct HealMemory {
    path: PathBuf,
    data: MemoryFile,
}

/// Stable fingerprint of a supervised command line.
pub fn fingerprint(command: &str, args: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(command.as_bytes());
    for a in args {
        hasher.update([0u8]);
        hasher.update(a.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

impl HealMemory {
    /// Load the memory at `path`, or start empty when missing/corrupt (a bad
    /// memory file must never break a run).
    pub fn load(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let data = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self { path, data }
    }

    pub fn record(&self, fp: &str) -> Option<&ProgramRecord> {
        self.data.programs.get(fp)
    }

    /// Re-create every remembered directory that is missing. Returns the dirs
    /// actually created now (the immunization applied to *this* environment).
    pub fn immunize(&self, fp: &str) -> Vec<PathBuf> {
        let Some(record) = self.data.programs.get(fp) else {
            return Vec::new();
        };
        let mut applied = Vec::new();
        for dir in &record.dirs {
            if !dir.exists() && std::fs::create_dir_all(dir).is_ok() {
                applied.push(dir.clone());
            }
        }
        applied
    }

    /// Fold one failure's tension into the program's CE history and classify it
    /// against what came before.
    pub fn observe_failure(&mut self, fp: &str, command_line: &str, ce: f32) -> Novelty {
        let record = self
            .data
            .programs
            .entry(fp.to_string())
            .or_insert_with(|| ProgramRecord {
                command: command_line.to_string(),
                ..ProgramRecord::default()
            });
        let novelty = if record.ce_count == 0 {
            Novelty::First
        } else if !ce.is_finite() || record.ce_mean == 0.0 {
            Novelty::Novel
        } else if ((ce - record.ce_mean) / record.ce_mean).abs() <= KNOWN_TOLERANCE {
            Novelty::Known
        } else {
            Novelty::Novel
        };
        if ce.is_finite() {
            let n = record.ce_count as f32;
            record.ce_mean = (record.ce_mean * n + ce) / (n + 1.0);
            record.ce_count += 1;
        }
        novelty
    }

    /// Remember directories a successful run needed. Deduplicates.
    pub fn remember_dirs(&mut self, fp: &str, command_line: &str, dirs: &[PathBuf]) {
        if dirs.is_empty() {
            return;
        }
        let record = self
            .data
            .programs
            .entry(fp.to_string())
            .or_insert_with(|| ProgramRecord {
                command: command_line.to_string(),
                ..ProgramRecord::default()
            });
        for d in dirs {
            if !record.dirs.contains(d) {
                record.dirs.push(d.clone());
            }
        }
    }

    /// Persist to disk (pretty JSON so the memory stays human-auditable).
    pub fn save(&self) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.data)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        std::fs::write(&self.path, json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("axiom_healmem_{tag}_{nanos}.json"))
    }

    #[test]
    fn fingerprint_is_stable_and_arg_sensitive() {
        let a = fingerprint("sh", &["-c".into(), "x".into()]);
        let b = fingerprint("sh", &["-c".into(), "x".into()]);
        let c = fingerprint("sh", &["-c".into(), "y".into()]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn remember_save_load_immunize_roundtrip() {
        let mem_path = tmp("roundtrip");
        let dir = std::env::temp_dir().join(tmp("dir").file_stem().unwrap().to_os_string());
        let fp = fingerprint("prog", &[]);

        let mut mem = HealMemory::load(&mem_path);
        mem.remember_dirs(&fp, "prog", &[dir.clone()]);
        mem.save().unwrap();

        // Fresh load in a "fresh environment" (dir does not exist yet).
        assert!(!dir.exists());
        let mem2 = HealMemory::load(&mem_path);
        let applied = mem2.immunize(&fp);
        assert_eq!(applied, vec![dir.clone()]);
        assert!(dir.exists(), "immunization must re-create the remembered dir");
        // Second immunization is a no-op (already present).
        assert!(mem2.immunize(&fp).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&mem_path);
    }

    #[test]
    fn novelty_classification_tracks_the_mean() {
        let mut mem = HealMemory::load(tmp("novelty"));
        let fp = fingerprint("prog", &[]);
        assert_eq!(mem.observe_failure(&fp, "prog", 5.0), Novelty::First);
        assert_eq!(mem.observe_failure(&fp, "prog", 5.01), Novelty::Known);
        assert_eq!(mem.observe_failure(&fp, "prog", 9.0), Novelty::Novel);
    }

    #[test]
    fn corrupt_memory_file_degrades_to_empty() {
        let path = tmp("corrupt");
        std::fs::write(&path, "{not json").unwrap();
        let mem = HealMemory::load(&path);
        assert!(mem.record("anything").is_none());
        let _ = std::fs::remove_file(&path);
    }
}
