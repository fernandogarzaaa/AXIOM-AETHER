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

/// What a swarm-immunity merge changed locally.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MergeReport {
    /// Programs we had never seen, adopted from the peer.
    pub programs_added: usize,
    /// Programs both sides knew, whose records were combined.
    pub programs_merged: usize,
    /// New remembered directories gained from the peer.
    pub dirs_added: usize,
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

    /// Serialize the whole memory for transfer to a swarm peer.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.data).unwrap_or_else(|_| "{}".into())
    }

    /// Merge a peer's exported memory into this one (swarm immunity).
    ///
    /// Semantics per program fingerprint: directory lists are unioned, and the
    /// failure-tension history is combined as a count-weighted mean — so a peer
    /// with 100 observed failures outweighs one with 2. Unknown programs are
    /// adopted wholesale. A malformed peer payload is rejected without touching
    /// local state.
    pub fn merge_json(&mut self, peer_json: &str) -> Result<MergeReport, String> {
        let peer: MemoryFile =
            serde_json::from_str(peer_json).map_err(|e| format!("invalid peer memory: {e}"))?;
        let mut report = MergeReport::default();
        for (fp, theirs) in peer.programs {
            match self.data.programs.get_mut(&fp) {
                None => {
                    report.programs_added += 1;
                    report.dirs_added += theirs.dirs.len();
                    self.data.programs.insert(fp, theirs);
                }
                Some(ours) => {
                    report.programs_merged += 1;
                    for d in theirs.dirs {
                        if !ours.dirs.contains(&d) {
                            ours.dirs.push(d);
                            report.dirs_added += 1;
                        }
                    }
                    let total = ours.ce_count + theirs.ce_count;
                    if total > 0 {
                        ours.ce_mean = (ours.ce_mean * ours.ce_count as f32
                            + theirs.ce_mean * theirs.ce_count as f32)
                            / total as f32;
                        ours.ce_count = total;
                    }
                }
            }
        }
        Ok(report)
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
    fn merge_adopts_unknown_and_unions_known_programs() {
        let fp_shared = fingerprint("shared", &[]);
        let fp_peer_only = fingerprint("peer-only", &[]);

        let mut ours = HealMemory::load(tmp("merge_ours"));
        ours.remember_dirs(&fp_shared, "shared", &[PathBuf::from("/a")]);
        ours.observe_failure(&fp_shared, "shared", 4.0); // count=1, mean=4.0

        let mut theirs = HealMemory::load(tmp("merge_theirs"));
        theirs.remember_dirs(&fp_shared, "shared", &[PathBuf::from("/a"), PathBuf::from("/b")]);
        // Three failures at 6.0 → their history outweighs ours 3:1.
        for _ in 0..3 {
            theirs.observe_failure(&fp_shared, "shared", 6.0);
        }
        theirs.remember_dirs(&fp_peer_only, "peer-only", &[PathBuf::from("/c")]);

        let report = ours.merge_json(&theirs.to_json()).unwrap();
        assert_eq!(report.programs_added, 1);
        assert_eq!(report.programs_merged, 1);
        assert_eq!(report.dirs_added, 2, "/b from shared + /c from peer-only");

        let shared = ours.record(&fp_shared).unwrap();
        assert_eq!(shared.dirs, vec![PathBuf::from("/a"), PathBuf::from("/b")]);
        assert_eq!(shared.ce_count, 4);
        assert!((shared.ce_mean - 5.5).abs() < 1e-5, "count-weighted: (4+18)/4");
        assert!(ours.record(&fp_peer_only).is_some());
    }

    #[test]
    fn merge_rejects_malformed_peer_payload() {
        let fp = fingerprint("prog", &[]);
        let mut ours = HealMemory::load(tmp("merge_bad"));
        ours.remember_dirs(&fp, "prog", &[PathBuf::from("/keep")]);
        assert!(ours.merge_json("{broken").is_err());
        // Local state untouched.
        assert_eq!(ours.record(&fp).unwrap().dirs, vec![PathBuf::from("/keep")]);
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
