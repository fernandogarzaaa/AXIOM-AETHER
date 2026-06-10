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

/// Current unix time in seconds (0 if the clock is before the epoch).
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// What the memory knows about one supervised program.
///
/// The confidence fields model an adaptive-immunity lifecycle: a freshly
/// learned heal is *tentative*; each time immunizing the program produces a
/// successful run the confidence matures toward 1.0 (affinity maturation);
/// and confidence wanes with time since last reinforcement (memory waning),
/// so heals that are never exercised again fade and can be pruned.
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
    /// Stored (pre-decay) immunity confidence in 0..1. Defaulted for records
    /// written before the confidence lifecycle existed.
    #[serde(default)]
    pub confidence: f32,
    /// Times immunizing this program preceded a successful run.
    #[serde(default)]
    pub immunizations: u32,
    /// Unix seconds of the last confidence reinforcement (for waning).
    #[serde(default)]
    pub last_reinforced: u64,
}

/// Half-life (days) of unreinforced immunity confidence.
const CONFIDENCE_HALFLIFE_DAYS: f32 = 30.0;
/// Confidence a heal starts at when first learned (tentative).
const INITIAL_CONFIDENCE: f32 = 0.5;
/// EMA rate at which successful reuse matures confidence toward 1.0.
const MATURATION_RATE: f32 = 0.34;
/// Decayed-confidence floor below which `prune_stale` forgets a record.
const PRUNE_FLOOR: f32 = 0.05;

impl ProgramRecord {
    /// Confidence after time-decay relative to `now` (unix secs). Heals never
    /// reinforced (`last_reinforced == 0`) report their stored confidence.
    pub fn confidence_now(&self, now: u64) -> f32 {
        if self.last_reinforced == 0 || self.confidence <= 0.0 {
            return self.confidence;
        }
        let age_days = now.saturating_sub(self.last_reinforced) as f32 / 86_400.0;
        self.confidence * 0.5f32.powf(age_days / CONFIDENCE_HALFLIFE_DAYS)
    }

    /// A human label for a decayed-confidence value.
    pub fn confidence_label(c: f32) -> &'static str {
        if c >= 0.85 {
            "established"
        } else if c >= 0.6 {
            "proven"
        } else if c >= 0.3 {
            "tentative"
        } else {
            "faded"
        }
    }
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

/// Shell wrappers whose raw `-c` snippets are too generic to advise on.
const SHELL_WRAPPERS: [&str; 9] = [
    "sh", "bash", "zsh", "dash", "fish", "cmd", "cmd.exe", "powershell", "pwsh",
];

/// A lowercase, matchable signature for a command line — the program name plus
/// its first non-flag argument (e.g. "cargo build", "pytest", "npm test").
/// Returns `None` for shell wrappers and trivially short signatures, so
/// advisory matching stays precise (no false positives on prose).
pub fn command_signature(command_line: &str) -> Option<String> {
    let tokens: Vec<&str> = command_line.split_whitespace().collect();
    let first = tokens.first()?;
    if SHELL_WRAPPERS.contains(&first.to_ascii_lowercase().as_str()) {
        return None;
    }
    // program + first argument that isn't a flag or a redirection/path.
    let mut sig: Vec<String> = vec![first.to_ascii_lowercase()];
    if let Some(arg) = tokens.iter().skip(1).find(|t| {
        !t.starts_with('-') && !t.starts_with('/') && !t.contains('>') && !t.contains('<')
    }) {
        sig.push(arg.to_ascii_lowercase());
    }
    let joined = sig.join(" ");
    (joined.len() >= 3).then_some(joined)
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

/// Express `dir` relative to `anchor` when it lives under it (making the heal
/// portable across checkouts/machines); otherwise return `dir` unchanged so
/// out-of-tree absolute heals are preserved verbatim.
pub fn relativize_dir(dir: &Path, anchor: &Path) -> PathBuf {
    match dir.strip_prefix(anchor) {
        Ok(rel) if rel.as_os_str().is_empty() => PathBuf::from("."),
        Ok(rel) => rel.to_path_buf(),
        Err(_) => dir.to_path_buf(),
    }
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

    /// Every program this memory knows about, newest-insertion-order-agnostic.
    pub fn all_records(&self) -> Vec<&ProgramRecord> {
        self.data.programs.values().collect()
    }

    /// Records whose command line contains `query` (case-insensitive). An empty
    /// query matches everything.
    pub fn find(&self, query: &str) -> Vec<&ProgramRecord> {
        let q = query.trim().to_ascii_lowercase();
        self.data
            .programs
            .values()
            .filter(|r| q.is_empty() || r.command.to_ascii_lowercase().contains(&q))
            .collect()
    }

    /// Human- and agent-readable summary of acquired immunity. `query` filters
    /// by command substring; `None`/empty lists everything.
    pub fn report_text(&self, query: Option<&str>) -> String {
        let mut records = self.find(query.unwrap_or(""));
        if records.is_empty() {
            return match query {
                Some(q) if !q.trim().is_empty() => {
                    format!("Axiom has no acquired immunity matching \"{q}\".")
                }
                _ => "Axiom has not learned any program failures yet.".to_string(),
            };
        }
        // Stable output: most-experienced programs first.
        records.sort_by(|a, b| b.ce_count.cmp(&a.ce_count).then(a.command.cmp(&b.command)));
        let now = now_secs();
        let mut out = format!("Acquired immunity ({} program(s)):\n", records.len());
        for r in records {
            let c = r.confidence_now(now);
            out.push_str(&format!("\n• {}\n", r.command));
            out.push_str(&format!(
                "    confidence: {:.2} ({}, immunizations: {})   failures observed: {}   mean tension (CE): {:.3}\n",
                c,
                ProgramRecord::confidence_label(c),
                r.immunizations,
                r.ce_count,
                r.ce_mean
            ));
            if r.dirs.is_empty() {
                out.push_str("    learned heals: none (no directory heals)\n");
            } else {
                out.push_str("    learned heals: pre-create directories\n");
                for d in &r.dirs {
                    out.push_str(&format!("      - {}\n", d.display()));
                }
            }
        }
        out
    }

    /// Advisory lines for any learned program whose command is referenced in
    /// `text`. Only programs with concrete heals (directory pre-creation) and a
    /// non-generic command signature qualify — so this never fires on bare
    /// shell snippets, and a match means Axiom genuinely knows a fix.
    pub fn advisories_for_text(&self, text: &str) -> Vec<String> {
        let haystack = text.to_ascii_lowercase();
        let mut out = Vec::new();
        let mut records: Vec<&ProgramRecord> = self
            .data
            .programs
            .values()
            .filter(|r| !r.dirs.is_empty())
            .collect();
        records.sort_by(|a, b| a.command.cmp(&b.command));
        for r in records {
            let Some(sig) = command_signature(&r.command) else {
                continue;
            };
            if !haystack.contains(&sig) {
                continue;
            }
            let dirs = r
                .dirs
                .iter()
                .map(|d| d.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let c = r.confidence_now(now_secs());
            // Phrase by matured confidence: an established fix is asserted, a
            // tentative one is offered as a possibility.
            let preamble = if c >= 0.6 {
                format!(
                    "`{}` reliably fails in this environment (fixed {} time(s)); Axiom's established fix",
                    r.command, r.immunizations
                )
            } else {
                format!(
                    "`{}` has failed in this environment before; Axiom's tentative fix",
                    r.command
                )
            };
            out.push(format!(
                "{preamble}: create director{} {}. Apply preemptively if it fails again.",
                if r.dirs.len() == 1 { "y" } else { "ies" },
                dirs
            ));
        }
        out
    }

    /// Resolve the default heal-memory path used across the engine:
    /// `AXIOM_HEAL_MEMORY` overrides it; `0`/`off` disables it (returns `None`);
    /// otherwise `~/.axiom/heal_memory.json`.
    pub fn default_path() -> Option<PathBuf> {
        match std::env::var("AXIOM_HEAL_MEMORY") {
            Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") => None,
            Ok(v) => Some(PathBuf::from(v)),
            Err(_) => dirs::home_dir().map(|h| h.join(".axiom").join("heal_memory.json")),
        }
    }

    /// Re-create every remembered directory that is missing. Returns the dirs
    /// actually created now (the immunization applied to *this* environment).
    pub fn immunize(&self, fp: &str) -> Vec<PathBuf> {
        let anchor = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        self.immunize_anchored(fp, &anchor)
    }

    /// Re-create remembered-but-missing directories, re-anchoring *relative*
    /// heals to `anchor` (the supervised process's working directory). This is
    /// what makes immunity location-invariant: a heal learned as `target` in
    /// one checkout is applied as `<anchor>/target` here. Absolute heals are
    /// used verbatim. Returns the resolved absolute paths actually created.
    pub fn immunize_anchored(&self, fp: &str, anchor: &Path) -> Vec<PathBuf> {
        let Some(record) = self.data.programs.get(fp) else {
            return Vec::new();
        };
        let mut applied = Vec::new();
        for dir in &record.dirs {
            let resolved = if dir.is_relative() {
                anchor.join(dir)
            } else {
                dir.clone()
            };
            if !resolved.exists() && std::fs::create_dir_all(&resolved).is_ok() {
                applied.push(resolved);
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
        // A freshly learned heal is tentative until reuse proves it.
        if record.confidence <= 0.0 {
            record.confidence = INITIAL_CONFIDENCE;
        }
    }

    /// Reinforce a program's immunity after immunizing it preceded a successful
    /// run — affinity maturation: confidence EMAs toward 1.0 and the waning
    /// clock resets. Pre-decays the stored confidence to `now` first so repeated
    /// reinforcement after long gaps doesn't over-credit a faded memory.
    pub fn reinforce_immunity(&mut self, fp: &str, now: u64) {
        if let Some(r) = self.data.programs.get_mut(fp) {
            let decayed = r.confidence_now(now).max(INITIAL_CONFIDENCE * 0.5);
            r.confidence = decayed + MATURATION_RATE * (1.0 - decayed);
            r.immunizations += 1;
            r.last_reinforced = now;
        }
    }

    /// Forget records whose decayed confidence has fallen below the prune floor
    /// (memory waning → clonal deletion). Returns the number forgotten. Heals
    /// never reinforced are kept (they have not had a chance to mature or fade).
    pub fn prune_stale(&mut self, now: u64) -> usize {
        let before = self.data.programs.len();
        self.data
            .programs
            .retain(|_, r| r.last_reinforced == 0 || r.confidence_now(now) >= PRUNE_FLOOR);
        before - self.data.programs.len()
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
                    // Combine fleet immunity experience: confidence takes the
                    // stronger of the two, immunizations sum, and the waning
                    // clock advances to the most recent reinforcement.
                    ours.confidence = ours.confidence.max(theirs.confidence);
                    ours.immunizations += theirs.immunizations;
                    ours.last_reinforced = ours.last_reinforced.max(theirs.last_reinforced);
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
    fn report_text_summarizes_and_filters() {
        let mut mem = HealMemory::load(tmp("report"));
        let fp_cargo = fingerprint("cargo", &["build".into()]);
        mem.remember_dirs(&fp_cargo, "cargo build", &[PathBuf::from("/target")]);
        mem.observe_failure(&fp_cargo, "cargo build", 4.0);
        let fp_py = fingerprint("python", &["app.py".into()]);
        mem.observe_failure(&fp_py, "python app.py", 6.0);

        let all = mem.report_text(None);
        assert!(all.contains("cargo build") && all.contains("python app.py"));
        assert!(all.contains("/target"), "remembered heals must be listed");

        let filtered = mem.report_text(Some("cargo"));
        assert!(filtered.contains("cargo build"));
        assert!(!filtered.contains("python app.py"), "filter must exclude non-matches");

        let miss = mem.report_text(Some("rustc"));
        assert!(miss.contains("no acquired immunity matching"));
    }

    #[test]
    fn report_text_empty_memory_is_friendly() {
        let mem = HealMemory::load(tmp("empty_report"));
        assert!(mem.report_text(None).contains("has not learned any program failures"));
    }

    #[test]
    fn command_signature_skips_shells_and_keeps_real_tools() {
        assert_eq!(command_signature("cargo build --release").as_deref(), Some("cargo build"));
        assert_eq!(command_signature("pytest").as_deref(), Some("pytest"));
        assert_eq!(command_signature("npm test").as_deref(), Some("npm test"));
        // shell wrappers and their snippets are too generic to advise on
        assert_eq!(command_signature("sh -c echo hi > /x/y"), None);
        assert_eq!(command_signature("bash -c make"), None);
    }

    #[test]
    fn advisories_fire_only_on_referenced_healable_commands() {
        let mut mem = HealMemory::load(tmp("advis"));
        let fp = fingerprint("cargo", &["build".into()]);
        mem.remember_dirs(&fp, "cargo build", &[PathBuf::from("/repo/target")]);
        // A program with no learned dirs must never produce an advisory.
        let fp2 = fingerprint("pytest", &[]);
        mem.observe_failure(&fp2, "pytest", 5.0);

        let hit = mem.advisories_for_text("why does cargo build keep failing in CI?");
        assert_eq!(hit.len(), 1);
        assert!(hit[0].contains("cargo build") && hit[0].contains("/repo/target"));

        // pytest is referenced but has no learned heal → no advisory.
        assert!(mem.advisories_for_text("my pytest run is slow").is_empty());
        // cargo not mentioned → no advisory.
        assert!(mem.advisories_for_text("unrelated question about npm").is_empty());
    }

    #[test]
    fn confidence_matures_with_reuse_and_wanes_with_time() {
        let mut mem = HealMemory::load(tmp("conf"));
        let fp = fingerprint("cargo", &["build".into()]);
        let now = 1_000_000_000u64;

        // Freshly learned → tentative.
        mem.remember_dirs(&fp, "cargo build", &[PathBuf::from("target")]);
        assert!((mem.record(&fp).unwrap().confidence - 0.5).abs() < 1e-6);

        // Each successful reuse matures confidence toward 1.0 (monotone up).
        let mut prev = mem.record(&fp).unwrap().confidence_now(now);
        for k in 1..=3 {
            mem.reinforce_immunity(&fp, now + k);
            let c = mem.record(&fp).unwrap().confidence_now(now + k);
            assert!(c > prev, "reuse {k} must raise confidence ({prev} -> {c})");
            prev = c;
        }
        assert_eq!(mem.record(&fp).unwrap().immunizations, 3);
        assert!(prev >= 0.6, "after 3 reuses confidence should be 'proven'+");

        // Waning: a full half-life later, confidence ~halves.
        let later = now + 3 + (CONFIDENCE_HALFLIFE_DAYS as u64) * 86_400;
        let decayed = mem.record(&fp).unwrap().confidence_now(later);
        assert!(decayed < prev * 0.6, "confidence must wane over a half-life");
    }

    #[test]
    fn prune_forgets_only_faded_reinforced_heals() {
        let mut mem = HealMemory::load(tmp("prune"));
        let now = 2_000_000_000u64;

        // A reinforced-then-ancient heal → should be pruned.
        let faded = fingerprint("old", &[]);
        mem.remember_dirs(&faded, "old", &[PathBuf::from("a")]);
        mem.reinforce_immunity(&faded, now - 3650 * 86_400); // ~10 years ago

        // A never-reinforced heal → kept (hasn't had a chance to mature/fade).
        let fresh = fingerprint("new", &[]);
        mem.remember_dirs(&fresh, "new", &[PathBuf::from("b")]);

        let pruned = mem.prune_stale(now);
        assert_eq!(pruned, 1);
        assert!(mem.record(&faded).is_none(), "ancient heal forgotten");
        assert!(mem.record(&fresh).is_some(), "unreinforced heal retained");
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
