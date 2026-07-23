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

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::belief::BetaBelief;

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
    /// Environment variables this program reported as required-but-unset. A
    /// value cannot be safely fabricated, so these are surfaced as diagnostics
    /// (in advisories / `axiom immunity`) rather than auto-applied.
    #[serde(default)]
    pub required_env: Vec<String>,
    /// Running mean of failure-trace cross-entropy.
    pub ce_mean: f32,
    /// Number of failures folded into `ce_mean`.
    pub ce_count: u32,
    /// Back-compat scalar mirror of `belief.mean()` (kept so older readers and
    /// JSON inspection still see a confidence number). The Beta belief is the
    /// source of truth.
    #[serde(default)]
    pub confidence: f32,
    /// Times immunizing this program preceded a successful run.
    #[serde(default)]
    pub immunizations: u32,
    /// Unix seconds of the last confidence reinforcement (for waning).
    #[serde(default)]
    pub last_reinforced: u64,
    /// Beta-distribution belief (estimate + uncertainty) — the source of truth
    /// for confidence. `None` on legacy records; migrated from the scalar on
    /// first access.
    #[serde(default)]
    pub belief: Option<BetaBelief>,
}

/// Half-life (days) of unreinforced immunity confidence.
const CONFIDENCE_HALFLIFE_DAYS: f32 = 30.0;
/// A belief whose time-decay has pushed its variance to/above this (near the
/// uniform prior's 0.083) has lost its evidence and is forgettable.
const FADED_VARIANCE: f32 = 0.07;
/// Cap on a peer-reported belief's total pseudocount mass — beyond this it is
/// treated as fabricated certainty (byzantine) and rejected from a merge.
const MAX_PEER_EVIDENCE: f32 = 1.0e4;

/// Trust applied to peer-reported beliefs before Dempster-Shafer fusion.
/// 1.0 = full trust (plain combine_ds); lower values shrink peer evidence
/// toward the uniform prior, bounding how hard a single gossiping peer can
/// move local belief. Overridable via `AXIOM_PEER_RELIABILITY`.
const PEER_RELIABILITY: f32 = 0.8;

/// Effective peer reliability: `AXIOM_PEER_RELIABILITY` env (clamped to
/// [0, 1]) or the `PEER_RELIABILITY` default.
fn peer_reliability() -> f32 {
    std::env::var("AXIOM_PEER_RELIABILITY")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(PEER_RELIABILITY)
}

impl ProgramRecord {
    /// The stored Beta belief, migrating a legacy scalar-confidence record into
    /// a Beta on first access (evidence ≈ the immunization count).
    fn base_belief(&self) -> BetaBelief {
        self.belief.unwrap_or_else(|| {
            if self.confidence > 0.0 {
                BetaBelief::from_confidence(self.confidence, (self.immunizations + 2) as f32)
            } else {
                BetaBelief::uniform()
            }
        })
    }

    /// Time-decay factor in [0,1] since the last reinforcement (0 = none).
    fn decay_factor(&self, now: u64) -> f32 {
        if self.last_reinforced == 0 {
            return 0.0;
        }
        let age_days = now.saturating_sub(self.last_reinforced) as f32 / 86_400.0;
        1.0 - 0.5f32.powf(age_days / CONFIDENCE_HALFLIFE_DAYS)
    }

    /// The belief regressed toward the uniform prior by its age (waning).
    pub fn belief_now(&self, now: u64) -> BetaBelief {
        self.base_belief().decayed(self.decay_factor(now))
    }

    /// Decayed mean — the scalar confidence, for callers that want a number.
    pub fn confidence_now(&self, now: u64) -> f32 {
        self.belief_now(now).mean()
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

/// Distinct programs that must independently have needed the same directory
/// before it counts as a *structural* heal — evidence the gap is in the
/// environment itself (nothing provisions this path) rather than something
/// specific to one program's interaction with it. Two is deliberately the
/// smallest threshold that is still corroboration rather than coincidence: a
/// single program's own heal stays exactly what it always was (scoped to that
/// program's fingerprint, per `remember_dirs`); this only fires once a second,
/// unrelated program independently hits the same missing path.
const STRUCTURAL_COROBORATION_THRESHOLD: usize = 2;

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
    /// Programs whose belief was irreconcilable under Dempster-Shafer (peer and
    /// local strongly disagreed) — the local belief was kept, peer's rejected.
    pub belief_conflicts: usize,
    /// Peer records rejected as byzantine (implausible belief parameters).
    pub byzantine_rejected: usize,
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

/// Inverse of [`relativize_dir`]: resolve a stored heal against `anchor`. A
/// relative heal is anchored here (portable immunity); an absolute heal is used
/// verbatim.
pub fn resolve_dir(dir: &Path, anchor: &Path) -> PathBuf {
    if dir.is_relative() {
        anchor.join(dir)
    } else {
        dir.to_path_buf()
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
            let belief = r.belief_now(now);
            out.push_str(&format!("\n• {}\n", r.command));
            out.push_str(&format!(
                "    confidence: {:.2} ±{:.2} ({}, immunizations: {})   failures observed: {}   mean tension (CE): {:.3}\n",
                belief.mean(),
                belief.variance().sqrt(),
                belief.label(),
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
            if !r.required_env.is_empty() {
                out.push_str(&format!(
                    "    requires env vars (set before running): {}\n",
                    r.required_env.join(", ")
                ));
            }
        }
        // Cross-program generalization only makes sense on the unfiltered,
        // whole-environment view — a query is asking about one program, and
        // this section is deliberately not about any one program.
        if query.map(str::trim).unwrap_or("").is_empty() {
            let structural = self.structural_dirs();
            if !structural.is_empty() {
                out.push_str(&format!(
                    "\nStructural (environment-wide) heals ({} director{}, corroborated \
                     across independent programs — applied even to programs Axiom has \
                     never run before):\n",
                    structural.len(),
                    if structural.len() == 1 { "y" } else { "ies" }
                ));
                for (dir, progs) in &structural {
                    out.push_str(&format!(
                        "\n• {}\n    corroborated by {} program(s): {}\n",
                        dir.display(),
                        progs.len(),
                        progs.join(", ")
                    ));
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
            .filter(|r| !r.dirs.is_empty() || !r.required_env.is_empty())
            .collect();
        records.sort_by(|a, b| a.command.cmp(&b.command));
        for r in records {
            let Some(sig) = command_signature(&r.command) else {
                continue;
            };
            if !haystack.contains(&sig) {
                continue;
            }
            if !r.dirs.is_empty() {
                let dirs = r
                    .dirs
                    .iter()
                    .map(|d| d.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                // Phrase by the Beta belief: an *established* fix (high mean AND
                // low variance — genuinely settled evidence) is asserted; a
                // high-mean-but-uncertain or low-mean fix is merely offered.
                let preamble = if r.belief_now(now_secs()).is_established() {
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
            if !r.required_env.is_empty() {
                out.push(format!(
                    "`{}` requires environment variable(s) {} to be set — Axiom cannot \
                     fabricate the value(s); ensure they are exported.",
                    r.command,
                    r.required_env.join(", ")
                ));
            }
        }
        out
    }

    /// Cross-reactive immunity: analogical hints from *sibling* commands in the
    /// same program family (same executable, different sub-command) that carry
    /// learned heals. Fires when the program name is referenced in `text` but
    /// the sibling's exact signature is not — generalizing immunity by analogy.
    /// These are advisory ONLY (never auto-applied), so a wrong analogy costs
    /// nothing. Capped to keep proxy payloads lean.
    pub fn cross_reactive_hints(&self, text: &str) -> Vec<String> {
        let hay = text.to_ascii_lowercase();
        let mut out = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
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
            // Cross-reactivity is for multi-sub-command families ("cargo build"),
            // not single-command tools, and the exact command must NOT already be
            // referenced (that path is a direct advisory, not an analogy).
            let mut parts = sig.splitn(2, ' ');
            let prog = parts.next().unwrap_or("");
            if parts.next().is_none() || prog.len() < 3 || hay.contains(&sig) {
                continue;
            }
            if !hay.contains(prog) || !seen.insert(format!("{prog}:{:?}", r.dirs)) {
                continue;
            }
            let dirs = r
                .dirs
                .iter()
                .map(|d| d.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            out.push(format!(
                "cross-reactive hint: a sibling `{}` previously needed director{} {}; \
                 a different `{prog} …` invocation here may need the same (Axiom has \
                 not applied it).",
                r.command,
                if r.dirs.len() == 1 { "y" } else { "ies" },
                dirs
            ));
            if out.len() >= 3 {
                break;
            }
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
            let resolved = resolve_dir(dir, anchor);
            if !resolved.exists() && std::fs::create_dir_all(&resolved).is_ok() {
                applied.push(resolved);
            }
        }
        applied
    }

    /// Anticipatory immunity: the program's learned prerequisite directories
    /// that are *currently missing* (re-anchored). A non-empty result predicts
    /// the program would fail right now for want of those directories — before
    /// it is ever run. Returns `(resolved_missing_dirs, confidence)`; an unknown
    /// program yields an empty list.
    pub fn missing_prerequisites(&self, fp: &str, anchor: &Path) -> (Vec<PathBuf>, f32) {
        let Some(record) = self.data.programs.get(fp) else {
            return (Vec::new(), 0.0);
        };
        let missing = record
            .dirs
            .iter()
            .map(|d| resolve_dir(d, anchor))
            .filter(|d| !d.exists())
            .collect();
        (missing, record.confidence_now(now_secs()))
    }

    /// Directories independently learned as needed by at least
    /// [`STRUCTURAL_COROBORATION_THRESHOLD`] distinct programs, with the
    /// corroborating command lines. Derived entirely from the existing
    /// per-program records (no separate persisted state to keep in sync) —
    /// grouping is by exact stored path, matching the equality `remember_dirs`
    /// itself already uses to dedupe a single program's own repeated heals.
    pub fn structural_dirs(&self) -> Vec<(PathBuf, Vec<&str>)> {
        let mut by_dir: HashMap<&Path, Vec<&str>> = HashMap::new();
        for record in self.data.programs.values() {
            for d in &record.dirs {
                by_dir.entry(d.as_path()).or_default().push(record.command.as_str());
            }
        }
        let mut out: Vec<(PathBuf, Vec<&str>)> = by_dir
            .into_iter()
            .filter(|(_, progs)| progs.len() >= STRUCTURAL_COROBORATION_THRESHOLD)
            .map(|(d, mut progs)| {
                progs.sort_unstable();
                progs.dedup();
                (d.to_path_buf(), progs)
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Structural directories currently missing for `fp` (re-anchored), and the
    /// mean matured confidence of their corroborating programs — the same
    /// anticipatory shape as [`Self::missing_prerequisites`], but generalized
    /// across programs instead of scoped to one. `fp`'s own already-remembered
    /// dirs are excluded so a program doesn't get double-counted against its
    /// own heal.
    pub fn missing_structural_prerequisites(&self, fp: &str, anchor: &Path) -> (Vec<PathBuf>, f32) {
        let own_dirs: HashSet<&Path> = self
            .data
            .programs
            .get(fp)
            .map(|r| r.dirs.iter().map(|d| d.as_path()).collect())
            .unwrap_or_default();
        let now = now_secs();
        let mut missing = Vec::new();
        let mut confidences = Vec::new();
        for (dir, corroborators) in self.structural_dirs() {
            if own_dirs.contains(dir.as_path()) {
                continue;
            }
            let resolved = resolve_dir(&dir, anchor);
            if resolved.exists() {
                continue;
            }
            // The structural claim is only as credible as the programs
            // corroborating it — averaged rather than taking this program's
            // own (nonexistent) confidence.
            let mean_conf: f32 = self
                .data
                .programs
                .values()
                .filter(|r| corroborators.contains(&r.command.as_str()))
                .map(|r| r.confidence_now(now))
                .sum::<f32>()
                / corroborators.len().max(1) as f32;
            confidences.push(mean_conf);
            missing.push(resolved);
        }
        let confidence = if confidences.is_empty() {
            0.0
        } else {
            confidences.iter().sum::<f32>() / confidences.len() as f32
        };
        (missing, confidence)
    }

    /// Pre-create structurally corroborated directories that are currently
    /// missing for `fp` — the generalized counterpart to
    /// [`Self::immunize_anchored`]: it applies even when `fp` has never been
    /// seen before, because the evidence is that the environment itself is
    /// missing the path, not that this specific program needs it. Returns the
    /// resolved paths actually created, with how many distinct programs
    /// corroborate each one.
    pub fn immunize_structural(&self, fp: &str, anchor: &Path) -> Vec<(PathBuf, usize)> {
        let own_dirs: HashSet<&Path> = self
            .data
            .programs
            .get(fp)
            .map(|r| r.dirs.iter().map(|d| d.as_path()).collect())
            .unwrap_or_default();
        let mut applied = Vec::new();
        for (dir, corroborators) in self.structural_dirs() {
            if own_dirs.contains(dir.as_path()) {
                continue;
            }
            let resolved = resolve_dir(&dir, anchor);
            if !resolved.exists() && std::fs::create_dir_all(&resolved).is_ok() {
                applied.push((resolved, corroborators.len()));
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
        // A freshly learned heal is tentative: the uniform prior Beta(1,1) —
        // mean 0.5, maximum uncertainty — until reuse accumulates evidence.
        if record.belief.is_none() {
            let b = BetaBelief::uniform();
            record.confidence = b.mean();
            record.belief = Some(b);
        }
    }

    /// Record environment variables a program reported as required-but-unset, so
    /// they can be surfaced as diagnostics. Deduplicates; creates the record if
    /// this is the program's first observation.
    pub fn remember_env_requirement(&mut self, fp: &str, command_line: &str, vars: &[String]) {
        if vars.is_empty() {
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
        for v in vars {
            if !record.required_env.contains(v) {
                record.required_env.push(v.clone());
            }
        }
    }

    /// Reinforce a program's immunity after immunizing it preceded a successful
    /// run — affinity maturation: add a success pseudocount to the Beta belief
    /// (raising the mean, lowering variance) and reset the waning clock. The
    /// belief is first decayed to `now` so reinforcement after a long gap builds
    /// on a faded (re-uncertain) memory rather than over-crediting it.
    pub fn reinforce_immunity(&mut self, fp: &str, now: u64) {
        if let Some(r) = self.data.programs.get_mut(fp) {
            let mut b = r.belief_now(now);
            b.reinforce();
            r.confidence = b.mean();
            r.belief = Some(b);
            r.immunizations += 1;
            r.last_reinforced = now;
        }
    }

    /// Forget records whose belief has decayed back to near-uniform uncertainty
    /// (its evidence lost to waning → clonal deletion). Returns the number
    /// forgotten. Never-reinforced heals are kept (they have not had a chance to
    /// mature or fade).
    pub fn prune_stale(&mut self, now: u64) -> usize {
        let before = self.data.programs.len();
        self.data
            .programs
            .retain(|_, r| r.last_reinforced == 0 || r.belief_now(now).variance() < FADED_VARIANCE);
        before - self.data.programs.len()
    }

    /// Serialize the whole memory for transfer to a swarm peer.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.data).unwrap_or_else(|_| "{}".into())
    }

    /// Merge a peer's exported memory into this one (swarm immunity), with
    /// epistemic + byzantine-resistant trust rules:
    ///
    /// * **Byzantine gate** — a peer record whose belief parameters are
    ///   implausible (NaN/∞/negative, or fabricated-certainty pseudocounts) is
    ///   rejected outright (`byzantine_rejected`).
    /// * **Dempster-Shafer combination** — for a program both sides know, the
    ///   two Beta beliefs are combined via DS evidence combination: agreeing
    ///   peers *compound* evidence (mean up, variance down); irreconcilable
    ///   peers raise a conflict, the local belief is kept, and the peer's is
    ///   rejected (`belief_conflicts`) rather than silently averaged.
    /// * Directory lists are unioned; failure-tension is count-weighted-mean.
    ///
    /// A malformed peer payload is rejected without touching local state.
    pub fn merge_json(&mut self, peer_json: &str) -> Result<MergeReport, String> {
        let peer: MemoryFile =
            serde_json::from_str(peer_json).map_err(|e| format!("invalid peer memory: {e}"))?;
        let mut report = MergeReport::default();
        let reliability = peer_reliability();
        for (fp, theirs) in peer.programs {
            // Byzantine gate: a peer cannot inject malformed/fabricated beliefs.
            if let Some(pb) = theirs.belief {
                if !pb.is_plausible(MAX_PEER_EVIDENCE) {
                    report.byzantine_rejected += 1;
                    continue;
                }
            }
            match self.data.programs.get_mut(&fp) {
                None => {
                    report.programs_added += 1;
                    report.dirs_added += theirs.dirs.len();
                    self.data.programs.insert(fp, theirs);
                }
                Some(ours) => {
                    report.programs_merged += 1;
                    for d in &theirs.dirs {
                        if !ours.dirs.contains(d) {
                            ours.dirs.push(d.clone());
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
                    // Reliability-discounted Dempster-Shafer combination.
                    // Peers are trusted at PEER_RELIABILITY (their evidence is
                    // shrunk toward the uniform prior before fusing); when the
                    // discounted beliefs still irreconcilably conflict, fall
                    // back to Murphy averaging instead of discarding the peer
                    // outright — dissent lowers confidence rather than being
                    // silently dropped.
                    let (combined, conflict) = ours
                        .base_belief()
                        .combine_ds_conflict_aware(&theirs.base_belief(), reliability);
                    if conflict.is_some() {
                        report.belief_conflicts += 1;
                    }
                    ours.confidence = combined.mean();
                    ours.belief = Some(combined);
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
    fn merge_flags_dempster_shafer_conflict_and_murphy_averages() {
        let fp = fingerprint("prog", &[]);
        let now = 1_000_000_000u64;
        // Ours: strongly-established belief (many successes).
        let mut ours = HealMemory::load(tmp("merge_conf_a"));
        ours.remember_dirs(&fp, "prog", &[PathBuf::from("/a")]);
        for k in 0..12 {
            ours.reinforce_immunity(&fp, now + k);
        }
        let our_mean = ours.record(&fp).unwrap().belief_now(now + 12).mean();

        // Theirs: a strongly-negative belief for the same program (penalized).
        let mut theirs = HealMemory::load(tmp("merge_conf_b"));
        theirs.remember_dirs(&fp, "prog", &[PathBuf::from("/a")]);
        if let Some(r) = theirs.data.programs.get_mut(&fp) {
            r.belief = Some(crate::belief::BetaBelief { alpha: 1.0, beta: 11.0 });
        }

        let report = ours.merge_json(&theirs.to_json()).unwrap();
        assert_eq!(report.belief_conflicts, 1, "irreconcilable beliefs flagged");
        // Conflict-aware fusion: the dissenting peer is Murphy-averaged in
        // (reliability-discounted), so confidence drops below our prior mean
        // instead of the peer being silently rejected — dissent is evidence.
        let fused_mean = ours.record(&fp).unwrap().belief_now(now + 12).mean();
        assert!(
            fused_mean < our_mean,
            "dissent must lower confidence: fused {fused_mean} vs local {our_mean}"
        );
        // But the discounted average must not capitulate to the peer's
        // strongly-negative view either — it stays above the peer's mean.
        let peer_mean = crate::belief::BetaBelief { alpha: 1.0, beta: 11.0 }.mean();
        assert!(
            fused_mean > peer_mean,
            "fusion must not adopt the peer outright: fused {fused_mean} vs peer {peer_mean}"
        );
    }

    #[test]
    fn merge_rejects_byzantine_fabricated_certainty() {
        let fp = fingerprint("prog", &[]);
        let mut ours = HealMemory::load(tmp("merge_byz_a"));
        ours.remember_dirs(&fp, "prog", &[PathBuf::from("/a")]);

        // Peer injects a fabricated-certainty belief (absurd pseudocount mass).
        let mut theirs = HealMemory::load(tmp("merge_byz_b"));
        theirs.remember_dirs(&fp, "prog", &[PathBuf::from("/evil")]);
        if let Some(r) = theirs.data.programs.get_mut(&fp) {
            r.belief = Some(crate::belief::BetaBelief { alpha: 1.0e9, beta: 1.0 });
        }

        let report = ours.merge_json(&theirs.to_json()).unwrap();
        assert_eq!(report.byzantine_rejected, 1);
        // The peer's directory must NOT have been merged in.
        assert_eq!(ours.record(&fp).unwrap().dirs, vec![PathBuf::from("/a")]);
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
    fn cross_reactive_hints_generalize_within_a_program_family() {
        let mut mem = HealMemory::load(tmp("xreact"));
        mem.remember_dirs(
            &fingerprint("cargo", &["build".into()]),
            "cargo build",
            &[PathBuf::from("target")],
        );

        // A sibling sub-command (`cargo test`) referenced but never learned →
        // analogical hint from `cargo build`.
        let hints = mem.cross_reactive_hints("why does cargo test fail in CI?");
        assert_eq!(hints.len(), 1);
        assert!(hints[0].contains("cargo build") && hints[0].contains("target"));

        // The exact command directly referenced → NOT a cross-reactive hint
        // (that's a direct advisory's job).
        assert!(mem.cross_reactive_hints("cargo build broke").is_empty());

        // A different program family → no analogy.
        assert!(mem.cross_reactive_hints("npm test failing").is_empty());
    }

    #[test]
    fn env_requirement_surfaces_in_advisory_and_report() {
        let mut mem = HealMemory::load(tmp("envadv"));
        let fp = fingerprint("cargo", &["run".into()]);
        mem.remember_env_requirement(&fp, "cargo run", &["DATABASE_URL".into()]);

        let adv = mem.advisories_for_text("why does cargo run fail?");
        assert!(
            adv.iter().any(|a| a.contains("DATABASE_URL") && a.contains("environment variable")),
            "env requirement must surface as an advisory: {adv:?}"
        );
        assert!(mem.report_text(None).contains("requires env vars"));
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
        let rec = mem.record(&fp).unwrap();
        let fresh = rec.belief_now(now + 3);
        assert!(fresh.is_established(), "after 3 reuses the belief is established");

        // Waning: long after the last reuse the belief regresses toward the
        // uniform prior — staleness becomes *uncertainty* (higher variance,
        // mean drifting back toward 0.5), not a confidence driven to zero.
        let later = now + 3 + 4 * (CONFIDENCE_HALFLIFE_DAYS as u64) * 86_400;
        let stale = rec.belief_now(later);
        assert!(!stale.is_established(), "a waned belief is no longer established");
        assert!(stale.variance() > fresh.variance(), "waning raises uncertainty");
        assert!(stale.mean() < fresh.mean(), "mean regresses toward 0.5");
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

    #[test]
    fn structural_dirs_require_corroboration_from_distinct_programs() {
        let mut mem = HealMemory::load(tmp("structural_basic"));
        let fp_a = fingerprint("make", &[]);
        let fp_b = fingerprint("npm", &["run".into(), "build".into()]);

        // A single program's own heal is not structural, no matter how many
        // times it's reinforced.
        mem.remember_dirs(&fp_a, "make", &[PathBuf::from("dist")]);
        assert!(mem.structural_dirs().is_empty(), "one program alone must not generalize");

        // A second, unrelated program independently needing the same path is
        // corroboration: this is now a structural heal.
        mem.remember_dirs(&fp_b, "npm run build", &[PathBuf::from("dist")]);
        let structural = mem.structural_dirs();
        assert_eq!(structural.len(), 1);
        assert_eq!(structural[0].0, PathBuf::from("dist"));
        assert_eq!(structural[0].1, vec!["make", "npm run build"]);
    }

    #[test]
    fn immunize_structural_applies_to_a_program_never_seen_before() {
        let dir = std::env::temp_dir().join(tmp("structural_dir").file_stem().unwrap().to_os_string());
        let mut mem = HealMemory::load(tmp("structural_apply"));
        mem.remember_dirs(&fingerprint("make", &[]), "make", &[dir.clone()]);
        mem.remember_dirs(&fingerprint("npm", &["run".into()]), "npm run", &[dir.clone()]);

        // A third program, with zero history of its own, still gets the
        // structurally-corroborated directory pre-created.
        let unknown_fp = fingerprint("gradle", &["build".into()]);
        assert!(mem.record(&unknown_fp).is_none(), "must genuinely be unknown to this program");
        assert!(!dir.exists());
        let applied = mem.immunize_structural(&unknown_fp, &std::env::current_dir().unwrap());
        assert_eq!(applied, vec![(dir.clone(), 2)]);
        assert!(dir.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn immunize_structural_does_not_duplicate_a_programs_own_known_dir() {
        let mut mem = HealMemory::load(tmp("structural_no_dup"));
        let shared = PathBuf::from("shared_dir_xyz");
        mem.remember_dirs(&fingerprint("a", &[]), "a", &[shared.clone()]);
        mem.remember_dirs(&fingerprint("b", &[]), "b", &[shared.clone()]);

        // `a` already knows about `shared_dir_xyz` itself -- immunize_structural
        // must not re-report it as a *structural* heal for `a` (that would
        // double-count the same directory as two different kinds of evidence).
        let applied = mem.immunize_structural(&fingerprint("a", &[]), &std::env::temp_dir());
        assert!(applied.is_empty(), "a program's own dir must not double up as structural: {applied:?}");
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join(&shared));
    }

    #[test]
    fn missing_structural_prerequisites_predicts_for_an_unknown_program() {
        let anchor = std::env::temp_dir().join(tmp("structural_predict_anchor").file_stem().unwrap().to_os_string());
        std::fs::create_dir_all(&anchor).unwrap();
        let mut mem = HealMemory::load(tmp("structural_predict"));
        mem.remember_dirs(&fingerprint("a", &[]), "a", &[PathBuf::from("needed")]);
        mem.remember_dirs(&fingerprint("b", &[]), "b", &[PathBuf::from("needed")]);

        let unknown_fp = fingerprint("c", &[]);
        let (missing, confidence) = mem.missing_structural_prerequisites(&unknown_fp, &anchor);
        assert_eq!(missing, vec![anchor.join("needed")]);
        assert!(confidence >= 0.0, "confidence must be a valid, non-negative estimate");

        let _ = std::fs::remove_dir_all(&anchor);
    }

    #[test]
    fn report_text_shows_structural_heals_only_on_the_unfiltered_view() {
        let mut mem = HealMemory::load(tmp("structural_report"));
        mem.remember_dirs(&fingerprint("make", &[]), "make", &[PathBuf::from("dist")]);
        mem.remember_dirs(&fingerprint("npm", &["run".into()]), "npm run", &[PathBuf::from("dist")]);

        let all = mem.report_text(None);
        assert!(all.contains("Structural (environment-wide) heals"));
        assert!(all.contains("corroborated by 2 program(s): make, npm run"));

        // A query for one specific program must not surface the cross-cutting
        // section -- it isn't about that one program.
        let filtered = mem.report_text(Some("make"));
        assert!(!filtered.contains("Structural (environment-wide) heals"));
    }
}
