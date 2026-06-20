//! AXIOM Agentic Core — the autonomy capstone.
//!
//! The `solve` loop (Pillar 3) already repairs a failing check: localize the
//! faulty file(s) across languages, propose a fix (deterministic Poly-JIT,
//! fleet-shared patch, or LLM), and keep it **only if the verifier goes green**.
//! This module generalizes that from *repair* to *goal-directed coding* and adds
//! the machinery a self-improving coding agent needs:
//!
//! 1. [`Transaction`] — all-or-nothing multi-file edits: snapshot a set of
//!    files, apply candidate contents, run the verifier, and **commit only if it
//!    passes — otherwise restore every file byte-for-byte**. Generalizes the
//!    single-file reversibility used in `solve` to a whole edit set.
//! 2. [`AttemptMemory`] — remembers candidate edit-sets that were already tried
//!    and rejected (by content hash) for a given objective, so the loop never
//!    re-proposes a known-failed change and can tell the proposer what failed.
//! 3. [`agentic_loop`] — the verifier-gated iterative core: ask a `Proposer`
//!    for an edit-set, apply it as a transaction, accept on green, else record
//!    the failure and re-prompt with feedback. The `Proposer` is a trait so the
//!    deterministic tests (and CI) need no LLM, while production plugs in the
//!    model backend.
//!
//! Everything is offline and reversible. The "intelligence" of the result is
//! *measured* by the evaluation harness ([`crate::agentic_eval`]) as a
//! success-rate over seeded broken-repo fixtures — a number, never a claim.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::provenance::sha256_hex;

/// One proposed edit: the full intended contents of a single file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEdit {
    pub path: PathBuf,
    pub content: String,
}

/// A set of file edits proposed as one atomic change.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EditSet {
    pub edits: Vec<FileEdit>,
}

impl EditSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, path: impl Into<PathBuf>, content: impl Into<String>) -> Self {
        self.edits.push(FileEdit {
            path: path.into(),
            content: content.into(),
        });
        self
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// Stable content hash of the *effective* edit-set — the identity used to
    /// dedup attempts. Duplicate paths collapse last-write-wins (matching
    /// [`Transaction::apply`]'s on-disk effect), so two edit-sets that leave the
    /// same files with the same bytes hash identically regardless of proposal
    /// order or redundant entries.
    pub fn fingerprint(&self) -> String {
        let mut latest: BTreeMap<&PathBuf, &String> = BTreeMap::new();
        for e in &self.edits {
            latest.insert(&e.path, &e.content); // last write wins, like apply()
        }
        let mut buf = String::new();
        for (p, c) in latest {
            buf.push_str(&p.to_string_lossy());
            buf.push('\0');
            buf.push_str(c);
            buf.push('\u{1}');
        }
        sha256_hex(buf.as_bytes())
    }
}

/// All-or-nothing application of an [`EditSet`]: snapshot every targeted file's
/// current bytes (recording which did not exist), write the new contents, and
/// hand control back to the caller to verify. On [`Transaction::rollback`] every
/// file is restored byte-for-byte (and created files are removed); on
/// [`Transaction::commit`] the changes simply stay. This is the safety boundary:
/// a rejected multi-file change never half-applies.
pub struct Transaction {
    /// path -> Some(original bytes) if it existed, None if it was newly created.
    /// Raw bytes (not String) so a non-UTF8 file is restored byte-for-byte
    /// rather than mistaken for "newly created" and deleted on rollback.
    snapshot: BTreeMap<PathBuf, Option<Vec<u8>>>,
    applied: bool,
}

impl Transaction {
    /// Snapshot and apply `edits`. Returns an error (after best-effort rollback)
    /// if any write fails, so a partial application never escapes.
    pub fn apply(edits: &EditSet) -> std::io::Result<Self> {
        let mut tx = Transaction {
            snapshot: BTreeMap::new(),
            applied: false,
        };
        for edit in &edits.edits {
            if tx.snapshot.contains_key(&edit.path) {
                // Same file targeted twice in one set: keep the first snapshot
                // (the true pre-transaction state) and let the later write win.
            } else {
                let original = std::fs::read(&edit.path).ok();
                tx.snapshot.insert(edit.path.clone(), original);
            }
            if let Some(parent) = edit.path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            if let Err(e) = std::fs::write(&edit.path, &edit.content) {
                // Roll back whatever we already wrote, then surface the error.
                tx.rollback();
                return Err(e);
            }
        }
        tx.applied = true;
        Ok(tx)
    }

    /// Keep the applied changes (consume the transaction without restoring).
    pub fn commit(mut self) {
        self.applied = false; // disarm Drop's rollback
        self.snapshot.clear();
    }

    /// Restore every touched file to its pre-transaction state, byte-for-byte;
    /// files that did not exist before are removed.
    pub fn rollback(&mut self) {
        for (path, original) in &self.snapshot {
            match original {
                Some(bytes) => {
                    let _ = std::fs::write(path, bytes);
                }
                None => {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        self.applied = false;
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        // A transaction dropped without commit() (e.g. an early return or panic
        // on a failed verify) must not leave edits applied.
        if self.applied {
            self.rollback();
        }
    }
}

/// Remembers edit-sets already tried and rejected for an objective, keyed by the
/// objective fingerprint, so the loop never re-proposes a known-failed change.
#[derive(Debug, Clone, Default)]
pub struct AttemptMemory {
    rejected: BTreeMap<String, Vec<String>>, // objective fp -> rejected edit-set fps
}

impl AttemptMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Has this exact edit-set already been tried and rejected for `objective`?
    pub fn was_rejected(&self, objective: &str, edits: &EditSet) -> bool {
        self.rejected
            .get(objective)
            .map(|v| v.contains(&edits.fingerprint()))
            .unwrap_or(false)
    }

    /// Record an edit-set as rejected for `objective`.
    pub fn record_rejected(&mut self, objective: &str, edits: &EditSet) {
        let fp = edits.fingerprint();
        let list = self.rejected.entry(objective.to_string()).or_default();
        if !list.contains(&fp) {
            list.push(fp);
        }
    }

    /// How many distinct rejected attempts are remembered for `objective`.
    #[allow(dead_code)] // public accessor; used by tests and downstream callers
    pub fn rejected_count(&self, objective: &str) -> usize {
        self.rejected.get(objective).map(|v| v.len()).unwrap_or(0)
    }
}

/// Feedback handed to a [`Proposer`] for the next attempt.
#[derive(Debug, Clone)]
pub struct ProposeContext<'a> {
    /// The objective in natural language (for repair this is "make the command
    /// pass"; for a feature it's the user's described change).
    pub objective: &'a str,
    /// Latest verifier output (empty on the first attempt).
    pub last_failure: &'a str,
    /// 1-based attempt number.
    pub attempt: usize,
    /// Fingerprints of edit-sets already rejected this run, so the proposer can
    /// be told not to repeat them.
    pub rejected_so_far: usize,
}

/// Something that proposes an [`EditSet`] toward an objective. Implemented by the
/// LLM backend in production and by deterministic fakes in tests/CI.
pub trait Proposer {
    /// Return a candidate edit-set, or `None` to give up this attempt.
    fn propose(&mut self, ctx: &ProposeContext) -> Option<EditSet>;
}

/// Outcome of an [`agentic_loop`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgenticOutcome {
    pub solved: bool,
    pub attempts: usize,
    /// Distinct candidates that were applied but rejected by the verifier.
    pub rejected: usize,
}

/// The verifier-gated, reversible, iterative agentic core. For up to
/// `max_attempts`: ask `propose` for an edit-set; skip it (without consuming a
/// distinct-rejection slot) if it is empty or already-rejected; otherwise apply
/// it as an all-or-nothing [`Transaction`], run `verify`, and **commit only if
/// it returns true** — else roll back byte-for-byte, remember the rejection, and
/// feed the fresh `capture` output into the next attempt.
///
/// `verify` returns whether the objective is satisfied; `capture` returns the
/// verifier output to show the proposer next (called after a rejection).
pub fn agentic_loop<P, V, C>(
    objective: &str,
    max_attempts: usize,
    memory: &mut AttemptMemory,
    proposer: &mut P,
    mut verify: V,
    mut capture: C,
) -> AgenticOutcome
where
    P: Proposer,
    V: FnMut() -> bool,
    C: FnMut() -> String,
{
    let mut last_failure = capture();
    let mut rejected = 0usize;
    let max = max_attempts.max(1);
    for attempt in 1..=max {
        let ctx = ProposeContext {
            objective,
            last_failure: &last_failure,
            attempt,
            rejected_so_far: rejected,
        };
        let Some(edits) = proposer.propose(&ctx) else {
            continue;
        };
        if edits.is_empty() || memory.was_rejected(objective, &edits) {
            // Nothing to do / already known bad — don't waste a verify run.
            continue;
        }
        let Ok(tx) = Transaction::apply(&edits) else {
            // Could not even apply (I/O error) — treat as rejected.
            memory.record_rejected(objective, &edits);
            rejected += 1;
            continue;
        };
        if verify() {
            tx.commit();
            return AgenticOutcome {
                solved: true,
                attempts: attempt,
                rejected,
            };
        }
        // Rejected: roll back the whole set, remember it, refresh feedback.
        drop(tx); // Drop rolls back (not committed)
        memory.record_rejected(objective, &edits);
        rejected += 1;
        last_failure = capture();
    }
    AgenticOutcome {
        solved: false,
        attempts: max,
        rejected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("axiom_agentic_{tag}_{n}"))
    }

    #[test]
    fn transaction_commit_keeps_changes() {
        let dir = tmp("commit");
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.txt");
        std::fs::write(&a, "old").unwrap();
        let edits = EditSet::new().with(a.clone(), "new");
        let tx = Transaction::apply(&edits).unwrap();
        tx.commit();
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "new");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transaction_rollback_restores_all_files_byte_for_byte() {
        let dir = tmp("rollback");
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.txt");
        let b = dir.join("sub/b.txt"); // newly created in a new dir
        std::fs::write(&a, "A0").unwrap();

        let edits = EditSet::new().with(a.clone(), "A1").with(b.clone(), "B1");
        {
            let _tx = Transaction::apply(&edits).unwrap();
            assert_eq!(std::fs::read_to_string(&a).unwrap(), "A1");
            assert_eq!(std::fs::read_to_string(&b).unwrap(), "B1");
            // dropped without commit -> rollback
        }
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "A0", "existing file restored");
        assert!(!b.exists(), "newly created file removed on rollback");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_set_fingerprint_is_order_independent() {
        let f1 = EditSet::new().with("a", "x").with("b", "y").fingerprint();
        let f2 = EditSet::new().with("b", "y").with("a", "x").fingerprint();
        assert_eq!(f1, f2, "same files+bytes hash identically regardless of order");
        let f3 = EditSet::new().with("a", "x").with("b", "z").fingerprint();
        assert_ne!(f1, f3, "different bytes hash differently");
    }

    #[test]
    fn edit_set_fingerprint_collapses_duplicate_paths_last_write_wins() {
        // Transaction::apply is last-write-wins for a repeated path, so the
        // fingerprint must match the single-edit form with the same final bytes.
        let dup = EditSet::new().with("a", "first").with("a", "final").fingerprint();
        let single = EditSet::new().with("a", "final").fingerprint();
        assert_eq!(dup, single, "redundant earlier write must not change identity");
    }

    #[test]
    fn transaction_restores_non_utf8_files_byte_for_byte() {
        let dir = tmp("binary");
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("data.bin");
        let original: &[u8] = &[0xff, 0x00, 0xfe, 0x80, b'h', b'i'];
        std::fs::write(&bin, original).unwrap();

        // Overwrite the (non-UTF8) file inside a transaction, then roll back.
        let edits = EditSet::new().with(bin.clone(), "text overwrite");
        {
            let _tx = Transaction::apply(&edits).unwrap();
            assert_eq!(std::fs::read_to_string(&bin).unwrap(), "text overwrite");
        }
        assert_eq!(
            std::fs::read(&bin).unwrap(),
            original,
            "a non-UTF8 original must be restored byte-for-byte, not deleted"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn attempt_memory_dedups_per_objective() {
        let mut m = AttemptMemory::new();
        let e = EditSet::new().with("a", "x");
        assert!(!m.was_rejected("goal", &e));
        m.record_rejected("goal", &e);
        m.record_rejected("goal", &e); // idempotent
        assert!(m.was_rejected("goal", &e));
        assert_eq!(m.rejected_count("goal"), 1);
        assert!(!m.was_rejected("other-goal", &e), "scoped by objective");
    }

    /// A proposer that returns a scripted sequence of edit-sets.
    struct ScriptedProposer {
        scripted: Vec<EditSet>,
        idx: usize,
    }
    impl Proposer for ScriptedProposer {
        fn propose(&mut self, _ctx: &ProposeContext) -> Option<EditSet> {
            let e = self.scripted.get(self.idx).cloned();
            self.idx += 1;
            e
        }
    }

    #[test]
    fn agentic_loop_keeps_only_the_verifying_edit_set() {
        let dir = tmp("loop");
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("prog.txt");
        std::fs::write(&target, "BROKEN").unwrap();

        // First proposal is wrong, second makes the file contain GOOD.
        let mut proposer = ScriptedProposer {
            scripted: vec![
                EditSet::new().with(target.clone(), "STILL BAD"),
                EditSet::new().with(target.clone(), "GOOD"),
            ],
            idx: 0,
        };
        let mut mem = AttemptMemory::new();
        let t = target.clone();
        let verify = move || std::fs::read_to_string(&t).map(|s| s == "GOOD").unwrap_or(false);
        let outcome = agentic_loop("make it GOOD", 5, &mut mem, &mut proposer, verify, String::new);

        assert!(outcome.solved);
        assert_eq!(outcome.attempts, 2);
        assert_eq!(outcome.rejected, 1, "the first wrong set was rolled back");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "GOOD");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn agentic_loop_restores_source_when_nothing_verifies() {
        let dir = tmp("loopfail");
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("prog.txt");
        std::fs::write(&target, "ORIGINAL").unwrap();

        let mut proposer = ScriptedProposer {
            scripted: vec![EditSet::new().with(target.clone(), "NOPE")],
            idx: 0,
        };
        let mut mem = AttemptMemory::new();
        let outcome = agentic_loop("impossible", 3, &mut mem, &mut proposer, || false, String::new);

        assert!(!outcome.solved);
        assert_eq!(outcome.rejected, 1);
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "ORIGINAL",
            "a never-verifying run leaves the source byte-for-byte intact"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn agentic_loop_skips_already_rejected_edit_sets() {
        let dir = tmp("loopskip");
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("prog.txt");
        std::fs::write(&target, "X").unwrap();

        // Same wrong proposal three times; memory should keep it from being
        // re-applied after the first rejection (rejected stays at 1).
        let same = EditSet::new().with(target.clone(), "WRONG");
        let mut proposer = ScriptedProposer {
            scripted: vec![same.clone(), same.clone(), same.clone()],
            idx: 0,
        };
        let mut mem = AttemptMemory::new();
        let outcome = agentic_loop("goal", 3, &mut mem, &mut proposer, || false, String::new);
        assert!(!outcome.solved);
        assert_eq!(outcome.rejected, 1, "identical proposal only counted/applied once");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
