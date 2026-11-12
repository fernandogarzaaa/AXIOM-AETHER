//! AXIOM Agentic Core — the autonomy capstone.
//!
//! The `solve` loop (Pillar 3) already repairs a failing check: localize the
//! faulty file(s) across languages, propose a fix (deterministic mutation,
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

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use crate::provenance::sha256_hex;

/// Last whole-second stamp handed out by [`mark_fresh`] (Unix seconds), so
/// every write in this process gets a strictly later second than the one
/// before it.
static LAST_FRESH_STAMP: AtomicU64 = AtomicU64::new(0);

/// Give `path` a modification time no other candidate write in this process has
/// used, so mtime-keyed build caches never serve a stale artifact for it.
///
/// Why this exists: the repair loops write a candidate, run the verifier, and
/// decide on the result. Python's `__pycache__` (and `make`-style tools)
/// decide whether to recompile by comparing the source's **mtime (seconds) and
/// size** to what they recorded. A candidate the same size as the original,
/// written within the same second, is therefore run from the *stale* bytecode
/// of the original — the verifier reports the original's failure and a correct
/// fix is rolled back as rejected. Observed on `py-swapped-args` (swapping two
/// arguments never changes the size).
///
/// The stamp must also never move **backwards**. Cargo, make, ninja and most
/// build tools rebuild only when a source is *newer* than its build output; an
/// earlier version of this function set the mtime into the past (to avoid
/// future-dated files) and every Rust candidate was then run from the stale
/// binary and rejected, correct fixes included (seeded-fault benchmark,
/// 2026-11-06: the authored fix for a `>=`/`>` fault failed with a past mtime
/// and passed with the current one).
///
/// *Now* is rounded **up** to the next whole second. Rounded down, a
/// candidate written in the same wall-clock second as the previous build
/// started got a stamp no newer than that build, and cargo reran the stale
/// binary: a seeded fault whose fix was the very first candidate passed when
/// run by hand and failed in the benchmark, depending on sub-second timing.
///
/// So the stamp is the largest of *now* (rounded up), *one whole second after
/// the previous stamp*, and one second after `base` (the file's mtime before
/// the write), in whole seconds. That gives:
/// * Python: a different second from whatever the `.pyc` recorded;
/// * cargo/make: strictly newer than any build that started before the write;
/// * monotonic within the process, so candidate N+1 is never "older" than N.
///
/// When writes come faster than one per second the stamps run ahead of the
/// wall clock by at most the number of writes in the burst (bounded by the
/// candidate cap, 64 by default); `make` may print a clock-skew notice, and
/// nothing else is affected. Best-effort: a filesystem that refuses
/// `set_modified` leaves the file as written.
pub fn mark_fresh(path: &Path, base: Option<SystemTime>) {
    let secs = |t: SystemTime| {
        t.duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    };
    let now = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() + u64::from(d.subsec_nanos() > 0))
        .unwrap_or(0);
    let floor = base.map(|b| secs(b) + 1).unwrap_or(0);
    let mut prev = LAST_FRESH_STAMP.load(Ordering::Relaxed);
    let stamp = loop {
        let next = now.max(prev + 1).max(floor);
        match LAST_FRESH_STAMP.compare_exchange_weak(
            prev,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break next,
            Err(actual) => prev = actual,
        }
    };
    if let Ok(f) = std::fs::OpenOptions::new().write(true).open(path) {
        let _ = f.set_modified(std::time::UNIX_EPOCH + Duration::from_secs(stamp));
    }
}

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
/// file whose write was attempted is restored byte-for-byte (and created files
/// are removed); on [`Transaction::commit`] the changes simply stay. This is
/// the safety boundary: a rejected multi-file change never half-applies.
/// Rollback restores only attempted targets, never files the write loop never
/// reached: rewriting an untouched file can corrupt it (a rollback write under
/// a file-size limit truncates a file the transaction never modified).
pub struct Transaction {
    /// path -> Some(original bytes) if it existed and was readable, None if it
    /// did not exist. An original that exists but cannot be read aborts
    /// [`Transaction::apply`] before any mutation: it is never recorded as
    /// "did not exist", so rollback can never delete it by mistake.
    /// Raw bytes (not String) so a non-UTF8 file is restored byte-for-byte
    /// rather than mistaken for "newly created" and deleted on rollback.
    snapshot: BTreeMap<PathBuf, Option<Vec<u8>>>,
    /// Targets whose writes were attempted, in path order. A path is recorded
    /// immediately before its write, so a failing write (which may have
    /// partially modified its target) is included, while a target the write
    /// loop never reached is never restored. `rollback` restores exactly
    /// these paths.
    attempted: BTreeSet<PathBuf>,
    applied: bool,
}

impl Transaction {
    /// Snapshot and apply `edits`. Returns an error (after best-effort rollback)
    /// if any step fails, so a partial application never escapes.
    ///
    /// Two phases. First, snapshot every distinct target BEFORE any mutation:
    /// a target that exists but cannot be read is an error here, not a
    /// "did not exist" snapshot, so apply refuses before touching anything.
    /// Second, write the edits with rollback armed, so any failure in the
    /// loop (including `create_dir_all`) drops the transaction and rolls
    /// back what was already written.
    pub fn apply(edits: &EditSet) -> std::io::Result<Self> {
        let mut tx = Transaction {
            snapshot: BTreeMap::new(),
            attempted: BTreeSet::new(),
            applied: false,
        };
        for edit in &edits.edits {
            if tx.snapshot.contains_key(&edit.path) {
                // Same file targeted twice in one set: keep the first snapshot
                // (the true pre-transaction state) and let the later write win.
                continue;
            }
            match std::fs::read(&edit.path) {
                Ok(bytes) => {
                    tx.snapshot.insert(edit.path.clone(), Some(bytes));
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    tx.snapshot.insert(edit.path.clone(), None);
                }
                Err(e) => return Err(e),
            }
        }
        // Arm rollback before the first write: from here on, any early
        // return drops `tx` and `Drop` rolls back what was written.
        tx.applied = true;
        for edit in &edits.edits {
            if let Some(parent) = edit.path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            // Record the attempt BEFORE the write: a failing write may have
            // partially modified its target, so it must be restored, while
            // a target the loop never reaches must never be touched.
            tx.attempted.insert(edit.path.clone());
            let before = std::fs::metadata(&edit.path)
                .and_then(|m| m.modified())
                .ok();
            std::fs::write(&edit.path, &edit.content)?;
            // A same-size rewrite in the same second must still look new to
            // mtime-keyed caches (see `mark_fresh`).
            mark_fresh(&edit.path, before);
        }
        Ok(tx)
    }

    /// Keep the applied changes (consume the transaction without restoring).
    pub fn commit(mut self) {
        self.applied = false; // disarm Drop's rollback
        self.snapshot.clear();
        self.attempted.clear();
    }

    /// Restore every file whose write was attempted to its pre-transaction
    /// state, byte-for-byte; files that did not exist before are removed.
    /// Targets the write loop never reached are left exactly as they were:
    /// rewriting one could corrupt it (a rollback write under a file-size
    /// limit truncates a file the transaction never modified). Every
    /// attempted file is restored even when one restoration fails, and the
    /// first error is returned so a failed rollback is observable instead of
    /// silently leaving files modified. Removing a created file that already
    /// vanished is not an error: the desired end state already holds.
    pub fn rollback(&mut self) -> std::io::Result<()> {
        let mut first_error: Option<std::io::Error> = None;
        for path in &self.attempted {
            let Some(original) = self.snapshot.get(path) else {
                continue;
            };
            let result = match original {
                Some(bytes) => {
                    let before = std::fs::metadata(path).and_then(|m| m.modified()).ok();
                    let r = std::fs::write(path, bytes);
                    if r.is_ok() {
                        // The restored bytes must look newer than the rejected
                        // candidate's build output, or the next verify runs it.
                        mark_fresh(path, before);
                    }
                    r
                }
                None => match std::fs::remove_file(path) {
                    Ok(()) => Ok(()),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                    Err(e) => Err(e),
                },
            };
            if let Err(e) = result {
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
        }
        self.applied = false;
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        // A transaction dropped without commit() (e.g. an early return or panic
        // on a failed verify) must not leave edits applied.
        if self.applied {
            let _ = self.rollback();
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
    verify: V,
    capture: C,
) -> AgenticOutcome
where
    P: Proposer,
    V: FnMut() -> bool,
    C: FnMut() -> String,
{
    // Single-gate loop: a candidate is accepted iff `verify()` passes.
    run_agentic(objective, max_attempts, memory, proposer, verify, capture)
}

/// **B.4 — held-out verification split (anti-overfitting).** Same verifier-gated
/// loop as [`agentic_loop`], but a candidate is committed only when it passes the
/// `verify_train` subset **and** the held-out `verify_holdout` subset. A patch
/// that passes the train subset yet fails the held-out one is the classic
/// test-overfit fix (green on the tests it was shaped against, wrong in general):
/// it is rolled back and counted as a rejection like any other, so the loop keeps
/// searching instead of trusting it.
///
/// `verify_holdout` is only run when `verify_train` already passed (cheap-first),
/// so the held-out suite costs nothing on the common failing path.
pub fn agentic_loop_with_holdout<P, Vt, Vh, C>(
    objective: &str,
    max_attempts: usize,
    memory: &mut AttemptMemory,
    proposer: &mut P,
    mut verify_train: Vt,
    mut verify_holdout: Vh,
    capture: C,
) -> AgenticOutcome
where
    P: Proposer,
    Vt: FnMut() -> bool,
    Vh: FnMut() -> bool,
    C: FnMut() -> String,
{
    run_agentic(
        objective,
        max_attempts,
        memory,
        proposer,
        || verify_train() && verify_holdout(),
        capture,
    )
}

/// Shared core for the agentic loop, generic over the accept predicate so the
/// single-gate and held-out-split variants share one audited control flow.
fn run_agentic<P, A, C>(
    objective: &str,
    max_attempts: usize,
    memory: &mut AttemptMemory,
    proposer: &mut P,
    mut accept: A,
    mut capture: C,
) -> AgenticOutcome
where
    P: Proposer,
    A: FnMut() -> bool,
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
        if accept() {
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
    fn mark_fresh_beats_a_build_that_started_this_second() {
        // A long-unchanged source (old mtime) whose build started a moment
        // before the candidate is written: the candidate must look newer
        // than that build, or cargo reruns the stale binary.
        let dir = tmp("fresh_build");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("lib.rs");
        std::fs::write(&f, "a").unwrap();
        let old = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        std::fs::File::options()
            .write(true)
            .open(&f)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let build_started = SystemTime::now();
        std::fs::write(&f, "b").unwrap();
        mark_fresh(&f, Some(old));
        assert!(std::fs::metadata(&f).unwrap().modified().unwrap() > build_started);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mark_fresh_is_monotonic_never_in_the_past_and_distinct_per_write() {
        let dir = tmp("fresh");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("m.py");
        std::fs::write(&f, "a").unwrap();
        let original = std::fs::metadata(&f).unwrap().modified().unwrap();
        let start = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut seen: Vec<u64> = Vec::new();
        for _ in 0..3 {
            std::fs::write(&f, "b").unwrap(); // same size as the original
            let before = std::fs::metadata(&f).unwrap().modified().ok();
            // Anything that started before the write (a build) must look older.
            let build_started = SystemTime::now();
            mark_fresh(&f, before);
            assert!(
                std::fs::metadata(&f).unwrap().modified().unwrap() > build_started,
                "strictly newer than a build that started just before the write"
            );
            let m = std::fs::metadata(&f)
                .unwrap()
                .modified()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            assert!(m >= start, "never earlier than the wall clock at start");
            assert!(
                SystemTime::UNIX_EPOCH + Duration::from_secs(m) > original,
                "newer than the original, so timestamp-based builds rebuild"
            );
            assert!(
                seen.iter().all(|p| *p < m),
                "strictly increasing: {seen:?} then {m}"
            );
            seen.push(m);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rollback_restores_bytes_with_a_newer_mtime_than_the_candidate() {
        let dir = tmp("rbfresh");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("lib.rs");
        std::fs::write(&f, "fn a() {}").unwrap();
        let mut tx = Transaction::apply(&EditSet::new().with(f.clone(), "fn b() {}")).unwrap();
        let cand = std::fs::metadata(&f).unwrap().modified().unwrap();
        tx.rollback().unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "fn a() {}");
        let restored = std::fs::metadata(&f).unwrap().modified().unwrap();
        assert!(
            restored > cand,
            "restored file must look newer than the candidate"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transaction_apply_changes_mtime_even_for_same_size_rewrite() {
        let dir = tmp("txfresh");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("m.py");
        std::fs::write(&f, "x = 1").unwrap();
        let before = std::fs::metadata(&f).unwrap().modified().unwrap();
        let tx = Transaction::apply(&EditSet::new().with(f.clone(), "x = 2")).unwrap();
        let after = std::fs::metadata(&f).unwrap().modified().unwrap();
        assert_ne!(before, after);
        tx.commit();
        let _ = std::fs::remove_dir_all(&dir);
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
        assert_eq!(
            std::fs::read_to_string(&a).unwrap(),
            "A0",
            "existing file restored"
        );
        assert!(!b.exists(), "newly created file removed on rollback");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transaction_apply_rolls_back_first_file_when_second_file_parent_cannot_be_created() {
        // Repro for audit P1-1: create_dir_all failure on the second file
        // returned without rolling back, leaving the first file modified.
        let dir = tmp("rollback_dirfail");
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.txt");
        std::fs::write(&a, "A0").unwrap();
        // A regular file where the second edit's parent directory must go.
        let blocker = dir.join("blocker");
        std::fs::write(&blocker, "not a dir").unwrap();
        let b = blocker.join("sub").join("b.txt");

        let edits = EditSet::new().with(a.clone(), "A1").with(b.clone(), "B1");
        let result = Transaction::apply(&edits);
        assert!(
            result.is_err(),
            "apply must fail when a parent cannot be created"
        );
        assert_eq!(
            std::fs::read_to_string(&a).unwrap(),
            "A0",
            "first file must be rolled back, not left modified"
        );
        assert!(!b.exists(), "second file must not exist");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transaction_apply_rolls_back_first_two_files_when_third_file_fails() {
        let dir = tmp("rollback_thirdfail");
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.txt");
        let b = dir.join("b.txt");
        std::fs::write(&a, "A0").unwrap();
        std::fs::write(&b, "B0").unwrap();
        let blocker = dir.join("blocker");
        std::fs::write(&blocker, "not a dir").unwrap();
        let c = blocker.join("sub").join("c.txt");

        let edits = EditSet::new()
            .with(a.clone(), "A1")
            .with(b.clone(), "B1")
            .with(c.clone(), "C1");
        let result = Transaction::apply(&edits);
        assert!(
            result.is_err(),
            "apply must fail when the third parent cannot be created"
        );
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "A0");
        assert_eq!(std::fs::read_to_string(&b).unwrap(), "B0");
        assert!(!c.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Repro for the Codex P2 on PR #276: rollback rewrote every
    /// snapshotted target, including files the write loop never reached.
    /// Under a file-size limit that rewrite itself corrupts a file the
    /// transaction never modified: with RLIMIT_FSIZE=1024 and SIGXFSZ
    /// ignored, a failing 2048-byte first edit leaves a later 4096-byte
    /// file truncated to 1024 bytes during rollback, while the parent
    /// commit leaves it intact.
    ///
    /// A file-size limit is process-wide, so the repro runs in a child
    /// process (this same test binary, filtered to this test) and the
    /// parent only asserts the child passed.
    #[test]
    #[cfg(unix)]
    fn transaction_rollback_leaves_unreached_files_untouched() {
        if std::env::var_os("AXIOM_ROLLBACK_RLIMIT_REPRO").is_none() {
            let exe = std::env::current_exe().unwrap();
            let out = std::process::Command::new(exe)
                .args([
                    "--exact",
                    "agentic::tests::transaction_rollback_leaves_unreached_files_untouched",
                    "--test-threads=1",
                ])
                .env("AXIOM_ROLLBACK_RLIMIT_REPRO", "1")
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "rlimit repro child failed: stdout={} stderr={}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        rollback_leaves_unreached_files_untouched_under_limit();
    }

    /// Child-process body of the test above: runs under RLIMIT_FSIZE=1024
    /// with SIGXFSZ ignored, so the first (2048-byte) write fails with
    /// EFBIG after a partial write and rollback runs under the limit.
    #[cfg(unix)]
    fn rollback_leaves_unreached_files_untouched_under_limit() {
        let dir = tmp("rollback_unreached");
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.txt");
        let b = dir.join("b.txt");
        std::fs::write(&a, "A0").unwrap();
        let b_original = vec![b'x'; 4096];
        std::fs::write(&b, &b_original).unwrap();
        let old = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        std::fs::File::options()
            .write(true)
            .open(&b)
            .unwrap()
            .set_modified(old)
            .unwrap();

        // Save the current limits, cap file size at 1024, and ignore the
        // signal that would otherwise kill the process on overflow.
        unsafe {
            let mut old_rlimit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            assert_eq!(libc::getrlimit(libc::RLIMIT_FSIZE, &mut old_rlimit), 0);
            assert_ne!(libc::signal(libc::SIGXFSZ, libc::SIG_IGN), libc::SIG_ERR);
            let capped = libc::rlimit {
                rlim_cur: 1024,
                rlim_max: old_rlimit.rlim_max,
            };
            assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &capped), 0);
            let result = Transaction::apply(
                &EditSet::new()
                    .with(a.clone(), "A".repeat(2048))
                    .with(b.clone(), "B".repeat(4096)),
            );
            // Restore the limits before asserting anything: no later file
            // operation in this process may run under the cap.
            assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &old_rlimit), 0);
            libc::signal(libc::SIGXFSZ, libc::SIG_DFL);
            assert!(
                result.is_err(),
                "apply must fail when the first write hits the file-size limit"
            );
        }
        assert_eq!(
            std::fs::read_to_string(&a).unwrap(),
            "A0",
            "attempted file must be rolled back after a partial write"
        );
        assert_eq!(
            std::fs::read(&b).unwrap(),
            b_original,
            "unreached file must keep its exact bytes: rollback must not have rewritten it under the limit"
        );
        assert_eq!(
            std::fs::metadata(&b).unwrap().modified().unwrap(),
            old,
            "unreached file must keep its exact mtime"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transaction_apply_refuses_unreadable_original_before_mutating() {
        // A directory as an edit target cannot be read (EISDIR), so its
        // original is unreadable, not nonexistent. Apply must fail BEFORE
        // touching any other file: content and mtime both untouched.
        let dir = tmp("rollback_unreadable");
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.txt");
        std::fs::write(&good, "G0").unwrap();
        let old = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        std::fs::File::options()
            .write(true)
            .open(&good)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let target_dir = dir.join("target_dir");
        std::fs::create_dir_all(&target_dir).unwrap();

        let edits = EditSet::new()
            .with(good.clone(), "G1")
            .with(target_dir.clone(), "x");
        let result = Transaction::apply(&edits);
        assert!(result.is_err(), "apply must fail on an unreadable original");
        assert_eq!(std::fs::read_to_string(&good).unwrap(), "G0");
        assert_eq!(
            std::fs::metadata(&good).unwrap().modified().unwrap(),
            old,
            "untouched file must keep its exact mtime: no write happened at all"
        );
        assert!(target_dir.is_dir(), "the directory target must still exist");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transaction_rollback_reports_failure_and_restores_remaining_files() {
        // Repro for audit P1-1: rollback swallowed restoration errors.
        // Replace one target with a directory after apply, so its restore
        // fails; rollback must report the error and still restore the other.
        let dir = tmp("rollback_fail");
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("aaa.txt");
        let good = dir.join("zzz.txt");
        std::fs::write(&bad, "BAD0").unwrap();
        std::fs::write(&good, "GOOD0").unwrap();

        let edits = EditSet::new()
            .with(bad.clone(), "BAD1")
            .with(good.clone(), "GOOD1");
        let mut tx = Transaction::apply(&edits).unwrap();
        std::fs::remove_file(&bad).unwrap();
        std::fs::create_dir_all(&bad).unwrap();

        let result = tx.rollback();
        assert!(result.is_err(), "rollback must report a failed restoration");
        assert_eq!(
            std::fs::read_to_string(&good).unwrap(),
            "GOOD0",
            "rollback must keep restoring the remaining files after one failure"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transaction_rollback_reports_failed_removal_of_created_file() {
        // The remove arm of rollback must report a real removal failure:
        // only a file that already vanished (NotFound) is tolerated. A
        // created file replaced by a directory cannot be removed, so
        // rollback must fail even though the other file still restores.
        let dir = tmp("rollback-remove-fail");
        std::fs::create_dir_all(&dir).unwrap();
        let existing = dir.join("aaa_existing.txt");
        std::fs::write(&existing, "before").unwrap();
        let created = dir.join("zzz_created.txt");
        let mut tx = Transaction::apply(
            &EditSet::new()
                .with(existing.clone(), "after")
                .with(created.clone(), "new"),
        )
        .unwrap();
        std::fs::remove_file(&created).unwrap();
        std::fs::create_dir(&created).unwrap();
        let result = tx.rollback();
        assert!(result.is_err(), "rollback must report a failed removal");
        assert_eq!(std::fs::read_to_string(&existing).unwrap(), "before");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transaction_rollback_tolerates_already_removed_created_file() {
        // A created file that vanished before rollback is already in the
        // desired end state, so rollback succeeds instead of reporting it.
        let dir = tmp("rollback_gone");
        std::fs::create_dir_all(&dir).unwrap();
        let created = dir.join("new.txt");

        let edits = EditSet::new().with(created.clone(), "N1");
        let mut tx = Transaction::apply(&edits).unwrap();
        assert!(created.exists());
        std::fs::remove_file(&created).unwrap();

        assert!(
            tx.rollback().is_ok(),
            "already-removed file is not an error"
        );
        assert!(!created.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_set_fingerprint_is_order_independent() {
        let f1 = EditSet::new().with("a", "x").with("b", "y").fingerprint();
        let f2 = EditSet::new().with("b", "y").with("a", "x").fingerprint();
        assert_eq!(
            f1, f2,
            "same files+bytes hash identically regardless of order"
        );
        let f3 = EditSet::new().with("a", "x").with("b", "z").fingerprint();
        assert_ne!(f1, f3, "different bytes hash differently");
    }

    #[test]
    fn edit_set_fingerprint_collapses_duplicate_paths_last_write_wins() {
        // Transaction::apply is last-write-wins for a repeated path, so the
        // fingerprint must match the single-edit form with the same final bytes.
        let dup = EditSet::new()
            .with("a", "first")
            .with("a", "final")
            .fingerprint();
        let single = EditSet::new().with("a", "final").fingerprint();
        assert_eq!(
            dup, single,
            "redundant earlier write must not change identity"
        );
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
        let verify = move || {
            std::fs::read_to_string(&t)
                .map(|s| s == "GOOD")
                .unwrap_or(false)
        };
        let outcome = agentic_loop(
            "make it GOOD",
            5,
            &mut mem,
            &mut proposer,
            verify,
            String::new,
        );

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
        let outcome = agentic_loop(
            "impossible",
            3,
            &mut mem,
            &mut proposer,
            || false,
            String::new,
        );

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
    fn holdout_rejects_a_patch_that_overfits_the_train_subset() {
        // The first proposal passes the train check (contains "TRAIN") but fails
        // the held-out check (must also contain "HOLD"): a classic overfit fix.
        // The split loop must reject it, roll back, and accept only the second
        // proposal that satisfies both — never trusting the overfit patch.
        let dir = tmp("holdout");
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("prog.txt");
        std::fs::write(&target, "BROKEN").unwrap();

        let mut proposer = ScriptedProposer {
            scripted: vec![
                EditSet::new().with(target.clone(), "TRAIN"), // passes train only
                EditSet::new().with(target.clone(), "TRAIN HOLD"), // passes both
            ],
            idx: 0,
        };
        let mut mem = AttemptMemory::new();
        let t1 = target.clone();
        let verify_train = move || {
            std::fs::read_to_string(&t1)
                .map(|s| s.contains("TRAIN"))
                .unwrap_or(false)
        };
        let t2 = target.clone();
        let verify_holdout = move || {
            std::fs::read_to_string(&t2)
                .map(|s| s.contains("HOLD"))
                .unwrap_or(false)
        };

        let outcome = agentic_loop_with_holdout(
            "fix it",
            5,
            &mut mem,
            &mut proposer,
            verify_train,
            verify_holdout,
            String::new,
        );

        assert!(outcome.solved);
        assert_eq!(outcome.attempts, 2);
        assert_eq!(
            outcome.rejected, 1,
            "the train-only (overfit) patch must be rejected by the held-out gate"
        );
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "TRAIN HOLD");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn holdout_leaves_source_intact_when_only_train_ever_passes() {
        // A patch that forever passes train but never the held-out gate must never
        // be committed; the source stays byte-for-byte original.
        let dir = tmp("holdoutfail");
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("prog.txt");
        std::fs::write(&target, "ORIGINAL").unwrap();

        let mut proposer = ScriptedProposer {
            scripted: vec![EditSet::new().with(target.clone(), "TRAIN")],
            idx: 0,
        };
        let mut mem = AttemptMemory::new();
        let outcome = agentic_loop_with_holdout(
            "impossible",
            3,
            &mut mem,
            &mut proposer,
            || true,  // train always passes
            || false, // held-out never passes
            String::new,
        );
        assert!(!outcome.solved);
        assert_eq!(outcome.rejected, 1);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "ORIGINAL");
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
        assert_eq!(
            outcome.rejected, 1,
            "identical proposal only counted/applied once"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
