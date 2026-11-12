//! Verified-patch memory — the substrate for fleet patch sharing.
//!
//! When the autonomous repair loop ([`crate::solve`]) makes a failing target go
//! green, the verified fix can be recorded here, exported to peers (signed), and
//! merged from peers. The point: a fix one node discovers becomes shared coding
//! skill across the fleet.
//!
//! # What a stored fix is
//!
//! A [`PatchCandidate`] is a **diff against one exact file**, not a file. It
//! carries the repository it was verified in ([`RepoScope::repo`]), the file's
//! path inside that repository, the SHA-256 of the file's bytes *before* the fix
//! (the pre-image), the changed span ([`PatchDiff`]), and the SHA-256 of the file
//! after the fix.
//!
//! # Safety invariants (the whole design rests on these)
//!
//! 1. **A stored fix is applied only to the bytes it was made against.**
//!    [`PatchMemory::try_candidates`] considers a candidate only when the
//!    repository identity and path match *and* the current file hashes to the
//!    candidate's pre-image. A file that has changed since (new functions,
//!    uncommitted edits, a different project's file with the same name) hashes
//!    differently and is never touched, so reuse cannot discard work.
//! 2. **A fix is never kept on trust.** A matching candidate is written, *this
//!    node's own* verify check is run, and the fix is kept **only if it goes
//!    green locally** — otherwise the file is restored byte-for-byte.
//! 3. **Only authenticated exports are merged** ([`crate::provenance`]: a fleet
//!    key is required and the HMAC must verify). Applying a candidate runs the
//!    verify command over it, so the store must never hold untrusted input.
//!    `verified_count` records how many independent nodes have confirmed a fix
//!    green: a ranking signal, never an authorization.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::export_ledger::ExportLedger;
use crate::provenance::{
    sha256_hex, sign_export, verify_export_authenticated, ProvenanceError, SignedExport,
};

/// A fix expressed as one replaced span of the pre-image: the file's first
/// `prefix_bytes` and last `suffix_bytes` are kept, and everything between them
/// is replaced by `replacement`.
///
/// The span is widened to whole lines, so `replacement` is the changed lines as
/// they read after the fix (human-auditable in the pretty-printed store). A diff
/// is only meaningful against its own pre-image; callers must check the
/// pre-image hash before [`PatchDiff::apply`] and the post-image hash after.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatchDiff {
    /// Leading bytes of the pre-image the fix leaves unchanged.
    #[serde(default)]
    pub prefix_bytes: usize,
    /// Trailing bytes of the pre-image the fix leaves unchanged.
    #[serde(default)]
    pub suffix_bytes: usize,
    /// What the span between them becomes.
    #[serde(default)]
    pub replacement: String,
}

impl PatchDiff {
    /// The diff that turns `pre` into `post`.
    pub fn between(pre: &str, post: &str) -> Self {
        let (a, b) = (pre.as_bytes(), post.as_bytes());
        // Longest common prefix, pulled back to the start of its line. A line
        // start is always a char boundary in both strings ('\n' is ASCII and
        // the bytes before it are identical).
        let common_prefix = a.iter().zip(b).take_while(|(x, y)| x == y).count();
        let prefix = a[..common_prefix]
            .iter()
            .rposition(|&c| c == b'\n')
            .map_or(0, |i| i + 1);
        // Longest common suffix of what remains, pushed forward to the start of
        // a line (dropped entirely when it contains no line break).
        let max_suffix = a.len().min(b.len()) - prefix;
        let common_suffix = a
            .iter()
            .rev()
            .zip(b.iter().rev())
            .take(max_suffix)
            .take_while(|(x, y)| x == y)
            .count();
        let tail = &a[a.len() - common_suffix..];
        let suffix = tail
            .iter()
            .position(|&c| c == b'\n')
            .map_or(0, |i| tail.len() - (i + 1));
        Self {
            prefix_bytes: prefix,
            suffix_bytes: suffix,
            replacement: post[prefix..post.len() - suffix].to_string(),
        }
    }

    /// Apply this diff to `pre`. `None` when the diff does not fit `pre` (the
    /// kept prefix and suffix overlap, or a cut falls inside a character) —
    /// i.e. `pre` is not the pre-image this diff was made against.
    pub fn apply(&self, pre: &str) -> Option<String> {
        let kept = self.prefix_bytes.checked_add(self.suffix_bytes)?;
        if kept > pre.len() {
            return None;
        }
        let head = pre.get(..self.prefix_bytes)?;
        let tail = pre.get(pre.len() - self.suffix_bytes..)?;
        let mut out = String::with_capacity(head.len() + self.replacement.len() + tail.len());
        out.push_str(head);
        out.push_str(&self.replacement);
        out.push_str(tail);
        Some(out)
    }
}

/// Where a fix belongs: which repository, and which file inside it. Computed by
/// [`repo_scope`] identically when a fix is recorded and when one is tried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoScope {
    /// Repository identity. `git:<root commit>` for a git checkout with full
    /// history (the same across clones of one repository, different between
    /// unrelated projects), otherwise `path:<sha256 of the canonical root
    /// directory>` (this directory on this machine only). A shallow checkout
    /// is the second kind: it does not hold its root commit.
    pub repo: String,
    /// Path of the file relative to the repository root, `/`-separated.
    pub rel_path: String,
}

impl RepoScope {
    pub fn new(repo: impl Into<String>, rel_path: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            rel_path: rel_path.into(),
        }
    }
}

/// Identify the repository `src` belongs to and its path inside it.
///
/// The root is the nearest ancestor of `src` that holds a `.git` entry; with no
/// git checkout it is `anchor` (the solve working directory) or, failing that,
/// the current directory. Returns `None` when `src` cannot be resolved or does
/// not live under any such root: with no identity there is nothing safe to key
/// a stored fix on, so the caller neither records nor reuses one.
pub fn repo_scope(src: &Path, anchor: Option<&Path>) -> Option<RepoScope> {
    let src = std::fs::canonicalize(src).ok()?;
    let (root, repo) = match git_root(&src) {
        Some(root) => {
            let repo = git_root_commit(&root)
                .map(|commit| format!("git:{commit}"))
                .unwrap_or_else(|| path_identity(&root));
            (root, repo)
        }
        None => {
            let root = match anchor {
                Some(dir) => dir.to_path_buf(),
                None => std::env::current_dir().ok()?,
            };
            let root = std::fs::canonicalize(root).ok()?;
            let repo = path_identity(&root);
            (root, repo)
        }
    };
    let rel = src.strip_prefix(&root).ok()?;
    let rel_path = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    if rel_path.is_empty() {
        return None;
    }
    Some(RepoScope { repo, rel_path })
}

/// Nearest ancestor directory of `file` containing a `.git` entry (a directory
/// in a normal checkout, a file in a worktree or submodule).
fn git_root(file: &Path) -> Option<PathBuf> {
    file.ancestors()
        .skip(1)
        .find(|dir| dir.join(".git").exists())
        .map(Path::to_path_buf)
}

/// Run `git -C root <args>` with a time bound; its stdout on success.
fn git_stdout(root: &Path, args: &[&str]) -> Option<String> {
    let mut cmd = std::process::Command::new("git");
    cmd.arg("-C").arg(root).args(args);
    let out = crate::proc_bound::output_bounded(&mut cmd, Some(std::time::Duration::from_secs(20)))
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The root commit of the checkout at `root` (the lexicographically smallest
/// when history has several roots), or `None` when git is unavailable, the
/// repository has no commits, git refuses to read it, or its history is
/// shallow.
///
/// A shallow checkout (`git clone --depth N`) does not hold its root commit.
/// Git reports the commits at the shallow boundary as parentless instead, and
/// that boundary moves whenever more history is fetched, so it is not an
/// identity. Such a checkout is identified by its directory, like one without
/// git, until its history is complete. A git too old to answer the shallow
/// question is treated the same way.
fn git_root_commit(root: &Path) -> Option<String> {
    let shallow = git_stdout(root, &["rev-parse", "--is-shallow-repository"])?;
    if shallow.trim() != "false" {
        return None;
    }
    git_stdout(root, &["rev-list", "--max-parents=0", "HEAD"])?
        .lines()
        .map(str::trim)
        .filter(|line| is_sha_hex(line, 40) || is_sha_hex(line, 64))
        .min()
        .map(str::to_string)
}

/// Machine-local identity for a directory that is not a usable git checkout.
fn path_identity(root: &Path) -> String {
    format!("path:{}", sha256_hex(root.to_string_lossy().as_bytes()))
}

fn is_sha_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// One verified candidate fix for a given failure fingerprint: a diff against
/// one exact file in one repository (see the module docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatchCandidate {
    /// Identity of the repository the fix was verified in ([`RepoScope::repo`]).
    #[serde(default)]
    pub repo: String,
    /// Repository-relative path of the file the fix targets.
    pub rel_path: String,
    /// SHA-256 of the file's bytes before the fix. The fix is applied only to a
    /// file that hashes to exactly this.
    #[serde(default)]
    pub pre_sha256: String,
    /// The fix itself, as a diff against the pre-image.
    #[serde(default)]
    pub diff: PatchDiff,
    /// SHA-256 of the file after the fix — dedup key, and checked after the
    /// diff is applied so a corrupt or forged diff is never written.
    pub sha256: String,
    /// How many independent nodes have verified this patch green. A trust signal
    /// used only for ranking; it never authorizes a blind apply.
    pub verified_count: u32,
}

impl PatchCandidate {
    /// Does this candidate carry everything needed to apply it safely? Entries
    /// written by older versions stored a whole file with no repository and no
    /// pre-image; they deserialize with those fields empty and are discarded,
    /// because nothing bounds what they would overwrite.
    pub fn is_well_formed(&self) -> bool {
        !self.repo.is_empty()
            && !self.rel_path.is_empty()
            && is_sha_hex(&self.pre_sha256, 64)
            && is_sha_hex(&self.sha256, 64)
    }

    /// Same fix: same repository, file, pre-image, post-image and diff.
    fn same_fix(&self, other: &PatchCandidate) -> bool {
        self.repo == other.repo
            && self.rel_path == other.rel_path
            && self.pre_sha256 == other.pre_sha256
            && self.sha256 == other.sha256
            && self.diff == other.diff
    }

    fn is_for(&self, scope: &RepoScope) -> bool {
        self.repo == scope.repo && self.rel_path == scope.rel_path
    }
}

/// A fleet-shareable store of verified patches, keyed by failure fingerprint
/// (`heal_memory::fingerprint(command, args)`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatchMemory {
    #[serde(default)]
    pub by_fingerprint: BTreeMap<String, Vec<PatchCandidate>>,
}

/// Summary of a peer merge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchMergeReport {
    pub new_candidates: usize,
    pub reinforced: usize,
    /// Set by [`PatchMemory::merge_signed_once`] when this exact export was
    /// merged before, or is one this node handed out itself. Nothing was
    /// changed and the other counts are zero.
    pub already_merged: bool,
    /// Candidates dropped during the merge: rejected by the
    /// [`FleetTrustPolicy`] (oversized span, per-fingerprint flood cap),
    /// repeated inside the export, or not [`PatchCandidate::is_well_formed`]
    /// (e.g. the whole-file format of older versions).
    pub byzantine_rejected: usize,
}

/// Byzantine-robustness policy for folding in a peer's patch export.
///
/// Provenance ([`crate::provenance`]) proves an export came from a key-holder and
/// was not tampered in transit — but a *malicious key-holder* can still try to
/// (a) inflate `verified_count` to force a bad-but-locally-verifiable patch to
/// the top of the try-order, or (b) flood the store with junk candidates. The
/// local re-verify gate ([`PatchMemory::try_candidates`]) already stops a wrong
/// patch from ever being *applied*; this policy additionally bounds how much
/// *trust and space* any single peer merge can consume, inspired by FLTrust-style
/// bounded-contribution aggregation.
#[derive(Debug, Clone, Copy)]
pub struct FleetTrustPolicy {
    /// Max `verified_count` a single merge may add to any one candidate. Counts
    /// are meant to reflect *independent* confirmations, so one peer export
    /// should move the needle by a bounded amount (robust default: `1`).
    pub max_verified_bump_per_candidate: u32,
    /// Max brand-new candidates a single merge may add per fingerprint (flood
    /// cap). Excess new candidates are rejected.
    pub max_new_candidates_per_fingerprint: usize,
    /// Reject any candidate whose changed span (`diff.replacement`) exceeds this
    /// many bytes.
    pub max_content_bytes: usize,
}

impl FleetTrustPolicy {
    /// The recommended Byzantine-robust policy: one peer = at most +1 trust per
    /// candidate, at most 8 new candidates per fingerprint per merge, 8 MiB cap.
    pub fn robust() -> Self {
        Self {
            max_verified_bump_per_candidate: 1,
            max_new_candidates_per_fingerprint: 8,
            max_content_bytes: 8 * 1024 * 1024,
        }
    }

    /// The legacy permissive policy (no bounds) — preserves the original
    /// `merge_signed` behavior for callers that have their own trust model.
    pub fn permissive() -> Self {
        Self {
            max_verified_bump_per_candidate: u32::MAX,
            max_new_candidates_per_fingerprint: usize::MAX,
            max_content_bytes: usize::MAX,
        }
    }
}

impl Default for FleetTrustPolicy {
    fn default() -> Self {
        Self::robust()
    }
}

impl PatchMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Default on-disk location, alongside the heal memory (`axiom_patch_memory.json`).
    pub fn default_path() -> Option<std::path::PathBuf> {
        crate::heal_memory::HealMemory::default_path()
            .map(|p| p.with_file_name("axiom_patch_memory.json"))
    }

    /// Load from disk; a missing/corrupt file yields an empty store. Entries
    /// that are not [`PatchCandidate::is_well_formed`] (including every entry
    /// written by a version that stored whole files) are dropped.
    pub fn load(path: impl AsRef<Path>) -> Self {
        let mut store: Self = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        store.drop_malformed();
        store
    }

    fn drop_malformed(&mut self) {
        self.by_fingerprint.retain(|_, candidates| {
            candidates.retain(PatchCandidate::is_well_formed);
            !candidates.is_empty()
        });
    }

    /// Persist to disk (pretty JSON).
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).expect("serialize patch memory");
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, json)
    }

    /// Record a locally-verified fix: in `scope`, the file whose content was
    /// `pre` verified green once it read `post`. Stored as the diff between the
    /// two, keyed by the repository, the path and the pre-image hash. If the
    /// identical fix is already stored for this fingerprint its
    /// `verified_count` is bumped; otherwise it is added. Candidates are kept
    /// ranked by `verified_count` (desc). A no-op "fix" (`pre == post`) is not
    /// recorded.
    pub fn record_verified(&mut self, fingerprint: &str, scope: &RepoScope, pre: &str, post: &str) {
        if pre == post {
            return;
        }
        let candidate = PatchCandidate {
            repo: scope.repo.clone(),
            rel_path: scope.rel_path.clone(),
            pre_sha256: sha256_hex(pre.as_bytes()),
            diff: PatchDiff::between(pre, post),
            sha256: sha256_hex(post.as_bytes()),
            verified_count: 1,
        };
        let list = self
            .by_fingerprint
            .entry(fingerprint.to_string())
            .or_default();
        if let Some(existing) = list.iter_mut().find(|c| c.same_fix(&candidate)) {
            existing.verified_count = existing.verified_count.saturating_add(1);
        } else {
            list.push(candidate);
        }
        list.sort_by_key(|c| std::cmp::Reverse(c.verified_count));
    }

    /// Candidate patches for a fingerprint, ranked most-verified first.
    pub fn candidates(&self, fingerprint: &str) -> &[PatchCandidate] {
        self.by_fingerprint
            .get(fingerprint)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Total candidates across all fingerprints.
    pub fn len(&self) -> usize {
        self.by_fingerprint.values().map(|v| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The stored fixes that may be applied to a file in `scope` whose current
    /// content is `current`: same repository, same path, and a pre-image hash
    /// equal to the hash of `current`. Each is returned with the post-image it
    /// produces; a candidate whose diff does not reproduce its recorded
    /// post-image hash (corrupt or forged) is left out.
    fn applicable<'a>(
        &'a self,
        fingerprint: &str,
        scope: &RepoScope,
        current: &str,
    ) -> Vec<(&'a PatchCandidate, String)> {
        let current_sha = sha256_hex(current.as_bytes());
        self.candidates(fingerprint)
            .iter()
            .filter(|c| c.is_well_formed() && c.is_for(scope) && c.pre_sha256 == current_sha)
            .filter_map(|c| {
                let post = c.diff.apply(current)?;
                (sha256_hex(post.as_bytes()) == c.sha256 && post != current).then_some((c, post))
            })
            .collect()
    }

    /// Try the stored fixes for `fingerprint` that were recorded against the
    /// exact current content of `source_path` (in `scope`), keeping the FIRST
    /// that makes `verify()` return true. Each is written then verified; one
    /// that does not verify green is rolled back byte-for-byte before the next
    /// is tried. Returns the post-image sha of the applied fix, or `None` if
    /// none applied (source left untouched).
    ///
    /// Two guards make this safe to run on a user's tree:
    /// * **pre-image match** — a fix is only considered when `source_path`
    ///   currently hashes to the fix's pre-image, in the same repository and at
    ///   the same path. Anything that has changed since the fix was recorded —
    ///   later edits, uncommitted work, another project's same-named file — is
    ///   not a match and is never written to.
    /// * **local verification** — a matching fix is kept only if *this node's*
    ///   `verify` confirms it.
    pub fn try_candidates<V>(
        &self,
        fingerprint: &str,
        scope: &RepoScope,
        source_path: &Path,
        mut verify: V,
    ) -> Option<String>
    where
        V: FnMut() -> bool,
    {
        let original = std::fs::read_to_string(source_path).ok()?;
        let matching = self.applicable(fingerprint, scope, &original);
        if matching.is_empty() {
            return None;
        }

        // B.1 per-bug test-time adaptation: try the matching fixes in
        // self-consistency-reranked order (consensus across alpha-rename /
        // whitespace-equivalent fixes first, weighted by verified_count) rather
        // than raw verified_count order. This only changes *which candidate the
        // verifier tries first* — the verify gate below is unchanged.
        let adapter_cands: Vec<crate::test_time_adapter::Candidate> = matching
            .iter()
            .map(|(c, post)| {
                crate::test_time_adapter::Candidate::new(post.clone(), c.verified_count as f32)
            })
            .collect();
        let order = crate::test_time_adapter::rerank_by_self_consistency(&adapter_cands);

        for &i in &order {
            let (c, post) = &matching[i];
            if std::fs::write(source_path, post).is_err() {
                let _ = std::fs::write(source_path, &original);
                continue;
            }
            if verify() {
                return Some(c.sha256.clone());
            }
            // Rejected locally → roll back byte-for-byte; never trust unverified.
            if std::fs::write(source_path, &original).is_err() {
                eprintln!(
                    "[axiom-patch] WARNING: failed to restore original after rejected candidate"
                );
            }
        }
        None
    }

    /// Candidate patches for `fingerprint` recorded for `scope`, ordered by the
    /// B.1 per-bug test-time adapter (self-consistency voting over the changed
    /// lines, weighted by `verified_count`). Exposed for callers that want the
    /// adapter's ranking without driving the verify loop; unlike
    /// [`Self::try_candidates`] it does not filter by pre-image.
    pub fn candidates_reranked(&self, fingerprint: &str, scope: &RepoScope) -> Vec<PatchCandidate> {
        let matching: Vec<&PatchCandidate> = self
            .candidates(fingerprint)
            .iter()
            .filter(|c| c.is_for(scope))
            .collect();
        let adapter_cands: Vec<crate::test_time_adapter::Candidate> = matching
            .iter()
            .map(|c| {
                crate::test_time_adapter::Candidate::new(
                    c.diff.replacement.clone(),
                    c.verified_count as f32,
                )
            })
            .collect();
        crate::test_time_adapter::rerank_by_self_consistency(&adapter_cands)
            .into_iter()
            .map(|i| matching[i].clone())
            .collect()
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).expect("serialize patch memory")
    }

    /// Export this store as a provenance-signed payload (SHA-256 + optional HMAC
    /// when `fleet_key` is set), ready to gossip to peers.
    pub fn export_signed(&self, fleet_key: Option<&[u8]>) -> SignedExport {
        sign_export(&self.to_json(), fleet_key)
    }

    /// Verify a peer's signed export and fold its candidates in. New patches are
    /// added; identical ones accumulate `verified_count`.
    /// Rejects the merge unless this node holds a fleet key and the export's
    /// HMAC verifies under it (see
    /// [`crate::provenance::verify_export_authenticated`]): with `fleet_key ==
    /// None` every export is refused. Merging only stores candidates, but a
    /// stored candidate is later written into a source tree and executed by the
    /// repair loop's verify run, so the store must never hold unauthenticated
    /// input.
    ///
    /// Uses the legacy [`FleetTrustPolicy::permissive`] (no contribution bounds)
    /// for backward compatibility. Prefer [`Self::merge_signed_guarded`] with
    /// [`FleetTrustPolicy::robust`] in fleets that may include Byzantine peers.
    pub fn merge_signed(
        &mut self,
        export: &SignedExport,
        fleet_key: Option<&[u8]>,
    ) -> Result<PatchMergeReport, ProvenanceError> {
        self.merge_signed_guarded(export, fleet_key, FleetTrustPolicy::permissive())
    }

    /// Byzantine-robust merge: authenticate the export (fleet key required, HMAC
    /// must verify), then fold the peer's candidates in subject to `policy` —
    /// bounding how much `verified_count` trust and how many new candidates a
    /// single peer export can contribute, and dropping oversized or malformed
    /// candidates. The pre-image match and the local re-verify gate still
    /// decide what is ever written or kept; this only bounds a malicious
    /// key-holder's influence on ranking and store size.
    pub fn merge_signed_guarded(
        &mut self,
        export: &SignedExport,
        fleet_key: Option<&[u8]>,
        policy: FleetTrustPolicy,
    ) -> Result<PatchMergeReport, ProvenanceError> {
        let payload = verify_export_authenticated(export, fleet_key)?;
        self.fold_in(payload, policy)
    }

    /// [`Self::merge_signed_guarded`], counting an export once.
    ///
    /// An export is one statement by one peer. Delivering it again (a retry,
    /// a sync job on a timer, a replay by anyone who captured it, or this
    /// node's own export sent back) must add nothing, or a count that is
    /// meant to reflect independent confirmations can be raised at will.
    /// `ledger` is the store's [`ExportLedger`]: an authenticated export that
    /// is already in it is reported as
    /// [`PatchMergeReport::already_merged`] and the store is left untouched;
    /// a new one is merged and recorded. The caller persists both with
    /// [`ExportLedger::save_around`]. A *different* export from a key-holder
    /// still counts.
    pub fn merge_signed_once(
        &mut self,
        export: &SignedExport,
        fleet_key: Option<&[u8]>,
        policy: FleetTrustPolicy,
        ledger: &mut ExportLedger,
    ) -> Result<PatchMergeReport, ProvenanceError> {
        // Authenticate first: the ledger answers nothing about an export
        // this node would refuse. The digest is taken from the payload.
        let payload = verify_export_authenticated(export, fleet_key)?;
        if ledger.contains(payload) {
            return Ok(PatchMergeReport {
                new_candidates: 0,
                reinforced: 0,
                already_merged: true,
                byzantine_rejected: 0,
            });
        }
        let report = self.fold_in(payload, policy)?;
        ledger.remember(payload);
        Ok(report)
    }

    /// Fold an authenticated export payload in, subject to `policy`.
    fn fold_in(
        &mut self,
        payload: &str,
        policy: FleetTrustPolicy,
    ) -> Result<PatchMergeReport, ProvenanceError> {
        let peer: PatchMemory =
            serde_json::from_str(payload).map_err(|_| ProvenanceError::BadSchema)?;
        let mut report = PatchMergeReport {
            new_candidates: 0,
            reinforced: 0,
            already_merged: false,
            byzantine_rejected: 0,
        };
        for (fp, peer_list) in peer.by_fingerprint {
            // Work on this fingerprint's list detached from the map, so a
            // fingerprint for which nothing is kept leaves no empty entry.
            let mut local = self.by_fingerprint.remove(&fp).unwrap_or_default();
            let known = local.len();
            let mut new_this_fp = 0usize;
            // Already-known fixes this export has reinforced. An export is one
            // peer's word, so each fix in it counts once however often the
            // export repeats it.
            let mut reinforced_here: HashSet<usize> = HashSet::new();
            for mut pc in peer_list {
                // Drop candidates that cannot be applied safely (no repository
                // or pre-image: the whole-file format of older versions) and
                // implausibly large ones.
                if !pc.is_well_formed() || pc.diff.replacement.len() > policy.max_content_bytes {
                    report.byzantine_rejected += 1;
                    continue;
                }
                // The peer's post-image hash cannot be recomputed here (this
                // node may not hold the pre-image). It is only a dedup key: a
                // fix is applied solely to a file hashing to `pre_sha256`, and
                // is discarded unless the result hashes to `sha256`.
                match local.iter().position(|c| c.same_fix(&pc)) {
                    // Already known before this export: reinforce, bounded. One
                    // export should count as at most a few independent
                    // confirmations.
                    Some(i) if i < known && reinforced_here.insert(i) => {
                        let bump = pc
                            .verified_count
                            .min(policy.max_verified_bump_per_candidate);
                        local[i].verified_count = local[i].verified_count.saturating_add(bump);
                        report.reinforced += 1;
                    }
                    // The same fix repeated inside one export counts once,
                    // whether it was new to this node or already known.
                    Some(_) => report.byzantine_rejected += 1,
                    None => {
                        // Flood cap: reject excess brand-new candidates per fingerprint.
                        if new_this_fp >= policy.max_new_candidates_per_fingerprint {
                            report.byzantine_rejected += 1;
                            continue;
                        }
                        // Also clamp the incoming count of a brand-new candidate so a
                        // peer cannot seed a fresh patch at the top of the ranking.
                        pc.verified_count = pc
                            .verified_count
                            .min(policy.max_verified_bump_per_candidate);
                        local.push(pc);
                        new_this_fp += 1;
                        report.new_candidates += 1;
                    }
                }
            }
            if !local.is_empty() {
                local.sort_by_key(|c| std::cmp::Reverse(c.verified_count));
                self.by_fingerprint.insert(fp, local);
            }
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(rel_path: &str) -> RepoScope {
        RepoScope::new("git:0123456789abcdef0123456789abcdef01234567", rel_path)
    }

    fn tmp_file(tag: &str, content: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("axiom_patchmem_{tag}_{nanos}.txt"));
        std::fs::write(&path, content).unwrap();
        path
    }

    // --- PatchDiff ----------------------------------------------------------

    #[test]
    fn diff_roundtrips_and_keeps_only_the_changed_lines() {
        let cases: &[(&str, &str)] = &[
            // one token on a middle line
            ("a\nlet x = 1 + 1;\nc\n", "a\nlet x = 1;\nc\n"),
            // line appended, line prepended, line removed
            ("a\nb\n", "a\nb\nc\n"),
            ("b\nc\n", "a\nb\nc\n"),
            ("a\nb\nc\n", "a\nc\n"),
            // no trailing newline, on either side
            ("a\nb", "a\nB"),
            ("a\nb\n", "a\nb"),
            // CRLF line endings are preserved byte-for-byte
            ("a\r\nb = 1\r\nc\r\n", "a\r\nb = 2\r\nc\r\n"),
            // multi-byte characters around and inside the change
            ("héllo\nwörld — ok\nend\n", "héllo\nwörld – ok\nend\n"),
            ("日本\n語\n", "日本\n語!\n"),
            // repeated lines: the common prefix and suffix must not overlap
            ("x\nx\nx\n", "x\nx\n"),
            ("x\nx\n", "x\nx\nx\n"),
            // empty on either side
            ("", "new file\n"),
            ("gone\n", ""),
        ];
        for (pre, post) in cases {
            let diff = PatchDiff::between(pre, post);
            assert_eq!(
                diff.apply(pre).as_deref(),
                Some(*post),
                "roundtrip failed for {pre:?} -> {post:?} ({diff:?})"
            );
            assert!(diff.prefix_bytes + diff.suffix_bytes <= pre.len());
        }
        // The stored span is the changed line, not the file.
        let diff = PatchDiff::between("a\nlet x = 1 + 1;\nc\n", "a\nlet x = 1;\nc\n");
        assert_eq!(diff.replacement, "let x = 1;\n");
        assert_eq!(diff.prefix_bytes, 2);
        assert_eq!(diff.suffix_bytes, 2);
    }

    #[test]
    fn diff_does_not_apply_to_a_file_it_does_not_fit() {
        let diff = PatchDiff::between("0123456789\nabc\n", "0123456789\nABC\n");
        // Shorter than the kept prefix + suffix.
        assert_eq!(diff.apply("short"), None);
        // A cut that would land inside a multi-byte character.
        let mid_char = PatchDiff {
            prefix_bytes: 1,
            suffix_bytes: 0,
            replacement: String::new(),
        };
        assert_eq!(mid_char.apply("é"), None);
        // Lengths that overflow.
        let huge = PatchDiff {
            prefix_bytes: usize::MAX,
            suffix_bytes: 2,
            replacement: String::new(),
        };
        assert_eq!(huge.apply("abc"), None);
    }

    // --- recording ----------------------------------------------------------

    #[test]
    fn record_dedups_identical_fixes_and_ranks_by_verified_count() {
        let mut m = PatchMemory::new();
        let s = scope("src/a.rs");
        m.record_verified("fp1", &s, "BROKEN\n", "FIX_A\n");
        m.record_verified("fp1", &s, "BROKEN\n", "FIX_A\n"); // identical → reinforce
        m.record_verified("fp1", &s, "BROKEN\n", "FIX_B\n"); // different → new candidate
        let cands = m.candidates("fp1");
        assert_eq!(cands.len(), 2, "two distinct candidates");
        assert_eq!(
            cands[0].diff.replacement, "FIX_A\n",
            "most-verified ranked first"
        );
        assert_eq!(cands[0].verified_count, 2);
        assert_eq!(cands[1].verified_count, 1);
        assert_eq!(m.candidates("nope").len(), 0);
    }

    #[test]
    fn the_same_edit_against_a_different_pre_image_or_repo_is_a_different_fix() {
        let mut m = PatchMemory::new();
        m.record_verified("fp", &scope("a.rs"), "x = 1 + 1\n", "x = 1\n");
        // Same resulting line, but made against a file with other content.
        m.record_verified("fp", &scope("a.rs"), "x = 1 + 1\ny = 2\n", "x = 1\ny = 2\n");
        // Same edit in another repository.
        m.record_verified(
            "fp",
            &RepoScope::new("path:other", "a.rs"),
            "x = 1 + 1\n",
            "x = 1\n",
        );
        assert_eq!(m.candidates("fp").len(), 3);
        assert!(m.candidates("fp").iter().all(|c| c.verified_count == 1));
    }

    #[test]
    fn a_no_op_fix_is_not_recorded() {
        let mut m = PatchMemory::new();
        m.record_verified("fp", &scope("a.rs"), "same\n", "same\n");
        assert!(m.is_empty());
    }

    #[test]
    fn a_stored_fix_holds_the_changed_span_not_the_file() {
        let pre: String = (0..200).map(|i| format!("line {i}\n")).collect();
        let post = pre.replace("line 100\n", "line one hundred\n");
        let mut m = PatchMemory::new();
        m.record_verified("fp", &scope("big.txt"), &pre, &post);
        let c = &m.candidates("fp")[0];
        assert_eq!(c.diff.replacement, "line one hundred\n");
        assert_eq!(c.pre_sha256, sha256_hex(pre.as_bytes()));
        assert_eq!(c.sha256, sha256_hex(post.as_bytes()));
        let json = m.to_json();
        assert!(!json.contains("line 42"), "unchanged lines are not stored");
    }

    #[test]
    fn json_roundtrip_preserves_store() {
        let mut m = PatchMemory::new();
        m.record_verified("fp", &scope("x.rs"), "BAD\n", "GOOD\n");
        let json = m.to_json();
        let back: PatchMemory = serde_json::from_str(&json).unwrap();
        assert_eq!(back.candidates("fp"), m.candidates("fp"));
    }

    #[test]
    fn whole_file_entries_from_older_versions_are_dropped_on_load() {
        // The pre-diff format: a full file, keyed by path only. Nothing bounds
        // what such an entry would overwrite, so it must never be applied.
        let legacy = serde_json::json!({
            "by_fingerprint": {
                "fp-legacy": [{
                    "rel_path": "calc.py",
                    "content": "def average(xs):\n    return sum(xs) / len(xs)\n",
                    "sha256": sha256_hex(b"def average(xs):\n    return sum(xs) / len(xs)\n"),
                    "verified_count": 3
                }]
            }
        });
        let path = tmp_file("legacy", &legacy.to_string());
        let store = PatchMemory::load(&path);
        assert!(store.is_empty(), "legacy whole-file entries are discarded");
        assert!(
            store.by_fingerprint.is_empty(),
            "no empty fingerprint left behind"
        );

        // A store mixing both keeps only the well-formed entry.
        let mut mixed = PatchMemory::new();
        mixed.record_verified("fp-new", &scope("a.rs"), "BAD\n", "GOOD\n");
        let mut value: serde_json::Value = serde_json::from_str(&mixed.to_json()).unwrap();
        value["by_fingerprint"]["fp-legacy"] = legacy["by_fingerprint"]["fp-legacy"].clone();
        std::fs::write(&path, value.to_string()).unwrap();
        let store = PatchMemory::load(&path);
        assert_eq!(store.len(), 1);
        assert_eq!(store.candidates("fp-new").len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    // --- applying -----------------------------------------------------------

    #[test]
    fn try_candidates_applies_only_a_locally_verified_patch() {
        let path = tmp_file("verify", "BROKEN\n");
        let s = scope("f.txt");
        let mut m = PatchMemory::new();
        m.record_verified("fp", &s, "BROKEN\n", "WRONG\n"); // won't verify
        m.record_verified("fp", &s, "BROKEN\n", "FIXED\n"); // will verify
        m.record_verified("fp", &s, "BROKEN\n", "FIXED\n");

        // verify() = the file currently contains FIXED.
        let p = path.clone();
        let applied = m.try_candidates("fp", &s, &path, || {
            std::fs::read_to_string(&p)
                .map(|s| s.contains("FIXED"))
                .unwrap_or(false)
        });
        assert_eq!(
            applied,
            Some(sha256_hex(b"FIXED\n")),
            "the locally-verifying patch is applied"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "FIXED\n");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn try_candidates_leaves_source_untouched_when_none_verify() {
        let path = tmp_file("none", "ORIGINAL\n");
        let s = scope("f.txt");
        let mut m = PatchMemory::new();
        m.record_verified("fp", &s, "ORIGINAL\n", "ALSO_WRONG\n");
        // verify() always false → no candidate accepted.
        let mut tried = 0;
        let applied = m.try_candidates("fp", &s, &path, || {
            tried += 1;
            false
        });
        assert!(applied.is_none());
        assert_eq!(tried, 1, "the matching candidate was tried, then rejected");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "ORIGINAL\n",
            "a never-verifying patch must leave the source byte-for-byte intact"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn try_candidates_never_cross_applies_to_a_different_rel_path() {
        // A patch recorded for `src/a.rs` must never be written to `src/b.rs`,
        // even when they share a fingerprint, identical content, and verify()
        // would pass.
        let path = tmp_file("xfile", "SAME_BYTES\n");
        let mut m = PatchMemory::new();
        m.record_verified("fp", &scope("src/a.rs"), "SAME_BYTES\n", "FIX_FOR_A\n");

        let mut verify_ran = false;
        let applied = m.try_candidates("fp", &scope("src/b.rs"), &path, || {
            verify_ran = true;
            true
        });
        assert!(
            applied.is_none(),
            "a patch recorded for a.rs is not applied to b.rs"
        );
        assert!(!verify_ran, "nothing was written, so nothing was verified");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "SAME_BYTES\n");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn try_candidates_refuses_a_file_that_changed_since_the_fix_was_recorded() {
        // The fix was recorded against OLD. The file now holds newer work.
        // Even a verify() that would accept anything must not get the chance:
        // the stored fix does not describe this file.
        let old = "fn average() { sum / (n + 1) }\n";
        let fixed = "fn average() { sum / n }\n";
        let newer = "fn average() { sum / 30 }\n\nfn median() { /* uncommitted */ }\n";
        let s = scope("src/stats.rs");
        let mut m = PatchMemory::new();
        m.record_verified("fp", &s, old, fixed);

        let path = tmp_file("newer", newer);
        let mut verify_ran = false;
        let applied = m.try_candidates("fp", &s, &path, || {
            verify_ran = true;
            true
        });
        assert!(applied.is_none(), "pre-image mismatch: not applied");
        assert!(
            !verify_ran,
            "the file was never written, so verify never ran"
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            newer,
            "newer work must survive byte-for-byte"
        );

        // Put the file back to the exact pre-image and the fix applies again.
        std::fs::write(&path, old).unwrap();
        assert!(m.try_candidates("fp", &s, &path, || true).is_some());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), fixed);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn try_candidates_refuses_another_repositorys_file_even_if_byte_identical() {
        let pre = "BROKEN\n";
        let mut m = PatchMemory::new();
        m.record_verified(
            "fp",
            &RepoScope::new("path:repo-one", "calc.py"),
            pre,
            "FIXED\n",
        );

        let path = tmp_file("otherrepo", pre);
        let mut verify_ran = false;
        let applied = m.try_candidates(
            "fp",
            &RepoScope::new("path:repo-two", "calc.py"),
            &path,
            || {
                verify_ran = true;
                true
            },
        );
        assert!(applied.is_none(), "different repository: not applied");
        assert!(!verify_ran);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), pre);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn try_candidates_discards_a_diff_that_does_not_reproduce_its_post_image() {
        // A forged or corrupt candidate: the right repository, path and
        // pre-image, but a diff whose result is not what `sha256` promises.
        let pre = "line one\nline two\n";
        let s = scope("f.txt");
        let mut m = PatchMemory::new();
        m.record_verified("fp", &s, pre, "line one\nline 2\n");
        m.by_fingerprint.get_mut("fp").unwrap()[0].diff.replacement =
            "import os; os.system('evil')\n".into();

        let path = tmp_file("forged", pre);
        let mut verify_ran = false;
        let applied = m.try_candidates("fp", &s, &path, || {
            verify_ran = true;
            true
        });
        assert!(applied.is_none());
        assert!(
            !verify_ran,
            "a diff that fails its hash check is never written"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), pre);
        let _ = std::fs::remove_file(&path);
    }

    // --- repository identity --------------------------------------------------

    fn tmp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("axiom_reposcope_{tag}_{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn repo_scope_without_git_is_the_anchor_directory() {
        let base = tmp_dir("anchor");
        let one = base.join("one");
        let two = base.join("two");
        for dir in [&one, &two] {
            std::fs::create_dir_all(dir.join("src")).unwrap();
            std::fs::write(dir.join("src").join("calc.py"), "x\n").unwrap();
        }
        let a = repo_scope(&one.join("src").join("calc.py"), Some(&one)).unwrap();
        let b = repo_scope(&two.join("src").join("calc.py"), Some(&two)).unwrap();
        assert_eq!(a.rel_path, "src/calc.py");
        assert_eq!(b.rel_path, "src/calc.py");
        assert!(a.repo.starts_with("path:"), "{}", a.repo);
        assert_ne!(a.repo, b.repo, "two directories are two repositories");
        // Stable: the same file resolves to the same scope every time.
        assert_eq!(
            a,
            repo_scope(&one.join("src").join("calc.py"), Some(&one)).unwrap()
        );
        // A file outside the anchor has no scope, and neither has a missing one.
        assert_eq!(
            repo_scope(&two.join("src").join("calc.py"), Some(&one)),
            None
        );
        assert_eq!(
            repo_scope(&one.join("src").join("nope.py"), Some(&one)),
            None
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn repo_scope_uses_the_enclosing_git_checkout_over_the_anchor() {
        // A `.git` entry with no commits behind it: the root is still the
        // checkout (so the path is repository-relative), and the identity
        // falls back to the directory because there is no root commit.
        let base = tmp_dir("gitroot");
        std::fs::create_dir_all(base.join(".git")).unwrap();
        std::fs::create_dir_all(base.join("pkg").join("src")).unwrap();
        let file = base.join("pkg").join("src").join("lib.rs");
        std::fs::write(&file, "x\n").unwrap();
        // The anchor is a subdirectory, as when `axiom solve` runs in `pkg/`.
        let scope = repo_scope(&file, Some(&base.join("pkg"))).unwrap();
        assert_eq!(scope.rel_path, "pkg/src/lib.rs");
        assert!(scope.repo.starts_with("path:"), "{}", scope.repo);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn repo_scope_of_a_git_repository_is_its_root_commit() {
        let git = |dir: &Path, args: &[&str]| {
            std::process::Command::new("git")
                .arg("-C")
                .arg(dir)
                .args([
                    "-c",
                    "user.name=axiom-test",
                    "-c",
                    "user.email=axiom-test@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                ])
                .args(args)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        };
        let base = tmp_dir("gitcommit");
        let origin = base.join("origin");
        std::fs::create_dir_all(&origin).unwrap();
        std::fs::write(origin.join("calc.py"), "x\n").unwrap();
        assert!(git(&origin, &["init", "-q"]), "this test needs git on PATH");
        assert!(git(&origin, &["add", "calc.py"]));
        assert!(git(&origin, &["commit", "-q", "-m", "root"]));

        let a = repo_scope(&origin.join("calc.py"), None).unwrap();
        assert!(a.repo.starts_with("git:"), "{}", a.repo);
        assert_eq!(a.repo.len(), "git:".len() + 40);
        assert_eq!(a.rel_path, "calc.py");

        // A clone of the same repository has the same identity...
        let clone = base.join("clone");
        assert!(git(
            &base,
            &[
                "clone",
                "-q",
                origin.to_str().unwrap(),
                clone.to_str().unwrap()
            ]
        ));
        let b = repo_scope(&clone.join("calc.py"), None).unwrap();
        assert_eq!(a, b, "clones of one repository share an identity");

        // ...and an unrelated repository does not.
        let unrelated = base.join("unrelated");
        std::fs::create_dir_all(&unrelated).unwrap();
        std::fs::write(unrelated.join("calc.py"), "y\n").unwrap();
        assert!(git(&unrelated, &["init", "-q"]));
        assert!(git(&unrelated, &["add", "calc.py"]));
        assert!(git(&unrelated, &["commit", "-q", "-m", "another root"]));
        let c = repo_scope(&unrelated.join("calc.py"), None).unwrap();
        assert_ne!(a.repo, c.repo);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn repo_scope_of_a_shallow_checkout_does_not_move_when_history_is_deepened() {
        let git = |dir: &Path, args: &[&str]| {
            std::process::Command::new("git")
                .arg("-C")
                .arg(dir)
                .args([
                    "-c",
                    "user.name=axiom-test",
                    "-c",
                    "user.email=axiom-test@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                ])
                .args(args)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        };
        let base = tmp_dir("gitshallow");
        let origin = base.join("origin");
        std::fs::create_dir_all(&origin).unwrap();
        assert!(git(&origin, &["init", "-q"]), "this test needs git on PATH");
        for n in 1..=3 {
            std::fs::write(origin.join("calc.py"), format!("x = {n}\n")).unwrap();
            assert!(git(&origin, &["add", "calc.py"]));
            assert!(git(&origin, &["commit", "-q", "-m", "change"]));
        }
        let full = repo_scope(&origin.join("calc.py"), None).unwrap();
        assert!(full.repo.starts_with("git:"), "{}", full.repo);

        // `file://` so that git honours `--depth` for a local clone.
        let shallow = base.join("shallow");
        let url = format!("file://{}", origin.display());
        assert!(git(
            &base,
            &[
                "clone",
                "-q",
                "--depth",
                "1",
                &url,
                shallow.to_str().unwrap()
            ]
        ));
        let before = repo_scope(&shallow.join("calc.py"), None).unwrap();
        // Git reports the shallow boundary commit as parentless. It is not the
        // root commit and it moves, so it must not be used as the identity.
        assert!(
            before.repo.starts_with("path:"),
            "a shallow checkout is identified by its directory: {}",
            before.repo
        );

        // Fetch one more commit of history: the boundary moves, the file does not.
        assert!(git(&shallow, &["fetch", "-q", "--deepen", "1"]));
        let after = repo_scope(&shallow.join("calc.py"), None).unwrap();
        assert_eq!(
            before, after,
            "deepening a shallow checkout must not change its identity"
        );

        // A fix recorded before deepening is still found afterwards.
        let mut pm = PatchMemory::new();
        pm.record_verified("fp", &before, "x = 3\n", "x = 4\n");
        assert_eq!(pm.candidates_reranked("fp", &after).len(), 1);

        // With its history complete the checkout holds the root commit, and
        // is identified like any other clone of the repository.
        assert!(git(&shallow, &["fetch", "-q", "--unshallow"]));
        let complete = repo_scope(&shallow.join("calc.py"), None).unwrap();
        assert_eq!(complete, full);
        let _ = std::fs::remove_dir_all(&base);
    }

    // --- fleet merge ----------------------------------------------------------

    /// A well-formed candidate as a peer would export it.
    fn peer_candidate(
        rel_path: &str,
        pre: &str,
        post: &str,
        verified_count: u32,
    ) -> PatchCandidate {
        PatchCandidate {
            repo: scope(rel_path).repo,
            rel_path: rel_path.into(),
            pre_sha256: sha256_hex(pre.as_bytes()),
            diff: PatchDiff::between(pre, post),
            sha256: sha256_hex(post.as_bytes()),
            verified_count,
        }
    }

    #[test]
    fn signed_export_merge_roundtrip_dedups_and_sums() {
        let key: &[u8] = b"fleet-secret";
        let mut a = PatchMemory::new();
        a.record_verified("fp", &scope("x.rs"), "BAD\n", "GOOD\n");
        let export = a.export_signed(Some(key));

        let mut b = PatchMemory::new();
        b.record_verified("fp", &scope("x.rs"), "BAD\n", "GOOD\n"); // same patch, independently verified
        b.record_verified("fp2", &scope("y.rs"), "BAD\n", "OTHER\n"); // unrelated
        let rep = b.merge_signed(&export, Some(key)).unwrap();
        assert_eq!(rep.reinforced, 1, "identical patch reinforced");
        assert_eq!(rep.new_candidates, 0);
        // The shared patch now has verified_count 2 (1 local + 1 from peer).
        assert_eq!(b.candidates("fp")[0].verified_count, 2);
        assert_eq!(b.candidates("fp2").len(), 1, "unrelated local patch intact");
    }

    #[test]
    fn merge_refuses_every_export_when_no_fleet_key_is_configured() {
        // A stored candidate is later written to disk and run by the repair
        // loop, so a node with no fleet key must not accept one from anybody:
        // not a hash-only export, and not one signed with a key it cannot check.
        let mut a = PatchMemory::new();
        a.record_verified("fp", &scope("x.rs"), "BAD\n", "ATTACKER CONTROLLED\n");
        for export in [a.export_signed(None), a.export_signed(Some(b"unknown"))] {
            for policy in [FleetTrustPolicy::robust(), FleetTrustPolicy::permissive()] {
                let mut local = PatchMemory::new();
                assert_eq!(
                    local
                        .merge_signed_guarded(&export, None, policy)
                        .unwrap_err(),
                    ProvenanceError::NoFleetKey
                );
                assert!(local.is_empty(), "a refused merge stores nothing");
            }
        }
        // And an unsigned export is refused even when this node has a key.
        let mut local = PatchMemory::new();
        assert_eq!(
            local
                .merge_signed(&a.export_signed(None), Some(b"fleet"))
                .unwrap_err(),
            ProvenanceError::SignatureMissing
        );
        assert!(local.is_empty());
    }

    #[test]
    fn merge_rejects_tampered_or_wrong_key() {
        let mut a = PatchMemory::new();
        a.record_verified("fp", &scope("x.rs"), "BAD\n", "GOOD\n");
        // Tampered payload → hash mismatch, even with the right key.
        let mut bad = a.export_signed(Some(b"right"));
        bad.payload.push_str("tampered");
        assert_eq!(
            PatchMemory::new()
                .merge_signed(&bad, Some(b"right"))
                .unwrap_err(),
            ProvenanceError::HashMismatch
        );
        // Wrong key.
        let signed = a.export_signed(Some(b"right"));
        assert!(PatchMemory::new()
            .merge_signed(&signed, Some(b"wrong"))
            .is_err());
    }

    #[test]
    fn merge_drops_whole_file_and_malformed_candidates_from_a_key_holder() {
        // Authenticated is not the same as safe: a peer still on the
        // whole-file format, or one sending junk hashes, contributes nothing.
        let key: &[u8] = b"fleet";
        let good = peer_candidate("a.rs", "BAD\n", "GOOD\n", 1);
        let payload = serde_json::json!({
            "by_fingerprint": {
                "fp-legacy": [{
                    "rel_path": "calc.py",
                    "content": "whole file\n",
                    "sha256": sha256_hex(b"whole file\n"),
                    "verified_count": 9
                }],
                "fp-mixed": [
                    serde_json::to_value(&good).unwrap(),
                    serde_json::to_value(PatchCandidate { pre_sha256: "not-a-hash".into(), ..good.clone() }).unwrap(),
                    serde_json::to_value(PatchCandidate { repo: String::new(), ..good.clone() }).unwrap(),
                    serde_json::to_value(PatchCandidate { sha256: "ABCDEF".into(), ..good.clone() }).unwrap()
                ]
            }
        });
        let export = sign_export(&payload.to_string(), Some(key));
        let mut local = PatchMemory::new();
        let rep = local
            .merge_signed_guarded(&export, Some(key), FleetTrustPolicy::robust())
            .unwrap();
        assert_eq!(rep.new_candidates, 1);
        assert_eq!(rep.byzantine_rejected, 4);
        assert_eq!(local.len(), 1);
        assert!(!local.by_fingerprint.contains_key("fp-legacy"));
        assert_eq!(local.candidates("fp-mixed"), &[good]);
    }

    #[test]
    fn robust_merge_caps_verified_count_inflation_from_one_peer() {
        // A Byzantine key-holder ships a candidate claiming a million confirmations
        // to force it to the top of the try-order. The robust policy must cap the
        // trust a single peer can inject to +1.
        let key: &[u8] = b"fleet";
        let mut peer = PatchMemory::new();
        peer.by_fingerprint.insert(
            "fp".into(),
            vec![peer_candidate("a.rs", "BAD\n", "FIX\n", 1_000_000)],
        );
        let export = peer.export_signed(Some(key));

        let mut local = PatchMemory::new();
        local.record_verified("fp", &scope("a.rs"), "BAD\n", "FIX\n"); // local count = 1
        let rep = local
            .merge_signed_guarded(&export, Some(key), FleetTrustPolicy::robust())
            .unwrap();
        assert_eq!(rep.reinforced, 1);
        assert_eq!(
            local.candidates("fp")[0].verified_count,
            2,
            "one peer may add at most +1 (1 local + 1 capped bump)"
        );
    }

    #[test]
    fn a_fix_repeated_inside_one_export_counts_once() {
        let key: &[u8] = b"fleet";
        let mut peer = PatchMemory::new();
        peer.by_fingerprint.insert(
            "fp".into(),
            vec![peer_candidate("a.rs", "BAD\n", "FIX\n", 1); 50],
        );
        let export = peer.export_signed(Some(key));
        let mut local = PatchMemory::new();
        let rep = local
            .merge_signed_guarded(&export, Some(key), FleetTrustPolicy::robust())
            .unwrap();
        assert_eq!(rep.new_candidates, 1);
        assert_eq!(rep.reinforced, 0);
        assert_eq!(rep.byzantine_rejected, 49);
        assert_eq!(local.candidates("fp")[0].verified_count, 1);
    }

    #[test]
    fn a_known_fix_repeated_inside_one_export_reinforces_once() {
        // The node already holds the fix (count 1). A key-holder repeats it 50
        // times in one export: that is one peer's confirmation, not fifty.
        let key: &[u8] = b"fleet";
        let mut peer = PatchMemory::new();
        peer.by_fingerprint.insert(
            "fp".into(),
            vec![peer_candidate("a.rs", "BAD\n", "FIX\n", 1); 50],
        );
        let export = peer.export_signed(Some(key));
        let mut local = PatchMemory::new();
        local.record_verified("fp", &scope("a.rs"), "BAD\n", "FIX\n");
        let rep = local
            .merge_signed_guarded(&export, Some(key), FleetTrustPolicy::robust())
            .unwrap();
        assert_eq!(rep.new_candidates, 0);
        assert_eq!(rep.reinforced, 1);
        assert_eq!(rep.byzantine_rejected, 49);
        assert_eq!(
            local.candidates("fp")[0].verified_count,
            2,
            "one export adds at most one capped bump per fix, however often it repeats it"
        );
    }

    #[test]
    fn robust_merge_caps_new_candidate_flood_per_fingerprint() {
        let key: &[u8] = b"fleet";
        let mut peer = PatchMemory::new();
        let flood: Vec<PatchCandidate> = (0..20)
            .map(|i| peer_candidate("a.rs", "BAD\n", &format!("C{i}\n"), 99))
            .collect();
        peer.by_fingerprint.insert("fp".into(), flood);
        let export = peer.export_signed(Some(key));

        let mut local = PatchMemory::new();
        let rep = local
            .merge_signed_guarded(&export, Some(key), FleetTrustPolicy::robust())
            .unwrap();
        assert_eq!(
            rep.new_candidates, 8,
            "flood capped to 8 new per fingerprint"
        );
        assert_eq!(rep.byzantine_rejected, 12);
        assert_eq!(local.candidates("fp").len(), 8);
        assert!(
            local.candidates("fp").iter().all(|c| c.verified_count == 1),
            "fresh peer candidates seeded at a bounded count, not their claimed 99"
        );
    }

    #[test]
    fn robust_merge_rejects_oversized_candidates() {
        let key: &[u8] = b"fleet";
        let mut peer = PatchMemory::new();
        peer.by_fingerprint.insert(
            "fp".into(),
            vec![peer_candidate("a.rs", "x\n", "TOOBIG\n", 1)], // 7-byte replacement
        );
        let export = peer.export_signed(Some(key));

        let policy = FleetTrustPolicy {
            max_content_bytes: 4, // < 7
            ..FleetTrustPolicy::robust()
        };
        let mut local = PatchMemory::new();
        let rep = local
            .merge_signed_guarded(&export, Some(key), policy)
            .unwrap();
        assert_eq!(rep.new_candidates, 0);
        assert_eq!(rep.byzantine_rejected, 1);
        assert!(local.candidates("fp").is_empty());
        assert!(local.by_fingerprint.is_empty());
    }

    #[test]
    fn merging_the_same_export_again_changes_nothing() {
        let key: &[u8] = b"fleet";
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("axiom_patch_memory.json");
        let mut peer = PatchMemory::new();
        peer.record_verified("fp", &scope("a.rs"), "BAD\n", "FIX\n");
        peer.record_verified("fp-new", &scope("b.rs"), "BAD\n", "OTHER\n");
        let export = peer.export_signed(Some(key));

        let mut local = PatchMemory::new();
        local.record_verified("fp", &scope("a.rs"), "BAD\n", "FIX\n");
        let first = ExportLedger::with(&store, |ledger| {
            let report = local
                .merge_signed_once(&export, Some(key), FleetTrustPolicy::robust(), ledger)
                .unwrap();
            ledger.save().unwrap();
            report
        });
        assert!(!first.already_merged);
        assert_eq!((first.reinforced, first.new_candidates), (1, 1));
        assert_eq!(local.candidates("fp")[0].verified_count, 2);
        let after_first = serde_json::to_value(&local).unwrap();

        // Delivered again, any number of times, under either policy. The
        // ledger is read from disk each time, as after a restart.
        for policy in [FleetTrustPolicy::robust(), FleetTrustPolicy::permissive()] {
            for _ in 0..3 {
                let again = ExportLedger::with(&store, |ledger| {
                    local
                        .merge_signed_once(&export, Some(key), policy, ledger)
                        .unwrap()
                });
                assert_eq!(
                    again,
                    PatchMergeReport {
                        new_candidates: 0,
                        reinforced: 0,
                        already_merged: true,
                        byzantine_rejected: 0,
                    }
                );
            }
        }
        assert_eq!(
            serde_json::to_value(&local).unwrap(),
            after_first,
            "a repeated export must not move any count"
        );

        // A different export from the peer still counts.
        peer.record_verified("fp-later", &scope("c.rs"), "BAD\n", "LATER\n");
        let next = ExportLedger::with(&store, |ledger| {
            local
                .merge_signed_once(
                    &peer.export_signed(Some(key)),
                    Some(key),
                    FleetTrustPolicy::robust(),
                    ledger,
                )
                .unwrap()
        });
        assert!(!next.already_merged);
        assert_eq!(next.new_candidates, 1);
        assert_eq!(local.candidates("fp")[0].verified_count, 3);
    }

    #[test]
    fn an_export_this_node_would_refuse_is_neither_merged_nor_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("axiom_patch_memory.json");
        let mut peer = PatchMemory::new();
        peer.record_verified("fp", &scope("a.rs"), "BAD\n", "FIX\n");
        let foreign = peer.export_signed(Some(b"other-fleet"));
        let mut local = PatchMemory::new();
        ExportLedger::with(&store, |ledger| {
            // Wrong key, and no key at all.
            for key in [Some(&b"fleet"[..]), None] {
                assert!(local
                    .merge_signed_once(&foreign, key, FleetTrustPolicy::robust(), ledger)
                    .is_err());
            }
            assert!(!ledger.contains(&foreign.payload));
            // Recorded as seen, but still refused: the ledger is consulted
            // only after the export has been authenticated.
            ledger.remember(&foreign.payload);
            assert!(local
                .merge_signed_once(&foreign, Some(b"fleet"), FleetTrustPolicy::robust(), ledger)
                .is_err());
        });
        assert!(local.is_empty());
    }

    #[test]
    fn permissive_merge_preserves_legacy_summing_behavior() {
        // merge_signed (permissive) must still sum counts unbounded, so existing
        // fleet behavior is unchanged for callers with their own trust model.
        let key: &[u8] = b"fleet";
        let mut peer = PatchMemory::new();
        peer.by_fingerprint.insert(
            "fp".into(),
            vec![peer_candidate("a.rs", "BAD\n", "FIX\n", 50)],
        );
        let export = peer.export_signed(Some(key));
        let mut local = PatchMemory::new();
        local.record_verified("fp", &scope("a.rs"), "BAD\n", "FIX\n"); // 1
        let rep = local.merge_signed(&export, Some(key)).unwrap();
        assert_eq!(rep.reinforced, 1);
        assert_eq!(rep.byzantine_rejected, 0);
        assert_eq!(
            local.candidates("fp")[0].verified_count,
            51,
            "1 + 50 unbounded"
        );
    }
}
