//! Verified-patch memory — the substrate for fleet patch sharing.
//!
//! When the autonomous repair loop ([`crate::solve`]) makes a failing target go
//! green, the verified fix can be recorded here, exported to peers (signed), and
//! merged from peers. The point: a fix one node discovers becomes shared coding
//! skill across the fleet.
//!
//! # Safety invariant (the whole design rests on this)
//!
//! **A patch received from a peer is NEVER applied on trust.** It is only ever
//! written through [`PatchMemory::try_candidates`], which writes the candidate,
//! runs *this node's own* verify check, and keeps the patch **only if it goes
//! green locally** — otherwise it is rolled back byte-for-byte. So an incorrect
//! or malicious peer patch cannot harm a node: it simply fails local
//! verification and is discarded. Provenance ([`crate::provenance`]: SHA-256 +
//! optional HMAC via `AXIOM_FLEET_KEY`) additionally gates which exports are even
//! considered, and `verified_count` records how many independent nodes have
//! confirmed a patch green (a trust signal for ranking, never for blind apply).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::provenance::{sha256_hex, sign_export, verify_export, ProvenanceError, SignedExport};

/// One verified candidate fix for a given failure fingerprint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatchCandidate {
    /// Repo-relative source path the patch targets (portable across checkouts).
    pub rel_path: String,
    /// The full verified file contents that made the verify command pass.
    pub content: String,
    /// SHA-256 of `content` — dedup key + integrity check.
    pub sha256: String,
    /// How many independent nodes have verified this patch green. A trust signal
    /// used only for ranking; it never authorizes a blind apply.
    pub verified_count: u32,
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
    /// Candidates rejected by the [`FleetTrustPolicy`] (oversized content or
    /// per-fingerprint flood cap). `0` under the permissive default policy.
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
    /// Reject any candidate whose content exceeds this many bytes.
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

    /// Load from disk; a missing/corrupt file yields an empty store.
    pub fn load(path: impl AsRef<Path>) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
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

    /// Record a locally-verified patch. If an identical patch (same content) is
    /// already stored for this fingerprint, bump its `verified_count`; otherwise
    /// add it. Candidates are kept ranked by `verified_count` (desc).
    pub fn record_verified(&mut self, fingerprint: &str, rel_path: &str, content: &str) {
        let sha = sha256_hex(content.as_bytes());
        let list = self.by_fingerprint.entry(fingerprint.to_string()).or_default();
        if let Some(existing) = list.iter_mut().find(|c| c.sha256 == sha) {
            existing.verified_count = existing.verified_count.saturating_add(1);
        } else {
            list.push(PatchCandidate {
                rel_path: rel_path.to_string(),
                content: content.to_string(),
                sha256: sha,
                verified_count: 1,
            });
        }
        list.sort_by(|a, b| b.verified_count.cmp(&a.verified_count));
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

    /// Try stored candidates for `fingerprint` against the live source file,
    /// keeping the FIRST that makes `verify()` return true. Each candidate is
    /// written then verified; a candidate that does not verify green is rolled
    /// back byte-for-byte before trying the next. Returns the sha of the applied
    /// patch, or `None` if none verified (source left untouched).
    ///
    /// This is the safety boundary for fleet patches: a peer's fix is only ever
    /// trusted after *this node's* `verify` independently confirms it.
    pub fn try_candidates<V>(
        &self,
        fingerprint: &str,
        rel_path: &str,
        source_path: &Path,
        mut verify: V,
    ) -> Option<String>
    where
        V: FnMut() -> bool,
    {
        let candidates = self.by_fingerprint.get(fingerprint)?;
        if candidates.is_empty() {
            return None;
        }
        // Only ever write a candidate to the file it was recorded for. A single
        // verify command (e.g. a broad `cargo test`) can cover many files and
        // share one fingerprint; without this guard a patch learned for `a.rs`
        // could be written to `b.rs` and kept if the suite still passes —
        // corrupting the wrong file. Match the exact recorded rel_path (computed
        // the same way at record time), so even same-named files in different
        // directories never cross-apply.
        let original = std::fs::read_to_string(source_path).ok()?;

        // B.1 per-bug test-time adaptation: among the candidates recorded for this
        // exact file, try them in self-consistency-reranked order (consensus
        // across alpha-rename/whitespace-equivalent fixes first, weighted by
        // verified_count) rather than raw verified_count order. This only changes
        // *which candidate the verifier tries first* — the verify gate below is
        // unchanged, so re-verify-before-trust is fully preserved.
        let matching: Vec<&PatchCandidate> =
            candidates.iter().filter(|c| c.rel_path == rel_path).collect();
        let adapter_cands: Vec<crate::test_time_adapter::Candidate> = matching
            .iter()
            .map(|c| crate::test_time_adapter::Candidate::new(c.content.clone(), c.verified_count as f32))
            .collect();
        let order = crate::test_time_adapter::rerank_by_self_consistency(&adapter_cands);

        for &i in &order {
            let c = matching[i];
            if c.content == original {
                continue; // already the current source
            }
            if std::fs::write(source_path, &c.content).is_err() {
                let _ = std::fs::write(source_path, &original);
                continue;
            }
            if verify() {
                return Some(c.sha256.clone());
            }
            // Rejected locally → roll back byte-for-byte; never trust unverified.
            if std::fs::write(source_path, &original).is_err() {
                eprintln!("[axiom-patch] WARNING: failed to restore original after rejected candidate");
            }
        }
        None
    }

    /// Candidate patches for `fingerprint` targeting `rel_path`, ordered by the
    /// B.1 per-bug test-time adapter (self-consistency voting weighted by
    /// `verified_count`) — the order [`try_candidates`] uses. Exposed for callers
    /// that want the adapter's ranking without driving the verify loop.
    pub fn candidates_reranked(&self, fingerprint: &str, rel_path: &str) -> Vec<PatchCandidate> {
        let Some(list) = self.by_fingerprint.get(fingerprint) else {
            return Vec::new();
        };
        let matching: Vec<&PatchCandidate> =
            list.iter().filter(|c| c.rel_path == rel_path).collect();
        let adapter_cands: Vec<crate::test_time_adapter::Candidate> = matching
            .iter()
            .map(|c| crate::test_time_adapter::Candidate::new(c.content.clone(), c.verified_count as f32))
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
    /// added; identical ones (same fingerprint+sha) accumulate `verified_count`.
    /// Rejects the merge if provenance fails (bad hash / wrong key / unsigned
    /// when a key is required). NEVER applies anything — merging only stores
    /// candidates for later re-verified application.
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

    /// Byzantine-robust merge: verify provenance, then fold the peer's candidates
    /// in subject to `policy` — bounding how much `verified_count` trust and how
    /// many new candidates a single peer export can contribute, and dropping
    /// oversized candidates. The local re-verify gate still decides what is ever
    /// applied; this only bounds a malicious key-holder's influence on ranking
    /// and store size.
    pub fn merge_signed_guarded(
        &mut self,
        export: &SignedExport,
        fleet_key: Option<&[u8]>,
        policy: FleetTrustPolicy,
    ) -> Result<PatchMergeReport, ProvenanceError> {
        let payload = verify_export(export, fleet_key)?;
        let peer: PatchMemory = serde_json::from_str(payload)
            .map_err(|_| ProvenanceError::BadSchema)?;
        let mut report = PatchMergeReport {
            new_candidates: 0,
            reinforced: 0,
            byzantine_rejected: 0,
        };
        for (fp, peer_list) in peer.by_fingerprint {
            let local = self.by_fingerprint.entry(fp).or_default();
            let mut new_this_fp = 0usize;
            for mut pc in peer_list {
                // Drop implausibly large candidates outright.
                if pc.content.len() > policy.max_content_bytes {
                    report.byzantine_rejected += 1;
                    continue;
                }
                // Never trust the peer-supplied hash: the signature proves the
                // bytes weren't changed in transit, not that sha256 identifies
                // content. Recompute from content so dedup, ranking, and logged
                // patch IDs always reflect the real bytes.
                pc.sha256 = sha256_hex(pc.content.as_bytes());
                if let Some(existing) = local.iter_mut().find(|c| c.sha256 == pc.sha256) {
                    // Bound the trust a single peer can inject: one export should
                    // count as at most a few independent confirmations.
                    let bump = pc.verified_count.min(policy.max_verified_bump_per_candidate);
                    existing.verified_count = existing.verified_count.saturating_add(bump);
                    report.reinforced += 1;
                } else {
                    // Flood cap: reject excess brand-new candidates per fingerprint.
                    if new_this_fp >= policy.max_new_candidates_per_fingerprint {
                        report.byzantine_rejected += 1;
                        continue;
                    }
                    // Also clamp the incoming count of a brand-new candidate so a
                    // peer cannot seed a fresh patch at the top of the ranking.
                    pc.verified_count =
                        pc.verified_count.min(policy.max_verified_bump_per_candidate);
                    local.push(pc);
                    new_this_fp += 1;
                    report.new_candidates += 1;
                }
            }
            local.sort_by(|a, b| b.verified_count.cmp(&a.verified_count));
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_dedups_by_content_and_ranks_by_verified_count() {
        let mut m = PatchMemory::new();
        m.record_verified("fp1", "src/a.rs", "FIX_A");
        m.record_verified("fp1", "src/a.rs", "FIX_A"); // identical → reinforce
        m.record_verified("fp1", "src/a.rs", "FIX_B"); // different → new candidate
        let cands = m.candidates("fp1");
        assert_eq!(cands.len(), 2, "two distinct candidates");
        assert_eq!(cands[0].content, "FIX_A", "most-verified ranked first");
        assert_eq!(cands[0].verified_count, 2);
        assert_eq!(cands[1].verified_count, 1);
        assert_eq!(m.candidates("nope").len(), 0);
    }

    #[test]
    fn json_roundtrip_preserves_store() {
        let mut m = PatchMemory::new();
        m.record_verified("fp", "x.rs", "GOOD");
        let json = m.to_json();
        let back: PatchMemory = serde_json::from_str(&json).unwrap();
        assert_eq!(back.candidates("fp"), m.candidates("fp"));
    }

    #[test]
    fn signed_export_merge_roundtrip_dedups_and_sums() {
        let key: &[u8] = b"fleet-secret";
        let mut a = PatchMemory::new();
        a.record_verified("fp", "x.rs", "GOOD");
        let export = a.export_signed(Some(key));

        let mut b = PatchMemory::new();
        b.record_verified("fp", "x.rs", "GOOD"); // same patch, independently verified
        b.record_verified("fp2", "y.rs", "OTHER"); // unrelated
        let rep = b.merge_signed(&export, Some(key)).unwrap();
        assert_eq!(rep.reinforced, 1, "identical patch reinforced");
        assert_eq!(rep.new_candidates, 0);
        // The shared patch now has verified_count 2 (1 local + 1 from peer).
        assert_eq!(b.candidates("fp")[0].verified_count, 2);
        assert_eq!(b.candidates("fp2").len(), 1, "unrelated local patch intact");
    }

    #[test]
    fn merge_rejects_tampered_or_wrong_key() {
        let mut a = PatchMemory::new();
        a.record_verified("fp", "x.rs", "GOOD");
        // Tampered payload → hash mismatch.
        let mut bad = a.export_signed(None);
        bad.payload.push_str("tampered");
        assert!(PatchMemory::new().merge_signed(&bad, None).is_err());
        // Wrong key.
        let signed = a.export_signed(Some(b"right"));
        assert!(PatchMemory::new()
            .merge_signed(&signed, Some(b"wrong"))
            .is_err());
    }

    #[test]
    fn try_candidates_applies_only_a_locally_verified_patch() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("axiom_patchmem_{}.txt", std::process::id()));
        std::fs::write(&path, "BROKEN").unwrap();

        let mut m = PatchMemory::new();
        m.record_verified("fp", "f.txt", "WRONG"); // won't verify
        m.record_verified("fp", "f.txt", "FIXED"); // will verify
        // Make FIXED outrank WRONG so order is deterministic-ish, but try() must
        // still skip WRONG (fails verify) and keep FIXED.
        m.record_verified("fp", "f.txt", "FIXED");

        // verify() = the file currently contains FIXED.
        let p = path.clone();
        let applied = m.try_candidates("fp", "f.txt", &path, || {
            std::fs::read_to_string(&p).map(|s| s.contains("FIXED")).unwrap_or(false)
        });
        assert!(applied.is_some(), "the locally-verifying patch is applied");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "FIXED");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn try_candidates_leaves_source_untouched_when_none_verify() {
        let path = std::env::temp_dir().join(format!("axiom_patchmem_none_{}.txt", std::process::id()));
        std::fs::write(&path, "ORIGINAL").unwrap();
        let mut m = PatchMemory::new();
        m.record_verified("fp", "f.txt", "ALSO_WRONG");
        // verify() always false → no candidate accepted.
        let applied = m.try_candidates("fp", "f.txt", &path, || false);
        assert!(applied.is_none());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "ORIGINAL",
            "a never-verifying patch must leave the source byte-for-byte intact"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn try_candidates_never_cross_applies_to_a_different_rel_path() {
        // A patch recorded for `src/a.rs` must never be written to `src/b.rs`,
        // even when they share a fingerprint and verify() would pass. Without the
        // rel_path guard a broad suite (one fingerprint, many files) could let an
        // `a.rs` fix corrupt `b.rs`.
        let path = std::env::temp_dir().join(format!("axiom_patchmem_xfile_{}.txt", std::process::id()));
        std::fs::write(&path, "ORIGINAL_B").unwrap();
        let mut m = PatchMemory::new();
        m.record_verified("fp", "src/a.rs", "FIX_FOR_A"); // recorded for a.rs only

        // Caller targets b.rs (rel_path "src/b.rs"); verify() would accept anything.
        let applied = m.try_candidates("fp", "src/b.rs", &path, || true);
        assert!(applied.is_none(), "a patch recorded for a.rs is not applied to b.rs");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "ORIGINAL_B",
            "the wrong-target file must be left byte-for-byte intact"
        );
        let _ = std::fs::remove_file(&path);
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
            vec![PatchCandidate {
                rel_path: "a.rs".into(),
                content: "FIX".into(),
                sha256: sha256_hex(b"FIX"),
                verified_count: 1_000_000,
            }],
        );
        let export = peer.export_signed(Some(key));

        let mut local = PatchMemory::new();
        local.record_verified("fp", "a.rs", "FIX"); // local count = 1
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
    fn robust_merge_caps_new_candidate_flood_per_fingerprint() {
        let key: &[u8] = b"fleet";
        let mut peer = PatchMemory::new();
        let flood: Vec<PatchCandidate> = (0..20)
            .map(|i| PatchCandidate {
                rel_path: "a.rs".into(),
                content: format!("C{i}"),
                sha256: String::new(), // recomputed on merge
                verified_count: 99,
            })
            .collect();
        peer.by_fingerprint.insert("fp".into(), flood);
        let export = peer.export_signed(Some(key));

        let mut local = PatchMemory::new();
        let rep = local
            .merge_signed_guarded(&export, Some(key), FleetTrustPolicy::robust())
            .unwrap();
        assert_eq!(rep.new_candidates, 8, "flood capped to 8 new per fingerprint");
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
            vec![PatchCandidate {
                rel_path: "a.rs".into(),
                content: "TOOBIG".into(), // 6 bytes
                sha256: String::new(),
                verified_count: 1,
            }],
        );
        let export = peer.export_signed(Some(key));

        let policy = FleetTrustPolicy {
            max_content_bytes: 4, // < 6
            ..FleetTrustPolicy::robust()
        };
        let mut local = PatchMemory::new();
        let rep = local.merge_signed_guarded(&export, Some(key), policy).unwrap();
        assert_eq!(rep.new_candidates, 0);
        assert_eq!(rep.byzantine_rejected, 1);
        assert!(local.candidates("fp").is_empty());
    }

    #[test]
    fn permissive_merge_preserves_legacy_summing_behavior() {
        // merge_signed (permissive) must still sum counts unbounded, so existing
        // fleet behavior is unchanged for callers with their own trust model.
        let key: &[u8] = b"fleet";
        let mut peer = PatchMemory::new();
        peer.by_fingerprint.insert(
            "fp".into(),
            vec![PatchCandidate {
                rel_path: "a.rs".into(),
                content: "FIX".into(),
                sha256: sha256_hex(b"FIX"),
                verified_count: 50,
            }],
        );
        let export = peer.export_signed(Some(key));
        let mut local = PatchMemory::new();
        local.record_verified("fp", "a.rs", "FIX"); // 1
        let rep = local.merge_signed(&export, Some(key)).unwrap();
        assert_eq!(rep.reinforced, 1);
        assert_eq!(rep.byzantine_rejected, 0);
        assert_eq!(local.candidates("fp")[0].verified_count, 51, "1 + 50 unbounded");
    }
}
