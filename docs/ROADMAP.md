# AXIOM-AETHER — Roadmap (2026-08-23)

Every item below is ranked by the mission brief's own priority scheme (P0
security/correctness/data-loss/runtime-breaking, P1 core architectural, P2
major product/research gaps, P3 performance, P4 documentation/cleanup).
Items marked **DONE** were implemented in this audit pass (see
[ARCHITECTURE-AUDIT.md](ARCHITECTURE-AUDIT.md) §8 for the full change list);
everything else is a recommendation, not a commitment — this document
doesn't speak for the maintainer's actual priorities, only for what the
evidence in this repo supports as important.

## P0 — Security / correctness / data-loss / runtime-breaking

- [x] **DONE** — `POST /v1/hypervisor/jit_run` gated behind
  `AXIOM_ENABLE_JIT_EXEC` (off by default), independent of `AXIOM_API_KEY`.
  See [SECURITY-AUDIT.md](SECURITY-AUDIT.md) §1.
- [ ] No other P0s found this pass. (The 2026-07-24 audit's P0-equivalent —
  the broken Docker build — was already fixed before this pass; reverified
  clean this pass via the existing CI-gated `docker-build` job's continued
  presence in `ci.yml`, not re-run live.)

## P1 — Core architectural problems

- [x] **DONE** — `Arc<Mutex<InferencePipeline>>` → `Arc<RwLock<InferencePipeline>>`,
  fixing the fake-parallelism bug in `responses_run_fingerprint`'s JoinSet
  fan-out. See [ARCHITECTURE-AUDIT.md](ARCHITECTURE-AUDIT.md) §4.
- [x] **DONE** — Checkpoint download integrity (`AXIOM_CHECKPOINT_SHA256`/
  `AXIOM_TOKENIZER_SHA256`, fail-closed on mismatch). See
  [SECURITY-AUDIT.md](SECURITY-AUDIT.md) §2.
- [ ] **Not done — architectural, not a quick fix**: no OS-level execution
  isolation anywhere self_heal/solve/poly_jit/sandbox run a subprocess (see
  SECURITY-AUDIT.md §4). Recommend: evaluate a lightweight sandboxing
  primitive (a `landlock`/seccomp profile, or an opt-in container runtime
  for the `jit_run` path specifically, since that's the one already gated
  and network-reachable) as a follow-up feature, not an audit fix. Scope
  before starting: this is a multi-week feature with real platform
  portability tradeoffs (Linux-only primitives vs. the Windows-first
  deployment story `scripts/*.ps1` implies), not a one-PR change.
- [x] **DONE (2026-10-29)** — **Checkpoint compatibility versioning**:
  `ModelMeta` now carries `schema_version` (current `1`; unversioned sidecars
  load as `0`) and `engine_version` provenance. `save` stamps both;
  `load_checked` refuses a sidecar from a newer schema, and the production
  load path (`InferencePipeline::with_checkpoint_and_options`) fails loudly
  instead of building the wrong model shape. See ARCHITECTURE-AUDIT.md §3.3.
- [x] **DONE (2026-10-29)** — **`graph_memory.rs` integration**: `recall`
  seeds spreading activation from its direct cosine hits over the persisted
  `edges.jsonl` plus `Supersedes` edges derived from records, and appends up
  to `GraphRecallParams::max_extra` (default 2) graph-only hits *after* the
  direct hits (never displacing one, never resurrecting a superseded or
  tombstoned record; cross-scope neighbours are fetched by id). Hits carry
  `via_graph` and `axiom_recall` labels them. Decisions taken: only
  `Supersedes` auto-seeds; no edge is written automatically; spread is 2
  hops, decay 0.5, 256 nodes. What remains is empirical — whether it helps
  precision — which is the AXIOMBench wrong-memory-rate metric below.
  (Original note: needed a design decision on which edge kinds auto-seed, how
  much `spread()` output widens a recall query, and the interaction with the
  recency/supersession rerank — evaluate it against
  [AXIOMBENCH.md](AXIOMBENCH.md)'s proposed wrong-memory-rate metric once
  that harness exists, not by feel.

## P2 — Major product / research gaps

- [ ] **Build the AXIOMBench task-success harness.** The mission brief's
  own top priority; not built this pass because it requires real agent-run
  infrastructure this audit cannot fabricate credibly. Full spec — task
  suite design, metrics already instrumented vs. missing, the
  measured/simulated/estimated labeling rule — is in
  [AXIOMBENCH.md](AXIOMBENCH.md) §3. This is the single highest-value next
  project for this repository's scientific credibility.
- [x] **DONE (2026-10-29)** — **Wire `graph_memory.rs` into recall** (see
  P1 above). The research question — does spreading activation improve or
  hurt recall precision — stays open until the AXIOMBench harness exists.
- [x] **DONE (2026-10-29)** — **`sandbox.rs` rename**: the module is
  `compile_verify.rs` and the type `CompileVerifier`; `AXIOM_COMPILE_VERIFY`,
  `AXIOM_COMPILE_VERIFY_ROOT`, `AXIOM_COMPILE_VERIFY_MAX_STEPS` and
  `AXIOM_COMPILE_VERIFY_TIMEOUT_SECS` are the names, with the `AXIOM_SANDBOX_*`
  names accepted as deprecated aliases (one stderr warning per process) for
  one release. `crate::sandbox` remains as a deprecated re-export.
- [ ] **Capability-model pattern, generalized.** `AXIOM_ENABLE_JIT_EXEC` is
  the first instance; the next capability-shaped surface should follow the
  same pattern (dedicated off-by-default flag, checked first, tested with an
  explicit disabled-by-default regression test) rather than accreting a
  generic policy engine speculatively. See SECURITY-AUDIT.md §7.
- [x] **DONE (2026-10-29)** — **Checksum manifest for releases.** The
  checkpoint job already published `SHA256SUMS.txt` for the checkpoint
  assets; the binary job now publishes a `.sha256` sidecar next to every
  archive, and `scripts/install.sh` / `scripts/install.ps1` verify the
  downloaded archive against it (refusing a mismatch; `AXIOM_REQUIRE_CHECKSUM=1`
  also refuses a release with no sidecar). Pin a checkpoint with
  `AXIOM_CHECKPOINT_SHA256` from the published `SHA256SUMS.txt`. Signing the
  manifest (Sigstore/minisign) is still open and needs a maintainer key.

- [x] **DONE (2026-10-29)** — **CVM store cross-session bound**
  (SECURITY-AUDIT.md addendum A9): `put` is append-only, the in-memory
  index is LRU-bounded by `AXIOM_CVM_MAX_INDEX_BYTES`, the on-disk total by
  `AXIOM_CVM_MAX_TOTAL_BYTES`, and the digest hook writes off the async
  worker.

## P3 — Performance improvements

- [ ] No new P3 performance findings beyond the P1 concurrency fix already
  made (which was scoped as P1 because it was blocking real parallelism a
  named product feature — Responses-path multi-run fingerprinting — was
  specifically built to have). Re-profile after
  [AXIOMBENCH.md](AXIOMBENCH.md)'s harness exists, once there's a real
  latency/throughput metric to optimize against instead of a structural
  code-reading argument.

## P4 — Documentation / cleanup

- [x] **DONE** — deleted `memory_pool.rs` (one-line dead stub, zero
  callers).
- [x] **DONE** — this document, plus ARCHITECTURE-AUDIT.md,
  SECURITY-AUDIT.md, COMPETITIVE-ANALYSIS.md, AXIOMBENCH.md, and a README
  terminology/security update.
- [x] **DONE (2026-10-29)** — **`axiombench`'s `RESULTS.md` overwrite bug**
  (see AXIOMBENCH.md §2): the writer now replaces only a marker-delimited
  table block, carries over rows for pillars that did not run, and preserves
  all hand-maintained prose.
- [ ] **Stale branch/PR cleanup** — this pass determined disposition for
  every open branch/PR (ARCHITECTURE-AUDIT.md §8, table) but did not delete
  or force-close any of them, as that's a visible, semi-destructive action
  better left to the repository owner. Recommended for cleanup once
  reviewed: close PR #81 (superseded by `docs/AUDIT_2026-07.md`); delete
  branches `claude/axiom-aether-audit-e66aji`, `claude/interactive-tui`,
  `claude/mesh-router-integration`, `claude/axiom-swarm-architecture-ntenzk`,
  `claude/generalizable-heal-rules`, `cvm/s8-rollout`,
  `docs/audit-2026-06` (all confirmed superseded/stale, evidence in the
  table). `claude/docs-audit-fixes` (95 lines of un-reviewed doc diff
  remaining) is worth a quick look before deciding, not blind deletion.
- [x] **DONE (2026-10-29)** — **`cargo fmt` is now a CI gate**: the three
  Rust crate graphs were formatted in one dedicated formatting-only commit
  and `ci.yml` runs `cargo fmt --check` on each of them. Clippy is also
  gated at deny-warnings across all targets (bins and tests), not just the
  library.
- [x] **DONE (2026-10-29)** — **Terminology table** folded into the README
  (`## Terminology`, before the audits section), with the compile-verifier
  rename and graph memory reflected.

## Explicitly not recommended (considered and rejected this pass)

- **The 5-service architectural split** (`TokenizerService`/`ModelRuntime`/
  `AdaptationRuntime`/`SessionStateStore`/`InferenceExecutor`) the mission
  brief raised as an option. Investigated in depth (ARCHITECTURE-AUDIT.md
  §4): the actual contention was a lock-granularity bug, already fixed by
  the RwLock change, with no remaining evidence a service decomposition
  would add real capability. Recommending it anyway would be scope growth
  without a measured justification.
- **A generic capability/policy engine**, before there are enough
  gated-capability instances to generalize from honestly (currently: one —
  `AXIOM_ENABLE_JIT_EXEC`). See SECURITY-AUDIT.md §7.
- **Moving EXPERIMENTAL modules into a literal `experimental/` directory**
  as a one-shot rename. Each already documents its own status accurately in
  its own doc comments (ARCHITECTURE-AUDIT.md §2); a mechanical move is a
  large diff for no behavior change. Do it opportunistically as those
  modules are touched for other reasons, not as a standalone PR.
