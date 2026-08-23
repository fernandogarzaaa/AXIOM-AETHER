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
- [ ] **Checkpoint compatibility versioning**: `ModelMeta` has three
  independent boolean architecture flags (`stabilize`, `last_token_only`,
  `learned_gate`) but no `min_engine_version` or monotonic schema version.
  Low risk today (3 flags); add before a 4th is needed. See
  ARCHITECTURE-AUDIT.md §3.3.
- [ ] **`graph_memory.rs` integration**: the edge store + spreading
  activation merged this pass (ARCHITECTURE-AUDIT.md §3.4) is tested in
  isolation but not wired into `memory_recall.rs`. Needs a real design
  decision (which edge kinds auto-seed, how much `spread()` output widens a
  recall query, interaction with the existing recency/supersession rerank)
  — do this as its own reviewed PR, and evaluate it against
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
- [ ] **Wire `graph_memory.rs` into recall** (see P1 above — listed twice
  deliberately, since it's both an architectural loose end and a research
  question: does spreading activation improve or hurt recall precision).
- [ ] **`sandbox.rs` rename**, with a deprecation path for the
  `AXIOM_SANDBOX_*` env var names. See SECURITY-AUDIT.md §3. Scoped as P2
  (not P1) because the current naming is a clarity gap, not an active
  vulnerability — the mechanism itself (`cargo check`, no deps, no
  build.rs) is low-risk as implemented.
- [ ] **Capability-model pattern, generalized.** `AXIOM_ENABLE_JIT_EXEC` is
  the first instance; the next capability-shaped surface should follow the
  same pattern (dedicated off-by-default flag, checked first, tested with an
  explicit disabled-by-default regression test) rather than accreting a
  generic policy engine speculatively. See SECURITY-AUDIT.md §7.
- [ ] **Checksum manifest for release checkpoints.** This pass added
  *support* for pinning a checksum; it didn't publish one. Recommend
  publishing a signed `SHA256SUMS` alongside GitHub releases so
  `AXIOM_CHECKPOINT_SHA256` has a maintainer-verified source to be pinned
  from, closing the residual gap noted in SECURITY-AUDIT.md §2.

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
- [ ] **`axiombench`'s `RESULTS.md` overwrite bug** (found this pass, see
  AXIOMBENCH.md §2): running the binary without the cost pillar silently
  drops hand-maintained prose and the cost row from `RESULTS.md`. Fix:
  either preserve non-generated sections on write, or move the
  generated table into its own file the hand-written prose doesn't share.
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
- [ ] **`cargo fmt` is not a CI gate**, and the repo is not currently
  `cargo fmt --check`-clean even in files untouched by this pass (confirmed:
  large diffs in `bin/axiombench/*.rs`, `bin/build_pairs.rs`, and others).
  Not fixed in this pass (would be a large, unrelated diff). Recommend
  either adding `cargo fmt --check` to CI and doing one dedicated
  formatting-only PR to establish the baseline, or explicitly documenting
  that this repo doesn't enforce rustfmt and removing any implication
  otherwise.
- [ ] **Terminology table** (ARCHITECTURE-AUDIT.md §6) should be folded into
  the top-level README's architecture section for new-contributor
  discoverability — done partially in this pass's README diff (the security
  env vars); the full terminology table itself is new to this pass and
  currently lives only in ARCHITECTURE-AUDIT.md.

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
