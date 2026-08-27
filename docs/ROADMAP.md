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
- [x] **DONE** — `graph_memory.rs` wired into `memory_recall.rs`
  (`recall_with_graph`) and the `axiom_recall` MCP tool: bounded
  spreading activation widens direct cosine hits, scope-confined, capped by
  `RecallParams.graph_k` (default 3), each expanded hit marked
  `via_graph = true` so precision can be measured separately once
  [AXIOMBENCH.md](AXIOMBENCH.md)'s harness exists. `axiom_recall` also now
  writes `CoOccurred` edges between hits returned together, so the graph
  grows from real usage instead of staying backfill-only. See
  ARCHITECTURE-AUDIT.md §3.4 for the full design-decision writeup.

## P2 — Major product / research gaps

- [x] **DONE (partial)** / **[ ] still open (the rest)** — **Build the
  AXIOMBench task-success harness.** The mission brief's own top priority.
  Round 2 built and ran a first, real, narrow slice: `axiombench --ablation`
  measures the self-healing repair loop's baseline-vs-AXIOM pass rate over
  the 9-fixture `eval-agentic` suite (0/9 → 9/9, real and reproducible, not
  simulated — see AXIOMBENCH.md §3 for the result and its scope caveats).
  **Still open**: this covers exactly one dimension (self-healing) on one
  hand-built fixture suite with no live agent involved. Compression,
  memory, and adaptation still have zero task-outcome measurement, and
  building that credibly needs real agent-run infrastructure this audit
  pass doesn't have. Full spec — task suite design, metrics already
  instrumented vs. missing, the measured/simulated/estimated labeling rule
  — is in [AXIOMBENCH.md](AXIOMBENCH.md) §4. Still the single highest-value
  next project for this repository's scientific credibility.
- [x] **DONE (wiring)** — see P1 above. The *research question* remains
  genuinely open and belongs here, not there: does spreading activation
  measurably improve or hurt recall precision, once
  [AXIOMBENCH.md](AXIOMBENCH.md)'s wrong-memory-rate metric exists to
  measure it? The `via_graph` flag exists specifically so that experiment
  is possible when the harness is built; it hasn't been run.
- [x] **DONE** — `sandbox.rs`/`SandboxController` renamed to
  `compile_verify.rs`/`CompileVerifier`, with `AXIOM_SANDBOX_*` accepted as
  deprecated aliases for one release. See SECURITY-AUDIT.md §3.
- [ ] **Capability-model pattern, generalized.** `AXIOM_ENABLE_JIT_EXEC` is
  the first instance; the next capability-shaped surface should follow the
  same pattern (dedicated off-by-default flag, checked first, tested with an
  explicit disabled-by-default regression test) rather than accreting a
  generic policy engine speculatively. See SECURITY-AUDIT.md §7.
- [x] **DONE (partially — corrected from round 1)** — `release.yml`'s
  `checkpoint` job *already* published `SHA256SUMS.txt` alongside every
  checkpoint/tokenizer release asset (`sha256sum ckpt_release/* >
  ckpt_release/SHA256SUMS.txt`); round 1 of this audit missed that and
  incorrectly reported it as missing. What actually was missing: the
  **Docker container entrypoint** (`scripts/docker_entrypoint.sh`) fetches
  checkpoints via a completely separate shell-based path from `axiom
  init`'s Rust path, and had no integrity verification at all — round 1's
  `AXIOM_CHECKPOINT_SHA256` fix only covered the Rust path. Fixed in round
  2: `docker_entrypoint.sh` now verifies against the same
  `AXIOM_CHECKPOINT_SHA256`/`AXIOM_TOKENIZER_SHA256` env vars, fail-closed
  on mismatch, tested directly (extracted-function unit test, not just
  read). **Still open**: `SHA256SUMS.txt` itself is unsigned — a compromised
  release pipeline could publish a new "correct" hash alongside a bad
  binary. GPG/sigstore signing needs real key material and CI secret
  configuration this audit pass doesn't have access to set up; left as a
  named gap, not implemented speculatively.

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
- [x] **DONE (documented; deliberately not reformatted)** — measured the
  actual size of the gap: `rustfmt --check` across every file in
  `axiom_engine_rs/src/` disagrees with **77 of 127 files (61%)**, ~12,200
  lines of diff, 894 individual hunks. `CONTRIBUTING.md` already tells
  contributors to run `cargo fmt` before a PR, but nothing has ever enforced
  it in CI, so the tree drifted this far unnoticed. A diff this size is not
  a same-pass fix: a mechanical repo-wide reformat touching 61% of files
  would swamp every other change in this audit's diff, and is real,
  deliberate, isolated work a maintainer should schedule on its own, not
  something to smuggle into an unrelated PR. Fixed the actual problem
  instead — `CONTRIBUTING.md` now says explicitly that `cargo fmt --check`
  isn't CI-enforced and that a contributor should format only the files
  they touch, not the whole tree, so nobody wastes a PR discovering this
  the hard way. **Still open, for whoever picks it up next**: do the
  dedicated formatting-only PR (large but mechanical and low-risk — rustfmt
  doesn't change semantics), then add `cargo fmt --check` to `ci.yml` so it
  can't drift this far again.
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
