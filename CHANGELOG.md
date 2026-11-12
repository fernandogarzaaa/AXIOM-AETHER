# Changelog

Notable changes to the `axiom_engine` crate / `axiom-aether` wheels. Older
releases are described on the GitHub Releases page.

## [Unreleased]

### Fixed

- `axiom solve` no longer "repairs" a failing check by editing it. The first
  source-repair phase only knew fixture-shaped rewrites (`exit 1` → `exit 0`,
  a marker rename, replacing the file with a script that exits 0) and reported
  SOLVED when one made the command pass. It is removed from the solve loop;
  source repair starts at the mutation layer.
- A source file longer than 4096 characters is no longer cut while a patch is
  being tried (`axiom solve`, and `POST /v1/hypervisor/jit_run`).
- `axiom solve` no longer leaves a candidate fix in your file when it is
  interrupted. The original of every file it writes to is saved to disk first
  (`~/.axiom/journal`, or `AXIOM_JOURNAL_DIR`) and restored on failure, on
  Ctrl-C / SIGTERM / SIGHUP (exit status 130 / 143 / 129), and by the next
  `axiom solve` after a kill. A file you edited after the kill is left alone
  and the saved original's path is printed. The journal directory and its
  files are owner-only; a directory other users could write to has its
  existing contents set aside, never restored. Two solves never repair the
  same file at once: the second one leaves that file alone.
- `POST /v1/hypervisor/jit_run` no longer replaces a file longer than 4096
  characters with a two-line script when the file contains `throw `,
  `Write-Error` or a fixture marker.
- A fleet export counts once. Delivering the same signed export again to
  `POST /v1/patches/merge` or `POST /v1/immunity/merge` (a retry, a sync job
  on a timer, a replay, or a node's own export posted back to it) used to
  count as a fresh confirmation each time: a stored fix's `verified_count`
  rose and a heal record's evidence compounded. It now answers
  `"already_merged": true` and changes nothing. A store remembers the last
  1024 exports it merged, in the store file, and the last 1024 it handed
  out, in `<store>.exports.json` beside it. `axiom immunity pull` says when
  a peer's export was merged before.
- The patch store and the heal memory are saved by rename instead of being
  written in place. A save that is interrupted, or read while it is in
  progress, used to leave or show a partly written file, and a store that
  does not parse loads as empty. The old file now stays whole until the new
  one is complete. The file keeps its permissions; a store that is a
  symbolic link stays a link; a store marked read-only is refused. Where a
  file cannot be replaced at all (a single file mounted into a container, a
  directory the process may not create files in) it is written in place, as
  before.

### Changed

- `axiom eval-agentic`: the 12 fixtures that were solved by the removed
  rewrites are now shell programs with a real one-token bug, checked by
  behaviour. Fixture names changed (`shell-exit-flip` → `sh-self-naming-frame`,
  `fixture-marker` → `sh-wrong-arith`, `rust-assert-flip` → `rust-arrow-frame`).
  The suite is still 32 cases.

## [0.7.0] - 2026-11-08

Headline: the deterministic repair engine (no LLM) now repairs **39/40**
seeded faults on the original held-out corpus and **38/40** on a fresh one,
up from 19/40 and 20/40 in 0.6.0. Against a strengthened test suite, 33/40 and
32/40 are verified correct (0.6.0: 16/40). On **real developer bugs**
(QuixBugs, 40 Python programs) it repairs **9/40 held out, all correct**
(14/40 in-sample after operators chosen from its failures). `axiom solve` no
longer hangs, aborts or leaks processes on a test that loops forever, and
never repairs by editing a test file.

**Crate versions:** `axiom_engine` 0.7.0 depends on `axiom_mesh_core`
**0.3.0** (public API moved to `rand` 0.10). **Minimum Rust:** `axiom_engine`
1.94 (was 1.90, set by `candle` 0.11); `axiom_mesh_core` stays at 1.90.

### Removed

Removed as announced in 0.6.0:

- `crate::hamiltonian` and `crate::q_manifold`. Use `crate::patch_ranker`
  and `crate::branch_weights`.
- The `/v1/hypervisor/quantum_coherent_state` route; it now returns 404. Use
  `/v1/hypervisor/patch_ranker_state`. The `quantum` JSON key in
  `/v1/hypervisor/jit_status` is unchanged.
- `crate::sandbox`, the `SandboxController` / `SandboxDiagnostic` /
  `SandboxRunReport` / `SandboxError` type aliases, and the
  `AXIOM_SANDBOX_*` environment names, which are now ignored. Use
  `crate::compile_verify` and `AXIOM_COMPILE_VERIFY*`.

### Developer bugs (QuixBugs)

On QuixBugs (40 Python programs with one human-written bug each) the
deterministic engine repairs **9/40, all correct**, with 0 rollback violations
([write-up](bench/results/QUIXBUGS-2026-11-07.md)).

### Dependencies

- `rand` 0.9 → 0.10 in `axiom_engine`, `axiom_mesh_core` and
  `axiom_mesh_rs`. Methods such as `random_range` and `random_bool` now come
  from `rand::RngExt`, `RngCore` is now `Rng`, and `StdRng::from_os_rng()` is
  replaced by `rand::make_rng::<StdRng>()`.
- **`axiom_mesh_core` 0.2.0 → 0.3.0.** Its public routing functions take
  `&mut impl rand::Rng`, which is a different trait in `rand` 0.10, so
  callers must use `rand` 0.10 as well.
- **Minimum Rust version for `axiom_engine`: 1.90 → 1.94.** `candle-core`
  0.11 does not compile on 1.93 or older: its `__cpuid_count` call outside
  `unsafe` is rejected there, and 1.94 is the first release that builds it.
  `axiom_mesh_core` still builds on 1.90 and keeps that minimum.
- `candle-core` / `candle-nn` / `candle-transformers` 0.10 → 0.11, moved
  together. Moving `candle-nn` alone left two `candle-core` versions in the
  graph. `safetensors` 0.7 → 0.8 to match `candle-core` 0.11, which also drops
  a duplicate `tokenizers` version from the lock.
- From Dependabot: `scraper` 0.29, `tokenizers` 0.23, `model2vec-rs` 0.4,
  `toml` 1.2, `libc` 0.2.191; `ndarray` 0.17 and `proptest` 1.12 in
  `axiom_mesh_rs`; `actions/checkout` v7, `actions/upload-artifact` v7,
  `docker/setup-buildx-action` v4; and minimums for `httpx`, `fastapi` and
  `setuptools`.

### Added (developer-bug run)

- Mutation repair: bitwise slips (`^=`/`&=`/`|=`, spaced `^`/`&`/`|`) in the
  arithmetic class, and a late `arg-shift` class: ±1 on a bare argument or
  index, a swapped two-element subscript, swapped operands of a lone `+`.
  QuixBugs goes to 14/40, all correct. That is in-sample (the families were
  chosen from its failures); held out it is 9/40. The seeded corpora are
  unchanged at 39/40 and 38/40.

### Fixed (developer-bug run)

- **`axiom solve` hung forever on a fault that loops forever.** The
  supervisor's first run of the verify command had no time limit.
  `SupervisorOptions::timeout` now bounds it (`None` for `axiom run`); `solve`
  passes its verify timeout.
- **A Poly-JIT verify timeout aborted the whole solve.** It now means "this
  phase did not fix it", and the solve moves on to mutation repair.
- **A test killed on a timeout left nothing to localize.**
  `fault_locate::targets_from_command` falls back to the files the verify
  command names (pytest node ids reduced to paths). When the localizer follows
  a test's imports, helpers beside the test now rank after the code under test.
- **A timed-out verifier was not fully killed.** Only the direct child was
  killed, so a subprocess holding its pipes still blocked the read, and
  Poly-JIT's timeout left its verifier running. 15 orphaned QuixBugs
  verifiers were found hours later. Verifiers now run in their own process
  group, which is `SIGKILL`ed on timeout, with a bounded 2 s drain
  (`proc_bound`, shared by the supervisor, `run_verify_capture` and
  Poly-JIT).
- **Mutation repair could go green by editing a test.** Seen as a swapped name
  in a QuixBugs test's graph setup. Phase 2.1 now never edits a test file.
- `scripts/bench_repair.sh` bounds its own baseline and post-solve verify runs
  (`AXIOM_BENCH_VERIFY_TIMEOUT_SECS`, default 300).

### Held-out seeded faults

Held-out seeded-fault repair goes from 19/40 to **39/40** on the
original corpus and from 20/40 to **38/40** on a fresh seed (15/17 on its
faults outside the tuning set), with 0 rollback violations; 33/40 and 32/40
are verified correct against a strengthened test suite (0.6.0: 16/40)
([write-up](bench/results/SEEDED-HELDOUT-v2-2026-11-06.md)).

### Added

- **Mutation repair: four fault classes it could not repair before.**
  `api-swap` swaps a call for its paired opposite (`min`↔`max`,
  `floor`↔`ceil`, `first`↔`last`, `starts_with`↔`ends_with`, `any`↔`all`)
  and swaps `break`↔`continue`. `ident-swap` exchanges every occurrence of two identifiers on
  a line (the "wrong variable" bug). The boolean class now also adds a dropped
  negation to an `if`/`while` condition or a one-expression closure body, and
  unwraps `!( … )` without leaving redundant parentheses behind.
- **Mutation repair: candidate ordering.** After the lines the trace blames,
  the planner now tries the rest of the function that raised, then the bodies
  of the functions the failing test calls, then what those call, and only
  then the rest of the file. Identifier swaps (up to 15 per line) are tried
  off the blamed lines only after every other edit.
- `axiom_mesh_core`: seven tests for gaps that wrong-but-accepted repairs
  exposed (`StateSmoother` alpha range, feature-hash stability, explicit soft
  routing, bandit gating on either knob, dimension checks on every entry
  point, batch capacity above two).
- `scripts/reverify_accepted_patches.py`: re-run the repairs a benchmark
  accepted against a stronger test suite, to separate equivalent edits from
  ones the original tests could not reject. The planner diagnostic can now
  emit candidates by label (`AXIOM_PLAN_DIAG_EMIT`).
- `AXIOM_REPAIR_VERBOSE=1` makes `axiom solve` print each mutation candidate
  as it is tried (`candidate 3/64: swap-args:L296 …`).
- `tests/mutation_plan_diag.rs` (ignored): offline planner diagnostic that
  reports, for each fixture of a seeded corpus, where the reversal of the
  seeded fault ranks in the plan. Iterates in seconds without a verifier.

### Fixed

- **A correct first repair candidate could be rejected, depending on timing.**
  `agentic::mark_fresh` rounded *now* down to the whole second, so a
  candidate written in the same second the previous build started looked no
  newer than that build and cargo reran the stale binary. It now rounds up.
  Seen as a seeded fault whose fix was candidate 1: solved by hand, failed in
  the benchmark run.
- **Fault localization lost its evidence on long test output.** The solve
  loop kept only the last 4000 characters of the verifier's output. With
  `RUST_BACKTRACE=1`, or a run of failing property tests, that tail held no
  `file:line` citing the faulty code, so the mutation budget went to the wrong
  lines. The kept window is now the first 2000 and the last 4000
  characters (`solve::bound_trace`).
- **Debug builds of every model-loading subcommand panicked at start-up**
  (`Cannot drop a runtime in a context where blocking is not allowed`). The
  pipeline's context streamer built a `reqwest::blocking` client inside the
  CLI's async runtime; reqwest's debug-only guard panics on that. The client
  is now built lazily, only when a context API URL is configured. Release
  binaries were not affected.
- `scripts/seed_faults.py` copied the crate with its original mtimes, so a
  test binary a previous fixture left in the shared `CARGO_TARGET_DIR` looked
  up to date and the pristine-crate check ran the wrong code. Copies now get
  fresh mtimes, and `proptest-regressions/` is no longer copied.

## [0.6.0] - 2026-11-06

Headline: the deterministic repair loop now repairs **19 of 40 seeded faults in
maintainer-written Rust that it was never shown** (19 of 29 in the fault
classes it has an operator for, 0 of 11 in the classes it has none for), with
zero rollback violations; the authored gate is 32/32. A JSON-aware digest
removes ~89% of the tokens of MCP tool results that previously were never
digested. Two repair-loop bugs that made those numbers impossible are fixed.

**Crate versions:** `axiom_engine` 0.6.0 depends on `axiom_mesh_core` **0.2.0**
(was 0.1.0). The mesh crate's public API moved to `rand` 0.9 and `ndarray` 0.17
types, which is breaking under semver for 0.x, so it is a new minor version.

**Deprecations (removed in the next minor):** `crate::hamiltonian` and
`crate::q_manifold` (use `patch_ranker` / `branch_weights`); the
`/v1/hypervisor/quantum_coherent_state` route (use
`/v1/hypervisor/patch_ranker_state`); the `AXIOM_SANDBOX_*` env names and
`crate::sandbox` (use `AXIOM_COMPILE_VERIFY_*` / `compile_verify`).

### Security
- The CVM L2 page store is bounded across sessions: the in-memory index by
  `AXIOM_CVM_MAX_INDEX_BYTES` (256 MiB, least-recently-used sessions drop
  from memory and reload lazily) and the on-disk total by
  `AXIOM_CVM_MAX_TOTAL_BYTES` (2 GiB, least-recently-used session files are
  deleted, logged). Previously a client rotating `X-Axiom-Session-Id` could
  grow both without limit.
- `POST /v1/cluster/merge` confines its input and output paths to
  `AXIOM_MERGE_ROOT` (default: current directory and `~/.axiom`); previously a
  data-plane caller could read any readable file (the error echoed the path)
  and overwrite any writable one.
- `/mcp` over HTTP now requires the `AXIOM_API_KEY` (as `X-Axiom-Key` or
  `Authorization: Bearer`) when no `AXIOM_MCP_TOKEN` is configured, so the
  remote MCP transport is never more open than the data plane it reaches.
- `/v1/hypervisor/read` only reads regular files up to
  `AXIOM_VFS_MAX_READ_BYTES` (default 16 MiB); `/v1/hypervisor/mount` honours
  an optional `AXIOM_VFS_ALLOWED_ROOTS` allowlist.
- Request `max_tokens` is clamped to `AXIOM_MAX_GENERATION_TOKENS` (default
  16384) before any backend sees it; a huge value used to size a
  pre-allocation and could abort the process.
- Session stores are bounded: `POST /v1/sessions` returns `409` past
  `AXIOM_MAX_SESSIONS` (default 1024); proxy TTT sessions keyed by
  `X-Axiom-Session-Id` are capped at `AXIOM_MAX_TTT_SESSIONS` (default 512)
  with least-recently-used eviction.
- API key and MCP token comparisons are constant-time.

### Fixed
- **Repair candidates are no longer run from stale `cargo`/`make` builds.**
  The stale-`.pyc` fix below first set each candidate's mtime into the past;
  timestamp-based build tools rebuild only when a source is newer than its
  output, so every Rust candidate was verified against the old binary and
  rejected, correct fixes included (reproduced on a seeded `>=`/`>` fault:
  fails with a past mtime, passes with the current one). `mark_fresh` now
  stamps `max(now, previous stamp + 1 s, base + 1 s)` in whole seconds,
  process-wide and monotonic, and rollback writes in both `solve` and
  `agentic::Transaction` get the same stamp. Bursts of writes run ahead of the
  wall clock by at most the number of writes (bounded by the candidate cap).
- Candidate writes in the repair loops (`solve` single-file path and
  `agentic::Transaction`) now give the file a modification time distinct from
  the original and from every earlier candidate (`agentic::mark_fresh`).
  Python's `__pycache__` and other mtime+size-keyed caches otherwise ran the
  *original* bytecode for a same-size candidate written within the same
  second, so the verifier rejected correct fixes (observed: swapping two call
  arguments failed 2 runs in 3).
- `scripts/proof_loop.sh` parses the current `axiom bench` wording again
  (`token savings : N% (A -> B tokens)`, `expand round-trip K/K = 100.0%`);
  the compression row of the proof-loop table had read `parse-failed` since
  the bench output was reworded.
- `axiom init` now installs a usable production model: it fetches the
  tokenizer and architecture sidecar next to the checkpoint it already
  downloaded, and the proxy, MCP server, `bench` and `run` pick that model up
  automatically when `AXIOM_PRODUCTION_BPE` is unset (`0` forces the legacy
  model; `1` with `AXIOM_BPE_CKPT`/`AXIOM_TOKENIZER` is unchanged). Before,
  the download went unused and a fresh install ran the legacy kernel.
- `CvmStore::put` appends one line per page instead of rewriting the whole
  session file, and the digest hook's store write runs on the blocking pool.
- Heal memory is no longer re-read and re-parsed from disk on every proxied
  `/v1/messages` and `GET /v1/immunity`: a process-wide cache validated
  against the file's modification time and size serves the read-only paths.
- Request-reachable panics: multi-byte UTF-8 at a `max_bytes` cut in
  `axiom_compress_path` and in MCP `fetch` id decoding; integer overflow on a
  client-supplied tensor shape in checkpoint upload and DWE fragment apply.
- Local generation (`/v1/completions`, `/v1/chat/completions`, local
  `/v1/messages`) and `/v1/adapt` run on the blocking pool instead of stalling
  an async worker (and panicking in debug builds on a blocking HTTP backend).
- `axiombench` no longer overwrites the hand-maintained prose in `RESULTS.md`:
  it rewrites only a marker-delimited table block and carries over rows for
  pillars that did not run.

### Added
- **Held-out repair benchmark.** `scripts/seed_faults.py` seeds random
  single-token faults into maintainer-written code, keeps only those the
  project's own tests catch, and deliberately includes fault classes the
  engine has no operator for. `scripts/import_quixbugs.py` converts QuixBugs'
  40 Python bugs (MIT) with the answer key removed. `scripts/bench_repair.sh`
  isolates solver state per fixture, pins in-file test regions as the oracle,
  and reports verifier-written test artifacts (`proptest-regressions/`, …)
  separately from rollback violations. First measurement: 19/40 seeded faults
  in `axiom_mesh_core` (66% of classes with an operator, 0% without), 0
  rollback violations: `bench/results/SEEDED-HELDOUT-2026-11-06.md`.
- **Callee-focused repair planning for in-file tests.** The mutation planner
  never edits a `#[cfg(test)]`-gated item (only that item: production code
  after a test-only `use` or helper stays repairable); when the failure
  blames an assertion inside one, it ranks the bodies of the functions that
  test calls ahead of the rest of the file.
- **Hybrid CLI integration** (`docs/CLI-INTEGRATION.md`): vendor CLIs keep
  their own login and point their base URL at the proxy, which relays their
  `Authorization` / `x-api-key` / `anthropic-version` / `anthropic-beta` /
  `OpenAI-Organization` / `OpenAI-Project` headers upstream untouched and holds
  no key (`tests/cli_passthrough_proxy.rs`; `/v1/chat/completions` now relays
  the two OpenAI scoping headers like `/v1/responses` already did).
  `scripts/start_axiom_for_cli.sh` launches the loopback proxy, waits for
  `/readyz` and prints the per-CLI exports. Experimental `AXIOM_BACKEND=cli`
  (`--features experimental`, `src/cli_backend.rs`) drives a logged-in
  `claude -p --output-format json` subprocess for `axiom solve` / `axiom task`
  / server generation behind the backend router: version-pinned, fail-closed
  on unrecognised output, bounded timeout, prompt on stdin, never logged.
  `backend_live::router_from_env` now returns `Result` so a rejected backend
  configuration stops the server instead of silently falling back. Measured
  run: `bench/cvm/CLI-PROXY-RESULTS-2026-11-06.md` (pass-through correctness,
  not a savings number — see the file).
- **Structure-preserving JSON digest** (`json_digest.rs`) and a JSON-aware
  admission estimate for the CVM tool-result digest. Every MCP server answers
  in JSON, and minified JSON has few whitespace tokens: a 37 KB GitHub
  `list_workflow_runs` response is ~9.3k model tokens but 2.7k whitespace
  tokens, so under the whitespace-only estimate none of the five real MCP
  responses recorded from a Claude Code session (2.1k–9.3k tokens each) ever
  crossed the 4000-token threshold — they reached upstream verbatim and then
  sat in the cached transcript for the rest of the session. Admission now
  estimates text that parses as a JSON object or array by
  `max(whitespace, chars/4)` (code and prose are unchanged), and the digest
  keeps what an agent reasons about — top-level scalars, array lengths, the
  union of item keys with types and a value histogram for enum-like fields,
  a sample of items, truncated-string markers — instead of the prose
  head+tail excerpt that kept part of item 1 and the end of item N. Still
  recoverable verbatim through the same `axiom_expand` stub. Measured on the
  five responses in `bench/cvm/JSON-DIGEST-2026-11-05.md`.
- **General deterministic repair** (`mutation_repair.rs`, Phase 2.1 of
  `axiom solve`): a bounded (`AXIOM_REPAIR_MAX_CANDIDATES`, default 64),
  trace-ranked set of single-edit mutations — comparison operators,
  off-by-one (`± 1`, `len(x) - 1`, `range(n + 1)`, Rust `..=`), boolean
  inversion, `and`/`or`, arithmetic operators, swapped call arguments,
  missing `return` (incl. Rust's trailing `;`), shell `test` operators, and
  a misspelled import rewritten to the closest sibling source file when the
  trace reports a missing module (Python, Node, Ruby, C, Rust, Go wordings) —
  focused on the lines the verifier blames, never on an assertion line, each
  kept only if the real verifier goes green (byte-for-byte rollback
  otherwise). Replaces reliance on the four fixture-shaped Poly-JIT flips.
- **Test-aware localization** (`fault_locate::repair_targets`,
  `is_test_path`, `import_targets`): test/spec files are demoted and the
  modules they import (Python, JS/TS, Ruby, shell `.`/`source`, C
  `#include` → `.c`, Rust `mod`, Go package siblings) are tried first, so a
  failing assertion is repaired in the code under test, not in the test.
- `bench/repair/` + `scripts/bench_repair.sh`: the on-disk repair corpus and
  runner from `docs/BENCH-REPAIR.md` Phase 2. Each fixture is a `broken/` tree
  plus `meta.json` (and an optional `expected.diff`); the runner checks the
  no-op baseline fails, drives the real `axiom solve` CLI from a scratch copy,
  re-verifies, and checks rollback correctness on failure, labelling passes
  *authored fix* / *different fix*. Three seed fixtures; `--check` runs in CI.
- `axiom eval-agentic`: eighteen real-interpreter fixtures (python3, node,
  sh, go, rustc, gcc, ruby) — a genuinely buggy module plus a genuine test,
  with the test file pinned so an overfit "fix" scores as a failure; fixtures
  whose toolchain is missing are reported `[SKIP]` and excluded from the
  score; `--require-all` / `AXIOM_EVAL_REQUIRE_ALL=1` fails on any skip (set
  in CI). Suite is now 32 fixtures, 32/32, ~15 s.
- Graph memory is wired into recall: `recall` seeds spreading activation
  from its direct hits over `edges.jsonl` (plus `Supersedes` edges derived
  from records) and appends up to two graph-only neighbours after the
  direct hits, flagged `via_graph`; `axiom_recall` labels them `via=graph`.
  `RecallParams.graph = None` disables the stage.
- Release archives ship a `.sha256` sidecar; `install.sh` / `install.ps1`
  verify the download against it and refuse a mismatch
  (`AXIOM_REQUIRE_CHECKSUM=1` also refuses a release without a sidecar).
- README `## Terminology` table.
- `docs/LIVE-TEST-2026-10-29.md`: live test of the release binary against the
  published Hub checkpoints (all three load; 86.3% bench savings; CVM digest
  97.4% smaller with verbatim expand; every hardening probe behaves as
  specified) and five findings, led by `axiom init` not wiring the checkpoint
  it downloads.
- `axiom bench` prints a caveat when the input is too small (under ~500
  tokens) for the savings percentage to mean anything.
- Checkpoint sidecars (`<ckpt>.meta.json`) carry `schema_version` and
  `engine_version`; a sidecar from a newer engine is refused instead of
  silently loading with default dimensions.
- `docs/ENVIRONMENT.md`: a complete reference of every `AXIOM_*` variable.
- Dependabot configuration for the Rust crates, Python package, Docker base
  images and GitHub Actions.

### Changed
- **Physics-flavoured module names renamed to what they do.** `hamiltonian.rs`
  is `patch_ranker.rs` (`VariationalHamiltonian` → `PatchRanker`,
  `HamiltonianFault` → `FaultContext`, `HamiltonianOptimization` →
  `RankedPatches` with `selected_patch` / `selected_label` /
  `energies_tensor`, `QuantumRuntimeStatus` → `PatchRankerStatus`,
  `optimize_fault` → `rank`); `q_manifold.rs` is `branch_weights.rs`
  (`QuantumStateManifold` → `BranchWeights`, `ManifoldVariation` →
  `BranchCandidate`, `QuantumManifoldTelemetry` → `BranchWeightTelemetry`,
  `evolve_probabilities` → `reweight_by_energy`, `collapse` → `select`). The
  arithmetic is unchanged: a deterministic score per candidate and an
  exponential reweighting, which is what the names now say. `crate::hamiltonian`
  and `crate::q_manifold` remain as deprecated re-exports for one release.
  New route `GET /v1/hypervisor/patch_ranker_state` (`{"ranker": …}`);
  `/v1/hypervisor/quantum_coherent_state` stays for one release with its old
  `{"quantum": …}` body, and `PolyJitStatus` keeps the `quantum` JSON key so
  `/v1/hypervisor/jit_status` consumers are unaffected.
- Dependency majors brought current together, since they cross crate
  boundaries and could not land one PR at a time: `ndarray` 0.17 and `rand`
  0.9 in both `axiom_engine_rs` and `axiom_mesh_core` (the engine passes
  `ndarray` arrays and `impl Rng` into the mesh crate, so the versions must
  match), `thiserror` 2 in `axiom_mesh_core`, `tokenizers` 0.22 (its
  `with_pre_tokenizer`/`with_decoder` now take `Option`, `initial_alphabet`
  a `std::collections::HashSet`), `model2vec-rs` 0.3 (`--features
  fuzzy-embed` verified to compile). `rand` 0.9 renames applied
  (`random_range`, `random_bool`, `rand::rng()`, `StdRng::from_os_rng`).
  Supersedes Dependabot PRs #229, #234, #236, #242, #243, #244.
- `sandbox.rs` is `compile_verify.rs` (`CompileVerifier`): it is a
  `cargo check` verifier, not a process sandbox. New env names
  `AXIOM_COMPILE_VERIFY`, `AXIOM_COMPILE_VERIFY_ROOT`,
  `AXIOM_COMPILE_VERIFY_MAX_STEPS`, `AXIOM_COMPILE_VERIFY_TIMEOUT_SECS`; the
  `AXIOM_SANDBOX_*` names and `crate::sandbox` path remain as deprecated
  aliases for one release.
- CI gates `cargo fmt --check` and clippy at deny-warnings across all targets
  for every Rust crate graph; the tree was formatted once to establish the
  baseline.
- Docs: HTTP method/route drift fixed (`/v1/hypervisor/list` and `/stat` are
  `POST`; `DELETE /v1/ttt/sessions/{id}` and `GET /v1/responses` listed),
  MCP tool counts corrected to 17 (20 with `--features experimental`),
  undocumented subcommands (`axiom skeleton`, `axiom verify-suggest`) and
  flags added, plugin manifest and Helm `appVersion` synced to 0.5.0.

## [0.5.0] - 2026-10-13

Changes since v0.4.3. Note: 0.4.3 was tagged and released on GitHub but never
reached crates.io/PyPI (see Fixed), so registry users go from 0.4.2 to 0.5.0.

### Security
- The HTTP server binds to loopback (`127.0.0.1`) by default and refuses to
  start on a non-loopback address unless `AXIOM_API_KEY` is set (#220).

### Added
- `axiom-agent-task`: standard MCP protocol support (#199), `--min-mutation-score`
  gate and mutation-score warning on `task_start` (#201, #203).
- `axiom verify-suggest`: heuristic verifier suggestions and weak-spot
  detection (#198, #204).
- `axiom skeleton` subcommand with `--diagnostic` mode (#196, #205).
- Lossy-context invariant tests, CI guard and receipt visibility (#211).

### Changed
- Predictive-reasoning MCP tools are gated behind `--features experimental` (#212).
- Digest is gated on `axiom_expand` availability and keeps symbol names (#207, #209).
- Install/first-run docs and Python server size presets (#215).

### Fixed
- Publish workflow: the "wait for `axiom_mesh_core` on crates.io" step called
  the crates.io API without a User-Agent, which crates.io answers with HTTP 403,
  so it always timed out and 0.4.3 was never published to crates.io or PyPI.
  The step now sends a descriptive User-Agent.
- Publish: tolerate an already-published `axiom_mesh_core`.
- CI: invalid YAML in `ci.yml` (#218); flaky passkey_recall test (#210).
