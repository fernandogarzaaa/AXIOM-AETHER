# Changelog

Notable changes to the `axiom_engine` crate / `axiom-aether` wheels. Older
releases are described on the GitHub Releases page.

## [Unreleased]

## [0.8.0] - 2026-11-13

Headline: 0.8.0 is a hardening release. On a default install a web page
could get a fix stored that `axiom solve` later wrote into your source and
ran; fleet merges now need a key and an allowed Host and Origin. A stored fix
could overwrite uncommitted work or another project's file of the same name;
a fix now applies only to the exact file it was made against. `axiom solve`
no longer passes a check by editing it, and an interrupted solve leaves your
files as they were. The proxy's local trivial-turn short-circuit, which could
end an agent's turn with a canned reply, is now opt-in. Cross-origin access is
deny-by-default and `/metrics` is behind the API key. Several of these change
behaviour: read **Upgrade notes** before upgrading a server or a fleet.

Proof loop: repair 32/32, compression 85.7% (legacy tokenizer) and 87.6%
(production BPE) at 3608/3608 round-trip, trust gate 100% supported coverage
([table](bench/results/proof-loop-2026-11-13.md)). Held-out repair on a
seed-7 corpus from the current, harder seeder: 26/40, against 28/40 for 0.7.0
on the same corpus; the two lost fixtures have their fault inside a multi-line
assertion, which 0.8.0 deliberately no longer edits.

### Upgrade notes

- **A fleet merge needs a fleet key.** `POST /v1/patches/merge` and
  `POST /v1/immunity/merge` answer 401 unless the node has `AXIOM_FLEET_KEY`
  set and the export's HMAC verifies under it. Unsigned exports and raw
  heal-memory payloads are no longer accepted. Set the same `AXIOM_FLEET_KEY`
  on every fleet member; in the Helm chart, set `secrets.fleetKey`. Without
  it, the gossip CronJob logs 401s and nothing propagates.
  `AXIOM_FLEET_KEY_PREV` is not accepted on these routes (#300).
- **Merge routes check `Host` and `Origin`.** `Host` must be `localhost`, an
  IP literal, or a name listed in the new `AXIOM_MERGE_ALLOWED_HOSTS`
  (comma-separated); anything else gets 403. A request that carries an
  `Origin` must name an origin listed in `AXIOM_CORS_ORIGINS`, and `null` is
  never accepted. Peers that call each other by pod IP, as the chart's gossip
  does, keep working. If you merge through a DNS name, a Service name or a
  reverse proxy, add that name to `AXIOM_MERGE_ALLOWED_HOSTS` (#300).
- **Stored fixes from 0.7.0 are discarded.** A fix in the patch store
  (`~/.axiom/axiom_patch_memory.json`, beside the heal memory) is now a diff
  tied to one repository and to the SHA-256 of the file it was made against.
  0.7.0 stored whole files keyed by file name. Those entries are dropped when
  the store is loaded, gone from the file at the next save, and refused in
  peer exports. Nothing can convert them: the fixes are learned again as
  `axiom solve` repairs. 0.7.0 and 0.8.0 nodes cannot exchange patches, so
  upgrade every node of a fleet together (#301).
- **The local trivial-turn short-circuit is off by default.** In 0.7.0,
  `/v1/messages` answered small, clean tool results itself ("Acknowledged …",
  `stop_reason: end_turn`) unless `AXIOM_LOCAL_TRIVIAL` was exactly `off`.
  Now it runs only when `AXIOM_LOCAL_TRIVIAL` is `1`, `true`, `yes` or `on`,
  *and* the checkpoint is verified (`AXIOM_CHECKPOINT_SHA256` pinned and
  matching). With the flag on and no verified checkpoint, the server refuses
  to start and the error names both remedies. If you rely on the
  short-circuit, pin the checkpoint hash. Otherwise remove the flag (#302).
- **With `AXIOM_LOCAL_TRIVIAL` on, the tokenizer must be pinned too.** If
  `AXIOM_TOKENIZER` is set, `AXIOM_TOKENIZER_SHA256` must be set to its hash,
  or the server refuses to start. It also refuses when `AXIOM_TOKENIZER` does
  not load. It refuses as well when the checkpoint's sidecar names a
  tokenizer and `AXIOM_TOKENIZER` is unset or names a different file. The
  hash tokenizer needs no pin (#302, #306).
- **`/metrics` requires `X-Axiom-Key` when `AXIOM_API_KEY` is set.** Without
  the key it now answers 401, the same as the data plane; `/healthz` and
  `/readyz` stay open. The chart's ServiceMonitor cannot send that header. So
  with an API key set (`secrets.apiKey`, or `AXIOM_API_KEY` in
  `existingSecret`), Prometheus scrapes fail until you configure the scrape
  credential yourself. This includes the new `values-staging.yaml` and
  `values-prod.yaml` overlays, which enable the ServiceMonitor. With no API key
  set, `/metrics` is open as before (#284).
- **`axiom_session_residency` no longer carries raw session ids.** The
  `session_id` label is now the first 16 hex characters of the id's SHA-256,
  and at most 100 sessions are emitted. Update any dashboard or alert that
  matched raw ids. `axiom_active_sessions` and `axiom_quantized_sessions`
  still carry the totals (#284).
- **CORS is deny-by-default.** The server used to answer every origin with a
  permissive CORS layer. Now it sends no `Access-Control-Allow-Origin` unless
  the origin is listed in the new `AXIOM_CORS_ORIGINS` (comma-separated exact
  origins, e.g. `https://app.example.com,http://localhost:3000`). A browser
  app that calls the API from another origin must be listed. `*` is ignored
  with a warning. Requests without an `Origin` header (curl, SDKs,
  same-origin pages) are unaffected (#285).
- **The Helm chart pins image digests.** `image.digest`, `gossip.digest` and
  `tests.digest` now default to sha256 pins, and the chart renders
  `repo:tag@digest`. The digest wins at pull time. So if you override
  `image.tag` (or `gossip.image` / `tests.image`), set the matching digest or
  clear it with `--set image.digest=`. Otherwise you get the pinned image,
  whatever tag you asked for. `values-gpu.yaml` clears `image.digest` for this
  reason. The Dockerfile's `FROM` lines are pinned the same way (#281).
- **Readiness can now require a verified checkpoint (opt-in).** With the new
  `AXIOM_REQUIRE_VERIFIED_CHECKPOINT=1` (Helm `config.requireVerifiedCheckpoint`),
  `GET /readyz` returns 503 unless the checkpoint exists, its tokenizer is
  present, and its SHA-256 matches `AXIOM_CHECKPOINT_SHA256` (Helm
  `checkpoint.sha256`). When `AXIOM_TOKENIZER_SHA256` is also set, the
  tokenizer must match it. The server still starts and `/healthz` stays 200.
  Off by default, so existing deployments are unchanged. The new staging and
  prod overlays turn it on: replace their placeholder hashes before using them
  (#283, #298).
- **The container entrypoint verifies pinned downloads.** When
  `AXIOM_CHECKPOINT_URL` / `AXIOM_TOKENIZER_URL` are set together with
  `AXIOM_CHECKPOINT_SHA256` / `AXIOM_TOKENIZER_SHA256`, a download that does
  not match is refused and the container exits non-zero. A file already at
  the target path that does not match is deleted and downloaded again. After
  a refused download no checkpoint is left in place, so a rollback has to
  install a good one. Without a pinned hash, nothing changes except that the
  computed hash is printed (#282).
- **Helm: gossip and `helm test` send `X-Axiom-Key`.** This is not breaking.
  To require the key in-cluster, set the new `secrets.apiKey`; app pods, the
  gossip CronJob and the test pod all pick it up. With `existingSecret`, put
  it in that Secret under the key `AXIOM_API_KEY`. Chart 0.3.0 → 0.3.1 (#280).

### Known issues

- The Helm chart in this tag still pins `image.digest` to the `latest`
  image resolved on 2026-11-09, which predates 0.8.0, and the digest wins over
  `image.tag`. Set `image.digest` to the 0.8.0 image's digest (given in the
  GitHub release notes) or clear it with `--set image.digest=` and set
  `image.tag`.
- A Prometheus ServiceMonitor cannot send `X-Axiom-Key`, so with
  `AXIOM_API_KEY` set, the chart's scrape of `/metrics` gets 401 (see Upgrade
  notes).
- `axiom_stream_backpressure_stalls_total` is exported but nothing increments
  it yet; it reads 0 (#291).
- `axiom init` downloads the model from Hugging Face by default. Those files
  are not the ones listed in a release's `SHA256SUMS.txt`, so pinning
  `AXIOM_CHECKPOINT_SHA256` to a release hash makes the default fetch fail.
  Pin the hash `axiom init` prints, or point `AXIOM_CHECKPOINT_URL` /
  `AXIOM_TOKENIZER_URL` at the release assets.

### Security

- **An unauthenticated request could get code run as the user.** On a default
  install (no `AXIOM_API_KEY`, no `AXIOM_FLEET_KEY`), the merge routes stored
  any export that matched its own SHA-256. A web page could send it as a
  `text/plain` POST, with no CORS preflight. `axiom solve` later wrote the
  stored fix into the blamed file and ran the verify command on it. Merged
  immunity records were injected into upstream prompts. Merges are now
  HMAC-authenticated and Host/Origin-checked; see Upgrade notes (#300).
- **A stored fix could destroy work.** `axiom solve` wrote a whole stored file
  back over the current one when the tests then passed. That could discard
  uncommitted changes, or replace a different project's file with the same name
  and test command, and it reported `SOLVED`. A fix is now considered only
  when the repository and path match and the file hashes to the fix's
  pre-image. A fix whose diff does not reproduce its recorded result is
  discarded (#301).
- Any website could read API responses cross-origin through the permissive
  CORS layer. Cross-origin access is now an explicit allowlist (#285).
- `/metrics` was public and labelled series with raw, client-supplied session
  ids, with no bound on their number (#284).
- Checkpoint installs are verified before they are trusted. The model fetch
  used by `axiom init` (and `axiom daemon start`) now honours pinned hashes
  from
  `~/.axiom/config.toml` (`base_model_sha256`, `tokenizer_sha256`,
  `meta_sha256` under `[models]`; config wins over env), and from the new
  `AXIOM_META_SHA256`. A fresh install is validated (non-empty checkpoint,
  strict sidecar parse with a supported schema and sane dims, matching
  tokenizer name, tokenizer loads with the sidecar's vocab size) and deleted
  if it fails. A failed or truncated download no longer leaves a partial
  `.download` file behind (#282).

### Added

- Real token streaming on the local route.
  `POST /v1/chat/completions` with `"stream": true`, when routed to a local
  Ollama model, now sends each token as Ollama produces it. The route used to
  buffer the whole answer and send it as one body. A client disconnect aborts
  the upstream request. If the local stream cannot start, the request falls
  back to the cloud path. A streamed request may run up to 30 minutes (#289).
- New Prometheus series on `/metrics`:
  - `axiom_first_token_latency_seconds` (#289);
  - `axiom_stage_latency_seconds{stage="compress"|"route"|"forward"|"fingerprint"}`;
  - `axiom_process_rss_bytes` (Linux; 0 elsewhere);
  - `axiom_session_memory_bytes`, including compression sessions;
  - `axiom_stream_backpressure_stalls_total`, which is not yet incremented
    anywhere and reads 0.

  Stage latencies count completed stages only. Streamed Anthropic forwards
  are not timed (#291).
- `/readyz` responses include `"checkpoint": "verified" | "absent" |
  "mismatched" | "unverified"`. A 503 from the readiness gate says why
  (#283).
- Helm values `checkpoint.sha256`, `checkpoint.tokenizerSha256`,
  `config.requireVerifiedCheckpoint` (#283) and `secrets.apiKey` (#280).
- Helm overlays `values-staging.yaml` (one replica, readiness gate on, one
  CORS origin) and `values-prod.yaml` (three replicas, PDB `minAvailable: 2`,
  autoscaling, CORS deny-all). Both expect an `existingSecret` and have
  placeholder checkpoint hashes. A rollback and recovery runbook is in
  `docs/ROLLBACK-RECOVERY.md`, the prod-like deploy checklist is in
  `docs/PROD-LIKE-DEPLOY.md`, and `scripts/rollback_drill.sh --local` runs a
  rollback drill without a cluster (#298).

### Changed

- `axiom solve` now reports `source_patched=true` (as well as
  `fleet_patched`) when a stored fix changed a file (#301).
- A fix is keyed to its repository:
  - `git:<root commit>` for a git checkout with full history, which its clones
    share;
  - otherwise `path:<hash of the root directory>`, which also covers shallow
    clones and folders without git.

  A fix from one machine's non-git folder or shallow clone is never reused
  elsewhere (#301).
- `/v1/responses` sends one stable `Idempotency-Key` with every upstream
  attempt. It uses yours if you send one, or derives one from the request
  otherwise.
  - A network error or 5xx is retried with the same compressed payload under
    the same key. It no longer falls back to the uncompressed payload.
  - A 400 falls back to your original body under a separate key that stays
    the same if you retry. A 400 that reports an idempotency-key conflict is
    not retried.
  - 401, 403, 429 and other 4xx are never retried (#290).
- The Helm chart's gossip and test images now run at pinned digests, and
  `NOTES.txt` prints the digest (#281).
- `axiom eval-agentic`: the 12 fixtures that were solved by the removed
  rewrites are now shell programs with a real one-token bug, checked by
  behaviour. Fixture names changed (`shell-exit-flip` → `sh-self-naming-frame`,
  `fixture-marker` → `sh-wrong-arith`, `rust-assert-flip` → `rust-arrow-frame`).
  The suite is still 32 cases (#304).

### Fixed

- `axiom solve` no longer "repairs" a failing check by editing it. The first
  source-repair phase only knew fixture-shaped rewrites (`exit 1` → `exit 0`,
  a marker rename, replacing the file with a script that exits 0) and reported
  SOLVED when one made the command pass. It is removed from the solve loop;
  source repair starts at the mutation layer (#304).
- A source file longer than 4096 characters is no longer cut while a patch is
  being tried (`axiom solve`, and `POST /v1/hypervisor/jit_run`) (#304).
- `axiom solve` no longer leaves a candidate fix in your file when it is
  interrupted. The original of every file it writes to is saved to disk first
  (`~/.axiom/journal`, or `AXIOM_JOURNAL_DIR`) and restored on failure, on
  Ctrl-C / SIGTERM / SIGHUP (exit status 130 / 143 / 129), and by the next
  `axiom solve` after a kill. A file you edited after the kill is left alone
  and the saved original's path is printed. The journal directory and its
  files are owner-only; a directory other users could write to has its
  existing contents set aside, never restored. Two solves never repair the
  same file at once: the second one leaves that file alone (#304).
- `POST /v1/hypervisor/jit_run` no longer replaces a file longer than 4096
  characters with a two-line script when the file contains `throw `,
  `Write-Error` or a fixture marker (#304).
- A fleet export counts once. Delivering the same signed export again to
  `POST /v1/patches/merge` or `POST /v1/immunity/merge` (a retry, a sync job
  on a timer, a replay, or a node's own export posted back to it) used to
  count as a fresh confirmation each time: a stored fix's `verified_count`
  rose and a heal record's evidence compounded. It now answers
  `"already_merged": true` and changes nothing. A store remembers the last
  1024 exports it merged, in the store file, and the last 1024 it handed
  out, in `<store>.exports.json` beside it. `axiom immunity pull` says when
  a peer's export was merged before (#307).
- The patch store and the heal memory are saved by rename instead of being
  written in place. A save that is interrupted, or read while it is in
  progress, used to leave or show a partly written file, and a store that
  does not parse loads as empty. The old file now stays whole until the new
  one is complete. The file keeps its permissions, owner and group; a store
  that is a symbolic link stays a link; a store the process could not have
  written in place is refused. Where a file cannot be replaced (a single
  file mounted into a container, a directory the process may not create
  files in), or could only be replaced by a file with a different owner or
  group, it is written in place, as before. Access control lists and other
  extended attributes on a store are not carried over to the new file (#309).
- A multi-file edit by the agentic loop (`agent_task_mcp`) could leave earlier
  files modified when a later write failed. It could also delete an original
  it could not read instead of restoring it. Every target is now snapshotted
  before the first write, only files a write was attempted on are restored,
  and a failed restoration is reported in the task record ("ROLLBACK FAILED
  …") instead of being dropped (#276).
- A verifier that prints without limit no longer grows memory without bound.
  Each output stream keeps its first and last 512 KiB, joined by an
  `[output truncated: …]` marker that gives the total (#277).
- `POST /v1/hypervisor/jit_run` left the candidate patch in the source file
  when the verifier timed out or failed to run. The original is now restored
  on every exit except a passing report (#278).
- The Rust compile verifier (`AXIOM_COMPILE_VERIFY`) now kills `cargo check`
  and everything it started on a timeout, and removes its temporary workspace
  on every exit (#279).
- Mutation repair could make a test pass by flipping a comparison on the
  second line of a multi-line `assert (…)` / `assert!(…)`. Continuation lines
  of an assertion are now never mutated. The rule covers assertions in
  production code too, so a fault inside a multi-line `assert!` precondition
  is no longer repaired: on a seed-7 held-out corpus this costs 2 of 40
  fixtures that 0.7.0 repaired (28/40 → 26/40) (#286).
- Planning mutation candidates for a large file no longer holds a full copy of
  the file for every candidate. Planning a 2,000-line file for one candidate
  took about 222 MB before and about 22 MB after; the two figures come from
  different measurement setups (#287).
- `/v1/responses` with compression could leave temporary per-run sessions
  registered when the client disconnected while fingerprints were being
  generated (#288). A request cancelled at the wrong moment could also still
  start a full model adaptation that nobody was waiting for (#308).
- A local stream that fails or ends early now ends without `data: [DONE]`, so
  the client can tell it is incomplete (#289).

### Release and CI

- `publish.yml` treats crates.io's "already uploaded" error like "already
  exists", so a re-run does not fail after a duplicate upload (#275).
- Each release archive and checkpoint asset ships with a source SBOM
  (`.sbom.json`), a syft SBOM (`.sbom.syft.json`) and a provenance
  attestation (`.provenance.json`). The release fails if the syft SBOM is
  missing or invalid. Container images carry buildx SBOM and provenance, and
  tagged releases attach `image.provenance.json` and `image.sbom.syft.json`.
  To verify an asset, run `python3 scripts/generate_attestation.py verify`;
  see `docs/RELEASE-PROVENANCE.md` (#297).
- New CI jobs:
  - `session-concurrency-stress`, a graded p95 / RSS / session-registry stress
    harness at 1 to 32 concurrent sessions (#292; RSS slack widened from 16 to
    24 MiB after flaky misses, #305);
  - `perf-regression`, which compares against
    `bench/results/perf-baseline.json` and requires the demo-regression arm to
    fail (#299);
  - `pinned-images`, which runs `scripts/check-pinned-images.sh` (#281);
  - `helm`, which lints the chart and checks its auth renders (#280);
  - `release-gate`, which combines all gates into one PASS/FAIL. It still
    reports two gates as not evaluated; see `docs/RELEASE-GATES.md` (#296).
- AxiomBench gained:
  - a paired A–E runner (`axiombench paired`) with fixtures for four more task
    families (#293);
  - a regenerated 36-fixture "unseen" seeded-fault suite with no tuning
    overlap (#294);
  - per-arm promotion verdicts in `bench/results/PAIRED-VERDICTS-2026-11-11.md`
    (#295).
- Test helpers write fake executables from a child process, which fixes
  intermittent "Text file busy" failures (#310).

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
