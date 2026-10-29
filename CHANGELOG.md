# Changelog

Notable changes to the `axiom_engine` crate / `axiom-aether` wheels. Older
releases are described on the GitHub Releases page.

## [Unreleased]

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
- Checkpoint sidecars (`<ckpt>.meta.json`) carry `schema_version` and
  `engine_version`; a sidecar from a newer engine is refused instead of
  silently loading with default dimensions.
- `docs/ENVIRONMENT.md`: a complete reference of every `AXIOM_*` variable.
- Dependabot configuration for the Rust crates, Python package, Docker base
  images and GitHub Actions.

### Changed
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
