# Changelog

Notable changes to the `axiom_engine` crate / `axiom-aether` wheels. Older
releases are described on the GitHub Releases page.

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
