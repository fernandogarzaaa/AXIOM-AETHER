# Experimental surfaces and the `experimental` feature

Three subsystems are wired and reachable but **not benchmarked, not covered by
the [proof loop](PROOF-LOOP.md), and not part of the supported product
surface**:

| Subsystem | What it is | Status |
|---|---|---|
| **ChimeraLang DSL** | In-tree interpreter for the [ChimeraLang](https://github.com/fernandogarzaaa/ChimeraLang) AI-cognition language (`belief`/`inquire`/`resolve`/`guard`/`evolve`), running on the engine's `BetaBelief` + provenance substrate, with tamper-evident run certificates. | Compiles and runs; no evaluation harness; single-author use only. |
| **VFS hypervisor** | User-mode "neural VFS" — `axiom daemon` background process, `axiom mount <dir>`, and `POST/GET /v1/hypervisor/*` including `jit_run` (executes a caller-supplied command, can overwrite files; gated separately by `AXIOM_ENABLE_JIT_EXEC`). | Prototype. `jit_run` is a real execution surface — see [`SECURITY-AUDIT.md`](SECURITY-AUDIT.md). |
| **DWE swarm / fleet** | Distributed Weight Exchange: nodes gossip signed fast-weight fragments and verified patches. `axiom swarm`, `axiom fleet`, `/v1/fleet/*`, `/v1/cluster/*`, `/v1/swarm/*`. | Design + safety model are sound (Byzantine-robust, provenance-gated); cross-node behaviour is only smoke-tested (n=2, see `RESULTS.md`). |

The predictive-reasoning MCP tools (`axiom_predict_states`,
`axiom_sample_trajectories`, `axiom_align_generation`) are already labelled
experimental and **untrained** in the README; they move under the same gate.

---

## The plan: a `cargo` feature named `experimental`

Add to `axiom_engine_rs/Cargo.toml`:

```toml
[features]
default = []
# ... existing: cuda, metal, tools, live-eval ...
# Experimental subsystems: ChimeraLang DSL, the VFS hypervisor, and the
# DWE swarm/fleet. Off by default so `cargo install`, the pip wheel, Docker,
# and release builds ship only the proven surface.
experimental = []
```

Then `cargo build` / `cargo test` / `cargo install` / the wheel / Docker / the
release workflow all exclude these subsystems, and the full surface is:

```bash
cargo build --features experimental
cargo test  --features experimental
```

CI keeps a dedicated job (`cargo test --features experimental`) so the gated
code still compiles and its own tests still run — it is demoted, not abandoned.

---

## File-by-file gating checklist (for the follow-up PR)

Each `mod`, subcommand variant, dispatch arm, route, and MCP tool for the three
subsystems gets `#[cfg(feature = "experimental")]`. Because `#[cfg]` on an enum
variant removes it from every `match`, the dispatch arms must be gated to match.

### `axiom_engine_rs/src/lib.rs` — module declarations

Gate: `alignment_loop` (4), `chimera` (12), `cluster` (15), `daemon` (18),
`dwe` (27), `hamiltonian` (34), `predictive_tools` (57), `q_manifold` (61),
`state_predictor` (74), `swarm_route` (76), `mesh_router` (77),
`swarm_router` (78), `trajectory_sampler` (83), `vfs` (88).

*(line numbers as of `dfb6bbc`; grep to re-confirm)*

### `axiom_engine_rs/src/cli.rs` — subcommand surface

- `#[cfg(feature = "experimental")]` on `AxiomCommand` variants: `Daemon`,
  `Mount`, `Swarm`, `Fleet`, `Chimera`.
- Same attribute on the sub-enums: `ChimeraCommand`, `DaemonCommand`,
  `FleetCommand`, `SwarmCommand`.

### `axiom_engine_rs/src/entrypoint.rs` — dispatch arms

Gate the `match` arms: `AxiomCommand::Daemon { .. }` (~675),
`AxiomCommand::Mount { .. }` (~699), `AxiomCommand::Chimera { .. }` (~1068),
`AxiomCommand::Swarm { .. }` (~1134), `AxiomCommand::Fleet { .. }` (~1205),
and the `fn print_daemon_status` helper (~1342). With the variants gated, the
outer `match` needs no wildcard — but confirm it still compiles exhaustively
under **both** feature states.

### `axiom_engine_rs/src/server/routes_verify.rs` — route registrations

Gate these `.route(...)` lines and their handler `fn`s:
`/v1/fleet/status`, `/v1/cluster/sync`, `/v1/cluster/merge`,
`/v1/hypervisor/mount|read|list|stat|jit_run|jit_status`,
`/v1/hypervisor/quantum_coherent_state`, `/v1/swarm/matrix_state`,
`/v1/chimera/run`. Build the experimental routes into a sub-`Router` added with
`#[cfg]` so the router expression stays valid when the feature is off.

Also check `server/routes_fleet.rs`, `server/routes_tools.rs`,
`server/prelude_state.rs` (state fields), `server/run.rs`, `server.rs`,
`swarm_route.rs`, `fault_locate.rs` — each references `dwe::` / fleet / chimera.

### `axiom_engine_rs/src/mcp_stdio.rs` — MCP tools

Gate the tool-dispatch arms and the tool-list entries for
`axiom_predict_states`, `axiom_sample_trajectories`, `axiom_align_generation`
(and the `predict_states_blocking` helper + the `assert!` in the tool-list
test). The tool count in the README (currently "20 tools") drops accordingly
when the feature is off — update it, or report it dynamically.

### `axiom_engine_rs/src/bin/axiombench/` (`tools` feature)

`main.rs` has `mod fleet;` and `fleet.rs` uses `dwe::`. Gate the fleet pillar in
axiombench behind `experimental` too, or make the pillar report "skipped
(experimental)" when the feature is off.

### Workflows

- `.github/workflows/ci.yml`: add a job `experimental` running
  `cargo test --features experimental --locked` so the gated code keeps
  compiling.
- `.github/workflows/release.yml`, `docker.yml`, `publish.yml`: no change —
  they already build without extra features, which is now the proven surface.

### Docs

- `README.md`: the *Experimental* section already lists these (added alongside
  this file). Once the feature lands, change "will be gated" → "build with
  `--features experimental`", and drop the gated routes from the default
  *HTTP API* list into the *Experimental* subsection.
- `CONTRIBUTING.md`: flip the `experimental` row in the features table from
  *(planned)* to shipped.
- `docs/PRODUCTION.md`: the fleet-mode section already says experimental.

### Acceptance

- `cargo build --locked` and `cargo test --release --locked` pass with the
  feature **off** and the three subsystems absent from `--help`, the route
  table, and the MCP tool list.
- `cargo build --features experimental --locked` and
  `cargo test --features experimental --locked` pass with everything present.
- `./scripts/demo_end_to_end.sh` still ends `FAIL 0` (it exercises the swarm
  step — that step must either be feature-gated in the script or the script
  built with `--features experimental`).
