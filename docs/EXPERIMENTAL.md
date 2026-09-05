# Experimental surfaces and the `experimental` feature

Three subsystems are wired and reachable but **not benchmarked, not covered by
the [proof loop](PROOF-LOOP.md), and not part of the supported product
surface**:

| Subsystem | What it is | Status |
|---|---|---|
| **ChimeraLang DSL** | In-tree interpreter for the [ChimeraLang](https://github.com/fernandogarzaaa/ChimeraLang) AI-cognition language (`belief`/`inquire`/`resolve`/`guard`/`evolve`), running on the engine's `BetaBelief` + provenance substrate, with tamper-evident run certificates. | Compiles and runs; no evaluation harness; single-author use only. |
| **VFS hypervisor** | User-mode "neural VFS" — `axiom daemon` background process, `axiom mount <dir>`, and `POST/GET /v1/hypervisor/*` including `jit_run` (executes a caller-supplied command, can overwrite files; gated separately by `AXIOM_ENABLE_JIT_EXEC`). | Prototype. `jit_run` is a real execution surface — see [`SECURITY-AUDIT.md`](SECURITY-AUDIT.md). |
| **DWE swarm / fleet** | Distributed Weight Exchange: nodes gossip signed fast-weight fragments and verified patches. `axiom swarm`, `axiom fleet`, `/v1/fleet/*`, `/v1/cluster/*`, `/v1/swarm/*`. | Design + safety model are sound (Byzantine-robust, provenance-gated); cross-node behaviour is only smoke-tested (n=2, see `RESULTS.md`). |

| **Fuzzy dedup tier** | Extends S4 `prefix_diet` (exact-byte dedup) with near-duplicate detection: a small, pretrained, externally-distilled static embedding model (Model2Vec, `minishlab/potion-base-8M`, ~7.5M params, CPU-only) used purely as a similarity oracle -- never for generation. Targets exactly the gap S4's own write-up found: this project's real `~/.claude` rule set measured 0% exact-dedup gain because the English/translated/web-override variants are near-duplicates, not byte-identical. Fail-safe by construction: no configured backend means it degrades to exact-dedup-only, the same "no signal" convention `embedder.rs` uses for a degenerate vector. An optional GPU-first tier (`fuzzy-embed-gpu`, `src/gpu_embed.rs`) swaps in a larger contextual encoder (`BAAI/bge-base-en-v1.5`, ~109M params, sized to fit a 2GB VRAM budget) on CUDA, falling back to the same model on CPU, then to the small Model2Vec model, then to no backend -- see `AutoBackend` in `src/fuzzy_dedup.rs`. | Implemented and unit-tested (`src/fuzzy_dedup.rs`, `src/gpu_embed.rs`) behind `--features fuzzy-embed` / `--features fuzzy-embed-gpu`, off by default. **Not yet measured on real traffic, not benchmarked for whether the larger model catches materially more real-world duplicates than the small one, and not cache-safe in the S1 sense** (embedding similarity isn't guaranteed byte/version-stable the way a hash is) -- needs its own S5-style eval gate before touching anything at or before a `cache_control` breakpoint. Treat as DONE-BUT-UNGATED. |

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

## Entanglement map (grep, as of `dfb6bbc`)

Not every experimental-looking module is cleanly severable. What actually
references each candidate, outside itself:

| Module | Referenced by | Severable? |
|---|---|---|
| `chimera` | `entrypoint.rs`, `server/routes_tools.rs` | **Yes** — contained. |
| `state_predictor` / `trajectory_sampler` / `alignment_loop` / `predictive_tools` | each other + `mcp_stdio.rs` only | **Yes** — as one cluster. |
| `dwe` / `cluster` / `swarm_route` / `swarm_router` / `mesh_router` / `vfs` | **`server/prelude_state.rs`** (shared state *struct fields*), `server/run.rs`, `server/routes_fleet.rs`, `bin/axiombench/fleet.rs` | Only via a `#[cfg]`-field refactor of the server state — real work. |
| `daemon` | **`config.rs`**, **`tui.rs`**, `entrypoint.rs` | Entangled with core config + the TUI. Gating needs those refs gated too. |
| `hamiltonian` / `q_manifold` | **`poly_jit.rs`** (the core repair engine), `server/prelude_state.rs` | **No — stays core.** Do not gate. |

So the earlier "gate 14 modules" plan was wrong: `hamiltonian`/`q_manifold`
are load-bearing for the spearhead, and `daemon` is wired into core config.

## Staged plan

**Stage 1 — `chimera` only** (contained; one PR, CI-verified):

| File | Change |
|---|---|
| `Cargo.toml` | add `experimental = []` to `[features]`. |
| `lib.rs` | `#[cfg(feature = "experimental")]` on `pub mod chimera;`. |
| `cli.rs` | `#[cfg(feature = "experimental")]` on the `AxiomCommand::Chimera { .. }` variant **and** the `ChimeraCommand` enum. |
| `entrypoint.rs` | `#[cfg(feature = "experimental")]` on the `AxiomCommand::Chimera { command } => { … }` match arm. Variant + arm both gated ⇒ the `match` stays exhaustive in both feature states, no wildcard. |
| `server/routes_tools.rs` | `#[cfg(feature = "experimental")]` on `async fn post_chimera_run` (its `crate::chimera::…` refs are fully-qualified — no `use` to gate). |
| `server/routes_verify.rs` | Split the `guarded` builder: drop `/v1/chimera/run` from the chain, then `#[cfg(feature = "experimental")] let guarded = guarded.route("/v1/chimera/run", post(post_chimera_run));` **before** `.route_layer(require_api_key)` so the route keeps the API-key guard. |
| `server/tests.rs` | `#[cfg(feature = "experimental")]` on `test_chimera_run_endpoint_executes_a_program`. |
| `.github/workflows/ci.yml` | new job `experimental`: `cargo test --features experimental --locked`. |
| `README.md` / `CONTRIBUTING.md` | flip "moving behind" → "build with `--features experimental`" for ChimeraLang. |

**Stage 2 — predictive-reasoning cluster** (`state_predictor` + `trajectory_sampler`
+ `alignment_loop` + `predictive_tools`): gate the four `mod`s, the three MCP
tool-dispatch arms + tool-list entries in `mcp_stdio.rs`, the
`predict_states_blocking` helper, and the tool-list test `assert!`s. Make the
README "20 tools" count dynamic or feature-aware.

**Stage 3 — DWE swarm / fleet + VFS hypervisor**: the big one. Requires making
the `dwe` / `cluster` / `swarm*` / `vfs` fields on `server/prelude_state.rs`'s
state struct `#[cfg]`-conditional, which ripples through every constructor and
`.field` access in `server/run.rs` and the fleet/hypervisor route modules, plus
gating `daemon` alongside its `config.rs` + `tui.rs` refs, plus the `Swarm` /
`Fleet` / `Daemon` / `Mount` CLI variants + arms + `print_daemon_status`, plus
the `bin/axiombench` fleet pillar. Do this only with a local build loop.

### Acceptance (per stage)

- `cargo build --locked` + `cargo test --release --locked` pass with the feature
  **off**, and the gated surface is absent from `--help`, the route table, and
  the MCP tool list.
- `cargo test --features experimental --locked` passes with it **on**.
- `./scripts/demo_end_to_end.sh` still ends `FAIL 0` (Stage 3: its swarm step
  must be feature-gated in the script or the script built with the feature).

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
