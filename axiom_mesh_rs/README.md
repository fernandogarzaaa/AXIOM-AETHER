# Axiom Mesh

An autonomous, control-theoretic AI orchestration engine. Axiom Mesh is a
closed-loop control system: terminal output, file diffs, and test logs are
sensor data; the controller minimizes the **residual** between the current
system state and the goal state; workers are actuators reached through a
sparse, dynamically re-shaped routing mesh.

> **Naming note:** this project is unrelated to the existing `axiom swarm
> connect`/`axiom swarm immunity` CLI commands and `/v1/swarm/*` routes in
> `axiom_engine_rs` (peer-to-peer immunity/fleet sharing). An earlier
> version of this workspace was itself called "Axiom Swarm," which
> collided with that unrelated feature — renamed to Axiom Mesh to remove
> the ambiguity.

This workspace is independent of `axiom_engine_rs` (the TTT engine) and
builds standalone — see [Status](#status) below for the full command list.

See [`docs/RESEARCH_BLUEPRINT.md`](docs/RESEARCH_BLUEPRINT.md) for the
research pass behind the fault-tolerance, annealing, backpressure,
batch-dispatch, and hierarchy mechanisms described below, including a real
correctness bug it found and fixed along the way.

## Architecture

```text
                        ┌─────────────────────────────┐
                        │   axiom_prime (Axiom Prime)  │
                        │  non-blocking FSM + health   │
                        │  Idle → Sensing → Routing →  │
                        │  AwaitingWorkers → Converged │
                        │  NodeHealth: quarantine on   │
                        │  repeated dispatch failure   │
                        └──────┬───────────────▲───────┘
              commands         │               │ events (async task
       (route, dispatch)       ▼               │  completions)
                        ┌─────────────────────────────┐
                        │        axiom_core            │
                        │  KNM: gravitational field →  │
                        │  annealed hard Gumbel-Softmax│
                        │  → sparse, backpressured     │
                        │  activation                  │
                        │  IDC: fuse → residual →      │
                        │  CorrectionVector             │
                        └──────┬───────────────────────┘
                               │ Arc<str> payloads (zero-copy)
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │  axiom_mcp   │ │  axiom_mcp   │ │  axiom_mcp   │
      │ Mini Aether  │ │ Mini Aether  │ │ Mini Aether  │
      │ sidecar +    │ │ sidecar +    │ │ sidecar +    │
      │ StdioTransport│ │ StdioTransport│ │ (mock, demo)│
      │ (timeout)    │ │ (timeout)    │ │              │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             ▼                ▼                ▼
           Codex            Claude           Gemini
```

The separation is deliberate: **axiom_core is the brain** (decides *where*
payloads go and *what* correction to apply), **axiom_mcp is the hands**
(conditions what each worker actually receives), and **axiom_prime is the
spine** (sequences the loop without ever blocking on a worker).

## Crates and modules

### `axiom_core` — KNM + IDC primitives (library)

| Module | Contents |
|---|---|
| `mesh` | `KineticNeuralMesh`: node registry, runtime topology reconfiguration (`add_node`/`remove_node` rebuild the affinity matrix), gravitational-field computation (`A·p + bias`, residual-modulated, plus opt-in capacity backpressure via `mark_active`/`mark_idle` and opt-in bandit-learned quality/exploration via `record_outcome`), residual-adaptive Gumbel-Softmax temperature annealing (`tau_gain`), `forward` returning an `Adhesion` (weights + sparse active set), and `forward_batch` — deterministic expert-choice batch dispatch returning a `BatchAdhesion` (see `examples/batch_dispatch.rs`). |
| `gumbel` | `gumbel_softmax`: hard-variant Gumbel-Softmax for discrete topology routing, plus the relaxed distribution for telemetry. Seedable and deterministic under a fixed RNG; `tau` is a deliberate departure from the textbook Concrete-distribution formula so it genuinely affects the hard decision (see the module docstring — the textbook version is provably temperature-invariant for its hard sample). |
| `node` | `WorkerNode` / `NodeId` / `NodeKind`: affinity embedding + routing bias per worker. |
| `residual` | `StateVector` and `Residual` (`goal − current`), norms, convergence predicate. |
| `idc` | `SensorReading` (terminal / diff / test log), deterministic feature-hash `fuse` into state space, `IdcController` with `actuate` producing a `CorrectionVector` of `Actuation` commands — never conversational text — plus `StateSmoother`, a composable EMA pre-filter for noisy fused state. |

`examples/batch_dispatch.rs` (`cargo run -p axiom_core --example batch_dispatch`)
demonstrates `forward_batch` against naive per-payload routing on a burst of
similar work.

### `axiom_prime` — Axiom Prime orchestrator (lib + binary)

A small library exposing the orchestration logic, plus a thin `main.rs`
binary that wires it into an async runner. The split exists so the FSM,
health tracking, and hierarchical composition below can be exercised by
this package's own tests without needing to *be* the executable.

| Module | Contents |
|---|---|
| `fsm` | `PrimeFsm`: a pure, non-blocking transition function `(state, event) → commands`. No I/O, no payloads — only control state. Late/duplicate events are no-ops; `is_terminal()` is monotonic under any event sequence (property-tested). |
| `health` | `NodeHealth`: counts consecutive dispatch failures per node and reports quarantine at a threshold — the orchestrator's cue to call `mesh.remove_node` on a worker that keeps failing. |
| `hierarchy` | `MeshSupervisor`: a mesh over *regions* (each a whole `KineticNeuralMesh`, composed rather than made recursive) — the hierarchical/decentralized topology leg. Picks a region the same way a leaf mesh picks a worker; what a region does once selected is the same `PrimeFsm`-driven machinery `main.rs` runs for the flat case (see its tests for a two-region demonstration). |
| `main` | The async runner: executes FSM commands, dispatches to real `StdioTransport` workers as tokio tasks (an in-flight LLM call is just a future `WorkerDone` event), tracks in-flight/health per node, and drives the demo loop to convergence — routing around a quarantined node if one goes down. |

### `axiom_mcp` — Mini Aether sidecars (library)

| Module | Contents |
|---|---|
| `filter` | `TokenScrubber`: deterministic, regex-free line rules — drop orchestrator-internal markers, redact credential-shaped assignments, strip control characters. Same input, same output, always. |
| `compress` | `compress_context`: strips conversational filler, keeps code fences verbatim, collapses prose whitespace; returns `CompressionStats` for cost accounting. |
| `sidecar` | `MiniAetherSidecar`: the per-worker pipeline (scrub → compress) emitting an `Arc<str>` `SidecarPayload`. |
| `protocol` | Line-delimited JSON-RPC 2.0 `aether/dispatch` request/response types — the Mini Aether wire protocol. |
| `transport` | `WorkerTransport` (dyn-safe async dispatch trait), `StdioTransport` (real child-process transport with a per-call timeout), `MockTransport` (tests/demos). |
| `bin/aether_worker` | Reference worker process speaking the wire protocol over stdio — what `StdioTransport` spawns. |

## Design decisions

* **Hard Gumbel-Softmax, top-k by masked redraw.** Each fan-out slot is an
  independent discrete draw with the previous winner masked to `−∞`, so
  `fan_out = 1` is exactly classic hard Gumbel-Softmax. There is no autograd
  tape in this runtime, so the straight-through trick reduces to returning
  the one-hot; the soft distribution is preserved on the sample for future
  gradient-based topology learning.
* **Residual-modulated routing.** The gravitational field can include an
  alignment term against the *normalized residual direction*, so nodes whose
  affinity matches the remaining gap pull harder than nodes aligned with
  work already done.
* **Deterministic sensor fusion.** `fuse` feature-hashes readings (FNV-1a)
  into a fixed-dimension space — the whole control loop is testable with no
  model in the loop, and a learned encoder can replace it later without
  touching the control law.
* **Zero-copy payload path.** Conditioned payloads are `Arc<str>`; the
  orchestrator, mesh telemetry, and worker tasks share one buffer, and the
  sidecar only allocates when it actually rewrites content.
* **The FSM owns nothing but control state.** LLM calls are tokio tasks;
  their completions come back as events. Axiom Prime never blocks on a
  worker, and a slow backend can't stall routing for the rest of the mesh.
* **A wedged worker can't stall the orchestrator.** Every `StdioTransport` call is
  bounded by a timeout, and `NodeHealth` quarantines (removes from the
  mesh) a worker after repeated consecutive failures — the FSM keeps
  converging on whatever nodes remain rather than hanging in
  `AwaitingWorkers` forever.
* **Routing anneals and backs off, opt-in.** `MeshConfig::tau_gain` scales
  Gumbel-Softmax temperature with the residual (explore far from the goal,
  sharpen near it); `MeshConfig::capacity_penalty` applies soft
  backpressure against nodes with dispatches already in flight. Both
  default to `0.0` (disabled) so existing behavior is unchanged unless
  opted into.
* **Routing learns from outcomes, opt-in.** `KineticNeuralMesh::record_outcome`
  feeds each dispatch's actual residual-reduction into a running-mean
  quality estimate per node; `MeshConfig::bandit_gain` weights it into
  routing and `MeshConfig::bandit_exploration` adds a UCB1-style bonus for
  under-sampled nodes. Kept separate from the static, user-set
  `WorkerNode.bias` — declared preference and learned quality are
  orthogonal. Defaults to `0.0` (disabled).
* **Batch dispatch flips the choice direction, deliberately not wired into
  the FSM.** `forward_batch` lets nodes choose their best payloads
  (capacity-bounded) instead of payloads choosing nodes — the actual fix
  for MoE-style pileup on a popular node. It's a tested, demonstrated
  library capability; Axiom Prime's own control loop stays one-payload-
  per-tick because no real caller needs batching yet, and building one
  speculatively would be exactly the premature abstraction these
  conventions rule out.
* **Hierarchy by composition, not recursion.** `MeshSupervisor` routes to
  *regions* using an ordinary, unmodified `KineticNeuralMesh` where each
  node represents a whole region — not a recursive `NodeKind` variant
  inside `axiom_core`. A region is free to run its own `PrimeFsm` exactly
  like the flat demo does; hierarchy falls out of using the same proven
  primitive twice rather than a parallel new architecture.
* **Demo affinities are fixed, not randomly drawn.** The three demo
  workers' affinities are chosen so the routing/failure/quarantine/
  convergence story is deterministic by construction — not a function of
  which way Gumbel noise happens to break a tie this run. A prior version
  drew affinities randomly per seed, which meant whether the quarantine
  path even fired was luck; that's not a property a demo whose job is to
  demonstrate a feature should depend on.

## Status

Two passes: a first vertical slice plus fault-tolerance/online-learning
pass, and a second pass that completed the remaining research-blueprint
roadmap. Implemented: real worker transports (JSON-RPC over stdio, with
timeout + quarantine), bandit-learned routing bias, EMA sensor-state
smoothing, property-based invariant tests (`proptest`), expert-choice
batch dispatch (`forward_batch`, demonstrated via
`examples/batch_dispatch.rs`), and hierarchical topology
(`MeshSupervisor`, composed from unmodified leaf meshes). The second pass
also found and fixed a real correctness bug in the temperature-annealing
feature from the first pass — see
[`docs/RESEARCH_BLUEPRINT.md`](docs/RESEARCH_BLUEPRINT.md) for the full
account, including which mechanisms are wired into the demo binary versus
implemented-and-tested-but-deliberately-not-forced-in (the sensor
smoother; batch dispatch), and why. Every item the original research pass
raised now has either an implementation or an explicit, reasoned "why
not" — nothing is left in an open-ended queued state. ROCm-accelerated
batch routing remains out of scope (this workspace has no GPU-backed
compute path at all yet, on any hardware).

A third pass integrated `axiom_core` into the actual shipped product:
`axiom_engine_rs` now depends on it directly (as a cross-workspace path
dependency — Cargo resolves this cleanly even though `axiom_engine_rs`
isn't a member of this workspace) to power `axiom_engine_rs`'s
`mesh_router` module, an opt-in (`AXIOM_MESH_ROUTING=1`) replacement for
its local-Ollama model selector's static "first available candidate"
logic. Building that integration surfaced a real gap in `forward` itself
— no way to say "route among only these currently-eligible nodes without
losing what the mesh has already learned about the excluded ones" — which
is now `KineticNeuralMesh::forward_restricted`. See
`axiom_engine_rs/src/mesh_router.rs` for the consumer.

```bash
cd axiom_mesh_rs                                    # this workspace is standalone — commands below won't find a Cargo.toml from the repo root
cargo test --workspace                             # 69 tests: unit + integration + property-based
cargo run --bin axiom_prime                         # closed-loop demo
cargo run -p axiom_core --example batch_dispatch    # expert-choice batch dispatch demo
```
