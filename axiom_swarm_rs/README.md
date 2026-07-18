# Axiom Swarm

An autonomous, control-theoretic AI orchestration engine. The swarm is a
closed-loop control system: terminal output, file diffs, and test logs are
sensor data; the controller minimizes the **residual** between the current
system state and the goal state; workers are actuators reached through a
sparse, dynamically re-shaped routing mesh.

This workspace is independent of `axiom_engine_rs` (the TTT engine) and
builds standalone:

```bash
cd axiom_swarm_rs
cargo test
cargo run --bin axiom_swarm   # closed-loop demo — real stdio-transport workers
```

See [`docs/RESEARCH_BLUEPRINT.md`](docs/RESEARCH_BLUEPRINT.md) for the
research pass behind the fault-tolerance, annealing, and backpressure
mechanisms below, plus what's queued for a design decision before it gets
built.

## Architecture

```text
                        ┌─────────────────────────────┐
                        │   axiom_swarm (Axiom Prime)  │
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
(conditions what each worker actually receives), and **axiom_swarm is the
spine** (sequences the loop without ever blocking on a worker).

## Crates and modules

### `axiom_core` — KNM + IDC primitives (library)

| Module | Contents |
|---|---|
| `mesh` | `KineticNeuralMesh`: node registry, runtime topology reconfiguration (`add_node`/`remove_node` rebuild the affinity matrix), gravitational-field computation (`A·p + bias`, residual-modulated, plus opt-in capacity backpressure via `mark_active`/`mark_idle`), residual-adaptive Gumbel-Softmax temperature annealing (`tau_gain`), and the `forward` pass returning an `Adhesion` (weights + sparse active set). |
| `gumbel` | `gumbel_softmax`: hard-variant Gumbel-Softmax for discrete topology routing, plus the relaxed distribution for telemetry. Seedable and deterministic under a fixed RNG. |
| `node` | `WorkerNode` / `NodeId` / `NodeKind`: affinity embedding + routing bias per worker. |
| `residual` | `StateVector` and `Residual` (`goal − current`), norms, convergence predicate. |
| `idc` | `SensorReading` (terminal / diff / test log), deterministic feature-hash `fuse` into state space, `IdcController` with `actuate` producing a `CorrectionVector` of `Actuation` commands — never conversational text. |

### `axiom_swarm` — Axiom Prime orchestrator (binary)

| Module | Contents |
|---|---|
| `fsm` | `SwarmFsm`: a pure, non-blocking transition function `(state, event) → commands`. No I/O, no payloads — only control state. Late/duplicate events are no-ops. |
| `health` | `NodeHealth`: counts consecutive dispatch failures per node and reports quarantine at a threshold — the orchestrator's cue to call `mesh.remove_node` on a worker that keeps failing. |
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
* **A wedged worker can't stall the swarm.** Every `StdioTransport` call is
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

## Status

First vertical slice plus a fault-tolerance pass: real worker transports
(JSON-RPC over stdio, with timeout + quarantine) are implemented and wired
into the demo binary. Learned affinity updates (bandit-style bias from
observed outcomes), Kalman/EMA sensor smoothing, expert-choice batch
dispatch, hierarchical topology, and ROCm-accelerated batch routing are
deliberately not implemented — see
[`docs/RESEARCH_BLUEPRINT.md`](docs/RESEARCH_BLUEPRINT.md) for why each is
queued for a decision rather than built.
