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
cargo run --bin axiom_swarm   # closed-loop demo with simulated workers
```

## Architecture

```text
                        ┌─────────────────────────────┐
                        │   axiom_swarm (Axiom Prime)  │
                        │  non-blocking FSM runner     │
                        │  Idle → Sensing → Routing →  │
                        │  AwaitingWorkers → Converged │
                        └──────┬───────────────▲───────┘
              commands         │               │ events (async task
       (route, dispatch)       ▼               │  completions)
                        ┌─────────────────────────────┐
                        │        axiom_core            │
                        │  KNM: gravitational field →  │
                        │  hard Gumbel-Softmax adhesion│
                        │  → sparse activation         │
                        │  IDC: fuse → residual →      │
                        │  CorrectionVector            │
                        └──────┬───────────────────────┘
                               │ Arc<str> payloads (zero-copy)
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │  axiom_mcp   │ │  axiom_mcp   │ │  axiom_mcp   │
      │ Mini Aether  │ │ Mini Aether  │ │ Mini Aether  │
      │  sidecar     │ │  sidecar     │ │  sidecar     │
      │ scrub+comprs │ │ scrub+comprs │ │ scrub+comprs │
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
| `mesh` | `KineticNeuralMesh`: node registry, runtime topology reconfiguration (`add_node`/`remove_node` rebuild the affinity matrix), gravitational-field computation (`A·p + bias`, optionally residual-modulated), and the `forward` pass returning an `Adhesion` (weights + sparse active set). |
| `gumbel` | `gumbel_softmax`: hard-variant Gumbel-Softmax for discrete topology routing, plus the relaxed distribution for telemetry. Seedable and deterministic under a fixed RNG. |
| `node` | `WorkerNode` / `NodeId` / `NodeKind`: affinity embedding + routing bias per worker. |
| `residual` | `StateVector` and `Residual` (`goal − current`), norms, convergence predicate. |
| `idc` | `SensorReading` (terminal / diff / test log), deterministic feature-hash `fuse` into state space, `IdcController` with `actuate` producing a `CorrectionVector` of `Actuation` commands — never conversational text. |

### `axiom_swarm` — Axiom Prime orchestrator (binary)

| Module | Contents |
|---|---|
| `fsm` | `SwarmFsm`: a pure, non-blocking transition function `(state, event) → commands`. No I/O, no payloads — only control state. Late/duplicate events are no-ops. |
| `main` | The async runner: executes FSM commands, spawns worker calls as tokio tasks (an in-flight LLM call is just a future `WorkerDone` event), shares payloads via `Arc<str>`, and drives the demo loop to convergence. |

### `axiom_mcp` — Mini Aether sidecars (library)

| Module | Contents |
|---|---|
| `filter` | `TokenScrubber`: deterministic, regex-free line rules — drop orchestrator-internal markers, redact credential-shaped assignments, strip control characters. Same input, same output, always. |
| `compress` | `compress_context`: strips conversational filler, keeps code fences verbatim, collapses prose whitespace; returns `CompressionStats` for cost accounting. |
| `sidecar` | `MiniAetherSidecar`: the per-worker pipeline (scrub → compress) emitting an `Arc<str>` `SidecarPayload`. |

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

## Status

Skeleton + first vertical slice. Real worker transports (the sidecar's MCP
wire protocol), learned affinity updates, and ROCm-accelerated batch routing
are intentionally not implemented yet — the demo binary simulates workers to
exercise the full closed loop end to end.
