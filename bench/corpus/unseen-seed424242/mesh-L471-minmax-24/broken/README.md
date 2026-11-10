# axiom_mesh_core

The canonical mesh crate for [AXIOM-AETHER](https://github.com/fernandogarzaaa/AXIOM-AETHER):
the "brain" of Axiom Mesh, owning the two load-bearing abstractions of the system.

* **Kinetic Neural Mesh (KNM)** — a sparse, dynamic routing graph over worker
  nodes. Every prompt payload projects a temporary "gravitational field" over
  the mesh; hard Gumbel-Softmax adhesion snaps the payload to the worker
  node(s) with the strongest pull, and only those nodes activate.
* **Intent-Driven Convergence (IDC)** — a control-theoretic feedback loop.
  Sensor readings (terminal output, file diffs, test logs) fuse into a
  `StateVector`; the residual `Goal − Current` drives an actuator that emits
  a `CorrectionVector` of concrete action commands rather than conversational
  text.

## Why this crate exists

The mesh code was previously vendored inside `axiom_engine` as the private
`crate::mesh_core` module (PR #190), because the crates.io name `axiom_core`
is squatted. This crate gives the mesh a canonical, publishable home under a
name that *is* available on crates.io, so `axiom_engine` depends on it as a
regular path (and eventually registry) dependency instead of carrying a
vendored copy.

## Usage

```toml
[dependencies]
axiom_mesh_core = "0.1"
```

```rust
use axiom_mesh_core::mesh::{KineticNeuralMesh, MeshConfig};
use axiom_mesh_core::node::{NodeId, WorkerNode};
```

## Layout

* `gumbel` — Gumbel-Softmax sampling with hard adhesion
* `idc` — Intent-Driven Convergence controller
* `mesh` — the Kinetic Neural Mesh routing graph
* `node` — worker node identity and metadata
* `residual` — residual state vectors (`Goal − Current`)

## License

MIT
