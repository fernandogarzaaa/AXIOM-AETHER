//! # axiom_core — the "brain" of the Axiom Swarm
//!
//! This crate owns the two load-bearing abstractions of the swarm:
//!
//! * **Kinetic Neural Mesh (KNM)** — a sparse, dynamic routing graph over
//!   worker nodes. Every prompt payload projects a temporary "gravitational
//!   field" over the mesh; hard Gumbel-Softmax adhesion snaps the payload to
//!   the worker node(s) with the strongest pull, and only those nodes
//!   activate. See [`mesh::KineticNeuralMesh`].
//!
//! * **Intent-Driven Convergence (IDC)** — a control-theoretic feedback
//!   loop. Sensor readings (terminal output, file diffs, test logs) fuse
//!   into a `StateVector`; the residual `Goal − Current` drives an actuator
//!   that emits a [`idc::CorrectionVector`] of concrete action commands
//!   rather than conversational text. See [`idc::IdcController`].
//!
//! Deliberately absent from this crate: process management, LLM transport,
//! and token filtering. Those live in `axiom_swarm` (orchestrator FSM) and
//! `axiom_mcp` (Mini Aether sidecars) respectively — the brain computes
//! *where* and *what*, the hands do the touching.

pub mod gumbel;
pub mod idc;
pub mod mesh;
pub mod node;
pub mod residual;

pub use gumbel::gumbel_softmax;
pub use idc::{CorrectionVector, IdcController, SensorReading, StateSmoother};
pub use mesh::{Adhesion, BatchAdhesion, KineticNeuralMesh, MeshError};
pub use node::{NodeId, WorkerNode};
pub use residual::{Residual, StateVector};
