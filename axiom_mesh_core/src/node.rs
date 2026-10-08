//! Worker node descriptors for the mesh.

use serde::{Deserialize, Serialize};

/// Stable identifier for a worker node in the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub usize);

/// What kind of backend a node fronts. The mesh routes on affinity vectors,
/// not on this tag — the tag exists for dispatch and telemetry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    /// An LLM worker reached through a Mini Aether sidecar (Codex, Claude,
    /// Gemini, a local model, ...). The string is the backend name.
    Llm(String),
    /// A deterministic tool runner (compiler, test harness, formatter).
    Tool(String),
}

/// A worker node: an addressable executor plus the affinity embedding the
/// mesh uses to compute its gravitational pull on incoming payloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerNode {
    pub id: NodeId,
    pub name: String,
    pub kind: NodeKind,
    /// Affinity embedding, length = mesh dimension. Encodes what this node
    /// is good at; dot-product against payload embeddings yields routing
    /// logits.
    pub affinity: Vec<f32>,
    /// Static routing bias (log-space). Positive values make the node
    /// "heavier" — a cheap way to prefer local/cheap workers at equal
    /// affinity.
    pub bias: f32,
}

impl WorkerNode {
    pub fn new(id: usize, name: impl Into<String>, kind: NodeKind, affinity: Vec<f32>) -> Self {
        Self { id: NodeId(id), name: name.into(), kind, affinity, bias: 0.0 }
    }

    pub fn with_bias(mut self, bias: f32) -> Self {
        self.bias = bias;
        self
    }
}
