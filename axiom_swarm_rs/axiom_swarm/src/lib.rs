//! Library surface for `axiom_swarm` — exposed so this package's own
//! integration tests (and, in principle, other workspace crates) can
//! exercise Axiom Prime's control logic — the FSM, node-health tracking,
//! hierarchical region composition — without needing to *be* the
//! `axiom_swarm` binary. `main.rs` is a thin async runner wired on top of
//! exactly these modules; it holds no logic of its own beyond sequencing.

pub mod fsm;
pub mod health;
pub mod hierarchy;
