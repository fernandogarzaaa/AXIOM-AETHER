# Axiom Swarm — Research-Backed Improvements & Blueprint

This document records a deliberate research + brainstorming pass over the
Axiom Swarm architecture: what the current literature on sparse routing,
multi-agent orchestration, and control theory suggests we're missing, what
was implemented as a direct result, and what's queued for a follow-up
decision rather than built unilaterally.

## Methodology

No installed skill in this environment covers "deep research" or
"brainstorming" specifically, so this pass adapted the methodology from two
public Claude Code skills rather than reinventing one:

- **[daymade/claude-code-skills — deep-research](https://github.com/daymade/claude-code-skills/blob/main/deep-research/SKILL.md)**:
  an 8-phase pipeline (scope → task board → parallel investigation →
  citation registry → evidence-mapped outline → draft → **mandatory
  counter-review** → verify). The load-bearing idea borrowed here is the
  counter-review step — every candidate improvement below was checked
  against its own downside/complexity cost before being recommended, not
  just its upside.
- **[Brainstorming skill for Claude Code](https://gist.github.com/scottd3v/1880ed0e96d5d7c6b1981fa3cb5767ef)**:
  develop 2-3 alternatives with trade-offs, lead with a recommendation, and
  get sign-off on anything that isn't a small, reversible change before
  building it. That's why the list below is split into **implemented now**
  (small, additive, non-breaking, already tested) and **queued for
  decision** (bigger, or genuinely has more than one reasonable design).

Research covered: sparse MoE routing and load balancing, Gumbel-Softmax vs.
Sinkhorn/optimal-transport alternatives for discrete routing, multi-agent
LLM orchestration + control-theory framings, AMD MI300X/ROCm inference
optimization, Kalman-filter sensor fusion, Rust actor-model supervision
trees, and bandit algorithms for cost-aware model routing.

## Implemented this pass

### 1. Dispatch timeout + quarantine (correctness fix, not just an optimization)

**Finding:** the Phase 2 `StdioTransport` (JSON-RPC over a child process's
stdio) had no timeout. A worker that wedges — crashes without exiting,
deadlocks, or just never writes a response — would hold the FSM's
`AwaitingWorkers` state forever, since nothing else can advance it. This
mirrors what the Rust actor-model research (kameo, ractor, bastion) treats
as a first-class concern: supervision trees exist specifically because a
crashed/stuck child must not silently stall its supervisor.

**What shipped:**
- `StdioTransport::spawn_with_timeout` wraps every `dispatch` call in
  `tokio::time::timeout`; `TransportError::Timeout` surfaces the failure
  instead of hanging. A late response to a timed-out call is safely
  discarded by the existing request-id correlation check.
- `axiom_swarm::health::NodeHealth` — a small, pure, unit-tested tracker
  that counts *consecutive* failures per node and reports quarantine at a
  configurable threshold. `main.rs` now calls `mesh.remove_node` when a
  node crosses it.
- The demo binary now dispatches two of its three workers to a **real**
  child process (`aether_worker`, speaking the wire protocol built in
  Phase 2) and wires the third to a transport that always fails, so the
  quarantine path is exercised end to end rather than only unit-tested.
  Observed demo output: `gemini` fails twice, gets quarantined, and the
  swarm still converges on `codex`/`claude` alone — sparse activation and
  fault tolerance working together.

**Counter-review:** a hard-coded timeout is a blunt instrument — a
legitimately slow worker looks identical to a wedged one. Accepted for now
because the alternative (no timeout) is strictly worse; a follow-up could
make the timeout adaptive per node (see "queued" section).

### 2. Residual-adaptive temperature annealing

**Finding:** comparative work on discrete routing mechanisms (Sinkhorn vs.
Gumbel-STE vs. ReinMax) reports that hard Gumbel-Softmax has a
**discretization gap** — committing to a hard winner early and then having
no way to escape a suboptimal assignment, because there's no gradient
signal correcting it later. Our IDC loop already has a natural annealing
signal sitting unused: the residual norm.

**What shipped:** `MeshConfig::tau_gain/tau_min/tau_max` (opt-in, `0.0`
gain by default so nothing existing changes). When enabled,
`tau_effective = clamp(tau_gain * |residual|, tau_min, tau_max)` — routing
is more exploratory while far from the goal and sharpens toward argmax as
the controller converges, the same explore-then-exploit shape as simulated
annealing. Demo now runs with annealing on.

**Counter-review:** this is a heuristic, not a principled fix for the
discretization gap (that would mean adopting Sinkhorn/ReinMax — see
"queued"). It's cheap, non-breaking, and directionally correct, which is
why it shipped now instead of waiting on a bigger routing-mechanism
decision.

### 3. Capacity-aware backpressure

**Finding:** MoE load-balancing research (expert-choice routing, MaxScore,
similarity-preserving routers) exists because unconstrained affinity-only
routing concentrates load on a few popular experts/workers. Our mesh had
no notion of "this node already has work in flight."

**What shipped:** `KineticNeuralMesh::mark_active`/`mark_idle` track
in-flight dispatch counts per node; `MeshConfig::capacity_penalty` (opt-in,
`0.0` by default) subtracts a per-busy-dispatch penalty from a node's
routing logits — the same shape as AIMD-style congestion control, applied
to routing instead of network throughput. Demo runs with a nonzero
penalty.

**Counter-review:** this is a soft penalty, not the hard per-expert
capacity limit MoE papers use (which drops/pads overflow tokens). A hard
capacity limit is a bigger behavior change — noted below rather than
shipped silently.

## Queued for your call before building

These have more than one reasonable design, or are large enough that
building the wrong one wastes real work — flagging for a decision rather
than picking unilaterally.

| Idea | Why it's promising | The actual decision needed |
|---|---|---|
| **Bandit-learned routing bias** — feed each node's empirical residual-reduction back into `WorkerNode.bias` (UCB/Thompson-style), so the mesh learns which workers are actually effective, not just structurally affine. | Directly matches the bandit-routing literature (MetaLLM, MixLLM) for cost/quality-aware online model selection — turns today's static bias into something that improves with use. | **Attribution**: when `fan_out > 1` and multiple workers land concurrently, how do you split credit for the residual reduction that tick? Per-worker dispatch (fan_out=1, current demo default) sidesteps this; multi-worker fan-out doesn't. |
| **Kalman/EMA smoothing on fused sensor state** — smooth `StateVector` across ticks before computing the residual, instead of raw per-tick fusion. | Classical control theory: unsmoothed sensor noise causes chattering actuation (routing flip-flopping tick to tick on noise, not signal). | **Determinism trade-off**: `fuse()` is deliberately deterministic/testable today. A stateful filter needs tuned process/measurement noise and changes what "same input → same output" means for the control loop. |
| **Expert-choice batch dispatch** — for the future case of routing *many* payloads to *few* workers at once (not today's one-payload-at-a-time loop), let workers choose their top payloads by affinity instead of payloads choosing workers, which is what actually solves load imbalance in MoE. | Directly cited as the highest-leverage 2025-2026 MoE improvement over naive top-1 routing. | Only applies once there's a real batch-dispatch use case; premature before then. |
| **Hierarchical/decentralized topology** — sub-swarms with their own Axiom Prime, per the centralized/decentralized/hierarchical + dynamic-adaptive taxonomy from the 2025-2026 multi-agent orchestration survey. | Matches how production multi-agent systems (LangGraph, AutoGen v0.4's actor-model rewrite) scale past a single controller. | Scale-driven; today's single-loop FSM has no evidence yet of being a bottleneck. Don't build ahead of the need. |
| **Property-based tests** (e.g. `proptest`) for FSM/mesh invariants — "pending count never goes negative under any WorkerDone interleaving," "forward() always returns exactly `min(fan_out, n_nodes)` active nodes." | The counter-review step above is exactly what property tests automate — adversarial input generation instead of hand-picked examples. | New dev-dependency; repo doesn't use `proptest` elsewhere today. Worth it once the state space (FSM × mesh × health) gets bigger than hand-written cases can cover well. |

## Sources

- [Mixture-of-experts with expert choice routing (Google Research)](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/)
- [Load Balancing Mixture of Experts with Similarity Preserving Routers](https://arxiv.org/abs/2506.14038)
- [A Survey on Mixture of Experts in LLMs](https://github.com/withinmiaov/A-Survey-on-Mixture-of-Experts-in-LLMs)
- [Expert Threshold Routing for Autoregressive LM (MaxScore)](https://arxiv.org/pdf/2603.11535)
- [Learning Latent Permutations with Gumbel-Sinkhorn Networks](https://openreview.net/forum?id=Byt3oJ-0W)
- [FlashSinkhorn: IO-Aware Entropic Optimal Transport on GPU](https://arxiv.org/pdf/2602.03067)
- [A Systematic Approach to Multi-Agent AI from Advanced Regulatory Control Theory](https://arxiv.org/html/2606.30877v1)
- [The Orchestration of Multi-Agent Systems: Architectures, Protocols, and Enterprise Adoption](https://arxiv.org/html/2601.13671v1)
- [LLM-Based Multi-Agent Orchestration: A Survey](https://doi.org/10.3390/fi18060326)
- [Multi-Armed Bandits Meet Large Language Models](https://arxiv.org/html/2505.13355v1)
- [Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey](https://arxiv.org/pdf/2603.04445)
- [Kalman Filters for Sensor Fusion](https://www.pleasedontcode.com/blog/kalman-filters-for-sensor-fusion)
- [kameo — fault-tolerant async actors for Rust](https://github.com/tqwewe/kameo)
- [AMD Instinct MI300X workload optimization — ROCm Documentation](https://rocm.docs.amd.com/en/docs-6.4.2/how-to/rocm-for-ai/inference-optimization/workload.html)
- [Best practices for competitive inference optimization on AMD Instinct MI300X](https://rocm.blogs.amd.com/artificial-intelligence/LLM_Inference/README.html)
- [daymade/claude-code-skills — deep-research SKILL.md](https://github.com/daymade/claude-code-skills/blob/main/deep-research/SKILL.md)
- [Brainstorming skill for Claude Code (gist)](https://gist.github.com/scottd3v/1880ed0e96d5d7c6b1981fa3cb5767ef)
