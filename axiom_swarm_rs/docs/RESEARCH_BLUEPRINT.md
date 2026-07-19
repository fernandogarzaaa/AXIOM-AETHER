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

### 4. Bandit-learned routing bias (UCB1)

**Finding:** the bandit-routing literature (MetaLLM, MixLLM) frames
"which worker should get this payload" as an online learning problem —
exactly the gap between our mesh's static, hand-set `WorkerNode.bias` and
what workers actually turn out to be effective in practice. Re-examining
the attribution concern flagged in the first version of this document: the
worry was credit-splitting when multiple workers land concurrently under
`fan_out > 1`. On reflection this doesn't need restricting to `fan_out=1`
— if reward is measured against the environment state *at the moment each
dispatch resolves* rather than a snapshot from dispatch time, each
completion naturally scores its own marginal contribution, interleaved or
not. That resolved the open decision, so this shipped instead of staying
queued.

**What shipped:** `KineticNeuralMesh::record_outcome(id, reward)` updates a
running-mean quality estimate per node (kept separate from the static,
user-set `WorkerNode.bias` rather than mutating it — the two are
orthogonal: declared preference vs. empirically learned quality).
`MeshConfig::bandit_gain` weights that estimate into the field;
`MeshConfig::bandit_exploration` adds an independent UCB1-style bonus,
`exploration * sqrt(ln(total_visits) / node_visits)`, biasing toward
under-sampled nodes — useful right after a quarantined node is manually
re-added, or when a new worker joins. Both default to `0.0`. `main.rs`
computes `reward = residual_norm_before - residual_norm_after` around each
successful dispatch and reports it; the demo runs with both terms on.

**Counter-review:** zero-visit nodes are given a bonus computed with
`node_visits` floored at 1 rather than the textbook UCB1 branch (which
uses infinity for unvisited arms) — true infinity in a routing logit would
propagate NaN through the softmax in `gumbel_softmax` (`exp(x - ∞) = 0`
for finite `x`, but `exp(∞ - ∞) = NaN` for the infinite entry itself). The
floored version is a well-known "optimistic initialization" simplification
of UCB1 and stays numerically safe; it's a slightly weaker cold-start
bonus than the textbook version, which is an acceptable trade for not
needing a NaN guard on every routing call. Also: `reward_mean` /
`visits` reset on topology change (same as `in_flight`), so a node that's
removed and re-added starts its learned quality from scratch rather than
carrying over a stale estimate — reasonable given the identity behind a
given `NodeId` may have genuinely changed by the time it's re-added.

### 5. EMA sensor-state smoothing (composable, not demo-wired)

**Finding:** re-examining the "determinism trade-off" this was originally
queued on — the objection conflated two different things. `fuse()` being a
*pure function* (same readings in, same state out) is unaffected by
adding an optional, separately-instantiated smoother the caller chooses to
route raw state through; nothing about `fuse()` itself needs to change.
The actual open question was only "is a stateful filter deterministic
given a fixed input *sequence*?" — yes, trivially, an EMA recurrence has no
randomness in it.

**What shipped:** `axiom_core::idc::StateSmoother` — a plain exponential
moving average (`estimate = (1-alpha)*prev + alpha*raw`), not a full
Kalman filter (no process/measurement noise covariance to tune, at the
cost of not separately modeling sensor vs. process uncertainty — a
genuinely simpler tool for a genuinely simpler job). `StateSmoother::disabled()`
(alpha=1.0) is a pure pass-through. Tested against hand-computed EMA
values and a synthetic oscillating-signal case showing >90% variance
reduction.

**Deliberately not done:** *not* wired into the demo binary. The demo's
"environment" is a synthetic vector with no noise in it — smoothing a
noise-free signal only adds lag with nothing to show for it, which would
misrepresent the mechanism rather than demonstrate it. It's implemented,
tested, and exported as a composable primitive (same category as
`KineticNeuralMesh::center_of_mass`, which also exists but isn't forced
into the demo) ready to sit in front of a real, noisy `fuse()` call once
one exists.

## Explicitly deferred (still not built, and why)

| Idea | Why it's promising | Why it's still not implemented |
|---|---|---|
| **Expert-choice batch dispatch** — for the future case of routing *many* payloads to *few* workers at once (not today's one-payload-at-a-time loop), let workers choose their top payloads by affinity instead of payloads choosing workers, which is what actually solves load imbalance in MoE. | Directly cited as the highest-leverage 2025-2026 MoE improvement over naive top-1 routing. | No real batch-dispatch use case exists in this codebase yet — the FSM dispatches one payload per tick. Building the mechanism ahead of a caller that needs it is exactly the kind of speculative abstraction this project's own engineering conventions rule out. Revisit when a real multi-payload workload shows up. |
| **Hierarchical/decentralized topology** — sub-swarms with their own Axiom Prime, per the centralized/decentralized/hierarchical + dynamic-adaptive taxonomy from the 2025-2026 multi-agent orchestration survey. | Matches how production multi-agent systems (LangGraph, AutoGen v0.4's actor-model rewrite) scale past a single controller. | Scale-driven; today's single-loop FSM has shown no evidence of being a bottleneck. Don't build ahead of the need — there's nothing to test it against yet, so it'd ship unvalidated. |

Property-based tests (`proptest`, for the FSM/mesh invariants this
document's counter-review step implies) and EMA sensor smoothing were also
on this list; both are now implemented above.

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
