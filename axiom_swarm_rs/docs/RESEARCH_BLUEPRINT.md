# Axiom Swarm — Research-Backed Improvements & Blueprint

This document records a deliberate research + brainstorming pass over the
Axiom Swarm architecture: what the current literature on sparse routing,
multi-agent orchestration, and control theory suggests we're missing, what
was implemented as a direct result, and what's queued for a follow-up
decision rather than built unilaterally. A second pass completed the two
items that had been queued as "explicitly deferred" — expert-choice batch
dispatch and hierarchical topology — and, in the process of building real
tests for them, found and fixed a genuine correctness bug in a
previously-shipped feature (`MeshConfig::tau_gain` annealing). Every item
originally raised by the research pass is now either implemented or has an
explicit, reasoned "not yet, and here's specifically why" — nothing is
still sitting in an open-ended "queued" state.

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

## Correctness fix found while completing the roadmap

Completing the two remaining roadmap items surfaced a real bug in an
already-shipped, already-documented feature — worth recording plainly
rather than folding into the "implemented" list below as if it were new
work, since the honest story is "a previous pass got this wrong."

**What was wrong:** `gumbel_softmax`'s hard (discrete) decision was
computed as `argmax(softmax((logits + gumbel_noise) / tau))`. Softmax and
division by a positive scalar are both order-preserving, so this reduces
to `argmax(logits + gumbel_noise)` for *every* `tau > 0` — the hard
decision is provably invariant to temperature. That's not a coding
mistake; it's a known, correct property of the textbook Gumbel-max trick.
The mistake was building `MeshConfig::tau_gain` ("residual-adaptive
temperature annealing," documented as making routing "explore more while
far from goal, sharpen as it converges") on top of that function and
assuming it would affect routing decisions. It provably couldn't — the
computed `tau` was real, `effective_tau()`'s own unit tests passed
correctly, but nothing downstream of it in hard-routing mode ever changed
behavior as a result.

**Why it went undetected:** the existing test for temperature sensitivity
(`low_temperature_tracks_dominant_logit`) used a logit gap (10) so large
that `argmax(logits + noise)` wins reliably on its own, with or without
any temperature scaling — the test validated that a dominant logit wins,
not that `tau` had anything to do with it.

**How it surfaced:** a new hierarchical-topology test
(`select_region_picks_the_best_affinity_region`, described below) used a
*modest* logit gap and a low `tau`, expecting temperature to make the
decision reliable. It failed on a specific seed — the failure was the
signal that led to this analysis rather than a seed swap.

**The fix:** changed the perturbation to `logits / tau + gumbel_noise`
(unscaled noise) instead of `(logits + gumbel_noise) / tau`. Now `tau`
sets the real signal-to-noise ratio of the discrete decision — low `tau`
amplifies genuine affinity gaps over fixed-scale noise, high `tau` lets
noise dominate — which is what `tau_gain` annealing actually needs to do
its job. This is a deliberate, documented departure from the textbook
Concrete-distribution formula, not an attempt to reproduce it more
faithfully; see the docstring on `gumbel_softmax` for the full reasoning.
A new test, `tau_genuinely_changes_the_hard_decision`, checks the property
the old test couldn't: a modest gap is won reliably at low `tau` and
becomes close to a coin flip at high `tau`.

**Knock-on effects, both fixed:** the demo's `MockTransport`-backed
"gemini always fails" quarantine path had been relying on Gumbel noise
occasionally routing to it under the old (temperature-invariant) formula —
a coincidence, not a design. Under the corrected formula, with the demo's
existing tuning, routing became sharp enough that gemini stopped being
selected at all, silently breaking the quarantine demonstration. Fixed by
giving the three demo workers fixed, purpose-chosen affinities (`gemini`
tracks the goal direction closely, guaranteeing it wins the first routing
decision and gets quarantined deterministically) instead of the
previously random per-run ones — the demo's narrative no longer depends on
which way any RNG happens to break a tie, matching the standard this
project already holds itself to for reproducible test seeds.

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

### 6. Expert-choice batch dispatch

**Finding:** MoE load-balancing research consistently identifies the same
fix for pileup on popular experts: flip the direction of choice. Instead
of each token picking its best expert (which is what `forward` already
does — one payload choosing among nodes), each expert independently ranks
*all* tokens and claims its best ones up to a capacity. A popular node
can't be swamped, because it's capacity-bounded regardless of how many
payloads want it.

**Originally deferred on:** "no real batch-dispatch use case exists in
this codebase yet... revisit when a real multi-payload workload shows
up." That's still true of Axiom Prime's own control loop — it dispatches
one payload per tick by design, and this pass didn't change that.

**What shipped instead:** `KineticNeuralMesh::forward_batch(payloads,
capacity, residual)` as a standalone library capability, deliberately not
wired into the FSM's single-payload loop. Assignment is a deterministic
greedy global match: every `(payload, node)` pair is scored via the same
gravitational field `forward` uses, sorted best-first with a fully
deterministic tie-break, and walked once — a payload is claimed by the
first node in that order with room left. No Gumbel-Softmax draw: with
`capacity` doing the real work of preventing pileup, a deterministic
best-match walk is simpler than a stochastic one and no less principled,
and this runtime has no autograd tape to keep differentiable either way.
A payload no node has room for by the time its turn comes is dropped —
the same fate real capacity-constrained MoE routing gives overflow
tokens, rather than forcing an imbalanced assignment.

Demonstrated (not just unit-tested) via `axiom_core/examples/batch_dispatch.rs`
(`cargo run -p axiom_core --example batch_dispatch`): a worker that's
genuinely the better fit for every payload in a burst — naive per-payload
routing isn't "wrong" about any single choice, it just has no way to
notice it's already full — floods that worker 6/0 under naive routing,
vs. a clean 3/3 split under capacity-bounded batch dispatch.

**Counter-review:** a real multi-payload use case in Axiom Prime's own
loop still doesn't exist, so the honest scope here is "a tested,
demonstrated library capability," not "wired into the swarm end to end."
Building the FSM-level caller for it remains deferred until a workload
that actually needs it shows up — adding one speculatively now would be
exactly the kind of premature abstraction this project's conventions rule
out, even though the underlying mechanism itself is no longer premature.

### 7. Hierarchical/decentralized topology

**Finding:** the centralized/decentralized/hierarchical (+ dynamic-
adaptive) taxonomy from the 2025-2026 multi-agent orchestration survey
matches how production multi-agent systems scale past a single
controller — "sub-swarms with their own Axiom Prime."

**Originally deferred on:** "scale-driven; today's single-loop FSM has
shown no evidence of being a bottleneck... there's nothing to test it
against yet, so it'd ship unvalidated." The "unvalidated" half of that
concern is what this pass addressed — a design can be implemented and
thoroughly tested without needing to prove a real scale bottleneck first,
as long as it's built from primitives already proven at the leaf level
rather than a parallel new architecture guessed at from a taxonomy.

**What shipped:** `axiom_swarm::hierarchy::SwarmSupervisor` — a mesh over
*regions*, where each region is a whole `KineticNeuralMesh` (the same
type `forward` already uses for leaf routing) rather than a new recursive
node type. The supervisor's job is deliberately narrow: given a payload
and residual, pick a region via the exact same gravitational-field +
Gumbel-Softmax adhesion a leaf mesh uses to pick a worker, one level up.
What a region does once selected — its own mesh routing, its own
`SwarmFsm` loop — is exactly the machinery `main.rs` already runs for the
flat case; a `#[test]` (`regions_run_independent_swarm_fsm_loops`)
demonstrates two regions each driven by their own independent `SwarmFsm`
instance, orchestrated only by which region a `SwarmSupervisor` selects.

**Counter-review:** this composition approach was chosen specifically to
avoid the alternative — making `NodeKind` recursive (`SubSwarm(Box<KineticNeuralMesh>)`)
directly inside `axiom_core`. That would have required new `Clone`/serde
derives on a mesh type containing `ndarray` buffers, a guard against a
region containing itself, and conflating "leaf worker descriptor" with
"recursive routing structure" throughout `WorkerNode`'s existing simple
role. Composing two ordinary, unmodified `KineticNeuralMesh` instances
gets the same hierarchical routing behavior with none of that risk.
Real limitation, stated rather than hidden: `SwarmSupervisor` has no
`remove_region` — region indices are assumed stable for the supervisor's
lifetime, which holds today because nothing removes one. Dynamic region
retirement would need to keep the supervisor's region-selection mesh and
its `regions()` vector in sync, a real design question left for when a
caller actually needs to retire a region at runtime.

Property-based tests (`proptest`, for the FSM/mesh invariants this
document's counter-review step implies) and EMA sensor smoothing were also
on this list from an earlier pass; both are implemented above.

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
