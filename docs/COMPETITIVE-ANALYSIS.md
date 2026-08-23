# AXIOM-AETHER — Competitive Analysis (2026-08-23)

**Basis and caveat, stated up front**: this analysis is written from public
documentation and general knowledge of each named project as of this
assistant's knowledge cutoff (January 2026), not from a fresh read of their
current source or a live benchmark run against them. Feature claims about
AXIOM are verified against this repository's code (cross-referenced to
[ARCHITECTURE-AUDIT.md](ARCHITECTURE-AUDIT.md)); feature claims about
competitors are **unverified against their current code** and should be
treated as directionally accurate, not as a substitute for reading their
docs before making a positioning decision. Where a claim is genuinely
uncertain, it's marked so rather than asserted.

## The field, grouped by what they actually compete on

AXIOM doesn't have one competitor — it overlaps partially with three
different categories. Comparing it to all of them on one axis (e.g., "which
has better memory") produces a false ranking; the honest comparison is
per-category.

### Category 1 — Agent memory frameworks: Letta, Mem0, Zep

| | Letta (formerly MemGPT) | Mem0 | Zep | AXIOM-AETHER |
|---|---|---|---|---|
| Core idea | An agent OS: memory-tiered context (in-context vs. archival), self-editing memory via tool calls | A memory *layer* you bolt onto any agent: extract, score, and retrieve facts as a hosted or embedded API | A memory service with a temporal knowledge graph (entities/relations change over time, queryable at any point) | A proxy/runtime the agent's traffic flows *through*, not a library the agent calls |
| Storage model | Its own agent state + memory store, typically Postgres-backed | Vector store + fact extraction (LLM-driven), hosted or self-run | Graph database (temporal), purpose-built for "what did we know when" | Multiple, purpose-specific: lossless JSONL facts (`memory_store.rs`), EMA-merged model *weights* (`vibe_memory.rs`), a directed edge graph (`graph_memory.rs`, new this pass) |
| Retrieval | Agent decides what to page in/out via explicit memory-management tool calls | Semantic search + recency/importance scoring | Graph traversal + temporal filtering | Two-stage hybrid: cheap relevance gate, then brute-force cosine top-k + recency/supersession rerank (`memory_recall.rs`); graph spreading activation exists (`graph_memory.rs::spread`) but is **not yet wired into recall** — see ARCHITECTURE-AUDIT.md §3.4 |
| Integration surface | Own SDK/agent framework; increasingly framework-agnostic | Drop-in memory API, integrates with LangChain/LlamaIndex/etc. | Drop-in memory API, similar integration story to Mem0 | **Transport-level**: an OpenAI/Anthropic-compatible HTTP proxy + MCP server. No SDK integration required — point an existing client at it |
| Model-level adaptation | None — memory is retrieved and stuffed into the prompt like any RAG system | None | None | **Genuinely different**: `vibe_memory.rs`'s EMA-merged fast-weight tensors are *model state*, not retrieved text — the model's own weights shift toward patterns it's seen, independent of what gets stuffed into the prompt. This is AXIOM's most defensible differentiator against this whole category (see below) |

**Where AXIOM is weaker than this category**: Letta/Mem0/Zep all have (per
public documentation) more mature multi-tenant hosted offerings, larger
adoption/ecosystem, and — critically — Zep's temporal graph and Letta's
self-editing memory are both *purpose-built and heavily tested* for exactly
the "what does the agent know and when did it learn it" problem, where
AXIOM's `graph_memory.rs` is new, unintegrated, and untested against real
retrieval workloads (§ AXIOMBENCH.md). If the buyer's problem is "I need
production-grade long-term agent memory today," these three are more proven.

**Where AXIOM is genuinely different, not just "also has memory"**: none of
Letta, Mem0, or Zep touch the model's own weights or activations — they are
all retrieval systems that change *what goes into the prompt*. AXIOM's TTT
fast-weight adaptation (`ttt_block.rs`) changes the *model's internal state*
in a way that's orthogonal to and compositional with retrieval memory —
theoretically closer to "the model gets a little bit fine-tuned on this
session" than "the model is shown more context." This is real, tested
mechanism (ARCHITECTURE-AUDIT.md §3.2), and no competitor in this category
does anything like it. **What isn't established**: whether this measurably
improves task success versus just doing better retrieval — see
AXIOMBENCH.md. The differentiation is architectural and defensible; the
*benefit* is unmeasured.

### Category 2 — Context/prompt compression: LLMLingua

| | LLMLingua (and LLMLingua-2) | AXIOM-AETHER |
|---|---|---|
| Core idea | Train a small model to score token-level importance; drop low-information tokens from a prompt before sending it to the target LLM | Two independent compression paths: structural (AST-aware, deterministic, reversible) and adaptive (fold into fast-weight state, return a short fingerprint instead of raw text) |
| Reversibility | Lossy by design — dropped tokens are gone; the target LLM never sees them | Structural path is **explicitly reversible**: `axiom bench` reports round-trip fidelity (100% signature recovery per `bench/RESULTS.md`, structure-only — see AXIOMBENCH.md for exactly what that measures), because it elides bodies, not signatures, and the elided content stays retrievable via `/v1/expand` |
| What gets compressed | General natural-language prompts | Two different things depending on path: **code/structured text** (structural path — AST-aware, works best on source files) and **conversational/session context** (adaptive path — the TTT fingerprint) |
| Model dependency | LLMLingua's compressor is itself a small trained model, shipped and versioned | Structural path needs no model at all (works with zero checkpoint); adaptive path needs the TTT checkpoint and its quality is tied to checkpoint training quality (honestly labeled — untrained checkpoint states this in API responses) |

**Where AXIOM is stronger**: the structural path's reversibility (you can
always get the elided body back via `/v1/expand`) is a real advantage over
LLMLingua's lossy token dropping for code-heavy or spec-heavy contexts,
where losing the wrong token can silently break a downstream tool call.
LLMLingua is a general-purpose, domain-agnostic compressor; AXIOM's
structural path is domain-specialized (source code, structured docs) and
loses nothing recoverable in that domain.

**Where AXIOM is weaker**: LLMLingua compresses arbitrary natural-language
prose well; AXIOM's structural path is largely a no-op on unstructured prose
(nothing to elide), and its adaptive path's compression *ratio* on prose is
unmeasured against LLMLingua's published numbers in this repo — see
AXIOMBENCH.md. This repo's compression numbers (`bench/ttt/RESULTS-2026-08-09.md`,
82–87% token savings) are measured on source code specifically, not on the
mixed conversational prompts LLMLingua's own benchmarks target — **the two
aren't measuring the same thing, and no doc in this repo should imply they
are** (checked: `RESULTS.md` already correctly scopes its claim to "over 120
files" of source, not general prompts — good practice worth preserving).

### Category 3 — Routing / gateway infrastructure: vLLM/SGLang, LiteLLM, Portkey

These are a different layer of the stack, and the honest comparison is
"complementary, with one real overlap area," not "competing."

- **vLLM/SGLang** are *inference servers* (how to serve a model fast — paged
  attention, continuous batching, speculative decoding). AXIOM is not an
  inference server in this sense at all for remote models (it forwards to
  Anthropic/OpenAI-compatible upstreams) and uses `candle` (not vLLM/SGLang)
  for its own local TTT model. **No real overlap** except that AXIOM's local
  model path is a (much smaller-scale) inference server too — comparing
  serving throughput would be comparing a research CPU/GPU TTT model against
  production-grade paged-attention servers, a comparison AXIOM would lose
  and shouldn't invite.
- **LiteLLM / Portkey** are *gateways*: unified API across providers,
  routing, retries, cost tracking, caching, virtual keys. This is the
  category AXIOM's `backend_router.rs`/`*_forwarder.rs`/`cost_ledger.rs`
  genuinely overlaps with — multi-provider routing with failover
  (`AXIOM_BACKEND=router`), an OpenAI/Anthropic-compatible surface, and cost
  metering (`cvm_store.rs`, the "CVM cost stack") are real, shipped features
  that do what LiteLLM/Portkey do at a basic level.

**Where AXIOM is weaker as a gateway**: LiteLLM and Portkey are
purpose-built for this and have (per public documentation) far more
provider integrations, more mature virtual-key/budget-management UX, and
dedicated teams whose whole product is the gateway. AXIOM's router is one
module among many in a much larger, differently-focused codebase, and this
audit found no evidence it's been used at gateway-scale (many providers,
high request volume) — the CVM cost pillar's own results
(`bench/ttt/RESULTS-2026-08-09.md`) are measured on 3 replayed session
records, explicitly labeled "indicative only."

**Where AXIOM is stronger, if you're already using it for compression/memory**:
you get provider routing, cost tracking, and TTT-based compression/memory in
one binary with one config surface, instead of composing a gateway (LiteLLM)
with a separate memory layer (Mem0/Zep) with a separate compressor
(LLMLingua). Whether that consolidation is a net win depends entirely on
whether the buyer wants AXIOM's specific compression/memory approach — it is
not a reason to choose AXIOM as *just* a gateway over LiteLLM/Portkey, which
are more mature at that one job.

---

## What AXIOM can defend, and what it can't (yet)

**Defensible claims** (verified against this repository's actual code and
tests):

- Structural compression is deterministic, reversible, and measured at
  scale on real source (2,181 files, 100% signature round-trip —
  `RESULTS.md`).
- TTT fast-weight adaptation is a real, tested, numerically-guarded
  mechanism, architecturally distinct from every named competitor's
  retrieval-only approach to "memory."
- Self-repair is verify-gated and reversible by construction, not
  trust-based (§3.5 of ARCHITECTURE-AUDIT.md) — a genuinely careful design,
  independently implemented at two different entry points.
- It runs as a transport-level proxy/MCP server, so it doesn't require
  adopting a new agent SDK the way Letta does, or wiring a memory API into
  application code the way Mem0/Zep typically do.

**Claims this repo cannot currently defend, and should not make**:

- That TTT adaptation, structural compression, or self-healing measurably
  improve *agent task success* — no benchmark in this repo measures this;
  see [AXIOMBENCH.md](AXIOMBENCH.md). Every existing benchmark measures a
  mechanism working (compression ratio, round-trip fidelity, repair
  pass/fail), never a downstream outcome.
- That AXIOM's memory subsystem is more capable than Zep's temporal graph or
  Letta's self-editing memory — it currently has less of the retrieval
  sophistication either offers (no temporal queries, no agent-directed
  memory management), and its own graph layer (`graph_memory.rs`) isn't
  wired into recall yet.
- That AXIOM's compression ratio numbers are comparable to LLMLingua's
  published numbers — they're measured on different content (source code
  vs. general prose) and aren't an apples-to-apples comparison.
- That AXIOM is a viable LiteLLM/Portkey replacement at gateway scale — it's
  never been measured at that scale in this repo.

## Positioning that survives scrutiny

The defensible thesis, narrower than "AXIOM beats X at Y": **AXIOM is the
only project in this comparison that changes the model's own adapted state
as a first-class mechanism, delivered as a transport-level proxy so it needs
no SDK adoption — everything else here is either a retrieval layer that
changes the prompt, a compressor that shrinks the prompt, or a gateway that
routes the request, and AXIOM does versions of all three but is the *only*
one that also does the fourth thing.** That's a real, checkable
differentiator. It is not evidence that any of it makes agents perform
better — that claim needs the benchmark work in
[AXIOMBENCH.md](AXIOMBENCH.md), which does not yet exist in this repo and
should be built before this positioning is used in anything customer-facing.
