# Axiom Phase 2 — Persistent Memory Layer (Design)

**Date:** 2026-06-06
**Status:** Approved design — pending implementation plan
**Author:** Brainstormed with Claude (Opus 4.8)
**Supersedes:** none (extends Phase 1 / Semantic v4)

---

## One-line

Axiom becomes a tiered, hierarchical, self-curating **memory layer** that silently
grounds Claude in your past work *and* answers direct memory queries — using the
TTT mechanism where lossy recall is acceptable, and exact stored text where Claude
must be correct.

---

## Why this, why now

Phase 1 proved Axiom's one genuine differentiator: **online Test-Time Training** —
a `[d_model, d_model]` fast-weight matrix `W̃` per layer that takes a real
self-supervised gradient step on every token at inference time (`ttt_block.rs:99`).
The model *learns from context by changing its weights*, O(1) memory per step, no
fine-tuning pipeline.

Token-saving (the proxy) was the easiest place to prove the mechanism. But `W̃` is
mathematically a **neural associative memory**: the update rule
`W̃ ← W̃ − η·(error ⊗ k)` learns key→value bindings, and recall is one matmul
`out = q × W̃` (`ttt_block.rs:123,147`). Nothing in the llama.cpp / vLLM / Ollama
world works this way. Phase 2 aims that mechanism at the **persistence /
long-term-memory bottleneck**: models forget everything between sessions and force
you to re-explain your codebase and decisions every time.

### Existing assets this builds on
- **Vibe memory** (`vibe_memory.rs`): EMA-merged persistent `W̃` ("codebase DNA").
- **Drift gate**: cross-entropy spike detector (clean ℒ≈4.1 vs anomaly ℒ≈7.2) — a
  built-in **out-of-distribution / surprise signal**, reused here as a salience scorer.
- **Skeleton + expand** (`skeleton.rs`, `/v1/expand`): the lossy-summary +
  lossless-on-demand pattern that good memory needs, already built.

---

## Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| **Phase 2 spine** | Persistent memory layer |
| **Consumer** | Both — silent auto-recall into Claude's prompt **and** an askable MCP query tool |
| **Storage fidelity** | **Tiered**: `W̃` gist (lossy, drives recall/ranking) + artifact store (lossless exact bodies; Claude only ever sees real text) |
| **Ingestion policy** | **Salience-gated** (drift-driven auto-capture) **+ explicit override** (remember/forget) |
| **Memory scope** | **Hierarchical**: personal layer (conventions/style, follows you) over per-project layers (repo specifics, never cross) |
| **Recall build** | **A — Hybrid two-stage**: cheap ambient `W̃` heat-check, then precise embedding retrieval for exact bodies |

---

## Architecture — memory middleware around the existing proxy

Today: `client → /v1/messages → compress heavy context → forward to Anthropic`.

Phase 2 wraps that with two hooks, both **purely additive** — the proxy is never
worse than Phase 1:

- **IN (recall, sync):** before forwarding, check memory; if relevant, inject a
  capped `<axiom_recall>` block into the prompt.
- **OUT (ingest, async):** after the exchange, salience-gate the content and write
  what is worth keeping.

### Honest engineering notes
- **Axiom is its own embedder.** Mean-pool the model's hidden states over a chunk →
  L2-normalized embedding vector. No external embedding model, no API call — keeps
  the single-Rust-binary property.
- **No vector DB (YAGNI).** At personal scale (one developer's projects ≈ thousands
  to tens-of-thousands of chunks), brute-force cosine is sub-millisecond. A flat
  on-disk store + linear scan is simpler, dependency-free, and fast enough. Add an
  ANN index only if/when scale demands it.
- **Reliability tension resolved by tiering.** Pure `W̃`→discrete-artifact addressing
  is lossy and unproven; we do **not** bet Claude's prompt on it. `W̃` drives the
  cheap ambient relevance signal; exact embedding retrieval pulls the real bodies
  that get injected.

---

## Components

Each is a focused module, consistent with the project's file-size discipline
(200–400 lines typical, 800 max).

| Module | Job |
|---|---|
| `memory_store.rs` | **Tier-2 lossless store.** Append-only JSONL + embedding sidecar, one partition per scope. Record = `{id, scope, ts, kind, body, embedding, drift_at_ingest, supersedes?, tombstone}`. Personal scale → flat file, linear scan. |
| `embedder.rs` | **Axiom-as-embedder.** Reuses the existing model via `inference.rs`: tokenize → forward → mean-pool hidden states → L2-norm. No new model. |
| `memory_recall.rs` | **Two-stage hybrid retrieval.** Stage 1: cheap `W̃` ambient "is anything relevant?" heat-check (gate — below threshold ⇒ skip, zero cost). Stage 2: embed query → cosine top-k over (personal ∪ project) → recency/supersession + drift rerank → budget-cap. |
| `memory_ingest.rs` | **Salience-gated writing.** Chunk → secret-scrub → drift-score → write if above salience threshold **or** explicit `remember`. Contradiction check links `supersedes`. |
| `memory_inject.rs` | **Prompt assembly.** Builds `<axiom_recall>` from ranked memories in *skeleton* form, each tagged `id + scope + date` so Claude can `axiom_expand` the exact body and the user can audit. |

**Record `kind` enum:** `decision | code | conversation | fix` (extensible).

**MCP tools (askable half):**
- `axiom_recall(query, scope?)` — explicit memory query → ranked memories.
- `axiom_remember(text, kind?)` — force-write a memory.
- `axiom_forget(id | query)` — tombstone.
- extend existing `axiom_expand` to pull memory bodies as well as compression bodies.

**Dashboard:** a "Memory" card (reusing the health-check card pattern added on
2026-06-06) — per-scope counts, recall hit-rate, tokens injected this session,
recent recalls (auditable list), silent-recall on/off toggle, and a budget slider.

---

## Data flow

### Ingest (async, after the response is returned)
```
exchange
  → chunk
  → secret-scrub (regex: API keys, tokens, secrets)
  → drift-score each chunk (cross-entropy via existing model)
  → (salient above threshold) OR (explicit remember)?
       → embed (Axiom embedder)
       → contradiction-check (cosine vs existing records in scope)
       → write record (+ supersedes link if conflicting)
       → EMA-merge gist into scope W̃
```

### Recall (sync, before forwarding upstream)
```
incoming request
  → W̃ ambient heat-check
       → below threshold? → skip entirely (zero cost), forward as Phase 1
       → above threshold?
            → embed query
            → cosine top-k over (personal ∪ current-project) store
            → recency / supersession + drift rerank
            → budget-cap (token ceiling, rank-truncate)
            → assemble <axiom_recall> skeleton block
            → inject into prompt
            → forward upstream
            → log injection (metrics + dashboard)
```

---

## The hard problems (handled, not hidden)

- **Contradiction / staleness.** A new record that semantically matches an existing
  one but differs is linked via `supersedes` + timestamp. Recall **surfaces the
  conflict** to Claude ("decided X on d1, then Y on d2 — latest is Y") rather than
  silently choosing. EMA decay ages out stale gist from `W̃`.
- **Token budget.** Hard cap on injected tokens (dashboard slider), rank-truncate.
  A memory layer that bloats context is self-defeating — skeleton-first,
  expand-on-demand.
- **Trust / transparency.** Every silent injection is logged and shown in the
  dashboard; it is togglable; the user can inspect exactly what was recalled.
- **Secrets / privacy.** Regex scrubber at ingest never memorizes keys/tokens; a
  scope can be marked no-memory.
- **Cross-scope leak.** Retrieval is hard-partitioned by scope. The personal layer
  is *conventions/style only* by policy — project specifics never cross repos.

---

## Error handling — fail-open everywhere

Any memory error (corrupt store, embed failure, empty store) → log, skip memory,
forward the request normally. Memory **never breaks the proxy**, mirroring Axiom's
existing fail-open ethos (the proxy already falls back to direct Anthropic routing
if down). Memory is strictly additive: an empty or broken memory behaves exactly
like Phase 1.

---

## Testing

- **Unit:** embedder determinism; cosine ranking correctness; salience threshold
  behavior; secret scrubber; supersession linking; **scope isolation** (a project-A
  query must never return project-B records); budget cap truncation.
- **Integration:** ingest→recall round-trip through the proxy; fail-open on a
  corrupt store; `remember` / `forget` tool behavior.
- **Eval harness** (extends the `eval_model` acceptance-suite pattern): seed known
  memories, query, measure **precision@k** and verify *exact* bodies are returned
  (no hallucinated recall). **Recall hit-rate is the headline metric.**
- **Reliability pairing:** recalled memories can be ChimeraLang-verified before
  Claude builds on them — closing the original "Axiom + ChimeraLang paired" loop
  (cheaper context *and* verified-correct grounding).

---

## Build phasing (de-risks by proving retrieval before betting the prompt on it)

- **2.0 — Store + embedder + askable recall (MCP only, no silent injection).**
  Prove retrieval quality with the eval harness. Lowest risk, immediately useful.
- **2.1 — Salience-gated ingest + contradiction/supersession + secret scrub.**
  Memory fills itself.
- **2.2 — Silent recall + injection + budget + dashboard card.** The ambient magic
  — only after 2.0 proves recall is trustworthy.
- **2.3 — Hierarchical personal layer + ChimeraLang verification pairing.**

> The ordering is the central risk-management decision: 2.0 deliberately ships the
> askable tool **without** silent injection, so recall precision is proven on real
> data before Claude's prompt quality depends on it.

---

## Out of scope (YAGNI for Phase 2)

- Vector DB / ANN index (flat brute-force is enough at personal scale).
- A separate embedding model (Axiom embeds itself).
- Multi-user / shared-team memory sync (personal scale first).
- Pure neural `W̃`→artifact addressing (research-grade; tiering avoids needing it).

---

## Success criteria

1. **Recall precision@5 ≥ a bar set in the eval harness** on seeded memories, with
   exact bodies returned (zero hallucinated recall).
2. **Proxy never regresses:** with memory empty/disabled, behavior is byte-equivalent
   to Phase 1; any memory error fails open.
3. **Scope isolation holds:** no cross-project leakage in tests.
4. **Net context win:** silent recall improves answer grounding without exceeding the
   configured token budget (memory pays for itself).
