# Generation backends

Axiom's own model does compression / recall / drift; **text generation** is
delegated to a "brain" you choose. None of axiom's other features require a
trained model — the brain is pluggable and optional.

## Backend selection

`AXIOM_BACKEND` picks the generation brain:

| `AXIOM_BACKEND` | Brain | Cost | Notes |
|---|---|---|---|
| `bootstrap` (default) | Axiom's tiny local TTT model | free, offline | auto-trained by `axiom init`; weak at generation, fine for the cognition loops |
| `opendrop` | Local OpenDrop server | free, local | defaults `OPENAI_BASE_URL` to `http://127.0.0.1:11434/v1` — just `opendrop run <model>` |
| `openai` | Any OpenAI-compatible endpoint | depends | cloud OpenAI **or** a local server (OpenDrop, Ollama, vLLM) |
| `anthropic` | Claude (Anthropic API) | metered | needs `ANTHROPIC_API_KEY` |

Env knobs: `OPENAI_API_KEY`, `OPENAI_BASE_URL` (or `OPENAI_API_BASE`),
`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`.

> **Multi-provider router** (GPT + Claude together, with failover and belief
> consensus) ships as a library (`backend_router.rs`, fully tested) and grounds
> ChimeraLang `inquire`. Wiring it as a runtime `AXIOM_BACKEND=router` mode in the
> server request path is the next step (see "Roadmap" below).

## Local models via OpenDrop (zero per-token cost)

[OpenDrop](https://github.com/fernandogarzaaa/OpenDrop) serves any open-weight
model behind an OpenAI-compatible API. Point axiom at it — **works today** with
the `openai` backend:

```bash
opendrop run Qwen2.5-Coder-32B          # serves OpenAI-compatible API on :11434
export AXIOM_BACKEND=opendrop            # auto-targets http://127.0.0.1:11434/v1
axiom --mode server                      # axiom now generates via the local model
```

(Use `AXIOM_BACKEND=openai` + `OPENAI_BASE_URL` instead if your local server runs
on a non-default host/port, e.g. Ollama or vLLM.)

### Recommended models (all GGUF / OpenDrop-ready)

| Need | Model | ~Q4 size | License |
|---|---|---|---|
| **Default / code repair** | **Qwen2.5-Coder-32B-Instruct** | ~20 GB | Apache-2.0 |
| Laptop | Qwen2.5-Coder-7B / 14B, Llama-3.1-8B, Phi-4 | 5–10 GB | Apache / Llama / MIT |
| Reasoning (ChimeraLang beliefs) | DeepSeek-R1-Distill-Qwen-32B, QwQ-32B | ~20 GB | Apache |
| GLM family, runnable | GLM-4.5-Air (106B MoE) | ~40–60 GB | MIT |
| Frontier (multi-GPU) | DeepSeek-V3, GLM-4.5 (355B) | large | MIT |

> GLM-5.2 (754B, ~1.5 TB BF16) is impractical for most hardware — prefer
> Qwen2.5-Coder-32B as the default and GLM-4.5-Air as the high-end GLM option.

## Using your ChatGPT / Claude **subscriptions** (not API)

A ChatGPT Pro / Claude Pro subscription is **not** API access — axiom can't call
those chat endpoints (no official API; doing so violates ToS). The sanctioned,
zero-cost way to combine a subscription with axiom is the **MCP** path: let your
subscription app call axiom's tools (see [`MCP-CLIENTS.md`](MCP-CLIENTS.md)).
Today that works with Claude Desktop / Claude Code over stdio; a remote HTTP
transport for the ChatGPT connector is on the roadmap below.

## Is a trained model required?

No. `axiom init` bootstraps a tiny model offline, and every feature —
compression, self-healing, autonomous repair, grounding, fleet gossip,
ChimeraLang — runs on it. A bigger model (OpenDrop-served or cloud) only improves
*generation* and local-fingerprint quality. See the model tiers above.

## Roadmap (next backend work)

1. **`AXIOM_BACKEND=router`** runtime mode — wire `backend_router` into the server
   so GPT + Claude run together (routing/failover/consensus) on live requests.
2. **Remote MCP transport (HTTP/SSE)** — so the **ChatGPT connector** and Claude
   *remote* connectors can attach, not just local stdio (`MCP-CLIENTS.md`).
