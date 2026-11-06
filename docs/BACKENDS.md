# Generation backends

Axiom's own model does compression / recall / drift; **text generation** is
delegated to a "brain" you choose. None of axiom's other features require a
trained model — the brain is pluggable and optional.

## Backend selection

`AXIOM_BACKEND` picks the generation brain:

| `AXIOM_BACKEND` | Brain | Cost | Notes |
|---|---|---|---|
| `bootstrap` (default) | Axiom's tiny local TTT model | free, offline | auto-trained by `axiom init`; weak at generation, fine for the cognition loops |
| `opendrop` | Local OpenDrop server | free, local | auto-targets a local OpenDrop server at `http://127.0.0.1:11434` — just `opendrop run <model>` |
| `openai` | Any OpenAI-compatible endpoint | depends | cloud OpenAI **or** a local server (OpenDrop, Ollama, vLLM) |
| `claude` | Claude (Anthropic API) | metered | needs `ANTHROPIC_API_KEY` (the value the code checks is `claude`; older docs said `anthropic`) |
| `router` | **GPT + Claude + local together** | metered | routes by task (code→Claude, else→GPT), with failover; opt-in consensus |
| `cli` | **A logged-in vendor CLI as a subprocess** (`claude -p`) | whatever the CLI's login bills | **experimental** (`--features experimental`); no key needed; see [CLI backend](#cli-backend-experimental-no-keys-fallback) and the vendor-terms caveat |

`openai`, `opendrop`, and `router` all build the same internal multi-provider
router (with the local TTT pipeline as a deterministic last-resort fallback);
`openai`/`opendrop` simply pin every task to the one OpenAI-compatible endpoint.

### Router mode (`AXIOM_BACKEND=router`)

Generation is routed across providers: code-repair → Claude, reasoning/general →
GPT, with deterministic failover to the local pipeline if a provider errors. It
registers whichever providers are configured:

```bash
export AXIOM_BACKEND=router
export ANTHROPIC_API_KEY=sk-ant-...     # registers Claude
export OPENAI_API_KEY=sk-...            # registers GPT (or set OPENAI_BASE_URL for a local server)
export AXIOM_OPENAI_MODEL=gpt-4o        # optional, default gpt-4o
axiom --mode server
```

The local TTT pipeline is always registered as the last-resort fallback, so the
server still answers even if every cloud provider is down.

Env knobs: `OPENAI_API_KEY`, `OPENAI_BASE_URL` (or `OPENAI_API_BASE`),
`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`.

#### Consensus mode

Set `AXIOM_ROUTER_CONSENSUS=1` to ask two providers for every generation and
fuse their answers via `BetaBelief` instead of returning the first success:

```bash
export AXIOM_BACKEND=router
export AXIOM_ROUTER_CONSENSUS=1
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
axiom --mode server
```

Any value other than `0`, `false`, or `off` enables consensus. Off by default.

> **Multi-provider router** (GPT + Claude together, with failover and belief
> consensus) is live as `AXIOM_BACKEND=router` (see below) and also grounds
> ChimeraLang `inquire`.

### CLI backend (experimental, no-keys fallback)

```bash
cargo build --release --features experimental --manifest-path axiom_engine_rs/Cargo.toml
export AXIOM_BACKEND=cli
export AXIOM_CLI_COMMAND=claude        # default; bare name on PATH or absolute path
export AXIOM_CLI_TIMEOUT_SECS=120      # default; per-call wall-clock bound
export AXIOM_CLI_MODEL=haiku           # optional --model
axiom solve ... / axiom task ... / axiom --mode server
```

Axiom's model-needing paths (`axiom solve` LLM repair, `axiom task`, and server
generation behind the router — every task pinned to `Provider::Cli`, local
pipeline as last-resort fallback) spawn
`claude -p --output-format json --tools "" --no-session-persistence
--strict-mcp-config --mcp-config '{"mcpServers":{}}'` with the prompt on stdin.
Guarantees (unit-tested with a fake CLI in `src/cli_backend.rs`): the CLI's
`--version` must parse as `2.x (Claude Code)` or the backend refuses to run
(the server exits; the repair paths skip LLM repair with a logged reason); only
the `type=result / subtype=success / is_error=false / result:string` shape is
accepted, anything else is an error rather than a guessed completion; every
call is killed past the timeout; no shell, no prompt in argv, no prompt/output
logging; `max_tokens` is advisory (the CLI has no output cap).

This is a fallback for machines without an API key, **not a cost lever** —
nothing is compressed on this path and the CLI bills whichever account it is
logged into. Anthropic's current guidance says products that drive Claude
programmatically should use API-key authentication; read
[`CLI-INTEGRATION.md` § Vendor terms](CLI-INTEGRATION.md#vendor-terms-checked-2026-11-06)
before using it with a subscription login. For cost savings use the proxy
pass-through path on that page instead.

## Local models via OpenDrop (zero per-token cost)

[OpenDrop](https://github.com/fernandogarzaaa/OpenDrop) serves any open-weight
model behind an OpenAI-compatible API. Point axiom at it — **works today** with
the `openai` backend:

```bash
opendrop run Qwen2.5-Coder-32B          # serves OpenAI-compatible API on :11434
export AXIOM_BACKEND=opendrop            # auto-targets the local server at :11434
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
| Frontier (multi-GPU) | DeepSeek-V3, GLM-4.5 (355B), GLM-5.2 (744B MoE) | large | MIT |

> GLM-5.2 is a released MIT-licensed MoE (~744B total / ~40B active per token).
> Even as a sparse MoE it needs multi-GPU / lots of RAM for the full weights, so
> for most hardware prefer Qwen2.5-Coder-32B as the default and GLM-4.5-Air as a
> runnable GLM option; reach for GLM-5.2 only when you have the capacity for a
> frontier model.

## Using your ChatGPT / Claude **subscriptions** (not API)

A ChatGPT Pro / Claude Pro subscription is **not** API access — axiom can't call
those chat endpoints (no official API; doing so violates ToS). The sanctioned,
zero-cost way to combine a subscription with axiom is the **MCP** path: let your
subscription app call axiom's tools (see [`MCP-CLIENTS.md`](MCP-CLIENTS.md)).
This works with Claude Desktop / Claude Code over stdio **and** with the ChatGPT
connector over the remote HTTP transport (`AXIOM_MCP_HTTP=1`), which now also
exposes the standard `search`/`fetch` tools — see [`MCP-CLIENTS.md`](MCP-CLIENTS.md).

The other direction — the vendor's **own CLI** (Claude Code, Codex CLI, …)
keeping its login and sending its traffic *through* the Axiom proxy via a base
URL override — is documented in [`CLI-INTEGRATION.md`](CLI-INTEGRATION.md),
together with each vendor's current terms on using subscription credentials
that way. Where those terms restrict or do not clearly permit it, the compliant
route is an API key.

## Is a trained model required?

No. `axiom init` bootstraps a tiny model offline, and every feature —
compression, self-healing, autonomous repair, grounding, fleet gossip,
ChimeraLang — runs on it. A bigger model (OpenDrop-served or cloud) only improves
*generation* and local-fingerprint quality. See the model tiers above.

## Roadmap (next backend work)

1. ~~**ChatGPT standard-mode `search`/`fetch` aliases**~~ — **done**: Axiom now
   exposes `search` and `fetch` tools (over its semantic memory) alongside its
   native tools, so it works as a standard ChatGPT connector, not just Developer
   Mode (see `MCP-CLIENTS.md`).
2. ~~**Consensus on the live router**~~ — **done**: set `AXIOM_ROUTER_CONSENSUS=1`
   to enable two-model consensus with `BetaBelief` fusion (see above).
