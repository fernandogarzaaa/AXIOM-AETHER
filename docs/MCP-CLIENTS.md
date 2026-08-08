# Running axiom as an MCP server (use it from Claude / ChatGPT / Cline)

Axiom ships an MCP server that exposes **20 tools** -- `axiom_compress_path`,
`axiom_evaluate_drift`, `axiom_expand`, `axiom_remember`, `axiom_recall`,
`axiom_forget`, `axiom_verify`, `axiom_validate_epistemic`, `axiom_immunity`,
`axiom_status`, `search`, `fetch`, `axiom_post_task`, `axiom_claim_task`,
`axiom_task_result`, `axiom_list_tasks`, `axiom_channels`, plus three
experimental, untrained predictive-reasoning tools --
`axiom_predict_states`, `axiom_sample_trajectories`,
`axiom_align_generation` -- to an LLM host. In this mode **the host model is
the brain and axiom is its toolkit**, which is the sanctioned,
zero-API-cost way to combine your ChatGPT/Claude *subscription* with axiom
(you drive your own app; it calls axiom's tools). The exact count is
enforced by a test (`tools_list_exposes_tools_with_schemas` in
`axiom_engine_rs/src/mcp_stdio.rs`), so it won't silently drift out of sync
with this doc again.

> **Want your AI agent to set this up for you?** See
> [`AGENT-SETUP.md`](AGENT-SETUP.md) for copy-paste prompts that make **Codex**
> (ChatGPT) or a **Claude** agent build, run, and connect Axiom end to end --
> including the Windows-build and `404 /mcp` gotchas. For **Cline** (VS Code),
> see [`CLINE-SETUP.md`](CLINE-SETUP.md). This page below is the transport
> reference.

## Claude Code plugin -- one-command install (recommended)

This repository is also a Claude Code **plugin marketplace**, so you can install
Axiom's MCP server without hand-editing any config file. From inside Claude Code:

```text
/plugin marketplace add fernandogarzaaa/AXIOM-AETHER
/plugin install axiom@axiom-aether
```

The plugin registers the `axiom` MCP server (`axiom --mode mcp`) for you.

**Prerequisite:** the `axiom` binary must be on your `PATH` -- the plugin
manifest declares the server, it does not compile the engine. Install it with
`scripts/install.sh` (Linux/macOS), `scripts/install.ps1` (Windows),
`cargo install --path axiom_engine_rs`, or `pip install axiom-engine`. Verify
with `axiom --version` before installing the plugin.

To pin a trained checkpoint instead of the auto-bootstrapped model, use the
manual `.mcp.json` route below and append
`"--checkpoint", "checkpoints/axiom_production_bpe.bin"` to `args`.

Manifest files: [`.claude-plugin/plugin.json`](../.claude-plugin/plugin.json),
[`.claude-plugin/marketplace.json`](../.claude-plugin/marketplace.json), and a
drop-in [`.mcp.json`](../.mcp.json) at the repository root for clients that read
it directly. The root `.mcp.json` is kept schema-only (no comment keys) so a
strict client cannot reject it; setup notes live here instead. It requires the
`axiom` binary on `PATH`, same as the plugin.

## Claude Desktop / Claude Code / Cline -- works today (stdio)

Axiom's MCP server speaks JSON-RPC over **stdio**, the transport Claude Desktop,
Claude Code, and Cline use for local MCP servers.

**Claude Desktop** -- add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "axiom": {
      "command": "axiom",
      "args": ["--mode", "mcp", "--checkpoint", "checkpoints/axiom_production_bpe.bin"]
    }
  }
}
```

(Omit `--checkpoint` to use the auto-bootstrapped model.)

**Claude Code** -- register the same server:

```bash
claude mcp add axiom -- axiom --mode mcp
```

**Cline (VS Code)** -- add to `cline_mcp_settings.json` (see
[`CLINE-SETUP.md`](CLINE-SETUP.md) for the full configuration):

```json
{
  "mcpServers": {
    "axiom": {
      "command": "axiom",
      "args": ["--mode", "mcp", "--checkpoint", "checkpoints/axiom_production_bpe.bin"]
    }
  }
}
```

Then ask Claude/Cline to "compress this repo with axiom" or "check this code for
drift" and it will call the tools.

## ChatGPT connector / Claude remote -- remote HTTP transport

ChatGPT connectors (and Claude *remote* connectors) need a **remote** MCP server
over HTTP, not local stdio. Axiom exposes one -- start the server with the
transport enabled:

```bash
AXIOM_MCP_HTTP=1 \
AXIOM_MCP_TOKEN=$(openssl rand -hex 24) \
axiom --mode server --host 0.0.0.0 --port 8080
```

This serves the **same MCP tools** as the stdio path at `/mcp`, sharing one
dispatch against the live pipeline:

- `POST /mcp` -- a JSON-RPC 2.0 request (`initialize`, `tools/list`, `tools/call`)
  returns the JSON-RPC response.
- `GET /mcp` -- an SSE stream (keep-alive) for Streamable-HTTP clients that open a
  stream for server-initiated messages.

> **Authenticate before exposing it.** `/mcp` lets a caller drive Axiom's tools
> (read/compress paths, memory, etc.), so set **`AXIOM_MCP_TOKEN`** to require an
> `Authorization: Bearer <token>` header on every `/mcp` request. If it is unset,
> `/mcp` is open and the server prints a warning -- only acceptable on a trusted
> local network. The HTTP MCP context uses the auto-bootstrapped model by default;
> point it at a trained checkpoint with `--checkpoint <path>` just like the stdio
> server.

Then expose the endpoint to ChatGPT (publicly or via a tunnel like `cloudflared`
/ `ngrok`) and add it as a **custom connector** pointing at `https://<host>/mcp`,
configuring the bearer token in the connector's auth settings.

Axiom also exposes the two tools ChatGPT's **standard** connector mode requires --
`search` and `fetch` -- as first-class aliases over its semantic memory:

- `search(query)` -> ranked memory hits as `{"results":[{"id","title","url"}]}`.
- `fetch(id)` -> the full record for a `search` result id as
  `{"id","title","text","url","metadata"}`.

Both tools advertise an `outputSchema` and return their payload in the MCP
result's `structuredContent` (with a JSON text mirror in `content`), so standard
ChatGPT / company-knowledge connectors can validate and read the structured
result, not just the text.

So Axiom works as a standard ChatGPT connector (not only Developer Mode). Axiom's
richer tools (`axiom_compress_path`, `axiom_evaluate_drift`, `axiom_expand`,
`axiom_remember`, `axiom_recall`, `axiom_forget`, `axiom_verify`,
`axiom_validate_epistemic`, `axiom_immunity`, `axiom_status`, plus the
inter-agent task board tools) remain available under **Developer Mode** (arbitrary
MCP tools).

> **Caveat (verify against current ChatGPT docs):** connector behavior must be
> tested against your live ChatGPT, so treat first-time setup as needing a quick
> smoke test -- the `search`/`fetch` contract ChatGPT expects can change.

## What this is *not*

This does not let axiom call ChatGPT/Claude's models for free -- MCP is the host
calling axiom's tools, not the reverse. For axiom-initiated generation, use a
[generation backend](BACKENDS.md) (local via OpenDrop, or a cloud API key).