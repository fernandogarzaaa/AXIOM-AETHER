# Running axiom as an MCP server (use it from Claude / ChatGPT)

Axiom ships an MCP server that exposes its cognition tools — `axiom_compress_path`,
`axiom_evaluate_drift`, `axiom_expand`, `axiom_remember`, `axiom_recall`,
`axiom_forget`, `axiom_verify` — to an LLM host. In this mode **the host model is
the brain and axiom is its toolkit**, which is the sanctioned, zero-API-cost way
to combine your ChatGPT/Claude *subscription* with axiom (you drive your own app;
it calls axiom's tools).

## Claude Desktop / Claude Code — works today (stdio)

Axiom's MCP server speaks JSON-RPC over **stdio**, which is exactly what Claude
Desktop and Claude Code use for local MCP servers.

**Claude Desktop** — add to `claude_desktop_config.json`:

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

**Claude Code** — register the same server:

```bash
claude mcp add axiom -- axiom --mode mcp
```

Then ask Claude to "compress this repo with axiom" or "check this code for drift"
and it will call the tools.

## ChatGPT connector / Claude remote — remote HTTP transport

ChatGPT connectors (and Claude *remote* connectors) need a **remote** MCP server
over HTTP, not local stdio. Axiom exposes one — start the server with the
transport enabled:

```bash
AXIOM_MCP_HTTP=1 axiom --mode server --host 0.0.0.0 --port 8080
```

This serves the **same MCP tools** as the stdio path at `/mcp`, sharing one
dispatch against the live pipeline:

- `POST /mcp` — a JSON-RPC 2.0 request (`initialize`, `tools/list`, `tools/call`)
  returns the JSON-RPC response.
- `GET /mcp` — an SSE stream (keep-alive) for Streamable-HTTP clients that open a
  stream for server-initiated messages.

Then expose the endpoint to ChatGPT (publicly or via a tunnel like `cloudflared`
/ `ngrok`) and add it as a **custom connector** pointing at `https://<host>/mcp`.

> **Caveats (verify against current ChatGPT docs):** ChatGPT's standard connector
> mode expects `search`/`fetch` tools — Axiom exposes its own tool names, which
> work under ChatGPT **Developer Mode** (arbitrary MCP tools). Adding
> `search`/`fetch` aliases (mapping to compress/expand/recall) for standard-mode
> compatibility is a tracked follow-up. Because connector behavior must be tested
> against your live ChatGPT, treat first-time setup as needing a quick smoke test.

## What this is *not*

This does not let axiom call ChatGPT/Claude's models for free — MCP is the host
calling axiom's tools, not the reverse. For axiom-initiated generation, use a
[generation backend](BACKENDS.md) (local via OpenDrop, or a cloud API key).
