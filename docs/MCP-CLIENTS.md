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

## ChatGPT connector — roadmap (remote HTTP/SSE)

ChatGPT connectors require a **remote** MCP server over HTTP (Streamable
HTTP / SSE), not local stdio. Axiom's HTTP transport is the next piece of work
(tracked in [`BACKENDS.md`](BACKENDS.md) → Roadmap):

- A `/mcp` endpoint on axiom's existing HTTP server that dispatches the same
  JSON-RPC tools against the live pipeline (POST → JSON response, GET → SSE).
- `search` / `fetch` tool aliases (mapping to axiom's compress/expand/recall) so
  axiom appears as a standard ChatGPT connector; arbitrary tools also work in
  ChatGPT **Developer Mode**.

Once landed, the setup will be: run `axiom --mode server`, expose `/mcp`
(directly or via a tunnel), and add it as a custom connector in ChatGPT. Until
then, the Claude stdio path above is the supported route.

## What this is *not*

This does not let axiom call ChatGPT/Claude's models for free — MCP is the host
calling axiom's tools, not the reverse. For axiom-initiated generation, use a
[generation backend](BACKENDS.md) (local via OpenDrop, or a cloud API key).
