# Set up Axiom with your AI agent (Codex / Claude)

This guide gets Axiom running as an MCP server and connected to your assistant. It
has two halves:

1. **[Setup reference](#setup-reference)** — the exact commands, so you (or your
   agent) know the target state.
2. **[Prompt your agent to do it](#prompt-your-agent)** — copy-paste prompts that
   make **Codex** (ChatGPT's coding agent) or a **Claude** agent configure Axiom
   for you, end to end, including the common failure modes.

There are two client paths, and they need different transports:

| Client | Transport | What you run |
|---|---|---|
| **Claude** (Desktop / Code) | **stdio** | `axiom --mode mcp` — registered locally, no network |
| **ChatGPT / Codex connector** | **remote HTTP** | `AXIOM_MCP_HTTP=1 axiom --mode server` + a public HTTPS URL |

In both modes **the host model is the brain and Axiom is its toolkit** — the
sanctioned, zero-API-cost way to use your ChatGPT/Claude subscription with Axiom.

---

## Setup reference

### Prerequisites — build a binary that has `/mcp`

The remote HTTP transport (`/mcp`) and the standard `search`/`fetch` connector
tools are recent. **An older release binary will return `404` for `/mcp`.** Build
(or install) current `main`:

```bash
git clone https://github.com/fernandogarzaaa/AXIOM-AETHER
cd AXIOM-AETHER/axiom_engine_rs
cargo build --release --locked
# binary: target/release/axiom_engine   (aka `axiom`)
```

> **Windows toolchain:** building can fail on missing linkers —
> `link.exe` (MSVC target) or `dlltool.exe` (GNU target). Fix by installing the
> **MSVC** toolchain and using it:
> ```powershell
> # Install "Desktop development with C++" via Visual Studio Build Tools, then:
> rustup default stable-x86_64-pc-windows-msvc
> cargo build --release --locked
> ```
> If you cannot build the HTTP server at all, see the
> [Node/Python bridge fallback](#fallback-stdio--http-bridge-for-chatgpt).

### Path A — Claude (stdio, simplest)

```bash
# Claude Code:
claude mcp add axiom -- axiom --mode mcp
```
Claude Desktop — add to `claude_desktop_config.json`:
```json
{ "mcpServers": { "axiom": { "command": "axiom", "args": ["--mode", "mcp"] } } }
```
Then ask Claude to "compress this repo with Axiom" or "recall what I decided
about X". (Add `"--checkpoint", "<path>"` to use a trained model.)

### Path B — ChatGPT / Codex (remote HTTP)

```bash
# 1. Pick a token once and keep it secret:
export AXIOM_MCP_TOKEN=$(openssl rand -hex 24)

# 2. Start the server with the HTTP transport on:
AXIOM_MCP_HTTP=1 axiom --mode server --host 0.0.0.0 --port 8080
#   add --checkpoint <path> for a trained model

# 3. Expose it over HTTPS (ChatGPT needs a public URL):
cloudflared tunnel --url http://localhost:8080      # or: ngrok http 8080
```

In ChatGPT → **Settings → Connectors → Add custom connector**:
- **URL:** `https://<your-tunnel-host>/mcp`
- **Auth:** Bearer token → your `AXIOM_MCP_TOKEN`

`GET/POST /mcp` share one dispatch; the connector sees the same tools as stdio,
plus the standard `search`/`fetch` aliases for ChatGPT's standard connector mode.

### Fallback: stdio → HTTP bridge (for ChatGPT)

If you **can't build the HTTP server** (e.g. an unresolved Windows toolchain), you
can still reach ChatGPT by front-ending the **stdio** server with a tiny bridge
that forwards `POST /mcp` to `axiom --mode mcp`'s stdin/stdout, then tunnel the
bridge. This works, but note the bridge fronts whatever binary you already have —
**an old binary won't expose `search`/`fetch`** (you'll see only the `axiom_*`
tools). Prefer the native server once you can build it. (Ask your agent to write
the bridge — see the Codex prompt below.)

### Verify (always do this first)

A `tools/list` round-trip is the smoke test. Through the tunnel it should return
the Axiom tools; from a current build that's **10**:
`search`, `fetch`, `axiom_compress_path`, `axiom_evaluate_drift`, `axiom_expand`,
`axiom_remember`, `axiom_recall`, `axiom_forget`, `axiom_verify`,
`axiom_immunity`.

---

## Prompt your agent

You don't have to do the above by hand — hand it to your coding agent. Paste the
matching prompt; it will detect your OS, build/run Axiom, wire the connector, and
verify.

### Prompt for **Codex** (ChatGPT's agent) — configure the ChatGPT connector

> **Goal: configure this machine so Axiom is usable from ChatGPT as a custom MCP
> connector, and verify it end to end.**
>
> Context: Axiom is a Rust project (`axiom_engine_rs`, binary `axiom` /
> `axiom_engine`). It exposes an MCP server. ChatGPT connectors need a **remote
> HTTP** endpoint at `/mcp`; Claude uses stdio. Repo:
> https://github.com/fernandogarzaaa/AXIOM-AETHER
>
> Do this, reporting each step and stopping to ask me only if a step needs a
> secret or a decision I must make:
> 1. **Use current `main`.** Older binaries return **404 on `/mcp`** — confirm the
>    binary supports `/mcp` (build `cargo build --release --locked` if unsure). On
>    Windows, if the build fails on `link.exe`/`dlltool.exe`, install the MSVC C++
>    build tools and `rustup default stable-x86_64-pc-windows-msvc`, then rebuild.
> 2. **Generate a secret token** and set `AXIOM_MCP_TOKEN` to it; never print the
>    full token back to me in plaintext beyond what I need to paste into ChatGPT.
> 3. **Start the server:** `AXIOM_MCP_HTTP=1 axiom --mode server --host 0.0.0.0
>    --port 8080` (use `--checkpoint` if a trained checkpoint exists).
> 4. **Expose it** over HTTPS with a tunnel (`cloudflared tunnel --url
>    http://localhost:8080` or `ngrok`). Capture the public URL.
> 5. **Smoke test** a `tools/list` JSON-RPC call against `https://<host>/mcp` with
>    the `Authorization: Bearer <token>` header, both locally and through the
>    tunnel. Show me the tool names returned; I expect to see `search` and `fetch`
>    plus the `axiom_*` tools.
> 6. **If — and only if — you cannot build the native HTTP server**, fall back to
>    writing a small HTTP→stdio bridge that forwards `POST /mcp` to `axiom --mode
>    mcp`, run that, and tunnel it instead. Flag clearly that a bridge over an old
>    binary won't expose `search`/`fetch`.
> 7. Print the final **connector settings** I should enter in ChatGPT (URL =
>    `https://<host>/mcp`, Auth = Bearer token), and remind me the trycloudflare
>    URL is public so I should treat the token as a password.
>
> Then save a one-line note of what you set up using the `axiom_remember` tool (if
> the connector is already live) so future sessions know the config.

### Prompt for a **Claude** agent — register the stdio server

> **Goal: register Axiom as a local MCP server for Claude, the simplest way, and
> verify it.**
>
> Context: Axiom is a Rust project (`axiom_engine_rs`, binary `axiom`). Claude uses
> MCP over **stdio** — no HTTP or tunnel needed. Repo:
> https://github.com/fernandogarzaaa/AXIOM-AETHER
>
> Do this and report each step:
> 1. Ensure an `axiom` binary exists (`cargo build --release --locked` in
>    `axiom_engine_rs` if needed; on Windows prefer the MSVC toolchain).
> 2. Register it for Claude Code: `claude mcp add axiom -- axiom --mode mcp`
>    (or, for Claude Desktop, add the `mcpServers.axiom` entry to
>    `claude_desktop_config.json` with `command: "axiom"`, `args:
>    ["--mode","mcp"]`). Use `--checkpoint <path>` if a trained checkpoint exists.
> 3. Verify the server starts and lists its tools (`tools/list`); show me the
>    names. I expect `search`, `fetch`, and the `axiom_*` tools.
> 4. Demonstrate one real call — e.g. `axiom_remember` a test note then
>    `axiom_recall` it — and show the result.
>
> Stop and ask me before changing any global config file.

### Tips that make agents succeed

- **Name the tool** in follow-up prompts ("use the Axiom `search` tool") so the
  model calls Axiom instead of answering from its own memory.
- **Seed memory first.** `search`/`recall` only return what you've saved — have
  the agent `axiom_remember` a few notes before expecting hits.
- **One concise memory per idea**, with a clear `scope` (`personal` or
  `project:<name>`).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `404` on `/mcp` | Binary predates the HTTP transport — build/run current `main`. |
| `401 Unauthorized` | Bearer token in the connector ≠ `AXIOM_MCP_TOKEN`; re-paste it. |
| `503` on `/mcp` | Server not started with `AXIOM_MCP_HTTP=1`, or tunnel URL is stale. |
| Build fails: `link.exe`/`dlltool.exe` | Windows linker missing — install MSVC C++ build tools, `rustup default stable-x86_64-pc-windows-msvc`, rebuild. |
| Only `axiom_*` tools, no `search`/`fetch` | You're on an old binary (often via a bridge) — rebuild from current `main`. |
| `search` returns nothing | Memory is empty; `axiom_remember` some notes first. |
| ChatGPT won't call Axiom | Be explicit: "use the Axiom `search` tool". |

> Connector behavior can change with ChatGPT updates — re-run the `tools/list`
> smoke test after any update before relying on it.

See also [`MCP-CLIENTS.md`](MCP-CLIENTS.md) (transport details) and
[`BACKENDS.md`](BACKENDS.md) (generation backends).
