# Hybrid CLI integration — vendor CLIs through the Axiom proxy

Your coding CLI keeps **its own login**; only its **base URL** changes. Axiom
runs on loopback, compresses what it safely can, and relays the CLI's own
credentials upstream verbatim — it never holds a vendor key and never needs an
Axiom key on loopback. This is the recommended integration for cost savings.
The experimental [`cli` backend](#the-cli-backend-experimental) at the end of
this page is the opposite direction (Axiom's *own* model calls go out through a
logged-in CLI) and is a no-keys fallback, not a savings tool.

**Before you wire a subscription login through anything, read
[Vendor terms](#vendor-terms-checked-2026-11-06).** Where the vendor's terms
restrict or do not clearly permit it, the compliant route is API-key billing.

## 1. Start the proxy

```bash
cargo build --release --locked --manifest-path axiom_engine_rs/Cargo.toml   # once
./scripts/start_axiom_for_cli.sh            # loopback 127.0.0.1:8080, waits for /readyz
./scripts/start_axiom_for_cli.sh --stop     # later
```

The script is idempotent (a healthy proxy already on the address is reused),
pins the server's own upstreams to the real vendor APIs (never itself), strips
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `AXIOM_API_KEY` from the server's
environment so the proxy runs in pure **auth pass-through**, and prints the
exact `export` lines below for your *client* shell. Knobs: `AXIOM_HOST`,
`AXIOM_PORT` (default `8080`), `AXIOM_UPSTREAM_URL`, `AXIOM_OPENAI_UPSTREAM_URL`,
`AXIOM_LOG_FILE`, `AXIOM_READY_TIMEOUT_S`, `AXIOM_CLI_PROXY_STATE_DIR`, plus the
usual compression flags (`AXIOM_TTT_COMPRESS*`, README § CVM Cost Stack).

Verify, from any shell (no credentials needed):

```bash
curl -fsS http://127.0.0.1:8080/healthz
curl -s http://127.0.0.1:8080/metrics | grep -E '^axiom_(savings|cost|quota)_'
```

`axiom_savings_*` and `axiom_cost_*` move while the CLI runs; savings are the
delta between two reads (`bench/cvm/CLI-PROXY-RESULTS-2026-11-06.md` shows a
full before/after).

## 2. Point the CLI at it

What the proxy relays upstream, untouched: `Authorization`, `x-api-key`,
`anthropic-version`, `anthropic-beta` (Anthropic shape); `Authorization`,
`OpenAI-Organization`, `OpenAI-Project` (OpenAI shapes; `/v1/responses` also
relays the Codex transport headers such as `chatgpt-account-id`). Proven by
`axiom_engine_rs/tests/cli_passthrough_proxy.rs` against a capturing upstream.

| CLI | Setting (confirmed in the vendor's docs on 2026-11-06) | API shape → Axiom route | Notes |
|---|---|---|---|
| **Claude Code** | `export ANTHROPIC_BASE_URL=http://127.0.0.1:8080` — "Override the API endpoint to route requests through a proxy or gateway" ([env vars](https://code.claude.com/docs/en/env-vars)) | Anthropic Messages → `POST /v1/messages` | Keeps whatever login is active. Anthropic's [gateway page](https://code.claude.com/docs/en/llm-gateway): "Setting only that variable, without a gateway credential, doesn't replace the subscription … a saved claude.ai login remains the active credential"; a gateway credential (`ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY`) replaces it and is billed per token. See [Vendor terms](#vendor-terms-checked-2026-11-06). |
| **OpenAI Codex CLI** | `~/.codex/config.toml`: `openai_base_url = "http://127.0.0.1:8080/v1"` — "Base URL override for the built-in openai model provider" ([config reference](https://developers.openai.com/codex/config-reference)); or a custom `[model_providers.<id>]` with `base_url`, `env_key` (API key) or `requires_openai_auth = true` | OpenAI Responses → `POST /v1/responses` (`wire_api`: "responses is the only supported value") | Project-local `.codex/config.toml` ignores `openai_base_url`; set it in the user-level file. The `OPENAI_BASE_URL` env var is **not** in the current reference (older releases read it); the launcher still exports it as a harmless extra. ChatGPT-plan sign-in sends `chatgpt-account-id`, which Axiom routes to the ChatGPT backend (`openai_forwarder.rs`); see terms below. |
| **Aider** | `export OPENAI_API_BASE=http://127.0.0.1:8080/v1` and `aider --model openai/<name>` ([OpenAI-compatible APIs](https://aider.chat/docs/llms/openai-compat.html)) | Chat Completions → `POST /v1/chat/completions` | Aider documents **no** base-URL override for its Anthropic path, so only OpenAI-compatible models route through Axiom. |
| **Cline** (VS Code) | Anthropic provider: "check *Use custom base URL* and enter the URL" → `http://127.0.0.1:8080` ([docs](https://docs.cline.bot/provider-config/anthropic)); OpenAI Compatible provider: *Base URL* → `http://127.0.0.1:8080/v1` ([docs](https://docs.cline.bot/provider-config/openai-compatible)) | `/v1/messages` or `/v1/chat/completions` | API-key providers. Cline's separate "Claude Code (Subscription)" provider drives the CLI itself and is not affected by a base URL. |
| **Continue** | `config.yaml` model entry: `apiBase: http://127.0.0.1:8080/v1` — "Can be used to override the default API base that is specified per model" ([reference](https://docs.continue.dev/reference)) | depends on `provider` (openai → `/v1/chat/completions`) | The reference shows `apiBase` on an `openai` provider; it does not list which other providers honour it. |
| **OpenCode** | `opencode.json`: `"provider": {"anthropic": {"options": {"baseURL": "http://127.0.0.1:8080/v1"}}}` — "You can customize the base URL for any provider by setting the baseURL option" ([providers](https://opencode.ai/docs/providers/)) | per provider | API keys only: "There are plugins that allow you to use your Claude Pro/Max models with OpenCode. Anthropic explicitly prohibits this." (removed in 1.3.0). |
| **Gemini CLI** | `GOOGLE_GEMINI_BASE_URL` / `GOOGLE_VERTEX_BASE_URL` ([configuration](https://geminicli.com/docs/reference/configuration/)) | Gemini API (`generateContent`) — **no Axiom route** | Not supported: Axiom speaks Anthropic/OpenAI shapes only. Also see terms below. |

Cursor and other IDEs with an "OpenAI base URL" field work the same way as the
OpenAI-compatible rows when they send plain Chat Completions; they were not
verified for this page.

### What a session looks like

`bench/cvm/CLI-PROXY-RESULTS-2026-11-06.md`: a real 4-turn Claude Code session
through the proxy with OAuth login — 11 upstream calls, all 200, only the CLI's
own `authorization` / `anthropic-beta` / `anthropic-version` reached Anthropic,
every cached prefix forwarded byte-identical. **Axiom removed 0 bytes** in that
run because the Axiom MCP server (and so the `axiom_expand` recovery tool) was
disabled in the CLI, which holds the lossy levers off by design
(`docs/safety/lossy-invariant.md`). For savings, run the CLI *with* the Axiom
MCP server configured (`docs/MCP-CLIENTS.md`) so digest/rebase may fire.

## The `cli` backend (experimental)

`AXIOM_BACKEND=cli` makes Axiom's **own** model-needing paths — `axiom solve`
LLM repair, the `axiom task` agent loop, and server generation behind the
backend router — spawn a logged-in vendor CLI as a subprocess instead of
calling an API with a key:

```text
claude -p --output-format json --tools "" --no-session-persistence \
       --strict-mcp-config --mcp-config '{"mcpServers":{}}' [--model …]  < prompt
```

Compiled only with `--features experimental` (`docs/EXPERIMENTAL.md`). Env:
`AXIOM_CLI_COMMAND` (default `claude`), `AXIOM_CLI_TIMEOUT_SECS` (default
120), `AXIOM_CLI_MODEL` (optional `--model`). Guarantees, each unit-tested
against a fake CLI script (`axiom_engine_rs/src/cli_backend.rs`):

- **Version pin.** `<command> --version` must parse as `MAJOR.MINOR.PATCH (Claude
  Code)` with major `2`; anything else refuses to start (the server exits, the
  repair paths skip LLM repair with a logged reason).
- **Fail closed.** Only `{"type":"result","subtype":"success","is_error":false,
  "result":"…"}` is accepted; any other output is an error, never a guess.
- **Bounded.** Every call is killed past the timeout.
- **No shell, no leaks.** Direct argv spawn; the prompt travels on stdin; tools,
  session persistence and MCP servers are disabled so the CLI answers as a plain
  completion and never touches your files; the prompt and output are never
  logged.

It is a fallback for machines with no key, not a cost lever: nothing is
compressed on this path, and the CLI bills whatever account it is logged into.
Read the vendor terms below before using it with a subscription login.

## Vendor terms (checked 2026-11-06)

Quotes are short excerpts from the pages linked; they are what the pages said
on the check date, not legal advice. Where a page could not be fetched by
automation it is marked as such. **Nothing here claims that routing a
subscription login through Axiom is permitted.**

### Anthropic — Claude Code, Claude Free/Pro/Max subscriptions

- [Legal and compliance → Usage policy → Authentication and credential use](https://code.claude.com/docs/en/legal-and-compliance):
  "OAuth authentication is intended exclusively for purchasers of Claude Free,
  Pro, Max, Ultra, Team, and Enterprise subscription plans and is designed to
  support ordinary use of Claude Code and other native Anthropic applications."
  "Developers building products or services that interact with Claude's
  capabilities, including those using the Agent SDK, should use API key
  authentication." "Anthropic does not permit third-party developers to offer
  Claude.ai login into their own applications, or to route requests through
  Free, Pro, Max, or Ultra plan credentials on behalf of their users."
  "developers may not collect, store, or intermediate Claude.ai credentials or
  session tokens." It also says this does not "prevent an end user from signing
  in to the unmodified Claude Code binary with their own Claude subscription",
  and that Anthropic "may … enforce these restrictions … without prior notice."
- [Other LLM gateways](https://code.claude.com/docs/en/llm-gateway): describes
  `ANTHROPIC_BASE_URL` as the gateway variable, says a saved claude.ai login
  stays the active credential when no gateway credential is set, that
  "Gateways that pass this traffic on to Anthropic must forward the OAuth
  capability in `anthropic-beta`" (Axiom does), and that "Anthropic doesn't
  endorse, maintain, or audit third-party gateway products."
- [Consumer Terms of Service](https://www.anthropic.com/legal/consumer-terms)
  (effective 2025-10-08), § 3: prohibits, "Except when you are accessing our
  Services via an Anthropic API Key or where we otherwise explicitly permit it,
  to access the Services through automated or non-human means, whether through
  a bot, script, or otherwise", and "bypassing any of our systems or protective
  measures."
- The [headless mode page](https://code.claude.com/docs/en/headless) documents
  `claude -p` and `--output-format json` but does not say which usage pool a
  subscription-authenticated `-p` run draws from. (This repository's
  `scripts/pss_eval.sh` recorded in July 2026 that `-p` spend drew from a
  separate Agent-SDK credit pool; that was not re-verified today.)

**Status: restricted / unclear for subscription logins.** A local Axiom proxy
relaying your own OAuth token sits between "an end user … with their own
Claude subscription" (allowed) and "intermediate Claude.ai … session tokens"
(not permitted for third-party developers); the `cli` backend is a product
driving Claude Code programmatically, which Anthropic's guidance says should
use API keys. Treat both as **not clearly permitted** with a subscription.
**Compliant route:** an API key from the Claude Console — either set
`ANTHROPIC_API_KEY` for the proxy (it injects it only when the client sends no
credential) or in the CLI's shell; the gateway page documents that this
replaces the subscription login and bills per token.

### OpenAI — Codex CLI, ChatGPT plans

- [Codex configuration reference](https://developers.openai.com/codex/config-reference):
  documents `openai_base_url`, `[model_providers.<id>].base_url` / `env_key` /
  `wire_api` ("responses is the only supported value"), `requires_openai_auth`
  ("The provider uses OpenAI authentication"), and `chatgpt_base_url`
  ("Override the base URL used during the ChatGPT login flow"). The
  [authentication page](https://developers.openai.com/codex/auth) distinguishes
  "Sign in with ChatGPT for subscription access" from "Sign in with an API key
  for usage-based access" (standard API pricing) and mentions custom providers
  with `requires_openai_auth = true` as "useful when you access OpenAI models
  through an LLM proxy server." Neither page addresses third-party clients or
  proxies holding ChatGPT credentials.
- The help article [Using Codex with your ChatGPT plan](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan)
  and OpenAI's [Terms of Use](https://openai.com/policies/terms-of-use) /
  [Usage Policies](https://openai.com/policies/usage-policies) returned HTTP 403
  to automated fetching on the check date, so they are **not quoted here**. A
  search excerpt of the help article states that signing in with a ChatGPT
  account places that use under the ChatGPT Terms of Use; the terms themselves
  were not read for this page.

**Status: unclear.** Codex's own configuration supports an "LLM proxy server"
in front of OpenAI with either sign-in mode, but no fetched page says whether a
proxy that relays ChatGPT-plan credentials is permitted. **Compliant route:**
an OpenAI API key (`env_key` on a `[model_providers]` entry, or
`OPENAI_API_KEY` in the proxy), billed at API rates.

### Google — Gemini CLI

- [Gemini CLI terms and privacy](https://geminicli.com/docs/resources/tos-privacy/)
  (last updated 2026-04-10): "Directly accessing the services powering Gemini
  CLI … using third-party software, tools, or services … is a violation of
  applicable terms and policies" and "may be grounds for suspension or
  termination of your account."
- The [individuals privacy notice](https://developers.google.com/gemini-code-assist/resources/privacy-notice-gemini-code-assist-individuals)
  says that starting 2026-06-18 the "Login with Google" path no longer serves
  Gemini CLI for the individual / Google AI Pro / Ultra tiers.

**Status: restricted, and not technically supported by Axiom** (different wire
protocol). Use a Gemini API key directly with the CLI; Axiom is not in that path.

### Everything else (Aider, Cline, Continue, OpenCode, Cursor)

These tools authenticate with API keys you supply, and their base-URL settings
exist precisely for proxies and gateways; the proxy relays your key unchanged
and the vendor bills it at API rates. OpenCode's docs are explicit that using a
Claude Pro/Max login through it is prohibited by Anthropic (see table).
