# CLI pass-through run — 2026-11-06

One real Claude Code session driven through a local Axiom proxy with the CLI's
**own OAuth login** (no API key anywhere), measured as `/metrics` deltas on a
fresh proxy process, the same way `PSS-RESULTS-2026-07-16.md` and
`scripts/pss_eval.sh` measure.

> **Read this as a pass-through correctness run, not a savings measurement.**
> The proxy relayed the CLI's credentials correctly, preserved the client's
> prompt cache, and forwarded every request — but Axiom's own compression
> removed **0 bytes** in this session (see "Why no savings" below). The
> cost-vs-uncached percentage reported by the ledger is native Anthropic prompt
> caching, which Claude Code gets with or without the proxy.

## Setup

| Item | Value |
|---|---|
| Proxy | `scripts/start_axiom_for_cli.sh` on `127.0.0.1:8080`, default flags (`AXIOM_TTT_COMPRESS=1`, threshold 400, cache-safe on, CVM digest `skeleton`), **no** `ANTHROPIC_API_KEY`, **no** `AXIOM_API_KEY` |
| Client | `claude` 2.1.318 (Claude Code), logged in via OAuth (`claude auth status`: `authMethod: oauth_token`), `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, `--model haiku`, `--tools Read,Grep,Glob`, `--strict-mcp-config --mcp-config '{"mcpServers":{}}'`, `--output-format json`, turns 2–4 with `--continue` |
| Workload | 4 turns over three copied source files (`backend_router.rs` 532 lines, `cache_safety.rs` 150, `cli_backend.rs` 952): read+summarize, read+answer, read+list, cross-file question answered from context |
| Upstream | `https://api.anthropic.com` **via a transparent local HTTP→HTTPS relay** (see "Environment caveat") |

### Environment caveat (sandbox only)

This cloud sandbox intercepts outbound TLS with a private CA. `curl` and the
`claude` CLI trust it through the system store; the Axiom binary's HTTP client
is built with `rustls` + webpki roots and rejected the intercepted certificate
(`invalid peer certificate: UnknownIssuer`), so a first attempt with the proxy
pointed straight at `https://api.anthropic.com` failed every turn with a 502.
To measure anyway, a 60-line Python stdlib relay (`http.server` →
`http.client.HTTPSConnection` with the system CA) listened on `127.0.0.1:8081`
and forwarded each request byte-for-byte (method, path, all non-hop-by-hop
headers, body; response streamed back chunked) to `api.anthropic.com`. The proxy
was started with `AXIOM_UPSTREAM_URL=http://127.0.0.1:8081`. The relay changes
nothing in what Axiom sends or receives; it only supplies a trust store the
sandbox needs. On a normal machine the proxy talks to Anthropic directly and
the relay is not involved.

## Pass-through evidence

The relay logged header **names** only (never values or bodies):

```
POST /v1/messages -> 200 req_bytes=17697 resp_bytes=4221 auth_headers=['anthropic-beta', 'anthropic-version', 'authorization'] 0.7s
POST /v1/messages -> 200 req_bytes=41118 resp_bytes=8401 auth_headers=['anthropic-beta', 'anthropic-version', 'authorization'] 1.4s
... (11 upstream calls in total, all 200, all with exactly these three auth headers)
```

- 11 upstream calls, 11 × HTTP 200. Every call carried the CLI's own
  `authorization` (OAuth bearer), `anthropic-beta` (incl. the OAuth capability
  flag) and `anthropic-version`; **no** `x-api-key` and **no** `x-axiom-key`
  were ever sent upstream.
- The proxy log shows `cache_safe: frozen_blocks=N mutable_blocks=0
  compressed=0` on every conversation turn (N growing 4 → 14 as the session
  grew): everything at or before the client's `cache_control` breakpoint was
  forwarded untouched.
- All 4 CLI turns returned `subtype: success, is_error: false` with correct
  answers (e.g. turn 4 correctly ranked the three files by length from context).

## `/metrics` deltas (fresh process, before → after)

| Counter | Before | After | Note |
|---|---|---|---|
| `axiom_savings_bytes_in_total` | 0 | **0** | Axiom compression removed nothing |
| `axiom_savings_bytes_forwarded_total` | 0 | 0 | |
| `axiom_cache_write_tokens_total` | 0 | 34,998 | matches the CLI's own `cache_creation_input_tokens` sum exactly (14,863 + 2,897 + 16,912 + 326) |
| `axiom_cache_read_tokens_total` | 0 | 131,347 | CLI reports 106,411 for the 7 conversation calls; the ledger also counts Claude Code's 4 small auxiliary calls (one ~20 KB request per turn) |
| `axiom_uncached_input_tokens_total` | 0 | 2,567 | |
| `axiom_cost_usd_total` | 0 | 0.205032 | **ledger priced `claude-haiku-5-5` at the Sonnet-5 fallback rate** (`cost_ledger.rs` only recognises `haiku-4`/`haiku-3`); the CLI's own client-side estimate for the 4 turns is $0.024191 |
| `axiom_cost_uncached_usd_total` | 0 | 0.533421 | same pricing caveat |
| `axiom_quota_units_total` | 0 | 102,516.3 | |
| `axiom_quota_units_uncached_total` | 0 | 266,710.5 | |
| `axiom_local_answered_turns_total` / `axiom_routed_turns_total` | 0 | 0 / 0 | L-B and R1 did not fire (Haiku traffic, no trivial turns) |

`1 − axiom_cost_usd_total / axiom_cost_uncached_usd_total = 61.6 %` (same ratio
in quota units). **This is the native prompt-cache effect** the proxy left
intact, not an Axiom saving: `axiom_savings_bytes_in_total` is 0.

## Why no savings, and what is needed for a savings number

Both lossy levers were correctly held off by the lossy-context invariant
(`docs/safety/lossy-invariant.md`): the proxy log says
`digest skipped: axiom_expand not in tools[]` on every turn, because the run
disabled all MCP servers (`--strict-mcp-config`) so the Axiom MCP server — and
with it the `axiom_expand` recovery tool — was not in the client's toolset.
Without a recovery path, Axiom must not drop or digest any content, and it
didn't. The TTT compressor saw `heavy_msgs=0` as well: the heavy payloads were
`tool_result` blocks inside the frozen (cached) prefix, which cache-safety
forbids touching.

To produce a real savings number for this integration, run the same harness
with:

1. the Axiom MCP server **enabled** in the CLI (so `axiom_expand` is in
   `tools[]` and S3 digest / R2 rebase may fire on heavy `tool_result`s above
   `AXIOM_CVM_DIGEST_THRESHOLD_TOKENS` = 4000);
2. a longer session whose new-turn tool results exceed that threshold (whole
   files, large greps), in a repository the operator actually works in;
3. a machine where the proxy can validate upstream TLS directly (any normal
   developer machine; the relay above is a sandbox artefact);
4. ideally a Sonnet/Opus tier, so `axiom_cost_*` prices the traffic at a rate
   the ledger knows — or add Haiku 5.5 to `cost_ledger.rs` first.

Spend for this run: the CLI's own estimate was $0.024 across 4 Haiku turns,
billed to the operator's subscription usage, exactly as `claude -p` without
the proxy would have been.
