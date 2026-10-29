# AXIOM-AETHER — Environment variable reference

Every `AXIOM_*` variable the engine reads, grouped by surface. The README's
*Useful environment variables* table is the curated subset; this page is the
complete one, generated from `grep -rhoE 'env::var(_os)?\("AXIOM_[A-Z0-9_]+"' axiom_engine_rs/src`
on 2026-10-29. Defaults are what the code falls back to when the variable is
unset. When you add a variable, add it here.

## Server, auth and limits

| Variable | Default | Effect |
|---|---|---|
| `AXIOM_HOST` / `AXIOM_PORT` | `127.0.0.1` / `3000` | Bind address for `--mode server`. A non-loopback bind requires `AXIOM_API_KEY` or `AXIOM_ALLOW_OPEN=1`. |
| `AXIOM_ALLOW_OPEN` | unset | Allow a non-loopback bind without `AXIOM_API_KEY` (loud startup warning). |
| `AXIOM_API_KEY` | unset (open) | Data-plane key; every route but `/healthz`, `/readyz`, `/metrics` needs `X-Axiom-Key`. |
| `AXIOM_MCP_HTTP` | unset (`/mcp` returns 503) | `1` enables the HTTP MCP transport. |
| `AXIOM_MCP_TOKEN` | unset | Bearer token for `/mcp`. Unset + `AXIOM_API_KEY` set ⇒ `/mcp` requires the API key instead. |
| `AXIOM_ENABLE_JIT_EXEC` | unset (403) | Enables `POST /v1/hypervisor/jit_run` (process execution). |
| `AXIOM_MAX_GENERATION_TOKENS` | `16384` | Ceiling on request `max_tokens` for local generation. |
| `AXIOM_MAX_SESSIONS` | `1024` | Cap on `/v1/sessions` W̃ sessions; `POST` returns 409 past it. |
| `AXIOM_MAX_TTT_SESSIONS` | `512` | Cap on proxy TTT sessions (`X-Axiom-Session-Id`, `/v1/ttt/feedback`); least-recently-used eviction. |
| `AXIOM_MERGE_ROOT` | cwd + `~/.axiom` | `;`-separated directories `POST /v1/cluster/merge` may read inputs from and write its output to. |
| `AXIOM_VFS_ALLOWED_ROOTS` | unset (any) | `;`-separated directories `/v1/hypervisor/mount` may mount. |
| `AXIOM_VFS_MAX_READ_BYTES` | `16777216` | Cap on one `/v1/hypervisor/read`; only regular files are read. |
| `AXIOM_COMPRESSION_CACHE` | `checkpoints/memory/axiom_compression_cache.bin` | Persisted W̃ cache written by `/v1/ttt/feedback` and read at boot. |

## Proxy / compression

| Variable | Default | Effect |
|---|---|---|
| `AXIOM_TTT_COMPRESS` | off | Enable proxy context compression on `/v1/messages` and `/v1/chat/completions`. |
| `AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS` / `AXIOM_TTT_COMPRESS_TOP_K` | see README | Compression threshold and recall depth. |
| `AXIOM_RESPONSES_COMPRESS` | on | `0` disables `/v1/responses` input compression. |
| `AXIOM_RESPONSES_RELAY_EXTRA` | empty | Extra lowercase client header names relayed to the Responses upstream (allowlist extension). |
| `AXIOM_OPENAI_RESPONSES_WS_UPSTREAM` | derived from headers | Explicit WebSocket upstream for `GET /v1/responses`. |
| `AXIOM_CACHE_SAFE`, `AXIOM_CVM_DIGEST`, `AXIOM_CVM_DIGEST_THRESHOLD_TOKENS`, `AXIOM_PREFIX_DEDUP`, `AXIOM_KEEPALIVE` | see README § CVM Cost Stack | Cost-stack levers. |
| `AXIOM_KEEPALIVE_HORIZON_S` | `1800` | Keep-alive ping horizon in seconds (only with `AXIOM_KEEPALIVE=1`; replays your own credentials, read `keepalive.rs` first). |
| `AXIOM_SESSION_RECORD` | off | `1` records scrubbed per-session request/response JSONL under `~/.axiom/sessions/`. |
| `AXIOM_PROXY_URL` | `http://127.0.0.1:3000` | Where the MCP server's `axiom_status` / `axiom_expand` reach the running proxy. |
| `AXIOM_BASE_URL` | `http://127.0.0.1:3000` | Proxy URL for `axiom fleet status` and `axiombench --live`. |

## Model and checkpoints

| Variable | Default | Effect |
|---|---|---|
| `AXIOM_DEVICE` | `auto` | `cpu`, `cuda`, `metal`, `auto`. |
| `AXIOM_PRODUCTION_BPE` | off | Use the trained BPE checkpoint path. |
| `AXIOM_TOKENIZER` / `AXIOM_BPE_CKPT` | `checkpoints/axiom_bpe.json` / `checkpoints/axiom_production_bpe.bin` | Tokenizer and checkpoint paths. |
| `AXIOM_CHECKPOINT_URL` / `AXIOM_TOKENIZER_URL` | unset | Fetch at boot; pair with `AXIOM_CHECKPOINT_SHA256` / `AXIOM_TOKENIZER_SHA256` to fail closed on a mismatch. |
| `AXIOM_EMB_CKPT` | `checkpoints/axiom_embedder.bin` | Embedding model used by MCP memory recall and `eval_recall`. |
| `AXIOM_FORGET_GATE` | `1.0` | Parameter-free forget gate α on the delta rule (α < 1 decays fast-weight memory). |
| `AXIOM_VIBE` / `AXIOM_VIBE_PRIME` | see README | Persistent fast-weight memory controls. |
| `AXIOM_VIBE_PATH` | `axiom_master_vibe.bin` | Master vibe file. |
| `AXIOM_VIBE_DECAY` | `0.9` | EMA decay when sessions are merged into the master vibe. |
| `AXIOM_DRIFT_THRESHOLD` | per build | Drift score threshold for `axiom_evaluate_drift`. |

## MCP server

| Variable | Default | Effect |
|---|---|---|
| `AXIOM_MCP_MAX_FILES` | `512` | Files `axiom_compress_path` will absorb per call. |
| `AXIOM_MCP_MAX_BYTES` | `1048576` | Bytes `axiom_compress_path` will absorb per call (cut at a char boundary). |
| `AXIOM_TASK_DIR` | `checkpoints/tasks/` | Task-board storage; set an absolute path because MCP hosts may start the server from a read-only cwd. |

## Backends and routing

| Variable | Default | Effect |
|---|---|---|
| `AXIOM_BACKEND` | local | `anthropic`, `openai`, `opendrop`, `router`, … (see `docs/BACKENDS.md`). |
| `AXIOM_CLAUDE_MODEL` / `AXIOM_CLAUDE_SYSTEM` | `DEFAULT_CLAUDE_MODEL` in `claude_backend.rs` / unset | Model id and default system prompt for the Claude backend. |
| `AXIOM_ROUTER_CONSENSUS` | off | Router asks two providers and fuses answers via `BetaBelief`. |
| `AXIOM_OLLAMA_URL` / `AXIOM_OLLAMA_MODELS` | `DEFAULT_OLLAMA_URL` / built-in candidate list (`swarm_router.rs`) | Local Ollama endpoint and model candidates. |
| `AXIOM_OLLAMA_NUM_CTX` / `AXIOM_OLLAMA_TIMEOUT_MS` | `DEFAULT_NUM_CTX` / `2000` | Context window and request timeout for Ollama calls. |

## Self-healing, solve and sandbox

| Variable | Default | Effect |
|---|---|---|
| `AXIOM_HEAL_MEMORY` | `~/.axiom/...` | Learned immunity JSON path, or `0`/`off`. |
| `AXIOM_PEER_RELIABILITY` | `0.8` | Weight given to a peer's immunity claims (clamped to [0, 1]). |
| `AXIOM_SOLVE_LLM_ATTEMPTS` | `3` | LLM repair attempts per `axiom solve` round. |
| `AXIOM_SOLVE_VERIFY_TIMEOUT_SECS` | `120` | Verifier timeout; a hung verifier is killed and the candidate rolled back. |
| `AXIOM_SANDBOX_ROOT` / `AXIOM_SANDBOX_MAX_STEPS` / `AXIOM_SANDBOX_VERIFY` / `AXIOM_SANDBOX_TIMEOUT_SECS` | see `sandbox.rs` | Staging-directory controls for candidate patches (`cargo check` gate; **not** an OS sandbox, see `docs/SECURITY-AUDIT.md` §3). |
| `AXIOM_EPISTEMIC_*` | see `docs/EPISTEMIC_DRIFT.md` | Epistemic validation thresholds. |

## Fleet / swarm (experimental)

| Variable | Default | Effect |
|---|---|---|
| `AXIOM_FLEET_KEY` / `AXIOM_FLEET_KEY_PREV` | unset | HMAC key (and rotation predecessor) for immunity exchange and DWE fragments. |
| `AXIOM_DWE_LISTEN` / `AXIOM_DWE_PEERS` | unset | Enable DWE and list comma-separated peers. |

## Grounding

| Variable | Default | Effect |
|---|---|---|
| `AXIOM_VERIFY_RESPONSES` | off | Response grounding advisories on the proxy. |
| `AXIOM_CONFORMAL_THRESHOLD` / `AXIOM_CONFORMAL_DELTA` | `0.75` / `0.10` | Conformal factuality gate τ and coverage tolerance δ. |

## Training and tooling binaries (`--features tools`)

`AXIOM_BPE`, `AXIOM_BPE_OUT`, `AXIOM_BPE_VOCAB`, `AXIOM_CORPUS_OUT`,
`AXIOM_PAIR_DIRS`, `AXIOM_PAIRS_OUT`, `AXIOM_SYNTH`, `AXIOM_META_TRAIN_LR`,
`AXIOM_META_TRAIN_MAX_FILES`, `AXIOM_META_TRAIN_MAX_SEQS`,
`AXIOM_META_TRAIN_SEED`, `AXIOM_TRAIN_LAST_TOKEN_ONLY`, `AXIOM_VRAM_BUDGET_MB`,
`AXIOM_HARVEST_CHECKPOINT`, `AXIOM_EMB_DEVICE`, `AXIOM_SEARCH_TOPK`,
`AXIOM_SEARCH_MAX_PAGES` — read only by the trainers, crawlers and evals under
`axiom_engine_rs/src/bin/`; each binary's `--help` or module doc comment
describes its own knobs.
