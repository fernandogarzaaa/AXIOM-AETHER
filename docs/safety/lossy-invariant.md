# Lossy-context invariant: LOSSY_CONTEXT implies RECOVERY_CAPABILITY_PRESENT

Every default-on path that drops, replaces, or summarizes content must have a
recovery path for the clients that trigger it, or be gated off for clients
without one. Verified against code at commit 35dcb50.

## Paths

| # | Path | Location | Default-on | What changes | Recovery mechanism | Status |
|---|------|----------|------------|--------------|-------------------|--------|
| 1 | Digest (tool result to stub) | `digest.rs`, `routes_messages.rs:337` | Yes (`AXIOM_CVM_DIGEST`) | Heavy `tool_result` replaced with `[AXIOM-PAGE page_id session=...]` stub | Gate: `expand_tool_available()` skips digesting when `axiom_expand` not in tools. Stub carries page_id + session_id, expandable via MCP `axiom_expand` or `POST /v1/expand`. | Gated (PR #207, restored PR #209) |
| 2 | Responses input compression | `responses_compressor.rs`, `routes_responses.rs` | Yes (opt-out `AXIOM_RESPONSES_COMPRESS`) | Old assistant text replaced with TTT fingerprint + source manifest | Fail-closed retry: on network failure or 400/5xx from compressed payload, proxy retries with the original payload held in memory. Manifest carries SHA-256 per item for verification. | Fail-closed retry |
| 3 | Rebase stubs (on cache break) | `rebase.rs`, `routes_messages.rs:337` | Yes (`AXIOM_REBASE_ON_BREAK`) | Old heavy `tool_result`s replaced with content-hashed stubs in L2 store | Same gate as digest: `expand_tool_available()` on the request body. Stubs expandable via same `axiom_expand` path. | Gated (PR #209) |
| 4 | Tool deferral | `tool_defer.rs`, `routes_messages.rs:655` | Yes (`AXIOM_TOOL_DEFER`) | Tool schemas hidden from cached prefix via `defer_loading: true` | Native recovery: Anthropic loads deferred schemas on demand as `tool_reference` blocks without breaking cache. Fail-closed: zero working-set overlap skips deferral entirely. Names, order, count unchanged. | Native on-demand recovery |
| 5 | Local-trivial answers | `local_trivial.rs`, `routes_messages.rs:252` | Yes (`AXIOM_LOCAL_TRIVIAL`) | Model turn replaced with fixed ACK text, no upstream call | Labeled: ACK text states "answered locally by Axiom -- no upstream call was made". Fail-closed admission: only small clean `tool_result`s with low surprisal and no error signature. Client sees the label. Counted in session receipt. | Labeled, fail-closed |
| 6 | Model routing (R1 downgrade) | `model_router.rs`, `routes_messages.rs:701` | Yes (`AXIOM_MODEL_ROUTE=auto`) | Model changed (e.g. Opus to Haiku); content untouched | Not content-lossy. Retry-once fallback to original tier on 4xx. Declared capability outranks downgrade. Downgrade is a cost decision, never a compression one. Counted in session receipt. | Fallback + receipt |

## Notes

- Path 1 and 3 share the `CvmStore` L2 and the `axiom_expand` recovery tool.
  The gate is checked against the request's own `tools` array, so a client
  that cannot call `axiom_expand` never receives a stub.
- Path 2's recovery is in-flight only: the original is retried on failure but
  not persisted. If the upstream accepts the fingerprint and returns a
  degraded answer, there is no post-hoc recovery. This is the documented
  tradeoff of the compression feature.
- Path 5's ACK is synthetic by design. The fail-closed admission is the
  safety mechanism: anything ambiguous is forwarded upstream.
- Path 6 changes which model answers, not what content it sees. The invariant
  applies via the receipt requirement: downgrades must be visible.

## Verification

- Proxy-level integration tests: `axiom_engine_rs/tests/lossy_invariant.rs`
  (one test per path, in a file feature work does not touch).
- CI guard: `.github/workflows/` check that fails if any gate test is deleted.
- Session receipt: `GET /v1/awareness/{id}` includes `local_answered_turns`
  and `routed_turns`; session-end receipt prints both.
