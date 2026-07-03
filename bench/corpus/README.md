# AxiomBench Cost Corpus

This directory stores scrubbed JSONL `ExchangeRecord` sessions promoted by:

```powershell
$env:CARGO_INCREMENTAL='0'; $env:CARGO_TARGET_DIR='target-test'; cargo run --features tools --bin axiombench --locked -- corpus promote <session-id-or-path>
```

The checked-in seed corpus contains synthetic multi-turn `/v1/responses` sessions with long assistant-history spans. It is intentionally secret-free and exists so the live cost replay path is reproducible even when no private `~/.axiom/sessions` recordings are available.

For headline release numbers, promote real recorded sessions with `AXIOM_SESSION_RECORD=1` and keep only files that pass the scrub-miss rejection gate.

Live replay uses `AXIOMBENCH_AUTHORIZATION` when set, or `OPENAI_API_KEY` as a bearer token. Without one, the replay still exercises the proxy path but upstream responses may be counted as errored.
