# Epistemic Drift Validation

AXIOM-AETHER can combine its deterministic grounding verifier with an
OpenAI-compatible LLM judge to detect fluent semantic drift that lexical
overlap cannot catch. The feature is local to AXIOM-AETHER. It does not use or
integrate with Axiom.co.

## Configuration

The judge is disabled until both its URL and model are configured:

```powershell
$env:AXIOM_EPISTEMIC_JUDGE_URL = 'http://127.0.0.1:11434'
$env:AXIOM_EPISTEMIC_JUDGE_MODEL = 'your-judge-model'
```

The URL may be an API root, a `/v1` root, or the full
`/v1/chat/completions` endpoint. Local OpenAI-compatible servers do not need a
key. For an authenticated endpoint, provide the secret only through the
process environment:

```powershell
$env:AXIOM_EPISTEMIC_JUDGE_API_KEY = '<secret>'
```

Optional settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AXIOM_EPISTEMIC_JUDGE_TIMEOUT_SECS` | `60` | Judge request timeout, clamped to 1-300 seconds |
| `AXIOM_EPISTEMIC_TELEMETRY_PATH` | `$AXIOM_MEMORY_DIR/epistemic_telemetry.jsonl` | Local append-only telemetry path |
| `AXIOM_EPISTEMIC_CAPTURE_TEXT` | unset | Set to `1` to store raw prompt/response text |
| `AXIOM_EPISTEMIC_AUTO` | unset | Set to `1` to monitor successful proxied responses asynchronously |

Telemetry stores SHA-256 hashes instead of raw prompt and response text by
default. Raw-text storage is an explicit privacy decision.

## HTTP API

`POST /v1/epistemic/validate` accepts:

```json
{
  "prompt": "Explain the mechanism.",
  "response": "The generated response.",
  "evidence": "Optional source evidence.",
  "target_model": "model identifier",
  "request_id": "optional trace identifier"
}
```

The result contains the grounding counts, semantic judge result, combined
`allow`, `review`, or `block` decision, judge latency, prompt version, and
whether telemetry was written. The endpoint returns `503` when the judge is not
configured and `502` when a configured judge fails or returns invalid output.

## MCP

`axiom_validate_epistemic` exposes the same operation to MCP clients. It
requires `prompt` and `response`; `evidence`, `target_model`, and `request_id`
are optional. Both `review` and `block` decisions set the MCP result's `isError`
flag so the calling agent cannot silently overlook drift or uncertain grounding.

## Automatic Monitoring

With `AXIOM_EPISTEMIC_AUTO=1`, successful Anthropic responses that pass through
Axiom's compression proxy are evaluated in a background task. Judge or telemetry
failures are written to server diagnostics and never fail or mutate the primary
inference response. Automatic monitoring of OpenAI-compatible upstream paths is
not yet implemented.

The judge prompt is versioned as `epistemic-drift-v1`. It treats abstract or
philosophical language as drift only when that language departs from the mode
requested by the original prompt; explicitly requested philosophical work is
not classified as drift merely because it is abstract.
