"""Hit Axiom's /v1/messages with the real Anthropic SDK.

Proves the integration works end-to-end: the Anthropic SDK is satisfied
with Axiom's response shape and successfully decodes the result.

Output text will be SHA-256 hash tokens (``tok_NNN``) because the local
Axiom model is untrained — that's expected. The test is about the wire
format, not the content.
"""

from __future__ import annotations

from anthropic import Anthropic

client = Anthropic(
    base_url="http://127.0.0.1:8080",
    api_key="not-needed-for-local-axiom",
)

response = client.messages.create(
    model="axiom-ttt-v1",
    max_tokens=8,
    messages=[{"role": "user", "content": "hello there from the anthropic sdk"}],
)

print("=== Anthropic SDK round-trip succeeded ===")
print(f"response.id        = {response.id}")
print(f"response.type      = {response.type}")
print(f"response.role      = {response.role}")
print(f"response.model     = {response.model}")
print(f"response.stop_reason = {response.stop_reason}")
print(f"response.usage     = input={response.usage.input_tokens} output={response.usage.output_tokens}")
print(f"response.content   = {response.content}")
print()
print("text:", "".join(b.text for b in response.content if b.type == "text"))
