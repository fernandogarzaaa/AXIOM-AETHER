// Non-billable token-savings measurement for the Axiom-TTT proxy.
//
// Stands up a mock Anthropic upstream that captures the EXACT payload the proxy
// forwards, fires an original heavy payload at the proxy, then compares:
//   * original input tokens  (what Claude Code would send WITHOUT Axiom)
//   * outbound input tokens   (the lean payload Axiom actually forwards upstream)
//
// Token count uses the same whitespace-splitting proxy the server uses
// (anthropic_forwarder.rs::whitespace_token_count) so the numbers line up with
// the [axiom-ttt] heavy_tokens log line. This is a wire-size compression ratio,
// NOT a semantic-fidelity claim.
const http = require("http");

const MOCK_PORT = Number(process.env.MOCK_PORT || 8788);
const PROXY = process.env.PROXY_URL || "http://127.0.0.1:3000";
const SESSION = process.env.SESSION_ID || "token-test-session";

// Whitespace token proxy — matches server's whitespace_token_count().
const wsTokens = (s) => (s.match(/\S+/g) || []).length;

// Recursively pull all text out of an Anthropic messages payload.
function payloadText(body) {
  let parts = [];
  if (body.system) parts.push(typeof body.system === "string" ? body.system : JSON.stringify(body.system));
  for (const m of body.messages || []) {
    const c = m.content;
    if (typeof c === "string") parts.push(c);
    else if (Array.isArray(c)) for (const blk of c) parts.push(blk.text || JSON.stringify(blk));
    else if (c) parts.push(JSON.stringify(c));
  }
  return parts.join("\n");
}

let captured = null;

const mock = http.createServer((req, res) => {
  let data = "";
  req.on("data", (c) => (data += c));
  req.on("end", () => {
    try {
      captured = JSON.parse(data);
    } catch {
      captured = { _raw: data };
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        id: "msg_mock",
        type: "message",
        role: "assistant",
        model: "claude-mock",
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 0, output_tokens: 1 },
      })
    );
  });
});

function buildHeavyPayload() {
  // ~600 "tokens" of heavy context + a short real question.
  const heavy =
    "ARCHITECTURE DUMP. " +
    Array.from({ length: 600 }, (_, i) => `module_${i}_does_thing_${i}`).join(" ");
  return {
    model: "claude-opus-4-7",
    max_tokens: 64,
    messages: [
      { role: "user", content: heavy },
      { role: "user", content: "Name one risk in that code. One sentence." },
    ],
  };
}

async function main() {
  await new Promise((r) => mock.listen(MOCK_PORT, "127.0.0.1", r));

  const original = buildHeavyPayload();
  const originalTokens = wsTokens(payloadText(original));

  const resp = await fetch(`${PROXY}/v1/messages`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-axiom-session-id": SESSION,
    },
    body: JSON.stringify(original),
  });
  const status = resp.status;
  await resp.text();

  // Give the mock a tick to finish parsing.
  await new Promise((r) => setTimeout(r, 150));

  if (!captured) {
    console.log(JSON.stringify({ error: "proxy did not forward to mock", proxy_status: status }));
    mock.close();
    process.exit(2);
  }

  const outboundTokens = wsTokens(payloadText(captured));
  const saved = originalTokens - outboundTokens;
  const ratio = originalTokens > 0 ? saved / originalTokens : 0;

  console.log(
    JSON.stringify(
      {
        proxy_status: status,
        original_input_tokens: originalTokens,
        outbound_input_tokens: outboundTokens,
        tokens_saved: saved,
        compression_ratio_pct: Number((ratio * 100).toFixed(1)),
        outbound_preview: payloadText(captured).slice(0, 320),
      },
      null,
      2
    )
  );
  mock.close();
  process.exit(0);
}

main().catch((e) => {
  console.error("test error:", e);
  mock.close();
  process.exit(1);
});
