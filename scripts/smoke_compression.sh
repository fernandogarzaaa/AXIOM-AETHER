#!/usr/bin/env bash
# End-to-end smoke of the active-compression pipeline.
#
# 1. Spawn a tiny mock Anthropic /v1/messages endpoint on :8765 that
#    captures and echoes the incoming payload.
# 2. Boot the Axiom server with compression enabled, pointed at the mock.
# 3. POST a /v1/messages request with a heavy code dump + a short query.
# 4. Print the captured upstream payload — proves heavy text was
#    stripped and replaced with the AXIOM-TTT fingerprint.
#
# Requires: python3, curl, an already-built `axiom_engine` release binary.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO_ROOT/axiom_engine_rs/target/release/axiom_engine"
MOCK_LOG="$(mktemp)"
AXIOM_LOG="$(mktemp)"
CAPTURE_PATH="$(mktemp)"

[ -x "$BIN" ] || {
    echo "release binary not built; run: cargo build --release --manifest-path axiom_engine_rs/Cargo.toml"
    exit 1
}

# --- Mock Anthropic upstream ----------------------------------------------
python3 - "$CAPTURE_PATH" <<'PY' > "$MOCK_LOG" 2>&1 &
import sys, json
from http.server import BaseHTTPRequestHandler, HTTPServer

CAPTURE = sys.argv[1]

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n).decode()
        with open(CAPTURE, "w") as f:
            f.write(body)
        resp = {
            "id": "msg_smoke",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "mock reply"}],
            "model": "claude-mock",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a, **kw):
        pass

HTTPServer(("127.0.0.1", 8765), H).serve_forever()
PY
MOCK_PID=$!
trap 'kill "$MOCK_PID" 2>/dev/null || true; pkill -P $$ 2>/dev/null || true' EXIT
sleep 0.5

# --- Axiom server with compression on -------------------------------------
AXIOM_TTT_COMPRESS=1 \
AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS=50 \
AXIOM_TTT_COMPRESS_TOP_K=8 \
ANTHROPIC_API_KEY=test-key \
ANTHROPIC_BASE_URL=http://127.0.0.1:8765 \
"$BIN" --mode server --host 127.0.0.1 --port 8766 > "$AXIOM_LOG" 2>&1 &
AXIOM_PID=$!
trap 'kill "$AXIOM_PID" "$MOCK_PID" 2>/dev/null || true; pkill -P $$ 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8766/v1/models > /dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 0.5
done
if [ "$READY" -ne 1 ]; then
    echo "FAIL: Axiom server never became ready on :8766 within 30s"
    echo "--- axiom server log ---"
    cat "$AXIOM_LOG"
    exit 1
fi
echo "==> Axiom up; compression-mode banner from server log:"
grep -E "Active-compression|listening" "$AXIOM_LOG" | sed 's/^/    /'

# --- Build a heavy code dump + send via /v1/messages ----------------------
HEAVY=$(python3 -c "print(' '.join(f'fn_chunk_{i}()' for i in range(400)))")
PAYLOAD=$(python3 - "$HEAVY" <<'PY'
import json, sys
heavy = sys.argv[1]
print(json.dumps({
    "model": "claude-opus-4-7",
    "max_tokens": 64,
    "messages": [
        {"role": "user", "content": heavy},
        {"role": "user", "content": "describe the architecture of that codebase"}
    ],
    "session_id": "smoke-session"
}))
PY
)

echo
echo "==> Sending POST /v1/messages with $(echo "$HEAVY" | wc -w) heavy tokens"
RESPONSE=$(curl -sS -X POST http://127.0.0.1:8766/v1/messages \
    -H "content-type: application/json" \
    -d "$PAYLOAD")
echo "==> proxy response: $RESPONSE"

echo
echo "==> Axiom compression log line:"
grep "axiom-ttt" "$AXIOM_LOG" | sed 's/^/    /' || echo "    (no log line found)"

echo
echo "==> Upstream payload captured by mock Anthropic:"
python3 - "$CAPTURE_PATH" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
messages = body.get("messages", [])
print(f"    upstream model:        {body.get('model')}")
print(f"    upstream max_tokens:   {body.get('max_tokens')}")
print(f"    upstream message count: {len(messages)}")
for i, m in enumerate(messages):
    c = m.get("content", "")
    if isinstance(c, list):
        c = "".join(b.get("text", "") for b in c)
    if len(c) > 320:
        c = c[:320] + f"... [+{len(c) - 320} chars]"
    print(f"    msg[{i}].role={m.get('role')!r}  content={c!r}")
PY

echo
echo "==> Confirming heavy raw text was NOT forwarded upstream:"
if grep -q "fn_chunk_399" "$CAPTURE_PATH"; then
    echo "    FAIL: raw heavy text leaked to upstream"
    exit 1
else
    echo "    OK: raw heavy text stripped from outbound payload"
fi
if grep -q "AXIOM-TTT-CONTEXT-FINGERPRINT" "$CAPTURE_PATH"; then
    echo "    OK: fingerprint block present in outbound payload"
else
    echo "    FAIL: fingerprint missing from outbound payload"
    exit 1
fi
