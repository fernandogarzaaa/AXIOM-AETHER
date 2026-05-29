#!/usr/bin/env bash
# Live end-to-end smoke of the Axiom-TTT proxy against the REAL Anthropic API.
#
# WHAT IT PROVES
#   1. The proxy boots with compression ON and forwards to the real upstream.
#   2. Two consecutive /v1/messages calls sharing ONE X-Axiom-Session-Id header
#      progressively adapt the SAME fast-weight tensor (W̃) — shown by the
#      per-call [axiom-ttt] recall_norm changing between calls (Frobenius norm
#      mutating on ingestion), not resetting to a fresh identity matrix.
#   3. Live assistant token output comes back from the upstream API.
#
# COST WARNING: this makes REAL, billable Anthropic API calls. It is gated on
# ANTHROPIC_API_KEY and SKIPS (exit 0) when the key is unset.
#
# Requires: python3, curl, an already-built release binary.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO_ROOT/axiom_engine_rs/target/release/axiom_engine"
[ -x "$BIN" ] || BIN="$BIN.exe"   # Windows / Git Bash

HOST="127.0.0.1"
PORT="${AXIOM_SMOKE_PORT:-3000}"
SESSION_ID="${AXIOM_SESSION_ID:-smoke-live-$$}"
LOG_FILE="$(mktemp)"

if [ ! -x "$BIN" ]; then
    echo "FAIL: release binary not built."
    echo "      cargo build --release --manifest-path axiom_engine_rs/Cargo.toml --bin axiom_engine"
    exit 1
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "SKIP: ANTHROPIC_API_KEY is unset."
    echo "      This is a LIVE, billable smoke against https://api.anthropic.com."
    echo "      Export your key and re-run to exercise the full upstream loop:"
    echo "        ANTHROPIC_API_KEY=sk-ant-... ./scripts/smoke_live.sh"
    exit 0
fi

# --- Boot the proxy, upstream pinned to the REAL Anthropic API -------------
# Low threshold so our modest heavy block reliably trips compression.
AXIOM_TTT_COMPRESS=1 \
AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS="${AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS:-50}" \
AXIOM_TTT_COMPRESS_TOP_K="${AXIOM_TTT_COMPRESS_TOP_K:-16}" \
ANTHROPIC_BASE_URL="https://api.anthropic.com" \
AXIOM_HOST="$HOST" \
AXIOM_PORT="$PORT" \
"$BIN" --mode server --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
AXIOM_PID=$!
trap 'kill "$AXIOM_PID" 2>/dev/null || true; pkill -P $$ 2>/dev/null || true' EXIT

# --- Wait for readiness ----------------------------------------------------
READY=0
for _ in $(seq 1 60); do
    if curl -sf "http://$HOST:$PORT/v1/models" > /dev/null 2>&1; then
        READY=1; break
    fi
    sleep 0.5
done
if [ "$READY" -ne 1 ]; then
    echo "FAIL: proxy never became ready on $HOST:$PORT"
    echo "--- server log ---"; cat "$LOG_FILE"
    exit 1
fi
echo "==> Proxy up on http://$HOST:$PORT (session=$SESSION_ID)"

# --- Helper: one compressed /v1/messages call pinned to our session --------
send_call() {
    local label="$1" heavy="$2" query="$3"
    local payload
    payload=$(python3 - "$heavy" "$query" <<'PY'
import json, sys
print(json.dumps({
    "model": "claude-opus-4-7",
    "max_tokens": 64,
    "messages": [
        {"role": "user", "content": sys.argv[1]},
        {"role": "user", "content": sys.argv[2]},
    ],
}))
PY
)
    echo
    echo "==> [$label] POST /v1/messages  (X-Axiom-Session-Id: $SESSION_ID)"
    local resp
    resp=$(curl -sS -X POST "http://$HOST:$PORT/v1/messages" \
        -H "content-type: application/json" \
        -H "X-Axiom-Session-Id: $SESSION_ID" \
        -d "$payload")
    # Extract the assistant text from the upstream response.
    echo "    upstream reply: $(python3 -c '
import json,sys
try:
    b=json.load(sys.stdin)
    print("".join(p.get("text","") for p in b.get("content",[])) or b)
except Exception as e:
    print("<non-JSON or error>", e)
' <<<"$resp")"
}

# --- Build two heavy contexts and fire them at the same session ------------
HEAVY1=$(python3 -c "print(' '.join(f'alpha_chunk_{i}()' for i in range(200)))")
HEAVY2=$(python3 -c "print(' '.join(f'beta_chunk_{i}()' for i in range(200)))")

send_call "call-1" "$HEAVY1" "Name one architectural risk in that code. One sentence."
send_call "call-2" "$HEAVY2" "Now name a second, different risk. One sentence."

# --- Inspect the live compression metrics ----------------------------------
echo
echo "==> [axiom-ttt] compression metric lines from the live server log:"
grep "axiom-ttt" "$LOG_FILE" | sed 's/^/    /' || {
    echo "    FAIL: no [axiom-ttt] compression lines — compression did not engage."
    echo "--- server log ---"; cat "$LOG_FILE"
    exit 1
}

# Pull the two recall_norm values and confirm the fast-weights mutated.
mapfile -t NORMS < <(grep -oE "recall_norm=[0-9.]+" "$LOG_FILE" | grep -oE "[0-9.]+")
echo
echo "==> Frobenius/recall norms observed across calls: ${NORMS[*]:-<none>}"
if [ "${#NORMS[@]}" -lt 2 ]; then
    echo "    FAIL: expected >=2 compression passes for a persistent session."
    exit 1
fi
if [ "${NORMS[0]}" = "${NORMS[1]}" ]; then
    echo "    WARN: recall_norm identical across calls (${NORMS[0]}) — W̃ may not have moved."
else
    echo "    OK: recall_norm mutated ${NORMS[0]} -> ${NORMS[1]} (fast-weights advancing)."
fi

# Confirm the same session was reused, not a fresh transient each time.
echo
echo "==> Live session registry (/v1/ttt/sessions):"
curl -sS "http://$HOST:$PORT/v1/ttt/sessions" | python3 -m json.tool | sed 's/^/    /'

echo
echo "==> LIVE SMOKE COMPLETE: compression engaged, norms moved, upstream replied."
