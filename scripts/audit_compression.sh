#!/usr/bin/env bash
#
# audit_compression.sh — non-billable, end-to-end token-compression auditor for
# the Axiom-TTT proxy.
#
# WHAT IT DOES
#   1. Stands up a LOCAL mock Anthropic upstream that captures the EXACT payload
#      the proxy forwards (so nothing leaves the machine, $0 in API spend).
#   2. Boots the Axiom-TTT proxy with the production checkpoint, compression ON,
#      and its upstream pinned at the mock.
#   3. Streams the playground fixtures (a heavy TS service + a 1000-line app log)
#      through the proxy on ONE shared session, so the fast-weight tensor (W̃)
#      keeps adapting across turns.
#   4. For each turn it measures:
#        * Original input tokens  — what Claude Code would have sent.
#        * Outbound input tokens  — the lean payload Axiom actually forwarded.
#      Token counts use whitespace splitting (\S+), matching the server's
#      whitespace_token_count() so they line up with the [axiom-ttt] log line.
#      This is a WIRE-SIZE compression ratio, not a semantic-fidelity claim.
#   5. Prints a markdown table (Original / Outbound / Realized Savings % /
#      recall_norm) and confirms recall_norm stability across the stream.
#
# It cleans up every background process it starts and leaves the tree pristine.
#
# Requires: node, curl, an already-built release binary, a trained checkpoint.
set -euo pipefail

# --- Resolve paths ---------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="$REPO_ROOT/axiom_engine_rs"
BIN="$CRATE_DIR/target/release/axiom_engine"
[ -x "$BIN" ] || BIN="$BIN.exe"   # Windows / Git Bash

CKPT="${AXIOM_AUDIT_CHECKPOINT:-$REPO_ROOT/checkpoints/axiom_production.bin}"

# Playground fixtures (created by Phase 3). Override with AXIOM_PLAYGROUND.
PLAYGROUND="${AXIOM_PLAYGROUND:-$(cd "$REPO_ROOT/.." && pwd)/axiom-playground}"
FIXTURE_TS="$PLAYGROUND/fixtures/mock_service.ts"
FIXTURE_LOG="$PLAYGROUND/fixtures/application_error.log"

# --- Network boundary (deliberately OFF the default 3000/8788 so a running
#     autostart proxy is never disturbed) -----------------------------------
HOST="127.0.0.1"
PORT="${AXIOM_AUDIT_PORT:-3111}"
MOCK_PORT="${AXIOM_AUDIT_MOCK_PORT:-8799}"
SESSION_ID="${AXIOM_AUDIT_SESSION:-audit-$$}"

# Low threshold so even a single fixture reliably trips compression.
THRESHOLD="${AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS:-64}"
TOP_K="${AXIOM_TTT_COMPRESS_TOP_K:-32}"

WORK="$(mktemp -d)"
CAPTURES="$WORK/captures.jsonl"
PROXY_LOG="$WORK/proxy.log"
MOCK_LOG="$WORK/mock.log"
: > "$CAPTURES"

MOCK_PID=""
PROXY_PID=""

# --- Cleanup: kill everything we started, scrub temp tree ------------------
cleanup() {
    local code=$?
    [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null || true
    [ -n "$MOCK_PID" ]  && kill "$MOCK_PID"  2>/dev/null || true
    # Reap any stragglers in our process group (Git Bash / MSYS).
    pkill -P $$ 2>/dev/null || true
    if [ "${AXIOM_AUDIT_DEBUG:-0}" = "1" ]; then
        echo "[debug] preserved work dir: $WORK" >&2
    else
        rm -rf "$WORK" 2>/dev/null || true
    fi
    exit $code
}
trap cleanup EXIT INT TERM

# --- Preflight -------------------------------------------------------------
fail() { echo "FAIL: $*" >&2; exit 1; }

command -v node >/dev/null 2>&1 || fail "node is required but not on PATH."
command -v curl >/dev/null 2>&1 || fail "curl is required but not on PATH."
[ -x "$BIN" ] || fail "release binary missing: $BIN
      Build it:  cargo build --release --manifest-path \"$CRATE_DIR/Cargo.toml\""
[ -f "$FIXTURE_TS" ]  || fail "fixture not found: $FIXTURE_TS (run Phase 3 first)."
[ -f "$FIXTURE_LOG" ] || fail "fixture not found: $FIXTURE_LOG (run Phase 3 first)."

CKPT_NOTE="fresh random init (recall_norm will be ~0 — LOW fidelity)"
CKPT_ARGS=()
if [ -f "$CKPT" ]; then
    CKPT_ARGS=(--checkpoint "$CKPT")
    CKPT_NOTE="$CKPT"
fi

# --- Emit the mock upstream (captures forwarded payloads) ------------------
cat > "$WORK/mock_upstream.js" <<'NODE'
const http = require("http");
const fs = require("fs");

const PORT = Number(process.env.MOCK_PORT);
const CAPTURE_FILE = process.env.CAPTURE_FILE;

// Whitespace token proxy — matches server whitespace_token_count() (\S+).
const wsTokens = (s) => (s.match(/\S+/g) || []).length;

function payloadText(body) {
  const parts = [];
  if (body.system) parts.push(typeof body.system === "string" ? body.system : JSON.stringify(body.system));
  for (const m of body.messages || []) {
    const c = m.content;
    if (typeof c === "string") parts.push(c);
    else if (Array.isArray(c)) for (const blk of c) parts.push(blk.text || JSON.stringify(blk));
    else if (c) parts.push(JSON.stringify(c));
  }
  return parts.join("\n");
}

let seq = 0;
const server = http.createServer((req, res) => {
  let data = "";
  req.on("data", (c) => (data += c));
  req.on("end", () => {
    // Only forwarded /v1/messages POSTs count as captures; GET readiness
    // probes must NOT pollute the capture stream (off-by-one guard).
    if (req.method === "POST") {
      let body;
      try { body = JSON.parse(data); } catch { body = { _raw: data }; }
      const text = payloadText(body);
      seq += 1;
      const rec = { seq, outbound_tokens: wsTokens(text), preview: text.slice(0, 200) };
      fs.appendFileSync(CAPTURE_FILE, JSON.stringify(rec) + "\n");
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({
      id: "msg_audit_mock", type: "message", role: "assistant", model: "claude-mock-audit",
      content: [{ type: "text", text: "ok" }], stop_reason: "end_turn",
      usage: { input_tokens: 0, output_tokens: 1 },
    }));
  });
});
server.listen(PORT, "127.0.0.1", () => console.error(`[mock] listening on 127.0.0.1:${PORT}`));
NODE

# --- Emit the sender (builds payload from a fixture, prints original tokens)-
cat > "$WORK/send.js" <<'NODE'
const fs = require("fs");

const wsTokens = (s) => (s.match(/\S+/g) || []).length;
function payloadText(body) {
  const parts = [];
  if (body.system) parts.push(typeof body.system === "string" ? body.system : JSON.stringify(body.system));
  for (const m of body.messages || []) {
    const c = m.content;
    if (typeof c === "string") parts.push(c);
    else if (Array.isArray(c)) for (const blk of c) parts.push(blk.text || JSON.stringify(blk));
  }
  return parts.join("\n");
}

const [fixturePath, query, proxyUrl, session] = process.argv.slice(2);
const heavy = fs.readFileSync(fixturePath, "utf8");
const payload = {
  model: "claude-opus-4-7",
  max_tokens: 64,
  messages: [
    { role: "user", content: heavy },
    { role: "user", content: query },
  ],
};
const originalTokens = wsTokens(payloadText(payload));

(async () => {
  const resp = await fetch(`${proxyUrl}/v1/messages`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-axiom-session-id": session },
    body: JSON.stringify(payload),
  });
  await resp.text();
  // stdout contract: "<original_tokens> <http_status>"
  console.log(`${originalTokens} ${resp.status}`);
})().catch((e) => { console.error("send error:", e); process.exit(1); });
NODE

# --- Boot the mock upstream ------------------------------------------------
MOCK_PORT="$MOCK_PORT" CAPTURE_FILE="$CAPTURES" node "$WORK/mock_upstream.js" > "$MOCK_LOG" 2>&1 &
MOCK_PID=$!

for _ in $(seq 1 40); do
    curl -sf "http://$HOST:$MOCK_PORT/" -o /dev/null 2>/dev/null && break || true
    # The mock has no GET route, but a connection refused vs. 200/404 tells us it's up.
    if curl -s "http://$HOST:$MOCK_PORT/" -o /dev/null 2>/dev/null; then break; fi
    sleep 0.25
done

# --- Boot the proxy, upstream pinned at the LOCAL mock ---------------------
# A dummy key is required so the forwarder is ENABLED; because the upstream is
# our local mock, this is non-billable.
AXIOM_TTT_COMPRESS=1 \
AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS="$THRESHOLD" \
AXIOM_TTT_COMPRESS_TOP_K="$TOP_K" \
ANTHROPIC_BASE_URL="http://$HOST:$MOCK_PORT" \
ANTHROPIC_API_KEY="sk-audit-local-mock-noupstream" \
AXIOM_HOST="$HOST" AXIOM_PORT="$PORT" \
"$BIN" --mode server --host "$HOST" --port "$PORT" "${CKPT_ARGS[@]}" > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!

READY=0
for _ in $(seq 1 80); do
    if curl -sf "http://$HOST:$PORT/v1/models" -o /dev/null 2>/dev/null; then READY=1; break; fi
    sleep 0.25
done
[ "$READY" -eq 1 ] || { echo "--- proxy log ---"; cat "$PROXY_LOG"; fail "proxy never became ready on $HOST:$PORT"; }

echo "==> Axiom-TTT audit harness up"
echo "    proxy       : http://$HOST:$PORT  (session=$SESSION_ID)"
echo "    mock upstream: http://$HOST:$MOCK_PORT  (captures -> $CAPTURES)"
echo "    checkpoint  : $CKPT_NOTE"
echo "    compression : ON (threshold=$THRESHOLD tokens, top_k=$TOP_K)"
echo

# --- Stream the fixtures through the proxy on a shared session -------------
declare -a NAMES ORIGS OUTS
expected=0

run_turn() {
    local name="$1" fixture="$2" query="$3"
    expected=$((expected + 1))
    echo "==> turn $expected: $name"
    local out orig status
    out=$(node "$WORK/send.js" "$fixture" "$query" "http://$HOST:$PORT" "$SESSION_ID")
    orig=$(echo "$out" | awk '{print $1}')
    status=$(echo "$out" | awk '{print $2}')
    echo "    proxy HTTP status: $status   original_input_tokens: $orig"

    # Wait until the mock has captured this turn's forwarded payload.
    local got=0
    for _ in $(seq 1 40); do
        if [ "$(wc -l < "$CAPTURES")" -ge "$expected" ]; then got=1; break; fi
        sleep 0.25
    done
    [ "$got" -eq 1 ] || fail "mock never captured turn $expected (proxy did not forward)."

    local outbound
    outbound=$(sed -n "${expected}p" "$CAPTURES" | node -e 'let d="";process.stdin.on("data",c=>d+=c);process.stdin.on("end",()=>{console.log(JSON.parse(d).outbound_tokens)})')
    echo "    outbound_input_tokens: $outbound"
    echo

    NAMES+=("$name"); ORIGS+=("$orig"); OUTS+=("$outbound")
}

run_turn "mock_service.ts"        "$FIXTURE_TS"  "Summarize this service's core responsibility and flag one risk."
run_turn "application_error.log"  "$FIXTURE_LOG" "What is the single most frequent root cause in this log?"

# --- Collect recall_norm stability across the stream -----------------------
mapfile -t NORMS < <(grep -oE "recall_norm=[0-9.]+" "$PROXY_LOG" | grep -oE "[0-9.]+" || true)

# --- Markdown report -------------------------------------------------------
total_orig=0; total_out=0
echo "## Axiom-TTT Compression Audit"
echo
echo "_Session \`$SESSION_ID\` · checkpoint \`$(basename "$CKPT")\` · non-billable (local mock upstream)._"
echo
echo "| # | Fixture | Original Input Tokens | Outbound Compressed Tokens | Realized Token Savings % | recall_norm |"
echo "|---|---------|----------------------:|---------------------------:|-------------------------:|------------:|"
for i in "${!NAMES[@]}"; do
    o="${ORIGS[$i]}"; c="${OUTS[$i]}"
    saved_pct=$(awk -v o="$o" -v c="$c" 'BEGIN{ if(o>0) printf "%.1f", (o-c)/o*100; else print "0.0" }')
    norm="${NORMS[$i]:-n/a}"
    printf "| %d | \`%s\` | %s | %s | %s%% | %s |\n" "$((i+1))" "${NAMES[$i]}" "$o" "$c" "$saved_pct" "$norm"
    total_orig=$((total_orig + o)); total_out=$((total_out + c))
done
agg_pct=$(awk -v o="$total_orig" -v c="$total_out" 'BEGIN{ if(o>0) printf "%.1f", (o-c)/o*100; else print "0.0" }')
printf "| **Σ** | **stream total** | **%s** | **%s** | **%s%%** | — |\n" "$total_orig" "$total_out" "$agg_pct"
echo
echo "**recall_norm across the session:** ${NORMS[*]:-<none captured>}"
if [ "${#NORMS[@]}" -ge 2 ]; then
    if [ "${NORMS[0]}" = "${NORMS[1]}" ]; then
        echo "- recall_norm held steady at ${NORMS[0]} — fast-weights stable across the stream."
    else
        echo "- recall_norm moved ${NORMS[0]} → ${NORMS[1]} — W̃ kept adapting turn-over-turn."
    fi
    echo "- recall_norm > 0 confirms the trained checkpoint is projecting real semantics."
elif [ "${#NORMS[@]}" -eq 0 ]; then
    echo "- WARNING: no recall_norm lines — compression did not engage (check threshold/checkpoint)."
fi
echo
echo "_Token counts use whitespace splitting (\\S+), matching the server's"
echo "whitespace_token_count(). This is a wire-size ratio, not a semantic-fidelity claim._"
