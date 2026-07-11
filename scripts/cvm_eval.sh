#!/usr/bin/env bash
# cvm_eval.sh — S5 CVM cost-stack behavior eval gate.
#
# THIS RUN SPENDS REAL MONEY: it drives 12 scripted agentic tasks through a
# live local Axiom proxy TWICE (flags off, then flags on --
# AXIOM_CVM_DIGEST=skeleton AXIOM_PREFIX_DEDUP=1) using headless
# `claude -p --model claude-haiku-4-5` against your own authenticated
# Anthropic account -- roughly 24 real API calls at Haiku pricing.
#
# This is the ONLY authority for flipping AXIOM_CVM_DIGEST -> skeleton and
# AXIOM_PREFIX_DEDUP -> 1 as defaults. See
# docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S5.
#
#   ./scripts/cvm_eval.sh
#
# Requires: a release build of axiom_engine, the `claude` CLI on PATH and
# already authenticated, curl. Never run this in CI or unattended -- it is
# a human-gated, deliberately manual step.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="$REPO/axiom_engine_rs"
# AXIOM_EVAL_BIN lets an operator point at an alternate build (e.g. one in
# an isolated CARGO_TARGET_DIR) instead of the default release path -- handy
# when a live proxy process already holds that default binary open.
BIN="${AXIOM_EVAL_BIN:-$ENGINE_DIR/target/release/axiom_engine}"
DATE_TAG="$(date +%Y-%m-%d)"
BENCH_DIR="$REPO/bench/cvm"
REPORT="$BENCH_DIR/RESULTS-$DATE_TAG.md"
TMP="$(mktemp -d)"
PORT_OFF=8931
PORT_ON=8932

cleanup() {
    [ -n "${PID_OFF:-}" ] && kill "$PID_OFF" 2>/dev/null
    [ -n "${PID_ON:-}" ] && kill "$PID_ON" 2>/dev/null
    rm -rf "$TMP"
}
trap cleanup EXIT

ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

step "build (release)"
if [ ! -x "$BIN" ]; then
    ( cd "$ENGINE_DIR" && cargo build --release ) || { echo "build failed"; exit 1; }
fi
[ -x "$BIN" ] || { echo "no release binary at $BIN"; exit 1; }

command -v claude >/dev/null 2>&1 || { echo "claude CLI not found on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl not found on PATH"; exit 1; }

mkdir -p "$BENCH_DIR"

# --- 12 tasks: file<TAB>question<TAB>expected-grep-pattern ----------------
# Every question requires >= 1 heavy file read from this repo; every answer
# is a single grep-able fact, verified against the source at authoring time
# (docs/superpowers/plans/2026-07-10-cvm-cost-stack.md, step S5, task 1).
TASKS_FILE="$TMP/tasks.tsv"
cat > "$TASKS_FILE" <<'EOF'
axiom_engine_rs/src/cost_ledger.rs	Read axiom_engine_rs/src/cost_ledger.rs and tell me: what USD price per million input tokens does PriceTable::SONNET use? Answer with just the number.	2\.00
axiom_engine_rs/src/cost_ledger.rs	Read axiom_engine_rs/src/cost_ledger.rs and tell me: what is the cache-read price per million tokens for PriceTable::SONNET_LEGACY (the legacy Sonnet 4.x family)? Answer with just the number.	0\.30
axiom_engine_rs/src/cache_safety.rs	Read axiom_engine_rs/src/cache_safety.rs. In frozen_prefix_len, when there is no per-message cache_control marker but the request body has a top-level cache_control field, how many of the leading messages are frozen? Answer in terms of messages.len().	messages\.len\(\) - 1
axiom_engine_rs/src/digest.rs	Read axiom_engine_rs/src/digest.rs and tell me the exact value of the DEFAULT_DIGEST_THRESHOLD_TOKENS constant.	4000
axiom_engine_rs/src/digest.rs	Read axiom_engine_rs/src/digest.rs and tell me what model string the haiku_digest function uses for AXIOM_CVM_DIGEST=haiku.	claude-haiku-4-5
axiom_engine_rs/src/cvm_store.rs	Read axiom_engine_rs/src/cvm_store.rs and tell me the exact value of MAX_SESSION_BYTES, in bytes (as written in the source, not simplified).	64 \* 1024 \* 1024
axiom_engine_rs/src/cvm_store.rs	Read axiom_engine_rs/src/cvm_store.rs and tell me how CvmStore::page_id_for derives a PageId from text -- name the hash algorithm and how many hex characters are kept.	16.*SHA-256|SHA-256.*16
axiom_engine_rs/src/skeleton.rs	Read axiom_engine_rs/src/skeleton.rs and tell me exactly how many entries the DECL_KEYWORDS array has.	18
axiom_engine_rs/src/context_compressor.rs	Read axiom_engine_rs/src/context_compressor.rs and tell me the exact value of MAX_ADAPT_CHUNK_TOKENS.	128
axiom_engine_rs/src/anthropic_forwarder.rs	Read axiom_engine_rs/src/anthropic_forwarder.rs and tell me the exact string value of the ANTHROPIC_VERSION constant.	2023-06-01
axiom_engine_rs/src/server/routes_responses.rs	Read axiom_engine_rs/src/server/routes_responses.rs and tell me: what HTTP status code does GET /v1/responses return for a request that is not a complete WebSocket handshake?	426
axiom_engine_rs/src/server/routes_tools.rs	Read axiom_engine_rs/src/server/routes_tools.rs and tell me the exact error message POST /mcp returns when AXIOM_MCP_HTTP is not set to 1.	MCP HTTP transport disabled
EOF
TASK_COUNT=$(wc -l < "$TASKS_FILE")

# Run all 12 tasks against $1=base_url, writing per-task pass/fail into
# $2=results_file (tsv: task_index<TAB>pass_or_fail).
run_tasks() {
    local base_url="$1"
    local results_file="$2"
    : > "$results_file"
    local i=0
    while IFS=$'\t' read -r _file question pattern; do
        i=$((i + 1))
        local out
        out=$(ANTHROPIC_BASE_URL="$base_url" claude -p "$question" --model claude-haiku-4-5 2>"$TMP/task_${i}_stderr.log")
        echo "$out" > "$TMP/task_${i}_stdout.log"
        if echo "$out" | grep -qE "$pattern"; then
            echo -e "${i}\tpass" >> "$results_file"
        else
            echo -e "${i}\tfail" >> "$results_file"
        fi
    done < "$TASKS_FILE"
}

# Start a proxy on $1=port with $2=extra env assignments (space-separated
# KEY=VAL), logging to $3=log_file. Echoes the PID.
start_proxy() {
    local port="$1" env_str="$2" log_file="$3"
    (
        eval "$env_str"
        export AXIOM_DEVICE=cpu
        export AXIOM_TTT_COMPRESS=1
        # Pin the CVM store to a known absolute path -- the default is a
        # cwd-relative "checkpoints/cvm", and this function's cwd isn't
        # guaranteed to be $ENGINE_DIR.
        export AXIOM_CVM_DIR="$ENGINE_DIR/checkpoints/cvm"
        "$BIN" --mode server --port "$port" > "$log_file" 2>&1 &
        echo $!
    )
}

wait_for_proxy() {
    local port="$1"
    for _ in $(seq 1 30); do
        curl -sf "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

metric() {
    local port="$1" name="$2"
    curl -s "http://127.0.0.1:$port/metrics" 2>/dev/null | awk -v n="$name" '$1==n {print $2}'
}

# --- Run 1: flags OFF -------------------------------------------------------
step "run 1/2: flags OFF (baseline)"
PID_OFF=$(start_proxy "$PORT_OFF" "" "$TMP/proxy_off.log")
wait_for_proxy "$PORT_OFF" || { echo "proxy (flags off) failed to start"; exit 1; }
run_tasks "http://127.0.0.1:$PORT_OFF" "$TMP/results_off.tsv"
cost_off=$(metric "$PORT_OFF" axiom_cost_usd_total)
kill "$PID_OFF" 2>/dev/null; wait "$PID_OFF" 2>/dev/null; PID_OFF=""

# --- Run 2: flags ON ---------------------------------------------------------
step "run 2/2: flags ON (AXIOM_CVM_DIGEST=skeleton AXIOM_PREFIX_DEDUP=1)"
# Clear any faults.jsonl left over from a prior eval run so the fault-rate
# count below reflects only this run.
rm -f "$ENGINE_DIR/checkpoints/cvm/faults.jsonl"
PID_ON=$(start_proxy "$PORT_ON" "export AXIOM_CVM_DIGEST=skeleton; export AXIOM_PREFIX_DEDUP=1" "$TMP/proxy_on.log")
wait_for_proxy "$PORT_ON" || { echo "proxy (flags on) failed to start"; exit 1; }
run_tasks "http://127.0.0.1:$PORT_ON" "$TMP/results_on.tsv"
cost_on=$(metric "$PORT_ON" axiom_cost_usd_total)
digested_pages=$(grep -o 'blocks=[0-9]*' "$TMP/proxy_on.log" | awk -F= '{sum+=$2} END {print sum+0}')
expand_calls=$(grep -c '"page_id"' "$ENGINE_DIR/checkpoints/cvm/faults.jsonl" 2>/dev/null || echo 0)
kill "$PID_ON" 2>/dev/null; wait "$PID_ON" 2>/dev/null; PID_ON=""

# --- Score -------------------------------------------------------------------
step "score"
pass_off=$(awk -F'\t' '$2=="pass"' "$TMP/results_off.tsv" | wc -l)
pass_on=$(awk -F'\t' '$2=="pass"' "$TMP/results_on.tsv" | wc -l)
fault_rate="0"
if [ "${digested_pages:-0}" -gt 0 ] 2>/dev/null; then
    fault_rate=$(awk -v e="$expand_calls" -v d="$digested_pages" 'BEGIN { printf "%.4f", e/d }')
fi

echo "correctness: flags_off=$pass_off/$TASK_COUNT flags_on=$pass_on/$TASK_COUNT"
echo "fault rate: $expand_calls expand calls / $digested_pages digested pages = $fault_rate"
echo "cost: flags_off=\$$cost_off flags_on=\$$cost_on"

# --- Pass bar (S5 task 3, hard asserts) --------------------------------------
overall_pass=true
if [ "$pass_on" -lt "$((pass_off - 1))" ]; then
    bad "correctness parity (flags-on $pass_on < flags-off-1 $((pass_off - 1)))"
    overall_pass=false
else
    ok "correctness parity"
fi

fault_pct=$(awk -v r="$fault_rate" 'BEGIN { printf "%.2f", r * 100 }')
if awk -v r="$fault_rate" 'BEGIN { exit !(r > 0.05) }'; then
    bad "fault rate ${fault_pct}% > 5%"
    overall_pass=false
else
    ok "fault rate ${fault_pct}% <= 5%"
fi

if awk -v a="$cost_on" -v b="$cost_off" 'BEGIN { exit !(a < b) }'; then
    ok "flags-on cost (\$$cost_on) strictly lower than flags-off (\$$cost_off)"
else
    bad "flags-on cost (\$$cost_on) NOT strictly lower than flags-off (\$$cost_off)"
    overall_pass=false
fi

# --- Report -------------------------------------------------------------------
{
    echo "# CVM S5 Eval Results — $DATE_TAG"
    echo
    echo "Live run against real Anthropic traffic (\`claude -p --model claude-haiku-4-5\`)."
    echo
    echo "| Metric | Flags off | Flags on | Pass bar |"
    echo "|---|---|---|---|"
    echo "| Correctness | $pass_off/$TASK_COUNT | $pass_on/$TASK_COUNT | flags-on >= flags-off - 1 |"
    echo "| Cost (\$) | $cost_off | $cost_on | flags-on strictly lower |"
    echo "| Fault rate | — | ${fault_pct}% ($expand_calls / $digested_pages) | <= 5% |"
    echo
    if [ "$overall_pass" = "true" ]; then
        echo "**Result: PASS** — defaults flipped in this PR."
    else
        echo "**Result: FAIL** — defaults left off; see per-task issues."
    fi
} > "$REPORT"
echo
echo "report written to $REPORT"

if [ "$overall_pass" = "true" ]; then
    exit 0
else
    exit 1
fi
