#!/usr/bin/env bash
# pss_eval.sh — Prolonged-Session Stack (PSS v2) live-eval gate.
#
# Companion to cvm_eval.sh (the S5 gate). Where cvm_eval.sh scores the S0-S8
# cost stack over 12 INDEPENDENT single-shot tasks, this scores the PSS levers
# (P0-P4) over a DEPENDENT chain run in ONE growing session -- the only shape
# that exercises tool deferral (L-A), free-window rebasing (R2), local
# short-circuiting (L-B), adaptive TTL (R3), and high-tier routing (R1).
#
# THIS RUN SPENDS REAL CREDITS. It drives bench/cvm/pss-eval-tasks.tsv through a
# live local Axiom proxy TWICE against your own authenticated Anthropic account:
#   1. flags OFF (baseline)
#   2. all PSS flags ON:
#      AXIOM_TOOL_DEFER=on AXIOM_LOCAL_TRIVIAL=on AXIOM_REBASE_ON_BREAK=on
#      AXIOM_ADAPTIVE_TTL=on AXIOM_MODEL_ROUTE=auto
#
# It is the ONLY authority for flipping those five flags to defaults. This
# script only MEASURES and REPORTS -- flipping the defaults is a SEPARATE PR
# made by hand after a human reads a PASS here (plan step P5.3).
#
#   ./scripts/pss_eval.sh
#
# CREDIT CAVEAT (deep-research 2026-07-11): since 2026-06-15 non-interactive
# `claude -p` draws from a SEPARATE monthly Agent-SDK credit pool ($20 Pro /
# $100 Max 5x / $200 Max 20x), NOT the main subscription window. The savings
# measured here are real token/quota savings, but the spend hits that pool --
# recorded in the report so the numbers are not misread as main-window savings.
#
# Requires: a release build of axiom_engine, the `claude` CLI on PATH and
# already authenticated, curl. Never run in CI or unattended -- human-gated.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="$REPO/axiom_engine_rs"
BIN="${AXIOM_EVAL_BIN:-$ENGINE_DIR/target/release/axiom_engine}"
DATE_TAG="$(date +%Y-%m-%d)"
BENCH_DIR="$REPO/bench/cvm"
TASKS_FILE="$BENCH_DIR/pss-eval-tasks.tsv"
REPORT="$BENCH_DIR/PSS-RESULTS-$DATE_TAG.md"
TMP="$(mktemp -d)"
PORT_OFF=8941
PORT_ON=8942
# Target from the design sim: +56.8% Sonnet / +66.4% Opus / +69.1% Fable.
TARGET_PCT=50

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
command -v curl   >/dev/null 2>&1 || { echo "curl not found on PATH"; exit 1; }
[ -f "$TASKS_FILE" ] || { echo "missing task file: $TASKS_FILE"; exit 1; }
TASK_COUNT=$(wc -l < "$TASKS_FILE")
SCORED_COUNT=$(awk -F'\t' '$2!="SKIP"' "$TASKS_FILE" | wc -l)

# Run the dependent chain in ONE session against $1=base_url, writing per-task
# pass/fail (scored turns only) into $2=results_file. The first turn starts a
# fresh conversation; every later turn uses `--continue` so the session history
# grows across the whole chain -- which is what makes the levers engage.
run_chain() {
    local base_url="$1" results_file="$2"
    : > "$results_file"
    local i=0
    while IFS=$'\t' read -r prompt pattern; do
        i=$((i + 1))
        local out
        if [ "$i" -eq 1 ]; then
            out=$(ANTHROPIC_BASE_URL="$base_url" claude -p "$prompt" \
                --model claude-haiku-4-5 2>"$TMP/t_${i}_err.log")
        else
            out=$(ANTHROPIC_BASE_URL="$base_url" claude -p --continue "$prompt" \
                --model claude-haiku-4-5 2>"$TMP/t_${i}_err.log")
        fi
        echo "$out" > "$TMP/t_${i}_out.log"
        [ "$pattern" = "SKIP" ] && continue
        if echo "$out" | grep -qiE "$pattern"; then
            echo -e "${i}\tpass" >> "$results_file"
        else
            echo -e "${i}\tfail" >> "$results_file"
        fi
    done < "$TASKS_FILE"
}

start_proxy() {
    local port="$1" env_str="$2" log_file="$3"
    (
        eval "$env_str"
        export AXIOM_DEVICE=cpu
        export AXIOM_TTT_COMPRESS=1
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

# --- Run 1: flags OFF (baseline) --------------------------------------------
step "run 1/2: flags OFF (baseline)"
PID_OFF=$(start_proxy "$PORT_OFF" "" "$TMP/proxy_off.log")
wait_for_proxy "$PORT_OFF" || { echo "proxy (off) failed to start"; exit 1; }
run_chain "http://127.0.0.1:$PORT_OFF" "$TMP/results_off.tsv"
quota_off=$(metric "$PORT_OFF" axiom_quota_units_total)
cost_off=$(metric "$PORT_OFF" axiom_cost_usd_total)
kill "$PID_OFF" 2>/dev/null; wait "$PID_OFF" 2>/dev/null; PID_OFF=""

# --- Run 2: all PSS flags ON -------------------------------------------------
step "run 2/2: PSS flags ON"
rm -f "$ENGINE_DIR/checkpoints/cvm/faults.jsonl"
PID_ON=$(start_proxy "$PORT_ON" \
    "export AXIOM_TOOL_DEFER=on; export AXIOM_LOCAL_TRIVIAL=on; export AXIOM_REBASE_ON_BREAK=on; export AXIOM_ADAPTIVE_TTL=on; export AXIOM_MODEL_ROUTE=auto" \
    "$TMP/proxy_on.log")
wait_for_proxy "$PORT_ON" || { echo "proxy (on) failed to start"; exit 1; }
run_chain "http://127.0.0.1:$PORT_ON" "$TMP/results_on.tsv"
quota_on=$(metric "$PORT_ON" axiom_quota_units_total)
cost_on=$(metric "$PORT_ON" axiom_cost_usd_total)
local_answered=$(metric "$PORT_ON" axiom_local_answered_turns_total)
routed_turns=$(metric "$PORT_ON" axiom_routed_turns_total)
routed_saved=$(metric "$PORT_ON" axiom_routed_quota_saved_units_total)
route_fallbacks=$(metric "$PORT_ON" axiom_route_fallbacks_total)
# Fault signals: CVM expand faults (a rebased/deferred page had to be reloaded)
# plus routing fallbacks (a downgrade the upstream rejected).
expand_faults=$(grep -c '"page_id"' "$ENGINE_DIR/checkpoints/cvm/faults.jsonl" 2>/dev/null || echo 0)
kill "$PID_ON" 2>/dev/null; wait "$PID_ON" 2>/dev/null; PID_ON=""

# --- Score -------------------------------------------------------------------
step "score"
pass_off=$(awk -F'\t' '$2=="pass"' "$TMP/results_off.tsv" | wc -l)
pass_on=$(awk -F'\t' '$2=="pass"' "$TMP/results_on.tsv" | wc -l)

# Numeric guards: treat missing metrics as 0 / avoid divide-by-zero.
quota_off=${quota_off:-0}; quota_on=${quota_on:-0}
cost_off=${cost_off:-0};   cost_on=${cost_on:-0}
local_answered=${local_answered:-0}; routed_turns=${routed_turns:-0}
routed_saved=${routed_saved:-0}; route_fallbacks=${route_fallbacks:-0}

savings_pct=$(awk -v a="$quota_off" -v b="$quota_on" \
    'BEGIN { if (a+0 > 0) printf "%.1f", (a-b)/a*100; else print "0.0" }')
total_faults=$(awk -v e="$expand_faults" -v f="$route_fallbacks" 'BEGIN { print e+f }')
fault_rate=$(awk -v ft="$total_faults" -v n="$TASK_COUNT" \
    'BEGIN { if (n+0 > 0) printf "%.4f", ft/n; else print "0.0" }')
fault_pct=$(awk -v r="$fault_rate" 'BEGIN { printf "%.2f", r*100 }')

echo "correctness: off=$pass_off/$SCORED_COUNT on=$pass_on/$SCORED_COUNT"
echo "quota units: off=$quota_off on=$quota_on  (savings ${savings_pct}%)"
echo "cost USD:    off=$cost_off on=$cost_on"
echo "L-B local-answered=$local_answered  R1 routed=$routed_turns saved_units=$routed_saved fallbacks=$route_fallbacks"
echo "fault rate:  ${fault_pct}% ($total_faults faults / $TASK_COUNT turns)"

# --- Pass bar ----------------------------------------------------------------
overall_pass=true

if [ "$pass_on" -lt "$((pass_off - 1))" ]; then
    bad "correctness parity (on $pass_on < off-1 $((pass_off - 1)))"; overall_pass=false
else
    ok "correctness parity"
fi

if awk -v a="$quota_on" -v b="$quota_off" 'BEGIN { exit !(a+0 < b+0) }'; then
    ok "quota units strictly lower (on $quota_on < off $quota_off)"
else
    bad "quota units NOT strictly lower (on $quota_on vs off $quota_off)"; overall_pass=false
fi

if awk -v r="$fault_rate" 'BEGIN { exit !(r+0 > 0.05) }'; then
    bad "fault rate ${fault_pct}% > 5%"; overall_pass=false
else
    ok "fault rate ${fault_pct}% <= 5%"
fi

# The headline goal is a >=50% quota reduction. Reported prominently; a miss is
# flagged but does not by itself fail the gate (a correctness-preserving 30%
# is still a real win) -- the human decides whether to flip defaults below 50%.
if awk -v p="$savings_pct" -v t="$TARGET_PCT" 'BEGIN { exit !(p+0 >= t+0) }'; then
    ok "quota savings ${savings_pct}% >= ${TARGET_PCT}% target"
    target_hit=true
else
    printf '  \033[33mMISS\033[0m quota savings %s%% < %s%% target (review before flipping)\n' "$savings_pct" "$TARGET_PCT"
    target_hit=false
fi

# --- Report ------------------------------------------------------------------
{
    echo "# PSS v2 Live-Eval Results — $DATE_TAG"
    echo
    echo "Live run of the dependent chain in \`bench/cvm/pss-eval-tasks.tsv\` through a"
    echo "local Axiom proxy, flags OFF vs all PSS levers ON, via"
    echo "\`claude -p --continue --model claude-haiku-4-5\` (one growing session)."
    echo
    echo "> **Credit note:** non-interactive \`claude -p\` spend since 2026-06-15 draws"
    echo "> from the separate Agent-SDK credit pool, not the main subscription window."
    echo "> These are real token/quota savings; the *spend* hit that pool."
    echo
    echo "| Metric | Flags off | PSS on | Gate |"
    echo "|---|---|---|---|"
    echo "| Correctness | $pass_off/$SCORED_COUNT | $pass_on/$SCORED_COUNT | on >= off - 1 |"
    echo "| Quota units | $quota_off | $quota_on | on strictly lower |"
    echo "| Quota savings | — | ${savings_pct}% | target >= ${TARGET_PCT}% |"
    echo "| Cost (USD) | $cost_off | $cost_on | (secondary) |"
    echo "| Fault rate | — | ${fault_pct}% ($total_faults/$TASK_COUNT) | <= 5% |"
    echo
    echo "PSS-on lever activity: L-B local-answered=$local_answered · R1 routed=$routed_turns (saved $routed_saved units, $route_fallbacks fallbacks)."
    echo
    if [ "$overall_pass" = "true" ] && [ "$target_hit" = "true" ]; then
        echo "**Result: PASS (>= ${TARGET_PCT}% target met).** Proceed to plan step P5.3:"
        echo "a SEPARATE PR flips the five lever defaults on and updates README/CAPABILITIES"
        echo "with these measured numbers (kept honestly distinct from the simulation)."
    elif [ "$overall_pass" = "true" ]; then
        echo "**Result: PASS on gates, BELOW ${TARGET_PCT}% target (${savings_pct}%).** Levers are"
        echo "correctness-preserving and cheaper, but under goal. Human decides whether to flip"
        echo "defaults now or return to brainstorming for more headroom."
    else
        echo "**Result: FAIL.** Defaults stay OFF. See the failed gate(s) above; per plan"
        echo "step P5.3-FAIL, return to brainstorming."
    fi
} > "$REPORT"
echo
echo "report written to $REPORT"

# Exit 0 only when every hard gate passed AND the 50% target was met, so an
# unattended caller can treat exit 0 as "clear to flip defaults".
if [ "$overall_pass" = "true" ] && [ "$target_hit" = "true" ]; then
    exit 0
else
    exit 1
fi
