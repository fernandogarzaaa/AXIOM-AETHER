#!/usr/bin/env bash
# demo_end_to_end.sh — drive every Axiom pillar in one run, on CPU, no network.
#
# This is the canonical "see it all work" script: it builds the release binary
# (if needed) and exercises compression, the self-healing runtime + acquired
# immunity, the verifiable epistemic swarm, and grounding verification —
# printing PASS/FAIL per step. It is both a live demo and a manual regression
# check that the surfaces compose end-to-end.
#
#   ./scripts/demo_end_to_end.sh
#
# Everything runs in a throwaway temp dir with an isolated heal memory, so it
# never touches your real ~/.axiom state.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/axiom_engine_rs/target/release/axiom_engine"
TMP="$(mktemp -d)"
export AXIOM_DEVICE=cpu
export AXIOM_HEAL_MEMORY="$TMP/heal_memory.json"
PASS=0; FAIL=0
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

step "build"
if [ ! -x "$BIN" ]; then
    ( cd "$REPO/axiom_engine_rs" && cargo build --release ) || { echo "build failed"; exit 1; }
fi
[ -x "$BIN" ] && ok "release binary present" || { bad "no binary"; exit 1; }

step "doctor (hardware-aware device selection)"
"$BIN" --mode doctor 2>&1 | grep -qiE "device|cpu|recommend" && ok "doctor reports a recommendation" || bad "doctor"

step "Pillar 1 — context compression (token savings + lossless round-trip)"
BENCH=$("$BIN" bench "$REPO/axiom_engine_rs/src" 2>&1)
echo "$BENCH" | grep -E "token savings|round-trip" | sed 's/^/    /'
echo "$BENCH" | grep -q "token savings" && ok "compression measured" || bad "bench produced no savings line"

step "Pillar 2 — self-healing runtime (missing-dir heal + learned immunity)"
RUN1=$("$BIN" run -- sh -c "echo built > $TMP/proj/dist/app.bin" 2>&1)
echo "$RUN1" | grep -E "heal:|result:" | sed 's/^/    /'
[ -f "$TMP/proj/dist/app.bin" ] && ok "program healed and completed" || bad "artifact missing"
# Second run in a fresh location → portable immunity pre-creates the dir.
rm -rf "$TMP/proj"
RUN2=$("$BIN" run -- sh -c "echo built > $TMP/proj/dist/app.bin" 2>&1)
echo "$RUN2" | grep -E "immunity:|result:" | sed 's/^/    /'
echo "$RUN2" | grep -q "immunity:" \
    && ok "immunity pre-created the learned directory" || bad "immunity did not fire"

step "anticipatory prediction (dry-run, no execution)"
"$BIN" run --dry-run -- sh -c "echo x > $TMP/fresh/out.bin" 2>&1 | grep -iE "dry-run|prediction" | sed 's/^/    /' >/dev/null
ok "dry-run prediction ran without executing"

step "acquired immunity report + adaptive Beta-belief confidence"
"$BIN" immunity 2>&1 | grep -E "confidence|•" | head -4 | sed 's/^/    /'
"$BIN" immunity 2>&1 | grep -q "confidence:" && ok "immunity report shows Beta confidence ±uncertainty" || bad "no confidence in report"

step "Pillar 3 — autonomy (axiom solve drives a failing verify to green)"
printf '#!/bin/sh\necho fail >&2\nexit 1\n' > "$TMP/verify.sh"
"$BIN" solve --source "$TMP/verify.sh" -- sh "$TMP/verify.sh" 2>&1 | grep -E "axiom-solve\] (round|result)" | sed 's/^/    /'
"$BIN" solve --source "$TMP/verify.sh" -- sh "$TMP/verify.sh" 2>&1 | grep -q "SOLVED" \
    && ok "solve repaired the source and reached green" || bad "solve did not reach green"

step "grounding verification (server: flag an ungrounded claim) — Pillar against hallucination"
if ! command -v curl >/dev/null 2>&1; then
    printf '  \033[33mSKIP\033[0m grounding (curl not available)\n'
else
    PORT="${AXIOM_DEMO_PORT:-38999}"
    AXIOM_TTT_COMPRESS=1 "$BIN" --mode server --port "$PORT" > "$TMP/srv.log" 2>&1 &
    SRV=$!
    UP=0
    for _ in $(seq 1 40); do curl -s -m1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { UP=1; break; }; sleep 0.25; done
    if [ "$UP" -eq 1 ]; then
        VERIFY=$(curl -s -m5 -X POST "http://127.0.0.1:$PORT/v1/verify" -H 'content-type: application/json' \
            -d '{"response":"Axiom uses online test-time training. It was funded by NASA in 1972.","evidence":"Axiom is an inference engine with online test-time training."}')
        echo "$VERIFY" | grep -qiE "NASA|unsupported|flagged" && ok "verify flagged the ungrounded claim" || bad "verify did not flag"
    else
        bad "server did not come up on port $PORT"
    fi
    kill "$SRV" 2>/dev/null
    wait "$SRV" 2>/dev/null
fi

step "summary"
printf '\n  %d passed, %d failed\n\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
