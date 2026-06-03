#!/usr/bin/env bash
# promote_d384_on_pass.sh — wait for the d384 run to finish, eval it, and promote
# to production ONLY if it passes acceptance. The verified d256 is backed up first
# and restored-by-default (we never ship an unproven model).
#
# Promotion = copy axiom_d384.bin -> axiom_production_bpe.bin (+ sidecar), then
# bounce the proxy so it loads the new model. The proxy's auto-device guard picks
# the right device (GPU is free once training exits, so the proxy may take CUDA).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/cuda-12.6/bin:$PATH"
export CUDA_PATH="${CUDA_PATH:-$HOME/cuda-12.6}"
CKPTS="$REPO/checkpoints"
D384="$CKPTS/axiom_d384.bin"
PROD="$CKPTS/axiom_production_bpe.bin"
LOG="$REPO/logs/promote_d384.log"
mkdir -p "$REPO/logs"
log() { echo "[promote] $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

log "waiting for train_semantic to finish…"
while tasklist 2>/dev/null | grep -qi train_semantic; do sleep 60; done
log "training finished."

if [ ! -f "$D384" ]; then
    log "ABORT: $D384 was never written (no val improvement) — d256 stays production."
    exit 1
fi

# --- Acceptance eval on the d384 checkpoint --------------------------------
log "evaluating d384…"
RESULT="$(AXIOM_BPE_CKPT="$D384" "$REPO/axiom_engine_rs/target/release/eval_model.exe" 2>>"$LOG" | tail -1)"
log "eval result: $RESULT"

if [ "$RESULT" != "PASS" ]; then
    log "d384 FAILED acceptance — keeping d256 production. No changes made."
    exit 2
fi

# --- Promote: back up d256, swap in d384 -----------------------------------
TS="$(date +%Y%m%d-%H%M%S)"
log "PASS — backing up d256 to axiom_production_bpe.d256.$TS.bak"
cp "$PROD" "$CKPTS/axiom_production_bpe.d256.$TS.bak" 2>/dev/null || true
cp "${PROD%.bin}.meta.json" "$CKPTS/axiom_production_bpe.d256.$TS.meta.bak" 2>/dev/null || true
cp "$D384" "$PROD"
cp "${D384%.bin}.meta.json" "${PROD%.bin}.meta.json"
log "promoted d384 -> production. eval already wrote the recalibrated drift gate."

# --- Bounce the proxy so it loads the new production model ------------------
log "restarting proxy to load promoted model…"
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -match 'start_axiom' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }; Stop-Process -Name axiom_engine -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
sleep 2
( cd "$REPO" && nohup bash start_axiom.sh >/dev/null 2>&1 & )
for i in $(seq 1 20); do
    if curl -s --max-time 2 -o /dev/null http://127.0.0.1:3000/ 2>/dev/null; then log "proxy back up (d384 production) after ${i}s"; break; fi
    sleep 1
done
log "DONE — d384 is live in production."
