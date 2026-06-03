#!/usr/bin/env bash
# train_d384.sh — launch the d384+stabilize convergence run on GPU.
#
# Trains to a SEPARATE checkpoint (axiom_d384.bin) so the verified-good d256
# production model (axiom_production_bpe.bin) is never clobbered. Promotion to
# production happens only after eval_model PASSes on the new checkpoint.
#
# GPU is reserved for this run; the proxy stays on CPU (co-tenancy guard), so
# the two never contend for the 6 GB card.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/cuda-12.6/bin:$PATH"
export CUDA_PATH="${CUDA_PATH:-$HOME/cuda-12.6}"

# Architecture: d384 / 6 layers, with inner-loop stabilization (required so deep
# models stay finite). Train to a dedicated checkpoint + sidecar.
export AXIOM_DMODEL=384
export AXIOM_NLAYERS=6
export AXIOM_TTT_STABILIZE=1
export AXIOM_BPE_CKPT="$REPO/checkpoints/axiom_d384.bin"

# Convergence controls. Early-stop on held-out CE keeps it from overfitting.
export AXIOM_EPOCHS="${AXIOM_EPOCHS:-12}"
export AXIOM_PATIENCE="${AXIOM_PATIENCE:-3}"
export AXIOM_LOG_EVERY="${AXIOM_LOG_EVERY:-50}"   # step heartbeat
export AXIOM_GRAD_CLIP="${AXIOM_GRAD_CLIP:-1.0}"
export AXIOM_WARMUP_STEPS="${AXIOM_WARMUP_STEPS:-100}"

BIN="$REPO/axiom_engine_rs/target/release/train_semantic.exe"
LOG="$REPO/logs/train_d384.log"
mkdir -p "$REPO/logs"

echo "[train_d384] starting d384/6L stabilize run -> $AXIOM_BPE_CKPT" | tee "$LOG"
echo "[train_d384] $(date)" | tee -a "$LOG"
"$BIN" 2>&1 | tee -a "$LOG"
echo "[train_d384] exited code=${PIPESTATUS[0]} at $(date)" | tee -a "$LOG"
