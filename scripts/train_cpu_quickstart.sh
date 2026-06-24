#!/usr/bin/env bash
# train_cpu_quickstart.sh — end-to-end CPU training of a real semantic checkpoint.
#
# Produces a converged, *discriminative* model on a commodity CPU (no GPU): it
# stages a corpus from the repo's own source + docs, trains a ByteLevel BPE
# tokenizer, trains the d128/2-layer TTT model under early-stopping on held-out
# cross-entropy, and runs the acceptance eval (clean-vs-anomaly drift margin +
# recalibrated drift gate). On PASS the proxy and `axiom run` can use it via
# AXIOM_PRODUCTION_BPE=1.
#
# Validated run (4-core CPU, ~17 min, step cap 4000):
#   val_ce 4.93  |  held-out CE 4.41  |  clean ~4.7  vs  anomaly 9.74
#   drift separation margin +4.93  →  ACCEPTANCE PASS, gate 7.28
#
# This is the CPU counterpart to train_d384.sh (which targets a 6 GB GPU). Tune
# the knobs below for a bigger/slower or smaller/faster model.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
BIN="axiom_engine_rs/target/release"

# --- Tunables (defaults are the validated CPU recipe) ----------------------
export AXIOM_DEVICE="${AXIOM_DEVICE:-cpu}"
export AXIOM_BPE_VOCAB="${AXIOM_BPE_VOCAB:-8000}"
export AXIOM_DMODEL="${AXIOM_DMODEL:-128}"
export AXIOM_NLAYERS="${AXIOM_NLAYERS:-2}"
export AXIOM_MAX_TOKENS="${AXIOM_MAX_TOKENS:-200000}"
export AXIOM_EPOCHS="${AXIOM_EPOCHS:-8}"
export AXIOM_STEP_CAP="${AXIOM_STEP_CAP:-4000}"
export AXIOM_PATIENCE="${AXIOM_PATIENCE:-3}"
export AXIOM_TRAIN_WIN="${AXIOM_TRAIN_WIN:-96}"
export AXIOM_LR="${AXIOM_LR:-3e-3}"
export AXIOM_WARMUP_STEPS="${AXIOM_WARMUP_STEPS:-100}"
export AXIOM_LOG_EVERY="${AXIOM_LOG_EVERY:-200}"

echo "[quickstart] building trainer binaries..."
# The trainer/eval binaries are gated behind the `tools` feature (kept out of
# default builds + the pip/crates packages), so enable it here.
cargo build --release --manifest-path axiom_engine_rs/Cargo.toml --features tools --bins >/dev/null

# --- 1. Stage a corpus from the repo's own source + docs -------------------
echo "[quickstart] staging corpus -> checkpoints/corpus/"
mkdir -p checkpoints/corpus
python3 - <<'PY'
import os, glob
roots = ['axiom_engine_rs/src', 'axiom_engine', 'docs', 'scripts']
files = []
for r in roots:
    for ext in ('*.rs', '*.py', '*.md'):
        files += glob.glob(os.path.join(r, '**', ext), recursive=True)
files = [f for f in files if 'target/' not in f]
blob = '\n\n'.join(open(f, encoding='utf-8', errors='ignore').read() for f in sorted(files))
n = 4
step = len(blob) // n + 1
for i in range(n):
    open(f'checkpoints/corpus/shard_{i:02d}.txt', 'w').write(blob[i*step:(i+1)*step])
print(f"  corpus: {len(files)} files, {len(blob)} chars, {n} shards")
PY

# --- 2. Tokenizer ----------------------------------------------------------
echo "[quickstart] training BPE tokenizer (vocab=$AXIOM_BPE_VOCAB)..."
"$BIN/train_tokenizer" >/dev/null

# --- 3. Semantic model (early-stopping on held-out CE) ---------------------
echo "[quickstart] training semantic model d${AXIOM_DMODEL}/${AXIOM_NLAYERS}L (this is the long step)..."
"$BIN/train_semantic"

# --- 4. Acceptance eval: drift separation + recalibrated gate --------------
echo "[quickstart] running acceptance eval..."
if "$BIN/eval_model"; then
    echo
    echo "[quickstart] PASS — the model separates clean code from anomalies."
    echo "[quickstart] Use it everywhere with:"
    echo "    export AXIOM_PRODUCTION_BPE=1"
    echo "    export AXIOM_TOKENIZER=$REPO/checkpoints/axiom_bpe.json"
    echo "    export AXIOM_BPE_CKPT=$REPO/checkpoints/axiom_production_bpe.bin"
else
    echo "[quickstart] eval did not PASS — inspect the margins above before promoting."
    exit 1
fi
