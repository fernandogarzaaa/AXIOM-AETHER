#!/usr/bin/env bash
# proof_loop.sh — regenerate AXIOM's three headline metrics and print a
# markdown table. See docs/PROOF-LOOP.md.
#
#   ./scripts/proof_loop.sh            # human run: table to stdout + bench/results/
#   ./scripts/proof_loop.sh --check    # CI: also exit non-zero if the repair gate fails
#
# Metrics:
#   1. Autonomous repair  — `axiom eval-agentic`         (deterministic, no LLM)
#   2. Context compression — `axiom bench <tree>`         (BPE off, and on if a checkpoint is present)
#   3. Grounding / trust  — `axiombench` trust pillar     (calibration set)
#
# CPU-only, no network, no API key. Runs from the repo root.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="$REPO/axiom_engine_rs"
BIN="$CRATE/target/release/axiom"
BENCHBIN="$CRATE/target/release/axiombench"
TREE="$CRATE/src"
OUTDIR="$REPO/bench/results"
DATE="$(date +%Y-%m-%d)"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

export AXIOM_DEVICE=cpu
export CARGO_TERM_COLOR=always

say() { printf '\033[1m%s\033[0m\n' "$*" >&2; }

# --- build --------------------------------------------------------------------
say "building axiom + axiombench (release)…"
( cd "$CRATE" && cargo build --release --locked --bin axiom ) \
  || { echo "build failed: axiom" >&2; exit 1; }
( cd "$CRATE" && cargo build --release --locked --features tools --bin axiombench ) \
  || { echo "build failed: axiombench" >&2; exit 1; }

# --- 1. autonomous repair ---------------------------------------------------- -
say "metric 1/3 — autonomous repair (eval-agentic)…"
REPAIR_OUT="$("$BIN" eval-agentic 2>&1)"
REPAIR_RC=$?
REPAIR="$(printf '%s\n' "$REPAIR_OUT" | grep -oE 'score: [0-9]+/[0-9]+ = [0-9]+%' | head -1)"
[ -n "$REPAIR" ] || REPAIR="parse-failed (rc=$REPAIR_RC)"

# --- 2. context compression ------------------------------------------------- --
say "metric 2/3 — context compression (bench)…"
parse_bench() {
  # extracts: "<savings>% token savings (<a> -> <b>) with <k>/<k> = <p>% ... round-trip"
  printf '%s\n' "$1" \
    | grep -iE 'token savings|round-trip' \
    | tr '\n' ' ' \
    | sed -E 's/.*?([0-9.]+% token savings[^)]*\)).*?([0-9]+\/[0-9]+ = [0-9.]+%[^.]*round-trip).*/\1 with \2/I' \
    | sed -E 's/  +/ /g'
}
COMP_LEGACY_OUT="$(AXIOM_PRODUCTION_BPE=0 "$BIN" bench "$TREE" 2>&1)"
COMP_LEGACY="$(parse_bench "$COMP_LEGACY_OUT")"
[ -n "$COMP_LEGACY" ] || COMP_LEGACY="parse-failed"

CKPT="$CRATE/checkpoints/axiom_production_bpe.bin"
TOK="$CRATE/checkpoints/axiom_bpe.json"
if [ -f "$CKPT" ] && [ -f "$TOK" ]; then
  COMP_BPE_OUT="$(AXIOM_PRODUCTION_BPE=1 AXIOM_BPE_CKPT="$CKPT" AXIOM_TOKENIZER="$TOK" "$BIN" bench "$TREE" 2>&1)"
  COMP_BPE="$(parse_bench "$COMP_BPE_OUT")"
  [ -n "$COMP_BPE" ] || COMP_BPE="parse-failed"
else
  COMP_BPE="skipped (no checkpoint at checkpoints/axiom_production_bpe.bin)"
fi

# --- 3. grounding / trust gate --------------------------------------------- ---
say "metric 3/3 — grounding / trust gate (axiombench)…"
TRUST_OUT="$("$BENCHBIN" 2>&1)"
TRUST="$(printf '%s\n' "$TRUST_OUT" \
  | grep -iE 'coverage|catch.rate|contradiction' \
  | tr '\n' ' ' | sed -E 's/  +/ /g' | sed -E 's/^ +//; s/ +$//')"
[ -n "$TRUST" ] || TRUST="parse-failed"

# --- emit ------------------------------------------------------------------- --
mkdir -p "$OUTDIR"
TABLE="$OUTDIR/proof-loop-$DATE.md"
{
  echo "# Proof loop — $DATE"
  echo
  echo "| Metric | Value | Config |"
  echo "|---|---|---|"
  echo "| Autonomous repair | ${REPAIR} | built-in fixtures, CPU, no LLM |"
  echo "| Compression (legacy tokenizer) | ${COMP_LEGACY} | \`AXIOM_PRODUCTION_BPE=0\`, \`axiom_engine_rs/src\` |"
  echo "| Compression (production BPE) | ${COMP_BPE} | \`AXIOM_PRODUCTION_BPE=1\`, \`axiom_engine_rs/src\` |"
  echo "| Trust gate | ${TRUST} | \`bench/trust/claims.jsonl\` |"
  echo
  echo "_Cost on real traffic is measured separately — see \`bench/cvm/*RESULTS*.md\`._"
} | tee "$TABLE"

echo >&2
say "wrote $TABLE"

if [ "$CHECK" = "1" ]; then
  case "$REPAIR" in
    *"= 100%") : ;;
    *) echo "PROOF LOOP FAIL: autonomous-repair gate is not 100% ($REPAIR)" >&2; exit 1 ;;
  esac
fi
