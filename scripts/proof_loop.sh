#!/usr/bin/env bash
# proof_loop.sh — regenerate AXIOM's three headline metrics and print a
# markdown table. See docs/PROOF-LOOP.md.
#
#   ./scripts/proof_loop.sh            # human run: table to stdout + bench/results/
#   ./scripts/proof_loop.sh --check    # CI: also exit non-zero if the repair
#                                    gate or the bench --strict round-trip
#                                    gate fails
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
  # pull the two figures out of the bench output independently, then join them
  local blob savings roundtrip
  blob="$(printf '%s\n' "$1" | tr '\n' ' ' | sed -E 's/  +/ /g')"
  # Current `axiom bench` wording:
  #   [bench] token savings : 84.1% (245070 -> 39004 tokens)
  #   [bench] signatures kept : 2978 (expand round-trip 2978/2978 = 100.0%)
  savings="$(printf '%s' "$blob" \
    | grep -oiE 'token savings ?: ?[0-9]+(\.[0-9]+)?% \([0-9]+ ?-> ?[0-9]+ tokens\)' | head -1 \
    | sed -E 's/^[Tt]oken savings ?: ?([0-9.]+%) \(([0-9]+) ?-> ?([0-9]+) tokens\)$/\1 token savings (\2 -> \3)/')"
  roundtrip="$(printf '%s' "$blob" \
    | grep -oiE 'round-trip [0-9]+/[0-9]+ = [0-9]+(\.[0-9]+)?%' | head -1 \
    | sed -E 's/^[Rr]ound-trip ([0-9]+\/[0-9]+ = [0-9.]+%)$/\1 round-trip/')"
  # Older wording ("84.1% token savings (245070 -> 39004)", "N/N = 100% round-trip").
  [ -n "$savings" ]   || savings="$(printf '%s' "$blob"   | grep -oiE '[0-9]+(\.[0-9]+)?% token savings( \([0-9,]+ ?-> ?[0-9,]+\))?' | head -1)"
  [ -n "$roundtrip" ] || roundtrip="$(printf '%s' "$blob" | grep -oiE '[0-9]+/[0-9]+ = [0-9]+(\.[0-9]+)?% (signature )?round-trip' | head -1)"
  if [ -n "$savings" ] && [ -n "$roundtrip" ]; then
    printf '%s with %s' "$savings" "$roundtrip"
  fi
}
COMP_LEGACY_OUT="$(AXIOM_PRODUCTION_BPE=0 "$BIN" bench --strict "$TREE" 2>&1)"
LEGACY_RC=$?
COMP_LEGACY="$(parse_bench "$COMP_LEGACY_OUT")"
[ -n "$COMP_LEGACY" ] || COMP_LEGACY="parse-failed"

CKPT="$CRATE/checkpoints/axiom_production_bpe.bin"
TOK="$CRATE/checkpoints/axiom_bpe.json"
if [ -f "$CKPT" ] && [ -f "$TOK" ]; then
  COMP_BPE_OUT="$(AXIOM_PRODUCTION_BPE=1 AXIOM_BPE_CKPT="$CKPT" AXIOM_TOKENIZER="$TOK" "$BIN" bench --strict "$TREE" 2>&1)"
  BPE_RC=$?
  COMP_BPE="$(parse_bench "$COMP_BPE_OUT")"
  [ -n "$COMP_BPE" ] || COMP_BPE="parse-failed"
else
  COMP_BPE="skipped (no checkpoint at checkpoints/axiom_production_bpe.bin)"
  BPE_RC=0
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
  [ "$LEGACY_RC" -eq 0 ] \
    || { echo "PROOF LOOP FAIL: legacy-tokenizer bench round-trip is not 100% (bench --strict exited $LEGACY_RC)" >&2; exit 1; }
  [ "$BPE_RC" -eq 0 ] \
    || { echo "PROOF LOOP FAIL: production-BPE bench round-trip is not 100% (bench --strict exited $BPE_RC)" >&2; exit 1; }
fi
