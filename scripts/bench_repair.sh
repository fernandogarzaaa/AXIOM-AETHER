#!/usr/bin/env bash
# bench_repair.sh — score the on-disk autonomous-repair corpus in bench/repair/
# (format and intent: docs/BENCH-REPAIR.md, "Fixture format for an on-disk corpus").
#
#   scripts/bench_repair.sh            # table to stdout + bench/results/repair-<date>.md
#   scripts/bench_repair.sh --check    # CI: non-zero if any fixture is not actually broken,
#                                      #     is not repaired, or a failed attempt left the
#                                      #     tree modified (AXIOM_EVAL_REQUIRE_ALL=1 also
#                                      #     fails on a skipped fixture)
#
# Per fixture: (1) No-op baseline — `verify` must FAIL on the untouched tree, or a
# pass means nothing; (2) `axiom solve` in a scratch copy with no file named, so
# the loop must localize and repair on its own; (3) re-run `verify`; (4) on
# failure, rollback correctness — the tree must be byte-identical to broken/.
# When expected.diff exists, a pass is labelled "authored fix" or "different fix".
#
# Env: AXIOM_BIN (default: axiom_engine_rs/target/release/axiom, built if absent),
#      AXIOM_REPAIR_CORPUS (default: bench/repair). CPU-only, no network, no API key.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="$REPO/axiom_engine_rs"
BIN="${AXIOM_BIN:-$CRATE/target/release/axiom}"
CORPUS="${AXIOM_REPAIR_CORPUS:-$REPO/bench/repair}"
OUTDIR="$REPO/bench/results"
DATE="$(date +%Y-%m-%d)"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1
export AXIOM_DEVICE=cpu

say() { printf '\033[1m%s\033[0m\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }
# Verifier by-products that are not part of the repaired tree.
DIFF_EXCLUDES=(-x __pycache__ -x '*.pyc' -x node_modules -x target -x '*.o' -x 't' -x 'calc_test')

if [ ! -x "$BIN" ]; then
  say "building axiom (release)…"
  ( cd "$CRATE" && cargo build --release --locked --bin axiom ) || { echo "build failed" >&2; exit 1; }
fi

pass=0; fail=0; skip=0; notbroken=0; dirty=0
rows=()
for dir in "$CORPUS"/*/; do
  dir="${dir%/}"
  name="$(basename "$dir")"
  meta="$dir/meta.json"
  [ -f "$meta" ] && [ -d "$dir/broken" ] || continue
  mapfile -t VERIFY < <(python3 -c 'import json,sys; [print(a) for a in json.load(open(sys.argv[1]))["verify"]]' "$meta")
  lang="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("lang","?"))' "$meta")"
  if [ "${#VERIFY[@]}" -eq 0 ]; then
    rows+=("| $name | $lang | SKIP (no verify command) | — | — |"); skip=$((skip+1)); continue
  fi
  if ! have "${VERIFY[0]}"; then
    rows+=("| $name | $lang | SKIP (missing \`${VERIFY[0]}\`) | — | — |"); skip=$((skip+1)); continue
  fi
  say "fixture $name ($lang): ${VERIFY[*]}"
  work="$(mktemp -d "${TMPDIR:-/tmp}/axiom_bench_repair_${name}_XXXXXX")"
  cp -R "$dir/broken/." "$work/"

  # 1. No-op baseline.
  if ( cd "$work" && "${VERIFY[@]}" >/dev/null 2>&1 ); then
    rows+=("| $name | $lang | NOT BROKEN | verify passes on the untouched tree | — |")
    notbroken=$((notbroken+1)); rm -rf "$work"; continue
  fi

  # 2. Autonomous solve, no file named.
  start="$(date +%s.%N)"
  log="$( cd "$work" && "$BIN" solve --max-rounds 2 -- "${VERIFY[@]}" 2>&1 )"
  end="$(date +%s.%N)"
  secs="$(python3 -c "print(f'{$end-$start:.1f}')")"
  how="$(printf '%s\n' "$log" | grep -oE 'mutation repair verified green \([^)]*\)|source repair solved it \(patched=[a-z]+\)|environment heal solved it|fleet patch [0-9a-f]+ verified green locally|LLM-proposed patch verified green' | head -1)"

  # 3. Re-verify.
  if ( cd "$work" && "${VERIFY[@]}" >/dev/null 2>&1 ); then
    authored="—"
    if [ -f "$dir/expected.diff" ] && have patch; then
      ref="$(mktemp -d)"; cp -R "$dir/broken/." "$ref/"
      if ( cd "$ref" && patch -s -p1 < "$dir/expected.diff" >/dev/null 2>&1 ); then
        if diff -rq "${DIFF_EXCLUDES[@]}" "$ref" "$work" >/dev/null 2>&1; then authored="authored fix"; else authored="different fix"; fi
      fi
      rm -rf "$ref"
    fi
    rows+=("| $name | $lang | PASS | ${how:-?} | ${secs}s, $authored |"); pass=$((pass+1))
  else
    # 4. Rollback correctness.
    if diff -rq "${DIFF_EXCLUDES[@]}" "$dir/broken" "$work" >/dev/null 2>&1; then rb="tree intact"; else rb="TREE MODIFIED"; dirty=$((dirty+1)); fi
    rows+=("| $name | $lang | FAIL | ${how:-no repair found} | ${secs}s, $rb |"); fail=$((fail+1))
  fi
  rm -rf "$work"
done

mkdir -p "$OUTDIR"
TABLE="$OUTDIR/repair-$DATE.md"
{
  echo "# Repair corpus — $DATE"
  echo
  echo "Corpus: \`${CORPUS#"$REPO"/}\` · runner: \`scripts/bench_repair.sh\` · deterministic, no LLM."
  echo
  echo "| Fixture | Lang | Result | How | Notes |"
  echo "|---|---|---|---|---|"
  printf '%s\n' ${rows[@]+"${rows[@]}"}
  echo
  echo "**pass: $pass · fail: $fail · skipped: $skip · not broken: $notbroken · rollback violations: $dirty**"
} | tee "$TABLE"
say "wrote $TABLE"

if [ "$CHECK" = 1 ]; then
  rc=0
  [ "$fail" -eq 0 ]      || { echo "BENCH REPAIR FAIL: $fail fixture(s) not repaired" >&2; rc=1; }
  [ "$notbroken" -eq 0 ] || { echo "BENCH REPAIR FAIL: $notbroken fixture(s) pass verify untouched" >&2; rc=1; }
  [ "$dirty" -eq 0 ]     || { echo "BENCH REPAIR FAIL: $dirty failed attempt(s) left the tree modified" >&2; rc=1; }
  if [ "${AXIOM_EVAL_REQUIRE_ALL:-0}" = 1 ] && [ "$skip" -gt 0 ]; then
    echo "BENCH REPAIR FAIL: $skip fixture(s) skipped and AXIOM_EVAL_REQUIRE_ALL=1" >&2; rc=1
  fi
  exit $rc
fi
