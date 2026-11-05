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
# pass means nothing; the tree is snapshotted AFTER this run so verifier
# by-products (__pycache__, .pytest_cache, build output…) are never mistaken
# for solver edits; (2) `axiom solve` in a scratch copy with no file named, so
# the loop must localize and repair on its own; (3) re-run `verify`, then check
# the PINNED files (meta.json "pinned", default: every test-looking file) are
# byte-identical to the fixture — a green verifier obtained by editing the
# oracle is scored OVERFIT, not PASS; (4) on failure, rollback correctness —
# the tree must be byte-identical to the post-baseline snapshot. When
# expected.diff exists, a pass is labelled "authored fix" or "different fix".
#
# Env: AXIOM_BIN (default: axiom_engine_rs/target/release/axiom, built if absent),
#      AXIOM_REPAIR_CORPUS (default: bench/repair), AXIOM_BENCH_KEEP=1 (keep each
#      scratch tree and print its path, for post-hoc inspection), AXIOM_BENCH_LABEL
#      (suffix for the results file). CPU-only, no network, no API key.
#
# meta.json may also carry "test_region": {"file": ..., "from_line": N} for
# fixtures whose tests live in the same file as the code (Rust #[cfg(test)]):
# any edit at or below that line is scored OVERFIT, like a pinned file.
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

# Files the solver must never have changed for a pass to count: meta.json
# "pinned" (paths relative to broken/), else every test-looking file in the
# fixture (test_*, *_test.*, *.test.*, *_spec.*, test.<ext>, and anything
# under test/, tests/, spec/, __tests__/).
pinned_files() { # $1 = meta.json, $2 = broken dir
  python3 - "$1" "$2" <<'PY'
import json, os, re, sys
meta, root = sys.argv[1], sys.argv[2]
pinned = json.load(open(meta)).get("pinned")
if pinned is None:
    pinned = []
    pat = re.compile(r'^(test_.*|.*_test\..*|.*\.test\..*|.*_spec\..*|.*\.spec\..*|spec_.*|test\.[A-Za-z0-9]+)$')
    for d, _, files in os.walk(root):
        rel_d = os.path.relpath(d, root)
        parts = [] if rel_d == "." else rel_d.split(os.sep)
        for f in files:
            if pat.match(f) or any(p in ("test", "tests", "spec", "specs", "__tests__") for p in parts):
                pinned.append(os.path.normpath(os.path.join(rel_d, f)))
for p in sorted(set(pinned)):
    print(p)
PY
}

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
  # Snapshot AFTER the baseline run: whatever the verifier itself writes
  # (caches, build output) is now part of the reference, so a later diff
  # attributes only the solver's edits to the solver.
  snap="$(mktemp -d "${TMPDIR:-/tmp}/axiom_bench_repair_${name}_snap_XXXXXX")"
  cp -R "$work/." "$snap/"
  mapfile -t PINNED < <(pinned_files "$meta" "$dir/broken")
  region_file="$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])).get("test_region") or {}; print(m.get("file",""))' "$meta")"
  region_from="$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])).get("test_region") or {}; print(m.get("from_line",0))' "$meta")"

  # 2. Autonomous solve, no file named. Solver state (heal memory and the
  #    verified-patch store derived from it, ~/.axiom) is isolated per fixture:
  #    fixtures in one corpus often share a verify command, and a patch learned
  #    on one would otherwise be replayed on the next and score as a repair.
  state="$(mktemp -d "${TMPDIR:-/tmp}/axiom_bench_repair_${name}_state_XXXXXX")"
  start="$(date +%s.%N)"
  log="$( cd "$work" && HOME="$state" AXIOM_HEAL_MEMORY="$state/heal_memory.json" "$BIN" solve --max-rounds 2 -- "${VERIFY[@]}" 2>&1 )"
  end="$(date +%s.%N)"
  secs="$(python3 -c "print(f'{$end-$start:.1f}')")"
  how="$(printf '%s\n' "$log" | grep -oE 'mutation repair verified green \([^)]*\)|source repair solved it \(patched=[a-z]+\)|environment heal solved it|fleet patch [0-9a-f]+ verified green locally|LLM-proposed patch verified green' | head -1)"

  # 3. Re-verify, then refuse a pass that edited the oracle.
  if ( cd "$work" && "${VERIFY[@]}" >/dev/null 2>&1 ); then
    touched=()
    for p in ${PINNED[@]+"${PINNED[@]}"}; do
      cmp -s "$dir/broken/$p" "$work/$p" || touched+=("$p")
    done
    if [ -n "$region_file" ] && [ "$region_from" -gt 0 ] && [ -f "$work/$region_file" ]; then
      if ! cmp -s <(tail -n "+$region_from" "$dir/broken/$region_file") <(tail -n "+$region_from" "$work/$region_file"); then
        touched+=("$region_file:$region_from+ (test region)")
      fi
    fi
    if [ "${#touched[@]}" -gt 0 ]; then
      rows+=("| $name | $lang | OVERFIT | ${how:-?} | ${secs}s, pinned file(s) modified: ${touched[*]} |"); fail=$((fail+1))
    else
      authored="—"
      if [ -f "$dir/expected.diff" ] && have patch; then
        ref="$(mktemp -d)"; cp -R "$dir/broken/." "$ref/"
        if ( cd "$ref" && patch -s -p1 < "$dir/expected.diff" >/dev/null 2>&1 ); then
          # Compare only the fixture's own files: anything the verifier
          # generated (present in the post-baseline snapshot) is ignored.
          same=1
          while IFS= read -r -d '' f; do
            rel="${f#"$dir/broken/"}"
            cmp -s "$ref/$rel" "$work/$rel" || { same=0; break; }
          done < <(find "$dir/broken" -type f -print0)
          if [ "$same" = 1 ]; then authored="authored fix"; else authored="different fix"; fi
        fi
        rm -rf "$ref"
      fi
      rows+=("| $name | $lang | PASS | ${how:-?} | ${secs}s, $authored, pinned ${#PINNED[@]} file(s) intact |"); pass=$((pass+1))
    fi
  else
    # 4. Rollback correctness against the post-baseline snapshot.
    if diff -rq "$snap" "$work" >/dev/null 2>&1; then rb="tree intact"; else rb="TREE MODIFIED: $(diff -rq "$snap" "$work" 2>/dev/null | head -3 | tr '\n' ' ')"; dirty=$((dirty+1)); fi
    rows+=("| $name | $lang | FAIL | ${how:-no repair found} | ${secs}s, $rb |"); fail=$((fail+1))
  fi
  if [ "${AXIOM_BENCH_KEEP:-0}" = 1 ]; then say "kept $work"; else rm -rf "$work"; fi
  rm -rf "$snap" "$state"
done

mkdir -p "$OUTDIR"
TABLE="$OUTDIR/repair${AXIOM_BENCH_LABEL:+-$AXIOM_BENCH_LABEL}-$DATE.md"
{
  echo "# Repair corpus — $DATE"
  echo
  echo "Corpus: \`${CORPUS#"$REPO"/}\` · runner: \`scripts/bench_repair.sh\` · deterministic, no LLM · candidate cap: \`AXIOM_REPAIR_MAX_CANDIDATES=${AXIOM_REPAIR_MAX_CANDIDATES:-64}\`."
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
  [ "$fail" -eq 0 ]      || { echo "BENCH REPAIR FAIL: $fail fixture(s) not repaired (or repaired by editing a pinned file)" >&2; rc=1; }
  [ "$notbroken" -eq 0 ] || { echo "BENCH REPAIR FAIL: $notbroken fixture(s) pass verify untouched" >&2; rc=1; }
  [ "$dirty" -eq 0 ]     || { echo "BENCH REPAIR FAIL: $dirty failed attempt(s) left the tree modified" >&2; rc=1; }
  if [ "${AXIOM_EVAL_REQUIRE_ALL:-0}" = 1 ] && [ "$skip" -gt 0 ]; then
    echo "BENCH REPAIR FAIL: $skip fixture(s) skipped and AXIOM_EVAL_REQUIRE_ALL=1" >&2; rc=1
  fi
  exit $rc
fi
