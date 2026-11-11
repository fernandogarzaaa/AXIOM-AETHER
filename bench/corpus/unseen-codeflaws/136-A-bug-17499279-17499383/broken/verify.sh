#!/bin/sh
# Fixture oracle, mirroring the defect's own test-genprog.sh semantics:
# compile with the Codeforces flags from the defect Makefile, run the
# repair-test inputs (input-posN must keep passing, input-negN must
# start passing), normalize program stdout the way the official script
# does (delete blank lines, strip leading whitespace), and diff against
# the expected output with trailing whitespace ignored. The compiled
# binary goes to a temp path OUTSIDE the fixture tree, so it never
# counts as a solver edit in the rollback comparison. Each run is
# bounded at 10 s: a hanging program fails the test, it does not hang
# the runner. The program's exit status is checked SEPARATELY from its
# stdout (Codex P2, PR #294): a patch that prints the expected output
# and then aborts must fail. Output is captured to a temp file first
# and `$?` read before any pipe, so a crash cannot hide behind a
# matching diff (a pipeline's status is the last command's, and `diff`
# cannot see that the program died).
set -u
PROG="$1"
[ -n "$PROG" ] || { echo "verify.sh needs the program file"; exit 2; }
BIN="${TMPDIR:-/tmp}/cf_verify_prog_$$"
gcc -fno-optimize-sibling-calls -fno-strict-aliasing -fno-asm -std=c99 -O2 -lm -o "$BIN" "$PROG" || exit 1
fail=0
for input in input-pos[0-9]* input-neg[0-9]*; do
    [ -f "$input" ] || continue
    case "$input" in
        input-pos*) n="${input#input-pos}"; expected="output-pos$n" ;;
        input-neg*) n="${input#input-neg}"; expected="output-neg$n" ;;
    esac
    [ -f "$expected" ] || { echo "missing $expected"; exit 2; }
    out_file="${TMPDIR:-/tmp}/cf_verify_out_$$"
    timeout 10 "$BIN" < "$input" > "$out_file"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        fail=1
    elif ! sed -e '/^$/d' -e 's/^[ \t]*//' "$out_file" | diff --brief --ignore-trailing-space - "$expected" > /dev/null; then
        fail=1
    fi
    rm -f "$out_file"
done
rm -f "$BIN"
exit $fail
