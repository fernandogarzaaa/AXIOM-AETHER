#!/usr/bin/env python3
"""Import a deterministic sample of Codeflaws defects as bench/repair fixtures.

Codeflaws (Tan et al., 3902 human-written C defects crawled from
Codeforces) is the Phase 4 held-out human-written bug suite: no repair
operator or ordering rule was ever tuned on it, and its programs are C
single-file submissions, a different language and shape from QuixBugs'
Python algorithms.

Selection rule (deterministic, no peeking at repair outcomes): walk
codeflaws-defect-detail-info.txt in file order and take the first N
defects with type WRONG_ANSWER whose buggy submission is at most 150
lines, whose buggy-vs-accepted diff is at most 15 lines, and which ship
at least 2 repair-test inputs including at least 1 failing one
(input-negN). Each imported defect is sanity-checked here: the accepted
program must PASS the fixture verifier and the buggy program must FAIL
it, or the defect is skipped. The accepted program is never shipped in
the fixture (the solver must not copy the answer out of its own tree);
the held-out inputs ship pinned for qualification-time validation, but
the repair oracle runs the repair tests only.

Usage:
    python3 scripts/import_codeflaws.py --tarball <codeflaws.tar.gz> \
        --detail all-script/codeflaws-defect-detail-info.txt \
        --out bench/corpus/unseen-codeflaws --n 12
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

VERIFY_SH = """#!/bin/sh
# Fixture oracle, mirroring the defect's own test-genprog.sh semantics:
# compile with the Codeforces flags from the defect Makefile, run the
# repair-test inputs (input-posN must keep passing, input-negN must
# start passing), normalize program stdout the way the official script
# does (delete blank lines, strip leading whitespace), and diff against
# the expected output with trailing whitespace ignored. The compiled
# binary goes to a temp path OUTSIDE the fixture tree, so it never
# counts as a solver edit in the rollback comparison. Each run is
# bounded at 10 s: a hanging program fails the test, it does not hang
# the runner.
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
    if ! timeout 10 "$BIN" < "$input" | sed -e '/^$/d' -e 's/^[ \\t]*//' | diff --brief --ignore-trailing-space - "$expected" > /dev/null; then
        fail=1
    fi
done
rm -f "$BIN"
exit $fail
"""


def read_detail(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rows.append(
            {
                "id": parts[0],
                "contest": parts[1],
                "type": parts[2],
                "diffinfo": parts[3],
            }
        )
    return rows


def build_index(tar: tarfile.TarFile) -> dict[str, dict[str, tarfile.TarInfo]]:
    """One pass over the tarball: defect id -> {member name -> member}.

    Built once, not per defect: `getmembers()` re-reads the whole gzip
    on every call, so a per-row scan would cost minutes per defect.
    """
    index: dict[str, dict[str, tarfile.TarInfo]] = {}
    for m in tar.getmembers():
        if not m.isfile() or not m.name.startswith("codeflaws/"):
            continue
        rest = m.name[len("codeflaws/"):]
        defect, _, name = rest.partition("/")
        if not name or "/" in name:
            continue
        index.setdefault(defect, {})[name] = m
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tarball", required=True)
    ap.add_argument("--detail", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-lines", type=int, default=150)
    ap.add_argument("--max-diff-lines", type=int, default=15)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_detail(Path(args.detail))
    print(f"{len(rows)} defects in detail file; selecting {args.n}", file=sys.stderr)

    imported: list[dict] = []
    skipped: list[dict] = []
    with tarfile.open(args.tarball, "r:gz") as tar:
        index = build_index(tar)
        for row in rows:
            if len(imported) >= args.n:
                break
            if row["type"] != "WRONG_ANSWER":
                skipped.append({"id": row["id"], "reason": f"type {row['type']}"})
                continue
            members = index.get(row["id"], {})
            c_files = sorted(n for n in members if n.endswith(".c"))
            if len(c_files) != 2:
                skipped.append({"id": row["id"], "reason": f"{len(c_files)} .c files"})
                continue
            # The buggy file carries the buggy submission id, the accepted
            # file the accepted id: <contest>-<problem>-<submission>.c.
            buggy_id = row["id"].split("-bug-")[1].split("-")[0]
            accepted_id = row["id"].split("-")[-1]
            buggy_name = next((n for n in c_files if buggy_id in n), None)
            accepted_name = next((n for n in c_files if accepted_id in n), None)
            if not buggy_name or not accepted_name:
                skipped.append({"id": row["id"], "reason": "could not identify buggy/accepted .c"})
                continue
            inputs = sorted(
                n for n in members if re.fullmatch(r"input-(pos|neg)\d+", n)
            )
            outputs = sorted(
                n for n in members if re.fullmatch(r"output-(pos|neg)\d+", n)
            )
            neg_inputs = [n for n in inputs if n.startswith("input-neg")]
            if len(inputs) < 2 or len(neg_inputs) < 1 or len(outputs) < len(inputs):
                skipped.append(
                    {
                        "id": row["id"],
                        "reason": f"only {len(inputs)} repair inputs ({len(neg_inputs)} failing)",
                    }
                )
                continue
            with tempfile.TemporaryDirectory(prefix="axiom_cf_import_") as td:
                td = Path(td)
                for name, member in members.items():
                    if "/" in name:
                        continue
                    f = tar.extractfile(member)
                    if f is not None:
                        (td / name).write_bytes(f.read())
                buggy = (td / buggy_name).read_text(errors="replace").splitlines()
                accepted = (td / accepted_name).read_text(errors="replace").splitlines()
                if len(buggy) > args.max_lines:
                    skipped.append({"id": row["id"], "reason": f"buggy {len(buggy)} lines"})
                    continue
                diff = list(
                    difflib.unified_diff(
                        buggy, accepted, fromfile=f"a/{buggy_name}", tofile=f"b/{accepted_name}", n=3
                    )
                )
                if len(diff) > args.max_diff_lines + 5:
                    skipped.append({"id": row["id"], "reason": f"diff {len(diff)} lines"})
                    continue
                # Sanity: accepted passes, buggy fails the oracle. Checked
                # in the temp dir, which holds both .c files; the fixture
                # itself never ships the accepted program (the solver must
                # not be able to copy the answer out of its own tree).
                (td / "verify.sh").write_text(VERIFY_SH)
                verify = ["sh", "verify.sh", buggy_name]
                accepted_ok = (
                    subprocess.run(
                        ["sh", "verify.sh", accepted_name],
                        cwd=td,
                        capture_output=True,
                        timeout=120,
                    ).returncode
                    == 0
                )
                buggy_ok = (
                    subprocess.run(
                        verify, cwd=td, capture_output=True, timeout=120
                    ).returncode
                    == 0
                )
                if not accepted_ok or buggy_ok:
                    skipped.append(
                        {
                            "id": row["id"],
                            "reason": f"oracle sanity failed (accepted_ok={accepted_ok}, buggy_ok={buggy_ok})",
                        }
                    )
                    continue
                fixture = out / row["id"]
                if fixture.exists():
                    shutil.rmtree(fixture)
                broken = fixture / "broken"
                broken.mkdir(parents=True)
                for name in members:
                    if "/" in name or name == accepted_name:
                        continue
                    f = tar.extractfile(members[name])
                    if f is not None:
                        (broken / name).write_bytes(f.read())
                (broken / "verify.sh").write_text(VERIFY_SH)
                # Pin everything except the buggy program: the oracle, the
                # test data (repair and held-out), and the build files are
                # not editable. Only the buggy .c may change.
                pinned = sorted(
                    ["verify.sh"]
                    + [n for n in members if n != accepted_name and n != buggy_name]
                )
                meta = {
                    "lang": "c",
                    "verify": verify,
                    "kind": "single",
                    "source": f"codeflaws:{row['id']}",
                    "pinned": pinned,
                }
                (fixture / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
                (fixture / "expected.diff").write_text("".join(diff))
                imported.append(
                    {
                        "id": row["id"],
                        "contest": row["contest"],
                        "buggy_lines": len(buggy),
                        "diff_lines": len(diff),
                        "inputs": len(inputs),
                        "buggy": buggy_name,
                    }
                )
                print(f"  [{len(imported):02d}] {row['id']}: {len(buggy)} lines, diff {len(diff)}", file=sys.stderr)
    summary = {
        "source": "Codeflaws (Tan et al.), https://github.com/codeflaws/codeflaws",
        "tarball": Path(args.tarball).name,
        "imported": len(imported),
        "skipped": len(skipped),
        "fixtures": imported,
        "skip_reasons": skipped,
    }
    (out / "IMPORT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0 if imported else 1


if __name__ == "__main__":
    sys.exit(main())
