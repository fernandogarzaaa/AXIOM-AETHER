#!/usr/bin/env python3
"""Validate the Codeflaws repair-test passes against the held-out inputs.

The repair oracle (verify.sh) runs only the repair-test inputs
(input-posN / input-negN). Each fixture also ships held-out inputs
(heldout-input-posN / heldout-output-posN) that the repair oracle never
runs. A repair-test PASS can therefore be a patch that overfits the
repair tests (e.g. 136-A: the solver's MXS=300 passes the repair tests
while the reference fix is MXS=1024). This script checks, for every
fixture the harness passed, whether the solver's accepted patch also
passes ALL held-out inputs, using the same compile flags and stdout
normalization as verify.sh.

It also runs the same held-out oracle on the reference program (the
fixture's expected.diff applied to the buggy program) as a sanity check:
if the reference fails a held-out input, that input is inconsistent with
the true fix and the fixture is flagged REFERENCE-INCONSISTENT rather
than blaming the solver.

Usage:
    AXIOM_BENCH_KEEP=1 AXIOM_BENCH_LABEL=unseen-codeflaws-rerun \
        AXIOM_REPAIR_CORPUS=bench/corpus/unseen-codeflaws \
        scripts/bench_repair.sh 2> /tmp/bench-keep.log
    python3 scripts/validate_heldout_codeflaws.py \
        --table bench/results/repair-unseen-codeflaws-rerun-2026-11-11.md \
        --keep-log /tmp/bench-keep.log \
        --out bench/results/heldout-codeflaws-2026-11-11.md

The keep-log is the harness run's stderr: bench_repair.sh prints
`kept <work dir>` for each fixture when AXIOM_BENCH_KEEP=1, and the
kept work dir holds the solver's accepted tree for that fixture.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GCC = [
    "gcc", "-fno-optimize-sibling-calls", "-fno-strict-aliasing", "-fno-asm",
    "-std=c99", "-O2", "-lm",
]


def run_heldout(prog: Path, data: Path, work: Path) -> tuple[int, int, list[str]]:
    """Compile prog and run every heldout-input-* in data.

    Returns (passed, total, failed_names). Same normalization as
    verify.sh: delete blank lines, strip leading whitespace, diff with
    trailing whitespace ignored. Each run bounded at 10 s.
    """
    binary = work / "prog"
    r = subprocess.run(GCC + ["-o", str(binary), str(prog)],
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        return 0, 0, ["COMPILE-FAILED"]
    passed, total, failed = 0, 0, []
    for inp in sorted(data.glob("heldout-input-*")):
        n = inp.name[len("heldout-input-"):]
        expected = data / f"heldout-output-{n}"
        if not expected.is_file():
            continue
        total += 1
        try:
            with open(inp, "rb") as fh:
                out = subprocess.run([str(binary)], stdin=fh, capture_output=True, timeout=10)
            got = out.stdout.decode(errors="replace").splitlines()
            got = [re.sub(r"^[ \t]*", "", line) for line in got if line != ""]
            want = expected.read_text(errors="replace").splitlines()
            want = [line.rstrip() for line in want]
            got = [line.rstrip() for line in got]
            # Exit status is judged separately from stdout (Codex P2,
            # PR #294): a program that prints the expected output and
            # then aborts must fail, not validate.
            if out.returncode != 0:
                failed.append(inp.name + f" (exit {out.returncode})")
            elif got == want:
                passed += 1
            else:
                failed.append(inp.name)
        except subprocess.TimeoutExpired:
            failed.append(inp.name + " (timeout)")
    return passed, total, failed


def parse_table(path: Path) -> list[str]:
    """Fixture names with a PASS row in a bench_repair.sh results table."""
    passes = []
    for line in path.read_text().splitlines():
        m = re.match(r"\| ([^ |]+) \| [^|]+ \| PASS \|", line)
        if m:
            passes.append(m.group(1))
    return passes


def parse_keep_log(path: Path) -> dict[str, Path]:
    """Fixture name -> kept work dir, from the harness run's stderr.

    bench_repair.sh's `say` wraps lines in ANSI bold escapes, so strip
    those before matching.
    """
    kept: dict[str, Path] = {}
    for raw in path.read_text().splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()
        m = re.match(r"kept (\S+)$", line)
        if not m:
            continue
        d = Path(m.group(1))
        base = d.name
        assert base.startswith("axiom_bench_repair_"), base
        rest = base[len("axiom_bench_repair_"):]
        name = rest.rsplit("_", 1)[0]
        kept[name] = d
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="bench/corpus/unseen-codeflaws")
    ap.add_argument("--table", required=True, help="bench_repair.sh results table to read PASS rows from")
    ap.add_argument("--keep-log", required=True, help="harness stderr log with 'kept <dir>' lines")
    ap.add_argument("--out", required=True, help="results markdown file to write")
    args = ap.parse_args()

    corpus = REPO / args.corpus
    table = Path(args.table)
    passes = parse_table(table)
    kept = parse_keep_log(Path(args.keep_log))
    if not passes:
        print("no PASS rows in table; refusing to report an empty validation", file=sys.stderr)
        return 2

    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    rows = []
    heldout_pass = 0
    for name in passes:
        fixture = corpus / name
        meta = json.loads((fixture / "meta.json").read_text())
        prog_name = meta["verify"][2]
        if name not in kept or not (kept[name] / prog_name).is_file():
            print(f"{name}: no kept work dir; cannot validate the accepted patch", file=sys.stderr)
            return 2
        with tempfile.TemporaryDirectory(prefix="axiom_cf_heldout_", dir=str(tmpdir)) as td:
            td = Path(td)
            repaired = kept[name] / prog_name
            # The solver's accepted edit, for the report.
            buggy = (fixture / "broken" / prog_name).read_text().splitlines()
            fixed = repaired.read_text().splitlines()
            edit = "\n".join(difflib.unified_diff(buggy, fixed, lineterm="")) or "(no diff?)"
            p, t, failed = run_heldout(repaired, fixture / "broken", td)
            # Reference program: expected.diff applied to the buggy program.
            refdir = td / "ref"
            shutil.copytree(fixture / "broken", refdir)
            r = subprocess.run(["patch", "-s", "-p1", "-i", str(fixture / "expected.diff")],
                               cwd=refdir, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{name}: expected.diff does not apply; cannot build the reference", file=sys.stderr)
                return 2
            rp, rt, rfailed = run_heldout(refdir / prog_name, fixture / "broken", td)
        verdict = "HELDOUT-PASS" if p == t and t > 0 else "HELDOUT-FAIL"
        if rp != rt:
            verdict = "REFERENCE-INCONSISTENT"
        if verdict == "HELDOUT-PASS":
            heldout_pass += 1
        rows.append((name, p, t, failed, rp, rt, rfailed, verdict, edit))

    out = Path(args.out)
    lines = [
        "# Held-out validation of Codeflaws repair-test passes",
        "",
        f"Source table: `{table.name}` ({len(passes)} repair-test passes). "
        f"Held-out inputs: `heldout-input-*` / `heldout-output-*` shipped pinned "
        "in each fixture, never run by the repair oracle. Same compile flags and "
        "stdout normalization as `verify.sh`; each run bounded at 10 s.",
        "",
        f"**Held-out validated: {heldout_pass} of {len(passes)} repair-test passes "
        f"also pass every held-out input.**",
        "",
        "| Fixture | Held-out (solver patch) | Held-out (reference fix) | Verdict |",
        "|---|---|---|---|",
    ]
    for name, p, t, failed, rp, rt, rfailed, verdict, _ in rows:
        fail_note = f" (failed: {', '.join(failed)})" if failed else ""
        ref_note = f" (failed: {', '.join(rfailed)})" if rfailed else ""
        lines.append(f"| {name} | {p}/{t}{fail_note} | {rp}/{rt}{ref_note} | {verdict} |")
    lines += ["", "## Solver accepted edits (repaired .c vs buggy .c)", ""]
    for name, p, t, failed, rp, rt, rfailed, verdict, edit in rows:
        lines += [f"### {name} ({verdict})", "", "```diff", edit, "```", ""]
    out.write_text("\n".join(lines))
    print(f"held-out validated {heldout_pass}/{len(passes)}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
