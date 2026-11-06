#!/usr/bin/env python3
"""Import the Python half of QuixBugs into `bench/repair/` fixture directories.

QuixBugs (Koppel, Lin et al.; MIT licence) is 40 small classic algorithms, each
with exactly one developer-style defect and a pytest oracle driven by JSON test
vectors or hand-written graph tests. Nobody on this project authored those bugs,
which is what makes it a held-out corpus for the deterministic repair loop.

The importer takes a checkout of https://github.com/jkoppel/QuixBugs and writes
one fixture per program:

    <out>/quixbugs-<program>/
      meta.json        verify = python3 -m pytest -q -p no:cacheprovider python_testcases/test_<program>.py
                       pinned = every file except python_programs/<program>.py
                       source = "quixbugs@<commit> python/<program>"
      broken/
        conftest.py                     (QuixBugs' own, verbatim)
        python_programs/<program>.py    (the buggy program)
        python_programs/node.py         (shared graph helper, when the program needs it)
        python_testcases/test_<program>.py   (the `--correct` import branch removed, see below)
        python_testcases/load_testdata.py, node.py
        json_testcases/<program>.json   (when the test uses vectors)

Two deliberate differences from a raw copy:

* The correct programs are NOT copied, and the test's `if pytest.use_correct:
  from correct_python_programs...` branch is removed. Otherwise the answer key
  would sit inside the tree the solver is allowed to touch.
* No `expected.diff`: QuixBugs' buggy files carry the docstring the correct files
  lack, so a file-level diff is noise. `meta.json` records the upstream commit;
  compare against `correct_python_programs/` there.

Usage:
    scripts/import_quixbugs.py --src /path/to/QuixBugs --out bench/repair-quixbugs
    AXIOM_REPAIR_CORPUS=bench/repair-quixbugs scripts/bench_repair.sh
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HELPERS = ("load_testdata.py", "node.py")


def upstream_commit(src: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - best effort provenance
        return "unknown"


def strip_correct_branch(test_src: str, program: str) -> str:
    """Replace the `if pytest.use_correct: ... else: ...` import with the plain import."""
    pattern = re.compile(
        r"if pytest\.use_correct:\n\s+from correct_python_programs\.(\w+) import (.+)\nelse:\n\s+(from python_programs\.\w+ import .+)\n",
    )
    new, n = pattern.subn(lambda m: m.group(3) + "\n", test_src)
    if n != 1:
        raise ValueError(f"{program}: expected exactly one use_correct import block, found {n}")
    return new


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="QuixBugs checkout")
    ap.add_argument("--out", required=True, help="corpus directory to create")
    ap.add_argument("--only", nargs="*", default=None, help="program names to import (default: all)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    programs_dir = src / "python_programs"
    tests_dir = src / "python_testcases"
    json_dir = src / "json_testcases"
    if not (programs_dir.is_dir() and tests_dir.is_dir()):
        print(f"{src} does not look like a QuixBugs checkout", file=sys.stderr)
        return 2
    commit = upstream_commit(src)
    out.mkdir(parents=True, exist_ok=True)

    programs = sorted(
        p.stem
        for p in programs_dir.glob("*.py")
        if not p.stem.endswith("_test") and p.stem != "node" and (tests_dir / f"test_{p.stem}.py").exists()
    )
    if args.only:
        programs = [p for p in programs if p in set(args.only)]

    verify = ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "PLACEHOLDER"]
    written = 0
    for prog in programs:
        fixture = out / f"quixbugs-{prog}"
        if fixture.exists():
            shutil.rmtree(fixture)
        broken = fixture / "broken"
        (broken / "python_programs").mkdir(parents=True)
        (broken / "python_testcases").mkdir(parents=True)
        shutil.copy2(src / "conftest.py", broken / "conftest.py")
        shutil.copy2(programs_dir / f"{prog}.py", broken / "python_programs" / f"{prog}.py")
        test_src = (tests_dir / f"test_{prog}.py").read_text()
        test_rel = f"python_testcases/test_{prog}.py"
        (broken / test_rel).write_text(strip_correct_branch(test_src, prog))
        pinned = ["conftest.py", test_rel]
        for helper in HELPERS:
            if (tests_dir / helper).exists():
                shutil.copy2(tests_dir / helper, broken / "python_testcases" / helper)
                pinned.append(f"python_testcases/{helper}")
        if "node" in test_src or "from node import" in (programs_dir / f"{prog}.py").read_text():
            shutil.copy2(programs_dir / "node.py", broken / "python_programs" / "node.py")
            pinned.append("python_programs/node.py")
        jf = json_dir / f"{prog}.json"
        if jf.exists():
            (broken / "json_testcases").mkdir()
            shutil.copy2(jf, broken / "json_testcases" / jf.name)
            pinned.append(f"json_testcases/{jf.name}")
        meta = {
            "lang": "python",
            "verify": verify[:-1] + [test_rel],
            "kind": "single",
            "source": f"quixbugs@{commit} python/{prog} (MIT; https://github.com/jkoppel/QuixBugs)",
            "pinned": pinned,
        }
        (fixture / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        written += 1
        print(f"  {fixture.name}", file=sys.stderr)
    (out / "README.md").write_text(
        "# QuixBugs (Python) as a `bench/repair` corpus\n\n"
        f"Generated by `scripts/import_quixbugs.py` from QuixBugs commit `{commit}`\n"
        "(MIT licence, https://github.com/jkoppel/QuixBugs). 40 classic algorithms, one\n"
        "developer-style defect each, pytest oracles. The correct programs are not\n"
        "included and the tests' `--correct` branch is removed, so the answer key is\n"
        "never inside the tree the solver may edit. Every file except the buggy\n"
        "program is pinned: a green test obtained by editing the oracle scores OVERFIT.\n\n"
        "Run: `AXIOM_REPAIR_CORPUS=<this dir> scripts/bench_repair.sh`\n"
    )
    print(f"wrote {written} fixture(s) to {out}", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
