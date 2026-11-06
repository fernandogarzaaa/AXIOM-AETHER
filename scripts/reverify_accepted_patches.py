#!/usr/bin/env python3
"""Re-verify repairs the benchmark accepted against a *stronger* test suite.

A seeded-fault repair "passes" when the crate's own tests accept it. Some
accepted edits are not the seeded reversal; this script tells the equivalent
ones from the ones the original tests were simply too weak to reject.

Inputs:
  --corpus DIR   seeded corpus (scripts/seed_faults.py output)
  --emit DIR     holds labels.tsv ("<fixture>\\t<candidate label>") and the
                 candidate sources written by the planner diagnostic:
                   AXIOM_PLAN_DIAG_EMIT=DIR cargo test --test mutation_plan_diag \\
                     -- --ignored --nocapture
                 (<fixture>.<k>.src, one per candidate carrying that label)
  --crate DIR    the pristine crate whose test region is the stronger oracle

For each fixture, the candidate that passes the fixture's own (original) tests
is the one the benchmark accepted; its code is then spliced in front of the
pristine crate's current `#[cfg(test)]` region and the tests are re-run.
Prints one line per accepted candidate and writes DIR/reverify.json.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile


def cut(lines):
    return next(i for i, line in enumerate(lines) if line.strip() == "#[cfg(test)]")


def run_tests(crate, rel, src, target_dir):
    with tempfile.TemporaryDirectory(prefix="axiom_reverify_") as td:
        work = os.path.join(td, "crate")
        # Fresh mtimes (shutil.copy), or cargo may reuse a stale test binary.
        shutil.copytree(
            crate,
            work,
            ignore=shutil.ignore_patterns("target", "proptest-regressions"),
            copy_function=shutil.copy,
        )
        with open(os.path.join(work, rel), "w") as f:
            f.write(src)
        r = subprocess.run(
            ["cargo", "test", "--lib", "-q"],
            cwd=work,
            capture_output=True,
            text=True,
            env={**os.environ, "CARGO_TARGET_DIR": target_dir, "RUST_BACKTRACE": "0"},
        )
    failed = [
        line.split()[0]
        for line in r.stdout.splitlines()
        if "---" in line and line.strip().endswith("FAILED")
    ]
    return r.returncode == 0, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--emit", required=True)
    ap.add_argument("--crate", required=True)
    ap.add_argument("--target-dir", default="/tmp/axiom_reverify_target")
    a = ap.parse_args()

    out = []
    for line in open(os.path.join(a.emit, "labels.tsv")):
        if not line.strip():
            continue
        name, label = line.rstrip("\n").split("\t")
        rel = json.load(open(os.path.join(a.corpus, name, "meta.json")))["fault"]["file"]
        stronger = open(os.path.join(a.crate, rel)).read().splitlines(True)
        for cand in sorted(glob.glob(os.path.join(a.emit, f"{name}.*.src"))):
            patched = open(cand).read()
            accepted, _ = run_tests(a.crate, rel, patched, a.target_dir)
            if not accepted:
                continue  # same label, different span: not the edit the benchmark kept
            lines = patched.splitlines(True)
            ok, failed = run_tests(
                a.crate, rel, "".join(lines[: cut(lines)] + stronger[cut(stronger) :]), a.target_dir
            )
            row = dict(fixture=name, label=label, candidate=os.path.basename(cand),
                       stronger_suite="PASS" if ok else "FAIL", killed_by=failed)
            out.append(row)
            print(f"{name} | {label} | {row['stronger_suite']} {failed}", flush=True)
    json.dump(out, open(os.path.join(a.emit, "reverify.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
