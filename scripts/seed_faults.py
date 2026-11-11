#!/usr/bin/env python3
"""Seed random single-token faults into maintainer-written Rust code and emit
them as `bench/repair/`-format fixtures (docs/BENCH-REPAIR.md).

Why this exists. The built-in `eval-agentic` fixtures and the seed corpus were
authored with the repair engine in mind, so they cannot say how the loop fares
on bugs nobody designed for it. This script produces such bugs the way the APR
literature does when a developer-bug corpus is not at hand: random single-token
faults ("mutants") in real, pre-existing code, kept only when the project's own
test suite catches them. The engine never sees which token was changed; it gets
the failing test output, exactly as in production.

Honesty notes, which travel with every result this produces:

* These are **seeded faults, not developer-committed bugs**. The fault model
  below deliberately includes classes the deterministic repair layer has no
  operator for (identifier swaps, min/max swaps, shift-direction flips,
  break/continue) alongside ones it does, so the rate is not tautological and
  the per-class breakdown shows where the engine stops.
* Tests live in the same files as the code (Rust `#[cfg(test)]`), so pinning
  whole files is impossible; each fixture records the line where the test module
  starts and the runner scores any edit at or below it as OVERFIT.
* A fault the suite does not catch ("survived") is excluded: it is not a bug the
  oracle can see, so repairing it is undefined.

Usage:
    scripts/seed_faults.py --crate axiom_mesh_core --out /tmp/seeded --n 40 --seed 2026
    AXIOM_REPAIR_CORPUS=/tmp/seeded scripts/bench_repair.sh

Phase 4 unseen suite (2026-11-10): seed 424242 is the qualification seed.
It uses a different generator revision than seeds 7/2026 (the `strlit` class
below did not exist when those corpora were generated) and a different seed,
so no operator or ordering rule was ever tuned against its faults. Its
fixtures live under bench/corpus/unseen-seed424242/ and are excluded from
all tuning until the qualification run.

Strengthened suite (2026-11-11, batch 2, seed 424243): three more fault
classes (`deref`, `cast`, `ret`) plus `--classes` (restrict sampling to
named classes) and `--exclude-dir` (never re-emit a fault already present
in a corpus directory). Batch 2 targets the thin classes of batch 1
(`logic`, `argswap`, `range`, `minmax` had 1-2 faults each) and the new
classes. In/out of the repair operator set, verified against
`MutationClass::ALL` in axiom_engine_rs/src/mutation_repair.rs (12
classes: Comparison, OffByOne, Boolean, Logic, Arithmetic, SwapArgs,
MissingReturn, ShellTest, ApiSwap, ArgShift, IdentSwap, ImportPath):
`ret` is INSIDE the set (MissingReturn re-adds a dropped `return`);
`deref` (drop `&`/`*`) and `cast` (change an `as` target type) are
OUTSIDE it, by construction. A `deref`/`cast` fault may fail to compile
rather than fail a test; the oracle is `cargo test --lib`, which fails
either way, and the corpus records the split per fixture in SEED.json.

Regenerated batch (2026-11-11, batch 3, seed 424244): Codex review of
the 36-fixture suite found 18 fixtures share a (file, line, class)
triple with the prior repair-seeded-mesh-core tuning tables, so the
suite was not genuinely unseen. Batch 3 replaces those 18 at fresh
sites, using `--exclude-triples` (never emit a site whose
(file, line, class) appears in a triples file: the tuning tables plus
the 18 retained fixtures), so the regenerated suite has zero triple
overlap with any tuning table. One fixture per triple per run is
enforced inside the generator as well.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------------
# Fault model
# ----------------------------------------------------------------------------

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


@dataclass
class Site:
    line_no: int  # 0-based
    cls: str
    before: str
    after: str
    start: int
    end: int

    def apply(self, line: str) -> str:
        return line[: self.start] + self.after + line[self.end :]


def code_mask(line: str) -> list[bool]:
    """False inside string/char literals and after `//`."""
    mask = [True] * len(line)
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            for k in range(i, n):
                mask[k] = False
            break
        if c == '"':
            j = i + 1
            while j < n:
                if line[j] == "\\":
                    j += 2
                    continue
                if line[j] == '"':
                    break
                j += 1
            for k in range(i, min(j, n - 1) + 1):
                mask[k] = False
            i = j + 1
            continue
        if c == "'" and i + 2 < n and line[i + 2] == "'":
            for k in range(i, i + 3):
                mask[k] = False
            i += 3
            continue
        i += 1
    return mask


def spaced(line: str, s: int, e: int) -> bool:
    return s > 0 and line[s - 1] == " " and e < len(line) and line[e] == " "


def string_spans(line: str) -> list[tuple[int, int, str]]:
    """(content_start, content_end, content) for genuine string literals.

    Unlike `code_mask`, which blanks string contents along with comments,
    this returns only real strings: `//` comments end the scan, and char
    literals are skipped.
    """
    out: list[tuple[int, int, str]] = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            break
        if c == '"':
            j = i + 1
            content: list[str] = []
            while j < n:
                if line[j] == "\\":
                    if j + 1 < n:
                        content.append(line[j + 1])
                    j += 2
                    continue
                if line[j] == '"':
                    break
                content.append(line[j])
                j += 1
            if j < n and j > i + 1:
                out.append((i + 1, j, "".join(content)))
            i = j + 1
            continue
        if c == "'" and i + 2 < n and line[i + 2] == "'":
            i += 3
            continue
        i += 1
    return out


def sites_for_line(line_no: int, line: str) -> list[Site]:
    """Every single-token fault we could inject into this line."""
    out: list[Site] = []
    mask = code_mask(line)
    stripped = line.strip()
    if (
        not stripped
        or stripped.startswith(("//", "#[", "use ", "mod ", "pub mod", "impl", "struct", "enum", "trait", "fn ", "pub fn", "///", "//!"))
        or "assert" in line
        or "panic!" in line
        or "unreachable!" in line
        or "debug_" in line
    ):
        return out

    def ok(s: int, e: int) -> bool:
        return all(mask[s:e])

    # Comparison / logic / arithmetic / range / shift operators.
    op_table = {
        "<=": ("cmp", ["<", ">="]),
        ">=": ("cmp", [">", "<="]),
        "==": ("cmp", ["!="]),
        "!=": ("cmp", ["=="]),
        "<": ("cmp", ["<=", ">"]),
        ">": ("cmp", [">=", "<"]),
        "&&": ("logic", ["||"]),
        "||": ("logic", ["&&"]),
        "+=": ("arith", ["-=", "="]),
        "-=": ("arith", ["+="]),
        "*=": ("arith", ["/="]),
        "+": ("arith", ["-", "*"]),
        "-": ("arith", ["+"]),
        "*": ("arith", ["/", "+"]),
        "/": ("arith", ["*"]),
        "..=": ("range", [".."]),
        "..": ("range", ["..="]),
        "<<": ("shift", [">>"]),
        ">>": ("shift", ["<<"]),
    }
    # Longest-first operator scan.
    ops_sorted = sorted(op_table, key=len, reverse=True)
    i = 0
    while i < len(line):
        if not mask[i]:
            i += 1
            continue
        hit = None
        for op in ops_sorted:
            if line.startswith(op, i) and ok(i, i + len(op)):
                hit = op
                break
        if hit is None:
            i += 1
            continue
        cls, alts = op_table[hit]
        e = i + len(hit)
        # Binary-position + spacing requirement avoids `->`, generics, derefs, `=>`.
        if hit in ("..", "..="):
            applicable = True
        else:
            applicable = spaced(line, i, e)
        if applicable:
            for a in alts:
                out.append(Site(line_no, cls, hit, a, i, e))
        i = e

    # Integer literals.
    for m in re.finditer(r"(?<![A-Za-z0-9_.])(\d+)(?![A-Za-z0-9_.])", line):
        s, e = m.span(1)
        if not ok(s, e):
            continue
        v = int(m.group(1))
        if v > 10_000:
            continue
        alts = {str(v + 1)}
        if v >= 1:
            alts.add(str(v - 1))
        if 1 <= v <= 64:
            alts.add(str(v * 2))
        for a in sorted(alts):
            out.append(Site(line_no, "const", m.group(1), a, s, e))

    # Booleans.
    for m in re.finditer(r"\b(true|false)\b", line):
        s, e = m.span(1)
        if ok(s, e):
            out.append(Site(line_no, "bool", m.group(1), "false" if m.group(1) == "true" else "true", s, e))

    # Negation: drop a `!` before an identifier/paren (not a macro), or negate an `if` condition.
    for m in re.finditer(r"(?<![A-Za-z0-9_!=])!(?=[A-Za-z_(])", line):
        s, e = m.span()
        if ok(s, e):
            out.append(Site(line_no, "neg", "!", "", s, e))
    m = re.match(r"^(\s*(?:\}\s*else\s+)?if\s+)(?!let\b)(.+?)(\s*\{\s*)$", line)
    if m and ok(m.start(2), m.end(2)) and "!(" not in m.group(2):
        out.append(Site(line_no, "neg", m.group(2), f"!({m.group(2)})", m.start(2), m.end(2)))

    # Off-by-one on lengths.
    for m in re.finditer(r"\.len\(\)", line):
        s, e = m.span()
        if ok(s, e) and not line[e:].lstrip().startswith(("-", "+")):
            out.append(Site(line_no, "offbyone", ".len()", ".len() - 1", s, e))
            out.append(Site(line_no, "offbyone", ".len()", ".len() + 1", s, e))

    # min/max swap (outside the repair engine's operator set).
    for m in re.finditer(r"\.(min|max)\(", line):
        s, e = m.span(1)
        if ok(s, e):
            out.append(Site(line_no, "minmax", m.group(1), "max" if m.group(1) == "min" else "min", s, e))
    for m in re.finditer(r"\b(floor|ceil)\(", line):
        s, e = m.span(1)
        if ok(s, e):
            out.append(Site(line_no, "minmax", m.group(1), "ceil" if m.group(1) == "floor" else "floor", s, e))

    # break/continue (outside the set).
    for m in re.finditer(r"\b(break|continue)\b", line):
        s, e = m.span(1)
        if ok(s, e):
            out.append(Site(line_no, "flow", m.group(1), "continue" if m.group(1) == "break" else "break", s, e))

    # String-literal mutation (added 2026-11-10 for the Phase 4 unseen suite,
    # seed 424242: this class did not exist when seeds 7/2026 were generated).
    # Appends "!" to a short plain string literal. The repair engine has no
    # operator that can guess the original literal, so this class is outside
    # its operator set by construction; a test only catches it when it asserts
    # the exact string. Skips format strings, paths, and URLs.
    for (start, end, content) in string_spans(line):
        if len(content) < 2 or len(content) > 32:
            continue
        if "{" in content or "}" in content or "/" in content or ":" in content:
            continue
        if not re.fullmatch(r"[A-Za-z0-9 _.,-]+", content):
            continue
        out.append(Site(line_no, "strlit", content, content + "!", start, end))

    # Deref drop (added 2026-11-11 for the strengthened suite): drop a
    # leading `&` borrow or `*` deref. Outside the repair engine's operator
    # set by construction (no operator adds `&`/`*`). Expression positions
    # only: signature lines are already skipped above, and these guards
    # exclude the lookalikes. `&` immediately after `<` is a generic type
    # argument (`Option<&T>`); a `&` before the `=`/` in ` of a `let`/`if
    # let`/`while let`/`for` header is a pattern binding, not a borrow;
    # an odd count of `|` before the site means it sits inside closure
    # params (`|a, &b|`); `-> &T` is a return type; `&mut` and `&&` are
    # not plain borrows. `*` needs an expression operator before it so
    # multiplication (`a * b`) and raw-pointer types (`*const T`, whose
    # previous non-space char is `:` or `s`) are excluded.
    stripped = line.strip()
    cutoff = 0
    if stripped.startswith(("let ", "if let ", "while let ")):
        eq = line.find("=")
        cutoff = eq + 1 if eq >= 0 else len(line)
    elif stripped.startswith("for "):
        inn = line.find(" in ")
        cutoff = inn + 4 if inn >= 0 else len(line)
    for m in re.finditer(r"&", line):
        s = m.start()
        if s < cutoff:
            continue
        if s + 1 >= len(line) or not (line[s + 1].isalpha() or line[s + 1] == "_" or line[s + 1] == "("):
            continue
        if line[s + 1 : s + 4] == "mut" or (s + 1 < len(line) and line[s + 1] == "&"):
            continue
        if s > 0 and line[s - 1] == "<":
            continue
        if not ok(s, s + 1):
            continue
        prev = line[:s].rstrip()
        if not prev or prev[-1] not in "(,=+-*/%<>!&|[":
            continue
        if line[:s].count("|") % 2 == 1:
            continue
        if re.search(r"->\s*&", line[: s + 1]):
            continue
        out.append(Site(line_no, "deref", "&", "", s, s + 1))
    for m in re.finditer(r"\*", line):
        s = m.start()
        if s < cutoff:
            continue
        if s + 1 >= len(line) or not (line[s + 1].isalpha() or line[s + 1] == "_" or line[s + 1] == "("):
            continue
        if not ok(s, s + 1):
            continue
        prev = line[:s].rstrip()
        if not prev or prev[-1] not in "=([!-+":
            continue
        out.append(Site(line_no, "deref", "*", "", s, s + 1))

    # Cast-target change (added 2026-11-11 for the strengthened suite):
    # `x as u64` -> `x as u32`, etc. Outside the operator set by
    # construction (no operator rewrites an `as` target type). Measured
    # on this crate (batch 2): all three emitted cast faults fail to
    # compile, because the new target type mismatches the operand type
    # (E0308/E0277, e.g. `f32` -> `f64` next to an `f32` operand). A cast
    # whose operand types still line up would be a behaviour change
    # (truncation, or a float-precision change the tests may or may not
    # catch). Either way the oracle kills it and the fixture records
    # which (see SEED.json).
    for m in re.finditer(r"\bas\s+(u8|u16|u32|u64|usize|i8|i16|i32|i64|isize|f32|f64)\b", line):
        s, e = m.span(1)
        if ok(s, e):
            out.append(Site(line_no, "cast", m.group(1), CAST_ALT[m.group(1)], s, e))

    # Dropped early return (added 2026-11-11 for the strengthened suite):
    # `return Err(...);` -> `Err(...);`. Measured on this crate (batch 2):
    # all three emitted `ret` faults fail to compile (E0282: type
    # annotations needed), because dropping `return` removes the type
    # ascription the `Err(...)`/`None` expression relied on; as a
    # discarded statement it has no expected type. INSIDE the operator
    # set: MissingReturn re-adds a dropped `return`.
    m = re.match(r"^(\s*)(return)(\s+)(.+);\s*$", line)
    if m and ok(m.start(2), m.end(3)):
        out.append(Site(line_no, "ret", "return", "", m.start(2), m.end(3)))

    # Identifier swap: two distinct short identifiers on one line, swapped (outside the set).
    idents = [m for m in re.finditer(rf"\b({IDENT})\b", line) if ok(*m.span(1))]
    names = []
    for m in idents:
        nm = m.group(1)
        if nm in names or nm in RUST_KEYWORDS or len(nm) > 12 or nm[0].isupper() or "::" in line[max(0, m.start() - 2) : m.start()]:
            continue
        names.append(nm)
    if len(names) >= 2:
        a, b = names[0], names[1]
        swapped = re.sub(rf"\b({a}|{b})\b", lambda mm: b if mm.group(1) == a else a, line)
        if swapped != line:
            out.append(Site(line_no, "idswap", f"{a}<->{b}", swapped, 0, len(line)))

    # Argument swap in a two-argument call (inside the set).
    for m in re.finditer(rf"\b{IDENT}\(([^(),]+),\s*([^(),]+)\)", line):
        s, e = m.span()
        if ok(s, e):
            new = line[: m.start(1)] + m.group(2) + ", " + m.group(1) + line[m.end(2) :]
            out.append(Site(line_no, "argswap", f"{m.group(1).strip()},{m.group(2).strip()}", new, 0, len(line)))
    return out


RUST_KEYWORDS = set(
    "as break const continue crate else enum extern false fn for if impl in let loop match mod move mut pub ref return "
    "self Self static struct super trait true type unsafe use where while async await dyn u8 u16 u32 u64 usize i8 i16 "
    "i32 i64 isize f32 f64 bool char str Some None Ok Err Vec String Box Option Result".split()
)

# Cast-target alternates for the `cast` fault class: each type maps to the
# nearest type whose values differ (width change, or float precision change).
CAST_ALT = {
    "u8": "u16",
    "u16": "u8",
    "u32": "u64",
    "u64": "u32",
    "usize": "u32",
    "i8": "i16",
    "i16": "i8",
    "i32": "i64",
    "i64": "i32",
    "isize": "i32",
    "f32": "f64",
    "f64": "f32",
}


def test_region_start(lines: list[str]) -> int:
    for i, l in enumerate(lines):
        if l.strip() == "#[cfg(test)]":
            return i
    return len(lines)


# ----------------------------------------------------------------------------
# Fixture emission
# ----------------------------------------------------------------------------


def run_verify(cwd: Path, verify: list[str]) -> bool:
    try:
        r = subprocess.run(verify, cwd=cwd, capture_output=True, timeout=600)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def copy_crate(src: Path, dst: Path) -> None:
    # `shutil.copy`, not the default `copy2`: the copy must get fresh mtimes.
    # With the original (older) mtimes, cargo sees sources older than the test
    # binary a previous fixture left in the shared CARGO_TARGET_DIR and reruns
    # that stale binary instead of rebuilding.
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("target", ".git", "*.orig", "proptest-regressions"),
        copy_function=shutil.copy,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crate", default="axiom_mesh_core", help="crate directory under the repo root")
    ap.add_argument("--out", required=True, help="corpus directory to create (bench/repair format)")
    ap.add_argument("--n", type=int, default=40, help="number of KILLED faults to emit")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-attempts", type=int, default=400)
    ap.add_argument("--files", nargs="*", default=None, help="restrict to these src files (relative to the crate)")
    ap.add_argument("--verify", default=None, help="override verify command (shell string)")
    ap.add_argument(
        "--balance",
        action="store_true",
        help="round-robin over fault classes so one prolific class (identifier swaps) cannot dominate the sample",
    )
    ap.add_argument(
        "--classes",
        default=None,
        help="comma-separated fault classes to sample from (default: all). "
        "Used to top up thin classes without re-rolling the whole suite.",
    )
    ap.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="corpus directory whose existing fixtures must never be re-emitted: "
        "a candidate site whose (file, line, class, before, after) matches an "
        "existing fixture's meta.json fault is skipped. Repeatable.",
    )
    ap.add_argument(
        "--exclude-triples",
        action="append",
        default=[],
        help="file listing (file, line, class) triples, one per line as "
        "FILE:LINE:CLASS (e.g. src/mesh.rs:489:flow), whose sites must never "
        "be emitted: a candidate sharing a triple with the file is skipped. "
        "Used to keep a regenerated suite disjoint from prior tuning tables. "
        "Repeatable. Triples emitted in this run are excluded the same way, "
        "so a suite never holds two fixtures at one (file, line, class).",
    )
    args = ap.parse_args()

    wanted = None
    if args.classes:
        wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
        known = {
            "argswap", "arith", "bool", "flow", "idswap", "minmax", "neg",
            "offbyone", "range", "shift", "const", "logic", "cmp", "strlit",
            "deref", "cast", "ret",
        }
        unknown = wanted - known
        if unknown:
            print(f"unknown fault class(es): {sorted(unknown)}", file=sys.stderr)
            return 2

    crate = (REPO / args.crate).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Debug profile on purpose: it builds incrementally, so each of the up to
    # 64 repair candidates costs ~1-2 s to verify instead of a full release
    # recompile (~10 s+), which makes a 40-fixture run ~1 h instead of ~8 h.
    # Portable target dir (Codex P1, PR #294): resolve the shared target
    # directory from the runtime environment at VERIFY time, never embed
    # the generation machine's absolute path in a committed fixture.
    # `${TMPDIR:-$HOME/.cache}` honours a caller-chosen scratch dir when
    # one is set, else the user's cache dir (on a real disk: /tmp is a
    # small tmpfs in this sandbox, so a bare /tmp default would strand
    # the build there). $HOME is safe to rely on: cargo/rustup need it.
    verify_sh = args.verify or 'CARGO_TARGET_DIR="${TMPDIR:-$HOME/.cache}/axiom_seeded_target" cargo test --lib -q'
    verify = ["sh", "-c", verify_sh]

    # Confirm the pristine crate is green with this verifier.
    with tempfile.TemporaryDirectory(prefix="axiom_seed_pristine_") as td:
        pristine = Path(td) / "crate"
        copy_crate(crate, pristine)
        if not run_verify(pristine, verify):
            print("pristine crate does not pass its own verifier; refusing to seed", file=sys.stderr)
            return 2

    files = args.files or sorted(
        str(p.relative_to(crate)) for p in (crate / "src").glob("*.rs") if p.name not in ("lib.rs", "node.rs")
    )
    excluded: set[tuple[str, int, str, str, str]] = set()
    for d in args.exclude_dir:
        for meta_path in Path(d).glob("*/meta.json"):
            try:
                fault = json.loads(meta_path.read_text()).get("fault") or {}
            except (json.JSONDecodeError, OSError):
                continue
            if fault.get("file") and fault.get("line") and fault.get("class") is not None:
                excluded.add(
                    (fault["file"], int(fault["line"]), fault["class"], fault.get("before", ""), fault.get("after", ""))
                )
    if excluded:
        print(f"excluding {len(excluded)} already-emitted fault(s) from {args.exclude_dir}", file=sys.stderr)
    excluded_triples: set[tuple[str, int, str]] = set()
    for triples_path in args.exclude_triples:
        for raw in Path(triples_path).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                f, ln, cls = line.rsplit(":", 2)
            except ValueError:
                print(f"bad triples line (want FILE:LINE:CLASS): {line!r}", file=sys.stderr)
                return 2
            excluded_triples.add((f, int(ln), cls))
    if excluded_triples:
        print(f"excluding {len(excluded_triples)} (file, line, class) triple(s) from {args.exclude_triples}", file=sys.stderr)
    all_sites: list[tuple[str, list[str], Site]] = []
    for rel in files:
        lines = (crate / rel).read_text().splitlines(keepends=True)
        region = test_region_start(lines)
        for i in range(region):
            content = lines[i].rstrip("\n")
            for s in sites_for_line(i, content):
                if wanted is not None and s.cls not in wanted:
                    continue
                all_sites.append((rel, lines, s))
    rng = random.Random(args.seed)
    rng.shuffle(all_sites)
    if args.balance:
        # Interleave classes: take one site from each class in turn, so the
        # emitted sample is as class-balanced as the kill filter allows.
        buckets: dict[str, list] = {}
        for item in all_sites:
            buckets.setdefault(item[2].cls, []).append(item)
        order = sorted(buckets)
        interleaved = []
        while any(buckets.values()):
            for c in order:
                if buckets[c]:
                    interleaved.append(buckets[c].pop())
        all_sites = interleaved
    print(f"{len(all_sites)} candidate fault sites across {len(files)} file(s); sampling with seed {args.seed}", file=sys.stderr)

    emitted = 0
    attempts = 0
    seen: set[str] = set()
    survived = 0
    by_class: dict[str, int] = {}
    for rel, lines, site in all_sites:
        if emitted >= args.n or attempts >= args.max_attempts:
            break
        content = lines[site.line_no].rstrip("\n")
        new_line = site.apply(content)
        if new_line == content:
            continue
        key = f"{rel}:{site.line_no}:{new_line}"
        if key in seen:
            continue
        seen.add(key)
        if (rel, site.line_no + 1, site.cls, site.before, site.after) in excluded:
            continue
        triple = (rel, site.line_no + 1, site.cls)
        if triple in excluded_triples:
            continue
        attempts += 1
        mutated = list(lines)
        mutated[site.line_no] = new_line + "\n"
        name = f"{Path(rel).stem}-L{site.line_no + 1}-{site.cls}-{emitted + 1:02d}"
        fixture = out / name
        if fixture.exists():
            shutil.rmtree(fixture)
        broken = fixture / "broken"
        copy_crate(crate, broken)
        (broken / rel).write_text("".join(mutated))
        # Kill filter: the suite must catch it.
        if run_verify(broken, verify):
            survived += 1
            shutil.rmtree(fixture)
            continue
        region = test_region_start(lines)
        meta = {
            "lang": "rust",
            "verify": verify,
            "kind": "single",
            "source": f"seeded fault, seed={args.seed}, {rel}:{site.line_no + 1}, class={site.cls}",
            "pinned": [],
            "test_region": {"file": rel, "from_line": region + 1},
            "fault": {"file": rel, "line": site.line_no + 1, "class": site.cls, "before": site.before, "after": site.after},
        }
        (fixture / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        diff = difflib.unified_diff(
            mutated, lines, fromfile=f"a/{rel}", tofile=f"b/{rel}", n=3
        )
        (fixture / "expected.diff").write_text("".join(diff))
        emitted += 1
        excluded_triples.add(triple)
        by_class[site.cls] = by_class.get(site.cls, 0) + 1
        print(f"  [{emitted:02d}] {name}: {site.before!r} -> {site.after!r}", file=sys.stderr)

    summary = {
        "crate": args.crate,
        "seed": args.seed,
        "emitted": emitted,
        "attempted": attempts,
        "survived_excluded": survived,
        "by_class": by_class,
        "files": files,
    }
    (out / "SEED.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0 if emitted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
