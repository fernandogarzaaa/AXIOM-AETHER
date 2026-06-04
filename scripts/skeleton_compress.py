#!/usr/bin/env python3
"""Prototype: Claude-READABLE compression via structural skeleton extraction.

Idea: instead of shipping Claude an opaque neural fingerprint (vocab indices +
norms, which a different model can't read), ship a compact *skeleton* of the
heavy content — doc comments, imports, and declaration signatures with bodies
elided. This is small (target <=20% of original) AND meaningful: Claude can
actually answer questions from real signatures.

Axiom's neural fingerprint still gets computed and kept INTERNALLY (session
recall / drift detection) — it just no longer wastes upstream tokens.

This prototype measures token savings and prints the skeleton so its readability
is self-evident. Language: Rust/Python/JS-ish heuristics (extensible).
"""
import re
import sys

# Lines that declare structure we keep (signature line only).
DECL = re.compile(
    r"""^\s*(
        pub\s+|public\s+|export\s+|default\s+
    )*(
        (async\s+)?fn\s|def\s|function\s|
        struct\s|enum\s|trait\s|impl\b|interface\s|class\s|type\s|
        const\s|static\s|let\s+[A-Z]|mod\s|namespace\s
    )""",
    re.VERBOSE,
)
DOC = re.compile(r"^\s*(//[/!]|#\s|\"\"\"|\*\s|/\*\*)")  # doc comments
IMPORT = re.compile(r"^\s*(use\s|import\s|from\s|#include|require\()")


def approx_tokens(s: str) -> int:
    return max(1, round(len(s) / 4))


def skeletonize(text: str, max_doc_lines: int = 3) -> str:
    lines = text.splitlines()
    out = []
    doc_budget = max_doc_lines
    elided = 0
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        if IMPORT.match(ln):
            out.append(ln.rstrip())
        elif DECL.match(ln):
            # keep the signature; cut at the opening brace so bodies vanish
            sig = ln.split("{")[0].rstrip()
            out.append(sig + (" { … }" if "{" in ln else ""))
        elif DOC.match(ln) and doc_budget > 0:
            out.append(ln.rstrip())
            doc_budget -= 1
        else:
            elided += 1
    if elided:
        out.append(f"// … {elided} implementation lines elided …")
    return "\n".join(out)


def build_readable_fingerprint(text: str, session_id: str = "demo") -> str:
    """The compact, Claude-readable block that replaces the heavy text."""
    skel = skeletonize(text)
    return (
        f'<axiom_context_digest session="{session_id}" '
        f'kind="structural-skeleton" original_tokens="{approx_tokens(text)}">\n'
        f"# Compact skeleton of elided heavy context (signatures kept, bodies dropped).\n"
        f"# Ask Axiom to expand a symbol if you need a specific body.\n"
        f"{skel}\n"
        f"</axiom_context_digest>"
    )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\garza\AXIOM-AETHER\axiom_engine_rs\src\server.rs"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()[:12000]  # same 12 KB slice as the proxy test

    digest = build_readable_fingerprint(text)
    o, d = approx_tokens(text), approx_tokens(digest)
    saved = (o - d) / o * 100

    print("=" * 60)
    print("CLAUDE-READABLE COMPRESSION — structural skeleton")
    print("=" * 60)
    print(f"Original heavy text : {len(text):>7} chars  (~{o:>5} tokens)")
    print(f"Readable digest     : {len(digest):>7} chars  (~{d:>5} tokens)")
    print(f"SAVED               : {saved:>6.1f}%   (target: 80%)")
    print("=" * 60)
    print("\n----- WHAT CLAUDE WOULD ACTUALLY SEE (first 1800 chars) -----\n")
    print(digest[:1800])
    if len(digest) > 1800:
        print(f"\n…[+{len(digest)-1800} more chars of signatures]")


if __name__ == "__main__":
    main()
