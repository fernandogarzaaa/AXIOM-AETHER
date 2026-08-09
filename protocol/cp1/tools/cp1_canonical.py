"""The single Python definition of CP/1 canonical form and sealing.

`build_fixtures.py` writes the golden corpus with these functions and
`verify.py` checks the corpus with them. When each script carried its own copy,
a change to one would have let the generator write a corpus under one rule while
the verifier declared it canonical under another — and `verify.py`'s canonical
check would have passed while the corpus disagreed with the Rust binding, which
is the exact drift the tool exists to detect.
"""

from __future__ import annotations

import hashlib
import json


def canonical(value: object) -> str:
    """CP/1 canonical form: sorted keys, no insignificant whitespace.

    `sort_keys=True` sorts by Python `str` comparison, which is Unicode code
    point order. For valid UTF-8 that is identical to UTF-8 byte order, which is
    what SPEC.md section 2 rule 1 requires and what Rust's `str: Ord` produces.
    (A JavaScript binding cannot use a bare `.sort()` here: that is UTF-16
    code-unit order, which disagrees for keys beyond the BMP.)

    `ensure_ascii=False` keeps non-ASCII as literal UTF-8 rather than `\\uXXXX`
    escapes, matching rule 3.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(document: dict) -> str:
    """SHA-256 over the canonical document with `provenance.content_hash` removed.

    A document cannot commit to its own hash. Everything else is inside it,
    including evidence and `derived_from`, so the provenance chain cannot be
    rewritten without changing the hash (SPEC.md section 4.1).
    """
    unsealed = json.loads(json.dumps(document))
    unsealed.get("provenance", {}).pop("content_hash", None)
    return hashlib.sha256(canonical(unsealed).encode("utf-8")).hexdigest()


def seal(document: dict) -> dict:
    """Return `document` with `provenance.content_hash` set to its true value."""
    document["provenance"]["content_hash"] = content_hash(document)
    return document
