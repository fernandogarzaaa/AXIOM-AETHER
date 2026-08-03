#!/usr/bin/env python3
"""Verify the CP/1 normative source is internally consistent.

Three independent checks, each of which has caught a different class of error:

1. **Manifest.** Every tracked file hashes to what `MANIFEST.sha256` records.
   Catches an edit to a schema or fixture that was not propagated by
   `build_fixtures.py`, which would silently desynchronize the vendored
   bindings that verify against the manifest.
2. **Schema.** Every fixture validates against `schema/cp1.schema.json`.
   Catches a fixture and a schema drifting apart — the failure mode where the
   conformance corpus and the contract disagree about what is legal.
3. **Canonicity and sealing.** Every fixture line is already in canonical form
   (re-encoding is a no-op) and its `provenance.content_hash` is the true hash
   of the document with that member removed. Catches a hand-edited fixture,
   which would make every binding's round-trip assertion fail for a reason
   that has nothing to do with the binding.

Exit code is non-zero on any failure, so this is usable directly as a CI step:

    python3 protocol/cp1/tools/verify.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

CP1_ROOT = Path(__file__).resolve().parent.parent


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def check_manifest() -> list[str]:
    manifest_path = CP1_ROOT / "MANIFEST.sha256"
    if not manifest_path.exists():
        return ["MANIFEST.sha256 is missing; run tools/build_fixtures.py"]

    failures = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, _, rel = line.partition("  ")
        target = CP1_ROOT / rel
        if not target.exists():
            failures.append(f"manifest lists {rel}, which does not exist")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"{rel}: manifest says {expected[:12]}…, file hashes {actual[:12]}…")
    return failures


def load_fixtures() -> list[tuple[int, str, dict]]:
    path = CP1_ROOT / "fixtures" / "canonical.jsonl"
    fixtures = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fixtures.append((lineno, line, json.loads(line)))
    return fixtures


def check_schema(fixtures: list[tuple[int, str, dict]]) -> list[str]:
    schema = json.loads((CP1_ROOT / "schema" / "cp1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    failures = []
    for lineno, _, doc in fixtures:
        errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
        for error in errors:
            location = "/".join(str(p) for p in error.path) or "<root>"
            failures.append(f"line {lineno} ({doc.get('type')}) at {location}: {error.message}")
    return failures


def check_canonical_and_sealed(fixtures: list[tuple[int, str, dict]]) -> list[str]:
    failures = []
    for lineno, raw, doc in fixtures:
        if canonical(doc) != raw:
            failures.append(f"line {lineno} ({doc.get('type')}): not in canonical form")

        recorded = doc["provenance"].get("content_hash")
        unsealed = json.loads(raw)
        unsealed["provenance"].pop("content_hash", None)
        actual = hashlib.sha256(canonical(unsealed).encode("utf-8")).hexdigest()
        if recorded != actual:
            failures.append(
                f"line {lineno} ({doc.get('type')}): content_hash is {recorded[:12]}…, "
                f"true hash is {actual[:12]}…"
            )
    return failures


def check_coverage(fixtures: list[tuple[int, str, dict]]) -> list[str]:
    """Every canonical type must appear in the corpus.

    Without this, adding a type to the schema and forgetting to add a fixture
    would leave that type untested in all three bindings simultaneously.
    """
    schema = json.loads((CP1_ROOT / "schema" / "cp1.schema.json").read_text(encoding="utf-8"))
    declared = {
        ref["$ref"].rsplit("/", 1)[-1] for ref in schema["oneOf"]
    }
    # Events carry their event name in `type`; they are the `Event` variant.
    event_names = set(schema["$defs"]["Event"]["properties"]["type"]["enum"])
    present = set()
    for _, _, doc in fixtures:
        kind = doc["type"]
        present.add("Event" if kind in event_names else kind)

    missing = sorted(declared - present)
    return [f"no fixture covers canonical type {name}" for name in missing]


def main() -> int:
    fixtures = load_fixtures()
    groups = {
        "manifest": check_manifest(),
        "schema": check_schema(fixtures),
        "canonical/sealed": check_canonical_and_sealed(fixtures),
        "coverage": check_coverage(fixtures),
    }

    total = 0
    for name, failures in groups.items():
        if failures:
            total += len(failures)
            print(f"FAIL {name}:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
        else:
            print(f"ok   {name}")

    if total:
        print(f"\n{total} problem(s) found", file=sys.stderr)
        return 1
    print(f"\nCP/1 normative source is consistent ({len(fixtures)} fixtures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
