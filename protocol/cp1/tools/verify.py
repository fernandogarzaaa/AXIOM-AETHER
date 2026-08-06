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
import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

CP1_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cp1_canonical import canonical, content_hash  # noqa: E402  (path set above)


def check_manifest() -> list[str]:
    manifest_path = CP1_ROOT / "MANIFEST.sha256"
    if not manifest_path.exists():
        return ["MANIFEST.sha256 is missing; run tools/build_fixtures.py"]

    failures = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, separator, rel = line.partition("  ")
        if not separator:
            # A manifest is an integrity control. Without this check the empty
            # path resolves to CP1_ROOT and the read below raises IsADirectoryError,
            # turning a reportable defect into a traceback.
            failures.append(f"malformed manifest line (expected `<sha256>  <path>`): {line!r}")
            continue
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
        # The schema's top level is a `oneOf`, so `iter_errors` yields a single
        # root error saying only "is not valid under any of the given schemas".
        # `best_match` descends into `error.context` and returns the branch
        # failure that actually names the offending member, which is the
        # difference between an actionable CI message and a shrug.
        error = best_match(validator.iter_errors(doc))
        if error is None:
            continue
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        failures.append(f"line {lineno} ({doc.get('type')}) at {location}: {error.message}")
    return failures


def check_canonical_and_sealed(fixtures: list[tuple[int, str, dict]]) -> list[str]:
    """Every fixture is already canonical and carries its true content hash.

    Every lookup here is guarded. `main` runs all four checks before reading any
    result, so `check_schema` does not gate this one — and a hand-edited fixture
    (the exact case this tool exists to catch) must produce a FAIL line rather
    than a traceback that kills the CI step.
    """
    failures = []
    for lineno, raw, doc in fixtures:
        if canonical(doc) != raw:
            failures.append(f"line {lineno} ({doc.get('type')}): not in canonical form")

        provenance = doc.get("provenance")
        if not isinstance(provenance, dict):
            failures.append(f"line {lineno} ({doc.get('type')}): provenance is missing")
            continue

        recorded = provenance.get("content_hash")
        actual = content_hash(json.loads(raw))
        if recorded != actual:
            shown = recorded[:12] if isinstance(recorded, str) else repr(recorded)
            failures.append(
                f"line {lineno} ({doc.get('type')}): content_hash is {shown}…, "
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
        kind = doc.get("type")
        present.add("Event" if kind in event_names else kind)

    missing = sorted(declared - present)
    failures = [f"no fixture covers canonical type {name}" for name in missing]
    failures.extend(check_binding_type_list(declared - {"Event"}))
    return failures


# `const COVERED_TYPES: [&str; 13] = [ "Identity", ... ];`
_COVERED_TYPES_RE = re.compile(
    r"const\s+COVERED_TYPES\s*:\s*\[&str;\s*\d+\]\s*=\s*\[(?P<body>[^\]]*)\]",
    re.DOTALL,
)


def check_binding_type_list(declared: set[str]) -> list[str]:
    """The Rust binding's hardcoded type list must match the schema's.

    `check_coverage` derives its expectation from the schema, so it tracks the
    schema automatically. The Rust binding hardcodes the same set as
    `COVERED_TYPES`. Without this, adding a fourteenth type to the schema would
    be reported here as a missing fixture, while COVERED_TYPES stayed at
    thirteen and the vendored copies in EVE and ADAM kept passing with a silent
    coverage hole.

    `declared` here excludes "Event": the Rust binding checks event coverage
    separately (`seen_types.contains("Event")`), so COVERED_TYPES never lists it.

    The binding lives outside `protocol/`, so a checkout that vendored only the
    protocol directory has nothing to compare against; that is a skip, not a
    failure.
    """
    binding = CP1_ROOT.parent.parent / "axiom_engine_rs" / "src" / "cp1" / "conformance.rs"
    if not binding.is_file():
        return []

    match = _COVERED_TYPES_RE.search(binding.read_text(encoding="utf-8"))
    if match is None:
        return [f"could not find COVERED_TYPES in {binding.name}; the drift guard is not running"]

    listed = set(re.findall(r'"([^"]+)"', match.group("body")))
    if listed == declared:
        return []
    return [
        f"COVERED_TYPES in {binding.name} disagrees with the schema: "
        f"only in schema {sorted(declared - listed)}, only in binding {sorted(listed - declared)}"
    ]


def check_provenance_edges(fixtures: list[tuple[int, str, dict]]) -> list[str]:
    """Check 5: derived_from edges resolve, and a measured FitnessResult names its run.

    See SPEC.md section 4.2. A FitnessResult asserts that baseline and candidate
    each ran n times at a given seed; without a reference to the
    SimulationCompleted that produced those runs, a measured result and a
    fabricated one are structurally identical, and the receiver cannot tell them
    apart. This is the one place a component reports on work only it can see.

    Scoped to the corpus: an edge pointing outside the supplied set is not a
    failure, because a binding cannot resolve an id it was never given.
    """
    type_by_id = {
        doc["id"]: doc.get("type") for _, _, doc in fixtures if isinstance(doc.get("id"), str)
    }

    failures = []
    for lineno, _, doc in fixtures:
        kind = doc.get("type")
        provenance = doc.get("provenance")
        if not isinstance(provenance, dict):
            continue
        derived = provenance.get("derived_from")
        if not isinstance(derived, list):
            continue

        if doc.get("id") in derived:
            failures.append(
                f"line {lineno} ({kind}): derives from itself, which is not a provenance edge"
            )

        # A result reporting no runs is the honest encoding of "EVE declined to
        # measure this". There is no simulation for it to name, and demanding
        # one would force it to invent the reference this rule exists to make
        # meaningful.
        baseline = doc.get("baseline")
        baseline_runs = baseline.get("runs", 0) if isinstance(baseline, dict) else 0
        if kind != "FitnessResult" or baseline_runs == 0:
            continue

        if not any(type_by_id.get(ref) == "SimulationCompleted" for ref in derived):
            failures.append(
                f"line {lineno} ({kind}): provenance.derived_from names no "
                "SimulationCompleted; a measurement that cannot be chained back to "
                "its run is indistinguishable from a fabricated one (SPEC.md 4.2)"
            )
    return failures


def main() -> int:
    fixtures = load_fixtures()
    groups = {
        "manifest": check_manifest(),
        "schema": check_schema(fixtures),
        "canonical/sealed": check_canonical_and_sealed(fixtures),
        "coverage": check_coverage(fixtures),
        "provenance edges": check_provenance_edges(fixtures),
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
