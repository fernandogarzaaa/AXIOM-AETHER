#!/usr/bin/env python3
"""Regenerate the CP/1 golden fixtures and the vendoring manifest.

The fixtures are the shared conformance corpus: every CP/1 binding, in every
repository, parses each line, re-encodes it in canonical form, and asserts the
bytes are identical to what it read. That test is only meaningful if the
fixtures are themselves exactly canonical and carry correct content hashes,
which is what this script guarantees.

Run from anywhere:

    python3 protocol/cp1/tools/build_fixtures.py

It rewrites `fixtures/canonical.jsonl` and `MANIFEST.sha256` in place. Both are
committed; CI verifies the manifest rather than regenerating, so an
uncommitted edit to a schema or fixture fails loudly instead of being silently
normalized away.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

CP1_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cp1_canonical import canonical, seal  # noqa: E402  (path set above)

# A fixed instant for every fixture. Fixtures must be byte-reproducible, so
# nothing here may read the clock.
T0 = "2026-01-01T00:00:00.000Z"
T1 = "2026-01-01T00:00:01.000Z"
T2 = "2026-01-01T00:00:02.000Z"

# Fixed UUIDs. Likewise: a fixture that generated random ids would produce a
# different manifest on every run and could never be verified.
U = {
    "identity": "11111111-1111-4111-8111-111111111111",
    "lineage": "22222222-2222-4222-8222-222222222222",
    "genome": "33333333-3333-4333-8333-333333333333",
    "genome_parent": "33333333-3333-4333-8333-333333333330",
    "capability": "44444444-4444-4444-8444-444444444444",
    "belief": "55555555-5555-4555-8555-555555555555",
    "memory": "66666666-6666-4666-8666-666666666666",
    "skill": "77777777-7777-4777-8777-777777777777",
    "mutation": "88888888-8888-4888-8888-888888888888",
    "reflection": "99999999-9999-4999-8999-999999999999",
    "observation": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    "experience": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    "fitness": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
    "context": "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
    "request": "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
    "event": "ffffffff-ffff-4fff-8fff-ffffffffffff",
    "correlation": "0f0f0f0f-0f0f-4f0f-8f0f-0f0f0f0f0f0f",
    "causation": "1f1f1f1f-1f1f-4f1f-8f1f-1f1f1f1f1f1f",
}


def prov(
    authored_by: str,
    origin: str,
    evidence: list[str] | None = None,
    derived_from: list[str] | None = None,
    produced_at: str = T0,
) -> dict:
    return {
        "authored_by": authored_by,
        "produced_at": produced_at,
        "origin": origin,
        "evidence": evidence or [],
        "derived_from": derived_from or [],
    }


def identity() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "Identity",
            "id": U["identity"],
            "lineage_id": U["lineage"],
            "name": "ADAM",
            "description": "A persistent cognitive organism whose identity outlives any single model.",
            "established_at": T0,
            "provenance": prov("adam", "genesis"),
        }
    )


def genome() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "Genome",
            "id": U["genome"],
            "version_label": "1.1",
            "parent_version_id": U["genome_parent"],
            "identity": identity(),
            "values": ["evidence over assertion", "reversibility"],
            "goals": ["survive model replacement"],
            "capabilities": ["context-compression", "experience-validation"],
            "skills": ["rust-debugging"],
            "policies": ["never accept an unvalidated genome amendment"],
            # Two of these keys are deliberately non-ASCII, and one is beyond
            # the BMP. UTF-8 byte order puts "\ufffd" before "\U0001d11e";
            # UTF-16 code-unit order puts them the other way round, because the
            # leading surrogate D834 sorts below FFFD. A binding that sorts keys
            # with a bare JavaScript `.sort()` fails the round-trip check on
            # this fixture, which is the point of carrying it.
            "preferences": {
                "tone": "direct",
                "verbosity": "low",
                "\ufffd": "replacement",
                "\U0001d11e": "g-clef",
            },
            "committed_at": T1,
            "commit_reason": "accepted mutation: amend goals.append",
            "provenance": prov("adam", "genome:commit", derived_from=[U["mutation"]]),
        }
    )


def capability() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "Capability",
            "id": U["capability"],
            "name": "experience-validation",
            "summary": "Measure a proposed change against deterministic scenarios before it is accepted.",
            "kind": "environment",
            "provider": "eve:cp1/validate",
            "requires": ["context-compression"],
            "stability": "stable",
            "provenance": prov("adam", "capability:declared"),
        }
    )


def belief() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "Belief",
            "id": U["belief"],
            "statement": "Stating a continuity goal improves task success.",
            "confidence_bp": 8200,
            "uncertainty_bp": 1500,
            "status": "held",
            "evidence_origin": "simulation",
            "formed_at": T0,
            "updated_at": T2,
            "provenance": prov(
                "adam",
                "belief:consolidation",
                # Must agree with the FitnessResult this derives from: +700bp,
                # no scenario regressing. A fixture whose stated evidence
                # contradicts the document it cites teaches every reader of the
                # corpus the wrong thing about how provenance is meant to hang
                # together.
                evidence=["fitness delta +700bp with no scenario regressing"],
                derived_from=[U["memory"], U["fitness"]],
            ),
        }
    )


def memory() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "Memory",
            "id": U["memory"],
            "kind": "semantic",
            "content": "Amendments to goals require an approving fitness result.",
            "confidence_bp": 9000,
            "salience_bp": 7500,
            "decay_rate_bp": 500,
            "access_count": 3,
            "created_at": T0,
            "last_accessed_at": T2,
            "provenance": prov(
                "adam",
                "reflection:consolidation",
                evidence=["3 episodic memories on the same topic"],
                derived_from=[U["experience"]],
            ),
        }
    )


def skill() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "Skill",
            "id": U["skill"],
            "name": "rust-debugging",
            "summary": "Diagnose and repair failing Rust builds.",
            "stage": "promoted",
            "procedure": [
                "read the first error, not the last",
                "check the dependency graph for version conflicts",
                "reproduce with --verbose before changing anything",
            ],
            "fitness_bp": 8750,
            "trials_passed": 7,
            "trials_total": 8,
            "provenance": prov("adam", "skill:promotion", evidence=["7/8 trials passed"]),
        }
    )


def mutation() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "Mutation",
            "id": U["mutation"],
            "kind": "amend_genome",
            "target": "goals.append",
            "current_value": "",
            "proposed_value": "survive model replacement",
            "rationale": "Recurring conflicts show the organism has no stated continuity goal.",
            "confidence_bp": 7000,
            "risk_bp": 6000,
            "status": "accepted",
            "provenance": prov(
                "adam",
                "evolution:analyze",
                evidence=["recurring conflict topic 'identity' seen 4 times"],
            ),
        }
    )


def reflection() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "Reflection",
            "id": U["reflection"],
            "genome_version_id": U["genome"],
            "genome_version_label": "1.1",
            "observed_at": T2,
            "memory_total": 42,
            "active_beliefs": 6,
            "promoted_skills": 1,
            "retired_skills": 0,
            "pending_mutations": 2,
            "accepted_mutations": 1,
            "notes": ["belief instability concentrated on one statement"],
            "provenance": prov("adam", "reflection:cycle"),
        }
    )


def observation() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "Observation",
            "id": U["observation"],
            "environment_id": "eve:scenario/excellent",
            "surface": "mock",
            "at": T0,
            "locator": "mock://excellent/signup",
            "title": "Create your account — Clarity",
            "signals": ["Create your account", "Step 1 of 1 — this is the only step."],
            "affordances": [
                {"label": "Create account", "role": "button", "enabled": True},
                {"label": "Back to home", "role": "link", "enabled": True},
            ],
            "latency_ms": 120,
            "error_perceived": False,
            "provenance": prov("eve", "eve:scenario/excellent"),
        }
    )


def experience() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "Experience",
            "id": U["experience"],
            "observation_id": U["observation"],
            "step": 4,
            "goal": "create an account and get to the main screen",
            "action": "click",
            "action_description": 'click "Create account"',
            "prediction": {
                "description": "the account is created and a main screen appears",
                "confidence_bp": 8000,
                "expects_change": True,
            },
            "outcome": "success",
            "surprise_bp": 500,
            "affect": {"frustration_bp": 1000, "trust_bp": 7200, "confidence_bp": 8100},
            "provenance": prov("eve", "eve:session/excellent#4", derived_from=[U["observation"]]),
        }
    )


def fitness_result() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "FitnessResult",
            "id": U["fitness"],
            "mutation_id": U["mutation"],
            "seed": 1337,
            "scenario_ids": ["excellent", "average", "bad"],
            "trials": 3,
            "baseline": {
                "composite_bp": 6400,
                "task_success_bp": 6667,
                "frustration_bp": 3100,
                "trust_bp": 6000,
                "cognitive_load_bp": 4200,
                "runs": 9,
            },
            "candidate": {
                "composite_bp": 7100,
                "task_success_bp": 7778,
                "frustration_bp": 2600,
                "trust_bp": 6500,
                "cognitive_load_bp": 3900,
                "runs": 9,
            },
            "delta_bp": 700,
            "recommendation": "approve",
            "reason": "candidate improved composite by 700bp across 9 seeded runs with no scenario regressing",
            "provenance": prov(
                "eve",
                "eve:cp1/validate",
                evidence=["seed=1337", "scenarios=excellent,average,bad", "trials=3"],
                derived_from=[U["mutation"]],
                produced_at=T2,
            ),
        }
    )


def context() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "Context",
            "id": U["context"],
            "purpose": "answer a question about the organism's continuity goal",
            "token_budget": 4096,
            "tokens_before": 31500,
            "tokens_after": 3780,
            "compression_ratio_bp": 1200,
            "grounded": True,
            "grounding_failures": [],
            "memory_ids": [U["memory"]],
            "belief_ids": [U["belief"]],
            "segments": [
                {"role": "identity", "content": "You are ADAM.", "source_id": U["identity"]},
                {"role": "goal", "content": "survive model replacement"},
                {
                    "role": "memory",
                    "content": "Amendments to goals require an approving fitness result.",
                    "source_id": U["memory"],
                },
            ],
            "provenance": prov(
                "axiom",
                "axiom:context/compress",
                evidence=["ttt-session=deadbeef"],
                derived_from=[U["memory"], U["belief"]],
                produced_at=T1,
            ),
        }
    )


def validation_request() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "ValidationRequest",
            "id": U["request"],
            "mutation": mutation(),
            "genome_before_hash": "a" * 64,
            "genome_after_hash": "b" * 64,
            "scenario_ids": ["excellent", "average", "bad"],
            "seed": 1337,
            "trials": 3,
            "provenance": prov("adam", "adam:evolution/validate", derived_from=[U["mutation"]]),
        }
    )


def event() -> dict:
    return seal(
        {
            "cp": "cp1",
            "type": "GenomeCommitted",
            "id": U["event"],
            "occurred_at": T2,
            "actor": "adam",
            "subject_id": U["genome"],
            "subject_type": "Genome",
            "correlation_id": U["correlation"],
            "causation_id": U["causation"],
            "payload": {
                "version_label": "1.1",
                "reason": "accepted mutation: amend goals.append",
                "fitness_delta_bp": 700,
                "validated": True,
            },
            "provenance": prov("adam", "adam:genome/commit", derived_from=[U["fitness"]], produced_at=T2),
        }
    )


BUILDERS = [
    identity,
    genome,
    capability,
    belief,
    memory,
    skill,
    mutation,
    reflection,
    observation,
    experience,
    fitness_result,
    context,
    validation_request,
    event,
]


def build_fixtures() -> str:
    return "".join(canonical(build()) + "\n" for build in BUILDERS)


def build_manifest() -> str:
    """SHA-256 of every file a vendored binding must copy verbatim."""
    tracked = sorted(
        [
            CP1_ROOT / "SPEC.md",
            CP1_ROOT / "VERSION",
            CP1_ROOT / "schema" / "cp1.schema.json",
            CP1_ROOT / "fixtures" / "canonical.jsonl",
        ],
        key=lambda p: p.relative_to(CP1_ROOT).as_posix(),
    )
    lines = []
    for path in tracked:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(CP1_ROOT).as_posix()}\n")
    return "".join(lines)


def main() -> None:
    fixtures_path = CP1_ROOT / "fixtures" / "canonical.jsonl"
    # `newline="\n"` rather than the platform default: a regeneration on
    # Windows would otherwise write CRLF, change every byte offset, and change
    # MANIFEST.sha256 — defeating the byte-reproducibility the corpus exists for.
    fixtures_path.write_text(build_fixtures(), encoding="utf-8", newline="\n")
    (CP1_ROOT / "MANIFEST.sha256").write_text(
        build_manifest(), encoding="utf-8", newline="\n"
    )
    print(f"wrote {fixtures_path.relative_to(CP1_ROOT)} ({len(BUILDERS)} documents)")
    print("wrote MANIFEST.sha256")


if __name__ == "__main__":
    main()
