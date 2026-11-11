#!/usr/bin/env python3
"""generate_attestation.py: SBOM plus provenance attestation per artifact.

Generates two files next to the artifact:
  <artifact>.sbom.json        SPDX 2.3 JSON, source-level SBOM
  <artifact>.provenance.json  build provenance (SLSA-style predicate)

The SBOM lists the artifact itself plus its build inputs as packages,
each with a SHA-256 checksum. It is a source SBOM, not a package-manager
SBOM. When `syft` is installed, --syft also runs it over the artifact
and writes <artifact>.sbom.syft.json, which is the package-level SBOM
the enforcement PR prefers. When syft is absent the script says so and
still writes the source SBOM, and exits 0: use --syft-required when a
missing or failed syft run must fail the build instead. With
--syft-required the script also verifies the syft output file exists,
parses as JSON, and carries an SPDX version before reporting success.

Provenance records the source commit, the builder, and the build
inputs (materials) with their hashes, so a verifier can re-hash the
tree and confirm the artifact was built from the claimed inputs.

Usage:
  python3 scripts/generate_attestation.py generate --artifact FILE [--repo-root DIR]
  python3 scripts/generate_attestation.py generate --artifact FILE --syft-required
  python3 scripts/generate_attestation.py verify --artifact FILE --provenance FILE [--sbom FILE]
  python3 scripts/generate_attestation.py attest-image --image REF --digest sha256:... --out FILE

Verify re-hashes the artifact, compares it with the attestation, and
checks every recorded material against the current tree. Exit 0 means
the attestation verifies. Exit 1 means a mismatch. No network access.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

# Build inputs recorded as provenance materials. Paths are relative to
# the repo root. Missing files are skipped, never invented.
MATERIAL_PATHS = [
    "axiom_engine_rs/Cargo.lock",
    "axiom_mesh_rs/Cargo.lock",
    "Dockerfile",
    "deploy/helm/axiom/values.yaml",
    "deploy/helm/axiom/Chart.yaml",
    ".github/workflows/release.yml",
    ".github/workflows/docker.yml",
    ".github/workflows/ci.yml",
]

SPDX_VERSION = "SPDX-2.3"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_root):
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def git_status_clean(repo_root):
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return out.stdout.strip() == ""
    except (subprocess.SubprocessError, OSError):
        return False


def repo_root_of(artifact, explicit):
    if explicit:
        return os.path.abspath(explicit)
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.dirname(here)


def materials(repo_root):
    found = []
    for rel in MATERIAL_PATHS:
        path = os.path.join(repo_root, rel)
        if os.path.isfile(path):
            found.append(
                {
                    "uri": rel,
                    "digest": {"sha256": sha256_file(path)},
                }
            )
    return found


def build_sbom(artifact, repo_root):
    name = os.path.basename(artifact)
    digest = sha256_file(artifact)
    packages = [
        {
            "SPDXID": "SPDXRef-Package-artifact",
            "name": name,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "checksums": [
                {
                    "algorithm": "SHA256",
                    "checksumValue": digest,
                }
            ],
        }
    ]
    for m in materials(repo_root):
        packages.append(
            {
                "SPDXID": "SPDXRef-Package-"
                + m["uri"].replace("/", "-").replace(".", "-"),
                "name": m["uri"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "checksums": [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": m["digest"]["sha256"],
                    }
                ],
            }
        )
    return {
        "spdxVersion": SPDX_VERSION,
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{name}-sbom",
        "documentNamespace": f"https://github.com/fernandogarzaaa/AXIOM-AETHER/sbom/{name}",
        "creationInfo": {
            "creators": ["Tool: generate_attestation.py"],
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "packages": packages,
    }


def build_provenance(artifact, repo_root, builder):
    commit = git_commit(repo_root)
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": os.path.basename(artifact),
                "digest": {"sha256": sha256_file(artifact)},
            }
        ],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://github.com/fernandogarzaaa/AXIOM-AETHER/generate_attestation.py",
                "externalParameters": {
                    "sourceCommit": commit,
                    "sourceTreeClean": git_status_clean(repo_root),
                    "builder": builder or platform.node(),
                    "platform": platform.platform(),
                },
                "resolvedDependencies": materials(repo_root),
            },
            "runDetails": {
                "builder": {"id": builder or platform.node()},
            },
        },
    }


def run_syft(artifact, out_path):
    """Run syft over the artifact and validate its SPDX output.

    Returns True only when syft exits 0 AND the output file exists,
    is non-empty, parses as JSON, and carries an SPDX version. A
    zero exit with no usable output is a failure, not a success.
    """
    if shutil.which("syft") is None:
        print("syft: not installed; package-level SBOM skipped")
        return False
    try:
        result = subprocess.run(
            ["syft", artifact, "-o", f"spdx-json={out_path}"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("syft: timed out; package-level SBOM skipped")
        return False
    if result.returncode != 0:
        print(f"syft: failed ({result.stderr.strip()[:200]})")
        return False
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        print(f"syft: exited 0 but wrote no output at {out_path}")
        return False
    try:
        with open(out_path) as f:
            doc = json.load(f)
    except ValueError as e:
        print(f"syft: output is not valid JSON: {e}")
        return False
    if not doc.get("spdxVersion", "").startswith("SPDX-"):
        print("syft: output has no SPDX version; package-level SBOM skipped")
        return False
    print(
        f"syft SBOM written: {out_path} ({doc['spdxVersion']}, "
        f"{len(doc.get('packages', []))} packages)"
    )
    return True


def cmd_generate(args):
    artifact = os.path.abspath(args.artifact)
    if not os.path.isfile(artifact):
        print(f"generate: artifact not found: {artifact}", file=sys.stderr)
        return 2
    root = repo_root_of(artifact, args.repo_root)
    sbom_path = args.sbom_out or artifact + ".sbom.json"
    prov_path = args.provenance_out or artifact + ".provenance.json"
    sbom = build_sbom(artifact, root)
    prov = build_provenance(artifact, root, args.builder)
    with open(sbom_path, "w") as f:
        json.dump(sbom, f, indent=2)
        f.write("\n")
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")
    print(f"artifact: {artifact}")
    print(f"sha256: {prov['subject'][0]['digest']['sha256']}")
    print(
        f"source commit: {prov['predicate']['buildDefinition']['externalParameters']['sourceCommit'] or '(unknown)'}"
    )
    print(
        f"tree clean: {prov['predicate']['buildDefinition']['externalParameters']['sourceTreeClean']}"
    )
    print(
        f"materials recorded: {len(prov['predicate']['buildDefinition']['resolvedDependencies'])}"
    )
    print(f"SBOM written: {sbom_path}")
    print(f"provenance written: {prov_path}")
    if args.syft or args.syft_required:
        ok = run_syft(artifact, artifact + ".sbom.syft.json")
        if args.syft_required and not ok:
            print(
                "generate: --syft-required and syft failed; failing the build",
                file=sys.stderr,
            )
            return 1
    return 0


def cmd_verify(args):
    artifact = os.path.abspath(args.artifact)
    if not os.path.isfile(artifact):
        print(f"verify: artifact not found: {artifact}", file=sys.stderr)
        return 2
    try:
        with open(args.provenance) as f:
            prov = json.load(f)
    except (OSError, ValueError) as e:
        print(f"verify: cannot read provenance: {e}", file=sys.stderr)
        return 2
    failures = []
    recorded = prov["subject"][0]["digest"]["sha256"]
    actual = sha256_file(artifact)
    if recorded == actual:
        print(f"PASS: artifact sha256 matches attestation ({actual[:16]}...)")
    else:
        failures.append("artifact sha256 mismatch")
        print(f"FAIL: artifact sha256 {actual} != attested {recorded}")
    root = repo_root_of(artifact, args.repo_root)
    commit = git_commit(root)
    attested_commit = prov["predicate"]["buildDefinition"]["externalParameters"].get(
        "sourceCommit", ""
    )
    if commit and attested_commit:
        if commit == attested_commit:
            print(f"PASS: source commit matches HEAD ({commit[:12]})")
        else:
            print(
                f"NOTE: HEAD is {commit[:12]}, attestation records {attested_commit[:12]} (artifact built from an older commit)"
            )
    for m in prov["predicate"]["buildDefinition"].get("resolvedDependencies", []):
        path = os.path.join(root, m["uri"])
        if not os.path.isfile(path):
            print(f"NOTE: material missing from tree: {m['uri']}")
            continue
        if sha256_file(path) == m["digest"]["sha256"]:
            print(f"PASS: material unchanged: {m['uri']}")
        else:
            failures.append(f"material changed: {m['uri']}")
            print(f"FAIL: material changed: {m['uri']}")
    if args.sbom:
        try:
            with open(args.sbom) as f:
                sbom = json.load(f)
            pkgs = {p["name"]: p for p in sbom.get("packages", [])}
            entry = pkgs.get(os.path.basename(artifact))
            if entry and entry["checksums"][0]["checksumValue"] == actual:
                print("PASS: SBOM artifact checksum matches")
            else:
                failures.append("SBOM artifact checksum mismatch")
                print("FAIL: SBOM artifact checksum mismatch")
        except (OSError, ValueError) as e:
            failures.append(f"SBOM unreadable: {e}")
            print(f"FAIL: SBOM unreadable: {e}")
    if failures:
        print(f"VERIFY: FAIL ({len(failures)} failure(s))")
        return 1
    print("VERIFY: PASS")
    return 0


def cmd_attest_image(args):
    """Write a provenance attestation for a pushed container image.

    The image itself is attested by buildx (SBOM and SLSA provenance
    attached in the registry). This file is the repo-side record: the
    image reference, its manifest digest, the source commit, the
    builder, and the same build-input materials as a file artifact.
    """
    digest = args.digest
    digest = digest.removeprefix("sha256:")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        print(f"attest-image: bad digest: {args.digest}", file=sys.stderr)
        return 2
    digest = digest.lower()
    root = repo_root_of(".", args.repo_root)
    commit = git_commit(root)
    prov = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": args.image,
                "digest": {"sha256": digest},
            }
        ],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://github.com/fernandogarzaaa/AXIOM-AETHER/docker.yml",
                "externalParameters": {
                    "sourceCommit": commit,
                    "sourceTreeClean": git_status_clean(root),
                    "builder": args.builder or platform.node(),
                    "platform": platform.platform(),
                    "imageDigest": f"sha256:{digest}",
                },
                "resolvedDependencies": materials(root),
            },
            "runDetails": {
                "builder": {"id": args.builder or platform.node()},
            },
        },
    }
    with open(args.out, "w") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")
    print(f"image: {args.image}@sha256:{digest}")
    print(f"source commit: {commit or '(unknown)'}")
    print(
        f"materials recorded: {len(prov['predicate']['buildDefinition']['resolvedDependencies'])}"
    )
    print(f"provenance written: {args.out}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SBOM and provenance attestation tooling."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser(
        "generate", help="Generate SBOM and provenance for an artifact."
    )
    gen.add_argument("--artifact", required=True)
    gen.add_argument("--repo-root", default=None)
    gen.add_argument(
        "--builder", default=None, help="Builder id (defaults to hostname)."
    )
    gen.add_argument("--sbom-out", default=None)
    gen.add_argument("--provenance-out", default=None)
    gen.add_argument(
        "--syft", action="store_true", help="Also run syft for a package-level SBOM."
    )
    gen.add_argument(
        "--syft-required",
        action="store_true",
        help="Fail (exit 1) when the syft SBOM is missing, fails, or invalid.",
    )
    ver = sub.add_parser("verify", help="Verify an artifact against its attestation.")
    ver.add_argument("--artifact", required=True)
    ver.add_argument("--provenance", required=True)
    ver.add_argument("--sbom", default=None)
    ver.add_argument("--repo-root", default=None)
    img = sub.add_parser(
        "attest-image", help="Write provenance for a pushed container image."
    )
    img.add_argument(
        "--image", required=True, help="Image reference, e.g. ghcr.io/org/repo:tag."
    )
    img.add_argument("--digest", required=True, help="Manifest digest, sha256:<hex>.")
    img.add_argument("--out", required=True, help="Provenance JSON to write.")
    img.add_argument("--repo-root", default=None)
    img.add_argument(
        "--builder", default=None, help="Builder id (defaults to hostname)."
    )
    args = parser.parse_args(argv)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "attest-image":
        return cmd_attest_image(args)
    return cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
