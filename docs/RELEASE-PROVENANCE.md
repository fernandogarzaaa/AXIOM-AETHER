# Provenance attestations

Audit section 7, Phase 5. Every release artifact ships with an SBOM
and a provenance attestation. This document defines the tooling, the
verification procedure, and the workflow wiring.

## What the tooling does

`scripts/generate_attestation.py` has three commands and no network
access.

- `generate --artifact FILE` writes `<artifact>.sbom.json` (SPDX 2.3
  JSON, source-level SBOM) and `<artifact>.provenance.json`
  (SLSA-style provenance). The provenance records the source commit,
  whether the tree was clean, the builder id, the platform, and every
  build input in `MATERIAL_PATHS` with its SHA-256 hash.
- `verify --artifact FILE --provenance FILE [--sbom FILE]` re-hashes
  the artifact, compares it with the attestation, checks the source
  commit against HEAD, and checks every recorded material against the
  current tree. Exit 0 means the attestation verifies.
- `attest-image --image REF --digest sha256:... --out FILE` writes a
  provenance attestation for a pushed container image. The subject is
  the image reference plus its manifest digest.

With `--syft`, generate also runs `syft` over the artifact and writes
`<artifact>.sbom.syft.json`. That is the package-level SBOM the
release pipeline requires. With `--syft-required`, a missing, failed,
or invalid syft run fails the build instead of printing a notice and
continuing. The release and checkpoint jobs use `--syft-required`.
Plain `--syft` keeps the lenient behavior for local runs. The source
SBOM is not a substitute for a package SBOM. It proves what tree the
artifact came from, not what packages are inside it.

## How to verify an attestation

The release notes link the artifact, its `.sbom.json`, and its
`.provenance.json`. A verifier runs this, with no trust in the
release notes themselves.

```sh
python3 scripts/generate_attestation.py verify \
  --artifact axiom-ttt-v0.7.0-linux-x86_64.tar.gz \
  --provenance axiom-ttt-v0.7.0-linux-x86_64.tar.gz.provenance.json \
  --sbom axiom-ttt-v0.7.0-linux-x86_64.tar.gz.sbom.json
```

Expected output, one line per check.

```text
PASS: artifact sha256 matches attestation (844f485adf925e51...)
PASS: source commit matches HEAD (bc98eab...)
PASS: material unchanged: axiom_engine_rs/Cargo.lock
PASS: material unchanged: Dockerfile
PASS: SBOM artifact checksum matches
VERIFY: PASS
```

A `NOTE: HEAD is ...` line means the artifact was built from an older
commit than the tree being verified against. That is expected when
verifying an old release against a newer checkout. It is not a
failure. A `FAIL` line is. A changed material means the tree no
longer matches the attested inputs, so the attestation cannot be
reproduced from this tree.

## Workflow wiring

In `release.yml`, after each archive is packaged and before upload,
one step per artifact generates the SBOM and provenance with
`--syft-required`, then verifies the archive against its own
attestation. The upload step lists the `.sbom.json`,
`.provenance.json`, and `.sbom.syft.json` files next to the archive
and its `.sha256` sidecar, with `fail_on_unmatched_files: true`, so
a missing sidecar fails the release. The checkpoint job does the same
for each checkpoint file in `ckpt_release/`.

In `docker.yml`, each per-arch build attaches a native SPDX SBOM and
SLSA provenance attestation to the pushed image (`sbom: true`,
`provenance: mode=max` on the build-push step). After the manifest
merge job, a step computes the multi-arch manifest digest as the
SHA-256 of the raw manifest bytes, writes `image.provenance.json`
with `attest-image`, and runs syft against the pushed image for
`image.sbom.syft.json`. Both files are uploaded as workflow artifacts
on every run, and as release assets on tag runs.

To verify a pushed image, compare the digest in
`image.provenance.json` with the live manifest digest.

```sh
DIGEST="sha256:$(docker buildx imagetools inspect ghcr.io/fernandogarzaaa/axiom-aether:v0.7.0 --raw | sha256sum | cut -d' ' -f1)"
python3 -c "import json; print(json.load(open('image.provenance.json'))['predicate']['buildDefinition']['externalParameters']['imageDigest'])"
```

The two digests must match. The source commit in the same file must
match the release tag's commit.

## Dry run

The dry-run transcript for this tooling is in
`docs/release-qualification/attestation-dry-run.md`. It runs the tool
against a real release binary and a real tar.gz, verifies the
attestations it produces, checks the syft SBOMs, and includes a
tamper check that flips the verdict to FAIL.
