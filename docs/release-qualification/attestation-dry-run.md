# Attestation dry run

Run 2026-11-11 in the `phase5/sbom-provenance` worktree at commit
`10ee468` (main HEAD; tree dirty with the uncommitted PR changes, so
the attestation records `tree clean: False`). The artifact is a real
release archive built the same way `release.yml` builds one: the
`axiom_engine` release binary from
`cargo build --release --locked --bin axiom_engine` (finished in
6m 59s), packaged with README.md and LICENSE into
`axiom-ttt-v0.7.0-test-linux-x86_64.tar.gz` (9,884,389 bytes).

## Generate

Command:

```sh
python3 scripts/generate_attestation.py generate \
  --artifact axiom-ttt-v0.7.0-test-linux-x86_64.tar.gz \
  --repo-root . --syft-required
```

Syft is the real 1.32.0 CLI, installed from the anchore release
binary. Output:

```text
artifact: /home/hatch/workspace/sbom-proof/real/axiom-ttt-v0.7.0-test-linux-x86_64.tar.gz
sha256: 9ee04ce8e8a63eedcc3b8fb2d27ba5e5d550b1c296459297476b545de91d2d23
source commit: 10ee468996e742adf6845dfa3f8bd8e7e6a78098
tree clean: False
materials recorded: 8
SBOM written: axiom-ttt-v0.7.0-test-linux-x86_64.tar.gz.sbom.json
provenance written: axiom-ttt-v0.7.0-test-linux-x86_64.tar.gz.provenance.json
syft SBOM written: axiom-ttt-v0.7.0-test-linux-x86_64.tar.gz.sbom.syft.json (SPDX-2.3, 1 packages)
```

The syft SBOM is valid SPDX-2.3 JSON. It lists one package, the
archive itself: a static Rust binary carries no package-manager
metadata for syft to catalog. Syft run directly against the bare
binary finds the same single package. The source SBOM's
`creationInfo.created` is a real UTC timestamp.

## Verify

Command:

```sh
python3 scripts/generate_attestation.py verify \
  --artifact axiom-ttt-v0.7.0-test-linux-x86_64.tar.gz \
  --provenance axiom-ttt-v0.7.0-test-linux-x86_64.tar.gz.provenance.json \
  --sbom axiom-ttt-v0.7.0-test-linux-x86_64.tar.gz.sbom.json
```

Output:

```text
PASS: artifact sha256 matches attestation (9ee04ce8e8a63eed...)
PASS: source commit matches HEAD (10ee468996e7)
PASS: material unchanged: axiom_engine_rs/Cargo.lock
PASS: material unchanged: axiom_mesh_rs/Cargo.lock
PASS: material unchanged: Dockerfile
PASS: material unchanged: deploy/helm/axiom/values.yaml
PASS: material unchanged: deploy/helm/axiom/Chart.yaml
PASS: material unchanged: .github/workflows/release.yml
PASS: material unchanged: .github/workflows/docker.yml
PASS: material unchanged: .github/workflows/ci.yml
PASS: SBOM artifact checksum matches
VERIFY: PASS
```

Exit code 0.

## Tamper check

One byte was appended to a copy of the archive, and verify ran again
against the original attestation. Output, abridged to the changed
lines:

```text
FAIL: artifact sha256 81c92414c3385d414f839320b0bb0a4b9702810589b416c853f1430da105b837 != attested 9ee04ce8e8a63eedcc3b8fb2d27ba5e5d550b1c296459297476b545de91d2d23
FAIL: SBOM artifact checksum mismatch
VERIFY: FAIL (2 failure(s))
```

Exit code 1. A modified artifact fails verification. The materials
still passed, which is correct: the tree did not change, the artifact
did.

## Syft failure modes (fixed in this PR)

The prep script's `--syft` path was untested. Testing it against the
real syft 1.32.0 found the CLI invocation itself works (tar.gz, zip,
and raw `.bin` all produce valid SPDX), but two defects around it:

- A missing or failing syft run printed a notice and the script still
  exited 0, so a release could ship with no package SBOM while the
  workflow reported success. Fixed with `--syft-required`, which the
  release and checkpoint jobs use. Verified: with syft absent from
  PATH, `generate --syft-required` exits 1; with a stub syft that
  exits 0 but writes nothing, it exits 1; with a stub that writes
  non-SPDX output, it exits 1.
- `run_syft` never validated the syft output file. It now checks the
  file exists, is non-empty, parses as JSON, and carries an SPDX
  version before reporting success.
- The source SBOM wrote an empty `creationInfo.created`, which is not
  a valid SPDX timestamp. It now writes a UTC timestamp.

The earlier prep dry run (2026-11-10, README.md stand-in, no syft)
is superseded by this run.

## Not covered by this dry run

- No pushed container image was attested locally: docker is not
  available in this sandbox. The image attestation (`attest-image`,
  syft against the pushed image, native buildx SBOM and provenance)
  is proven by the `docker.yml` workflow_dispatch run on the PR
  branch instead. `attest-image` was smoke-tested locally with a
  placeholder digest (exit 0; a bad digest exits 2).
- The Windows and macOS archives are attested by the `release.yml`
  workflow_dispatch run on the PR branch, not locally.
