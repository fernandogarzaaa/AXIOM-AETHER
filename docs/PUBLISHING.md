# Publishing AXIOM

The `axiom` engine ships through three channels, all driven by a single version
tag. End users install with one of:

```bash
pip install axiom-aether            # prebuilt binary wheel (no toolchain)
cargo install axiom_engine          # build from source
curl -fsSL https://raw.githubusercontent.com/fernandogarzaaa/AXIOM-AETHER/main/scripts/install.sh | bash   # GitHub Release binary
```

## What is published where

| Registry | Package | Command(s) installed | Built by |
|---|---|---|---|
| **PyPI** | `axiom-aether` | `axiom` (+ `axiom_engine` alias) | maturin wheels (`publish.yml` → `wheels`/`pypi`) |
| **crates.io** | `axiom_engine` | `axiom`, `axiom_engine` | `cargo publish` (`publish.yml` → `crates-io`) |
| **GitHub Releases** | tarball/zip | `axiom_engine` → installed as `axiom` | `release.yml` |

The pure-Python package at the repo root (`pyproject.toml`, **`axiom-engine`**)
is a *separate* reference implementation — distinct from the `axiom-aether`
binary wheel. Keep the names distinct to avoid confusion.

### Release checksums

Every binary archive on a GitHub Release ships with a `.sha256` sidecar,
written by the `build` job in `release.yml`:

```
axiom-ttt-<version>-linux-x86_64.tar.gz      axiom-ttt-<version>-linux-x86_64.tar.gz.sha256
axiom-ttt-<version>-macos-arm64.tar.gz       axiom-ttt-<version>-macos-arm64.tar.gz.sha256
axiom-ttt-<version>-windows-x86_64.zip       axiom-ttt-<version>-windows-x86_64.zip.sha256
```

Each sidecar is one line in `sha256sum -c` format (`<lowercase hex>  <filename>`),
so you can check a manual download with `sha256sum -c <asset>.sha256` (or
`shasum -a 256 -c` on macOS) from the directory holding both files.

- **Installers verify by default.** `scripts/install.sh` and `scripts/install.ps1`
  download the sidecar for the archive they selected, hash the archive, and
  abort on a mismatch ("refusing to install a corrupted or substituted
  archive"). A match prints `checksum verified (<hex>)`.
- **Older releases have no sidecar.** The installers then warn
  (`no .sha256 sidecar published for this release; skipping verification`)
  and continue. Set `AXIOM_REQUIRE_CHECKSUM=1` to fail instead. `install.sh`
  also skips with a warning when neither `sha256sum` nor `shasum` is
  installed, unless `AXIOM_REQUIRE_CHECKSUM=1` is set.
- **Checkpoint assets** are covered by the separate `SHA256SUMS.txt` published
  by the `checkpoint` job. To make `axiom init` fail closed on a substituted
  checkpoint, copy the `axiom_production_bpe.bin` line's hash from that file
  and export it before fetching: `AXIOM_CHECKPOINT_SHA256=<hex>` (and
  `AXIOM_TOKENIZER_SHA256` for the tokenizer, from the `axiom_bpe.json` line).
- **Not signed.** The sidecars and `SHA256SUMS.txt` are plain, unsigned
  manifests hosted next to the assets. They catch corruption and partial or
  mismatched downloads, but anyone who can replace a release asset can replace
  its checksum too. They are not a defence against a compromised release or
  account. Signing (e.g. cosign/minisign or build attestations) is still open.

## How the binary set is controlled

The crate defines two user-facing bins (`axiom`, `axiom_engine`) from the same
entrypoint plus the `axiom-agent-task` MCP server, and gates its 11
dev/training/eval bins (`axiombench`, trainers, crawlers, evals) behind the **`tools`**
feature (`Cargo.toml`). A default build — `cargo install`, the pip wheel, Docker,
and release — therefore ships only `axiom` (+ the `axiom_engine` alias), not the
training toolchain. Build the dev tools with `cargo build --features tools`.

## Release procedure

1. Bump `version` in **only** `axiom_engine_rs/Cargo.toml`; the
   `axiom_engine_rs/pyproject.toml` project version is dynamic, so maturin
   derives the `axiom-aether` wheel version from Cargo metadata.
2. Commit, then tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. `publish.yml` runs on the tag:
   - `crates-io`: `cargo publish --locked`.
   - `wheels`: maturin builds wheels for linux-x86_64 (manylinux), macOS-arm64,
     windows-x64.
   - `pypi`: uploads all wheels via PyPI Trusted Publishing.
   - `release.yml` (separately) attaches the GitHub Release binaries.

## One-time setup

- **crates.io:** configure
  [Trusted Publishing](https://crates.io/docs/trusted-publishing) for this repo +
  the `publish.yml` workflow (no API token needed). Verify the crate name
  `axiom_engine` is available (or pick another and update `Cargo.toml`).
- **PyPI:** verify the name `axiom-aether` is free, then configure
  [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) for this repo +
  the `publish.yml` workflow + a `pypi` environment (no API token needed). As a
  fallback, set `MATURIN_PYPI_TOKEN` and switch the `pypi` job to a token upload.

## Pre-publish verification (run locally before tagging)

```bash
cd axiom_engine_rs
cargo publish --dry-run --allow-dirty          # packaging + metadata + size
cargo build --release --bin axiom              # the shipped command
maturin build --release --bindings bin --out dist && \
  pip install dist/axiom_aether-*.whl           # wheel installs `axiom`
../scripts/new_user_simulation.sh              # full first-run journey, 9/9
```

## Notes & caveats

- Tag == published artifact. The registry release for a version must be built
  from that version's tag commit, so `git checkout vX.Y.Z` reproduces exactly
  what `cargo install axiom_engine@X.Y.Z` compiles. Provenance break (2026-10):
  crates.io `axiom_engine` 0.4.2 was published from a post-`v0.4.2`-tag `main`
  commit (it contains PR #176 surprise-triage and PR #190 vendored
  `mesh_core`, neither of which is in the `v0.4.2` tag; the tag's
  `config.rs` also points at a different checkpoint URL). Do not repeat this:
  if a publish must run from a non-tagged commit, bump the version first so
  the registry artifact always matches its tag.

- crates.io enforces a package-size limit; `Cargo.toml`'s `exclude` keeps
  checkpoints/models/`target` out (dry-run packages ~1.3 MiB).
- `cargo install axiom_engine` compiles candle on the user's machine (slower,
  needs a toolchain) — that is expected; pip/Releases are the no-compile paths.
- The wheel carries a precompiled binary, so it is `py3-none-<platform>` (any
  Python 3), one wheel per OS/arch.
