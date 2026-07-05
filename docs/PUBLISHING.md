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

## How the binary set is controlled

The crate defines two user-facing bins (`axiom`, `axiom_engine`) from the same
entrypoint, and gates its ~10 dev/training/eval bins behind the **`tools`**
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

- **crates.io:** create an API token and add it as the repo secret
  `CARGO_REGISTRY_TOKEN`. Verify the crate name `axiom_engine` is available (or
  pick another and update `Cargo.toml`).
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

- crates.io enforces a package-size limit; `Cargo.toml`'s `exclude` keeps
  checkpoints/models/`target` out (dry-run packages ~1.3 MiB).
- `cargo install axiom_engine` compiles candle on the user's machine (slower,
  needs a toolchain) — that is expected; pip/Releases are the no-compile paths.
- The wheel carries a precompiled binary, so it is `py3-none-<platform>` (any
  Python 3), one wheel per OS/arch.
