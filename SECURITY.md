# Security policy

## Reporting a vulnerability

Report privately through
[GitHub security advisories](https://github.com/fernandogarzaaa/AXIOM-AETHER/security/advisories/new).
Please do not open a public issue for a vulnerability.

Include what you would need to reproduce it yourself: version or commit,
the component involved (inference runtime, training loop, API server),
and a minimal case. You should get an initial response within a week.

## Supported versions

AXIOM-AETHER is pre-release software with no stable release yet. Only
the latest commit on the default branch receives security fixes.

## Scope notes

A few things about AXIOM-AETHER's design are worth knowing before
reporting:

- **The runtime executes model code locally.** AXIOM-AETHER loads model
  weights and runs inference plus online test-time training on your
  machine. Only load weights from sources you trust; a malicious weight
  file or training callback can execute arbitrary code during load.
- **Training mutates the model.** The online test-time training loop
  updates weights from observed data. Do not train on untrusted or
  sensitive data you cannot afford to have reflected in the model.
- **The API server is a local operator surface.** It assumes a trusted
  local user; do not expose it to a network.
- **Native builds compile Rust code.** Building from source runs the
  Rust toolchain on this repository's code. Review what you build, and
  prefer the prebuilt artifacts when they are available.
