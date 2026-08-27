# AXIOM-AETHER — Security Audit (2026-08-23, updated same day — round 2)

Scope: every path in the repository that executes a subprocess, writes to
the filesystem outside its own scratch space, accepts network input, or
fetches/loads external artifacts. Written treating autonomous execution as
hostile by default, per the mission brief. Findings 1, 2, and 3 were fixed
across two passes on the same audit (marked **FIXED**, with which pass
noted); the rest are documented findings with a recommended disposition,
not code changes, because they are either already adequately mitigated, or
the fix requires a product decision this audit shouldn't make unilaterally.

## Summary table

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | `POST /v1/hypervisor/jit_run` — unauthenticated-by-default arbitrary command execution | **P0** | **FIXED** (pass 1) |
| 2 | Checkpoint downloads had no integrity verification | **P1** | **FIXED** (pass 1) |
| 3 | `sandbox.rs` naming implies a security boundary it doesn't provide | P2 | **FIXED** (pass 2) — renamed to `compile_verify.rs`/`CompileVerifier` |
| 4 | No execution isolation anywhere (self_heal, solve, poly_jit, compile_verify all run real subprocesses) | P1 (architectural) | Documented, not built — see below |
| 5 | Data-plane bind defaults to `0.0.0.0` with auth opt-in | P1, already flagged and partly mitigated (2026-07-24 audit) | Documented, reinforced |
| 6 | Patch-memory trust model | — | **Strength**, not a finding |
| 7 | No formal capability model | P2 | Documented, scoped recommendation below |

---

## 1. `POST /v1/hypervisor/jit_run` — arbitrary command execution (P0, FIXED)

### What it was

`routes_hypervisor.rs::hypervisor_jit_run` builds a `PolyJitRunRequest` from
the **caller-supplied JSON body** — `command: String`, `args: Vec<String>`,
`working_dir: Option<String>`, `source_path: Option<String>` — and passes it
straight to `PolyJitEngine::run_with_feedback`, which calls
`tokio::process::Command::new(&request.command).args(&request.args)` with no
allowlist, no argument validation, and no sandboxing. On failure it also
overwrites `source_path` on disk with a synthesized or LLM-proposed patch.
This is not a hypothetical: the route is a real, always-registered Axum
handler at `POST /v1/hypervisor/jit_run`.

The route sat behind `AXIOM_API_KEY` (the general data-plane guard, opt-in
and unset by default — see finding 5), but nothing gated the specific
**capability** of remote command execution independently of that general
auth. A server started with zero configuration — the documented quick-start
path — exposed this route on whatever interface it bound (default
`0.0.0.0`, see finding 5) with no key required at all.

### Why this is P0

- Reachable over the network by default configuration, not just from a
  local trusted caller.
- No allowlist: `command` can be `sh`, `curl`, `rm`, anything on `$PATH`.
- Runs with the full privileges of the Axiom server process.
- Can also overwrite an arbitrary file the caller names as `source_path`.
- The mission brief's own framing applies directly: *"Do not introduce a
  dangerous execution capability without an explicit policy boundary."*
  None existed.

### Fix

A dedicated capability flag, independent of `AXIOM_API_KEY`:

- `AppState.jit_exec_enabled: bool`, **defaults to `false`**
  (`prelude_state.rs`).
- `run.rs` reads `AXIOM_ENABLE_JIT_EXEC` (`1`/`true`/`on`) at startup; when
  enabled without `AXIOM_API_KEY` also set, logs an explicit `WARNING` that
  process execution is exposed with no auth.
- `hypervisor_jit_run` checks the flag first and returns `403 Forbidden`
  with a message naming the env var, before touching the request body at
  all, when the capability isn't enabled.
- Regression test: `jit_run_endpoint_is_disabled_by_default` (asserts `403`
  and that the body names `AXIOM_ENABLE_JIT_EXEC`); the two existing
  `jit_run_*` tests were updated to explicitly opt in
  (`.with_jit_exec_enabled(true)`), which is itself documentation — a test
  exercising a dangerous capability now has to say so.

This is a **capability gate**, in the vocabulary the mission brief asks for
(`process.execute` + `filesystem.write`), scoped to exactly the one route
that needs it. It is intentionally not a broader capability *system*
(roles, policies, per-key scopes) — see finding 7 for why that's future work,
not this fix.

### What this does NOT fix

Once `AXIOM_ENABLE_JIT_EXEC=1` is set, the route is exactly as unrestricted
as before for anyone who can call it — there is still no command allowlist,
no argument sanitization, no resource limits, no process isolation. The gate
answers "should this capability exist on this server at all," not "is this
particular command call safe." An operator who enables it is expected to
also set `AXIOM_API_KEY` and to understand they're exposing a repair loop
that executes what it's told. This is documented in the README's new
"Process execution capability" section and should be treated as the honest
ceiling of what this fix does.

---

## 2. Checkpoint download integrity (P1, FIXED)

`config.rs::fetch_base_model` downloads a checkpoint or tokenizer file from
an operator-configured URL (`AXIOM_CHECKPOINT_URL`/`AXIOM_TOKENIZER_URL`, or
`cfg.models.base_model_url` from `axiom init`'s TOML config) over
`reqwest::blocking::get` and, before this pass, wrote it straight to the
target path with **no integrity check at all**. The file is then loaded
directly into the runtime as the model's weights.

This is a supply-chain gap, not a remote-attacker one — `AXIOM_CHECKPOINT_URL`
is operator-configured, not attacker-controlled in normal operation — but a
compromised release mirror, a MITM on a plain-`http://` URL, or a simple
copy/paste of the wrong URL would install silently with no signal to the
operator that anything was wrong.

**Fix**: `fetch_base_model` now hashes the download while streaming
(SHA-256) and, when `AXIOM_CHECKPOINT_SHA256`/`AXIOM_TOKENIZER_SHA256` is
set, verifies before the file is renamed into its final path — a mismatch
deletes the partial download and returns `Err` (propagated as a real
`axiom init` failure, not swallowed as best-effort). When no hash is pinned
(the default — this stays fully backward compatible), the computed digest is
printed so an operator can capture and pin it for next time. Unit-tested in
isolation (`verify_checksum_*`, three tests, no network required).

**Update (round 2)**: `release.yml`'s `checkpoint` job already publishes a
`SHA256SUMS.txt` release asset for every checkpoint/tokenizer/drift-gate
file (`sha256sum ckpt_release/* > ckpt_release/SHA256SUMS.txt`) — round 1
of this audit missed that and incorrectly recommended it as future work.
That manifest is the "maintainer-verified source to pin from" this fix
needed; it already existed. What round 1's fix *didn't* cover: the Docker
container entrypoint (`scripts/docker_entrypoint.sh`) fetches checkpoints
via a separate, shell-based path — not `config.rs::fetch_base_model` at
all — and had zero verification. Fixed in round 2: the same
`AXIOM_CHECKPOINT_SHA256`/`AXIOM_TOKENIZER_SHA256` env vars now verify
there too, fail-closed on mismatch, deleting the partial download.

**Residual risk, still not fixed**: this is opt-in verification.
`AXIOM_CHECKPOINT_URL` itself remains untrusted-input-shaped (an
operator-supplied string, fetched over whatever scheme it specifies —
nothing forces HTTPS). `SHA256SUMS.txt` itself is **unsigned** — SHA-256
alone protects against corruption/substitution once a hash is pinned by an
out-of-band-verified source, not against a compromised release pipeline
publishing a new "correct" hash alongside a bad binary. Real GPG/sigstore
signing needs key material and CI secret configuration this audit pass
doesn't have access to set up; tracked in [ROADMAP.md](ROADMAP.md) as a
named, deliberately-not-implemented gap rather than something faked with a
placeholder key.

---

## 3. `sandbox.rs` naming (P2, FIXED in pass 2)

`SandboxController` (`sandbox.rs`) was, by its own module doc, honest about
what it did: *"a closed-loop local compilation sandbox... creates an isolated
temp Cargo package, runs compiler checks... The sandbox never writes into the
real repository."* Mechanically, it ran `cargo check --message-format=json`
against a synthesized `Cargo.toml` with **zero external dependencies and no
`build.rs`** — no proc-macros, no network access during the check, no
arbitrary code actually executes (`cargo check` type-checks; it doesn't run
`main` or tests). In this exact configuration, the practical risk was low.

The finding was naming, not behavior: `AXIOM_SANDBOX_VERIFY`/
`AXIOM_SANDBOX_ROOT`/`AXIOM_SANDBOX_MAX_STEPS` and the type name
`SandboxController` read, to an operator skimming env vars, like they imply
a process/OS security boundary (seccomp, a container, a VM) — they don't.
The mission brief is explicit: *"Do not call something a 'sandbox' unless
it provides an actual security boundary... rename it appropriately unless
you can safely implement real isolation."*

**Fixed**: `sandbox.rs` → `compile_verify.rs`, `SandboxController` →
`CompileVerifier`, `SandboxDiagnostic` → `CompileDiagnostic`,
`SandboxRunReport`/`SandboxError` → `CompileVerifyReport`/
`CompileVerifyError`. The module doc now leads with *"This is not a process
isolation boundary"* instead of three sentences in. Env vars renamed to
`AXIOM_COMPILE_VERIFY[_ROOT|_MAX_STEPS|_TIMEOUT_SECS]`, with the old
`AXIOM_SANDBOX_*` names still accepted for one release as deprecated
aliases (a warning is logged when an alias is used) — checked before
renaming that these vars were undocumented anywhere in README, scripts, or
config, so the actual blast radius was two source files, not a public
config surface; the alias path is defensive insurance, not a response to a
known dependency. Regression test:
`deprecated_env_alias_is_used_when_new_name_unset` (also asserts the new
name wins when both are set).

---

## 4. No execution isolation anywhere (P1, architectural, documented)

Every subprocess this codebase runs — `self_heal.rs`'s verify commands,
`solve.rs`'s repair-loop re-verification, `poly_jit.rs`'s
`run_process`/`run_with_feedback`, `compile_verify.rs`'s `cargo check` — executes as
a **real child process of the Axiom server or CLI**, with that process's own
filesystem and network access. There is no container, no seccomp/AppArmor
profile, no chroot, no resource cgroup, anywhere in this codebase.

This is not a new finding this audit invented — it's the honest state of
"autonomous execution" across the whole repository, and it's why finding 1
(the network-reachable, ungated version of this) was P0 while this general
architectural fact is P1: the mitigating factor everywhere *except* the
now-fixed `jit_run` route is that the caller is either (a) the local
operator, invoking `axiom run`/`axiom solve` on their own machine against
their own repo — the same trust model as running `npm test` yourself — or
(b) gated behind the new `AXIOM_ENABLE_JIT_EXEC` capability flag.

**What is done well despite the lack of isolation**: every repair path is
verify-gated and reversible (§3.5 of ARCHITECTURE-AUDIT.md) — a bad patch
doesn't survive, and a bad *heal* (e.g. an unwanted directory) is additive
and non-destructive by construction (heals only ever create a directory or
add an executable bit; nothing in `self_heal.rs` deletes or overwrites). That
containment is real and valuable, but it is a **correctness/reversibility**
property, not a **security isolation** property, and this audit does not
conflate the two.

**Recommendation** (ROADMAP.md, not done this pass — a real sandbox
integration, e.g. a container runtime or a `landlock`/seccomp profile for
the child process, is a substantial feature, not an audit fix): define the
capability vocabulary explicitly (`process.execute`, `filesystem.write`,
`filesystem.read`, `network.request`, `package.install`, `git.write`,
`memory.write`, `model.route` — exactly the list the mission brief proposes)
and make every capability-touching entry point declare which ones it needs,
even before real OS-level isolation exists. `AXIOM_ENABLE_JIT_EXEC` (finding
1's fix) is the first instance of that pattern in the codebase; it should be
the template, not a one-off.

---

## 5. Data-plane default bind and auth (P1, prior finding, reinforced)

Already identified in `docs/AUDIT_2026-07.md` and partly mitigated then
(`AXIOM_API_KEY`, opt-in, off by default; a startup warning when binding off
loopback without it). Reinforced here because finding 1 shows why "one
opt-in key gates everything" is an incomplete model once one of "everything"
is a process-execution capability and not just a data-plane read/write: the
severity of being unauthenticated is not uniform across routes.
**No further code change recommended here beyond finding 1's fix** — the
existing `AXIOM_API_KEY` mechanism is sound for what it protects; the gap
was capability-blindness, not authentication-blindness, and that's now
closed for the one route where it mattered most.

---

## 6. Patch-memory trust model (strength, not a finding)

`patch_memory.rs`'s own doc comment states the invariant plainly: *"A patch
received from a peer is NEVER applied on trust."* Verified by reading
`PatchMemory::try_candidates` and the fleet-merge path
(`routes_verify.rs`/`post_patches_merge`): every candidate — local or
peer-sourced — is written, then re-verified by **this node's own** verify
check, and kept only if it passes. This is the correct design for exactly
the threat this audit is otherwise concerned with (memory poisoning via a
compromised or malicious peer): a peer cannot get code executed on another
node merely by claiming a fix works. Called out explicitly here because a
security audit that only lists problems is as misleading as one that only
lists strengths — this is architecture worth preserving as new peer-facing
features are added, not something to "improve."

---

## 7. Toward a formal capability model (P2, scoped recommendation)

The mission brief asks for a capability model: `filesystem.read`,
`filesystem.write`, `process.execute`, `network.request`, `package.install`,
`git.write`, `memory.write`, `model.route`. This audit does **not**
implement that system wholesale — doing so credibly (a policy engine,
per-caller scopes, audit logging of capability grants) is a multi-week
feature, not an audit-pass fix, and building it speculatively without a real
second or third capability-gated route to generalize from would be exactly
the premature abstraction the mission brief warns against.

What this audit *does* establish, concretely, as the seed:

- `AXIOM_ENABLE_JIT_EXEC` is now the first explicit, named, off-by-default
  capability gate in the codebase, for `process.execute` (+ implicit
  `filesystem.write` for the patch path).
- `AXIOM_API_KEY` is an authentication gate (who), not a capability gate
  (what) — the two are orthogonal and this pass keeps them that way rather
  than conflating them.
- The next capability-shaped surface to gate the same way, in priority
  order, is `network.request` via `poly_jit`'s `command`/`args` (a caller
  who can already run `curl` through the now-gated route effectively has
  `network.request` too — gating `jit_exec` transitively covers this one for
  now) and `git.write` if/when any endpoint gains the ability to commit or
  push (none currently does — checked: no route calls `git` as a
  subprocess).

Recommendation for the next pass that touches this: promote the pattern
(`AppState.<capability>_enabled: bool`, read from a dedicated env var,
checked first in the handler, tested with an explicit "disabled by default"
regression test) to every future route that executes, writes outside a
scoped store, or reaches the network on the caller's behalf — without
building a generic policy engine until there are enough instances of the
pattern to generalize from honestly.
