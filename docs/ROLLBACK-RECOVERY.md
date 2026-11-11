# Rollback and recovery runbook

Audit section 7, Phase 5. A release is not qualified until a rollback
has been drilled and the evidence recorded. This runbook defines the
three rollback paths and what each one must prove. One local drill has
been run: `docs/release-qualification/drills/2026-11-11-local-rollback.md`.
No cluster drill has been run yet. The drill evidence template is
`docs/release-qualification/rollback-drill-evidence.md`.

## When to roll back

- The readiness gate never clears after install (checkpoint hash
  mismatch, bad image, bad config).
- `/readyz` flaps or the error rate jumps after a release, and the
  cause is not yet known.
- A bad checkpoint or tokenizer was published and pods boot with it.
- Gossip spreads a bad learned state across the fleet.

Roll back first, diagnose second. A rollback is not an admission of a
root cause. It is the safe state while the cause is found.

## Path 1: Helm rollback

Restores the previous chart revision, which restores the previous
image, config, and values.

```sh
helm history <release> -n <namespace>
helm rollback <release> <revision> -n <namespace> --wait --timeout 10m
kubectl -n <namespace> rollout status deploy/<release>
```

What it must prove: the previous revision number, the image digest
the pods run after rollback, all pods ready, `/readyz` 200 on every
pod, `helm test` passing. The old revision stays available because
Helm keeps release history. Do not run `helm uninstall` as a rollback.
Uninstall deletes the history the rollback needs.

## Path 2: checkpoint re-install via the atomic installer

Use this when the code is fine but the checkpoint or tokenizer is bad.
The container entrypoint (`scripts/docker_entrypoint.sh`) installs
checkpoints atomically. Every download goes to a temp file in the same
directory, is verified against its pinned SHA-256 there, and is
renamed onto the final path only on success. A failed or interrupted
download never leaves a partial file under the final name. Note the
entrypoint deletes an installed file whose hash mismatches the pinned
one before it re-downloads, so a refused download leaves no checkpoint
file at all. The rollback must re-install a good checkpoint after a
refused attempt. It cannot rely on the previous file surviving one.

```sh
helm upgrade <release> deploy/helm/axiom -n <namespace> \
  -f deploy/helm/axiom/values-<staging|prod>.yaml \
  --set checkpoint.url=<previous good checkpoint URL> \
  --set checkpoint.tokenizerUrl=<previous good tokenizer URL> \
  --set checkpoint.sha256=<previous good sha256> \
  --set checkpoint.tokenizerSha256=<previous good sha256> \
  --reuse-values --wait --timeout 10m
```

What it must prove: the previous good hashes, the entrypoint log
lines showing the verified install, all pods ready, `/readyz` 200.
If the bad checkpoint is already on disk with a matching hash, the
entrypoint skips the download. A rollback to different hashes forces
a re-download, which is the behavior the drill checks.

Two checkpoint failure modes behave differently, measured in the
2026-11-11 local drill. A hash-mismatched but parseable checkpoint
lets the server start, and the readiness gate holds it out with
`/readyz` 503 and reason mismatched. A corrupt (unparseable)
checkpoint fails server boot outright with a non-zero exit and no
listener, so there is no `/readyz` 503 to observe. The rollback is
the same in both cases: re-install the previous good checkpoint
through the entrypoint and restart. Do not expect a 503 from a
corrupt file. Expect a crash loop, and read the boot log.

## Path 3: config revert

Use this when a config or secret change broke serving and the image
and checkpoint are good. Revert the values change in git, then
re-apply.

```sh
git revert <config commit>
helm upgrade <release> deploy/helm/axiom -n <namespace> \
  -f deploy/helm/axiom/values-<staging|prod>.yaml --reuse-values \
  --set checkpoint.sha256=<deployed checkpoint sha256> \
  --set checkpoint.tokenizerSha256=<deployed tokenizer sha256> \
  --wait --timeout 10m
```

The `--set` hashes are not optional. The overlay files carry
placeholder hashes, and values from `-f` take priority over
`--reuse-values`, so a revert without them replaces the deployed
hashes with placeholders. The entrypoint then rejects the checkpoint
download and the replacement pods never start. Capture the deployed
hashes first with `helm get values <release> -n <namespace>` and
pass them on the upgrade command. A config-only override file that
does not mention checkpoint settings works too.

Config changes roll the pods because the Deployment carries a
`checksum/config` annotation. What it must prove: the reverted commit,
the rendered ConfigMap diff, all pods ready, `/readyz` 200. Secrets
are reverted the same way, but through the out-of-band Secret, never
through a values file. A secret revert must not commit the old secret
value to git.

## Recovery after a crash

- A pod crash loses only per-pod local state (heal memory, TTT
  sessions). The fleet re-converges through the gossip CronJob, which
  runs every 5 minutes. After recovery, confirm the gossip job
  succeeds at least once before calling the fleet converged.
- A checkpoint on disk survives a pod restart because it lives on the
  `checkpoints` emptyDir volume of the pod. A node loss loses it, and
  the entrypoint re-downloads it on the new node. That is the designed
  behavior. It is not data loss.
- Multi-file rollback inside the runtime (agent patch application) is
  a separate mechanism from deploy rollback. It is covered by the
  Phase 1 fix and its fault-injection tests, not by this runbook.

## What a drill must record

Every drill fills in
`docs/release-qualification/rollback-drill-evidence.md` in full.
Pre-state, trigger, every command run with its output, post-state
verification, and timings. A drill with an empty evidence template
did not happen. The drill script `scripts/rollback_drill.sh` runs
the checks in order and prints the evidence block. It refuses to run
against a release that does not exist, and `--dry-run` prints the
commands without running them. Cluster mode exits non-zero when the
rollback or any post-state check fails, so a failed recovery never
reports as a passing drill.

## Local drill (no cluster)

`scripts/rollback_drill.sh --local` runs the three rollback paths
end to end without a cluster, a container runtime, or a registry.
It uses the real Helm binary (`helm lint` and `helm template` on
both overlays), the real container entrypoint (atomic checkpoint
install against a local file server), and the real server binary as
a local process (the `/readyz` gate). It needs a built server
binary (`cargo build --release --bin axiom_engine`). If Helm is not
on PATH it downloads the pinned Helm release into the drill work
dir and verifies its SHA-256 first.

What the local drill exercises: Helm lint and template rendering
with the readiness gate on and the image digest pinned, atomic
checkpoint install plus wrong-hash refusal plus skip-when-present,
`/readyz` 200 on the good checkpoint, `/readyz` 503 on a
hash-mismatched checkpoint, `/readyz` 200 again after re-installing
the good checkpoint, boot failure on a corrupt checkpoint, and
config revert at the render layer (a changed value alters the
ConfigMap and the `checksum/config` annotation, reverting it
restores the baseline render byte for byte).

What it does NOT exercise: `helm rollback`, `kubectl rollout
status`, `helm test`, pod readiness, the gossip CronJob, and
multi-replica behavior. None of those exist without a cluster.
A cluster drill is still required before a release is qualified.
The local drill is a rehearsal of the mechanics, not a substitute.

The local drill sets `AXIOM_PRODUCTION_BPE=1`, exactly as the
chart does through the ConfigMap. Without that flag the server
ignores `AXIOM_BPE_CKPT` and boots the legacy model, while the
readiness gate hashes a different path. A manual server run that
omits the flag is misleading. Always copy the chart env, not just
the checkpoint paths.

## Drill findings (follow-ups, not silent fixes)

Findings from the 2026-11-11 local drill. Each one is a follow-up
item. None of them changed runtime code in this PR.

1. No live cluster exists in the sandbox, so the cluster drill
   (Helm rollback, `helm test`, gossip CronJob) is still unproven.
   It needs a kind or minikube environment and a real release.
2. The server silently ignores `AXIOM_BPE_CKPT` unless
   `AXIOM_PRODUCTION_BPE=1` is also set, while the readiness gate
   hashes the default checkpoint path instead. In-cluster behavior
   is consistent because the chart sets the flag. A manual run
   without it can report a verified checkpoint for a model that is
   not serving. Consider failing closed or warning when the gate is
   on, the flag is unset, and `AXIOM_BPE_CKPT` is set.
3. A corrupt checkpoint fails boot instead of yielding `/readyz`
   503. Operators should read the boot log on a crash loop, not
   wait for a readiness signal that never arrives. The runbook now
   states this in Path 2.
4. The ServiceMonitor cannot scrape `/metrics` once `AXIOM_API_KEY`
   is set. The server requires `X-Axiom-Key` on `/metrics`, and the
   chart's ServiceMonitor has no way to present that header, so
   scrapes get 401. The overlays enable the ServiceMonitor while
   the checklist requires the key, so a deploy that follows the
   guide has no working metrics gating. Fixing it needs either
   server support for an `Authorization` bearer on `/metrics` or a
   scrape path that can send the header. It is not a chart-only
   fix, so it stays a follow-up. The deploy guide no longer claims
   metrics has no auth.
5. The Path 3 revert command as first written would have replaced
   the deployed checkpoint hashes with the overlay placeholders,
   because `-f` values take priority over `--reuse-values`. Found
   by Codex review, fixed in this PR: the command now re-applies
   the deployed hashes, and the runbook says how to capture them.
