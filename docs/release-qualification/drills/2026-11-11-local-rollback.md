# Rollback drill evidence: local drill

Local-mode run of `scripts/rollback_drill.sh --local`. No cluster, no
container runtime, no registry existed in this sandbox, so the drill ran
against the real Helm binary (lint and template), the real container
entrypoint (atomic checkpoint install), and the real server binary as a
local process (the /readyz gate). Cluster steps (helm rollback, kubectl
rollout status, helm test, gossip CronJob) were NOT exercised.

## Drill identity

- Date and time (UTC): 2026-11-11T08:07:07Z to 2026-11-11T08:07:17Z
- Operator: rollback drill script (local mode)
- Release name and namespace: axiom / axiom-staging (rendered only; no release exists)
- Chart revision before the drill: n/a (no cluster)
- Chart revision rolled back to: n/a (no cluster)
- Rollback path used: checkpoint re-install (local process) plus config revert (render layer)
- Environment: local process drill, no live cluster

## Pre-state

- Server binary: /home/hatch/workspace/axiom-aether-phase5-item4/axiom_engine_rs/target/release/axiom_engine
- Helm: v3.18.4+gd80839c
- v1 checkpoint sha256: 411a485216e432ece6b9af94fa32154cf79a2a56d4f81266baa50063f45092bd
- v1 tokenizer sha256: ee2ec7c52d81ba38ae4a6d66428a64779c7b6d12362868abb82af575674325c9
- v2 checkpoint sha256: a7d901b2971ddd71bf71ccc985632d624a3301234cc50e7650ca0138ccae03a6
- helm lint: pass (see phase 1)
- Staging and prod overlays render with the readiness gate on, the image
  digest pinned, staging CORS at exactly one origin, prod CORS deny-all.

## Trigger

- What was broken, in one sentence: a bad checkpoint release (v2 bytes) was installed while the pinned hash still named v1.
- How the break was introduced (bad image, bad checkpoint, bad config, simulated): simulated, by copying v2 bytes over the installed checkpoint.
- Time the break was introduced (UTC): 2026-11-11T08:07:10Z
- Time the break was detected (UTC): 2026-11-11T08:07:12Z
- Detection method (probe, metrics, manual check): /readyz probe, 503 with reason mismatched.

## Rollback steps (commands run, in order)

1. Command: helm lint deploy/helm/axiom; helm template for staging and prod overlays with the drill hashes.
   Output: lint passes; both renders succeed; gate on, image digest pinned, CORS as above.
2. Command: entrypoint install of v1 (AXIOM_CHECKPOINT_URL=<local file server>/v1.bin, pinned sha256, AXIOM_ENTRYPOINT_NOEXEC=1).
   Output: v1 installed with matching hash; a wrong-hash attempt refused and left nothing under the final name (the entrypoint deletes the installed file before re-downloading, so v1 was re-installed after); a repeat install skipped without re-download.
3. Command: start server with v1 installed, AXIOM_PRODUCTION_BPE=1, gate on, pinned hashes; curl /readyz.
   Output: /readyz 200; /healthz 200.
4. Command: copy v2 bytes over the checkpoint, restart the server, curl /readyz.
   Output: /readyz 503 with the mismatch reason; /healthz 200 (process alive, gate holding it out).
5. Command: entrypoint re-install of v1, restart the server, curl /readyz.
   Output: v1 re-installed; /readyz 200.
6. Command: write unparseable bytes over the checkpoint, start the server.
   Output: server exits non-zero (exit 1) with no listener; v1 restored afterwards.
7. Command: change the staging CORS origin in a copy of values-staging.yaml, render, revert, render again.
   Output: the bad render differs in the ConfigMap and checksum/config annotation; the reverted render matches the baseline byte for byte.

## Post-state verification

- Pods running and ready after rollback: n/a (no cluster); the local server process reports /readyz 200 after rollback.
- Image digest on the pods after rollback: n/a (no cluster); the rendered staging image is ghcr.io/fernandogarzaaa/axiom-aether:latest@sha256:844f485adf925e515070a1680630eaefe4423a54ea8cbc971b045e6acb5d8afe.
- Checkpoint SHA-256 after rollback: 411a485216e432ece6b9af94fa32154cf79a2a56d4f81266baa50063f45092bd (matches the pinned v1 hash).
- /readyz result per pod after rollback: 200 on the local process.
- helm test result after rollback: NOT exercised (needs a cluster).
- Gossip CronJob result after rollback: NOT exercised (needs a cluster).
- Error rate or smoke request result after rollback: /healthz 200, /readyz 200.

## Timings

- Trigger to detection: 2s (restart plus /readyz poll)
- Detection to rollback start: 0s
- Rollback start to all pods ready: 2s (local process /readyz 200; no pods exist)
- Rollback start to /readyz 200 on every pod: 2s (one local process)
- Total time from trigger to verified recovery: 4s
- Whole drill wall clock: 10s

## Verdict

- Rollback succeeded: yes
- Recovery verified by the post-state checks above: yes
- Gaps found during the drill (each one becomes a follow-up item, not a silent fix):
  - No live cluster, so helm rollback, kubectl rollout status, helm test, and the gossip CronJob were not exercised. A cluster drill is still required before a release is qualified.
  - The drill needs AXIOM_PRODUCTION_BPE=1 for the server to load AXIOM_BPE_CKPT at all; with the flag unset the server silently boots the legacy model while the gate hashes a different path. The chart sets the flag, so in-cluster behavior is consistent, but a manual run without it is misleading.
  - A corrupt (unparseable) checkpoint fails server boot outright instead of yielding /readyz 503. The runbook now says so. The 503 path covers hash-mismatched but parseable files.
- Follow-up issue numbers: none filed as GitHub issues; the three gaps above are named as follow-up items in docs/ROLLBACK-RECOVERY.md and in the PR.

## Drill result

- pass=21 fail=0
