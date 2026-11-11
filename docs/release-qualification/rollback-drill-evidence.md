# Rollback drill evidence template

Fill this in for every rollback drill. A drill with a blank field is
incomplete. Copy this file to
`docs/release-qualification/drills/<date>-<release>-rollback.md` and
fill in the copy. Never edit this template in place with drill data.

## Drill identity

- Date and time (UTC):
- Operator:
- Release name and namespace:
- Chart revision before the drill:
- Chart revision rolled back to:
- Rollback path used: Helm rollback / checkpoint re-install / config revert (pick one)
- Environment: staging / prod-like local cluster (pick one; no drill runs against real prod)

## Pre-state (recorded before the trigger)

- Pods running and ready (paste `kubectl get pods` output):
- Image digest on the running pods:
- Checkpoint SHA-256 on the running pods:
- `/readyz` result per pod:
- `helm test` result before the drill:
- Gossip CronJob last run result:

## Trigger

- What was broken, in one sentence:
- How the break was introduced (bad image, bad checkpoint, bad config, simulated):
- Time the break was introduced (UTC):
- Time the break was detected (UTC):
- Detection method (probe, metrics, manual check):

## Rollback steps (commands run, in order)

Paste every command and its output. Do not summarize. Include the
`helm history` output, the rollback or upgrade command, and the
rollout status.

1. Command:
   Output:
2. Command:
   Output:
3. Command:
   Output:

## Post-state verification

- Pods running and ready after rollback (paste `kubectl get pods` output):
- Image digest on the pods after rollback (must match the previous good revision):
- Checkpoint SHA-256 after rollback:
- `/readyz` result per pod after rollback (must be 200 on every pod):
- `helm test` result after rollback (must pass):
- Gossip CronJob result after rollback (must succeed at least once):
- Error rate or smoke request result after rollback:

## Timings

- Trigger to detection:
- Detection to rollback start:
- Rollback start to all pods ready:
- Rollback start to `/readyz` 200 on every pod:
- Total time from trigger to verified recovery:

## Verdict

- Rollback succeeded: yes / no
- Recovery verified by the post-state checks above: yes / no
- Gaps found during the drill (each one becomes a follow-up item, not a silent fix):
- Follow-up issue numbers:
