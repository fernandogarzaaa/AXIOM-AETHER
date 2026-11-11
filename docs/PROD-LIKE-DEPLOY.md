# Production-like deployment

How to deploy AXIOM-AETHER to a production-like environment with the
Phase 2 Helm work: authenticated gossip, pinned images, atomic
checkpoints, the readiness gate, and restricted CORS. Nothing here
deploys anything by itself. Each step names its check.

## Before you start

- Pick a release tag. The tag and its image digest come from the
  release notes. The digest in `values.yaml` is the pinned digest of
  the `latest` image as of 2026-11-09. If you deploy a different tag,
  resolve its digest from GHCR first and set both `--set
  image.tag=<tag>` and `--set image.digest=<digest>`. A tag without
  its digest is a failed deploy, not a shortcut.
- Pick a namespace. Staging and prod never share one.
- Have the checkpoint and tokenizer release asset URLs plus their
  SHA-256 hashes. They come from the release `SHA256SUMS.txt`. The
  hashes are what the readiness gate verifies.

## Pre-flight checklist

1. `helm lint deploy/helm/axiom` passes.
2. `./scripts/check-pinned-images.sh` passes.
3. `helm template` renders with your overlay and `--set` values. Read
   the rendered Deployment. Confirm the image line ends in
   `@sha256:...`, the `AXIOM_REQUIRE_VERIFIED_CHECKPOINT` env is `1`,
   and no Secret in the render carries a real key.
4. Create the secrets Secret first, out of band. Example shape only,
   with placeholder values. Generate real keys with
   `openssl rand -hex 32`.
   ```sh
   kubectl -n <namespace> create secret generic <existingSecret> \
     --from-literal=AXIOM_API_KEY=<generated> \
     --from-literal=AXIOM_FLEET_KEY=<generated>
   ```
   The overlay files name the Secret. They never contain the keys.
   `existingSecret` must match the Secret you created, or the pods
   fail to start.
5. Confirm the CORS allowlist. Prod denies all origins
   (`AXIOM_CORS_ORIGINS` empty). Staging allows exactly one named
   origin. `*` is never a valid value. The server ignores a wildcard
   and logs it.

## Install

```sh
helm install <release> deploy/helm/axiom \
  -n <namespace> --create-namespace \
  -f deploy/helm/axiom/values-<staging|prod>.yaml \
  --set checkpoint.url=<release checkpoint URL> \
  --set checkpoint.tokenizerUrl=<release tokenizer URL> \
  --set checkpoint.sha256=<sha256> \
  --set checkpoint.tokenizerSha256=<sha256> \
  --set image.tag=<release tag> \
  --set image.digest=<image digest> \
  --wait --timeout 10m
```

`--wait` blocks until the Deployment reports available. That is the
first readiness signal, not the last.

## Readiness-gate expectations

The readiness gate is `config.requireVerifiedCheckpoint: "1"` in both
overlays. It changes what `/readyz` means.

- `/healthz` answers 200 once the process is up. It proves the
  process runs. It proves nothing about the model.
- `/readyz` answers 200 only when the production checkpoint file
  exists, the tokenizer is present, and the checkpoint SHA-256
  matches `checkpoint.sha256`. When `checkpoint.tokenizerSha256` is
  pinned, the tokenizer hash must match too. Anything else is a 503,
  and the pod stays out of the Service endpoints.
- The startup probe allows up to 5 minutes for checkpoint download
  and model load. A pod that is still not ready after that is a failed
  deploy, not a slow one. Check the entrypoint logs for a hash
  mismatch before re-running the install.
- After install, verify readiness directly. All three pods in prod
  must report ready. One ready pod out of three is not a pass.
  ```sh
  kubectl -n <namespace> get pods -l app.kubernetes.io/name=axiom
  kubectl -n <namespace> exec deploy/<release> -- curl -fsS localhost:8080/readyz
  ```
- `helm test <release> -n <namespace>` runs the chart connectivity
  test against `/readyz` and `/v1/models`. It presents `X-Axiom-Key`
  when the API key is set. It must pass before the deploy counts as
  successful.

## Metrics-gating expectations

- `/metrics` is scraped on the API Service when
  `metrics.serviceMonitor.enabled` is true (both overlays). Confirm
  the ServiceMonitor exists and Prometheus reports the target up.
- After the pods are ready, confirm these signals exist with sane
  values: total tokens prefilled, active and quantized session gauges,
  prefill latency histogram, session residency. These are the four
  signals the server exposes today. Per-stage latency, memory gauges,
  first-token latency, and stream backpressure signals arrive with
  the Phase 3 observability work. A deploy checklist that claims
  those signals exist today is wrong.
- Gossip health is a deploy signal too. The gossip CronJob runs every
  5 minutes and fans signed immunity exports across the fleet. After
  10 minutes, check the last CronJob run succeeded. A failing gossip
  job does not block serving, but it blocks calling the fleet
  converged.
- Metrics access has no auth of its own in the chart. Do not expose
  `/metrics` outside the cluster. If the environment needs it, put it
  behind the same network policy or ingress auth as the data plane.

## Rollback

See `docs/ROLLBACK-RECOVERY.md`. The short form: `helm rollback
<release> <revision> -n <namespace>` restores the previous chart
revision, and the checkpoint re-installs atomically through the
entrypoint. Record the drill evidence. A rollback with no evidence
template filled in did not happen.
