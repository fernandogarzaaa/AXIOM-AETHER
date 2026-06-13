# Deploying Axiom on Kubernetes

Axiom ships an OpenAI/Anthropic-compatible HTTP server in a single static-ish
binary. This guide covers a production-grade **Kubernetes** deployment with a
self-converging **anti-fragile fleet**: every replica learns locally, and a
gossip CronJob propagates each node's immunity (heal-memory) across the fleet so
a fix any one node discovers becomes shared knowledge.

## TL;DR

```sh
# 1. The image is published to GHCR on every push to main:
#    ghcr.io/fernandogarzaaa/axiom-aether:latest

# 2. Install with a trained checkpoint seeded at boot + a fleet key:
helm install axiom deploy/helm/axiom \
  --namespace axiom --create-namespace \
  --set checkpoint.url=https://<your-store>/axiom_production_bpe.bin \
  --set checkpoint.tokenizerUrl=https://<your-store>/axiom_bpe.json \
  --set secrets.fleetKey=$(openssl rand -hex 32) \
  --set secrets.anthropicApiKey=$ANTHROPIC_API_KEY

# 3. Verify:
kubectl -n axiom rollout status deploy/axiom
kubectl -n axiom port-forward svc/axiom 8080:80
curl localhost:8080/readyz        # {"status":"ready","model":"..."}
```

## What the chart creates

| Object | Purpose |
|---|---|
| **Deployment** | N replicas, runs as non-root, `readOnlyRootFilesystem`, with `startup`/`liveness`/`readiness` probes. |
| **Service** (ClusterIP) | Load-balanced traffic entrypoint (`:80` → container `:8080`). |
| **Headless Service** | Per-pod DNS + Endpoints — the substrate the gossip job uses to enumerate the fleet. |
| **ConfigMap** | Non-secret env (device, ports, feature flags, checkpoint URLs). |
| **Secret** | `AXIOM_FLEET_KEY`, `ANTHROPIC_API_KEY`, `AXIOM_CONTEXT_API_KEY`. |
| **CronJob + RBAC** | Immunity gossip (see below). |
| **HPA** (optional) | CPU-based autoscaling. |
| **PVC** (optional) | Persistent state for single-replica mode. |

## The three cloud problems, and how the chart solves them

1. **Health probes.** The server only binds its socket *after* the model loads,
   so reaching `/readyz` already implies readiness; `/healthz` is an unconditional
   liveness check that never touches the inference lock (a long generation can't
   trip a restart). The `startupProbe` gives the pod up to ~5 min to download the
   checkpoint and load before liveness kicks in.

2. **Ephemeral, multi-replica state.** The image ships **without weights**
   (`.dockerignore` excludes `checkpoints/`). At boot, the entrypoint seeds the
   read-only checkpoint + tokenizer from `checkpoint.url` / `checkpoint.tokenizerUrl`
   (object storage or a GitHub release). Mutable learned state (heal-memory,
   immunity) lives on a per-pod `emptyDir` and **converges across the fleet via
   gossip** — no shared filesystem required.

3. **Secrets.** API keys and the fleet key come from a Kubernetes Secret
   (chart-created or `secrets.existingSecret`), never baked into the image.

## Fleet convergence (anti-fragile swarm immunity)

Each node exposes:
- `GET /v1/immunity` — its heal-memory, **HMAC-signed** with `AXIOM_FLEET_KEY`.
- `POST /v1/immunity/merge` — folds a peer's signed export in (verifies first;
  local learning is never weakened — dirs are unioned, tensions count-weighted).

The **gossip CronJob** (default every 5 min) reads the headless service's
Endpoints to find live pods, pulls each pod's signed export, and fans it out to
the others' merge endpoint. Because exports are signed, a receiver rejects any
unsigned or tampered payload when a fleet key is set.

```
       ┌─ pod A ─┐   GET /v1/immunity (signed)    ┌─ gossip CronJob ─┐
       │ learns  │ ───────────────────────────────▶│  every 5 min     │
       └─────────┘                                  │  fan-out merge   │
       ┌─ pod B ─┐ ◀── POST /v1/immunity/merge ─────└──────────────────┘
       │ inherits│
       └─────────┘
```

Tune or disable:
```sh
--set gossip.schedule="*/2 * * * *"   # faster convergence
--set gossip.enabled=false            # turn the fleet sync off
```

> The gossip image needs `curl` + `jq` (default `dwdraju/alpine-curl-jq`).
> Override `gossip.image` if you mirror images internally.

## Scaling

```sh
helm upgrade axiom deploy/helm/axiom --reuse-values \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=3 --set autoscaling.maxReplicas=12
```

Replicas are stateless for serving (model is read-only; TTT fast-weights are
per-session in memory). New pods seed the checkpoint on boot and inherit the
fleet's accumulated immunity at the next gossip round.

## GPU

Set `config.device=cuda` and add GPU resources/affinity via values
(`resources.limits."nvidia.com/gpu": 1`, plus the matching `nodeSelector`).
The CPU `d128/2L` checkpoint is the default; for a larger model see the
"Scaling note" in the README — capacity must match the corpus.

## Local validation

```sh
helm lint deploy/helm/axiom
helm template axiom deploy/helm/axiom --set secrets.fleetKey=test | kubectl apply --dry-run=client -f -
```
