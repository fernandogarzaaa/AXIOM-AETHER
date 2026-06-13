FROM rust:1-slim AS builder

# OCI image labels (used by GHCR)
LABEL org.opencontainers.image.source="https://github.com/fernandogarzaaa/AXIOM-AETHER"
LABEL org.opencontainers.image.description="Axiom-TTT Inference Engine — OpenAI-compatible API with online Test-Time Training"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /usr/src/axiom_engine
COPY . .

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/axiom_engine/axiom_engine_rs
RUN cargo build --release

FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/axiom_engine/axiom_engine_rs/target/release/axiom_engine /app/axiom_engine
COPY scripts/docker_entrypoint.sh /app/docker_entrypoint.sh
RUN chmod +x /app/docker_entrypoint.sh

RUN mkdir -p /app/checkpoints /app/tokenizer_cache

# Run as a non-root user (k8s Pod Security Standards / restricted profile friendly).
RUN useradd --system --uid 10001 --home-dir /app axiom \
    && chown -R axiom:axiom /app
USER axiom

EXPOSE 8080

ENV AXIOM_HOST="0.0.0.0"
ENV AXIOM_PORT="8080"
ENV AXIOM_DEVICE="cpu"
ENV RUST_LOG="info"

# Liveness for orchestrators that honor HEALTHCHECK (compose, Swarm).
# Kubernetes uses the httpGet probes in the Helm chart instead.
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${AXIOM_PORT}/healthz" || exit 1

# The entrypoint optionally seeds the checkpoint/tokenizer, then execs the engine.
ENTRYPOINT ["/app/docker_entrypoint.sh"]
CMD ["--mode", "server"]
