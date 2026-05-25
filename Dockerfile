FROM rust:1.82-slim AS builder

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
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/axiom_engine/axiom_engine_rs/target/release/axiom_engine /app/axiom_engine

RUN mkdir -p /app/checkpoints /app/tokenizer_cache

EXPOSE 8080

ENV AXIOM_HOST="0.0.0.0"
ENV AXIOM_PORT="8080"
ENV AXIOM_DEVICE="cpu"
ENV RUST_LOG="info"

ENTRYPOINT ["/app/axiom_engine"]
CMD ["--mode", "server"]
