# Dockerfile — multi-stage build for the Axiom-TTT engine.
#
# Builds the Rust `axiom_engine` binary in a builder stage, then copies it into
# a lean runtime image. The image ships WITHOUT trained weights (see
# .dockerignore) — they are seeded at runtime from AXIOM_CHECKPOINT_URL /
# AXIOM_TOKENIZER_URL or mounted as a volume (see scripts/docker_entrypoint.sh).
#
# Platforms: linux/amd64 + linux/arm64 (see .github/workflows/docker.yml).

# ---------------------------------------------------------------------------
# Stage 1: Builder — compile the Rust binary
# ---------------------------------------------------------------------------
FROM rust:1.85-bookworm AS builder

# Install build dependencies (tree-sitter needs a C compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy the crate source first (better layer caching — dependencies change less
# often than the application code).
COPY axiom_engine_rs/ ./axiom_engine_rs/

# Build the release binary. We build from the crate directory so Cargo.toml
# paths resolve correctly. The `--locked` flag uses the committed Cargo.lock
# for reproducible builds.
RUN cd axiom_engine_rs && \
    cargo build --release --locked --bin axiom_engine && \
    strip target/release/axiom_engine

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean image with just the binary and entrypoint
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim

# Install runtime dependencies (ca-certificates for HTTPS, curl for checkpoint
# download in the entrypoint, libssl for TLS).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create the app directory structure
WORKDIR /app
RUN mkdir -p /app/checkpoints /app/tokenizer_cache

# Copy the binary from the builder stage
COPY --from=builder /build/axiom_engine_rs/target/release/axiom_engine /app/axiom_engine

# Copy the entrypoint script
COPY scripts/docker_entrypoint.sh /app/docker_entrypoint.sh
RUN chmod +x /app/docker_entrypoint.sh

# Expose the default server port
EXPOSE 8080

# Environment defaults (overridable at runtime)
ENV AXIOM_HOST=0.0.0.0 \
    AXIOM_PORT=8080 \
    AXIOM_DEVICE=cpu \
    RUST_LOG=info \
    AXIOM_BPE_CKPT=/app/checkpoints/axiom_production_bpe.bin \
    AXIOM_TOKENIZER=/app/checkpoints/axiom_bpe.json

# Health check — the /healthz endpoint returns 200 when the server is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${AXIOM_PORT}/healthz || exit 1

# Use the entrypoint script which optionally downloads checkpoints, then
# hands off to the server.
ENTRYPOINT ["/app/docker_entrypoint.sh"]
CMD ["--mode", "server", "--host", "0.0.0.0", "--port", "8080"]