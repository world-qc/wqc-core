# syntax=docker/dockerfile:1
# Build context must include BOTH repositories as siblings under one parent:
#
#   parent/
#     wqc-core/          ← this Dockerfile
#     wqc-stark-engine/wqc-stark-core/
#
# From the parent directory (sibling checkouts under one folder):
#   docker build -f wqc-core/Dockerfile -t world-qc/wqc-core:latest .
#
# Uses Cargo.toml [patch] → ../wqc-stark-engine/wqc-stark-core (no git fetch in builder).

FROM rust:1.95-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# /build/wqc-core + /build/wqc-stark-engine/wqc-stark-core (patch path)
WORKDIR /build/wqc-core

COPY wqc-stark-engine/wqc-stark-core /build/wqc-stark-engine/wqc-stark-core

COPY wqc-core/Cargo.toml wqc-core/Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs

ARG WQC_FEATURES=webgpu
RUN cargo build --release --features "${WQC_FEATURES}"
RUN rm -f target/release/deps/wqc_core*

COPY wqc-core/src ./src
COPY wqc-stark-engine/wqc-stark-core /build/wqc-stark-engine/wqc-stark-core
RUN cargo build --release --features "${WQC_FEATURES}"

RUN cp target/release/wqc-core /usr/local/bin/wqc-core
RUN cp Cargo.lock /Cargo.lock

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/bin/wqc-core /usr/local/bin/
COPY --from=builder /Cargo.lock /

CMD ["wqc-core"]
