# syntax=docker/dockerfile:1
# Build context must include BOTH repositories as siblings under one parent:
#
#   parent/
#     wqc-core/          ← this Dockerfile
#     wqc-stark-engine/wqc-stark-core/
#
# From the parent directory:
#   docker build --platform linux/amd64 -f wqc-core/Dockerfile \
#     -t world-qc/wqc-core:latest .
#
# On Apple Silicon, the builder runs on BUILDPLATFORM (native) and
# cross-compiles to amd64. Full linux/amd64 under QEMU SIGSEGVs in collect2/cc
# (same pattern as wqc-composer / wqc-p2p-proxy).
#
# Uses Cargo.toml [patch] → ../wqc-stark-engine/wqc-stark-core (no git fetch).

FROM --platform=$BUILDPLATFORM rust:1.95-slim-bookworm AS builder

ARG TARGETARCH
ARG BUILDARCH
ARG WQC_FEATURES=webgpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    pkg-config \
    $(if [ "$TARGETARCH" = "amd64" ] && [ "$BUILDARCH" != "amd64" ]; then \
        echo build-essential gcc-x86-64-linux-gnu g++-x86-64-linux-gnu libc6-dev-amd64-cross; \
      else \
        echo build-essential libssl-dev; \
      fi) \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# /build/wqc-core + /build/wqc-stark-engine/wqc-stark-core (patch path)
WORKDIR /build/wqc-core

COPY wqc-stark-engine/wqc-stark-core /build/wqc-stark-engine/wqc-stark-core

COPY wqc-core/Cargo.toml wqc-core/Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs

RUN set -eux; \
    if [ "$TARGETARCH" = "amd64" ] && [ "$BUILDARCH" != "amd64" ]; then \
      rustup target add x86_64-unknown-linux-gnu; \
      export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc; \
      export CC_x86_64_unknown_linux_gnu=x86_64-linux-gnu-gcc; \
      export CXX_x86_64_unknown_linux_gnu=x86_64-linux-gnu-g++; \
      export AR_x86_64_unknown_linux_gnu=x86_64-linux-gnu-ar; \
      export CARGO_TERM_PROGRESS_WHEN=never; \
      cargo build --release --target x86_64-unknown-linux-gnu --features "${WQC_FEATURES}"; \
      rm -f target/x86_64-unknown-linux-gnu/release/deps/wqc_core*; \
    else \
      cargo build --release --features "${WQC_FEATURES}"; \
      rm -f target/release/deps/wqc_core*; \
    fi

COPY wqc-core/src ./src
COPY wqc-stark-engine/wqc-stark-core /build/wqc-stark-engine/wqc-stark-core

RUN set -eux; \
    if [ "$TARGETARCH" = "amd64" ] && [ "$BUILDARCH" != "amd64" ]; then \
      export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc; \
      export CC_x86_64_unknown_linux_gnu=x86_64-linux-gnu-gcc; \
      export CXX_x86_64_unknown_linux_gnu=x86_64-linux-gnu-g++; \
      export AR_x86_64_unknown_linux_gnu=x86_64-linux-gnu-ar; \
      export CARGO_TERM_PROGRESS_WHEN=never; \
      cargo build --release --target x86_64-unknown-linux-gnu --features "${WQC_FEATURES}"; \
      cp target/x86_64-unknown-linux-gnu/release/wqc-core /usr/local/bin/wqc-core; \
    else \
      cargo build --release --features "${WQC_FEATURES}"; \
      cp target/release/wqc-core /usr/local/bin/wqc-core; \
    fi; \
    cp Cargo.lock /Cargo.lock

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/bin/wqc-core /usr/local/bin/
COPY --from=builder /Cargo.lock /

CMD ["wqc-core"]
