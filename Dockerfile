# --- Build Stage ---
FROM rust:1.95-slim-bookworm AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Pre-compile dependencies only
# Create a dummy main.rs to build dependencies
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -f target/release/deps/wqc_core*

# Step 2: Build actual source code
COPY src ./src
RUN cargo build --release
RUN cp target/release/wqc-core /usr/local/bin/
RUN cp Cargo.lock /

# --- Runtime Stage ---
FROM debian:bookworm-slim

WORKDIR /app
COPY --from=builder /usr/local/bin/wqc-core /usr/local/bin/
COPY --from=builder /Cargo.lock /

# Run the core
CMD ["wqc-core"]
