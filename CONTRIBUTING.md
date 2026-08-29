# Contributing to wqc-core

Thank you for your interest in contributing to **wqc-core**, the quantum circuit executor and zk-STARK proving engine at the heart of the World Quantum Computer protocol.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it. Report unacceptable behavior using the contact details in that document.

## How to Contribute

Contributions are welcome in many forms:

- Bug reports and feature requests via [GitHub Issues](https://github.com/world-qc/wqc-core/issues)
- Documentation improvements (including `doc/`)
- Code changes via pull requests

If you plan a larger change, please open an issue first so we can discuss the approach and avoid duplicate work.

## Development Setup

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) **1.95** or newer
- Sibling checkout of [`wqc-stark-engine`](https://github.com/world-qc/wqc-stark-engine) (required by `Cargo.toml` `[patch]`)

### Clone and build

```bash
git clone https://github.com/world-qc/wqc-core.git
git clone https://github.com/world-qc/wqc-stark-engine.git   # sibling directory
cd wqc-core
cargo build
```

Optional WebGPU MPS backend:

```bash
cargo build --features webgpu
```

### Run locally

```bash
cargo run
# Default: Unix domain socket at /tmp/wqc-core.sock (see README for env vars)
```

See [`openapi/openapi.yaml`](openapi/openapi.yaml) for the HTTP contract, and
[README.md](README.md) for configuration and trace semantics (`doc/trace-spec.md`).

## Making Changes

1. Fork the repository and create a branch from `main`.
2. Make your changes in a focused, reviewable scope.
3. Run the checks below before opening a pull request.
4. Open a pull request against `main` with a clear description of the change and why it is needed.

### Branch naming

Use short, descriptive names, for example:

- `fix/mps-bond-truncation`
- `docs/trace-spec-update`
- `feat/sample-counts-validation`

## Coding Guidelines

- Write all source code, documentation, and comments in **English**.
- Keep STARK public-input binding and trace layout changes backward-compatible unless versioned intentionally.
- Follow common Rust conventions (`cargo fmt`, idiomatic error handling).
- Align execution-trace changes with `doc/trace-spec.md` and cross-check against `wqc-stark-engine` when touching proofs.
- HTTP request/response JSON lives in [`openapi/openapi.yaml`](openapi/openapi.yaml). Update it in the same pull request as handler or payload struct changes. `cargo test --test openapi_spec` checks the path inventory against `src/main.rs`.

## Checks

Before submitting a pull request, run:

```bash
cargo fmt --all
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
cargo build --release --locked
```

If you add new behavior, include tests where practical.

## Pull Request Guidelines

A good pull request:

- Has a concise title and description
- Explains the problem and the chosen solution
- Links related issues (for example, `Fixes #123`)
- Passes local checks listed above
- Keeps unrelated changes out of the diff
- Notes any API, trace-format, or proof-binding changes clearly (HTTP: update `openapi/openapi.yaml`)

Maintainers may request changes or suggest an alternative approach. Once approved, your contribution will be merged.

## Licensing

By contributing, you agree that your contributions will be licensed under the same terms as the project: the [GNU General Public License v3.0](LICENSE).

## Questions

If something is unclear, open a [GitHub Issue](https://github.com/world-qc/wqc-core/issues) or ask in your pull request. We are happy to help you get started.
