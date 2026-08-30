# wqc-core (The Engine)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-orange.svg)]()
[![CI](https://github.com/world-qc/wqc-core/actions/workflows/ci.yml/badge.svg)](https://github.com/world-qc/wqc-core/actions/workflows/ci.yml)
[![API Reference](https://img.shields.io/badge/API-Reference-blue)](https://world-qc.github.io/wqc-docs/core/)

`wqc-core` is the computational heart of the World Quantum Computer (WQC) protocol.
It is a Rust quantum circuit executor optimized for **Decentralized Proof of Useful Work (D-PoUW)**:
each slice is contracted as a **tensor network** (default: bond-truncated MPS), then proven with a zk-STARK.

## Role in the WQC pipeline

```
client → wqc-orchestrator → wqc-node → wqc-core (this repo)
```

In production, **[`wqc-node`](https://github.com/world-qc/wqc-node)** on the same host
starts this process and calls its HTTP API for each slice. Direct `/compute` access is
for local development, integration tests, and debugging — not for end-user clients.

Swarm slice delivery (orchestrator split → libp2p dispatch → one core instance per slice)
is described in [wqc-docs `spec/architecture-current.md` §3](https://github.com/world-qc/wqc-docs/blob/main/spec/architecture-current.md#3-task-lifecycle)
(cut, dispatch, compute). MPI-style in-core state sharding is not a whitepaper goal.

## Key Features

- **Tensor-network execution**: Gate-by-gate MPS contraction with SVD bond truncation (`O(N · χ²)` memory).
- **zk-STARK D-PoUW**: Plonky3 uni-STARK over the execution trace (`wqc-stark-engine`); no re-execution on verify.
- **Compact-register slices**: Width `qubit_count = N − |assignments|`; scalar output = ⟨0…0|ψ⟩ amplitude.
- **Gate set**: H, X, Y, Z, T, S, CNOT, CZ, RX, RY, RZ, CCNOT, terminal Z-basis **MEASURE**, mid-circuit **RESET** and classical **IF**.
- **Output modes**: `statevector_scalar` (default), `sample_counts` (seed-bound histograms, Qiskit bitstring order), and `expectation` (Pauli observables; no `MEASURE` gates).
- **Public-input binding**: `circuit_id`, `sub_task_id`, `node_id`, `slice_id`, `output_result_hash`, plus `measurement_spec_hash` and `security_level` when the proof binds them.
- **Orchestrator χ hints**: Per-task `mps_max_bond_dim` capped by node env `WQC_MPS_MAX_BOND_DIM`.
- **WebGPU MPS kernels**: `--features webgpu`, `WQC_TN_BACKEND=webgpu` (see `doc/tn-engine.md`).
- **WorkReport metrics**: `trace_rows`, `gate_count`, `compute_wall_ms`, `prove_wall_ms`, `proof_bytes`, plus `tn_backend` and `vram_peak_bytes` on the WebGPU path — forwarded upstream for gas settlement.

## Upcoming

- Physical QPU proxy.
- On-chain economic layer (orchestrator / L2).

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `WQC_MPS_MAX_BOND_DIM` | `128` | Node ceiling on MPS bond dimension χ |
| `WQC_TN_BACKEND` | `cpu` | `webgpu` offloads 1q/merge kernels (needs `--features webgpu`) |
| `WQC_CONNECTION_MODE` | `uds` | `uds` (Unix socket) or `tcp` |
| `WQC_SOCKET_PATH` | `/var/run/wqc-core.sock` | Unix socket path when `WQC_CONNECTION_MODE=uds` |
| `WQC_CORE_TCP_PORT` | `3000` | TCP port when `WQC_CONNECTION_MODE=tcp` (binds `0.0.0.0`) |
| `WQC_MAX_MEMORY_GB` | (unset) | PCS memory budget (GiB); unset disables the gate |
| `WQC_PCS_MEMORY_POLICY` | `refuse` | `refuse` (fail prove) or `spill` (auto-lower Mmcs chunk) when over budget. Exposed to nodes via `GET /sysinfo` → `pcs_memory_policy` for open-call bid eligibility. |
| `WQC_PCS_MMCS_GROUP_CHUNK` | `24` | Mmcs group chunk size for leaf/agg PCS prove (time vs wire trade-off) |
| `RAYON_NUM_THREADS` | `1` | Worker threads for prove (lower on memory-constrained hosts) |

Memory per slice: `≈ N · χ² · 32` bytes. Orchestrator may send a lower `mps_max_bond_dim` per task.

On Windows, `WQC_CONNECTION_MODE=uds` is unsupported; the process falls back to TCP automatically.

## API Reference

[`openapi/openapi.yaml`](openapi/openapi.yaml) is the source of truth for request and
response JSON. A rendered reference is published at
<https://world-qc.github.io/wqc-docs/core/>. The table below is an index only.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/compute` | Contract slice + generate STARK proof |
| `POST` | `/leaf_pcs` | Build leaf PCS bundle from an existing leaf STARK proof |
| `POST` | `/verify` | Stateless STARK verification |
| `GET` | `/gates` | Supported gate names (feature discovery) |
| `GET` | `/sysinfo` | Host RAM / CPU snapshot + TN backend + PCS memory policy |
| `GET` | `/health` | Liveness probe polled by `wqc-node` |

There is no auth layer and CORS is permissive. This API is for `wqc-node` on the same
host; do not expose it to a network. The default transport is a **Unix domain socket**.
When `WQC_CONNECTION_MODE=tcp`, the process binds **`0.0.0.0`** (all interfaces) with
no TLS or auth — use only on loopback-trusted hosts.

## Payload semantics

JSON field types and required keys live in [`openapi/openapi.yaml`](openapi/openapi.yaml).
The notes below document **behavior that schemas alone cannot express** — validation
rules, cross-process normalization, and proof-binding edge cases specific to this process.

Gate grammar, measurement rules, `output_mode`, `counts` key order, determinism, and
scale limits are normative in
[wqc-docs `spec/circuit-payload.md`](https://github.com/world-qc/wqc-docs/blob/main/spec/circuit-payload.md).
They apply to the whole pipeline, not just this process.

**Gate `params` shape**: single-parameter gates (`H`, `X`, `Y`, `Z`, `T`, `S`, `RESET`)
take a **bare integer** here, not a one-element array — `{"type": "H", "params": [0]}` is
a `400`. Circuits submitted to the orchestrator may use `[0]`; `wqc-node` flattens
one-element arrays (`normalize_gate_params`), including inside `IF.params.gate`, before
calling `/compute`. Do not post a client payload to `/compute` unchanged.

**Unknown fields are ignored**, so `wqc-node` may send extra keys such as
`parent_task_id` or `required_votes`.

**`security_level`**: orchestrator tier (`low`, `normal`, `high`, `ultra`) copied onto
each slice. It selects the FRI query ladder for leaf prove/verify and is bound into STARK
public inputs when non-empty. Empty or absent uses the engine default (40 queries). See
[wqc-docs `spec/zk-STARK.md` §5.1](https://github.com/world-qc/wqc-docs/blob/main/spec/zk-STARK.md).

**`noise_model`**: optional depolarizing / readout noise applied during mid-circuit
trajectory sampling only. Noise is **not bound into the STARK**; a noisy run downgrades
distribution binding to unbound. Treat as simulation/research, not verifiable production.
See [circuit-payload.md §6](https://github.com/world-qc/wqc-docs/blob/main/spec/circuit-payload.md).

**Sampling strategy**: terminal `MEASURE` uses full statevector projection. Mid-circuit
semantics (`RESET`, `IF`, or a unitary after the first `MEASURE`) switch to trajectory
sampling, which this process caps at `qubit_count ≤ 20`. Multi-slice `sample_counts` is
not supported; use single-slice or small circuits.

**`distribution_proof`** reports how strongly the transcript binds the sampled
distribution. `unitary_trace_only` means quorum still relies on the canonical counts
hash under the shared seed. The `*_linked_*` schemes are derived by the orchestrator, not
returned here.

**`complex_result`** is always present, even in `sample_counts` and `expectation`
modes: it is the amplitude at computational basis |0…0⟩ on free wires after TN
contraction.

**`/verify` is a local convenience path.** Consensus verification of worker proofs runs
in the orchestrator through `libwqc_stark_verifier` (FFI), not here.

**`pcs_memory_policy`** in `/sysinfo` mirrors this process's `WQC_PCS_MEMORY_POLICY`.
`wqc-node` requires `spill` before bidding on PCS open calls.

**Error bodies are `text/plain`**, except `/verify`, which returns a JSON
`VerifyResponse` on `403`. There is no `503` path and no `memory_cost_kb` request field.

**X/Y basis** helpers live in `src/basis.rs`; reference payloads are under
[`wqc-docs/examples/circuits/`](https://github.com/world-qc/wqc-docs/tree/main/examples/circuits).

## Testing

```bash
cargo test --release --locked          # CI suite (skips #[ignore])
cargo test --release --locked -- --ignored   # Plonky3 STARK prove/verify roundtrips (local; can take a long time)
```

GitHub Actions runs the fast release suite only (`timeout-minutes: 45`). Heavy STARK
proves are `#[ignore]` — same policy as `wqc-stark-engine`. Before opening a pull request,
also run the full check list in [CONTRIBUTING.md](CONTRIBUTING.md) (`cargo test --locked`
without `--release` is acceptable for faster local iteration).

## Documentation

- [`doc/tn-engine.md`](doc/tn-engine.md) — MPS backend, χ configuration, execution flow
- [`doc/trace-spec.md`](doc/trace-spec.md) — STARK trace columns (v2, `TRACE_WIDTH = 11`)
- [wqc-docs `spec/architecture-current.md`](https://github.com/world-qc/wqc-docs/blob/main/spec/architecture-current.md) — live swarm topology and task lifecycle
- [wqc-docs `spec/circuit-payload.md`](https://github.com/world-qc/wqc-docs/blob/main/spec/circuit-payload.md) — shared payload semantics
- [wqc-docs `spec/zk-STARK.md`](https://github.com/world-qc/wqc-docs/blob/main/spec/zk-STARK.md) — proof system and verification
- [`wqc-stark-engine`](https://github.com/world-qc/wqc-stark-engine) — AIR prover/verifier crate this process calls

## Requirements

- **Rust**: 1.95+ (see `AGENTS.md`)
- **Build layout**: clone [`wqc-stark-engine`](https://github.com/world-qc/wqc-stark-engine) as a **sibling** directory (`../wqc-stark-engine/`). `Cargo.toml` `[patch]` uses the local checkout; see [CONTRIBUTING.md](CONTRIBUTING.md).
- **RAM**: Scales with circuit size; [`wqc-miner`](https://github.com/world-qc/wqc-miner) caps host usage via `max_memory_gb` in [`settings.toml`](https://github.com/world-qc/wqc-miner/blob/main/settings.toml.example). Default tensor-network accuracy ceiling is χ=`128` in core; advanced operators may `export WQC_MPS_MAX_BOND_DIM=…` before launch (see [`doc/tn-engine.md`](doc/tn-engine.md))
- **Key deps**: `nalgebra` (SVD), `wqc-stark-engine` (Plonky3), `axum`, `sha3`

### Quick start

```bash
git clone https://github.com/world-qc/wqc-core.git
git clone https://github.com/world-qc/wqc-stark-engine.git   # sibling: ../wqc-stark-engine
cd wqc-core && cargo build && cargo run
# curl --unix-socket /var/run/wqc-core.sock http://localhost/health
```

One concurrent `/compute` per process is recommended; the node orchestrates single-task execution per core instance.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding guidelines, and the pull request process.

## License

GNU General Public License v3.0 (GPLv3). See `LICENSE`.
