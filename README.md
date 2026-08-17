# wqc-core (The Engine)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-yellow.svg)]()
[![CI](https://github.com/world-qc/wqc-core/actions/workflows/ci.yml/badge.svg)](https://github.com/world-qc/wqc-core/actions/workflows/ci.yml)

`wqc-core` is the computational heart of the World Quantum Computer (WQC) protocol.
It is a Rust quantum circuit executor optimized for **Decentralized Proof of Useful Work (D-PoUW)**:
each slice is contracted as a **tensor network** (default: bond-truncated MPS), then proven with a zk-STARK.

## Key Features

- **Tensor-network execution**: Gate-by-gate MPS contraction with SVD bond truncation (`O(N · χ²)` memory).
- **zk-STARK D-PoUW**: Plonky3 uni-STARK over the execution trace (`wqc-stark-engine`); no re-execution on verify.
- **Compact-register slices**: Width `qubit_count = N − |assignments|`; scalar output = ⟨0…0|ψ⟩ amplitude.
- **Public-input binding**: `circuit_id`, `sub_task_id`, `node_id`, `slice_id`, `output_result_hash` in every proof.
- **Orchestrator χ hints**: Per-task `mps_max_bond_dim` capped by node env `WQC_MPS_MAX_BOND_DIM`.
- **WorkReport metrics**: `trace_rows`, `gate_count`, `compute_wall_ms`, `prove_wall_ms`, `proof_bytes` for Gas settlement upstream.

## Technical Architecture & Roadmap

### Phase 1: Foundation (completed)

- Universal gate set: H, X, Y, Z, T, S, CNOT, CZ, RX, RY, RZ, CCNOT, **MEASURE** (terminal Z-basis).
- Compute → prove → verify cycle with cross-language hash alignment (Rust / Go).
- HTTP API over Unix domain socket (default) or TCP.

### Phase 2: Scaling & distribution (current)

- [x] zk-STARK integration (trace-schema v2, multi-target AIR).
- [x] TN engine Phase 2a (`src/tn/`, compact-register boundaries).
- [x] MPS bond truncation Phase 2b (default backend). See `doc/tn-engine.md`.
- [x] Orchestrator `mps_max_bond_dim` per slice (`min(env χ, task χ)`).
- [x] WebGPU MPS kernels (`--features webgpu`, `WQC_TN_BACKEND=webgpu`).
- [x] **`sample_counts` execution**: terminal `MEASURE`, seed-bound histograms, Qiskit bitstring order.
- [x] **Swarm slice delivery (§3.1)**: orchestrator + `wqc-node` responsibility (compact-register split → libp2p dispatch → 1 core = 1 slice). MPI-style in-core state sharding is not a whitepaper goal.

### Phase 3: Sovereign network (upcoming)

- Physical QPU proxy.
- On-chain economic layer (orchestrator / L2).

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `WQC_MPS_MAX_BOND_DIM` | `128` | Node ceiling on MPS bond dimension χ |
| `WQC_TN_BACKEND` | `cpu` | `webgpu` offloads 1q/merge kernels (needs `--features webgpu`) |
| `WQC_CONNECTION_MODE` | `uds` | `uds` (Unix socket) or `tcp` |
| `WQC_CORE_TCP_PORT` | `3000` | TCP port if connection mode is `tcp` |
| `WQC_MAX_MEMORY_GB` | (unset) | PCS memory budget (GiB); unset disables the gate |
| `WQC_PCS_MEMORY_POLICY` | `refuse` | `refuse` (fail prove) or `spill` (auto-lower Mmcs chunk) when over budget. Exposed to nodes via `GET /sysinfo` → `pcs_memory_policy` for open-call bid eligibility. |
| `WQC_PCS_MMCS_GROUP_CHUNK` | `24` | Mmcs group chunk size for leaf/agg PCS prove (time vs wire trade-off) |
| `RAYON_NUM_THREADS` | `1` | Worker threads for prove (lower on memory-constrained hosts) |

Memory per slice: `≈ N · χ² · 32` bytes. Orchestrator may send a lower `mps_max_bond_dim` per task.

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/compute` | Contract slice + generate STARK proof |
| `POST` | `/leaf_pcs` | Build leaf PCS bundle from an existing leaf STARK proof |
| `POST` | `/verify` | Stateless STARK verification |
| `GET` | `/gates` | Supported gate names (feature discovery) |
| `GET` | `/sysinfo` | Host RAM / CPU snapshot + TN backend + PCS memory policy |

### `POST /compute` — request fields

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `string` | Sub-task ID (STARK `sub_task_id`) |
| `circuit_id` | `string` | SHA3-256 of pruned circuit (public input) |
| `node_id` | `string` | Executing node identity |
| `qubit_count` | `int` | Effective compact width after slice cuts |
| `original_qubit_count` | `int` | Parent circuit width before slicing |
| `slice_id` | `string` | Binary path in slice tree (e.g. `"0"`, `"01"`) |
| `slice_assignments` | `array` | Fixed legs `{ "edge_id": "e_0", "value": 0\|1 }` |
| `circuit` | `array` | Pruned gates with local qubit indices |
| `mps_max_bond_dim` | `int` (optional) | Orchestrator χ recommendation; effective χ = `min(this, WQC_MPS_MAX_BOND_DIM)` |
| `output_mode` | `string` (optional) | `statevector_scalar` (default), `sample_counts`, or `expectation` |
| `classical_bit_count` | `int` (optional) | Classical register width (`sample_counts` required) |
| `shots` | `int` (optional) | Shot count (`sample_counts` required) |
| `sample_seed` | `int` (optional) | PRNG seed from orchestrator (`sample_counts` required) |
| `observables` | `array` (optional) | Pauli sums for `expectation` mode |

`WorkReport` also returns `tn_backend` and `vram_peak_bytes` (WebGPU path).

### `output_mode`

| Mode | Returns | STARK proves |
|------|---------|--------------|
| `statevector_scalar` | `complex_result` (amplitude at \|0…0⟩) | Unitary TN trace |
| `sample_counts` | `sample_result.counts` + `shots` | Unitary TN trace only; `output_result_hash` binds canonical counts JSON |
| `expectation` | `expectation_result.values` (Pauli sums) | Unitary TN trace only; `output_result_hash` binds canonical expectation JSON |

**X/Y basis**: `MEASURE` is Z-only. For `sample_counts`, insert `H` (X) or `RX(-π/2)` (Y) before `MEASURE`. For `expectation`, use Pauli `X`/`Y` in `observables`. See `src/basis.rs` and `wqc-docs/examples/circuits/sample/`.

**`counts` bitstring**: Qiskit-compatible — **rightmost character = `cbit 0`**.
**Scope**: terminal `MEASURE` gates only; mid-circuit measure is rejected. Sampling uses full statevector projection (`qubit_count ≤ 20`). Multi-slice `sample_counts` is not supported; use single-slice / small circuits.

**Example `sample_counts` circuit** (Bell state):

```json
{
  "output_mode": "sample_counts",
  "classical_bit_count": 2,
  "shots": 1024,
  "sample_seed": 42,
  "circuit": [
    { "type": "H", "params": [0] },
    { "type": "CNOT", "params": [0, 1] },
    { "type": "MEASURE", "params": { "qubit": 0, "cbit": 0 } },
    { "type": "MEASURE", "params": { "qubit": 1, "cbit": 1 } }
  ]
}
```

**Example response (`sample_counts`)**:

```json
{
  "task_id": "sub-task-1",
  "status": "success",
  "result_type": "sample_counts",
  "complex_result": { "real": 0.7071067811865475, "imag": 0.0 },
  "sample_result": {
    "counts": { "00": 512, "11": 512 },
    "shots": 1024
  },
  "proof": { "…": "…" },
  "work_report": { "…": "…" }
}
```

**Example request:**

```json
{
  "task_id": "sub-task-1",
  "circuit_id": "abc123…",
  "node_id": "12D3Koo…",
  "qubit_count": 3,
  "original_qubit_count": 26,
  "slice_id": "0",
  "slice_assignments": [],
  "mps_max_bond_dim": 128,
  "circuit": [
    { "type": "H", "params": [0] },
    { "type": "CCNOT", "params": [0, 1, 2] }
  ]
}
```

**Example response:**

```json
{
  "task_id": "sub-task-1",
  "status": "success",
  "complex_result": { "real": 0.3535533905932738, "imag": 0.0 },
  "proof": {
    "public_inputs": {
      "circuit_id": "abc123…",
      "sub_task_id": "sub-task-1",
      "node_id": "12D3Koo…",
      "slice_id": "0",
      "output_result_hash": "deadbeef…"
    },
    "stark_proof_b64": "…"
  },
  "work_report": {
    "trace_rows": 42,
    "gate_count": 2,
    "compute_wall_ms": 12,
    "prove_wall_ms": 340,
    "proof_bytes": 65536
  }
}
```

`complex_result` is the amplitude at computational basis |0…0⟩ on free wires after TN contraction.

### `POST /verify`

Verifies `stark_proof_b64` against the five public inputs. Does not re-run the circuit.

```json
{
  "proof": {
    "public_inputs": { "…": "…" },
    "stark_proof_b64": "…"
  }
}
```

Success: `200` with `{ "valid": true, "reason": null }`.
Invalid proof: `403` with `{ "valid": false, "reason": "…" }`.

### `GET /sysinfo`

Returns host metrics and prove-time configuration for node scheduling and PCS open-call gating:

```json
{
  "system_memory_used_kb": 1048576,
  "system_memory_total_kb": 8388608,
  "cpu_usage_percent": 12.5,
  "tn_backend_requested": "webgpu",
  "tn_backend_active": "webgpu",
  "mps_max_bond_dim": 128,
  "pcs_memory_policy": "spill"
}
```

`pcs_memory_policy` mirrors this core process's `WQC_PCS_MEMORY_POLICY`. `wqc-node` probes it before bidding on PCS open calls.

### Error handling

| HTTP | When | Typical cause |
|------|------|----------------|
| `400` | Bad request | Invalid qubit index, compact-register mismatch, insufficient RAM for requested χ |
| `403` | Forbidden | `/verify` — STARK transcript or public inputs invalid |
| `500` | Internal error | Contraction or proving failure |

There is no `503` path and no `memory_cost_kb` request field.

## Testing

```bash
cargo test --release                 # CI suite (skips #[ignore])
cargo test --release -- --ignored  # Plonky3 STARK prove/verify roundtrips (local; can take a long time)
```

GitHub Actions runs the fast suite only (`timeout-minutes: 45`). Heavy STARK proves are `#[ignore]` — same policy as `wqc-stark-engine`.

## Documentation

- `doc/tn-engine.md` — MPS backend, χ configuration, execution flow
- `doc/trace-spec.md` — STARK trace columns (v2, `TRACE_WIDTH = 11`)
- `whitepaper_gap.md` — WP v0.3 alignment and remaining gaps

## Requirements

- **Rust**: 1.95+ (see `AGENTS.md`)
- **RAM**: Depends on `N` and χ; devnet compose often sets `WQC_MPS_MAX_BOND_DIM=256`
- **Key deps**: `nalgebra` (SVD), `wqc-stark-engine` (Plonky3), `axum`, `sha3`

One concurrent `/compute` per process is recommended; the node orchestrates single-task execution per core instance.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding guidelines, and the pull request process.

## License

GNU General Public License v3.0 (GPLv3). See `LICENSE`.
