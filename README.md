# wqc-core (The Engine)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-yellow.svg)]()

`wqc-core` is the computational heart of the World Quantum Computer (WQC) protocol.
It is a Rust quantum circuit executor optimized for **Decentralized Proof of Useful Work (D-PoUW)**:
each slice is contracted as a **tensor network** (default: bond-truncated MPS), then proven with a zk-STARK.

## Key Features

- **Tensor-network execution**: Gate-by-gate MPS contraction with SVD bond truncation (`O(N · χ²)` memory).
- **zk-STARK D-PoUW**: Plonky3 uni-STARK over the execution trace (`wqc-stark-engine`); no re-execution on verify.
- **Policy C slices**: Compact register width `qubit_count = N − |assignments|`; scalar output = ⟨0…0|ψ⟩ amplitude.
- **Public-input binding**: `circuit_id`, `sub_task_id`, `node_id`, `slice_id`, `output_result_hash` in every proof.
- **Orchestrator χ hints**: Per-task `mps_max_bond_dim` capped by node env `WQC_MPS_MAX_BOND_DIM`.
- **WorkReport metrics**: `trace_rows`, `gate_count`, `compute_wall_ms`, `prove_wall_ms`, `proof_bytes` for Gas settlement upstream.

## Technical Architecture & Roadmap

### Phase 1: Foundation (completed)

- Universal gate set: H, X, Y, Z, T, S, CNOT, CZ, RX, RY, RZ, CCNOT (Toffoli decomposition in MPS).
- Compute → prove → verify cycle with cross-language hash alignment (Rust / Go).
- HTTP API over Unix domain socket (default) or TCP.

### Phase 2: Scaling & distribution (current)

- [x] zk-STARK integration (trace-schema v2, multi-target AIR).
- [x] TN engine Phase 2a (`src/tn/`, Policy C boundaries).
- [x] MPS bond truncation Phase 2b (default backend). See `doc/tn-engine.md`.
- [x] Orchestrator `mps_max_bond_dim` per slice (`min(env χ, task χ)`).
- [ ] WebGPU MPS kernels (`wgpu`).
- [ ] Distributed processing / P2P state sharding (orchestrator + node responsibility).

### Phase 3: Sovereign network (upcoming)

- Physical QPU proxy.
- On-chain economic layer (orchestrator / L2).

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `WQC_MPS_MAX_BOND_DIM` | `128` | Node ceiling on MPS bond dimension χ |
| `WQC_CONNECTION_MODE` | `uds` | `uds` (Unix socket) or `tcp` |

Memory per slice: `≈ N · χ² · 32` bytes. Orchestrator may send a lower `mps_max_bond_dim` per task.

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/compute` | Contract slice + generate STARK proof |
| `POST` | `/verify` | Stateless STARK verification |
| `GET` | `/gates` | Supported gate names (feature discovery) |
| `GET` | `/sysinfo` | Host RAM / CPU snapshot |

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

### Error handling

| HTTP | When | Typical cause |
|------|------|----------------|
| `400` | Bad request | Invalid qubit index, Policy C mismatch, insufficient RAM for requested χ |
| `403` | Forbidden | `/verify` — STARK transcript or public inputs invalid |
| `500` | Internal error | Contraction or proving failure |

There is no `503` path and no `memory_cost_kb` request field (removed with Argon2 PoUW).

## Documentation

- `doc/tn-engine.md` — MPS backend, χ configuration, execution flow
- `doc/trace-spec.md` — STARK trace columns (v2, `TRACE_WIDTH = 11`)
- `whitepaper_gap.md` — WP v0.3 alignment and remaining gaps

## Requirements

- **Rust**: 1.75+ (workspace uses 2021 edition)
- **RAM**: Depends on `N` and χ; devnet compose often sets `WQC_MPS_MAX_BOND_DIM=256`
- **Key deps**: `nalgebra` (SVD), `wqc-stark-engine` (Plonky3), `axum`, `sha3`

One concurrent `/compute` per process is recommended; the node orchestrates single-task execution per core instance.

## License

GNU General Public License v3.0 (GPLv3). See `LICENSE`.
