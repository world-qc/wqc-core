# WQC Tensor-Network Engine

`wqc-core` executes each slice as a **tensor-network (TN) contraction**. The default backend is
**bond-truncated MPS** (`O(N · χ²)` memory). A dense reference backend remains in `dense.rs` for tests.

## Module layout

| File | Role |
|------|------|
| `src/tn/mps.rs` | **Default backend** — MPS with SVD bond truncation (`χ = WQC_MPS_MAX_BOND_DIM`) |
| `src/tn/gates.rs` | Gate unitary tensors (1- and 2-qubit); CCNOT via Toffoli decomposition |
| `src/tn/boundary.rs` | Slice leg fixation (`e_<k>`), Policy C validation |
| `src/tn/trace.rs` | STARK trace during MPS contraction (dense marginal sampling when `N ≤ 16`) |
| `src/tn/contract.rs` | `contract_slice()` entry |
| `src/tn/dense.rs` | Exact `2^N` reference (unit tests only) |

`QuantumRegister` is a type alias for `MpsState`.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `WQC_MPS_MAX_BOND_DIM` | `128` | Maximum MPS bond dimension `χ` after each 2-qubit SVD split |

Memory estimate per slice: `≈ N · χ² · 32` bytes (vs `2^N · 16` for dense).

## Execution flow

```
POST /compute
  → ContractionWorkspace::try_allocate (MPS, χ from env)
  → TensorNetwork::contract → contract_slice()
       1. Policy C boundary check
       2. |0…0⟩ product MPS
       3. Per gate: pre-trace → apply gate tensor(s) → SVD truncate → post-trace
       4. Scalar = ⟨0…0|ψ⟩
  → STARK prove
```

## Gate application (MPS)

| Arity | Method |
|-------|--------|
| 1-qubit | Local tensor contraction on site `t` |
| 2-qubit | Bubble SWAPs → adjacent pair → merge → apply `U` → SVD split (truncate to `χ`) |
| CCNOT | Standard Toffoli decomposition (H / CNOT / T / T†) using 2-site kernels only |

## Policy C boundaries

Unchanged from Phase 2a — see prior sections in git history. `slice_assignments` validate
`effective_qubit_count == original_qubit_count − |assignments|`.

## Trace / STARK

- Circuits with `N ≤ 16`: trace marginals sampled from `MpsState::to_dense_state()` (AIR-compatible).
- Larger `N`: MPS environment marginals (approximate trace; devnet-scale circuits use slicing to keep `N` small per slice).

## Tests

```bash
cargo test -p wqc-core
```

## Roadmap

- [x] Phase 2b: MPS + bond truncation (default backend)
- [ ] Orchestrator optimal-cut hints → contraction order
- [ ] Full dense-free trace marginals for large `N`
- [ ] WebGPU MPS kernels (`wgpu`)
