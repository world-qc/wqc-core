# WQC Tensor-Network Engine

`wqc-core` executes each slice as a **tensor-network (TN) contraction**: gates are complex unitary
tensors; each application contracts the gate into the open wire indices of the network state.

## Module layout

| File | Role |
|------|------|
| `src/tn/dense.rs` | **Dense exact backend** — rank-N state tensor (`2^N` amplitudes), gate-by-gate contraction |
| `src/tn/boundary.rs` | Slice leg fixation (`e_<k>` → qubit `k`), Policy C validation |
| `src/tn/trace.rs` | STARK execution trace (11 columns) during contraction |
| `src/tn/contract.rs` | `contract_slice()` entry used by `TensorNetwork::contract` |

`QuantumRegister` is a type alias for `DenseTnState`.

## Execution flow

```
POST /compute
  → ContractionWorkspace::try_allocate(effective_qubits)
  → TensorNetwork::from_parts(qubits, pruned_gates, slice_assignments)
  → contract_slice():
       1. Parse + verify boundary (Policy C: effective = original − |assignments|)
       2. Initialise |0…0⟩ on compact wires
       3. For each gate: emit pre-trace row → contract gate tensor → post-trace row
       4. Scalar readout = amplitude at compact |0…0⟩
  → STARK prove on trace
```

## Policy C boundaries

The orchestrator prunes fixed legs and remaps surviving gates to local indices `0 .. qubit_count−1`.
Workers receive:

- `qubit_count` — effective register width
- `original_qubit_count` — parent circuit width
- `slice_assignments` — classical bits fixed on cut legs `e_<global_qubit>`

Core validates `qubit_count == original_qubit_count − len(slice_assignments)` and binds
assignments into STARK public inputs. The TN executor starts from |0…0⟩ on free wires; upstream
prune must have folded fixed legs into the gate list.

`BoundaryConditions::global_basis_index_for_compact_zero()` maps assignments to the parent-register
basis index for cross-layer audits.

## Backends

| Backend | Status | Memory |
|---------|--------|--------|
| `dense` (default) | ✅ | `O(2^N)` — reference + devnet |
| `mps` (bond-truncated) | 🔜 | `O(N · χ²)` — low-entanglement slices |
| `wgpu` | 🔜 | GPU tensor cores, same trace schema |

Select future backends with `WQC_TN_BACKEND` (not wired yet; dense is always used).

## Tests

```bash
cargo test -p wqc-core
```

Key cases:

- `tn::contract::h_then_ccnot_matches_circuit_executor` — TN path ≡ legacy circuit executor
- `tn::boundary::policy_c_verifies_effective_qubit_count`
- `engine::trace_tests::executor_traces_satisfy_stark_air_for_h_and_cnot`

## Next steps

1. **MPS exact / truncated** — bond-dimension cap for memory sub-linear in `N`
2. **Contraction order** — integrate with orchestrator optimal-cut hints
3. **Non-pruned mode** — contract on full `original_qubit_count` with in-core boundary projectors
4. **WebGPU** — portable accelerator path (`wgpu`)
