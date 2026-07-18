# WQC Tensor-Network Engine

`wqc-core` executes each slice as a **tensor-network (TN) contraction**. The default backend is
**bond-truncated MPS** (`O(N · χ²)` memory). A dense reference backend remains in `dense.rs` for tests.

## Module layout

| File | Role |
|------|------|
| `src/tn/mps.rs` | **Default backend** — MPS with SVD bond truncation (`χ = WQC_MPS_MAX_BOND_DIM`) |
| `src/tn/gates.rs` | Gate unitary tensors (1- and 2-qubit); CCNOT via Toffoli decomposition |
| `src/tn/boundary.rs` | Slice leg fixation (`e_<k>`), Policy C validation |
| `src/tn/site_order.rs` | Optional `mps_site_order` validation + logical→site gate/observable remap |
| `src/tn/trace.rs` | STARK trace during MPS contraction (dense marginal sampling when `N ≤ 16`) |
| `src/tn/contract.rs` | `contract_slice()` entry |
| `src/tn/dense.rs` | Exact `2^N` reference (unit tests only) |

`QuantumRegister` is a type alias for `MpsState`.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `WQC_MPS_MAX_BOND_DIM` | `128` | Node/core ceiling on MPS bond dimension `χ` |
| `mps_max_bond_dim` (per task) | — | Orchestrator recommendation; effective χ = `min(env, task)` |
| `WQC_TN_BACKEND` | `cpu` | `cpu` or `webgpu` (requires `--features webgpu` build) |

`GET /sysinfo` also returns TN probe fields for node startup banners:

- `tn_backend_requested` — value of `WQC_TN_BACKEND`
- `tn_backend_active` — `cpu` or `webgpu` after adapter probe
- `tn_backend_note` — set when falling back (e.g. adapter not found)
- `mps_max_bond_dim` — effective χ ceiling from env

Memory estimate per slice: `≈ N · χ² · 32` bytes (vs `2^N · 16` for dense).

## WebGPU backend (Phase 2c)

Build with `cargo build --features webgpu`. At runtime set `WQC_TN_BACKEND=webgpu`.

| Component | GPU | CPU fallback |
|-----------|-----|----------------|
| 1-qubit site apply | `apply_one_qubit` compute shader | native loops |
| 2-site merge (θ tensor) | `merge_two_site` compute shader | native loops |
| 2-qubit unitary on θ | — | native (f64) |
| SVD bond truncation | — | `nalgebra` (f64) |

`WorkReport` includes `tn_backend` (`cpu` / `webgpu`) and `vram_peak_bytes` (peak GPU buffer allocation per task).

Shaders: `src/tn/gpu/shaders.wgsl`. Complex numbers use `vec2<f32>` on GPU; results are promoted to `f64` before SVD and trace emission.

## Execution flow

```
POST /compute
  → ContractionWorkspace::try_allocate_with_bond (χ = min(env, task))
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
- [x] Orchestrator `mps_max_bond_dim` per slice (χ recommendation + bond proxy cap)
- [x] WebGPU MPS kernels (`--features webgpu`, `WQC_TN_BACKEND=webgpu`)
- [x] Orchestrator TN cut: Stoer–Wagner exact min-cut (+ Phase A/B fallback)
- [x] MPS site-order hints (`mps_site_order` orch → node → `tn/site_order.rs` remap)
