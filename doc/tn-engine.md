# WQC Tensor-Network Engine

`wqc-core` executes each slice as a **tensor-network (TN) contraction**. The default backend is
**bond-truncated MPS** (`O(N · χ²)` memory). A dense reference backend remains in `dense.rs` for tests.

## Module layout

| File | Role |
|------|------|
| `src/tn/mps.rs` | **Default backend** — MPS with SVD bond truncation (`χ = WQC_MPS_MAX_BOND_DIM`) |
| `src/tn/gates.rs` | Gate unitary tensors (1- and 2-qubit); CCNOT via Toffoli decomposition |
| `src/tn/boundary.rs` | Slice leg fixation (`e_<k>`), compact-register validation |
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

| Component | GPU | CPU |
|-----------|-----|-----|
| 1-qubit site apply | — | always. A 2×2 is below the merge side threshold, even when `WQC_TN_BACKEND=webgpu` |
| 2-site merge (θ tensor) | tiled f32 GEMM (`gemm.wgsl`) when the multiply count reaches the side measured on the first webgpu start | native f64 below that side |
| 2-qubit unitary on θ | — | native f64. The split stays on CPU, so the unitary stays with it |
| SVD bond truncation | fused Jacobi is tested through side 256 | `nalgebra` f64 for every production split. The fused kernel was slower at sides 8…256 |

`WorkReport` includes `tn_backend` (`cpu` / `webgpu`) and `vram_peak_bytes` (peak GPU buffer allocation per task).

`GET /sysinfo` includes `vram_budget_bytes` when WebGPU is active: the adapter’s
`max_buffer_size` (wgpu-visible allocation ceiling, typically ≤ device VRAM). This is a
**proxy**, not free VRAM. `wqc-node` may take `min(RAM budget, VRAM budget)` for bidding;
PCS gates still use system RAM / `WQC_MAX_MEMORY_GB` only.

Shaders: `src/tn/gpu/gemm.wgsl` (tiled complex f32, `vec2f`) is the merge used at or above the measured side. `src/tn/gpu/svd.wgsl` (double-float, `vec4f`) holds the Jacobi kernel used by the conformance test, not by production splits. The side is the square matrix with the same multiply count as that one product, not the qubit count. The first `WQC_TN_BACKEND=webgpu` start times CPU f64 against a GPU upload/multiply/readback (up to 2 min) and caches the side (`WQC_GEMM_SIDE_PATH`, or `~/Library/Caches/wqc/gemm-min-side` on macOS). Later starts read that file. `WQC_GEMM_MIN_SIDE` skips the measurement. Until a measurement exists the process uses side 160. On one Mac the measured crossover was side 160. The same 20q / χ=128 circuit still snaps to `(16861, 0)` and stays about 370 ms, because the SVD remains on CPU. `src/tn/gpu/shaders.wgsl` still has the older `vec2<f32>` 1-qubit and merge kernels. Nodes do not re-check the CPU SVD at runtime; agreement is the offline conformance test (`gpu_svd_matches_f64_on_separated_spectra`, χ = 2, 16, 32, 128).

## Execution flow

```
POST /compute
  → ContractionWorkspace::try_allocate_with_bond (χ = min(env, task))
  → TensorNetwork::contract → contract_slice()
       1. Compact-register boundary check
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

## Compact-register boundaries

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
