# WQC Trace Spec (v1)

This document defines the canonical STARK trace contract between `wqc-core` executors and `wqc-stark-core`.

## Goals

- Keep proof logic independent from execution backend (`CPU` now, `wgpu/WebGPU` later).
- Allow backend migration without rewriting AIR rules.
- Make failures debuggable by freezing row/column semantics.

## Slice contract (Policy C)

Sub-tasks use a **compact register** after tensor slicing:

| Concept | Rule |
| :--- | :--- |
| Register size | `qubit_count = original_qubit_count - len(slice_assignments)` |
| Circuit indices | Local wires `0 .. qubit_count-1` (orchestrator remaps after prune) |
| Initial state | `\|0…0⟩` on the compact register |
| Scalar output | Amplitude at **basis index `0`** (`complex_result`); assignments are metadata + STARK binding, not a readout bit-mask |

The trace below is generated from the **remapped** gate list the node receives. Any CPU or `wgpu` backend must execute that list on the compact register and use the same `state[0]` readout rule for hash alignment with the orchestrator.

## Row Layout (`TRACE_WIDTH = 10`)

Per-step row emitted by `execute_with_trace`:

1. `gate_id` — see gate id table below
2. `ctrl_active` — **discrete** `0.0` or `1.0` (marginal control probability thresholded at `0.5`)
3. `ctrl_active_2` — second control for CCNOT (same discretization)
4. `p_cos`
5. `p_sin`
6. `v0_re`
7. `v0_im`
8. `v1_re`
9. `v1_im`
10. `padding` (currently `0`)

The final boundary row reuses the same 10-column shape with `gate_id = 0`.

### Gate ids (`gate_id` column)

| Id | Gate |
|----|------|
| 1 | X |
| 2 | Y |
| 3 | Z |
| 4 | H |
| 5 | S |
| 6 | T |
| 7 | CNOT |
| 8 | CZ |
| 9 | CCNOT |
| 10–12 | RX / RY / RZ |
| 0 | Padding / terminal boundary |

## AIR Expansion (`AIR_WIDTH = 19`)

`wqc-stark-core` expands each row into:

- 1 gate id column
- 10 selector columns (`X, Y, Z, H, S, T, CNOT, CZ, CCNOT, ROT`)
- 8 payload columns (`ctrl*`, trig params, amplitudes)

Selector index for gate id `g`:

- `1..=6` → selector `g - 1`
- `7` (CNOT) → `6`
- `8` (CZ) → `7`
- `9` (CCNOT) → `8`
- `10..=12` (RX/RY/RZ) → `9`

## Fixed-Point Mapping

- `FIXED_POINT_SCALE = 10_000.0`
- Floating values are rounded to integer before Mersenne31 mapping.
- Control columns are written as discrete `0` / `1` before AIR ingestion.

Any backend (`CPU` / `wgpu`) must preserve this exact mapping rule for proof compatibility.

## Proof transcript (v1)

```text
<sub_task_id><_M31_QUANTUM_AIR_V1_><circuit_id\0><node_id\0><slice_id\0><output_hash\0>
<trace_row_count: u32 LE><trace f64 LE bytes><air_sum: u32 LE><boundary u32×4>
```

The verifier re-expands the embedded trace, recomputes `air_sum`, and checks `air_sum == 0` plus boundary amplitudes.
