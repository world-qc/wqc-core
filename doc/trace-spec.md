# WQC Trace Spec (v2 — multi-target AIR)

This document defines the canonical STARK trace contract between `wqc-core` executors and `wqc-stark-core`.

## Goals

- Keep proof logic independent from execution backend (`CPU` now, `wgpu/WebGPU` later).
- Allow backend migration without rewriting AIR rules.
- Make failures debuggable by freezing row/column semantics.
- Support **multi-target gate sequences** (e.g. `H(0)` → `CCNOT(0,1,2)`) via per-gate pre/post rows and transition links.

## Compact-register slice contract

Sub-tasks use a **compact register** after tensor slicing:

| Concept | Rule |
| :--- | :--- |
| Register size | `qubit_count = original_qubit_count - len(slice_assignments)` |
| Circuit indices | Local wires `0 .. qubit_count-1` (orchestrator remaps after prune) |
| Initial state | `\|0…0⟩` on the compact register |
| Scalar output | Amplitude at **basis index `0`** (`complex_result`); assignments are metadata + STARK binding, not a readout bit-mask |

The trace below is generated from the **remapped** gate list the node receives. Any CPU or `wgpu` backend must execute that list on the compact register and use the same `state[0]` readout rule for hash alignment with the orchestrator.

## Row layout (`TRACE_WIDTH = 11`)

Each gate emits **two trace rows** on the gate's **target qubit**:

1. **Pre-gate row** — `gate_id` = active gate; amplitudes sampled before applying the gate.
2. **Post-gate row** — `gate_id = 0`; amplitudes sampled after applying the gate (same target wire).

After all gates, a **terminal boundary row** (`gate_id = 0`) samples the last gate's target qubit with `transition_link = 0`.

An empty circuit emits a single terminal boundary row.

### Column indices

| Col | Name | Description |
|-----|------|-------------|
| 0 | `gate_id` | See gate id table below |
| 1 | `ctrl_active` | Discrete `0.0` or `1.0` (marginal control probability thresholded at `0.5`) |
| 2 | `ctrl_active_2` | Second control for CCNOT (same discretization) |
| 3 | `p_cos` | Rotation parameter cosine |
| 4 | `p_sin` | Rotation parameter sine |
| 5 | `v0_re` | Target-qubit `\|0⟩` amplitude (real) |
| 6 | `v0_im` | Target-qubit `\|0⟩` amplitude (imag) |
| 7 | `v1_re` | Target-qubit `\|1⟩` amplitude (real) |
| 8 | `v1_im` | Target-qubit `\|1⟩` amplitude (imag) |
| 9 | `target_qubit` | Logical wire index sampled in columns 5–8 |
| 10 | `transition_link` | `1.0` if the **next** row continues the same target wire; `0.0` otherwise |

`transition_link` on row `i` is set by the executor after the full trace is built:

- Pre → post for the same gate: `link = 1`
- Post → next gate's pre when targets match: `link = 1`
- Post → next gate's pre when targets differ (cross-wire): `link = 0` on the post row
- Terminal row: `link = 0`

AIR transition constraints apply only when `transition_link = 1` on the current row.

### Consecutive unary gates (trace fold)

Fixed-point Hadamard / rotation constraints can fail when **both** target amplitudes are
non-zero (e.g. `H(0); H(0)` on |0⟩ passes through |+⟩). Executors therefore **fold** runs
before emitting trace rows:

| Pattern | MPS execution | Trace emission |
| --- | --- | --- |
| `H(t)^n` (n even) | apply all n gates | no rows (H∘H = I) |
| `H(t)^n` (n odd) | apply all n gates | one H pre/post pair |
| `RX(t,θ₁)…RX(t,θₖ)` with Σθᵢ ≡ 0 (mod 2π) | apply all | no rows (net identity) |
| `RX` run with non-zero net angle | apply all | one pre/post pair per gate (unchanged) |

Physics (MPS state) always applies the full gate list. Trace folding is an AIR-encoding
detail only; proofs bind `output_result_hash` from the executed state.

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
| 0 | Padding / post-gate / terminal boundary |

## AIR expansion (`AIR_WIDTH = 21`)

`wqc-stark-core` expands each row into:

- 1 gate id column
- 10 selector columns (`X, Y, Z, H, S, T, CNOT, CZ, CCNOT, ROT`)
- 8 payload columns (`ctrl×2`, trig params, amplitudes×4)
- `target_qubit` and `transition_link`

Selector index for gate id `g`:

- `1..=6` → selector `g - 1`
- `7` (CNOT) → `6`
- `8` (CZ) → `7`
- `9` (CCNOT) → `8`
- `10..=12` (RX/RY/RZ) → `9`

## Fixed-point mapping

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

**Breaking change**: traces produced under v1 (10 columns, one row per gate) do not verify against the v2 AIR. Devnet nodes and stark-engine must be upgraded together.

## Status

Trace alignment between `wqc-core` and `wqc-stark-core` is complete for multi-target circuits. See
`wqc-stark-engine/docs/PHASE2_TRACE_ALIGNMENT.md` for the checklist and test coverage.
