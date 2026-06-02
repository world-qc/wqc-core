# WQC Trace Spec (v1)

This document defines the canonical STARK trace contract between `wqc-core` executors and `wqc-stark-core`.

## Goals

- Keep proof logic independent from execution backend (`CPU` now, `wgpu/WebGPU` later).
- Allow backend migration without rewriting AIR rules.
- Make failures debuggable by freezing row/column semantics.

## Row Layout (`TRACE_WIDTH = 10`)

Per-step row emitted by `execute_with_trace`:

1. `gate_id`
2. `ctrl_active`
3. `ctrl_active_2`
4. `p_cos`
5. `p_sin`
6. `v0_re`
7. `v0_im`
8. `v1_re`
9. `v1_im`
10. `padding` (currently `0`)

The final boundary row reuses the same 10-column shape.

## AIR Expansion (`AIR_WIDTH = 18`)

`wqc-stark-core` expands each row into:

- 1 gate id column
- 9 selector columns (`X,Y,Z,H,S,T,CTRL,CCNOT,ROT`)
- 8 payload columns (`ctrl*`, trig params, amplitudes)

## Fixed-Point Mapping

- `FIXED_POINT_SCALE = 10_000.0`
- Floating values are rounded to integer before Mersenne31 mapping.

Any backend (`CPU` / `wgpu`) must preserve this exact mapping rule for proof compatibility.
