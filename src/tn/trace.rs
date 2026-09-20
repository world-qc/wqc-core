//! STARK execution trace emission during MPS gate-by-gate contraction.

use std::collections::HashMap;

use crate::engine::{EngineError, Gate};
use ndarray::Array1;
use num_complex::Complex64;
use p3_field::PrimeCharacteristicRing;
use p3_mersenne_31::Mersenne31;
use wqc_stark_engine::air::{f64_to_m31, m31_to_f64, AirConstants};
use wqc_stark_engine::trace_spec::TRACE_WIDTH;

use super::mps::MpsState;

/// Carried AIR amplitudes for a logical target (fixed-point shadow of the locked subspace).
#[derive(Clone, Copy, Debug)]
struct AirAmpState {
    v0_re: f64,
    v0_im: f64,
    v1_re: f64,
    v1_im: f64,
    subspace: Option<usize>,
}

/// Gate-by-gate MPS contraction with pre/post trace rows (11-column schema).
pub fn execute_with_trace(
    qubit_count: usize,
    gates: &[Gate],
    state: &mut MpsState,
) -> Result<Vec<f64>, EngineError> {
    if qubit_count != state.qubit_count {
        return Err(EngineError::MismatchedRegister);
    }

    let mut trace = Vec::new();
    let mut air_amps: HashMap<usize, AirAmpState> = HashMap::new();
    let mut i = 0;
    while i < gates.len() {
        let gate = &gates[i];
        if matches!(gate, Gate::MEASURE(_) | Gate::RESET(_) | Gate::IF(_)) {
            i += 1;
            continue;
        }

        if let Some(run) = consecutive_h_run(&gates[i..]) {
            let traced = run % 2;
            let silent = run - traced;
            for gate in &gates[i..i + silent] {
                state.apply_gate(gate)?;
                // Physics advanced without AIR rows — drop carried amps for that target.
                air_amps.remove(&gate_logical_target(gate));
            }
            if traced == 1 {
                emit_gate_trace(state, gates[i + silent].clone(), &mut trace, &mut air_amps)?;
            }
            i += run;
            continue;
        }

        if let Some((run, angle_sum)) = consecutive_rotation_run(&gates[i..]) {
            if is_identity_rotation(angle_sum) {
                for gate in &gates[i..i + run] {
                    state.apply_gate(gate)?;
                    air_amps.remove(&gate_logical_target(gate));
                }
            } else {
                // Emit one net-angle rotation so intermediate both-nonzero states are not
                // fixed-point constrained gate-by-gate (matches H-fold motivation).
                let net = match &gates[i] {
                    Gate::RX(t, _) => Gate::RX(*t, angle_sum),
                    Gate::RY(t, _) => Gate::RY(*t, angle_sum),
                    Gate::RZ(t, _) => Gate::RZ(*t, angle_sum),
                    _ => unreachable!("rotation run"),
                };
                emit_gate_trace(state, net, &mut trace, &mut air_amps)?;
            }
            i += run;
            continue;
        }

        emit_gate_trace(state, gate.clone(), &mut trace, &mut air_amps)?;
        i += 1;
    }

    let terminal_target = gates
        .iter()
        .rev()
        .find(|g| !matches!(g, Gate::MEASURE(_) | Gate::RESET(_) | Gate::IF(_)))
        .map(gate_logical_target)
        .unwrap_or(0);
    push_terminal_trace_row(state, terminal_target, &air_amps, &mut trace);
    apply_transition_links(&mut trace);

    Ok(trace)
}

fn consecutive_h_run(gates: &[Gate]) -> Option<usize> {
    let Gate::H(target) = gates.first()? else {
        return None;
    };
    let mut run = 1usize;
    while run < gates.len() {
        match &gates[run] {
            Gate::H(t) if t == target => run += 1,
            _ => break,
        }
    }
    Some(run)
}

fn consecutive_rotation_run(gates: &[Gate]) -> Option<(usize, f64)> {
    let (target, first_angle, axis) = match gates.first()? {
        Gate::RX(t, a) => (*t, *a, 0u8),
        Gate::RY(t, a) => (*t, *a, 1u8),
        Gate::RZ(t, a) => (*t, *a, 2u8),
        _ => return None,
    };
    let mut run = 1usize;
    let mut angle_sum = first_angle;
    while run < gates.len() {
        match (&gates[run], axis) {
            (Gate::RX(t, a), 0) if *t == target => {
                angle_sum += *a;
                run += 1;
            }
            (Gate::RY(t, a), 1) if *t == target => {
                angle_sum += *a;
                run += 1;
            }
            (Gate::RZ(t, a), 2) if *t == target => {
                angle_sum += *a;
                run += 1;
            }
            _ => break,
        }
    }
    Some((run, angle_sum))
}

fn is_identity_rotation(angle_sum: f64) -> bool {
    let reduced = angle_sum.rem_euclid(std::f64::consts::TAU);
    reduced.abs() < 1e-9 || (std::f64::consts::TAU - reduced).abs() < 1e-9
}

fn emit_gate_trace(
    state: &mut MpsState,
    gate: Gate,
    trace: &mut Vec<f64>,
    air_amps: &mut HashMap<usize, AirAmpState>,
) -> Result<(), EngineError> {
    let logical_target = gate_logical_target(&gate);
    let (pre, ctrl_active, ctrl_active_2, p_cos, p_sin) =
        resolve_pre_amps(state, &gate, logical_target, air_amps);
    push_trace_row(
        trace,
        gate.to_stark_id().unwrap_or(0.0),
        ctrl_active,
        ctrl_active_2,
        p_cos,
        p_sin,
        pre.v0_re,
        pre.v0_im,
        pre.v1_re,
        pre.v1_im,
        logical_target as f64,
    );

    state.apply_gate(&gate)?;

    let post = air_exact_post_amps(&gate, &pre, ctrl_active, ctrl_active_2, p_cos, p_sin);
    push_trace_row(
        trace,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        post.v0_re,
        post.v0_im,
        post.v1_re,
        post.v1_im,
        logical_target as f64,
    );
    air_amps.insert(logical_target, post);
    Ok(())
}

fn resolve_pre_amps(
    state: &MpsState,
    gate: &Gate,
    logical_target: usize,
    air_amps: &HashMap<usize, AirAmpState>,
) -> (AirAmpState, f64, f64, f64, f64) {
    let (ctrl_prob_1, ctrl_prob_2) = control_probabilities(state, gate);
    let ctrl_active = if ctrl_prob_1 > 0.5 { 1.0 } else { 0.0 };
    let ctrl_active_2 = if ctrl_prob_2 > 0.5 { 1.0 } else { 0.0 };
    let (_, p_cos, p_sin) = gate.to_stark_payload(ctrl_active > 0.5);

    if let Some(prev) = air_amps.get(&logical_target) {
        return (*prev, ctrl_active, ctrl_active_2, p_cos, p_sin);
    }

    let pre = if state.qubit_count <= 16 {
        let (v0_re, v0_im, v1_re, v1_im, s) =
            sample_target_from_dense(&state.to_dense_state(), logical_target, None);
        AirAmpState {
            v0_re,
            v0_im,
            v1_re,
            v1_im,
            subspace: Some(s),
        }
    } else {
        let (a0, a1) = state.site_amplitudes(logical_target);
        AirAmpState {
            v0_re: a0.re,
            v0_im: a0.im,
            v1_re: a1.re,
            v1_im: a1.im,
            subspace: None,
        }
    };
    (pre, ctrl_active, ctrl_active_2, p_cos, p_sin)
}

/// Post amplitudes that satisfy the host AIR transition exactly (Mersenne31 twin of constraints).
fn air_exact_post_amps(
    gate: &Gate,
    pre: &AirAmpState,
    ctrl_active: f64,
    ctrl_active_2: f64,
    p_cos: f64,
    p_sin: f64,
) -> AirAmpState {
    let c = AirConstants::mersenne31_defaults();
    let v0_re = f64_to_m31(pre.v0_re);
    let v0_im = f64_to_m31(pre.v0_im);
    let v1_re = f64_to_m31(pre.v1_re);
    let v1_im = f64_to_m31(pre.v1_im);
    let pc = f64_to_m31(p_cos);
    let ps = f64_to_m31(p_sin);
    let one = c.one;
    let two = c.two;
    let ca = Mersenne31::new(ctrl_active.round() as u32);
    let ca2 = Mersenne31::new(ctrl_active_2.round() as u32);

    let (n0_re, n0_im, n1_re, n1_im) = match gate {
        Gate::X(_) => (v1_re, v1_im, v0_re, v0_im),
        Gate::Y(_) => (
            v1_im,
            Mersenne31::ZERO - v1_re,
            Mersenne31::ZERO - v0_im,
            v0_re,
        ),
        Gate::Z(_) => (
            v0_re,
            v0_im,
            Mersenne31::ZERO - v1_re,
            Mersenne31::ZERO - v1_im,
        ),
        Gate::H(_) => {
            let s = c.inv_sqrt2 * c.scale_inverse;
            (
                (v0_re + v1_re) * s,
                (v0_im + v1_im) * s,
                (v0_re - v1_re) * s,
                (v0_im - v1_im) * s,
            )
        }
        Gate::S(_) => (v0_re, v0_im, Mersenne31::ZERO - v1_im, v1_re),
        Gate::T(_) => {
            let s = c.inv_sqrt2 * c.scale_inverse;
            (v0_re, v0_im, (v1_re - v1_im) * s, (v1_re + v1_im) * s)
        }
        Gate::CNOT(..) => {
            let inactive = one - ca;
            (
                inactive * v0_re + ca * v1_re,
                inactive * v0_im + ca * v1_im,
                inactive * v1_re + ca * v0_re,
                inactive * v1_im + ca * v0_im,
            )
        }
        Gate::CZ(..) => {
            let phase = one - (two * ca);
            (v0_re, v0_im, v1_re * phase, v1_im * phase)
        }
        Gate::CCNOT(..) => {
            let active = ca * ca2;
            let inactive = one - active;
            (
                inactive * v0_re + active * v1_re,
                inactive * v0_im + active * v1_im,
                inactive * v1_re + active * v0_re,
                inactive * v1_im + active * v0_im,
            )
        }
        Gate::RX(..) => {
            let inv = c.scale_inverse;
            (
                (v0_re * pc + v1_im * ps) * inv,
                (v0_im * pc - v1_re * ps) * inv,
                (v1_re * pc + v0_im * ps) * inv,
                (v1_im * pc - v0_re * ps) * inv,
            )
        }
        Gate::RY(..) => {
            let inv = c.scale_inverse;
            (
                (v0_re * pc - v1_re * ps) * inv,
                (v0_im * pc - v1_im * ps) * inv,
                (v1_re * pc + v0_re * ps) * inv,
                (v1_im * pc + v0_im * ps) * inv,
            )
        }
        Gate::RZ(..) => {
            let inv = c.scale_inverse;
            (
                (v0_re * pc + v0_im * ps) * inv,
                (v0_im * pc - v0_re * ps) * inv,
                (v1_re * pc - v1_im * ps) * inv,
                (v1_im * pc + v1_re * ps) * inv,
            )
        }
        Gate::MEASURE(_) | Gate::RESET(_) | Gate::IF(_) => (v0_re, v0_im, v1_re, v1_im),
    };

    AirAmpState {
        v0_re: m31_to_f64(n0_re),
        v0_im: m31_to_f64(n0_im),
        v1_re: m31_to_f64(n1_re),
        v1_im: m31_to_f64(n1_im),
        subspace: pre.subspace,
    }
}

fn apply_transition_links(trace: &mut [f64]) {
    let row_count = trace.len() / TRACE_WIDTH;
    for row in 0..row_count {
        let base = row * TRACE_WIDTH;
        let link = if row + 1 < row_count {
            let curr_target = trace[base + 9];
            let next_target = trace[(row + 1) * TRACE_WIDTH + 9];
            if (curr_target - next_target).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        trace[base + 10] = link;
    }
}

fn gate_logical_target(gate: &Gate) -> usize {
    match gate {
        Gate::X(t) | Gate::Y(t) | Gate::Z(t) | Gate::H(t) | Gate::S(t) | Gate::T(t) => *t,
        Gate::CNOT(_, t) | Gate::CZ(_, t) => *t,
        Gate::CCNOT(_, _, t) => *t,
        Gate::RX(t, _) | Gate::RY(t, _) | Gate::RZ(t, _) => *t,
        Gate::MEASURE(spec) => spec.qubit,
        Gate::RESET(t) => *t,
        Gate::IF(params) => gate_logical_target(&params.gate),
    }
}

fn sample_target_amplitudes(state: &MpsState, phys_target: usize) -> (f64, f64, f64, f64) {
    if state.qubit_count <= 16 {
        let (v0_re, v0_im, v1_re, v1_im, _) =
            sample_target_from_dense(&state.to_dense_state(), phys_target, None);
        return (v0_re, v0_im, v1_re, v1_im);
    }
    let (a0, a1) = state.site_amplitudes(phys_target);
    (a0.re, a0.im, a1.re, a1.im)
}

/// Samples the densest (or locked) computational-basis pair that differs only on `phys_target`.
///
/// Returns `(v0_re, v0_im, v1_re, v1_im, subspace_index)`. When `locked_subspace` is `Some(s)`,
/// that ancillary basis is used so a pre/post gate pair sees the same 2D slice (unitary on the
/// target acts within each slice independently). Independent max-prob picks across the gate
/// can jump slices and break AIR rotation / H constraints.
fn sample_target_from_dense(
    state: &Array1<Complex64>,
    phys_target: usize,
    locked_subspace: Option<usize>,
) -> (f64, f64, f64, f64, usize) {
    let subspace_limit = state.len() >> 1;

    let pick = |s: usize| -> (usize, usize, f64) {
        let low_mask = (1 << phys_target) - 1;
        let high_bits = (s & !low_mask) << 1;
        let low_bits = s & low_mask;
        let idx_v0 = high_bits | low_bits;
        let idx_v1 = idx_v0 | (1 << phys_target);
        let p0 = state[idx_v0].re * state[idx_v0].re + state[idx_v0].im * state[idx_v0].im;
        let p1 = state[idx_v1].re * state[idx_v1].re + state[idx_v1].im * state[idx_v1].im;
        (idx_v0, idx_v1, p0 + p1)
    };

    let (best_v0_idx, best_v1_idx, best_s) = if let Some(s) = locked_subspace {
        let s = s.min(subspace_limit.saturating_sub(1));
        let (v0, v1, _) = pick(s);
        (v0, v1, s)
    } else {
        let mut max_pair_prob = -1.0;
        let mut best_v0_idx = 0;
        let mut best_v1_idx = 0;
        let mut best_s = 0usize;
        for s in 0..subspace_limit {
            let (idx_v0, idx_v1, combined_prob) = pick(s);
            if combined_prob > max_pair_prob {
                max_pair_prob = combined_prob;
                best_v0_idx = idx_v0;
                best_v1_idx = idx_v1;
                best_s = s;
            }
        }
        (best_v0_idx, best_v1_idx, best_s)
    };

    (
        state[best_v0_idx].re,
        state[best_v0_idx].im,
        state[best_v1_idx].re,
        state[best_v1_idx].im,
        best_s,
    )
}

fn control_probabilities(state: &MpsState, gate: &Gate) -> (f64, f64) {
    if state.qubit_count <= 16 {
        let dense = state.to_dense_state();
        return match gate {
            Gate::CNOT(c, _) | Gate::CZ(c, _) => {
                let mut prob = 0.0;
                for (idx, amplitude) in dense.iter().enumerate() {
                    if (idx >> c) & 1 == 1 {
                        prob += amplitude.re * amplitude.re + amplitude.im * amplitude.im;
                    }
                }
                (prob, 0.0)
            }
            Gate::CCNOT(c1, c2, _) => {
                let mut prob_1 = 0.0;
                let mut prob_2 = 0.0;
                for (idx, amplitude) in dense.iter().enumerate() {
                    let p = amplitude.re * amplitude.re + amplitude.im * amplitude.im;
                    if (idx >> c1) & 1 == 1 {
                        prob_1 += p;
                    }
                    if (idx >> c2) & 1 == 1 {
                        prob_2 += p;
                    }
                }
                (prob_1, prob_2)
            }
            _ => (0.0, 0.0),
        };
    }
    match gate {
        Gate::CNOT(c, _) | Gate::CZ(c, _) => (state.control_probability(*c), 0.0),
        Gate::CCNOT(c1, c2, _) => (
            state.control_probability(*c1),
            state.control_probability(*c2),
        ),
        _ => (0.0, 0.0),
    }
}

#[allow(clippy::too_many_arguments)]
fn push_trace_row(
    trace: &mut Vec<f64>,
    gate_id: f64,
    ctrl_active: f64,
    ctrl_active_2: f64,
    p_cos: f64,
    p_sin: f64,
    v0_re: f64,
    v0_im: f64,
    v1_re: f64,
    v1_im: f64,
    target_qubit: f64,
) {
    trace.push(gate_id);
    trace.push(ctrl_active);
    trace.push(ctrl_active_2);
    trace.push(p_cos);
    trace.push(p_sin);
    trace.push(v0_re);
    trace.push(v0_im);
    trace.push(v1_re);
    trace.push(v1_im);
    trace.push(target_qubit);
    trace.push(0.0);
}

fn push_terminal_trace_row(
    state: &MpsState,
    logical_target: usize,
    air_amps: &HashMap<usize, AirAmpState>,
    trace: &mut Vec<f64>,
) {
    let (v0_re, v0_im, v1_re, v1_im) = if let Some(prev) = air_amps.get(&logical_target) {
        (prev.v0_re, prev.v0_im, prev.v1_re, prev.v1_im)
    } else {
        sample_target_amplitudes(state, logical_target)
    };
    push_trace_row(
        trace,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        v0_re,
        v0_im,
        v1_re,
        v1_im,
        logical_target as f64,
    );
}
