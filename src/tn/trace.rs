//! STARK execution trace emission during MPS gate-by-gate contraction.

use crate::engine::{EngineError, Gate};
use ndarray::Array1;
use num_complex::Complex64;
use wqc_stark_engine::trace_spec::TRACE_WIDTH;

use super::mps::MpsState;

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
            }
            if traced == 1 {
                emit_gate_trace(state, gates[i].clone(), &mut trace)?;
            }
            i += run;
            continue;
        }

        if let Some(run) = consecutive_rx_run(&gates[i..]) {
            let angle_sum = rx_angle_sum(&gates[i..i + run]);
            if is_identity_rotation(angle_sum) {
                for gate in &gates[i..i + run] {
                    state.apply_gate(gate)?;
                }
            } else {
                for gate in &gates[i..i + run] {
                    emit_gate_trace(state, gate.clone(), &mut trace)?;
                }
            }
            i += run;
            continue;
        }

        emit_gate_trace(state, gate.clone(), &mut trace)?;
        i += 1;
    }

    let terminal_target = gates
        .iter()
        .rev()
        .find(|g| !matches!(g, Gate::MEASURE(_) | Gate::RESET(_) | Gate::IF(_)))
        .map(gate_logical_target)
        .unwrap_or(0);
    push_terminal_trace_row(state, terminal_target, &mut trace);
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

fn consecutive_rx_run(gates: &[Gate]) -> Option<usize> {
    let Gate::RX(target, _) = gates.first()? else {
        return None;
    };
    let mut run = 1usize;
    while run < gates.len() {
        match &gates[run] {
            Gate::RX(t, _) if t == target => run += 1,
            _ => break,
        }
    }
    Some(run)
}

fn rx_angle_sum(gates: &[Gate]) -> f64 {
    gates
        .iter()
        .filter_map(|gate| match gate {
            Gate::RX(_, angle) => Some(*angle),
            _ => None,
        })
        .sum()
}

fn is_identity_rotation(angle_sum: f64) -> bool {
    let reduced = angle_sum.rem_euclid(std::f64::consts::TAU);
    reduced.abs() < 1e-9 || (std::f64::consts::TAU - reduced).abs() < 1e-9
}

fn emit_gate_trace(
    state: &mut MpsState,
    gate: Gate,
    trace: &mut Vec<f64>,
) -> Result<(), EngineError> {
    let logical_target = gate_logical_target(&gate);
    push_gate_snapshot_row(state, &gate, logical_target, trace);
    state.apply_gate(&gate)?;
    push_post_gate_row(state, logical_target, trace);
    Ok(())
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
        return sample_target_from_dense(&state.to_dense_state(), phys_target);
    }
    let (a0, a1) = state.site_amplitudes(phys_target);
    (a0.re, a0.im, a1.re, a1.im)
}

fn sample_target_from_dense(
    state: &Array1<Complex64>,
    phys_target: usize,
) -> (f64, f64, f64, f64) {
    let mut max_pair_prob = -1.0;
    let mut best_v0_idx = 0;
    let mut best_v1_idx = 0;
    let subspace_limit = state.len() >> 1;

    for s in 0..subspace_limit {
        let low_mask = (1 << phys_target) - 1;
        let high_bits = (s & !low_mask) << 1;
        let low_bits = s & low_mask;

        let idx_v0 = high_bits | low_bits;
        let idx_v1 = idx_v0 | (1 << phys_target);

        let p0 = state[idx_v0].re * state[idx_v0].re + state[idx_v0].im * state[idx_v0].im;
        let p1 = state[idx_v1].re * state[idx_v1].re + state[idx_v1].im * state[idx_v1].im;
        let combined_prob = p0 + p1;

        if combined_prob > max_pair_prob {
            max_pair_prob = combined_prob;
            best_v0_idx = idx_v0;
            best_v1_idx = idx_v1;
        }
    }

    (
        state[best_v0_idx].re,
        state[best_v0_idx].im,
        state[best_v1_idx].re,
        state[best_v1_idx].im,
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

fn push_gate_snapshot_row(state: &MpsState, gate: &Gate, logical_target: usize, trace: &mut Vec<f64>) {
    let gate_id = gate.to_stark_id().unwrap_or(0.0);
    let (ctrl_prob_1, ctrl_prob_2) = control_probabilities(state, gate);
    let ctrl_active = if ctrl_prob_1 > 0.5 { 1.0 } else { 0.0 };
    let ctrl_active_2 = if ctrl_prob_2 > 0.5 { 1.0 } else { 0.0 };
    let (_, p_cos, p_sin) = gate.to_stark_payload(ctrl_active > 0.5);
    let (v0_re, v0_im, v1_re, v1_im) = sample_target_amplitudes(state, logical_target);

    push_trace_row(
        trace,
        gate_id,
        ctrl_active,
        ctrl_active_2,
        p_cos,
        p_sin,
        v0_re,
        v0_im,
        v1_re,
        v1_im,
        logical_target as f64,
    );
}

fn push_post_gate_row(state: &MpsState, logical_target: usize, trace: &mut Vec<f64>) {
    let (v0_re, v0_im, v1_re, v1_im) = sample_target_amplitudes(state, logical_target);
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

fn push_terminal_trace_row(state: &MpsState, logical_target: usize, trace: &mut Vec<f64>) {
    let (v0_re, v0_im, v1_re, v1_im) = sample_target_amplitudes(state, logical_target);
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
