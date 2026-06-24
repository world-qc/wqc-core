//! STARK execution trace emission during TN gate-by-gate contraction.

use crate::engine::{EngineError, Gate};
use ndarray::Array1;
use num_complex::Complex64;
use wqc_stark_engine::trace_spec::TRACE_WIDTH;

use super::dense::DenseTnState;

/// Gate-by-gate contraction with pre/post trace rows (11-column schema).
pub fn execute_with_trace(
    qubit_count: usize,
    gates: &[Gate],
    state: &mut DenseTnState,
) -> Result<Vec<f64>, EngineError> {
    use crate::engine::EngineError;

    if qubit_count != state.qubit_count {
        return Err(EngineError::MismatchedRegister);
    }

    let total_rows = gates.len() * 2 + 1;
    let mut trace = Vec::with_capacity(total_rows * TRACE_WIDTH);

    for gate in gates {
        let logical_target = gate_logical_target(gate);
        push_gate_snapshot_row(&state.state, gate, logical_target, &mut trace);
        state.apply_gate(gate);
        push_post_gate_row(&state.state, logical_target, &mut trace);
    }

    let terminal_target = gates.last().map(gate_logical_target).unwrap_or(0);
    push_terminal_trace_row(&state.state, terminal_target, &mut trace);
    apply_transition_links(&mut trace);

    Ok(trace)
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
    }
}

fn sample_target_amplitudes(
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

fn push_gate_snapshot_row(
    state: &Array1<Complex64>,
    gate: &Gate,
    logical_target: usize,
    trace: &mut Vec<f64>,
) {
    let gate_id = gate.to_stark_id().unwrap_or(0.0);
    let phys_target = logical_target;

    let (ctrl_prob_1, ctrl_prob_2) = match gate {
        Gate::CNOT(c, _) | Gate::CZ(c, _) => {
            let phys_ctrl = *c;
            let mut prob = 0.0;
            for (idx, amplitude) in state.iter().enumerate() {
                if (idx >> phys_ctrl) & 1 == 1 {
                    prob += amplitude.re * amplitude.re + amplitude.im * amplitude.im;
                }
            }
            (prob, 0.0)
        }
        Gate::CCNOT(c1, c2, _) => {
            let mut prob_1 = 0.0;
            let mut prob_2 = 0.0;
            for (idx, amplitude) in state.iter().enumerate() {
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

    let ctrl_active = if ctrl_prob_1 > 0.5 { 1.0 } else { 0.0 };
    let ctrl_active_2 = if ctrl_prob_2 > 0.5 { 1.0 } else { 0.0 };
    let (_, p_cos, p_sin) = gate.to_stark_payload(ctrl_active > 0.5);
    let (v0_re, v0_im, v1_re, v1_im) = sample_target_amplitudes(state, phys_target);

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

fn push_post_gate_row(state: &Array1<Complex64>, logical_target: usize, trace: &mut Vec<f64>) {
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

fn push_terminal_trace_row(state: &Array1<Complex64>, logical_target: usize, trace: &mut Vec<f64>) {
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
