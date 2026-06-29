//! Phase C1: mid-circuit measurement, RESET, and classical IF gates.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::engine::{EngineError, Gate, IfParams, MeasureParams};
use crate::noise::NoiseModel;
use crate::sample::{SampleResult, outcome_key_from_classical};
use crate::tn::dense::DenseTnState;

/// Returns true when the circuit needs trajectory sampling (not terminal-block semantics).
pub fn uses_mid_circuit_semantics(gates: &[Gate]) -> bool {
    gates.iter().any(|g| matches!(g, Gate::RESET(_) | Gate::IF(_)))
        || has_unitary_after_first_measure(gates)
}

fn has_unitary_after_first_measure(gates: &[Gate]) -> bool {
    let Some(idx) = gates.iter().position(|g| matches!(g, Gate::MEASURE(_))) else {
        return false;
    };
    gates[idx + 1..].iter().any(|g| is_unitary_or_conditional(g))
}

fn is_unitary_or_conditional(gate: &Gate) -> bool {
    !matches!(gate, Gate::MEASURE(_))
}

/// Collects every MEASURE in program order.
pub fn collect_measures(gates: &[Gate]) -> Vec<MeasureParams> {
    gates
        .iter()
        .filter_map(|g| match g {
            Gate::MEASURE(spec) => Some(spec.clone()),
            _ => None,
        })
        .collect()
}

/// Gates included in the unitary STARK trace (non-destructive / non-classical controls).
pub fn extract_unitary_gates_for_proof(gates: &[Gate]) -> Vec<Gate> {
    gates
        .iter()
        .filter(|g| matches!(g, Gate::H(_) | Gate::X(_) | Gate::Y(_) | Gate::Z(_) | Gate::S(_) | Gate::T(_)
            | Gate::CNOT(_, _) | Gate::CZ(_, _) | Gate::CCNOT(_, _, _)
            | Gate::RX(_, _) | Gate::RY(_, _) | Gate::RZ(_, _)))
        .cloned()
        .collect()
}

/// Phase C circuit validation for `sample_counts`.
pub fn validate_phase_c_sample_circuit(
    gates: &[Gate],
    classical_bit_count: usize,
) -> Result<Vec<MeasureParams>, EngineError> {
    if classical_bit_count == 0 {
        return Err(EngineError::ExecutionFailed(
            "classical_bit_count must be > 0 for sample_counts".into(),
        ));
    }

    let mut measured_qubits = vec![false; 64]; // resized per qubit check
    let mut measures = Vec::new();

    for (i, gate) in gates.iter().enumerate() {
        match gate {
            Gate::MEASURE(spec) => {
                if spec.cbit >= classical_bit_count {
                    return Err(EngineError::ExecutionFailed(format!(
                        "gate {i} MEASURE cbit {} out of bounds (classical_bit_count {classical_bit_count})",
                        spec.cbit
                    )));
                }
                if spec.qubit >= measured_qubits.len() {
                    measured_qubits.resize(spec.qubit + 1, false);
                }
                measured_qubits[spec.qubit] = true;
                measures.push(spec.clone());
            }
            Gate::RESET(q) => {
                if *q >= measured_qubits.len() {
                    measured_qubits.resize(*q + 1, false);
                }
                measured_qubits[*q] = false;
            }
            Gate::IF(params) => {
                if params.cbit >= classical_bit_count {
                    return Err(EngineError::ExecutionFailed(format!(
                        "gate {i} IF cbit {} out of bounds",
                        params.cbit
                    )));
                }
                if params.value > 1 {
                    return Err(EngineError::ExecutionFailed(format!(
                        "gate {i} IF value must be 0 or 1"
                    )));
                }
                ensure_gate_avoids_measured(i, &params.gate, &measured_qubits)?;
            }
            other => {
                ensure_gate_avoids_measured(i, other, &measured_qubits)?;
            }
        }
    }

    if measures.is_empty() {
        return Err(EngineError::ExecutionFailed(
            "sample_counts requires at least one MEASURE gate".into(),
        ));
    }

    Ok(measures)
}

fn ensure_gate_avoids_measured(
    gate_index: usize,
    gate: &Gate,
    measured_qubits: &[bool],
) -> Result<(), EngineError> {
    for q in gate_qubit_operands(gate) {
        if q < measured_qubits.len() && measured_qubits[q] {
            return Err(EngineError::ExecutionFailed(format!(
                "gate {gate_index} operates on measured qubit {q} without intervening RESET"
            )));
        }
    }
    Ok(())
}

fn gate_qubit_operands(gate: &Gate) -> Vec<usize> {
    match gate {
        Gate::H(q) | Gate::X(q) | Gate::Y(q) | Gate::Z(q) | Gate::S(q) | Gate::T(q)
        | Gate::RX(q, _) | Gate::RY(q, _) | Gate::RZ(q, _) | Gate::RESET(q) => vec![*q],
        Gate::CNOT(c, t) | Gate::CZ(c, t) => vec![*c, *t],
        Gate::CCNOT(c1, c2, t) => vec![*c1, *c2, *t],
        Gate::MEASURE(spec) => vec![spec.qubit],
        Gate::IF(params) => gate_qubit_operands(&params.gate),
    }
}

/// Deterministic shot-by-shot simulation with optional noise (C3).
pub fn sample_mid_circuit_measurements(
    gates: &[Gate],
    qubit_count: usize,
    classical_bit_count: usize,
    shots: u64,
    seed: u64,
    noise: Option<&NoiseModel>,
) -> Result<SampleResult, EngineError> {
    if qubit_count > 20 {
        return Err(EngineError::ExecutionFailed(
            "mid-circuit trajectory sampling requires qubit_count <= 20".into(),
        ));
    }

    validate_phase_c_sample_circuit(gates, classical_bit_count)?;

    let mut counts = std::collections::BTreeMap::new();
    for shot in 0..shots {
        let shot_seed = seed.wrapping_add(shot);
        let outcome = simulate_one_shot(gates, qubit_count, classical_bit_count, shot_seed, noise)?;
        *counts.entry(outcome).or_insert(0) += 1;
    }

    Ok(SampleResult { counts, shots })
}

fn simulate_one_shot(
    gates: &[Gate],
    qubit_count: usize,
    classical_bit_count: usize,
    seed: u64,
    noise: Option<&NoiseModel>,
) -> Result<String, EngineError> {
    let mut state = DenseTnState::try_new(qubit_count)?;
    let mut classical = vec![0u8; classical_bit_count];
    let mut rng = StdRng::seed_from_u64(seed);

    for gate in gates {
        match gate {
            Gate::MEASURE(spec) => {
                let (p0, p1) = z_marginal(&state, spec.qubit);
                let mut outcome = if rng.gen::<f64>() < p0 / (p0 + p1).max(1e-30) {
                    0
                } else {
                    1
                };
                if let Some(noise) = noise {
                    outcome = noise.apply_readout(outcome, &mut rng);
                }
                collapse_z(&mut state, spec.qubit, outcome);
                classical[spec.cbit] = outcome;
            }
            Gate::RESET(q) => reset_qubit(&mut state, *q),
            Gate::IF(params) => {
                if classical[params.cbit] == params.value {
                    apply_noisy_unitary(&mut state, &params.gate, noise, &mut rng);
                }
            }
            unitary => {
                apply_noisy_unitary(&mut state, unitary, noise, &mut rng);
            }
        }
    }

    Ok(outcome_key_from_classical(&classical))
}

fn apply_noisy_unitary(
    state: &mut DenseTnState,
    gate: &Gate,
    noise: Option<&NoiseModel>,
    rng: &mut StdRng,
) {
    state.apply_gate(gate);
    if let Some(noise) = noise {
        if let Some(q) = single_qubit_target(gate) {
            noise.apply_depolarizing(state, q, rng);
        }
    }
}

fn single_qubit_target(gate: &Gate) -> Option<usize> {
    match gate {
        Gate::H(q) | Gate::X(q) | Gate::Y(q) | Gate::Z(q) | Gate::S(q) | Gate::T(q)
        | Gate::RX(q, _) | Gate::RY(q, _) | Gate::RZ(q, _) => Some(*q),
        _ => None,
    }
}

fn z_marginal(state: &DenseTnState, qubit: usize) -> (f64, f64) {
    let mut p0 = 0.0;
    let mut p1 = 0.0;
    for (i, amp) in state.state.iter().enumerate() {
        let prob = amp.norm_sqr();
        if (i >> qubit) & 1 == 0 {
            p0 += prob;
        } else {
            p1 += prob;
        }
    }
    (p0, p1)
}

fn collapse_z(state: &mut DenseTnState, qubit: usize, outcome: u8) {
    for (i, amp) in state.state.iter_mut().enumerate() {
        if ((i >> qubit) & 1) as u8 != outcome {
            *amp = num_complex::Complex64::new(0.0, 0.0);
        }
    }
    normalize_dense(state);
}

fn reset_qubit(state: &mut DenseTnState, qubit: usize) {
    collapse_z(state, qubit, 0);
}

fn normalize_dense(state: &mut DenseTnState) {
    let norm_sq: f64 = state.state.iter().map(|a| a.norm_sqr()).sum();
    if norm_sq <= 0.0 {
        return;
    }
    let scale = norm_sq.sqrt();
    for amp in state.state.iter_mut() {
        *amp /= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Gate;

    #[test]
    fn mid_circuit_measure_then_conditional_x() {
        let gates = vec![
            Gate::H(0),
            Gate::MEASURE(MeasureParams { qubit: 0, cbit: 0 }),
            Gate::IF(IfParams {
                cbit: 0,
                value: 1,
                gate: Box::new(Gate::X(1)),
            }),
            Gate::MEASURE(MeasureParams { qubit: 1, cbit: 1 }),
        ];
        let sample = sample_mid_circuit_measurements(&gates, 2, 2, 1024, 7, None).expect("sample");
        assert_eq!(sample.shots, 1024);
        assert!(sample.counts.contains_key("01") || sample.counts.contains_key("11"));
    }

    #[test]
    fn reset_allows_reuse_of_measured_qubit() {
        let gates = vec![
            Gate::H(0),
            Gate::MEASURE(MeasureParams { qubit: 0, cbit: 0 }),
            Gate::RESET(0),
            Gate::H(0),
            Gate::MEASURE(MeasureParams { qubit: 0, cbit: 0 }),
        ];
        assert!(validate_phase_c_sample_circuit(&gates, 1).is_ok());
    }

    #[test]
    fn unitary_on_measured_qubit_without_reset_is_rejected() {
        let gates = vec![
            Gate::MEASURE(MeasureParams { qubit: 0, cbit: 0 }),
            Gate::H(0),
        ];
        let err = validate_phase_c_sample_circuit(&gates, 1).unwrap_err().to_string();
        assert!(err.contains("measured qubit"));
    }
}
