//! Phase C1: mid-circuit measurement, RESET, and classical IF gates.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use wqc_stark_engine::{canonicalize_terminal_statevector, z_marginal_from_statevector};

use crate::engine::{EngineError, Gate, MeasureParams};
use crate::noise::NoiseModel;
use crate::sample::{outcome_key_from_classical, SampleResult};
use crate::tn::dense::DenseTnState;

/// Returns true when the circuit needs trajectory sampling (not terminal-block semantics).
pub fn uses_mid_circuit_semantics(gates: &[Gate]) -> bool {
    gates
        .iter()
        .any(|g| matches!(g, Gate::RESET(_) | Gate::IF(_)))
        || has_unitary_after_first_measure(gates)
}

fn has_unitary_after_first_measure(gates: &[Gate]) -> bool {
    let Some(idx) = gates.iter().position(|g| matches!(g, Gate::MEASURE(_))) else {
        return false;
    };
    gates[idx + 1..].iter().any(is_unitary_or_conditional)
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

/// One observed MEASURE event during a single trajectory shot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrajectoryMeasureEvent {
    pub gate_index: usize,
    pub qubit: usize,
    pub cbit: usize,
    pub p0: f64,
    pub p1: f64,
    pub outcome: u8,
    /// Dense pre-measure statevector amplitudes (C2c marginal binding).
    pub pre_measure_statevector: Vec<(f64, f64)>,
}

/// One deterministic trajectory shot (seed-fixed).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrajectoryShotTrace {
    pub shot_index: u64,
    pub shot_seed: u64,
    pub final_outcome: String,
    pub classical_bits: Vec<u8>,
    pub measures: Vec<TrajectoryMeasureEvent>,
}

/// Full mid-circuit trajectory trace across all sampled shots.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrajectoryTrace {
    pub shots: u64,
    pub traces: Vec<TrajectoryShotTrace>,
}

/// Gates included in the unitary STARK trace (non-destructive / non-classical controls).
pub fn extract_unitary_gates_for_proof(gates: &[Gate]) -> Vec<Gate> {
    gates
        .iter()
        .filter(|g| {
            matches!(
                g,
                Gate::H(_)
                    | Gate::X(_)
                    | Gate::Y(_)
                    | Gate::Z(_)
                    | Gate::S(_)
                    | Gate::T(_)
                    | Gate::CNOT(_, _)
                    | Gate::CZ(_, _)
                    | Gate::CCNOT(_, _, _)
                    | Gate::RX(_, _)
                    | Gate::RY(_, _)
                    | Gate::RZ(_, _)
            )
        })
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
        Gate::H(q)
        | Gate::X(q)
        | Gate::Y(q)
        | Gate::Z(q)
        | Gate::S(q)
        | Gate::T(q)
        | Gate::RX(q, _)
        | Gate::RY(q, _)
        | Gate::RZ(q, _)
        | Gate::RESET(q) => vec![*q],
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
    let (sample, _trace) = sample_mid_circuit_measurements_with_trace(
        gates,
        qubit_count,
        classical_bit_count,
        shots,
        seed,
        noise,
    )?;
    Ok(sample)
}

/// Deterministic shot-by-shot simulation plus per-shot trajectory trace (C2c foundation).
pub fn sample_mid_circuit_measurements_with_trace(
    gates: &[Gate],
    qubit_count: usize,
    classical_bit_count: usize,
    shots: u64,
    seed: u64,
    noise: Option<&NoiseModel>,
) -> Result<(SampleResult, TrajectoryTrace), EngineError> {
    if qubit_count > 20 {
        return Err(EngineError::ExecutionFailed(
            "mid-circuit trajectory sampling requires qubit_count <= 20".into(),
        ));
    }

    validate_phase_c_sample_circuit(gates, classical_bit_count)?;

    let mut counts = std::collections::BTreeMap::new();
    let mut traces = Vec::with_capacity(shots as usize);
    for shot in 0..shots {
        let shot_seed = seed.wrapping_add(shot);
        let trace = simulate_one_shot_with_trace(
            gates,
            qubit_count,
            classical_bit_count,
            shot,
            shot_seed,
            noise,
        )?;
        *counts.entry(trace.final_outcome.clone()).or_insert(0) += 1;
        traces.push(trace);
    }

    Ok((
        SampleResult { counts, shots },
        TrajectoryTrace { shots, traces },
    ))
}

fn simulate_one_shot_with_trace(
    gates: &[Gate],
    qubit_count: usize,
    classical_bit_count: usize,
    shot_index: u64,
    seed: u64,
    noise: Option<&NoiseModel>,
) -> Result<TrajectoryShotTrace, EngineError> {
    let mut state = DenseTnState::try_new(qubit_count)?;
    let mut classical = vec![0u8; classical_bit_count];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut measures = Vec::new();

    for (gate_index, gate) in gates.iter().enumerate() {
        match gate {
            Gate::MEASURE(spec) => {
                let pre_measure_statevector: Vec<(f64, f64)> =
                    state.state.iter().map(|amp| (amp.re, amp.im)).collect();
                // The trajectory witness stores this vector on the 1e-4 digest
                // grid and the verifier allows only 1e-5 between shot p0/p1 and
                // that witness. Sample and record the marginal of the same grid.
                let canonical = canonicalize_terminal_statevector(&pre_measure_statevector);
                let (p0, p1) = z_marginal_from_statevector(&canonical, spec.qubit, qubit_count)
                    .unwrap_or_else(|| z_marginal(&state, spec.qubit));
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
                measures.push(TrajectoryMeasureEvent {
                    gate_index,
                    qubit: spec.qubit,
                    cbit: spec.cbit,
                    p0,
                    p1,
                    outcome,
                    pre_measure_statevector,
                });
            }
            Gate::RESET(q) => reset_qubit(&mut state, *q, &mut rng),
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

    Ok(TrajectoryShotTrace {
        shot_index,
        shot_seed: seed,
        final_outcome: outcome_key_from_classical(&classical),
        classical_bits: classical,
        measures,
    })
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
        Gate::H(q)
        | Gate::X(q)
        | Gate::Y(q)
        | Gate::Z(q)
        | Gate::S(q)
        | Gate::T(q)
        | Gate::RX(q, _)
        | Gate::RY(q, _)
        | Gate::RZ(q, _) => Some(*q),
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

/// RESET discards the qubit's value (zk-STARK §6.5: the state-vector leaf
/// swaps it into an ancilla that is traced out). On one trajectory that is a
/// measurement whose outcome is drawn from the shot's RNG and not recorded,
/// then an X when it read 1. Projecting onto |0⟩ instead would post-select:
/// it changes the other qubits' distribution, and leaves no state at all
/// when the qubit is |1⟩.
fn reset_qubit(state: &mut DenseTnState, qubit: usize, rng: &mut StdRng) {
    let (p0, p1) = z_marginal(state, qubit);
    let outcome = if rng.gen::<f64>() < p0 / (p0 + p1).max(1e-30) {
        0
    } else {
        1
    };
    collapse_z(state, qubit, outcome);
    if outcome == 1 {
        state.apply_gate(&Gate::X(qubit));
    }
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
    use crate::engine::{Gate, IfParams};

    /// Exact distribution of the classical register, branching on every
    /// MEASURE and RESET outcome (circuit-payload §1, §2.1). Independent of
    /// the trajectory code: it keeps every branch with its weight.
    fn exact_distribution(
        gates: &[Gate],
        n: usize,
        ncb: usize,
    ) -> std::collections::BTreeMap<String, f64> {
        let mut branches = vec![(DenseTnState::try_new(n).expect("dense"), vec![0u8; ncb])];
        for gate in gates {
            let mut next = Vec::new();
            for (state, cbits) in branches {
                let split = |q: usize| {
                    (0..2u8)
                        .map(|bit| {
                            let mut s = DenseTnState {
                                state: state.state.clone(),
                                qubit_count: state.qubit_count,
                            };
                            for (i, a) in s.state.iter_mut().enumerate() {
                                if ((i >> q) & 1) as u8 != bit {
                                    *a = num_complex::Complex64::new(0.0, 0.0);
                                }
                            }
                            (bit, s)
                        })
                        .filter(|(_, s)| s.state.iter().map(|a| a.norm_sqr()).sum::<f64>() > 1e-15)
                        .collect::<Vec<_>>()
                };
                match gate {
                    Gate::MEASURE(m) => {
                        for (bit, s) in split(m.qubit) {
                            let mut c = cbits.clone();
                            c[m.cbit] = bit;
                            next.push((s, c));
                        }
                    }
                    Gate::RESET(q) => {
                        for (bit, mut s) in split(*q) {
                            if bit == 1 {
                                s.apply_gate(&Gate::X(*q));
                            }
                            next.push((s, cbits.clone()));
                        }
                    }
                    Gate::IF(p) => {
                        let mut s = state;
                        if cbits[p.cbit] == p.value {
                            s.apply_gate(&p.gate);
                        }
                        next.push((s, cbits));
                    }
                    g => {
                        let mut s = state;
                        s.apply_gate(g);
                        next.push((s, cbits));
                    }
                }
            }
            branches = next;
        }
        let mut dist = std::collections::BTreeMap::new();
        for (state, cbits) in branches {
            let weight: f64 = state.state.iter().map(|a| a.norm_sqr()).sum();
            *dist
                .entry(outcome_key_from_classical(&cbits))
                .or_insert(0.0) += weight;
        }
        dist
    }

    /// The sampled counts are within 5.5 sigma of the exact distribution,
    /// and no outcome of probability 0 is drawn.
    fn assert_matches_exact(gates: &[Gate], n: usize, ncb: usize, shots: u64, label: &str) {
        let want = exact_distribution(gates, n, ncb);
        let got = sample_mid_circuit_measurements(gates, n, ncb, shots, 11, None)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(got.counts.values().sum::<u64>(), shots, "{label}");
        let keys: std::collections::BTreeSet<_> = want.keys().chain(got.counts.keys()).collect();
        for key in keys {
            let p = want.get(key).copied().unwrap_or(0.0);
            let c = got.counts.get(key).copied().unwrap_or(0) as f64;
            if p < 1e-12 {
                assert_eq!(c, 0.0, "{label}: outcome {key} has probability 0");
                continue;
            }
            let sigma = (p * (1.0 - p) * shots as f64).sqrt().max(1e-6);
            let z = (c - p * shots as f64).abs() / sigma;
            assert!(
                z < 5.5,
                "{label}: outcome {key} p={p} count={c} ({z:.1} sigma)"
            );
        }
    }

    fn m(qubit: usize, cbit: usize) -> Gate {
        Gate::MEASURE(MeasureParams { qubit, cbit })
    }

    /// RESET of an entangled qubit discards it: the partner stays 50:50
    /// (projecting onto |0⟩ gave 0 every shot). RESET of a measured |1⟩
    /// returns it to |0⟩ (projecting left no state, then 1 or an error).
    #[test]
    fn reset_discards_the_qubit_instead_of_post_selecting() {
        let entangled = [Gate::H(0), Gate::CNOT(0, 1), Gate::RESET(0), m(1, 0)];
        assert_matches_exact(&entangled, 2, 1, 4000, "H CNOT RESET M");
        let partial = [Gate::RY(0, 1.2), Gate::CNOT(0, 1), Gate::RESET(0), m(1, 0)];
        assert_matches_exact(&partial, 2, 1, 4000, "RY CNOT RESET M");
        let measured_one = [Gate::X(0), m(0, 0), Gate::RESET(0), m(0, 1)];
        let got = sample_mid_circuit_measurements(&measured_one, 1, 2, 200, 3, None)
            .expect("X M RESET M");
        assert_eq!(got.counts.get("01"), Some(&200), "{:?}", got.counts);
        let unmeasured_one = [Gate::X(0), Gate::RESET(0), Gate::H(0), m(0, 0)];
        assert_matches_exact(&unmeasured_one, 1, 1, 4000, "X RESET H M");
    }

    /// Seeded random mid-circuit circuits (MEASURE, RESET, IF, every unitary
    /// gate; measured qubits frozen until RESET) match the exact distribution.
    #[test]
    fn random_mid_circuits_match_the_exact_distribution() {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(0x0d1d_2026_1128);
        for case in 0..40 {
            let n = rng.gen_range(1..=4usize);
            let ncb = rng.gen_range(1..=n.min(3) + 1);
            let mut frozen = vec![false; n];
            let mut measured_any = false;
            let mut gates = Vec::new();
            for _ in 0..rng.gen_range(2..=14) {
                let live: Vec<usize> = (0..n).filter(|&q| !frozen[q]).collect();
                let k: f64 = rng.gen();
                let unitary = |rng: &mut StdRng| -> Gate {
                    let pick = |rng: &mut StdRng, not: &[usize]| loop {
                        let w = live[rng.gen_range(0..live.len())];
                        if !not.contains(&w) {
                            break w;
                        }
                    };
                    let q = pick(rng, &[]);
                    let j: f64 = rng.gen();
                    if live.len() >= 3 && j < 0.08 {
                        let b = pick(rng, &[q]);
                        let t = pick(rng, &[q, b]);
                        Gate::CCNOT(q, b, t)
                    } else if live.len() >= 2 && j < 0.35 {
                        let t = pick(rng, &[q]);
                        if rng.gen() {
                            Gate::CNOT(q, t)
                        } else {
                            Gate::CZ(q, t)
                        }
                    } else if j < 0.65 {
                        let theta = rng.gen_range(-7.0..7.0);
                        match rng.gen_range(0..3) {
                            0 => Gate::RX(q, theta),
                            1 => Gate::RY(q, theta),
                            _ => Gate::RZ(q, theta),
                        }
                    } else {
                        match rng.gen_range(0..6) {
                            0 => Gate::H(q),
                            1 => Gate::X(q),
                            2 => Gate::Y(q),
                            3 => Gate::Z(q),
                            4 => Gate::S(q),
                            _ => Gate::T(q),
                        }
                    }
                };
                if k < 0.2 && !live.is_empty() {
                    let q = live[rng.gen_range(0..live.len())];
                    gates.push(m(q, rng.gen_range(0..ncb)));
                    frozen[q] = true;
                    measured_any = true;
                } else if k < 0.32 {
                    let q = rng.gen_range(0..n);
                    gates.push(Gate::RESET(q));
                    frozen[q] = false;
                } else if k < 0.44 && measured_any && !live.is_empty() {
                    let g = unitary(&mut rng);
                    gates.push(Gate::IF(IfParams {
                        cbit: rng.gen_range(0..ncb),
                        value: rng.gen_range(0..2),
                        gate: Box::new(g),
                    }));
                } else if !live.is_empty() {
                    let g = unitary(&mut rng);
                    gates.push(g);
                }
            }
            for (q, &is_frozen) in frozen.iter().enumerate() {
                if !is_frozen {
                    gates.push(m(q, rng.gen_range(0..ncb)));
                }
            }
            assert_matches_exact(&gates, n, ncb, 4000, &format!("case {case}: {gates:?}"));
        }
    }

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
        let err = validate_phase_c_sample_circuit(&gates, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("measured qubit"));
    }

    #[test]
    fn trajectory_trace_records_measurement_probabilities_and_outcomes() {
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
        let (sample, trace) =
            sample_mid_circuit_measurements_with_trace(&gates, 2, 2, 4, 7, None).expect("trace");
        assert_eq!(sample.shots, 4);
        assert_eq!(trace.shots, 4);
        assert_eq!(trace.traces.len(), 4);
        assert_eq!(trace.traces[0].shot_index, 0);
        assert_eq!(trace.traces[0].shot_seed, 7);
        assert_eq!(trace.traces[0].measures.len(), 2);
        let first = &trace.traces[0].measures[0];
        assert_eq!(first.gate_index, 1);
        assert_eq!(first.qubit, 0);
        assert_eq!(first.cbit, 0);
        let canonical = canonicalize_terminal_statevector(&first.pre_measure_statevector);
        let (expected_p0, expected_p1) =
            z_marginal_from_statevector(&canonical, first.qubit, 2).expect("marginal");
        assert_eq!(first.p0, expected_p0);
        assert_eq!(first.p1, expected_p1);
        assert!((first.p0 - 0.5).abs() < 1e-4, "p0={}", first.p0);
        assert!((first.p1 - 0.5).abs() < 1e-4, "p1={}", first.p1);
        assert!(trace
            .traces
            .iter()
            .all(|shot| shot.final_outcome == outcome_key_from_classical(&shot.classical_bits)));
    }
}
