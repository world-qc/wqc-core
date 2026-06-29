//! Terminal Z-basis measurement and seed-bound deterministic sampling (Phase A).

use std::collections::BTreeMap;

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::engine::{EngineError, Gate, MeasureParams};
use crate::tn::MpsState;

/// Client output mode for `/compute`.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    #[default]
    StatevectorScalar,
    SampleCounts,
    Expectation,
}

/// Histogram returned for `sample_counts` mode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SampleResult {
    pub counts: BTreeMap<String, u64>,
    pub shots: u64,
}

/// Split a circuit into unitary gates and a terminal MEASURE suffix.
pub fn split_unitary_and_measures(
    gates: &[Gate],
) -> Result<(Vec<Gate>, Vec<MeasureParams>), EngineError> {
    let first_measure = gates.iter().position(|g| matches!(g, Gate::MEASURE(_)));
    let Some(idx) = first_measure else {
        return Ok((gates.to_vec(), Vec::new()));
    };

    let (prefix, suffix) = gates.split_at(idx);
    if prefix.iter().any(|g| matches!(g, Gate::MEASURE(_))) {
        return Err(EngineError::ExecutionFailed(
            "mid-circuit MEASURE is not supported in Phase A".into(),
        ));
    }

    let mut measures = Vec::with_capacity(suffix.len());
    for gate in suffix {
        match gate {
            Gate::MEASURE(spec) => measures.push(spec.clone()),
            _ => {
                return Err(EngineError::ExecutionFailed(
                    "gates after the first MEASURE must all be MEASURE".into(),
                ));
            }
        }
    }

    Ok((prefix.to_vec(), measures))
}

/// Canonical JSON for SHA3-256 hashing — must match orchestrator `SampleResult` marshaling.
pub fn format_go_sample_result_json(result: &SampleResult) -> String {
    let mut counts_pairs = String::new();
    for (key, value) in &result.counts {
        if !counts_pairs.is_empty() {
            counts_pairs.push(',');
        }
        counts_pairs.push_str(&format!(r#""{key}":{value}"#));
    }
    format!(
        r#"{{"counts":{{{counts_pairs}}},"shots":{}}}"#,
        result.shots
    )
}

/// SHA3-256 hex digest of the canonical sample-result JSON.
pub fn calculate_sample_result_hash(result: &SampleResult) -> String {
    use sha3::{Digest, Sha3_256};
    hex::encode(Sha3_256::digest(format_go_sample_result_json(result).as_bytes()))
}

/// Build outcome probabilities for terminal Z measurements on the full statevector.
pub fn compute_outcome_probabilities(
    statevector: &[Complex64],
    qubit_count: usize,
    measures: &[MeasureParams],
    classical_bit_count: usize,
) -> Result<BTreeMap<String, f64>, EngineError> {
    if classical_bit_count == 0 {
        return Err(EngineError::ExecutionFailed(
            "classical_bit_count must be > 0 for sample_counts".into(),
        ));
    }
    for spec in measures {
        if spec.qubit >= qubit_count {
            return Err(EngineError::QubitIndexOutOfBounds {
                index: spec.qubit,
                limit: qubit_count,
            });
        }
        if spec.cbit >= classical_bit_count {
            return Err(EngineError::ExecutionFailed(format!(
                "classical bit index {} out of bounds (limit: {})",
                spec.cbit, classical_bit_count
            )));
        }
    }

    let dim = 1usize << qubit_count;
    if statevector.len() != dim {
        return Err(EngineError::ExecutionFailed(format!(
            "statevector length {} does not match 2^{qubit_count}",
            statevector.len()
        )));
    }

    let mut probs: BTreeMap<String, f64> = BTreeMap::new();
    for (basis, amp) in statevector.iter().enumerate() {
        let p = amp.re * amp.re + amp.im * amp.im;
        if p == 0.0 {
            continue;
        }
        let key = outcome_key(basis, measures, classical_bit_count);
        *probs.entry(key).or_insert(0.0) += p;
    }
    Ok(probs)
}

fn outcome_key(basis_index: usize, measures: &[MeasureParams], classical_bit_count: usize) -> String {
    let mut bits = vec![b'0'; classical_bit_count];
    for spec in measures {
        let bit = (basis_index >> spec.qubit) & 1;
        // Qiskit convention: rightmost character is cbit 0 (and typically qubit 0).
        let pos = classical_bit_count - 1 - spec.cbit;
        bits[pos] = if bit == 1 { b'1' } else { b'0' };
    }
    // SAFETY: bits are only ASCII '0'/'1'.
    unsafe { String::from_utf8_unchecked(bits) }
}

/// Deterministic shot sampling from a probability table and PRNG seed.
pub fn sample_counts_from_probabilities(
    probabilities: &BTreeMap<String, f64>,
    shots: u64,
    seed: u64,
) -> SampleResult {
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    if shots == 0 || probabilities.is_empty() {
        return SampleResult { counts, shots };
    }

    let outcomes: Vec<(&String, f64)> = probabilities.iter().map(|(k, v)| (k, *v)).collect();
    let total: f64 = outcomes.iter().map(|(_, p)| p).sum();
    let mut cumulative = Vec::with_capacity(outcomes.len());
    let mut acc = 0.0;
    for (label, prob) in &outcomes {
        acc += prob / total;
        cumulative.push((*label, acc));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..shots {
        let r: f64 = rng.gen();
        let label = cumulative
            .iter()
            .find(|(_, c)| r <= *c)
            .map(|(label, _)| (*label).clone())
            .unwrap_or_else(|| cumulative.last().unwrap().0.clone());
        *counts.entry(label).or_insert(0) += 1;
    }

    SampleResult { counts, shots }
}

/// Project an MPS state to a dense statevector and sample terminal measurements.
pub fn sample_terminal_measurements(
    state: &MpsState,
    measures: &[MeasureParams],
    classical_bit_count: usize,
    shots: u64,
    seed: u64,
) -> Result<SampleResult, EngineError> {
    let statevector = state.contract_to_statevector()?;
    let probabilities =
        compute_outcome_probabilities(&statevector, state.qubit_count, measures, classical_bit_count)?;
    Ok(sample_counts_from_probabilities(&probabilities, shots, seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Circuit, ContractionWorkspace, Gate};

    #[test]
    fn outcome_key_uses_qiskit_bit_order() {
        let measures = vec![
            MeasureParams { qubit: 0, cbit: 0 },
            MeasureParams { qubit: 1, cbit: 1 },
        ];
        // |01⟩: q0=1, q1=0 → Qiskit "01" (rightmost char = cbit 0 = q0)
        let key = outcome_key(0b01, &measures, 2);
        assert_eq!(key, "01");
        // |10⟩: q0=0, q1=1 → Qiskit "10"
        let key = outcome_key(0b10, &measures, 2);
        assert_eq!(key, "10");
    }

    #[test]
    fn bell_state_terminal_measure_produces_00_and_11() {
        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::H(0)).expect("h");
        circuit.add(Gate::CNOT(0, 1)).expect("cnot");
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");

        let measures = vec![
            MeasureParams { qubit: 0, cbit: 0 },
            MeasureParams { qubit: 1, cbit: 1 },
        ];
        let sample = sample_terminal_measurements(
            workspace.register_mut(),
            &measures,
            2,
            1024,
            42,
        )
        .expect("sample");

        assert_eq!(sample.shots, 1024);
        assert_eq!(sample.counts.values().sum::<u64>(), 1024);
        assert!(sample.counts.contains_key("00"));
        assert!(sample.counts.contains_key("11"));
        assert!(sample.counts.get("01").is_none());
        assert!(sample.counts.get("10").is_none());
    }

    #[test]
    fn sample_seed_is_reproducible() {
        let mut workspace = ContractionWorkspace::try_allocate(1, 1).expect("allocate");
        let mut circuit = Circuit::new(1);
        circuit.add(Gate::H(0)).expect("h");
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");

        let measures = vec![MeasureParams { qubit: 0, cbit: 0 }];
        let a = sample_terminal_measurements(workspace.register_mut(), &measures, 1, 256, 99)
            .expect("a");
        let b = sample_terminal_measurements(workspace.register_mut(), &measures, 1, 256, 99)
            .expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn mid_circuit_measure_is_rejected() {
        let gates = vec![
            Gate::H(0),
            Gate::MEASURE(MeasureParams { qubit: 0, cbit: 0 }),
            Gate::H(0),
        ];
        let err = split_unitary_and_measures(&gates).unwrap_err().to_string();
        assert!(err.contains("after the first MEASURE"));
    }

    #[test]
    fn sample_result_hash_uses_sorted_count_keys() {
        let mut counts = BTreeMap::new();
        counts.insert("11".to_string(), 512);
        counts.insert("00".to_string(), 512);
        let hash = calculate_sample_result_hash(&SampleResult {
            counts,
            shots: 1024,
        });
        assert_eq!(
            hash,
            calculate_sample_result_hash(&SampleResult {
                counts: {
                    let mut c = BTreeMap::new();
                    c.insert("00".to_string(), 512);
                    c.insert("11".to_string(), 512);
                    c
                },
                shots: 1024,
            })
        );
    }
}
