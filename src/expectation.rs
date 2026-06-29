//! Phase B B1: exact Pauli expectation values from terminal statevector projection.

use std::collections::{BTreeMap, BTreeSet};

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::engine::{ComplexResult, EngineError, format_go_complex_result_json, MeasureParams};
use crate::tn::MpsState;

/// Complex coefficient for a Pauli term (matches orchestrator JSON).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComplexCoeff {
    pub real: f64,
    pub imag: f64,
}

/// Single Pauli string term in a linear combination.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PauliTerm {
    pub label: String,
    pub coeff: ComplexCoeff,
}

/// Named observable: O = Σ coeff_k · P_k.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObservableSpec {
    pub id: String,
    pub terms: Vec<PauliTerm>,
}

/// Expectation values keyed by observable `id`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExpectationResult {
    pub values: BTreeMap<String, ComplexResult>,
}

/// Validate observables and circuit shape for `expectation` mode.
pub fn validate_expectation_task(
    qubit_count: usize,
    measures: &[MeasureParams],
    observables: &[ObservableSpec],
) -> Result<(), String> {
    if !measures.is_empty() {
        return Err("expectation mode forbids MEASURE gates".into());
    }
    if observables.is_empty() {
        return Err("observables must contain at least one entry for expectation".into());
    }
    if qubit_count == 0 {
        return Err("qubit_count must be > 0 for expectation".into());
    }

    let mut seen_ids = BTreeSet::new();
    for obs in observables {
        if obs.id.is_empty() {
            return Err("observable id must not be empty".into());
        }
        if !seen_ids.insert(obs.id.clone()) {
            return Err(format!("duplicate observable id {:?}", obs.id));
        }
        if obs.terms.is_empty() {
            return Err(format!("observable {:?} must have at least one term", obs.id));
        }
        for term in &obs.terms {
            validate_pauli_label(&term.label, qubit_count)?;
        }
    }
    Ok(())
}

fn validate_pauli_label(label: &str, qubit_count: usize) -> Result<(), String> {
    if label.len() != qubit_count {
        return Err(format!(
            "pauli label {:?} length {} does not match qubit_count {}",
            label,
            label.len(),
            qubit_count
        ));
    }
    for ch in label.chars() {
        if !matches!(ch, 'I' | 'X' | 'Y' | 'Z') {
            return Err(format!("invalid pauli character {:?} in label {:?}", ch, label));
        }
    }
    Ok(())
}

/// Apply a Pauli string to a computational basis index (Qiskit: rightmost char = qubit 0).
fn apply_pauli_string(index: usize, label: &str) -> Result<(usize, Complex64), EngineError> {
    let n = label.len();
    let mut j = index;
    let mut phase = Complex64::new(1.0, 0.0);
    let bytes = label.as_bytes();

    for q in 0..n {
        let ch = bytes[n - 1 - q];
        let bit = (index >> q) & 1;
        match ch {
            b'I' => {}
            b'Z' => {
                if bit == 1 {
                    phase *= -1.0;
                }
            }
            b'X' => {
                j ^= 1 << q;
            }
            b'Y' => {
                j ^= 1 << q;
                // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                phase *= Complex64::new(0.0, if bit == 0 { 1.0 } else { -1.0 });
            }
            _ => {
                return Err(EngineError::ExecutionFailed(format!(
                    "invalid pauli character in label {:?}",
                    label
                )));
            }
        }
    }

    Ok((j, phase))
}

/// ⟨ψ|P|ψ⟩ for a single Pauli string P.
pub fn pauli_string_expectation(
    statevector: &[Complex64],
    label: &str,
) -> Result<f64, EngineError> {
    let n = label.len();
    if n == 0 {
        return Err(EngineError::ExecutionFailed("empty pauli label".into()));
    }
    let dim = 1usize << n;
    if statevector.len() != dim {
        return Err(EngineError::ExecutionFailed(format!(
            "statevector length {} does not match 2^{n}",
            statevector.len()
        )));
    }

    let mut sum = Complex64::new(0.0, 0.0);
    for i in 0..dim {
        let (j, phase) = apply_pauli_string(i, label)?;
        sum += statevector[i].conj() * phase * statevector[j];
    }
    Ok(sum.re)
}

/// Evaluate one observable (weighted sum of Pauli terms).
pub fn observable_expectation(
    statevector: &[Complex64],
    observable: &ObservableSpec,
) -> Result<ComplexResult, EngineError> {
    let mut real = 0.0;
    let mut imag = 0.0;

    for term in &observable.terms {
        let pauli_exp = pauli_string_expectation(statevector, &term.label)?;
        let coeff = Complex64::new(term.coeff.real, term.coeff.imag);
        let product = coeff * pauli_exp;
        real += product.re;
        imag += product.im;
    }

    Ok(ComplexResult { real, imag })
}

/// Compute all named observables from a contracted MPS register.
pub fn compute_expectations(
    state: &MpsState,
    observables: &[ObservableSpec],
) -> Result<ExpectationResult, EngineError> {
    let statevector = state.contract_to_statevector()?;
    let mut values = BTreeMap::new();
    for obs in observables {
        let value = observable_expectation(&statevector, obs)?;
        values.insert(obs.id.clone(), value);
    }
    Ok(ExpectationResult { values })
}

fn format_go_float(val: f64) -> String {
    if val == (val as i64) as f64 {
        format!("{:.1}", val)
    } else {
        format!("{val}")
    }
}

fn format_go_complex_coeff_json(coeff: &ComplexCoeff) -> String {
    format!(
        r#"{{"imag":{},"real":{}}}"#,
        format_go_float(coeff.imag),
        format_go_float(coeff.real),
    )
}

/// Canonical JSON for observable specs — must match orchestrator hashing.
pub fn format_go_observable_spec_json(observables: &[ObservableSpec]) -> String {
    let mut sorted: Vec<&ObservableSpec> = observables.iter().collect();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));

    let mut obs_parts = Vec::with_capacity(sorted.len());
    for obs in sorted {
        let mut terms = obs.terms.clone();
        terms.sort_by(|a, b| a.label.cmp(&b.label));
        let term_parts: Vec<String> = terms
            .iter()
            .map(|t| {
                format!(
                    r#"{{"coeff":{},"label":"{}"}}"#,
                    format_go_complex_coeff_json(&t.coeff),
                    t.label
                )
            })
            .collect();
        obs_parts.push(format!(
            r#"{{"id":"{}","terms":[{}]}}"#,
            obs.id,
            term_parts.join(",")
        ));
    }
    format!(r#"{{"observables":[{}]}}"#, obs_parts.join(","))
}

/// SHA3-256 hex digest of the canonical observable-spec JSON.
pub fn calculate_observable_spec_hash(observables: &[ObservableSpec]) -> String {
    use sha3::{Digest, Sha3_256};
    hex::encode(Sha3_256::digest(
        format_go_observable_spec_json(observables).as_bytes(),
    ))
}

/// Canonical JSON for expectation results — must match orchestrator hashing.
pub fn format_go_expectation_result_json(result: &ExpectationResult) -> String {
    let mut pairs = String::new();
    for (id, value) in &result.values {
        if !pairs.is_empty() {
            pairs.push(',');
        }
        pairs.push_str(&format!(
            r#""{id}":{}"#,
            format_go_complex_result_json(value)
        ));
    }
    format!(r#"{{"values":{{{pairs}}}}}"#)
}

/// SHA3-256 hex digest of the canonical expectation-result JSON.
pub fn calculate_expectation_result_hash(result: &ExpectationResult) -> String {
    use sha3::{Digest, Sha3_256};
    hex::encode(Sha3_256::digest(
        format_go_expectation_result_json(result).as_bytes(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Circuit, ContractionWorkspace, Gate};

    fn zero_state_expectation(observables: &[ObservableSpec]) -> ExpectationResult {
        let mut workspace = ContractionWorkspace::try_allocate(1, 1).expect("allocate");
        let circuit = Circuit::new(1);
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");
        compute_expectations(workspace.register_mut(), observables).expect("expect")
    }

    #[test]
    fn z_expectation_on_zero_is_one() {
        let obs = vec![ObservableSpec {
            id: "Z0".into(),
            terms: vec![PauliTerm {
                label: "Z".into(),
                coeff: ComplexCoeff {
                    real: 1.0,
                    imag: 0.0,
                },
            }],
        }];
        let result = zero_state_expectation(&obs);
        let z0 = result.values.get("Z0").expect("Z0");
        assert!((z0.real - 1.0).abs() < 1e-12);
        assert!(z0.imag.abs() < 1e-12);
    }

    #[test]
    fn z_expectation_on_plus_is_zero() {
        let mut workspace = ContractionWorkspace::try_allocate(1, 1).expect("allocate");
        let mut circuit = Circuit::new(1);
        circuit.add(Gate::H(0)).expect("h");
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");

        let obs = vec![ObservableSpec {
            id: "Z0".into(),
            terms: vec![PauliTerm {
                label: "Z".into(),
                coeff: ComplexCoeff {
                    real: 1.0,
                    imag: 0.0,
                },
            }],
        }];
        let result = compute_expectations(workspace.register_mut(), &obs).expect("exp");
        let z0 = result.values.get("Z0").expect("Z0");
        assert!(z0.real.abs() < 1e-12);
    }

    #[test]
    fn bell_state_zz_expectation_is_one() {
        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::H(0)).expect("h");
        circuit.add(Gate::CNOT(0, 1)).expect("cnot");
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");

        let obs = vec![ObservableSpec {
            id: "ZZ".into(),
            terms: vec![PauliTerm {
                label: "ZZ".into(),
                coeff: ComplexCoeff {
                    real: 1.0,
                    imag: 0.0,
                },
            }],
        }];
        let result = compute_expectations(workspace.register_mut(), &obs).expect("exp");
        let zz = result.values.get("ZZ").expect("ZZ");
        assert!((zz.real - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pauli_linear_combination() {
        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::H(0)).expect("h");
        circuit.add(Gate::CNOT(0, 1)).expect("cnot");
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");

        // H = 0.5 ZZ + 0.5 II on Bell → 0.5*1 + 0.5*1 = 1
        let obs = vec![ObservableSpec {
            id: "H".into(),
            terms: vec![
                PauliTerm {
                    label: "ZZ".into(),
                    coeff: ComplexCoeff {
                        real: 0.5,
                        imag: 0.0,
                    },
                },
                PauliTerm {
                    label: "II".into(),
                    coeff: ComplexCoeff {
                        real: 0.5,
                        imag: 0.0,
                    },
                },
            ],
        }];
        let result = compute_expectations(workspace.register_mut(), &obs).expect("exp");
        let h = result.values.get("H").expect("H");
        assert!((h.real - 1.0).abs() < 1e-12);
    }

    #[test]
    fn expectation_hash_is_order_independent() {
        let mut values_a = BTreeMap::new();
        values_a.insert(
            "H".into(),
            ComplexResult {
                real: -0.81,
                imag: 0.0,
            },
        );
        values_a.insert(
            "Z0".into(),
            ComplexResult {
                real: 0.12,
                imag: 0.0,
            },
        );
        let mut values_b = BTreeMap::new();
        values_b.insert(
            "Z0".into(),
            ComplexResult {
                real: 0.12,
                imag: 0.0,
            },
        );
        values_b.insert(
            "H".into(),
            ComplexResult {
                real: -0.81,
                imag: 0.0,
            },
        );
        assert_eq!(
            calculate_expectation_result_hash(&ExpectationResult { values: values_a }),
            calculate_expectation_result_hash(&ExpectationResult { values: values_b }),
        );
    }

    #[test]
    fn observable_spec_hash_sorts_terms() {
        let obs_a = vec![ObservableSpec {
            id: "H".into(),
            terms: vec![
                PauliTerm {
                    label: "ZZ".into(),
                    coeff: ComplexCoeff {
                        real: 0.5,
                        imag: 0.0,
                    },
                },
                PauliTerm {
                    label: "IX".into(),
                    coeff: ComplexCoeff {
                        real: 0.3,
                        imag: 0.0,
                    },
                },
            ],
        }];
        let obs_b = vec![ObservableSpec {
            id: "H".into(),
            terms: vec![
                PauliTerm {
                    label: "IX".into(),
                    coeff: ComplexCoeff {
                        real: 0.3,
                        imag: 0.0,
                    },
                },
                PauliTerm {
                    label: "ZZ".into(),
                    coeff: ComplexCoeff {
                        real: 0.5,
                        imag: 0.0,
                    },
                },
            ],
        }];
        assert_eq!(
            calculate_observable_spec_hash(&obs_a),
            calculate_observable_spec_hash(&obs_b),
        );
    }

    #[test]
    fn measure_gates_rejected_in_validation() {
        let measures = vec![MeasureParams { qubit: 0, cbit: 0 }];
        let obs = vec![ObservableSpec {
            id: "Z".into(),
            terms: vec![PauliTerm {
                label: "Z".into(),
                coeff: ComplexCoeff {
                    real: 1.0,
                    imag: 0.0,
                },
            }],
        }];
        let err = validate_expectation_task(1, &measures, &obs).unwrap_err();
        assert!(err.contains("MEASURE"));
    }
}
