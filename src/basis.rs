//! Phase B B2: X/Y/Z measurement basis helpers (circuit composition, no new gates).

use std::f64::consts::FRAC_PI_2;

use crate::engine::{Gate, MeasureParams};

/// Measurement basis for terminal Z-readout convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementBasis {
    Z,
    X,
    Y,
}

/// Unitary gates to insert immediately before `MEASURE` on `qubit` for non-Z bases.
///
/// WQC always executes `MEASURE` as **Z-basis** projection. Other bases use a fixed
/// pre-rotation on the same qubit:
///
/// | Basis | Pre-rotation (before `MEASURE`) | Qiskit equivalent |
/// | --- | --- | --- |
/// | Z | (none) | direct `measure` |
/// | X | `H` | `measure_x` |
/// | Y | `RX(-π/2)` | `measure_y` (equivalent to `Sdg`→`H` up to global phase) |
pub fn unitary_gates_before_z_measure(basis: MeasurementBasis, qubit: usize) -> Vec<Gate> {
    match basis {
        MeasurementBasis::Z => Vec::new(),
        MeasurementBasis::X => vec![Gate::H(qubit)],
        MeasurementBasis::Y => vec![Gate::RX(qubit, -FRAC_PI_2)],
    }
}

/// Pauli character for single-qubit `expectation` observables (`label` length 1).
pub fn pauli_char_for_basis(basis: MeasurementBasis) -> char {
    match basis {
        MeasurementBasis::Z => 'Z',
        MeasurementBasis::X => 'X',
        MeasurementBasis::Y => 'Y',
    }
}

/// Build terminal `MEASURE` suffix for one qubit in the given basis.
pub fn terminal_measure_in_basis(
    basis: MeasurementBasis,
    qubit: usize,
    cbit: usize,
) -> Vec<Gate> {
    let mut gates = unitary_gates_before_z_measure(basis, qubit);
    gates.push(Gate::MEASURE(MeasureParams { qubit, cbit }));
    gates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Circuit, ContractionWorkspace, Gate};
    use crate::expectation::{ComplexCoeff, ObservableSpec, PauliTerm, compute_expectations};
    use crate::sample::sample_terminal_measurements;

    #[test]
    fn x_basis_sample_on_plus_is_deterministic_zero() {
        let mut workspace = ContractionWorkspace::try_allocate(1, 1).expect("allocate");
        let mut circuit = Circuit::new(1);
        circuit.add(Gate::H(0)).expect("h");
        circuit.add(Gate::H(0)).expect("x basis");
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");

        let sample = sample_terminal_measurements(
            workspace.register_mut(),
            &[MeasureParams { qubit: 0, cbit: 0 }],
            1,
            64,
            7,
        )
        .expect("sample");
        assert_eq!(sample.counts.get("0"), Some(&64));
        assert!(sample.counts.get("1").is_none());
    }

    #[test]
    fn y_basis_sample_on_plus_i_is_deterministic_zero() {
        let mut workspace = ContractionWorkspace::try_allocate(1, 1).expect("allocate");
        let mut circuit = Circuit::new(1);
        circuit.add(Gate::RX(0, FRAC_PI_2)).expect("prep");
        circuit.add(Gate::RX(0, -FRAC_PI_2)).expect("y basis");
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");

        let sample = sample_terminal_measurements(
            workspace.register_mut(),
            &[MeasureParams { qubit: 0, cbit: 0 }],
            1,
            64,
            11,
        )
        .expect("sample");
        assert_eq!(sample.counts.get("0"), Some(&64));
    }

    #[test]
    fn x_and_y_expectation_via_pauli_labels() {
        let mut workspace = ContractionWorkspace::try_allocate(1, 1).expect("allocate");
        let mut circuit = Circuit::new(1);
        circuit.add(Gate::H(0)).expect("h");
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");

        let obs = vec![
            ObservableSpec {
                id: "X".into(),
                terms: vec![PauliTerm {
                    label: "X".into(),
                    coeff: ComplexCoeff {
                        real: 1.0,
                        imag: 0.0,
                    },
                }],
            },
            ObservableSpec {
                id: "Z".into(),
                terms: vec![PauliTerm {
                    label: "Z".into(),
                    coeff: ComplexCoeff {
                        real: 1.0,
                        imag: 0.0,
                    },
                }],
            },
        ];
        let result = compute_expectations(workspace.register_mut(), &obs).expect("exp");
        let x = result.values.get("X").expect("X");
        assert!((x.real - 1.0).abs() < 1e-12);
        let z = result.values.get("Z").expect("Z");
        assert!(z.real.abs() < 1e-12);
    }

    #[test]
    fn terminal_measure_in_basis_appends_measure_gate() {
        let gates = terminal_measure_in_basis(MeasurementBasis::X, 0, 0);
        assert_eq!(gates.len(), 2);
        assert!(matches!(gates[0], Gate::H(0)));
        assert!(matches!(gates[1], Gate::MEASURE(_)));
    }
}
