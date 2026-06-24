//! Tensor-network contraction entry point for slice execution.

use crate::engine::{ComplexResult, EngineError, Gate, SliceAssignment};

use super::boundary::BoundaryConditions;
use super::dense::DenseTnState;
use super::trace;

/// Contract a pruned gate list on the compact register and return scalar + STARK trace.
pub fn contract_slice(
    qubit_count: usize,
    gates: &[Gate],
    assignments: &[SliceAssignment],
    original_qubit_count: usize,
    state: &mut DenseTnState,
) -> Result<(ComplexResult, Vec<f64>), EngineError> {
    let boundary = BoundaryConditions::from_assignments(assignments)?;
    boundary.verify_policy_c(original_qubit_count, qubit_count)?;

    let trace = trace::execute_with_trace(qubit_count, gates, state)?;
    let amp = state.amplitude_at_compact_zero();

    Ok((
        ComplexResult {
            real: amp.re,
            imag: amp.im,
        },
        trace,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Circuit;

    #[test]
    fn h_then_ccnot_matches_circuit_executor() {
        let gates = vec![Gate::H(0), Gate::CCNOT(0, 1, 2)];
        let mut tn_state = DenseTnState::try_new(3).expect("allocate");
        let (tn_result, tn_trace) =
            contract_slice(3, &gates, &[], 3, &mut tn_state).expect("tn contract");

        let mut workspace = crate::engine::ContractionWorkspace::try_allocate(3, 3).expect("ws");
        let mut circuit = Circuit::new(3);
        for gate in gates {
            circuit.add(gate).expect("add");
        }
        let sv_trace = circuit
            .execute_with_trace(workspace.register_mut())
            .expect("sv trace");
        let sv_amp = workspace.register_mut().state[0];

        assert!((tn_result.real - sv_amp.re).abs() < 1e-12);
        assert!((tn_result.imag - sv_amp.im).abs() < 1e-12);
        assert_eq!(tn_trace, sv_trace);
    }

    #[test]
    fn boundary_assignments_validate_policy_c() {
        let gates = vec![Gate::H(0)];
        let assignments = vec![SliceAssignment {
            edge_id: "e_1".into(),
            value: 1,
        }];
        let mut state = DenseTnState::try_new(2).expect("allocate");

        contract_slice(2, &gates, &assignments, 3, &mut state).expect("2 = 3-1");
        let mut bad = DenseTnState::try_new(3).expect("allocate");
        assert!(contract_slice(3, &gates, &assignments, 3, &mut bad).is_err());
    }
}
