//! Tensor-network contraction entry point for slice execution.

use crate::engine::{ComplexResult, EngineError, Gate, SliceAssignment};

use super::boundary::BoundaryConditions;
use super::mps::MpsState;
use super::trace;

/// Contract a pruned gate list via bond-truncated MPS; returns scalar + STARK trace.
pub fn contract_slice(
    qubit_count: usize,
    gates: &[Gate],
    assignments: &[SliceAssignment],
    original_qubit_count: usize,
    state: &mut MpsState,
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
    use crate::tn::gates::exact_bond_dim;

    #[test]
    fn mps_h_ccnot_exact_bond_produces_expected_amplitude() {
        let gates = vec![Gate::H(0), Gate::CCNOT(0, 1, 2)];
        let chi = exact_bond_dim(3);
        let mut mps = MpsState::try_new_with_bond(3, chi).expect("mps");
        let (result, _) = contract_slice(3, &gates, &[], 3, &mut mps).expect("contract");
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        assert!((result.real - inv_sqrt2).abs() < 1e-5);
        assert!(result.imag.abs() < 1e-5);
    }

    #[test]
    fn boundary_assignments_validate_policy_c() {
        let gates = vec![Gate::H(0)];
        let assignments = vec![SliceAssignment {
            edge_id: "e_1".into(),
            value: 1,
        }];
        let chi = exact_bond_dim(2);
        let mut state = MpsState::try_new_with_bond(2, chi).expect("allocate");

        contract_slice(2, &gates, &assignments, 3, &mut state).expect("2 = 3-1");
        let mut bad = MpsState::try_new_with_bond(3, chi).expect("allocate");
        assert!(contract_slice(3, &gates, &assignments, 3, &mut bad).is_err());
    }

    #[test]
    fn truncated_bond_still_runs() {
        let gates = vec![Gate::H(0), Gate::CNOT(0, 1)];
        let mut mps = MpsState::try_new_with_bond(2, 4).expect("mps");
        assert!(contract_slice(2, &gates, &[], 2, &mut mps).is_ok());
    }

    #[test]
    fn reverse_site_order_matches_identity_amplitude() {
        use crate::tn::site_order::{logical_to_site_map, remap_gates, validate_site_order};

        let gates = vec![Gate::H(0), Gate::CNOT(0, 1)];
        let chi = exact_bond_dim(2);

        let mut mps_id = MpsState::try_new_with_bond(2, chi).expect("mps");
        let (r_id, _) = contract_slice(2, &gates, &[], 2, &mut mps_id).expect("id");

        let order = vec![1, 0];
        validate_site_order(&order, 2).unwrap();
        let remapped = remap_gates(&gates, &logical_to_site_map(&order)).unwrap();
        let mut mps_rev = MpsState::try_new_with_bond(2, chi).expect("mps");
        let (r_rev, _) = contract_slice(2, &remapped, &[], 2, &mut mps_rev).expect("rev");

        // |0…0⟩ is invariant under qubit permutation, so ⟨00|ψ⟩ matches.
        assert!((r_id.real - r_rev.real).abs() < 1e-8);
        assert!((r_id.imag - r_rev.imag).abs() < 1e-8);
    }
}
