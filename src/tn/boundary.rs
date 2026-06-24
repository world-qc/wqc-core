//! Slice boundary metadata: classical leg fixation for tensor-network contraction.
//!
//! Policy C (current devnet): the orchestrator prunes fixed legs upstream and dispatches a
//! compact register. Assignments are still validated here and bound into STARK public inputs.
//! The TN executor starts from |0…0⟩ on free wires; `original_qubit_count` records the parent width.

use crate::engine::{EngineError, SliceAssignment};

/// Parsed boundary: global qubit index → classical bit (`0` or `1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryConditions {
    pub fixed_qubits: Vec<(usize, u8)>,
}

impl BoundaryConditions {
    pub fn from_assignments(assignments: &[SliceAssignment]) -> Result<Self, EngineError> {
        let mut fixed_qubits = Vec::with_capacity(assignments.len());
        for assignment in assignments {
            if assignment.value > 1 {
                return Err(EngineError::InvalidAssignmentValue {
                    edge_id: assignment.edge_id.clone(),
                    value: assignment.value,
                });
            }
            let qubit = parse_edge_index(&assignment.edge_id)?;
            fixed_qubits.push((qubit, assignment.value));
        }
        fixed_qubits.sort_by_key(|(q, _)| *q);
        Ok(Self { fixed_qubits })
    }

    /// Policy C consistency: `effective = original - |assignments|`.
    pub fn verify_policy_c(
        &self,
        original_qubit_count: usize,
        effective_qubit_count: usize,
    ) -> Result<(), EngineError> {
        if self.fixed_qubits.len() > original_qubit_count {
            return Err(EngineError::InvalidQubitCount(original_qubit_count));
        }

        let expected_effective = original_qubit_count
            .checked_sub(self.fixed_qubits.len())
            .ok_or(EngineError::InvalidQubitCount(original_qubit_count))?;

        if effective_qubit_count != expected_effective {
            return Err(EngineError::ExecutionFailed(format!(
                "Policy C qubit mismatch: effective={effective_qubit_count}, expected={expected_effective} (original={original_qubit_count}, fixed_legs={})",
                self.fixed_qubits.len()
            )));
        }

        for (qubit, _) in &self.fixed_qubits {
            if *qubit >= original_qubit_count {
                return Err(EngineError::QubitIndexOutOfBounds {
                    index: *qubit,
                    limit: original_qubit_count,
                });
            }
        }

        Ok(())
    }

    /// Global computational-basis index for |free=0…0, fixed=assignments| on the parent register.
    pub fn global_basis_index_for_compact_zero(&self) -> usize {
        let mut index = 0usize;
        for (qubit, value) in &self.fixed_qubits {
            if *value == 1 {
                index |= 1 << qubit;
            }
        }
        index
    }
}

fn parse_edge_index(edge_id: &str) -> Result<usize, EngineError> {
    let suffix = edge_id
        .strip_prefix("e_")
        .ok_or_else(|| EngineError::InvalidEdgeId(edge_id.to_string()))?;
    suffix
        .parse::<usize>()
        .map_err(|_| EngineError::InvalidEdgeId(edge_id.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_c_verifies_effective_qubit_count() {
        let boundary = BoundaryConditions::from_assignments(&[
            SliceAssignment {
                edge_id: "e_2".into(),
                value: 1,
            },
            SliceAssignment {
                edge_id: "e_0".into(),
                value: 0,
            },
        ])
        .expect("parse");

        boundary.verify_policy_c(3, 1).expect("3-2=1");
        assert_eq!(boundary.global_basis_index_for_compact_zero(), 0b100);
    }

    #[test]
    fn rejects_inconsistent_effective_width() {
        let boundary = BoundaryConditions::from_assignments(&[SliceAssignment {
            edge_id: "e_1".into(),
            value: 0,
        }])
        .expect("parse");

        assert!(boundary.verify_policy_c(3, 3).is_err());
    }
}
