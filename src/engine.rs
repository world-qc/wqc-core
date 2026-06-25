use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumIter};
use std::fmt;

// --- Shared task / result types (serialized by api and wqc-node) ---

/// Classical fixation on one tensor-network leg (edge) for this slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceAssignment {
    /// Leg identifier from the orchestrator (e.g. `"e_0"`, `"e_1"`).
    pub edge_id: String,
    /// Classical bit value: `0` or `1`.
    pub value: u8,
}

/// Final scalar after full contraction of the slice tensor network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexResult {
    pub real: f64,
    pub imag: f64,
}

/// Canonical JSON for SHA3-256 hashing — must match orchestrator `ComplexResult.MarshalJSON`.
pub fn format_go_complex_result_json(result: &ComplexResult) -> String {
    fn format_go_float(val: f64) -> String {
        if val == (val as i64) as f64 {
            format!("{:.1}", val)
        } else {
            format!("{val}")
        }
    }

    format!(
        r#"{{"real":{},"imag":{}}}"#,
        format_go_float(result.real),
        format_go_float(result.imag),
    )
}

/// SHA3-256 of the canonical JSON-encoded complex result.
pub fn calculate_complex_result_hash(result: &ComplexResult) -> String {
    use sha3::{Digest, Sha3_256};
    let bytes = format_go_complex_result_json(result).into_bytes();
    hex::encode(Sha3_256::digest(&bytes))
}

// --- Error definitions for robustness ---

#[derive(Debug)]
pub enum EngineError {
    InvalidQubitCount(usize),
    QubitIndexOutOfBounds { index: usize, limit: usize },
    InsufficientMemory { required: u64, available: u64 },
    MismatchedRegister,
    InvalidAssignmentValue { edge_id: String, value: u8 },
    InvalidEdgeId(String),
    BasisIndexOutOfBounds { index: usize, dim: usize },
    ExecutionFailed(String),
}

impl std::error::Error for EngineError {}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InvalidQubitCount(c) => {
                write!(f, "Invalid qubit count: {}. Max limit is 40.", c)
            }
            EngineError::QubitIndexOutOfBounds { index, limit } => {
                write!(f, "Qubit index {} out of bounds (limit: {})", index, limit)
            }
            EngineError::InsufficientMemory { required, available } => {
                write!(
                    f,
                    "Insufficient memory: need {} bytes, only {} bytes available (80% safety threshold)",
                    required, available
                )
            }
            EngineError::MismatchedRegister => {
                write!(f, "Workspace/circuit qubit count mismatch")
            }
            EngineError::InvalidAssignmentValue { edge_id, value } => {
                write!(f, "Assignment for edge '{}' has invalid classical value {}", edge_id, value)
            }
            EngineError::InvalidEdgeId(edge_id) => {
                write!(f, "Cannot parse tensor edge id '{}'", edge_id)
            }
            EngineError::BasisIndexOutOfBounds { index, dim } => {
                write!(f, "Contracted basis index {} out of workspace bounds (dim {})", index, dim)
            }
            EngineError::ExecutionFailed(msg) => write!(f, "{}", msg),
        }
    }
}

// --- Step 1: Pre-allocation workspace ---

/// Working memory for MPS tensor contraction (`O(N · χ²)` with bond dimension χ).
pub struct ContractionWorkspace {
    state: crate::tn::MpsState,
    /// Global circuit width before slicing; used for Policy C boundary validation.
    pub original_qubit_count: usize,
    /// Rough byte estimate: `N · χ² · 32`.
    reserved_bytes: u64,
}

impl ContractionWorkspace {
    pub fn try_allocate(qubit_count: usize, original_qubit_count: usize) -> Result<Self, EngineError> {
        Self::try_allocate_with_bond(qubit_count, original_qubit_count, None)
    }

    pub fn try_allocate_with_bond(
        qubit_count: usize,
        original_qubit_count: usize,
        mps_max_bond_dim: Option<usize>,
    ) -> Result<Self, EngineError> {
        let chi = crate::tn::resolve_bond_dim(mps_max_bond_dim);
        let reserved_bytes = (qubit_count as u64)
            .saturating_mul(chi as u64)
            .saturating_mul(chi as u64)
            .saturating_mul(32);
        let state = crate::tn::MpsState::try_new_with_bond(qubit_count, chi)?;
        Ok(Self {
            state,
            original_qubit_count,
            reserved_bytes,
        })
    }

    pub fn qubit_count(&self) -> usize {
        self.state.qubit_count
    }

    pub fn register_mut(&mut self) -> &mut QuantumRegister {
        &mut self.state
    }

    pub fn tn_backend_label(&self) -> &'static str {
        self.state.backend_label()
    }

    pub fn peak_vram_bytes(&self) -> u64 {
        self.state.peak_vram_bytes()
    }
}

// --- Step 2: Tensor network contraction executive ---

/// Partial tensor graph: pruned `circuit` plus boundary `assignments` for this slice.
pub struct TensorNetwork {
    qubit_count: usize,
    gates: Vec<Gate>,
    assignments: Vec<SliceAssignment>,
}

impl TensorNetwork {
    /// Builds the network and validates gates / assignments before any allocation-heavy work.
    pub fn from_parts(
        qubit_count: usize,
        gates: Vec<Gate>,
        assignments: Vec<SliceAssignment>,
    ) -> Result<Self, EngineError> {
        crate::tn::boundary::BoundaryConditions::from_assignments(&assignments)?;

        // Reject out-of-bounds qubit indices early (replaces panicking asserts).
        let mut circuit = Circuit::new(qubit_count);
        for gate in gates {
            circuit.add(gate)?;
        }

        Ok(Self {
            qubit_count,
            gates: circuit.gates,
            assignments,
        })
    }

    /// Gate-by-gate TN contraction on the compact register; returns scalar amplitude + STARK trace.
    pub fn contract(
        &self,
        workspace: &mut ContractionWorkspace,
    ) -> Result<(ComplexResult, Vec<f64>), EngineError> {
        if workspace.qubit_count() != self.qubit_count {
            return Err(EngineError::MismatchedRegister);
        }

        crate::tn::contract_slice(
            self.qubit_count,
            &self.gates,
            &self.assignments,
            workspace.original_qubit_count,
            &mut workspace.state,
        )
    }
}

// --- Gate definitions (API + STARK trace metadata) ---

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug, EnumIter, Display)]
#[serde(tag = "type", content = "params")]
#[strum(serialize_all = "UPPERCASE")]
pub enum Gate {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    T(usize),
    S(usize),
    CNOT(usize, usize),
    CZ(usize, usize),
    RX(usize, f64),
    RY(usize, f64),
    RZ(usize, f64),
    CCNOT(usize, usize, usize),
}

impl Gate {
    /// Maps gates to float identifiers for the STARK execution-trace constraint selector.
    pub fn to_stark_id(&self) -> Option<f64> {
        match self {
            Gate::X(_) => Some(1.0),
            Gate::Y(_) => Some(2.0),
            Gate::Z(_) => Some(3.0),
            Gate::H(_) => Some(4.0),
            Gate::S(_) => Some(5.0),
            Gate::T(_) => Some(6.0),
            Gate::CNOT(..) => Some(7.0),
            Gate::CZ(..) => Some(8.0),
            Gate::CCNOT(..) => Some(9.0),
            Gate::RX(..) => Some(10.0),
            Gate::RY(..) => Some(11.0),
            Gate::RZ(..) => Some(12.0),
        }
    }

    /// Extracts supplementary execution payload fields for the STARK 10-column chunk format.
    /// Returns `(ctrl_active, p_cos, p_sin)` — aligned with the `wqc-stark-core` ingest spec.
    pub fn to_stark_payload(&self, is_control_active: bool) -> (f64, f64, f64) {
        match self {
            Gate::CNOT(..) | Gate::CZ(..) | Gate::CCNOT(..) => {
                (if is_control_active { 1.0 } else { 0.0 }, 1.0, 0.0)
            }
            Gate::RX(_, theta) | Gate::RY(_, theta) | Gate::RZ(_, theta) => {
                ((theta / 2.0).cos(), (theta / 2.0).sin(), 0.0)
            }
            _ => (0.0, 1.0, 0.0),
        }
    }
}

/// Bond-truncated MPS TN state (default contraction backend).
pub use crate::tn::MpsState as QuantumRegister;

pub struct Circuit {
    pub qubit_count: usize,
    pub gates: Vec<Gate>,
}

impl Circuit {
    pub fn new(qubit_count: usize) -> Self {
        Self {
            qubit_count,
            gates: Vec::new(),
        }
    }

    /// Appends a gate after bound checking (replaces panicking `assert!`).
    pub fn add(&mut self, gate: Gate) -> Result<(), EngineError> {
        match &gate {
            // 1-qubit gates
            Gate::H(t) | Gate::X(t) | Gate::Y(t) | Gate::Z(t) | Gate::T(t) | Gate::S(t) => {
                if *t >= self.qubit_count {
                    return Err(EngineError::QubitIndexOutOfBounds { index: *t, limit: self.qubit_count });
                }
            }
            // 1-qubit rotation gates
            Gate::RX(t, _) | Gate::RY(t, _) | Gate::RZ(t, _) => {
                if *t >= self.qubit_count {
                    return Err(EngineError::QubitIndexOutOfBounds { index: *t, limit: self.qubit_count });
                }
            }
            // 2-qubit gates
            Gate::CNOT(c, t) | Gate::CZ(c, t) => {
                if *c >= self.qubit_count {
                    return Err(EngineError::QubitIndexOutOfBounds { index: *c, limit: self.qubit_count });
                }
                if *t >= self.qubit_count {
                    return Err(EngineError::QubitIndexOutOfBounds { index: *t, limit: self.qubit_count });
                }
            }
            Gate::CCNOT(c1, c2, t) => {
                if *c1 >= self.qubit_count {
                    return Err(EngineError::QubitIndexOutOfBounds { index: *c1, limit: self.qubit_count });
                }
                if *c2 >= self.qubit_count {
                    return Err(EngineError::QubitIndexOutOfBounds { index: *c2, limit: self.qubit_count });
                }
                if *t >= self.qubit_count {
                    return Err(EngineError::QubitIndexOutOfBounds { index: *t, limit: self.qubit_count });
                }
            }
        }
        self.gates.push(gate);
        Ok(())
    }

    /// Executes the circuit while capturing the 11-column STARK trace via the TN executor.
    pub fn execute_with_trace(&self, register: &mut QuantumRegister) -> Result<Vec<f64>, String> {
        if self.qubit_count != register.qubit_count {
            return Err("Register/circuit qubit count mismatch".to_string());
        }

        crate::tn::trace::execute_with_trace(self.qubit_count, &self.gates, register)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod trace_tests {
    use super::{Circuit, ContractionWorkspace, Gate};
    use wqc_stark_engine::trace_spec::TRACE_WIDTH;

    fn trace_at(trace: &[f64], row: usize, col: usize) -> f64 {
        trace[row * TRACE_WIDTH + col]
    }

    fn assert_approx_eq(a: f64, b: f64) {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-12,
            "assert_approx_eq failed: left={a} right={b} diff={diff}"
        );
    }

    #[test]
    fn empty_circuit_emits_terminal_trace_row() {
        let mut workspace = ContractionWorkspace::try_allocate(3, 30).expect("allocate");
        let circuit = Circuit::new(3);
        let trace = circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");
        assert_eq!(trace.len(), 11, "empty circuit should emit one 11-column boundary row");
    }

    #[test]
    fn h_gate_trace_has_expected_fields() {
        let mut workspace = ContractionWorkspace::try_allocate(1, 1).expect("allocate");
        let mut circuit = Circuit::new(1);
        circuit.add(Gate::H(0)).expect("add gate");

        let trace = circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");

        // 1 gate => 3 rows (pre, post, terminal)
        assert_eq!(trace.len(), 3 * TRACE_WIDTH);

        // Pre-gate snapshot row
        assert_approx_eq(trace_at(&trace, 0, 0), 4.0); // H gate id
        assert_approx_eq(trace_at(&trace, 0, 9), 0.0); // target qubit
        assert_approx_eq(trace_at(&trace, 0, 5), 1.0); // v0_re for |0>

        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        // Post-gate row
        assert_approx_eq(trace_at(&trace, 1, 0), 0.0);
        assert_approx_eq(trace_at(&trace, 1, 9), 0.0);
        assert_approx_eq(trace_at(&trace, 1, 5), inv_sqrt2);
        assert_approx_eq(trace_at(&trace, 1, 7), inv_sqrt2);

        // Terminal boundary row
        assert_approx_eq(trace_at(&trace, 2, 0), 0.0);
        assert_approx_eq(trace_at(&trace, 2, 9), 0.0);
        assert_approx_eq(trace_at(&trace, 2, 5), inv_sqrt2);
        assert_approx_eq(trace_at(&trace, 2, 7), inv_sqrt2);
    }

    #[test]
    fn cnot_with_control_zero_has_ctrl_active_zero() {
        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::CNOT(0, 1)).expect("add gate");

        let trace = circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");

        // 1 gate => 3 rows
        assert_eq!(trace.len(), 3 * TRACE_WIDTH);

        // Pre-gate snapshot row (before applying CNOT on |00>)
        assert_approx_eq(trace_at(&trace, 0, 0), 7.0); // CNOT gate id
        assert_approx_eq(trace_at(&trace, 0, 1), 0.0); // ctrl_active must be 0
        assert_approx_eq(trace_at(&trace, 0, 9), 1.0); // target qubit
        assert_approx_eq(trace_at(&trace, 0, 5), 1.0); // target v0_re for |00>
        assert_approx_eq(trace_at(&trace, 0, 7), 0.0); // target v1_re
    }

    #[test]
    fn cnot_with_control_one_discretizes_ctrl_active_to_one() {
        // Prepare control=1 by applying X on control qubit 0: |00> -> |01>, then CNOT(0,1): |01> -> |11>
        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::X(0)).expect("add gate");
        circuit.add(Gate::CNOT(0, 1)).expect("add gate");

        let trace = circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");

        // 2 gates => 5 rows
        assert_eq!(trace.len(), 5 * TRACE_WIDTH);

        // Pre-gate snapshot row for CNOT is row=2
        assert_approx_eq(trace_at(&trace, 2, 0), 7.0); // CNOT
        assert_approx_eq(trace_at(&trace, 2, 1), 1.0); // ctrl_active must be 1 (marginal prob = 1.0)
        assert_approx_eq(trace_at(&trace, 2, 9), 1.0); // target qubit
        assert_approx_eq(trace_at(&trace, 2, 5), 1.0); // target v0_re (target=0) for |01>
        assert_approx_eq(trace_at(&trace, 2, 7), 0.0); // target v1_re (target=1)

        // Terminal boundary row (after CNOT -> |11>, target qubit=1 => v0_re=0, v1_re=1)
        assert_approx_eq(trace_at(&trace, 4, 0), 0.0); // boundary
        assert_approx_eq(trace_at(&trace, 4, 9), 1.0);
        assert_approx_eq(trace_at(&trace, 4, 5), 0.0);
        assert_approx_eq(trace_at(&trace, 4, 7), 1.0);
    }

    #[test]
    fn cnot_control_half_probability_sets_ctrl_active_zero() {
        // H on control qubit => marginal control probability = 0.5, and we use strict `> 0.5` discretization.
        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::H(0)).expect("add gate");
        circuit.add(Gate::CNOT(0, 1)).expect("add gate");

        let trace = circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");

        // 2 gates => 5 rows; CNOT pre-gate snapshot is row=2
        assert_approx_eq(trace_at(&trace, 2, 0), 7.0); // CNOT
        assert_approx_eq(trace_at(&trace, 2, 1), 0.0); // ctrl_active should be 0 for prob == 0.5
    }

    #[test]
    fn cz_gate_trace_uses_gate_id_eight_and_discrete_control() {
        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::X(0)).expect("add gate");
        circuit.add(Gate::CZ(0, 1)).expect("add gate");

        let trace = circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");

        assert_eq!(trace.len(), 5 * TRACE_WIDTH);
        assert_approx_eq(trace_at(&trace, 2, 0), 8.0); // CZ gate id (pre-gate row)
        assert_approx_eq(trace_at(&trace, 2, 1), 1.0); // ctrl_active discretized to 1
    }

    #[test]
    fn ccnot_gate_trace_uses_gate_id_nine_and_dual_controls() {
        let mut workspace = ContractionWorkspace::try_allocate(3, 3).expect("allocate");
        let mut circuit = Circuit::new(3);
        circuit.add(Gate::X(0)).expect("add gate");
        circuit.add(Gate::X(1)).expect("add gate");
        circuit.add(Gate::CCNOT(0, 1, 2)).expect("add gate");

        let trace = circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");

        assert_eq!(trace.len(), 7 * TRACE_WIDTH);
        assert_approx_eq(trace_at(&trace, 4, 0), 9.0); // CCNOT pre-gate row
        assert_approx_eq(trace_at(&trace, 4, 1), 1.0); // ctrl_active
        assert_approx_eq(trace_at(&trace, 4, 2), 1.0); // ctrl_active_2
        assert_approx_eq(trace_at(&trace, 4, 9), 2.0); // target qubit
    }

    #[test]
    fn executor_traces_satisfy_stark_air_for_h_and_cnot() {
        use p3_field::PrimeField32;
        use wqc_stark_engine::evaluate_execution_trace;

        let cases: Vec<(&str, Vec<Gate>)> = vec![
            ("h", vec![Gate::H(0)]),
            ("cnot_inactive", vec![Gate::CNOT(0, 1)]),
            (
                "h_ccnot_devnet",
                vec![Gate::H(0), Gate::CCNOT(0, 1, 2)],
            ),
        ];

        for (name, gates) in cases {
            let qubits = gates
                .iter()
                .map(|gate| match gate {
                    Gate::CCNOT(_, _, t) | Gate::CNOT(_, t) | Gate::CZ(_, t) => *t + 1,
                    Gate::X(t)
                    | Gate::Y(t)
                    | Gate::Z(t)
                    | Gate::H(t)
                    | Gate::S(t)
                    | Gate::T(t)
                    | Gate::RX(t, _)
                    | Gate::RY(t, _)
                    | Gate::RZ(t, _) => *t + 1,
                })
                .max()
                .unwrap_or(1);

            let mut workspace = ContractionWorkspace::try_allocate(qubits, qubits).expect("allocate");
            let mut circuit = Circuit::new(qubits);
            for gate in gates {
                circuit.add(gate).expect("add gate");
            }

            let trace = circuit
                .execute_with_trace(workspace.register_mut())
                .expect("trace");

            let air_sum = evaluate_execution_trace(&trace)
                .unwrap_or_else(|| panic!("{name}: trace should expand to AIR"));
            assert_eq!(
                air_sum.as_canonical_u32(),
                0,
                "{name}: executor trace should satisfy AIR constraints"
            );
        }
    }
}

#[cfg(test)]
mod hash_tests {
    use super::{calculate_complex_result_hash, format_go_complex_result_json, ComplexResult};

    #[test]
    fn complex_result_json_uses_go_integer_style() {
        let json = format_go_complex_result_json(&ComplexResult {
            real: 0.0,
            imag: 1.0,
        });
        assert_eq!(json, r#"{"real":0.0,"imag":1.0}"#);
    }

    #[test]
    fn complex_result_hash_matches_orchestrator_for_zero_imag() {
        let hash = calculate_complex_result_hash(&ComplexResult {
            real: 0.7071067811865475,
            imag: 0.0,
        });
        // SHA3-256 of `{"real":0.7071067811865475,"imag":0.0}` — imag must be `0.0`, not `0`.
        assert_eq!(
            hash,
            "ee10cd493b1b6773f6e947b471b3fc8c94009eac7ac09d189e64ead838dfc0d5"
        );
    }
}
