use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumIter};
use sysinfo::System;
use std::fmt;
use wqc_stark_engine::trace_spec::TRACE_WIDTH;

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

/// Working memory for tensor contraction: one contiguous complex buffer sized `2^qubit_count`.
///
/// `qubit_count` is the effective complexity for this slice; `original_qubit_count` records
/// how much the global circuit was shrunk before dispatch (engine tuning / logging).
pub struct ContractionWorkspace {
    register: QuantumRegister,
    /// Global circuit width before slicing; consumed by the future TN backend initializer.
    #[allow(dead_code)]
    pub original_qubit_count: usize,
    /// Bytes reserved up front (`2^qubit_count * 16`) to avoid fragmentation before contraction.
    #[allow(dead_code)]
    reserved_bytes: u64,
}

impl ContractionWorkspace {
    /// Reserves `2^qubit_count * 16` bytes and rejects tasks that exceed available RAM.
    pub fn try_allocate(qubit_count: usize, original_qubit_count: usize) -> Result<Self, EngineError> {
        let reserved_bytes = (1u64 << qubit_count).saturating_mul(16);
        let register = QuantumRegister::new(qubit_count)?;
        Ok(Self {
            register,
            original_qubit_count,
            reserved_bytes,
        })
    }

    pub fn qubit_count(&self) -> usize {
        self.register.qubit_count
    }

    pub fn register_mut(&mut self) -> &mut QuantumRegister {
        &mut self.register
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
        validate_assignments(&assignments)?;

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

    /// Executes gate contractions on the pre-allocated workspace and returns the scalar amplitude.
    pub fn contract(
        &self,
        workspace: &mut ContractionWorkspace,
    ) -> Result<(ComplexResult, Vec<f64>), EngineError> {
        if workspace.qubit_count() != self.qubit_count {
            return Err(EngineError::MismatchedRegister);
        }

        let mut circuit = Circuit::new(self.qubit_count);
        for gate in &self.gates {
            circuit.add(gate.clone())?;
        }

        // Run the interim state-vector kernel and capture the 10-column STARK execution trace.
        let trace = circuit
            .execute_with_trace(workspace.register_mut())
            .map_err(EngineError::ExecutionFailed)?;

        // Compact register: orchestrator prunes fixed legs and remaps free qubits to 0..N-C-1.
        // Slice boundary values live in assignments metadata; scalar readout is |0…0⟩ on free wires.
        let _ = &self.assignments;
        let state = &workspace.register_mut().state;
        if state.is_empty() {
            return Err(EngineError::BasisIndexOutOfBounds {
                index: 0,
                dim: 0,
            });
        }

        let amp = state[0];
        Ok((
            ComplexResult {
                real: amp.re,
                imag: amp.im,
            },
            trace,
        ))
    }
}

fn validate_assignments(assignments: &[SliceAssignment]) -> Result<(), EngineError> {
    for a in assignments {
        if a.value > 1 {
            return Err(EngineError::InvalidAssignmentValue {
                edge_id: a.edge_id.clone(),
                value: a.value,
            });
        }
        parse_edge_index(&a.edge_id)?;
    }
    Ok(())
}

fn parse_edge_index(edge_id: &str) -> Result<usize, EngineError> {
    let suffix = edge_id
        .strip_prefix("e_")
        .ok_or_else(|| EngineError::InvalidEdgeId(edge_id.to_string()))?;
    suffix
        .parse::<usize>()
        .map_err(|_| EngineError::InvalidEdgeId(edge_id.to_string()))
}

// --- Gate definitions and dense-kernel contraction backend (current executor) ---
//
// NOTE:
// - WQC targets heterogeneous decentralized nodes (servers, laptops, mobile, browser-class devices).
// - The long-term accelerator path is WebGPU via `wgpu` (portable), not vendor-locked CUDA/cuTensorNet.
// - This dense kernel remains the reference executor until a `wgpu` backend emits the same trace schema.

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

/// Dense complex buffer backing the interim contraction kernel (`|ψ⟩` amplitudes).
pub struct QuantumRegister {
    pub state: Array1<Complex64>,
    pub qubit_count: usize,
}

impl QuantumRegister {
    /// Creates a register after a dynamic memory check (prevents OOM before contraction).
    pub fn new(qubit_count: usize) -> Result<Self, EngineError> {
        // Required bytes: (2^N) * 16 for Complex64 amplitudes.
        let required_memory = (1u64 << qubit_count).saturating_mul(16);

        let mut sys = System::new_all();
        sys.refresh_memory();

        let available_memory = sys.available_memory();
        let total_memory = sys.total_memory();

        // Hardening: fit within 80% of currently available RAM and 90% of total physical RAM.
        let available_threshold = (available_memory as f64 * 0.8) as u64;
        let total_threshold = (total_memory as f64 * 0.9) as u64;

        if required_memory > available_threshold || required_memory > total_threshold {
            return Err(EngineError::InsufficientMemory {
                required: required_memory,
                available: available_memory,
            });
        }

        // Safety cap against accidental huge allocations (e.g. mis-reported qubit counts).
        if qubit_count == 0 || qubit_count > 40 {
            return Err(EngineError::InvalidQubitCount(qubit_count));
        }

        let dim = 1 << qubit_count;
        let mut state = Array1::from_elem(dim, Complex64::new(0.0, 0.0));
        state[0] = Complex64::new(1.0, 0.0); // |0…0⟩
        Ok(Self { state, qubit_count })
    }

    pub fn apply_gate(&mut self, gate: &Gate) {
        match gate {
            Gate::H(t) => {
                // Hadamard: creates superposition on the target qubit.
                let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
                self.apply_unary(*t, |v0, v1| {
                    ((v0 + v1) * inv_sqrt2, (v0 - v1) * inv_sqrt2)
                });
            }
            Gate::X(t) => {
                // Pauli-X: quantum NOT (swaps |0⟩ and |1⟩ on the target).
                self.apply_unary(*t, |v0, v1| (v1, v0))
            }
            Gate::Y(t) => {
                // Pauli-Y: π rotation around the Y axis.
                let i = Complex64::i();
                self.apply_unary(*t, |v0, v1| (v1 * (-i), v0 * i));
            }
            Gate::Z(t) => {
                // Pauli-Z: phase flip on |1⟩.
                self.apply_unary(*t, |v0, v1| (v0, -v1))
            }
            Gate::S(t) => {
                // S gate: Z^{1/2} (90° phase).
                let i = Complex64::i();
                self.apply_unary(*t, |v0, v1| (v0, v1 * i));
            }
            Gate::T(t) => {
                // T gate: Z^{1/4} (π/4 phase).
                let factor = Complex64::new(
                    std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                );
                self.apply_unary(*t, |v0, v1| (v0, v1 * factor));
            }
            Gate::CNOT(c, t) => {
                // Controlled-NOT: flip target when control is |1⟩.
                self.apply_controlled(*c, *t, |v0, v1| (v1, v0))
            }
            Gate::CZ(c, t) => {
                // Controlled-Z: phase on target when control is |1⟩.
                self.apply_controlled(*c, *t, |v0, v1| (v0, -v1))
            }
            Gate::RX(t, theta) => {
                // Rotation around X by theta.
                let (sin, cos) = (theta / 2.0).sin_cos();
                let cos_c = Complex64::new(cos, 0.0);
                let n_i_sin_c = Complex64::new(0.0, -sin);
                self.apply_unary(*t, |v0, v1| {
                    (v0 * cos_c + v1 * n_i_sin_c, v0 * n_i_sin_c + v1 * cos_c)
                });
            }
            Gate::RY(t, theta) => {
                // Rotation around Y by theta.
                let (sin, cos) = (theta / 2.0).sin_cos();
                self.apply_unary(*t, |v0, v1| {
                    (v0 * cos - v1 * sin, v0 * sin + v1 * cos)
                });
            }
            Gate::RZ(t, theta) => {
                // Rotation around Z by theta.
                let (sin, cos) = (theta / 2.0).sin_cos();
                let exp_p = Complex64::new(cos, -sin); // e^{-iθ/2}
                let exp_m = Complex64::new(cos, sin);  // e^{+iθ/2}
                self.apply_unary(*t, |v0, v1| (v0 * exp_p, v1 * exp_m));
            }
            Gate::CCNOT(c1, c2, t) => {
                // Toffoli (CCNOT): universal for classical reversible logic.
                self.apply_ccnot(*c1, *c2, *t)
            }
        }
    }

    /// Applies a single-qubit unitary in parallel over all amplitude pairs for qubit `t`.
    fn apply_unary<F>(&mut self, t: usize, f: F)
    where
        F: Fn(Complex64, Complex64) -> (Complex64, Complex64) + Sync + Send,
    {
        let size = 1 << self.qubit_count;
        let step = 1 << t;

        (0..size).into_par_iter().step_by(step * 2).for_each(|i| {
            for j in i..i + step {
                unsafe {
                    // Raw pointers allow rayon to mutate disjoint index pairs concurrently.
                    let ptr = self.state.as_ptr() as *mut Complex64;
                    let v0 = *ptr.add(j);
                    let v1 = *ptr.add(j + step);
                    let (new_v0, new_v1) = f(v0, v1);
                    *ptr.add(j) = new_v0;
                    *ptr.add(j + step) = new_v1;
                }
            }
        });
    }

    /// Applies a controlled unitary only on basis states where the control qubit is |1⟩.
    fn apply_controlled<F>(&mut self, c: usize, t: usize, f: F)
    where
        F: Fn(Complex64, Complex64) -> (Complex64, Complex64) + Sync + Send,
    {
        let size = 1 << self.qubit_count;
        let step_t = 1 << t;
        let mask_c = 1 << c;

        (0..size).into_par_iter().step_by(step_t * 2).for_each(|i| {
            for j in i..i + step_t {
                if (j & mask_c) != 0 {
                    unsafe {
                        let ptr = self.state.as_ptr() as *mut Complex64;
                        let v0 = *ptr.add(j);
                        let v1 = *ptr.add(j + step_t);
                        let (new_v0, new_v1) = f(v0, v1);
                        *ptr.add(j) = new_v0;
                        *ptr.add(j + step_t) = new_v1;
                    }
                }
            }
        });
    }

    /// Toffoli gate implementation: swap target amplitudes when both controls are |1⟩.
    fn apply_ccnot(&mut self, c1: usize, c2: usize, target: usize) {
        let dim = self.state.len();
        let m1 = 1 << c1;
        let m2 = 1 << c2;
        let mt = 1 << target;

        let state_ptr = self.state.as_slice_mut().expect("Memory error");
        let raw_ptr = state_ptr.as_mut_ptr() as usize;

        (0..dim).into_par_iter().for_each(|i| {
            if (i & m1) != 0 && (i & m2) != 0 {
                let target_bit = (i & mt) != 0;
                let flipped_idx = if target_bit { i & !mt } else { i | mt };

                // Touch each swap pair only once to avoid double-flipping.
                if i < flipped_idx {
                    unsafe {
                        let ptr = raw_ptr as *mut Complex64;
                        let v1 = *ptr.add(i);
                        let v2 = *ptr.add(flipped_idx);
                        *ptr.add(i) = v2;
                        *ptr.add(flipped_idx) = v1;
                    }
                }
            }
        });
    }
}

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

    /// Executes the circuit while capturing a 10-column structured trace aligned with `wqc-stark-core`.
    pub fn execute_with_trace(&self, register: &mut QuantumRegister) -> Result<Vec<f64>, String> {
        if self.qubit_count != register.qubit_count {
            return Err("Register/circuit qubit count mismatch".to_string());
        }

        let total_rows = self.gates.len() + 1;
        let mut trace = Vec::with_capacity(total_rows * TRACE_WIDTH);

        // Steps 1..N: snapshot the register immediately BEFORE each gate is applied.
        for gate in &self.gates {
            let state = &register.state;
            let gate_id = gate.to_stark_id().unwrap_or(0.0);

            // Keep trace extraction aligned with the simulator bit indexing.
            // `apply_gate` interprets qubit ids directly as bit positions.
            let get_phys_bit = |logical_q: usize| -> usize { logical_q };

            // Columns 1 & 2: control-qubit marginal probabilities at this trace row.
            let (ctrl_prob_1, ctrl_prob_2) = match gate {
                Gate::CNOT(c, _) | Gate::CZ(c, _) => {
                    let phys_ctrl = get_phys_bit(*c);
                    let mut prob = 0.0;
                    for (idx, amplitude) in state.iter().enumerate() {
                        if (idx >> phys_ctrl) & 1 == 1 {
                            prob += amplitude.re * amplitude.re + amplitude.im * amplitude.im;
                        }
                    }
                    (prob, 0.0)
                }
                Gate::CCNOT(c1, c2, _) => {
                    let phys_c1 = get_phys_bit(*c1);
                    let phys_c2 = get_phys_bit(*c2);
                    let mut prob_1 = 0.0;
                    let mut prob_2 = 0.0;
                    for (idx, amplitude) in state.iter().enumerate() {
                        let p = amplitude.re * amplitude.re + amplitude.im * amplitude.im;
                        if (idx >> phys_c1) & 1 == 1 { prob_1 += p; }
                        if (idx >> phys_c2) & 1 == 1 { prob_2 += p; }
                    }
                    (prob_1, prob_2)
                }
                _ => (0.0, 0.0),
            };

            let ctrl_active = if ctrl_prob_1 > 0.5 { 1.0 } else { 0.0 };
            let ctrl_active_2 = if ctrl_prob_2 > 0.5 { 1.0 } else { 0.0 };

            // Columns 3 & 4: trigonometric rotation components (RX/RY/RZ) or defaults.
            let (_, p_cos, p_sin) = gate.to_stark_payload(ctrl_active > 0.5);

            let logical_target = match gate {
                Gate::X(t) | Gate::Y(t) | Gate::Z(t) | Gate::H(t) | Gate::S(t) | Gate::T(t) => *t,
                Gate::CNOT(_, t) | Gate::CZ(_, t) => *t,
                Gate::CCNOT(_, _, t) => *t,
                Gate::RX(t, _) | Gate::RY(t, _) | Gate::RZ(t, _) => *t,
            };
            let phys_target = get_phys_bit(logical_target);

            // Columns 5–8: dominant |v0⟩, |v1⟩ amplitude pair on the target qubit subspace.
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

            let v0_re = state[best_v0_idx].re;
            let v0_im = state[best_v0_idx].im;
            let v1_re = state[best_v1_idx].re;
            let v1_im = state[best_v1_idx].im;

            let padding = 0.0;

            trace.push(gate_id);       // Column 0
            trace.push(ctrl_active);   // Column 1
            trace.push(ctrl_active_2); // Column 2
            trace.push(p_cos);         // Column 3
            trace.push(p_sin);         // Column 4
            trace.push(v0_re);         // Column 5
            trace.push(v0_im);         // Column 6
            trace.push(v1_re);         // Column 7
            trace.push(v1_im);         // Column 8
            trace.push(padding);       // Column 9

            // Advance the simulator state for the next trace row.
            register.apply_gate(gate);
        }

        // Step N+1: terminal boundary row (validates final register conditions for the STARK AIR).
        let logical_target = self
            .gates
            .last()
            .map(|gate| match gate {
                Gate::X(t) | Gate::Y(t) | Gate::Z(t) | Gate::H(t) | Gate::S(t) | Gate::T(t) => *t,
                Gate::CNOT(_, t) | Gate::CZ(_, t) => *t,
                Gate::CCNOT(_, _, t) => *t,
                Gate::RX(t, _) | Gate::RY(t, _) | Gate::RZ(t, _) => *t,
            })
            .unwrap_or(0);
        push_terminal_trace_row(&register.state, logical_target, &mut trace);

        Ok(trace)
    }
}

fn push_terminal_trace_row(state: &Array1<Complex64>, logical_target: usize, trace: &mut Vec<f64>) {
    let phys_target = logical_target;

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

    let v0_re = state[best_v0_idx].re;
    let v0_im = state[best_v0_idx].im;
    let v1_re = state[best_v1_idx].re;
    let v1_im = state[best_v1_idx].im;

    trace.push(0.0);     // Column 0
    trace.push(0.0);   // Column 1
    trace.push(0.0);   // Column 2
    trace.push(1.0);   // Column 3
    trace.push(0.0);   // Column 4
    trace.push(v0_re); // Column 5
    trace.push(v0_im); // Column 6
    trace.push(v1_re); // Column 7
    trace.push(v1_im); // Column 8
    trace.push(0.0);   // Column 9
}

#[cfg(test)]
mod trace_tests {
    use super::{Circuit, ContractionWorkspace};

    #[test]
    fn empty_circuit_emits_terminal_trace_row() {
        let mut workspace = ContractionWorkspace::try_allocate(3, 30).expect("allocate");
        let circuit = Circuit::new(3);
        let trace = circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");
        assert_eq!(trace.len(), 10, "empty circuit should emit one 10-column boundary row");
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
