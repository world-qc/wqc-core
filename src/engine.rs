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
    ///
    /// Each gate emits two rows: a pre-gate snapshot (active `gate_id`) and a post-gate snapshot
    /// (`gate_id = 0`) on the same target qubit. Column 9 records that target index so the AIR can
    /// skip amplitude constraints on cross-wire transitions (multi-target circuits).
    pub fn execute_with_trace(&self, register: &mut QuantumRegister) -> Result<Vec<f64>, String> {
        if self.qubit_count != register.qubit_count {
            return Err("Register/circuit qubit count mismatch".to_string());
        }

        let total_rows = self.gates.len() * 2 + 1;
        let mut trace = Vec::with_capacity(total_rows * TRACE_WIDTH);

        for gate in &self.gates {
            let logical_target = gate_logical_target(gate);
            push_gate_snapshot_row(&register.state, gate, logical_target, &mut trace);
            register.apply_gate(gate);
            push_post_gate_row(&register.state, logical_target, &mut trace);
        }

        let terminal_target = self
            .gates
            .last()
            .map(gate_logical_target)
            .unwrap_or(0);
        push_terminal_trace_row(&register.state, terminal_target, &mut trace);
        apply_transition_links(&mut trace);

        Ok(trace)
    }
}

/// Sets column 10 on each row: `1` when the next row samples the same target qubit.
fn apply_transition_links(trace: &mut [f64]) {
    let row_count = trace.len() / TRACE_WIDTH;
    for row in 0..row_count {
        let base = row * TRACE_WIDTH;
        let link = if row + 1 < row_count {
            let curr_target = trace[base + 9];
            let next_target = trace[(row + 1) * TRACE_WIDTH + 9];
            if (curr_target - next_target).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        trace[base + 10] = link;
    }
}

fn gate_logical_target(gate: &Gate) -> usize {
    match gate {
        Gate::X(t) | Gate::Y(t) | Gate::Z(t) | Gate::H(t) | Gate::S(t) | Gate::T(t) => *t,
        Gate::CNOT(_, t) | Gate::CZ(_, t) => *t,
        Gate::CCNOT(_, _, t) => *t,
        Gate::RX(t, _) | Gate::RY(t, _) | Gate::RZ(t, _) => *t,
    }
}

fn sample_target_amplitudes(
    state: &Array1<Complex64>,
    phys_target: usize,
) -> (f64, f64, f64, f64) {
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

    (
        state[best_v0_idx].re,
        state[best_v0_idx].im,
        state[best_v1_idx].re,
        state[best_v1_idx].im,
    )
}

fn push_trace_row(
    trace: &mut Vec<f64>,
    gate_id: f64,
    ctrl_active: f64,
    ctrl_active_2: f64,
    p_cos: f64,
    p_sin: f64,
    v0_re: f64,
    v0_im: f64,
    v1_re: f64,
    v1_im: f64,
    target_qubit: f64,
) {
    trace.push(gate_id);
    trace.push(ctrl_active);
    trace.push(ctrl_active_2);
    trace.push(p_cos);
    trace.push(p_sin);
    trace.push(v0_re);
    trace.push(v0_im);
    trace.push(v1_re);
    trace.push(v1_im);
    trace.push(target_qubit);
    trace.push(0.0); // transition_link filled by apply_transition_links
}

fn push_gate_snapshot_row(
    state: &Array1<Complex64>,
    gate: &Gate,
    logical_target: usize,
    trace: &mut Vec<f64>,
) {
    let gate_id = gate.to_stark_id().unwrap_or(0.0);
    let phys_target = logical_target;

    let (ctrl_prob_1, ctrl_prob_2) = match gate {
        Gate::CNOT(c, _) | Gate::CZ(c, _) => {
            let phys_ctrl = *c;
            let mut prob = 0.0;
            for (idx, amplitude) in state.iter().enumerate() {
                if (idx >> phys_ctrl) & 1 == 1 {
                    prob += amplitude.re * amplitude.re + amplitude.im * amplitude.im;
                }
            }
            (prob, 0.0)
        }
        Gate::CCNOT(c1, c2, _) => {
            let mut prob_1 = 0.0;
            let mut prob_2 = 0.0;
            for (idx, amplitude) in state.iter().enumerate() {
                let p = amplitude.re * amplitude.re + amplitude.im * amplitude.im;
                if (idx >> c1) & 1 == 1 {
                    prob_1 += p;
                }
                if (idx >> c2) & 1 == 1 {
                    prob_2 += p;
                }
            }
            (prob_1, prob_2)
        }
        _ => (0.0, 0.0),
    };

    let ctrl_active = if ctrl_prob_1 > 0.5 { 1.0 } else { 0.0 };
    let ctrl_active_2 = if ctrl_prob_2 > 0.5 { 1.0 } else { 0.0 };
    let (_, p_cos, p_sin) = gate.to_stark_payload(ctrl_active > 0.5);
    let (v0_re, v0_im, v1_re, v1_im) = sample_target_amplitudes(state, phys_target);

    push_trace_row(
        trace,
        gate_id,
        ctrl_active,
        ctrl_active_2,
        p_cos,
        p_sin,
        v0_re,
        v0_im,
        v1_re,
        v1_im,
        logical_target as f64,
    );
}

fn push_post_gate_row(state: &Array1<Complex64>, logical_target: usize, trace: &mut Vec<f64>) {
    let (v0_re, v0_im, v1_re, v1_im) = sample_target_amplitudes(state, logical_target);
    push_trace_row(
        trace,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        v0_re,
        v0_im,
        v1_re,
        v1_im,
        logical_target as f64,
    );
}

fn push_terminal_trace_row(state: &Array1<Complex64>, logical_target: usize, trace: &mut Vec<f64>) {
    let (v0_re, v0_im, v1_re, v1_im) = sample_target_amplitudes(state, logical_target);
    push_trace_row(
        trace,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        v0_re,
        v0_im,
        v1_re,
        v1_im,
        logical_target as f64,
    );
}

#[cfg(test)]
mod trace_tests {
    use super::{Circuit, ContractionWorkspace, Gate, TRACE_WIDTH};

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
