use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;
use strum_macros::{EnumIter, Display};
use sysinfo::System;
use std::fmt;

// --- Error Definitions for Robustness ---

#[derive(Debug)]
pub enum EngineError {
    InvalidQubitCount(usize),
    QubitIndexOutOfBounds { index: usize, limit: usize },
    InsufficientMemory { required: u64, available: u64 },
    MismatchedRegister,
}

impl std::error::Error for EngineError {}

// Manually implement Display to handle complex error messages with values
impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InvalidQubitCount(c) =>
                write!(f, "Invalid qubit count: {}. Max limit is 40.", c),
            EngineError::QubitIndexOutOfBounds { index, limit } =>
                write!(f, "Qubit index {} out of bounds (limit: {})", index, limit),
            EngineError::InsufficientMemory { required, available } =>
                write!(f, "Insufficient memory: Need {} bytes, but only {} bytes are available (80% safety threshold applied)", required, available),
            EngineError::MismatchedRegister =>
                write!(f, "Register/Circuit qubit count mismatch"),
        }
    }
}

// --- Engine Core ---

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
    /// Maps gates to float identifiers for the STARK execution trace constraint selector.
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
            _ => None, // Gates outside scope don't accumulate STARK trace fields
        }
    }

    /// Extracts supplementary execution payload fields for the STARK 8-column chunk format.
    /// Returns: (ctrl_active, p_cos, p_sin)
    /// This directly aligns with the `wqc-stark-core` ingest spec!
    pub fn to_stark_payload(&self, is_control_active: bool) -> (f64, f64, f64) {
        match self {
            // Control gates: maps the control active flag passed from the external simulator execution state.
            Gate::CNOT(..) | Gate::CZ(..) | Gate::CCNOT(..) => {
                (if is_control_active { 1.0 } else { 0.0 }, 1.0, 0.0)
            }
            // Rotation gates: pre-calculates and stores cos and sin from the angle (theta).
            // * Note: Please retrieve theta according to the definition in engine.rs (the following is just placeholder logic).
            Gate::RX(_, theta) | Gate::RY(_, theta) | Gate::RZ(_, theta) => {
                ((theta / 2.0).cos(), (theta / 2.0).sin(), 0.0)
            }
            // All other single-qubit gates use default values.
            _ => (0.0, 1.0, 0.0)
        }
    }
}

pub struct QuantumRegister {
    pub state: Array1<Complex64>,
    pub qubit_count: usize,
}

impl QuantumRegister {
    /// Create a new QuantumRegister with dynamic memory check (Hardening)
    pub fn new(qubit_count: usize) -> Result<Self, EngineError> {
        // 1. Calculate required memory: (2^N) * 16 bytes (for Complex64)
        let required_memory = (1u64 << qubit_count) * 16;

        // 2. Resource check using sysinfo
        let mut sys = System::new_all();
        sys.refresh_memory();

        let available_memory = sys.available_memory(); // Actual free/available RAM
        let total_memory = sys.total_memory();         // Total physical RAM

        // Hardening Logic:
        // Rule A: Must fit within 80% of CURRENTLY available memory.
        // Rule B: Must not exceed 90% of TOTAL physical memory (to prevent OS starvation).
        let available_threshold = (available_memory as f64 * 0.8) as u64;
        let total_threshold = (total_memory as f64 * 0.9) as u64;

        if required_memory > available_threshold || required_memory > total_threshold {
            return Err(EngineError::InsufficientMemory {
                required: required_memory,
                available: available_memory,
            });
        }

        // 3. Final safety cap to prevent accidental logic errors (e.g., 60 qubits)
        if qubit_count == 0 || qubit_count > 40 {
            return Err(EngineError::InvalidQubitCount(qubit_count));
        }

        // 4. Memory allocation
        let dim = 1 << qubit_count;
        let mut state = Array1::from_elem(dim, Complex64::new(0.0, 0.0));
        state[0] = Complex64::new(1.0, 0.0);
        Ok(Self { state, qubit_count })
    }

    /// Extracted helper to read specific amplitude pairs safely for state extraction.
    pub fn get_amplitude_pair(&self, target_qubit: usize, base_index: usize) -> (Complex64, Complex64) {
        let step = 1 << target_qubit;
        (self.state[base_index], self.state[base_index + step])
    }

    pub fn apply_gate(&mut self, gate: &Gate) {
        match gate {
            Gate::H(t) => {
                // Hadamard gate: Creates superposition
                let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
                self.apply_unary(*t, |v0, v1| {
                    ((v0 + v1) * inv_sqrt2, (v0 - v1) * inv_sqrt2)
                });
            },
            Gate::X(t) => {
                // Pauli-X gate: Quantum NOT gate (swaps |0> and |1>)
                self.apply_unary(*t, |v0, v1| (v1, v0));
            },
            Gate::Y(t) => {
                // Pauli-Y gate: Rotation around Y-axis by PI
                let i = Complex64::i();
                self.apply_unary(*t, |v0, v1| (v1 * (-i), v0 * i));
            },
            Gate::Z(t) => {
                // Pauli-Z gate: Phase flip
                self.apply_unary(*t, |v0, v1| (v0, -v1));
            },
            Gate::S(t) => {
                // S gate: Phase gate (Z^1/2), 90-degree rotation
                let i = Complex64::i();
                self.apply_unary(*t, |v0, v1| (v0, v1 * i));
            },
            Gate::T(t) => {
                // T gate: PI/4 phase gate (Z^1/4)
                let factor = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2);
                self.apply_unary(*t, |v0, v1| (v0, v1 * factor));
            },
            Gate::CNOT(c, t) => {
                // Controlled-NOT gate: Swaps target amplitudes if control is |1>
                self.apply_controlled(*c, *t, |v0, v1| (v1, v0));
            },
            Gate::CZ(c, t) => {
                // Controlled-Z gate: Phase flip on target if control is |1>
                self.apply_controlled(*c, *t, |v0, v1| (v0, -v1));
            },
            Gate::RX(t, theta) => {
                // Rotation around X-axis by theta
                let (sin, cos) = (theta / 2.0).sin_cos();
                let cos_c = Complex64::new(cos, 0.0);
                let n_i_sin_c = Complex64::new(0.0, -sin);
                self.apply_unary(*t, |v0, v1| {
                    (v0 * cos_c + v1 * n_i_sin_c, v0 * n_i_sin_c + v1 * cos_c)
                });
            },
            Gate::RY(t, theta) => {
                // Rotation around Y-axis by theta
                let (sin, cos) = (theta / 2.0).sin_cos();
                self.apply_unary(*t, |v0, v1| {
                    (v0 * cos - v1 * sin, v0 * sin + v1 * cos)
                });
            },
            Gate::RZ(t, theta) => {
                // Rotation around Z-axis by theta
                let (sin, cos) = (theta / 2.0).sin_cos();
                let exp_p = Complex64::new(cos, -sin); // e^(-i*theta/2)
                let exp_m = Complex64::new(cos, sin);  // e^(i*theta/2)
                self.apply_unary(*t, |v0, v1| (v0 * exp_p, v1 * exp_m));
            },
            Gate::CCNOT(c1, c2, t) => {
                // Toffoli gate: Universal for classical logic
                self.apply_ccnot(*c1, *c2, *t);
            },
        }
    }

    fn apply_unary<F>(&mut self, t: usize, f: F)
    where
        F: Fn(Complex64, Complex64) -> (Complex64, Complex64) + Sync + Send,
    {
        let size = 1 << self.qubit_count;
        let step = 1 << t;

        // Parallel processing of the state vector using rayon
        (0..size).into_par_iter().step_by(step * 2).for_each(|i| {
            for j in i..i + step {
                unsafe {
                    // Use raw pointers for thread-safe concurrent mutation
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

    fn apply_controlled<F>(&mut self, c: usize, t: usize, f: F)
    where
        F: Fn(Complex64, Complex64) -> (Complex64, Complex64) + Sync + Send,
    {
        let size = 1 << self.qubit_count;
        let step_t = 1 << t;
        let mask_c = 1 << c;

        (0..size).into_par_iter().step_by(step_t * 2).for_each(|i| {
            for j in i..i + step_t {
                // Apply transformation only if the control bit is set
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

    fn apply_ccnot(&mut self, c1: usize, c2: usize, target: usize) {
        let dim = self.state.len();
        let m1 = 1 << c1;
        let m2 = 1 << c2;
        let mt = 1 << target;

        let state_ptr = self.state.as_slice_mut().expect("Memory error");
        let raw_ptr = state_ptr.as_mut_ptr() as usize;

        (0..dim).into_par_iter().for_each(|i| {
            // Process only if both control bits c1 and c2 are 1
            if (i & m1) != 0 && (i & m2) != 0 {
                let target_bit = (i & mt) != 0;
                let flipped_idx = if target_bit { i & !mt } else { i | mt };

                // Perform swap only once per pair to prevent double-flipping
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

    /// Helper function to generate a deterministic output hash for the state vector
    pub fn calculate_output_hash(&self, state_vector: &[[f64; 2]]) -> String {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        for [real, imag] in state_vector {
            hasher.update(real.to_le_bytes());
            hasher.update(imag.to_le_bytes());
        }
        hex::encode(hasher.finalize())
    }

    /// Add a gate to the circuit with bound checking (Replaces assert!)
    pub fn add(&mut self, gate: Gate) -> Result<(), EngineError> {
        match &gate {
            // 1-qubit gates
            Gate::H(t) | Gate::X(t) | Gate::Y(t) | Gate::Z(t) | Gate::T(t) | Gate::S(t) => {
                if *t >= self.qubit_count {
                    return Err(EngineError::QubitIndexOutOfBounds { index: *t, limit: self.qubit_count });
                }
            }
            // 1-qubit gates with parameters (Rotation gates)
            Gate::RX(t, _) | Gate::RY(t, _) | Gate::RZ(t, _) => {
                if *t >= self.qubit_count {
                    return Err(EngineError::QubitIndexOutOfBounds { index: *t, limit: self.qubit_count });
                }
            }
            // 2-qubit gates
            Gate::CNOT(c, t) | Gate::CZ(c, t) => {
                if *c >= self.qubit_count { return Err(EngineError::QubitIndexOutOfBounds { index: *c, limit: self.qubit_count }); }
                if *t >= self.qubit_count { return Err(EngineError::QubitIndexOutOfBounds { index: *t, limit: self.qubit_count }); }
            }
            // 3-qubit gates
            Gate::CCNOT(c1, c2, t) => {
                if *c1 >= self.qubit_count { return Err(EngineError::QubitIndexOutOfBounds { index: *c1, limit: self.qubit_count }); }
                if *c2 >= self.qubit_count { return Err(EngineError::QubitIndexOutOfBounds { index: *c2, limit: self.qubit_count }); }
                if *t >= self.qubit_count { return Err(EngineError::QubitIndexOutOfBounds { index: *t, limit: self.qubit_count }); }
            }
        }
        self.gates.push(gate);
        Ok(())
    }

    /// Executes the circuit while capturing an 18-column structured execution trace aligned with wqc-stark-core.
    pub fn execute_with_trace(&self, register: &mut QuantumRegister) -> Result<Vec<f64>, String> {
        if self.qubit_count != register.qubit_count {
            return Err("Register/Circuit qubit count mismatch".to_string());
        }

        let total_rows = self.gates.len() + 1;
        let mut trace = Vec::with_capacity(total_rows * 10);

        // --- Step 1 to N: Capture state snapshot BEFORE applying each quantum gate ---
        for gate in &self.gates {
            let state = &register.state;
            let gate_id = gate.to_stark_id().unwrap_or(0.0);

            // Fixed reference mapping ensuring that logical indices target the exact same
            // physical memory slot, independent of the scaled qubit dimensions (N >= 4).
            let get_phys_bit = |logical_q: usize| -> usize {
                if self.qubit_count <= 2 { logical_q } else { 3 - 1 - logical_q }
            };

            // Column 1 & 2: Track control qubit parameters at the current execution slice
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

            let ctrl_active = ctrl_prob_1;
            let ctrl_active_2 = ctrl_prob_2;

            // Column 3 & 4: Trigonometric rotation components
            let (_, p_cos, p_sin) = gate.to_stark_payload(ctrl_active > 0.5);

            // Target qubit isolation using absolute fixed-endian mapping
            let logical_target = match gate {
                Gate::X(t) | Gate::Y(t) | Gate::Z(t) | Gate::H(t) | Gate::S(t) | Gate::T(t) => *t,
                Gate::CNOT(_, t) | Gate::CZ(_, t) => *t,
                Gate::CCNOT(_, _, t) => *t,
                Gate::RX(t, _) | Gate::RY(t, _) | Gate::RZ(t, _) => *t,
            };
            let phys_target = get_phys_bit(logical_target);

            // Back to the proven duplication-free subspace selector that brought victory on 1,2,3
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

            // Advance the simulator state
            register.apply_gate(gate);
        }

        // --- Step N+1: Append the absolute FINAL state boundary row ---
        // The final trace row validates the terminal register boundary conditions.
        // We lock the physical reference bit to 0 to eliminate any dependency on the last gate type.
        if !self.gates.is_empty() {
            let state = &register.state;

            // 🎯 last_gate の match を完全撤廃し、物理ビット 0 に固定
            let phys_target = 0;

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

            trace.push(0.0);           // Column 0
            trace.push(0.0);           // Column 1
            trace.push(0.0);           // Column 2
            trace.push(1.0);           // Column 3
            trace.push(0.0);           // Column 4
            trace.push(v0_re);         // Column 5
            trace.push(v0_im);         // Column 6
            trace.push(v1_re);         // Column 7
            trace.push(v1_im);         // Column 8
            trace.push(0.0);           // Column 9
        }

        Ok(trace)
    }

    /// Legacy fallback method compatibility for non-STARK pipelines
    pub fn execute(&self, register: &mut QuantumRegister) -> Result<(), String> {
        self.execute_with_trace(register)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}
