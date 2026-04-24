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

    /// Execute the circuit on a QuantumRegister with safety checks
    pub fn execute(&self, register: &mut QuantumRegister) -> Result<(), String> {
        if self.qubit_count != register.qubit_count {
            return Err("Register/Circuit qubit count mismatch".to_string());
        }
        for gate in &self.gates {
            register.apply_gate(gate);
        }
        Ok(())
    }
}
