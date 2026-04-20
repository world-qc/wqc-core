use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)] // Derive Serde traits
#[serde(tag = "type", content = "params")]    // Use "adjacently tagged" for clear JSON structure
pub enum Gate {
    #[serde(rename = "H")]
    H(usize),
    #[serde(rename = "X")]
    X(usize),
    #[serde(rename = "Y")]
    Y(usize),
    #[serde(rename = "Z")]
    Z(usize),
    #[serde(rename = "T")]
    T(usize),
    #[serde(rename = "CNOT")]
    CNOT(usize, usize), // (control, target)
    #[serde(rename = "CCNOT")]
    CCNOT(usize, usize, usize), // (control1, control2, target)
}

pub struct QuantumRegister {
    pub state: Array1<Complex64>,
    pub qubit_count: usize,
}

impl QuantumRegister {
    pub fn new(qubit_count: usize) -> Self {
        let dim = 1 << qubit_count;
        let mut state = Array1::from_elem(dim, Complex64::new(0.0, 0.0));
        state[0] = Complex64::new(1.0, 0.0);
        Self { state, qubit_count }
    }

    pub fn apply_gate(&mut self, gate: &Gate) {
        match gate {
            Gate::H(t) => self.apply_1q_gate(*t, [
                Complex64::new(1.0 / 2.0f64.sqrt(), 0.0), Complex64::new(1.0 / 2.0f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0f64.sqrt(), 0.0), Complex64::new(-1.0 / 2.0f64.sqrt(), 0.0)
            ]),
            Gate::X(t) => self.apply_1q_gate(*t, [
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)
            ]),
            Gate::Y(t) => self.apply_1q_gate(*t, [
                Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)
            ]),
            Gate::Z(t) => self.apply_1q_gate(*t, [
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)
            ]),
            Gate::T(t) => self.apply_1q_gate(*t, [
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2)
            ]),
            Gate::CNOT(c, t) => self.apply_cnot(*c, *t),
            Gate::CCNOT(c1, c2, t) => self.apply_ccnot(*c1, *c2, *t),
        }
    }

    fn apply_1q_gate(&mut self, target: usize, matrix: [Complex64; 4]) {
        let dim = self.state.len();
        let dist = 1 << target;
        let state_ptr = self.state.as_slice_mut().expect("Failed to get mutable slice");
        let raw_ptr = state_ptr.as_mut_ptr() as usize;

        // Parallelize loops to maximize CPU-RAM bandwidth
        (0..(dim / (2 * dist))).into_par_iter().for_each(|i| {
            for j in 0..dist {
                let idx0 = i * 2 * dist + j;
                let idx1 = idx0 + dist;

                unsafe {
                    let ptr = raw_ptr as *mut Complex64;
                    let v0 = *ptr.add(idx0);
                    let v1 = *ptr.add(idx1);

                    *ptr.add(idx0) = matrix[0] * v0 + matrix[1] * v1;
                    *ptr.add(idx1) = matrix[2] * v0 + matrix[3] * v1;
                }
            }
        });
    }

    fn apply_cnot(&mut self, control: usize, target: usize) {
        let dim = self.state.len();
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        let state_ptr = self.state.as_slice_mut().expect("Failed to get mutable slice");
        let raw_ptr = state_ptr.as_mut_ptr() as usize;

        (0..dim).into_par_iter().for_each(|i| {
            if (i & c_mask) != 0 {
                let target_bit = (i & t_mask) != 0;
                let flipped_idx = if target_bit { i & !t_mask } else { i | t_mask };

                // Only swap once per pair
                if i < flipped_idx {
                    unsafe {
                        let ptr = raw_ptr as *mut Complex64;
                        let v_current = *ptr.add(i);
                        let v_flipped = *ptr.add(flipped_idx);
                        *ptr.add(i) = v_flipped;
                        *ptr.add(flipped_idx) = v_current;
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
    pub gates: Vec<Gate>,
    pub qubit_count: usize,
}

impl Circuit {
    pub fn new(qubit_count: usize) -> Self {
        Self {
            gates: Vec::new(),
            qubit_count,
        }
    }

    /// Add a gate to the circuit
    pub fn add(&mut self, gate: Gate) {
        // Simple safety check: ensures the gate doesn't target out-of-bounds qubits
        match gate {
            Gate::H(t) | Gate::X(t) | Gate::Y(t) | Gate::Z(t) | Gate::T(t) => {
                assert!(
                    t < self.qubit_count,
                    "Target qubit index out of bounds"
                );
            }
            Gate::CNOT(c, t) => {
                assert!(
                    c < self.qubit_count && t < self.qubit_count,
                    "Control or Target index out of bounds"
                );
            }
            Gate::CCNOT(c1, c2, t) => {
                assert!(
                    c1 < self.qubit_count && c2 < self.qubit_count && t < self.qubit_count,
                    "Control or Target index out of bounds"
                );
            }
        }
        self.gates.push(gate);
    }

    /// Execute the entire circuit on a QuantumRegister
    pub fn execute(&self, register: &mut QuantumRegister) {
        assert_eq!(self.qubit_count, register.qubit_count, "Circuit/Register qubit mismatch");

        println!("Executing circuit with {} gates on {} qubits...", self.gates.len(), self.qubit_count);
        for gate in &self.gates {
            register.apply_gate(gate);
        }
    }
}
