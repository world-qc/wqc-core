use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumIter};

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
    pub fn new(qubit_count: usize) -> Self {
        let dim = 1 << qubit_count;
        let mut state = Array1::from_elem(dim, Complex64::new(0.0, 0.0));
        state[0] = Complex64::new(1.0, 0.0);
        Self { state, qubit_count }
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
        // Validation: ensures all target/control qubits are within the allowed range
        match &gate {
            // 1-qubit gates
            Gate::H(t) | Gate::X(t) | Gate::Y(t) | Gate::Z(t) | Gate::T(t) | Gate::S(t) => {
                assert!(t < &self.qubit_count, "Target qubit index {} out of bounds", t);
            }
            // 1-qubit gates with parameters (Rotation gates)
            Gate::RX(t, _) | Gate::RY(t, _) | Gate::RZ(t, _) => {
                assert!(t < &self.qubit_count, "Target qubit index {} out of bounds", t);
            }
            // 2-qubit gates
            Gate::CNOT(c, t) | Gate::CZ(c, t) => {
                assert!(
                    c < &self.qubit_count && t < &self.qubit_count,
                    "Control ({}) or Target ({}) index out of bounds", c, t
                );
            }
            // 3-qubit gates
            Gate::CCNOT(c1, c2, t) => {
                assert!(
                    c1 < &self.qubit_count && c2 < &self.qubit_count && t < &self.qubit_count,
                    "Control ({}, {}) or Target ({}) index out of bounds", c1, c2, t
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
