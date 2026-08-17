//! Dense exact tensor-network backend: gate tensors contracted into a rank-N state tensor
//! (`2^N` complex amplitudes). This is the reference TN executor until bond-truncated MPS / GPU paths land.

use crate::engine::{EngineError, Gate};
use crate::memory_budget::max_wqc_memory_bytes_from_total;
use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;
use sysinfo::System;

/// Rank-N product-state tensor `|ψ⟩` with one physical index per qubit wire (dim 2 each).
pub struct DenseTnState {
    pub state: Array1<Complex64>,
    pub qubit_count: usize,
}

impl DenseTnState {
    /// Allocates `2^N` amplitudes after the RAM guard used by the PoUW worker.
    pub fn try_new(qubit_count: usize) -> Result<Self, EngineError> {
        if qubit_count == 0 || qubit_count > 40 {
            return Err(EngineError::InvalidQubitCount(qubit_count));
        }

        let required_memory = (1u64 << qubit_count).saturating_mul(16);
        if !cfg!(test) {
            let mut sys = System::new_all();
            sys.refresh_memory();

            let total_memory = sys.total_memory();
            if total_memory > 0 {
                let budget = max_wqc_memory_bytes_from_total(total_memory);
                if required_memory > budget {
                    return Err(EngineError::InsufficientMemory {
                        required: required_memory,
                        available: budget,
                    });
                }
            }
        }

        let dim = 1 << qubit_count;
        let mut state = Array1::from_elem(dim, Complex64::new(0.0, 0.0));
        state[0] = Complex64::new(1.0, 0.0);
        Ok(Self { state, qubit_count })
    }

    /// Amplitude at compact computational basis |0…0⟩ (compact-register scalar readout).
    pub fn amplitude_at_compact_zero(&self) -> Complex64 {
        self.state[0]
    }

    /// Contract a gate tensor into the open wire indices of the network state.
    pub fn apply_gate(&mut self, gate: &Gate) {
        match gate {
            Gate::H(t) => {
                let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
                self.apply_unary(*t, |v0, v1| ((v0 + v1) * inv_sqrt2, (v0 - v1) * inv_sqrt2));
            }
            Gate::X(t) => self.apply_unary(*t, |v0, v1| (v1, v0)),
            Gate::Y(t) => {
                let i = Complex64::i();
                self.apply_unary(*t, |v0, v1| (v1 * (-i), v0 * i));
            }
            Gate::Z(t) => self.apply_unary(*t, |v0, v1| (v0, -v1)),
            Gate::S(t) => {
                let i = Complex64::i();
                self.apply_unary(*t, |v0, v1| (v0, v1 * i));
            }
            Gate::T(t) => {
                let factor = Complex64::new(
                    std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                );
                self.apply_unary(*t, |v0, v1| (v0, v1 * factor));
            }
            Gate::CNOT(c, t) => self.apply_controlled(*c, *t, |v0, v1| (v1, v0)),
            Gate::CZ(c, t) => self.apply_controlled(*c, *t, |v0, v1| (v0, -v1)),
            Gate::RX(t, theta) => {
                let (sin, cos) = (theta / 2.0).sin_cos();
                let cos_c = Complex64::new(cos, 0.0);
                let n_i_sin_c = Complex64::new(0.0, -sin);
                self.apply_unary(*t, |v0, v1| {
                    (v0 * cos_c + v1 * n_i_sin_c, v0 * n_i_sin_c + v1 * cos_c)
                });
            }
            Gate::RY(t, theta) => {
                let (sin, cos) = (theta / 2.0).sin_cos();
                self.apply_unary(*t, |v0, v1| (v0 * cos - v1 * sin, v0 * sin + v1 * cos));
            }
            Gate::RZ(t, theta) => {
                let (sin, cos) = (theta / 2.0).sin_cos();
                let exp_p = Complex64::new(cos, -sin);
                let exp_m = Complex64::new(cos, sin);
                self.apply_unary(*t, |v0, v1| (v0 * exp_p, v1 * exp_m));
            }
            Gate::CCNOT(c1, c2, t) => self.apply_ccnot(*c1, *c2, *t),
            Gate::MEASURE(_) | Gate::RESET(_) | Gate::IF(_) => {}
        }
    }

    fn apply_unary<F>(&mut self, t: usize, f: F)
    where
        F: Fn(Complex64, Complex64) -> (Complex64, Complex64) + Sync + Send,
    {
        let size = 1 << self.qubit_count;
        let step = 1 << t;

        (0..size).into_par_iter().step_by(step * 2).for_each(|i| {
            for j in i..i + step {
                unsafe {
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
            if (i & m1) != 0 && (i & m2) != 0 {
                let target_bit = (i & mt) != 0;
                let flipped_idx = if target_bit { i & !mt } else { i | mt };
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
