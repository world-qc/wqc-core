//! Gate unitary tensors for TN / MPS contraction.

use crate::engine::Gate;
use num_complex::Complex64;

pub type Mat2 = [[Complex64; 2]; 2];
pub type Mat4 = [[[[Complex64; 2]; 2]; 2]; 2]; // [ctrl_out][tgt_out][ctrl_in][tgt_in]

pub fn unary_matrix(gate: &Gate) -> Mat2 {
    match gate {
        Gate::H(_) => {
            let k = 1.0 / 2.0f64.sqrt();
            let kc = Complex64::new(k, 0.0);
            [[kc, kc], [kc, -kc]]
        }
        Gate::X(_) => [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ],
        Gate::Y(_) => {
            let i = Complex64::i();
            [
                [Complex64::new(0.0, 0.0), -i],
                [i, Complex64::new(0.0, 0.0)],
            ]
        }
        Gate::Z(_) => [
            [Complex64::ONE, Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), -Complex64::ONE],
        ],
        Gate::S(_) => {
            let i = Complex64::i();
            [
                [Complex64::ONE, Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), i],
            ]
        }
        Gate::T(_) => {
            let factor = Complex64::new(
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            );
            [
                [Complex64::ONE, Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), factor],
            ]
        }
        Gate::RX(_, theta) => {
            let (sin, cos) = (theta / 2.0).sin_cos();
            let cos_c = Complex64::new(cos, 0.0);
            let n_i_sin = Complex64::new(0.0, -sin);
            [[cos_c, n_i_sin], [n_i_sin, cos_c]]
        }
        Gate::RY(_, theta) => {
            let (sin, cos) = (theta / 2.0).sin_cos();
            [
                [Complex64::new(cos, 0.0), Complex64::new(-sin, 0.0)],
                [Complex64::new(sin, 0.0), Complex64::new(cos, 0.0)],
            ]
        }
        Gate::RZ(_, theta) => {
            let (sin, cos) = (theta / 2.0).sin_cos();
            [
                [Complex64::new(cos, -sin), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(cos, sin)],
            ]
        }
        _ => panic!("not a unary gate: {gate:?}"),
    }
}

pub fn cnot_matrix() -> Mat4 {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    [
        [[[o, z], [z, o]], [[z, z], [z, z]]],
        [[[z, z], [z, z]], [[z, o], [o, z]]],
    ]
}

pub fn cz_matrix() -> Mat4 {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    let m = -Complex64::ONE;
    [
        [[[o, z], [z, z]], [[z, z], [z, z]]],
        [[[z, z], [o, z]], [[z, z], [z, m]]],
    ]
}

pub fn swap_matrix() -> Mat4 {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    [
        [[[o, z], [z, z]], [[z, o], [z, z]]],
        [[[z, z], [o, z]], [[z, z], [z, o]]],
    ]
}

pub fn two_qubit_matrix(gate: &Gate) -> Mat4 {
    match gate {
        Gate::CNOT(..) => cnot_matrix(),
        Gate::CZ(..) => cz_matrix(),
        _ => panic!("not a two-qubit gate: {gate:?}"),
    }
}

pub fn max_bond_dim_from_env() -> usize {
    std::env::var("WQC_MPS_MAX_BOND_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&chi| chi > 0)
        .unwrap_or(128)
}

/// Per-task χ from the orchestrator, capped by the node/core env default.
pub fn resolve_bond_dim(task_override: Option<usize>) -> usize {
    let env = max_bond_dim_from_env();
    match task_override.filter(|&chi| chi > 0) {
        Some(task) => env.min(task),
        None => env,
    }
}

/// Bond dimension large enough for exact simulation on `qubit_count` wires.
pub fn exact_bond_dim(qubit_count: usize) -> usize {
    1usize << (qubit_count.min(20) / 2 + 1)
}
