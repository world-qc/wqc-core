//! Bond-truncated matrix-product-state (MPS) tensor-network backend.
//!
//! Memory scales as `O(N · χ²)` with bond dimension `χ` (`WQC_MPS_MAX_BOND_DIM`, default 128).
//! Set `WQC_TN_BACKEND=webgpu` (with `--features webgpu`) to offload 1-qubit and merge kernels.

#[cfg(feature = "webgpu")]
use std::sync::Arc;

use crate::engine::{EngineError, Gate};
use ndarray::{Array1, Array3, Array4};
use nalgebra::DMatrix;
use num_complex::Complex64;

use super::gates::{
    max_bond_dim_from_env, swap_matrix, two_qubit_matrix, unary_matrix, Mat4,
};

#[cfg(feature = "webgpu")]
use super::backend::{tn_backend_from_env, TnBackend};
#[cfg(feature = "webgpu")]
use super::gpu::GpuMpsDevice;

/// MPS executor: one site tensor per qubit wire `[left_bond × 2 × right_bond]`.
pub struct MpsState {
    pub sites: Vec<Array3<Complex64>>,
    pub qubit_count: usize,
    pub max_bond_dim: usize,
    #[cfg(feature = "webgpu")]
    gpu: Option<Arc<GpuMpsDevice>>,
    #[cfg(feature = "webgpu")]
    using_webgpu: bool,
}

impl MpsState {
    pub fn try_new(qubit_count: usize) -> Result<Self, EngineError> {
        Self::try_new_with_bond(qubit_count, max_bond_dim_from_env())
    }

    pub fn try_new_with_bond(qubit_count: usize, max_bond_dim: usize) -> Result<Self, EngineError> {
        if qubit_count == 0 || qubit_count > 40 {
            return Err(EngineError::InvalidQubitCount(qubit_count));
        }
        if max_bond_dim == 0 {
            return Err(EngineError::ExecutionFailed(
                "MPS max bond dimension must be > 0".into(),
            ));
        }

        let estimated_bytes = (qubit_count as u64)
            .saturating_mul(max_bond_dim as u64)
            .saturating_mul(max_bond_dim as u64)
            .saturating_mul(32);

        if !cfg!(test) {
            let mut sys = sysinfo::System::new_all();
            sys.refresh_memory();
            let available = sys.available_memory();
            if available > 0 && estimated_bytes > (available as f64 * 0.8) as u64 {
                return Err(EngineError::InsufficientMemory {
                    required: estimated_bytes,
                    available,
                });
            }
        }

        let mut sites = Vec::with_capacity(qubit_count);
        for i in 0..qubit_count {
            let left = 1usize;
            let right = 1usize;
            let mut tensor = Array3::<Complex64>::zeros((left, 2, right));
            tensor[[0, 0, 0]] = Complex64::ONE;
            let _ = i;
            sites.push(tensor);
        }

        Ok(Self {
            sites,
            qubit_count,
            max_bond_dim,
            #[cfg(feature = "webgpu")]
            gpu: init_gpu_device(),
            #[cfg(feature = "webgpu")]
            using_webgpu: false,
        })
    }

    /// Human-readable TN backend label (`cpu` or `webgpu`).
    pub fn backend_label(&self) -> &'static str {
        #[cfg(feature = "webgpu")]
        {
            if self.using_webgpu {
                return "webgpu";
            }
        }
        "cpu"
    }

    /// Peak GPU buffer bytes observed during this state lifetime (0 on CPU-only).
    pub fn peak_vram_bytes(&self) -> u64 {
        #[cfg(feature = "webgpu")]
        {
            if let Some(ref gpu) = self.gpu {
                return gpu.peak_vram_bytes();
            }
        }
        0
    }

    #[cfg(feature = "webgpu")]
    fn mark_webgpu_used(&mut self) {
        self.using_webgpu = true;
    }

    pub fn amplitude_at_compact_zero(&self) -> Complex64 {
        let mut vec = vec![Complex64::ONE];
        for site in &self.sites {
            let left = site.dim().0;
            let right = site.dim().2;
            let mut next = vec![Complex64::new(0.0, 0.0); right];
            for a in 0..left.min(vec.len()) {
                for b in 0..right {
                    next[b] += vec[a] * site[[a, 0, b]];
                }
            }
            vec = next;
        }
        vec.first().copied().unwrap_or(Complex64::new(0.0, 0.0))
    }

    pub fn apply_gate(&mut self, gate: &Gate) -> Result<(), EngineError> {
        match gate {
            Gate::H(t)
            | Gate::X(t)
            | Gate::Y(t)
            | Gate::Z(t)
            | Gate::S(t)
            | Gate::T(t)
            | Gate::RX(t, _)
            | Gate::RY(t, _)
            | Gate::RZ(t, _) => {
                self.apply_one_qubit(*t, &unary_matrix(gate));
            }
            Gate::CNOT(c, t) => self.apply_two_qubit_gate(*c, *t, &two_qubit_matrix(gate))?,
            Gate::CZ(c, t) => self.apply_two_qubit_gate(*c, *t, &two_qubit_matrix(gate))?,
            Gate::CCNOT(c1, c2, t) => self.apply_ccnot(*c1, *c2, *t)?,
        }
        Ok(())
    }

    fn apply_one_qubit(&mut self, qubit: usize, u: &[[Complex64; 2]; 2]) {
        #[cfg(feature = "webgpu")]
        if let Some(ref gpu) = self.gpu {
            let site = &mut self.sites[qubit];
            if gpu.apply_one_qubit(site, u).is_ok() {
                self.mark_webgpu_used();
                return;
            }
        }

        let site = &mut self.sites[qubit];
        let (left, _, right) = site.dim();
        let mut updated = Array3::<Complex64>::zeros((left, 2, right));
        for a in 0..left {
            for b in 0..right {
                let v0 = site[[a, 0, b]];
                let v1 = site[[a, 1, b]];
                updated[[a, 0, b]] = u[0][0] * v0 + u[0][1] * v1;
                updated[[a, 1, b]] = u[1][0] * v0 + u[1][1] * v1;
            }
        }
        *site = updated;
    }

    fn apply_two_qubit_gate(
        &mut self,
        ctrl: usize,
        tgt: usize,
        u: &Mat4,
    ) -> Result<(), EngineError> {
        let (c, t) = if ctrl < tgt { (ctrl, tgt) } else { (tgt, ctrl) };
        if c + 1 != t {
            for pos in c..t - 1 {
                self.apply_swap_positions(pos, pos + 1)?;
            }
        }
        self.apply_two_qubit_at_positions(c, c + 1, u, ctrl < tgt)
    }

    fn apply_two_qubit_at_positions(
        &mut self,
        p: usize,
        q: usize,
        u: &Mat4,
        ctrl_is_first: bool,
    ) -> Result<(), EngineError> {
        debug_assert_eq!(q, p + 1);
        let left = self.sites[p].clone();
        let right = self.sites[q].clone();
        let dl = left.dim().0;
        let dr = right.dim().2;
        let theta = self.merge_two_site_tensors(&left, &right)?;

        let mut updated = Array4::<Complex64>::zeros((dl, 2, 2, dr));
        for a in 0..dl {
            for b in 0..dr {
                for s_out in 0..2 {
                    for t_out in 0..2 {
                        let mut sum = Complex64::new(0.0, 0.0);
                        for s_in in 0..2 {
                            for t_in in 0..2 {
                                let (c_out, t_out_i) = if ctrl_is_first {
                                    (s_out, t_out)
                                } else {
                                    (t_out, s_out)
                                };
                                let (c_in, t_in_i) = if ctrl_is_first {
                                    (s_in, t_in)
                                } else {
                                    (t_in, s_in)
                                };
                                sum += u[c_out][t_out_i][c_in][t_in_i] * theta[[a, s_in, t_in, b]];
                            }
                        }
                        updated[[a, s_out, t_out, b]] = sum;
                    }
                }
            }
        }

        let (site_l, site_r) = svd_split_two_site(updated, self.max_bond_dim)?;
        self.sites[p] = site_l;
        self.sites[q] = site_r;
        Ok(())
    }

    fn merge_two_site_tensors(
        &mut self,
        left: &Array3<Complex64>,
        right: &Array3<Complex64>,
    ) -> Result<Array4<Complex64>, EngineError> {
        #[cfg(feature = "webgpu")]
        if let Some(ref gpu) = self.gpu {
            if let Ok(theta) = gpu.merge_two_site(left, right) {
                self.mark_webgpu_used();
                return Ok(theta);
            }
        }

        let dl = left.dim().0;
        let dr = right.dim().2;
        let bond = left.dim().2.min(right.dim().0);
        let mut theta = Array4::<Complex64>::zeros((dl, 2, 2, dr));
        for a in 0..dl {
            for b in 0..dr {
                for s in 0..2 {
                    for t in 0..2 {
                        let mut sum = Complex64::new(0.0, 0.0);
                        for g in 0..bond {
                            sum += left[[a, s, g]] * right[[g, t, b]];
                        }
                        theta[[a, s, t, b]] = sum;
                    }
                }
            }
        }
        Ok(theta)
    }

    fn apply_swap_positions(&mut self, p: usize, q: usize) -> Result<(), EngineError> {
        self.apply_two_qubit_at_positions(p, q, &swap_matrix(), true)
    }

    fn apply_ccnot(&mut self, c1: usize, c2: usize, t: usize) -> Result<(), EngineError> {
        let tdg = tdg_matrix();
        self.apply_one_qubit(t, &unary_matrix(&Gate::H(t)));
        self.apply_two_qubit_gate(c2, t, &super::gates::cnot_matrix())?;
        self.apply_one_qubit(t, &tdg);
        self.apply_two_qubit_gate(c1, t, &super::gates::cnot_matrix())?;
        self.apply_one_qubit(t, &unary_matrix(&Gate::T(t)));
        self.apply_two_qubit_gate(c2, t, &super::gates::cnot_matrix())?;
        self.apply_one_qubit(t, &tdg);
        self.apply_two_qubit_gate(c1, t, &super::gates::cnot_matrix())?;
        self.apply_one_qubit(t, &unary_matrix(&Gate::T(t)));
        self.apply_one_qubit(t, &unary_matrix(&Gate::H(t)));
        Ok(())
    }

    pub fn site_amplitudes(&self, target: usize) -> (Complex64, Complex64) {
        let left_env = self.left_environment(target);
        let right_env = self.right_environment(target);
        let site = &self.sites[target];
        let mut a0 = Complex64::new(0.0, 0.0);
        let mut a1 = Complex64::new(0.0, 0.0);
        for (l, &le) in left_env.iter().enumerate() {
            for r in 0..site.dim().2 {
                let re = right_env.get(r).copied().unwrap_or(Complex64::ONE);
                a0 += le * site[[l, 0, r]] * re;
                a1 += le * site[[l, 1, r]] * re;
            }
        }
        (a0, a1)
    }

    pub fn control_probability(&self, control: usize) -> f64 {
        let (_, a1) = self.site_amplitudes(control);
        a1.re * a1.re + a1.im * a1.im
    }

    fn left_environment(&self, target: usize) -> Vec<Complex64> {
        if target == 0 {
            return vec![Complex64::ONE];
        }
        let mut vec = vec![Complex64::ONE];
        for site in self.sites.iter().take(target) {
            let left = site.dim().0;
            let right = site.dim().2;
            let mut next = vec![Complex64::new(0.0, 0.0); right];
            for a in 0..left.min(vec.len()) {
                for s in 0..2 {
                    for b in 0..right {
                        next[b] += vec[a] * site[[a, s, b]];
                    }
                }
            }
            vec = next;
        }
        vec
    }

    fn right_environment(&self, target: usize) -> Vec<Complex64> {
        if target + 1 >= self.qubit_count {
            return vec![Complex64::ONE];
        }
        let mut vec = vec![Complex64::ONE];
        for site in self.sites.iter().rev().take(self.qubit_count - target - 1) {
            let left = site.dim().0;
            let right = site.dim().2;
            let mut next = vec![Complex64::new(0.0, 0.0); left];
            for b in 0..right.min(vec.len()) {
                for s in 0..2 {
                    for a in 0..left {
                        next[a] += vec[b] * site[[a, s, b]];
                    }
                }
            }
            vec = next;
        }
        vec
    }

    pub fn to_dense_state(&self) -> Array1<Complex64> {
        let dim = 1usize << self.qubit_count;
        let mut state = Array1::zeros(dim);
        self.fill_dense_recursive(0, 0, 0, Complex64::ONE, &mut state);
        state
    }

    fn fill_dense_recursive(
        &self,
        site_idx: usize,
        bond_in: usize,
        basis_prefix: usize,
        coeff: Complex64,
        state: &mut Array1<Complex64>,
    ) {
        if site_idx == self.qubit_count {
            state[basis_prefix] += coeff;
            return;
        }
        let site = &self.sites[site_idx];
        if bond_in >= site.dim().0 {
            return;
        }
        for s in 0..2 {
            for b in 0..site.dim().2 {
                let amp = site[[bond_in, s, b]];
                if amp.norm_sqr() < 1e-30 {
                    continue;
                }
                self.fill_dense_recursive(
                    site_idx + 1,
                    b,
                    basis_prefix | (s << site_idx),
                    coeff * amp,
                    state,
                );
            }
        }
    }
}

fn truncated_svd_split(
    mat: DMatrix<Complex64>,
    max_bond: usize,
    left_dim: usize,
    right_dim: usize,
) -> Result<(Array3<Complex64>, Array3<Complex64>), EngineError> {
    let rows = mat.nrows();
    let cols = mat.ncols();
    let svd = mat.svd(true, true);
    let singular = svd.singular_values;
    let rank = singular
        .iter()
        .filter(|&&s| s > 1e-12)
        .count()
        .max(1);
    let chi = rank.min(max_bond).min(rows).min(cols);

    let u = svd.u.ok_or_else(|| EngineError::ExecutionFailed("SVD missing U".into()))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| EngineError::ExecutionFailed("SVD missing V^T".into()))?;

    let mut left = Array3::<Complex64>::zeros((left_dim, 2, chi));
    let mut right = Array3::<Complex64>::zeros((chi, 2, right_dim));

    for m in 0..chi {
        let s = singular[m].sqrt();
        for l in 0..left_dim {
            for p in 0..2 {
                left[[l, p, m]] = u[(l * 2 + p, m)] * s;
            }
        }
        for r in 0..right_dim {
            for t in 0..2 {
                right[[m, t, r]] = v_t[(m, t * right_dim + r)] * s;
            }
        }
    }

    Ok((left, right))
}

fn tdg_matrix() -> [[Complex64; 2]; 2] {
    let z = Complex64::new(0.0, 0.0);
    let factor = Complex64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    );
    [[Complex64::ONE, z], [z, factor]]
}

fn svd_split_two_site(
    theta: Array4<Complex64>,
    max_bond: usize,
) -> Result<(Array3<Complex64>, Array3<Complex64>), EngineError> {
    let (dl, _, _, dr) = theta.dim();
    let rows = dl * 2;
    let cols = 2 * dr;
    let mut mat = DMatrix::<Complex64>::zeros(rows, cols);
    for l in 0..dl {
        for s in 0..2 {
            for t in 0..2 {
                for r in 0..dr {
                    mat[(l * 2 + s, t * dr + r)] = theta[[l, s, t, r]];
                }
            }
        }
    }
    truncated_svd_split(mat, max_bond, dl, dr)
}

#[cfg(feature = "webgpu")]
fn init_gpu_device() -> Option<Arc<GpuMpsDevice>> {
    if tn_backend_from_env() != TnBackend::WebGpu {
        return None;
    }
    match GpuMpsDevice::try_new() {
        Some(device) => {
            eprintln!("wqc-core: WebGPU MPS backend initialized");
            Some(device)
        }
        None => {
            eprintln!("wqc-core: WebGPU adapter not found; using CPU MPS kernels");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Gate;
    use crate::tn::gates::exact_bond_dim;

    #[test]
    fn mps_h_gate_matches_dense_amplitude() {
        let gates = vec![Gate::H(0)];
        let chi = exact_bond_dim(1);
        let mut mps = MpsState::try_new_with_bond(1, chi).expect("mps");
        for g in &gates {
            mps.apply_gate(g).expect("apply");
        }
        let amp = mps.amplitude_at_compact_zero();
        assert!((amp.re - 1.0 / 2.0f64.sqrt()).abs() < 1e-8);
        assert!(amp.im.abs() < 1e-8);
    }

    #[test]
    fn mps_h_ccnot_exact_bond_matches_dense() {
        let gates = vec![Gate::H(0), Gate::CCNOT(0, 1, 2)];
        let chi = exact_bond_dim(3);
        let mut mps = MpsState::try_new_with_bond(3, chi).expect("mps");
        for g in &gates {
            mps.apply_gate(g).expect("apply");
        }
        let dense = mps.to_dense_state();
        let expected = {
            let mut d = crate::tn::dense::DenseTnState::try_new(3).expect("dense");
            for g in &gates {
                d.apply_gate(g);
            }
            d.state[0]
        };
        assert!((dense[0].re - expected.re).abs() < 1e-6);
        assert!((dense[0].im - expected.im).abs() < 1e-6);
    }
}
