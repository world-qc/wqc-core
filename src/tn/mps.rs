//! Bond-truncated matrix-product-state (MPS) tensor-network backend.
//!
//! Memory scales as `O(N · χ²)` with bond dimension `χ` (`WQC_MPS_MAX_BOND_DIM`, default 128).
//! Set `WQC_TN_BACKEND=webgpu` (with `--features webgpu`) to run a 2-site merge
//! on the GPU when that product is at least the side measured on the first
//! webgpu start of this machine. A layer of disjoint adjacent 2-qubit gates is
//! one batched product, and the bond cut is a GEMM range-finder when that
//! product is large. Smaller cuts stay on CPU f64. 1-qubit gates stay on CPU.

#[cfg(feature = "webgpu")]
use std::sync::Arc;

use crate::engine::{EngineError, Gate};
use crate::memory_budget::max_wqc_memory_bytes_from_total;
use nalgebra::DMatrix;
use ndarray::{Array1, Array3, Array4};
use num_complex::Complex64;

use super::gates::{max_bond_dim_from_env, swap_matrix, two_qubit_matrix, unary_matrix, Mat4};

#[cfg(feature = "webgpu")]
use super::engine_status::shared_gpu_device;
#[cfg(feature = "webgpu")]
use super::gpu::GpuMpsDevice;

/// Bytes for `qubit_count` site tensors at bond `max_bond_dim`: `N · χ² · 32`.
pub fn mps_workspace_bytes(qubit_count: usize, max_bond_dim: usize) -> u64 {
    (qubit_count as u64)
        .saturating_mul(max_bond_dim as u64)
        .saturating_mul(max_bond_dim as u64)
        .saturating_mul(32)
}

/// Reject when the MPS workspace exceeds the host budget (total RAM minus reserve).
/// A zero `total_memory` skips the check, matching a host that did not report RAM.
pub fn ensure_mps_workspace_budget(required: u64, total_memory: u64) -> Result<(), EngineError> {
    if total_memory == 0 {
        return Ok(());
    }
    let budget = max_wqc_memory_bytes_from_total(total_memory);
    if required > budget {
        return Err(EngineError::InsufficientMemory {
            required,
            available: budget,
        });
    }
    Ok(())
}

/// MPS executor: one site tensor per qubit wire `[left_bond × 2 × right_bond]`.
/// Product of the two boundary bonds in a time-split overlap. 16×16 is the
/// largest product that stays inside the existing 256-side two-site matrix.
const OVERLAP_ENV_LIMIT: usize = 256;

pub struct MpsState {
    pub sites: Vec<Array3<Complex64>>,
    pub qubit_count: usize,
    pub max_bond_dim: usize,
    /// When set, a split that would drop a singular value above the cutoff
    /// returns an error instead of shortening the bond.
    reject_truncation: bool,
    #[cfg(feature = "webgpu")]
    gpu: Option<Arc<GpuMpsDevice>>,
    #[cfg(feature = "webgpu")]
    resident: Vec<Option<super::gpu::GpuSite>>,
    #[cfg(feature = "webgpu")]
    using_webgpu: bool,
}

/// Two-qubit GPU splits kept during tests. Process-wide.
#[cfg(all(test, feature = "webgpu"))]
pub static GPU_TWO_Q_ACCEPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Two-qubit GPU splits sent back to CPU f64 during tests. Process-wide.
#[cfg(all(test, feature = "webgpu"))]
pub static GPU_TWO_Q_FALLBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl MpsState {
    pub fn try_new(qubit_count: usize) -> Result<Self, EngineError> {
        Self::try_new_with_bond(qubit_count, max_bond_dim_from_env())
    }

    pub fn try_new_with_bond(qubit_count: usize, max_bond_dim: usize) -> Result<Self, EngineError> {
        if qubit_count == 0 {
            return Err(EngineError::InvalidQubitCount(qubit_count));
        }
        if max_bond_dim == 0 {
            return Err(EngineError::ExecutionFailed(
                "MPS max bond dimension must be > 0".into(),
            ));
        }

        let estimated_bytes = mps_workspace_bytes(qubit_count, max_bond_dim);

        if !cfg!(test) {
            let mut sys = sysinfo::System::new_all();
            sys.refresh_memory();
            let total = sys.total_memory();
            ensure_mps_workspace_budget(estimated_bytes, total)?;
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

        #[cfg(feature = "webgpu")]
        Self::log_backend_once();

        Ok(Self {
            sites,
            qubit_count,
            max_bond_dim,
            reject_truncation: false,
            #[cfg(feature = "webgpu")]
            gpu: shared_gpu_device(),
            #[cfg(feature = "webgpu")]
            resident: Vec::new(),
            #[cfg(feature = "webgpu")]
            using_webgpu: false,
        })
    }

    #[cfg(feature = "webgpu")]
    fn log_backend_once() {
        use std::sync::Once;
        static LOG_ONCE: Once = Once::new();
        LOG_ONCE.call_once(|| {
            let status = super::engine_status::tn_engine_status();
            match status.active.as_str() {
                "webgpu" => println!("wqc-core: WebGPU MPS backend active"),
                _ if status.requested == "webgpu" => {
                    if let Some(note) = &status.note {
                        println!("wqc-core: {note}");
                    }
                }
                _ => {}
            }
        });
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

    /// Test hook: force this state onto a caller-owned device.
    #[cfg(all(test, feature = "webgpu"))]
    pub fn try_new_with_gpu(
        qubit_count: usize,
        max_bond_dim: usize,
        gpu: Arc<GpuMpsDevice>,
    ) -> Result<Self, EngineError> {
        let mut state = Self::try_new_with_bond(qubit_count, max_bond_dim)?;
        state.gpu = Some(gpu);
        state.resident.clear();
        Ok(state)
    }

    #[cfg(feature = "webgpu")]
    fn ensure_resident(&mut self) -> bool {
        let Some(gpu) = self.gpu.clone() else {
            return false;
        };
        if self.resident.len() == self.sites.len() && self.resident.iter().all(|s| s.is_some()) {
            return true;
        }
        self.resident = self
            .sites
            .iter()
            .map(|site| Some(gpu.upload_site(site)))
            .collect();
        true
    }

    #[cfg(feature = "webgpu")]
    fn refresh_resident_pair(&mut self, p: usize, q: usize) {
        let Some(gpu) = self.gpu.clone() else {
            return;
        };
        if self.resident.len() != self.sites.len() {
            return;
        }
        self.resident[p] = Some(gpu.upload_site(&self.sites[p]));
        self.resident[q] = Some(gpu.upload_site(&self.sites[q]));
    }

    /// Contract the MPS to a dense statevector (Phase A sampling; `n ≤ 20`).
    pub fn contract_to_statevector(&self) -> Result<Vec<Complex64>, EngineError> {
        let n = self.qubit_count;
        if n > 20 {
            return Err(EngineError::ExecutionFailed(
                "full statevector projection requires qubit_count <= 20".into(),
            ));
        }

        let mut state = vec![Complex64::ONE];

        for (q_idx, site) in self.sites.iter().enumerate() {
            let l = site.dim().0;
            let r = site.dim().2;
            let prefix = 1usize << q_idx;
            if state.len() != l * prefix {
                return Err(EngineError::ExecutionFailed(
                    "MPS bond layout mismatch during statevector projection".into(),
                ));
            }

            let mut next = vec![Complex64::new(0.0, 0.0); r * prefix * 2];
            for a in 0..l {
                for basis_prefix in 0..prefix {
                    let src = a * prefix + basis_prefix;
                    let v = state[src];
                    for phys in 0..2 {
                        for b in 0..r {
                            let dst_basis = basis_prefix + phys * prefix;
                            let dst = b * (prefix * 2) + dst_basis;
                            next[dst] += v * site[[a, phys, b]];
                        }
                    }
                }
            }
            state = next;
        }

        let expected = 1usize << n;
        if state.len() != expected {
            return Err(EngineError::ExecutionFailed(format!(
                "projected statevector length {} != 2^{n}",
                state.len()
            )));
        }
        Ok(state)
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
            Gate::MEASURE(_) | Gate::RESET(_) | Gate::IF(_) => {}
        }
        Ok(())
    }

    /// Apply `U†`. H, X, Y, Z, CNOT, CZ, and CCNOT are their own adjoints.
    /// S, T, and the rotations are the conjugate transpose of the gate matrix.
    fn apply_gate_adjoint(&mut self, gate: &Gate) -> Result<(), EngineError> {
        match gate {
            Gate::MEASURE(_) | Gate::RESET(_) | Gate::IF(_) => {
                return Err(EngineError::ExecutionFailed(
                    "time-split overlap does not apply measurement or reset".into(),
                ));
            }
            Gate::CCNOT(c1, c2, t) => self.apply_ccnot(*c1, *c2, *t)?,
            Gate::CNOT(c, t) => {
                let u = adjoint_mat4(&two_qubit_matrix(gate));
                self.apply_two_qubit_gate(*c, *t, &u)?;
            }
            Gate::CZ(c, t) => {
                let u = adjoint_mat4(&two_qubit_matrix(gate));
                self.apply_two_qubit_gate(*c, *t, &u)?;
            }
            Gate::H(t)
            | Gate::X(t)
            | Gate::Y(t)
            | Gate::Z(t)
            | Gate::S(t)
            | Gate::T(t)
            | Gate::RX(t, _)
            | Gate::RY(t, _)
            | Gate::RZ(t, _) => {
                let u = adjoint_mat2(&unary_matrix(gate));
                self.apply_one_qubit(*t, &u);
            }
        }
        Ok(())
    }

    fn apply_one_qubit(&mut self, qubit: usize, u: &[[Complex64; 2]; 2]) {
        #[cfg(feature = "webgpu")]
        if self.gpu_one_qubit(qubit, u) {
            return;
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
        let (left, right) = if ctrl < tgt { (ctrl, tgt) } else { (tgt, ctrl) };
        // Slide `right` down to `left + 1`, apply, then undo the swaps.
        // Leaving the qubits permuted makes the next gate hit the wrong pair.
        let span = right - left;
        if span > 1 {
            for pos in (left + 1..right).rev() {
                self.apply_swap_positions(pos, pos + 1)?;
            }
        }
        self.apply_two_qubit_at_positions(left, left + 1, u, ctrl < tgt)?;
        if span > 1 {
            for pos in left + 1..right {
                self.apply_swap_positions(pos, pos + 1)?;
            }
        }
        Ok(())
    }

    fn apply_two_qubit_at_positions(
        &mut self,
        p: usize,
        q: usize,
        u: &Mat4,
        ctrl_is_first: bool,
    ) -> Result<(), EngineError> {
        debug_assert_eq!(q, p + 1);
        #[cfg(feature = "webgpu")]
        if self.try_gpu_two_qubit(p, q, u, ctrl_is_first)? {
            return Ok(());
        }
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

        let (site_l, site_r) = self.split_updated(updated)?;
        self.sites[p] = site_l;
        self.sites[q] = site_r;
        #[cfg(feature = "webgpu")]
        self.refresh_resident_pair(p, q);
        Ok(())
    }

    #[cfg(feature = "webgpu")]
    fn gpu_one_qubit(&mut self, qubit: usize, u: &[[Complex64; 2]; 2]) -> bool {
        let _ = (qubit, u);
        // A 1-qubit apply is a 2×2. That is below the GEMM side threshold, so
        // the f64 loop runs even when the backend setting is webgpu.
        false
    }

    #[cfg(feature = "webgpu")]
    fn try_gpu_two_qubit(
        &mut self,
        p: usize,
        q: usize,
        u: &Mat4,
        ctrl_is_first: bool,
    ) -> Result<bool, EngineError> {
        let rows = self.sites[p].dim().0.saturating_mul(2);
        let cols = self.sites[q].dim().2.saturating_mul(2);
        if !super::gpu::svd_uses_gpu(rows.max(cols)) {
            return Ok(false);
        }
        let Some(gpu) = self.gpu.clone() else {
            return Ok(false);
        };
        if !self.ensure_resident() {
            return Ok(false);
        }
        let outcome = {
            let (Some(left), Some(right)) = (self.resident[p].as_ref(), self.resident[q].as_ref())
            else {
                return Ok(false);
            };
            gpu.split_two_qubit(left, right, u, ctrl_is_first, self.max_bond_dim)?
        };
        match outcome {
            super::gpu::SplitOutcome::Accepted(got) => {
                self.sites[p] = got.left;
                self.sites[q] = got.right;
                self.resident[p] = Some(gpu.upload_site(&self.sites[p]));
                self.resident[q] = Some(gpu.upload_site(&self.sites[q]));
                self.mark_webgpu_used();
                #[cfg(test)]
                GPU_TWO_Q_ACCEPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(true)
            }
            super::gpu::SplitOutcome::Fallback => {
                #[cfg(test)]
                GPU_TWO_Q_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(false)
            }
        }
    }

    fn merge_two_site_tensors(
        &mut self,
        left: &Array3<Complex64>,
        right: &Array3<Complex64>,
    ) -> Result<Array4<Complex64>, EngineError> {
        #[cfg(feature = "webgpu")]
        if let Some(gpu) = self.gpu.clone() {
            if let Some(theta) = gpu.try_gpu_merge(left, right) {
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

    /// Apply a layer of adjacent, qubit-disjoint CNOT/CZ gates as one product.
    ///
    /// Pre-state samples stay on the caller. This writes the sites once, after
    /// the batched multiply and the GEMM truncation.
    pub(crate) fn apply_adjacent_disjoint_gates(
        &mut self,
        gates: &[Gate],
    ) -> Result<(), EngineError> {
        if gates.len() < 2 {
            for gate in gates {
                self.apply_gate(gate)?;
            }
            return Ok(());
        }
        let mut jobs = Vec::with_capacity(gates.len());
        for gate in gates {
            let (ctrl, tgt) = match gate {
                Gate::CNOT(c, t) | Gate::CZ(c, t) => (*c, *t),
                _ => {
                    return Err(EngineError::ExecutionFailed(
                        "disjoint layer contains a gate that is not CNOT or CZ".into(),
                    ))
                }
            };
            let (p, q, ctrl_is_first) = if ctrl < tgt {
                (ctrl, tgt, true)
            } else {
                (tgt, ctrl, false)
            };
            jobs.push(LayerJob {
                p,
                q,
                ctrl_is_first,
                u: two_qubit_matrix(gate),
                left: self.sites[p].clone(),
                right: self.sites[q].clone(),
            });
        }
        let thetas = self.merge_layer(&jobs)?;
        for (job, theta) in jobs.iter().zip(thetas) {
            let updated = apply_two_qubit_unitary(theta, &job.u, job.ctrl_is_first);
            let (site_l, site_r) = self.split_updated(updated)?;
            self.sites[job.p] = site_l;
            self.sites[job.q] = site_r;
            #[cfg(feature = "webgpu")]
            self.refresh_resident_pair(job.p, job.q);
        }
        Ok(())
    }

    fn merge_layer(&mut self, jobs: &[LayerJob]) -> Result<Vec<Array4<Complex64>>, EngineError> {
        #[cfg(feature = "webgpu")]
        {
            let shapes: Vec<(usize, usize, usize)> = jobs
                .iter()
                .map(|job| {
                    let dl = job.left.dim().0;
                    let dr = job.right.dim().2;
                    let bond = job.left.dim().2.min(job.right.dim().0);
                    (dl, dr, bond)
                })
                .collect();
            let m = shapes.iter().map(|(dl, _, _)| dl * 2).max().unwrap_or(0);
            let k = shapes.iter().map(|(_, _, bond)| *bond).max().unwrap_or(0);
            let n = shapes.iter().map(|(_, dr, _)| dr * 2).max().unwrap_or(0);
            if m > 0 && k > 0 && n > 0 {
                if let Some(gpu) = self.gpu.clone() {
                    if let Some(packed) = pack_layer_gemm(&gpu, jobs, &shapes, m, k, n) {
                        self.mark_webgpu_used();
                        return Ok(unpack_layer_thetas(&packed, &shapes, m, n));
                    }
                }
            }
        }
        let mut out = Vec::with_capacity(jobs.len());
        for job in jobs {
            out.push(cpu_merge(&job.left, &job.right));
        }
        Ok(out)
    }

    fn split_updated(
        &mut self,
        updated: Array4<Complex64>,
    ) -> Result<(Array3<Complex64>, Array3<Complex64>), EngineError> {
        #[cfg(feature = "webgpu")]
        {
            split_theta(
                self.gpu.as_deref(),
                updated,
                self.max_bond_dim,
                self.reject_truncation,
            )
        }
        #[cfg(not(feature = "webgpu"))]
        {
            split_theta(updated, self.max_bond_dim, self.reject_truncation)
        }
    }

    fn apply_swap_positions(&mut self, p: usize, q: usize) -> Result<(), EngineError> {
        self.apply_two_qubit_at_positions(p, q, &swap_matrix(), true)
    }

    pub(crate) fn apply_unary_at(&mut self, site: usize, gate: &Gate) -> Result<(), EngineError> {
        self.apply_one_qubit(site, &unary_matrix(gate));
        Ok(())
    }

    pub(crate) fn apply_tdg_at(&mut self, site: usize) -> Result<(), EngineError> {
        self.apply_one_qubit(site, &tdg_matrix());
        Ok(())
    }

    pub(crate) fn apply_swap_at(&mut self, site: usize) -> Result<(), EngineError> {
        self.apply_swap_positions(site, site + 1)
    }

    pub(crate) fn apply_pair_at(
        &mut self,
        site: usize,
        gate: &Gate,
        ctrl_on_left: bool,
    ) -> Result<(), EngineError> {
        self.apply_two_qubit_at_positions(site, site + 1, &two_qubit_matrix(gate), ctrl_on_left)
    }

    pub(crate) fn apply_cnot_at(
        &mut self,
        site: usize,
        ctrl_on_left: bool,
    ) -> Result<(), EngineError> {
        self.apply_two_qubit_at_positions(
            site,
            site + 1,
            &super::gates::cnot_matrix(),
            ctrl_on_left,
        )
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

struct LayerJob {
    p: usize,
    q: usize,
    ctrl_is_first: bool,
    u: Mat4,
    left: Array3<Complex64>,
    right: Array3<Complex64>,
}

/// GEMM truncation starts at the same square side the Mac round-trip crossed.
/// Both backends use it, so a CPU miner and a GPU miner cut the same way.
const RSVD_MIN_SIDE: usize = 160;
const RSVD_OVERSAMPLE: usize = 16;

#[cfg(test)]
pub(crate) static RSVD_KEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
pub(crate) static RSVD_FALLBACKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

struct SplitShape {
    rows: usize,
    cols: usize,
    ell: usize,
    dl: usize,
    dr: usize,
    max_bond: usize,
}

fn cpu_merge(left: &Array3<Complex64>, right: &Array3<Complex64>) -> Array4<Complex64> {
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
    theta
}

fn apply_two_qubit_unitary(
    theta: Array4<Complex64>,
    u: &Mat4,
    ctrl_is_first: bool,
) -> Array4<Complex64> {
    let (dl, _, _, dr) = theta.dim();
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
    updated
}

#[cfg(feature = "webgpu")]
fn pack_layer_gemm(
    gpu: &super::gpu::GpuMpsDevice,
    jobs: &[LayerJob],
    shapes: &[(usize, usize, usize)],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<Complex64>> {
    let batch = jobs.len();
    let mut a = vec![Complex64::new(0.0, 0.0); batch * m * k];
    let mut b = vec![Complex64::new(0.0, 0.0); batch * k * n];
    for (bi, job) in jobs.iter().enumerate() {
        let (dl, dr, bond) = shapes[bi];
        for aa in 0..dl {
            for s in 0..2 {
                for g in 0..bond {
                    let row = aa * 2 + s;
                    a[(bi * m + row) * k + g] = job.left[[aa, s, g]];
                }
            }
        }
        for g in 0..bond {
            for t in 0..2 {
                for bb in 0..dr {
                    let col = t * dr + bb;
                    b[(bi * k + g) * n + col] = job.right[[g, t, bb]];
                }
            }
        }
    }
    gpu.try_complex_gemm(batch, m, k, n, &a, &b)
}

fn unpack_layer_thetas(
    packed: &[Complex64],
    shapes: &[(usize, usize, usize)],
    m: usize,
    n: usize,
) -> Vec<Array4<Complex64>> {
    shapes
        .iter()
        .enumerate()
        .map(|(bi, (dl, dr, _))| {
            let mut theta = Array4::<Complex64>::zeros((*dl, 2, 2, *dr));
            for aa in 0..*dl {
                for s in 0..2 {
                    for t in 0..2 {
                        for bb in 0..*dr {
                            let row = aa * 2 + s;
                            let col = t * *dr + bb;
                            theta[[aa, s, t, bb]] = packed[(bi * m + row) * n + col];
                        }
                    }
                }
            }
            theta
        })
        .collect()
}

fn rsvd_worth(rows: usize, cols: usize, ell: usize) -> bool {
    let flops = (rows as u128)
        .saturating_mul(cols as u128)
        .saturating_mul(ell as u128);
    let side = RSVD_MIN_SIDE as u128;
    flops >= side.saturating_mul(side).saturating_mul(side)
}

fn split_theta(
    #[cfg(feature = "webgpu")] gpu: Option<&super::gpu::GpuMpsDevice>,
    theta: Array4<Complex64>,
    max_bond: usize,
    reject_truncation: bool,
) -> Result<(Array3<Complex64>, Array3<Complex64>), EngineError> {
    // An exact half must not take the range-finder. That path keeps χ modes
    // and does not report the weight it dropped.
    if reject_truncation {
        return svd_split_two_site(theta, max_bond, true);
    }
    let (dl, _, _, dr) = theta.dim();
    let rows = dl * 2;
    let cols = 2 * dr;
    let ell = max_bond.saturating_add(RSVD_OVERSAMPLE).min(rows).min(cols);
    if ell == 0 || ell >= rows.min(cols) || !rsvd_worth(rows, cols, ell) {
        return svd_split_two_site(theta, max_bond, false);
    }
    let mut mat = vec![Complex64::new(0.0, 0.0); rows * cols];
    for a in 0..dl {
        for s in 0..2 {
            for t in 0..2 {
                for b in 0..dr {
                    mat[(a * 2 + s) * cols + (t * dr + b)] = theta[[a, s, t, b]];
                }
            }
        }
    }
    let shape = SplitShape {
        rows,
        cols,
        ell,
        dl,
        dr,
        max_bond,
    };
    #[cfg(feature = "webgpu")]
    let split = rsvd_sites(gpu, &mat, &shape);
    #[cfg(not(feature = "webgpu"))]
    let split = rsvd_sites(&mat, &shape);
    if let Some(pair) = split {
        return Ok(pair);
    }
    svd_split_two_site(theta, max_bond, false)
}

fn rsvd_sites(
    #[cfg(feature = "webgpu")] gpu: Option<&super::gpu::GpuMpsDevice>,
    mat: &[Complex64],
    shape: &SplitShape,
) -> Option<(Array3<Complex64>, Array3<Complex64>)> {
    let SplitShape {
        rows,
        cols,
        ell,
        dl,
        dr,
        max_bond,
    } = *shape;
    let omega = rademacher(cols, ell, mix_seed(rows, cols, ell));
    #[cfg(feature = "webgpu")]
    let y0 = matmul(gpu, rows, cols, ell, mat, &omega);
    #[cfg(not(feature = "webgpu"))]
    let y0 = matmul(rows, cols, ell, mat, &omega);
    let ah = adjoint(mat, rows, cols);
    #[cfg(feature = "webgpu")]
    let z = matmul(gpu, cols, rows, ell, &ah, &y0);
    #[cfg(not(feature = "webgpu"))]
    let z = matmul(cols, rows, ell, &ah, &y0);
    #[cfg(feature = "webgpu")]
    let y = matmul(gpu, rows, cols, ell, mat, &z);
    #[cfg(not(feature = "webgpu"))]
    let y = matmul(rows, cols, ell, mat, &z);
    let q = thin_q(&y, rows, ell)?;
    let qh = adjoint(&q, rows, ell);
    #[cfg(feature = "webgpu")]
    let b = matmul(gpu, ell, rows, cols, &qh, mat);
    #[cfg(not(feature = "webgpu"))]
    let b = matmul(ell, rows, cols, &qh, mat);
    let bmat = DMatrix::from_row_slice(ell, cols, &b);
    let svd = bmat.svd(true, true);
    let mut values: Vec<f64> = svd.singular_values.iter().copied().collect();
    // Oversampled modes of an exact gap land around 1e-15. The GPU cutoff
    // band is 1e-10 wide and would treat that dust as an ambiguous cut.
    for s in &mut values {
        if *s < super::svd_guard::SINGULAR_CUTOFF * 1e-2 {
            *s = 0.0;
        }
    }
    let chi = super::svd_guard::retained_chi(&values, max_bond, rows, cols);
    if chi == 0 || super::svd_guard::needs_f64_fallback(&values, chi) {
        #[cfg(test)]
        RSVD_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    #[cfg(test)]
    RSVD_KEPT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut left = Array3::<Complex64>::zeros((dl, 2, chi));
    let mut right = Array3::<Complex64>::zeros((chi, 2, dr));
    for mode in 0..chi {
        let scale = svd.singular_values[mode].sqrt();
        for r in 0..rows {
            let mut acc = Complex64::new(0.0, 0.0);
            for t in 0..ell {
                acc += q[r * ell + t] * u[(t, mode)];
            }
            left[[r / 2, r % 2, mode]] = acc * scale;
        }
        for c in 0..cols {
            right[[mode, c / dr, c % dr]] = v_t[(mode, c)] * scale;
        }
    }
    Some((left, right))
}

fn matmul(
    #[cfg(feature = "webgpu")] gpu: Option<&super::gpu::GpuMpsDevice>,
    m: usize,
    k: usize,
    n: usize,
    a: &[Complex64],
    b: &[Complex64],
) -> Vec<Complex64> {
    #[cfg(feature = "webgpu")]
    if let Some(gpu) = gpu {
        if let Some(c) = gpu.try_complex_gemm(1, m, k, n, a, b) {
            return c;
        }
    }
    let ma = DMatrix::from_row_slice(m, k, a);
    let mb = DMatrix::from_row_slice(k, n, b);
    let mc = &ma * &mb;
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = mc[(i, j)];
        }
    }
    out
}

fn adjoint(a: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); cols * rows];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c].conj();
        }
    }
    out
}

fn thin_q(y: &[Complex64], rows: usize, ell: usize) -> Option<Vec<Complex64>> {
    let qr = DMatrix::from_row_slice(rows, ell, y).qr();
    let q = qr.q();
    let mut out = vec![Complex64::new(0.0, 0.0); rows * ell];
    for r in 0..rows {
        for c in 0..ell {
            out[r * ell + c] = q[(r, c)];
        }
    }
    Some(out)
}

fn rademacher(cols: usize, ell: usize, salt: u64) -> Vec<Complex64> {
    let mut state = salt | 1;
    let scale = 1.0 / (cols as f64).sqrt();
    let mut omega = Vec::with_capacity(cols * ell);
    for _ in 0..cols * ell {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let re = if state & 0x100 == 0 { scale } else { -scale };
        let im = if state & 0x200 == 0 { scale } else { -scale };
        omega.push(Complex64::new(re, im));
    }
    omega
}

fn mix_seed(rows: usize, cols: usize, ell: usize) -> u64 {
    (rows as u64)
        .wrapping_mul(0x9E37)
        .wrapping_add(cols as u64)
        .wrapping_mul(0x9E37)
        .wrapping_add(ell as u64)
}

fn truncated_svd_split(
    mat: DMatrix<Complex64>,
    max_bond: usize,
    left_dim: usize,
    right_dim: usize,
    reject_truncation: bool,
) -> Result<(Array3<Complex64>, Array3<Complex64>), EngineError> {
    let rows = mat.nrows();
    let cols = mat.ncols();
    let svd = mat.svd(true, true);
    let singular = svd.singular_values;
    let values: Vec<f64> = singular.iter().copied().collect();
    let chi = super::svd_guard::retained_chi(&values, max_bond, rows, cols);
    if reject_truncation
        && values
            .iter()
            .skip(chi)
            .any(|s| *s > super::svd_guard::SINGULAR_CUTOFF)
    {
        return Err(EngineError::ExecutionFailed("mps bond exceeded chi".into()));
    }

    let u = svd
        .u
        .ok_or_else(|| EngineError::ExecutionFailed("SVD missing U".into()))?;
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
    reject_truncation: bool,
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
    truncated_svd_split(mat, max_bond, dl, dr, reject_truncation)
}

fn adjoint_mat2(u: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [u[0][0].conj(), u[1][0].conj()],
        [u[0][1].conj(), u[1][1].conj()],
    ]
}

fn adjoint_mat4(u: &Mat4) -> Mat4 {
    let z = Complex64::new(0.0, 0.0);
    let mut out = [[[[z; 2]; 2]; 2]; 2];
    for c_out in 0..2 {
        for t_out in 0..2 {
            for c_in in 0..2 {
                for t_in in 0..2 {
                    out[c_out][t_out][c_in][t_in] = u[c_in][t_in][c_out][t_out].conj();
                }
            }
        }
    }
    out
}

/// Work-report mark: both half sweeps are in the unitary AIR. The overlap is
/// a separate transcript tail (claimed MPS tensors → amplitude), not an AIR
/// row on the half sweeps. Informational on the node: the orchestrator
/// overwrites from `MpsTimeSplit`.
pub const PROOF_SCOPE_TIME_SPLIT_HALVES: &str = "mps_time_split_halves";
/// Work-report mark: only the first half is in the unitary AIR (second half
/// has a gate with no Gate-level adjoint expansion, e.g. MEASURE). Overlap
/// still ships as a separate tail when the path emits one.
pub const PROOF_SCOPE_TIME_SPLIT_FIRST_HALF: &str = "mps_time_split_first_half";

/// `<0|U|0>` as an overlap of two MPS.
///
/// `|ψ⟩ = U_first|0⟩` uses `gates[..split]`. `|φ⟩ = U_second†|0⟩` uses the
/// adjoint of `gates[split..]` in reverse. The scalar is `⟨φ|ψ⟩`. A half whose
/// bond would pass `max_bond`, or an overlap environment wider than
/// [`OVERLAP_ENV_LIMIT`], is an error. Nothing is truncated.
pub fn contract_time_split(
    qubit_count: usize,
    gates: &[Gate],
    split: usize,
    max_bond: usize,
) -> Result<Complex64, EngineError> {
    let (ket, bra) = time_split_boundaries(qubit_count, gates, split, max_bond)?;
    overlap(&ket, &bra)
}

/// Same as [`contract_time_split`], plus an AIR trace for the half sweeps.
///
/// Returns the ket/bra MPS so the caller can emit an [`OverlapSegment`]
/// binding the tensors to the published amplitude. When every gate in the
/// second half has a Gate-level adjoint sequence, both halves are traced and
/// the scope is [`PROOF_SCOPE_TIME_SPLIT_HALVES`]. Otherwise only the first
/// half is traced ([`PROOF_SCOPE_TIME_SPLIT_FIRST_HALF`]).
pub fn contract_time_split_with_trace(
    qubit_count: usize,
    gates: &[Gate],
    split: usize,
    max_bond: usize,
) -> Result<(Complex64, Vec<f64>, &'static str, MpsState, MpsState), EngineError> {
    if split > gates.len() {
        return Err(EngineError::ExecutionFailed(
            "time-split index past the gate list".into(),
        ));
    }
    let mut ket = MpsState::try_new_with_bond(qubit_count, max_bond)?;
    ket.reject_truncation = true;
    let mut trace = super::trace::execute_with_trace(qubit_count, &gates[..split], &mut ket)?;

    let mut bra = MpsState::try_new_with_bond(qubit_count, max_bond)?;
    bra.reject_truncation = true;
    let scope = if let Some(bra_gates) = bra_unitary_gates(&gates[split..]) {
        let bra_trace = super::trace::execute_with_trace(qubit_count, &bra_gates, &mut bra)?;
        trace.extend(bra_trace);
        PROOF_SCOPE_TIME_SPLIT_HALVES
    } else {
        apply_second_half_adjoint(&mut bra, &gates[split..])?;
        PROOF_SCOPE_TIME_SPLIT_FIRST_HALF
    };
    let amp = overlap(&ket, &bra)?;
    Ok((amp, trace, scope, ket, bra))
}

/// Build an overlap STARK witness from the time-split ket/bra MPS and snapped amplitude.
pub fn build_overlap_segment(
    ket: &MpsState,
    bra: &MpsState,
    amplitude: &crate::engine::ComplexResult,
) -> Result<wqc_stark_engine::OverlapSegment, EngineError> {
    use wqc_stark_engine::{
        calculate_amplitude_digest, OverlapSegment, OVERLAP_ENV_LIMIT, OVERLAP_MAX_BOND,
    };

    if ket.qubit_count != bra.qubit_count {
        return Err(EngineError::MismatchedRegister);
    }
    let n = ket.qubit_count;
    let mut ket_sites = Vec::with_capacity(n);
    let mut bra_sites = Vec::with_capacity(n);
    for site in 0..n {
        let kt = &ket.sites[site];
        let bt = &bra.sites[site];
        let (k_left, k_phys, k_right) = kt.dim();
        let (b_left, b_phys, b_right) = bt.dim();
        if k_phys != 2 || b_phys != 2 {
            return Err(EngineError::ExecutionFailed(
                "mps site physical dim must be 2".into(),
            ));
        }
        if k_left > OVERLAP_MAX_BOND
            || k_right > OVERLAP_MAX_BOND
            || b_left > OVERLAP_MAX_BOND
            || b_right > OVERLAP_MAX_BOND
        {
            return Err(EngineError::ExecutionFailed(
                "mps overlap site bond exceeds STARK witness limit 16".into(),
            ));
        }
        if k_right.saturating_mul(b_right) > OVERLAP_ENV_LIMIT {
            return Err(EngineError::ExecutionFailed(
                "mps overlap environment exceeds 256".into(),
            ));
        }
        ket_sites.push(site_to_overlap_tensor(kt));
        bra_sites.push(site_to_overlap_tensor(bt));
    }
    Ok(OverlapSegment {
        qubit_count: n as u32,
        amplitude_re: amplitude.real,
        amplitude_im: amplitude.imag,
        amplitude_digest: calculate_amplitude_digest(amplitude.real, amplitude.imag),
        ket_sites,
        bra_sites,
    })
}

fn site_to_overlap_tensor(site: &Array3<Complex64>) -> wqc_stark_engine::OverlapSiteTensor {
    let (left, phys, right) = site.dim();
    let mut values = Vec::with_capacity(left * phys * right);
    for a in 0..left {
        for s in 0..phys {
            for b in 0..right {
                let c = site[[a, s, b]];
                values.push((c.re, c.im));
            }
        }
    }
    wqc_stark_engine::OverlapSiteTensor {
        left_dim: left as u32,
        right_dim: right as u32,
        values,
    }
}

fn time_split_boundaries(
    qubit_count: usize,
    gates: &[Gate],
    split: usize,
    max_bond: usize,
) -> Result<(MpsState, MpsState), EngineError> {
    if split > gates.len() {
        return Err(EngineError::ExecutionFailed(
            "time-split index past the gate list".into(),
        ));
    }
    let mut ket = MpsState::try_new_with_bond(qubit_count, max_bond)?;
    ket.reject_truncation = true;
    for gate in &gates[..split] {
        ket.apply_gate(gate)?;
    }
    let mut bra = MpsState::try_new_with_bond(qubit_count, max_bond)?;
    bra.reject_truncation = true;
    if let Some(bra_gates) = bra_unitary_gates(&gates[split..]) {
        for gate in &bra_gates {
            bra.apply_gate(gate)?;
        }
    } else {
        apply_second_half_adjoint(&mut bra, &gates[split..])?;
    }
    Ok((ket, bra))
}

fn apply_second_half_adjoint(bra: &mut MpsState, second_half: &[Gate]) -> Result<(), EngineError> {
    for gate in second_half.iter().rev() {
        bra.apply_gate_adjoint(gate)?;
    }
    Ok(())
}

/// Adjoint of `second_half` in reverse, as Gate ops the AIR already knows.
fn bra_unitary_gates(second_half: &[Gate]) -> Option<Vec<Gate>> {
    let mut out = Vec::new();
    for gate in second_half.iter().rev() {
        out.extend(adjoint_gate_sequence(gate)?);
    }
    Some(out)
}

fn adjoint_gate_sequence(gate: &Gate) -> Option<Vec<Gate>> {
    match gate {
        Gate::H(t) => Some(vec![Gate::H(*t)]),
        Gate::X(t) => Some(vec![Gate::X(*t)]),
        Gate::Y(t) => Some(vec![Gate::Y(*t)]),
        Gate::Z(t) => Some(vec![Gate::Z(*t)]),
        Gate::CNOT(c, t) => Some(vec![Gate::CNOT(*c, *t)]),
        Gate::CZ(c, t) => Some(vec![Gate::CZ(*c, *t)]),
        Gate::CCNOT(c1, c2, t) => Some(vec![Gate::CCNOT(*c1, *c2, *t)]),
        Gate::RX(t, th) => Some(vec![Gate::RX(*t, -*th)]),
        Gate::RY(t, th) => Some(vec![Gate::RY(*t, -*th)]),
        Gate::RZ(t, th) => Some(vec![Gate::RZ(*t, -*th)]),
        // S† = S·Z. T† = T³·Z. No dagger variants on Gate.
        Gate::S(t) => Some(vec![Gate::S(*t), Gate::Z(*t)]),
        Gate::T(t) => Some(vec![Gate::T(*t), Gate::T(*t), Gate::T(*t), Gate::Z(*t)]),
        Gate::MEASURE(_) | Gate::RESET(_) | Gate::IF(_) => None,
    }
}

fn overlap(ket: &MpsState, bra: &MpsState) -> Result<Complex64, EngineError> {
    if ket.qubit_count != bra.qubit_count {
        return Err(EngineError::MismatchedRegister);
    }
    let mut env = vec![Complex64::ONE];
    let mut ket_left = 1usize;
    let mut bra_left = 1usize;
    for site in 0..ket.qubit_count {
        let kt = &ket.sites[site];
        let bt = &bra.sites[site];
        let (k_left, _, k_right) = kt.dim();
        let (b_left, phys, bra_right) = bt.dim();
        if k_left != ket_left || b_left != bra_left || phys != 2 {
            return Err(EngineError::ExecutionFailed(
                "mps site shape does not match the overlap".into(),
            ));
        }
        let env_dim = k_right.saturating_mul(bra_right);
        if env_dim > OVERLAP_ENV_LIMIT {
            return Err(EngineError::ExecutionFailed(
                "mps overlap environment exceeds 256".into(),
            ));
        }
        let mut next = vec![Complex64::new(0.0, 0.0); env_dim];
        for a in 0..k_left {
            for b in 0..b_left {
                let coeff = env[a * bra_left + b];
                if coeff.norm_sqr() == 0.0 {
                    continue;
                }
                for s in 0..2 {
                    for a_right in 0..k_right {
                        let kv = coeff * kt[[a, s, a_right]];
                        if kv.norm_sqr() == 0.0 {
                            continue;
                        }
                        for b_out in 0..bra_right {
                            next[a_right * bra_right + b_out] += kv * bt[[b, s, b_out]].conj();
                        }
                    }
                }
            }
        }
        env = next;
        ket_left = k_right;
        bra_left = bra_right;
    }
    if env.len() != 1 {
        return Err(EngineError::ExecutionFailed(
            "mps overlap left an open bond".into(),
        ));
    }
    Ok(env[0])
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

    #[test]
    fn nonadjacent_two_qubit_gates_match_dense_statevector() {
        // Gates that are not neighbors on the MPS path. The zero amplitude
        // alone stays 1/sqrt(2) even if a swap is applied to the wrong pair.
        let gates = vec![
            Gate::H(0),
            Gate::H(1),
            Gate::CNOT(0, 2),
            Gate::CZ(3, 1),
            Gate::CNOT(2, 0),
        ];
        let chi = exact_bond_dim(4);
        let mut mps = MpsState::try_new_with_bond(4, chi).expect("mps");
        for g in &gates {
            mps.apply_gate(g).expect("apply");
        }
        let mut dense = crate::tn::dense::DenseTnState::try_new(4).expect("dense");
        for g in &gates {
            dense.apply_gate(g);
        }
        let got = mps.to_dense_state();
        assert_eq!(got.len(), dense.state.len());
        for (i, (x, y)) in got.iter().zip(dense.state.iter()).enumerate() {
            assert!((x - y).norm() < 1e-8, "amp[{i}] mps={x} dense={y}");
        }
    }

    fn ry_cnot_brickwork(n: usize, depth: usize) -> (Vec<Gate>, usize) {
        let mut gates = Vec::new();
        let mut split = 0usize;
        for layer in 0..depth {
            if layer == depth / 2 {
                split = gates.len();
            }
            for q in 0..n {
                gates.push(Gate::RY(q, 0.17 + 0.01 * (q + layer) as f64));
            }
            let mut q = layer % 2;
            while q + 1 < n {
                gates.push(Gate::CNOT(q, q + 1));
                q += 2;
            }
        }
        (gates, split)
    }

    #[test]
    fn time_split_hadamard_matches_dense() {
        let gates = vec![Gate::H(0), Gate::S(0), Gate::RZ(0, 0.4)];
        let got = contract_time_split(1, &gates, 2, 128).expect("split");
        let mut dense = crate::tn::dense::DenseTnState::try_new(1).expect("dense");
        for g in &gates {
            dense.apply_gate(g);
        }
        assert!(
            (got - dense.state[0]).norm() < 1e-8,
            "{got} vs {}",
            dense.state[0]
        );
    }

    #[test]
    fn time_split_brickwork_matches_dense() {
        // 12 qubits caps the middle cut at 2^6 = 64, so one χ=128 sweep is
        // already exact. 16 qubits leaves 8 sites on each side of that cut.
        // CNOT brickwork on |+⟩ stays a product state, so each layer also
        // rotates. Eight layers a side keep the bond at 16. The full depth
        // needs 256 and must not be truncated down to 128.
        let n = 16usize;
        let (gates, split) = ry_cnot_brickwork(n, 16);
        let got = contract_time_split(n, &gates, split, 128).expect("split");
        let mut dense = crate::tn::dense::DenseTnState::try_new(n).expect("dense");
        for g in &gates {
            dense.apply_gate(g);
        }
        assert!(
            (got - dense.state[0]).norm() < 1e-8,
            "{got} vs {}",
            dense.state[0]
        );
        let err = contract_time_split(n, &gates, gates.len(), 128);
        assert!(err.is_err(), "one sweep of bond 256 must not truncate");
    }

    #[test]
    fn time_split_proof_scope_marks_halves_for_cnot() {
        // Keep this tiny: execute_with_trace on a 16q depth-16 half is minutes
        // in debug. The dense match above covers the amplitude path.
        let gates = vec![Gate::H(0), Gate::CNOT(0, 1), Gate::RY(1, 0.3)];
        let (_, _, scope, _, _) =
            contract_time_split_with_trace(2, &gates, 2, 128).expect("traced");
        assert_eq!(scope, PROOF_SCOPE_TIME_SPLIT_HALVES);
    }

    #[test]
    fn time_split_with_trace_matches_plain_on_wide_shallow() {
        // n>16 takes the disjoint-layer path inside execute_with_trace. The
        // plain path does not. Amplitudes must still agree.
        let n = 18usize;
        let (gates, split) = ry_cnot_brickwork(n, 4);
        let plain = contract_time_split(n, &gates, split, 128).expect("plain");
        let (traced, _, scope, _, _) =
            contract_time_split_with_trace(n, &gates, split, 128).expect("traced");
        assert_eq!(scope, PROOF_SCOPE_TIME_SPLIT_HALVES);
        assert!(
            (plain - traced).norm() < 1e-8,
            "plain={plain} traced={traced}"
        );
    }

    #[test]
    fn overlap_rejects_environment_above_256() {
        let ket = forced_bond_mps(3, 17);
        let bra = forced_bond_mps(3, 17);
        let err = overlap(&ket, &bra).expect_err("17×17 environment");
        assert!(err.to_string().contains("256"), "unexpected error: {err}");
    }

    fn forced_bond_mps(n: usize, bond: usize) -> MpsState {
        let mut m = MpsState::try_new_with_bond(n, bond.max(32)).expect("mps");
        m.reject_truncation = true;
        for i in 0..n {
            let left = if i == 0 { 1 } else { bond };
            let right = if i + 1 == n { 1 } else { bond };
            let mut t = Array3::zeros((left, 2, right));
            let scale = Complex64::new(1.0 / (left as f64).sqrt(), 0.0);
            for a in 0..left {
                t[[a, 0, a.min(right - 1)]] = scale;
            }
            m.sites[i] = t;
        }
        m
    }

    #[test]
    fn long_range_on_width16_half_rejects_at_chi16() {
        // Final cutwidth 16 plus a far CNOT. Swaps along the path push the
        // bond past 16; reject_truncation must error instead of truncating.
        let n = 16usize;
        let (mut gates, _) = ry_cnot_brickwork(n, 8);
        gates.push(Gate::CNOT(0, n - 1));
        let err = contract_time_split(n, &gates, gates.len(), 16);
        assert!(
            err.is_err(),
            "swap along a width-16 cut must not truncate under chi 16"
        );
    }

    #[test]
    fn long_range_on_width8_half_matches_dense_at_chi16() {
        // Depth 6 is final width 8. Chi 16 leaves room for the long-range
        // swaps, which is the margin TimeSplitPlan keeps.
        let n = 16usize;
        let (mut gates, _) = ry_cnot_brickwork(n, 6);
        gates.push(Gate::CNOT(0, n - 1));
        let got = contract_time_split(n, &gates, gates.len(), 16).expect("swap headroom");
        let mut dense = crate::tn::dense::DenseTnState::try_new(n).expect("dense");
        for g in &gates {
            dense.apply_gate(g);
        }
        assert!(
            (got - dense.state[0]).norm() < 1e-8,
            "{got} vs {}",
            dense.state[0]
        );
    }

    #[test]
    fn time_split_with_s_and_t_keeps_halves_proof_scope() {
        let gates = vec![Gate::H(0), Gate::S(0), Gate::T(0)];
        let plain = contract_time_split(1, &gates, 1, 128).expect("plain");
        let (traced, _, scope, _, _) =
            contract_time_split_with_trace(1, &gates, 1, 128).expect("traced");
        assert_eq!(scope, PROOF_SCOPE_TIME_SPLIT_HALVES);
        assert!((plain - traced).norm() < 1e-8, "{plain} vs {traced}");
        let mut dense = crate::tn::dense::DenseTnState::try_new(1).expect("dense");
        for g in &gates {
            dense.apply_gate(g);
        }
        assert!((plain - dense.state[0]).norm() < 1e-8);
    }

    #[test]
    fn overlap_segment_from_time_split_verifies_and_rejects_tamper() {
        use wqc_stark_engine::verify_overlap_segment;

        let gates = vec![Gate::H(0), Gate::S(0), Gate::RZ(0, 0.4)];
        let (amp, _, _, ket, bra) =
            contract_time_split_with_trace(1, &gates, 2, 16).expect("traced");
        let snapped = crate::engine::snap_complex_result(crate::engine::ComplexResult {
            real: amp.re,
            imag: amp.im,
        });
        let segment = build_overlap_segment(&ket, &bra, &snapped).expect("segment");
        assert!(verify_overlap_segment(&segment));

        let mut bad = segment.clone();
        bad.amplitude_re = 0.0;
        bad.amplitude_digest =
            wqc_stark_engine::calculate_amplitude_digest(bad.amplitude_re, bad.amplitude_im);
        assert!(!verify_overlap_segment(&bad));

        let cnot = vec![Gate::H(0), Gate::CNOT(0, 1), Gate::RY(1, 0.25)];
        let (amp2, _, _, ket2, bra2) =
            contract_time_split_with_trace(2, &cnot, 2, 16).expect("cnot");
        let snapped2 = crate::engine::snap_complex_result(crate::engine::ComplexResult {
            real: amp2.re,
            imag: amp2.im,
        });
        let seg2 = build_overlap_segment(&ket2, &bra2, &snapped2).expect("cnot segment");
        assert!(verify_overlap_segment(&seg2));
    }

    #[test]
    fn overlap_segment_brickwork_nontrivial_amplitude_verifies() {
        // Regression for M31 1e-4 snap vs 1e-5 eps: honest RY+CNOT splits with
        // O(1) amplitude must verify. Depth-6 midpoint on 4–12q is enough.
        use wqc_stark_engine::verify_overlap_segment;

        for n in [4usize, 8, 12] {
            let (gates, split) = ry_cnot_brickwork(n, 6);
            assert!(split > 0 && split < gates.len(), "n={n} split={split}");
            let (amp, _, _, ket, bra) =
                contract_time_split_with_trace(n, &gates, split, 16).expect("traced");
            assert!(
                amp.norm() > 1e-3,
                "n={n}: need nontrivial amplitude, got {amp}"
            );
            let snapped = crate::engine::snap_complex_result(crate::engine::ComplexResult {
                real: amp.re,
                imag: amp.im,
            });
            let segment = build_overlap_segment(&ket, &bra, &snapped).expect("segment");
            assert!(
                verify_overlap_segment(&segment),
                "n={n} amp=({}, {})",
                snapped.real,
                snapped.imag
            );
        }
    }

    #[test]
    fn build_overlap_segment_rejects_env_above_256() {
        let ket = forced_bond_mps(3, 17);
        let bra = forced_bond_mps(3, 17);
        let zero = crate::engine::ComplexResult {
            real: 0.0,
            imag: 0.0,
        };
        let err = build_overlap_segment(&ket, &bra, &zero).expect_err("bond 17");
        assert!(
            err.to_string().contains("16") || err.to_string().contains("256"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn disjoint_adjacent_layer_matches_sequential() {
        let prep = [Gate::H(0), Gate::H(2), Gate::RY(1, 0.3), Gate::RY(3, 0.7)];
        let layer = [Gate::CNOT(0, 1), Gate::CZ(3, 2)];
        let chi = exact_bond_dim(4);
        let mut seq = MpsState::try_new_with_bond(4, chi).expect("mps");
        let mut batched = MpsState::try_new_with_bond(4, chi).expect("mps");
        for g in prep.iter().chain(layer.iter()) {
            seq.apply_gate(g).expect("apply");
        }
        for g in &prep {
            batched.apply_gate(g).expect("apply");
        }
        batched
            .apply_adjacent_disjoint_gates(&layer)
            .expect("layer");
        let a = seq.to_dense_state();
        let b = batched.to_dense_state();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).norm() < 1e-8, "{x} vs {y}");
        }
    }

    #[test]
    fn rsvd_recovers_gapped_rank4() {
        let rows = 32;
        let cols = 24;
        let ell = 8;
        let dl = 16;
        let dr = 12;
        let sigmas = [8.0, 4.0, 2.0, 1.0];
        let u_raw = DMatrix::from_fn(rows, 4, |r, c| {
            Complex64::new(((r * 3 + c * 5) as f64).sin(), ((r + c) as f64 * 0.3).cos())
        });
        let v_raw = DMatrix::from_fn(cols, 4, |r, c| {
            Complex64::new(
                ((r * 7 + c * 11) as f64).cos(),
                ((r * 2 + c) as f64 * 0.2).sin(),
            )
        });
        let uq = u_raw.qr().q();
        let vq = v_raw.qr().q();
        let mut mat = vec![Complex64::new(0.0, 0.0); rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = Complex64::new(0.0, 0.0);
                for mode in 0..4 {
                    acc += uq[(r, mode)] * sigmas[mode] * vq[(c, mode)].conj();
                }
                mat[r * cols + c] = acc;
            }
        }
        let shape = SplitShape {
            rows,
            cols,
            ell,
            dl,
            dr,
            max_bond: 4,
        };
        #[cfg(feature = "webgpu")]
        let split = rsvd_sites(None, &mat, &shape);
        #[cfg(not(feature = "webgpu"))]
        let split = rsvd_sites(&mat, &shape);
        let (left, right) = split.expect("rsvd");
        assert_eq!(left.dim().2, 4);
        let mut err = 0.0f64;
        for a in 0..dl {
            for s in 0..2 {
                for t in 0..2 {
                    for b in 0..dr {
                        let mut sum = Complex64::new(0.0, 0.0);
                        for g in 0..4 {
                            sum += left[[a, s, g]] * right[[g, t, b]];
                        }
                        let got = mat[(a * 2 + s) * cols + (t * dr + b)];
                        err = err.max((sum - got).norm());
                    }
                }
            }
        }
        assert!(err < 1e-6, "rebuild err {err}");
    }

    #[test]
    fn mps_accepts_registers_above_40_qubits() {
        assert!(MpsState::try_new_with_bond(0, 8).is_err());
        assert!(MpsState::try_new_with_bond(41, 8).is_ok());
        assert!(MpsState::try_new_with_bond(100, 128).is_ok());
    }

    #[test]
    fn statevector_projection_still_rejects_above_20() {
        let mps = MpsState::try_new_with_bond(21, 4).expect("mps");
        let err = mps.contract_to_statevector().expect_err("21q statevector");
        assert!(err.to_string().contains("qubit_count <= 20"));
    }

    #[test]
    fn mps_workspace_over_host_budget_is_insufficient_memory() {
        let required = mps_workspace_bytes(100, 128);
        assert_eq!(required, 52_428_800);
        let total = 8 * 1024 * 1024 * 1024;
        let budget = max_wqc_memory_bytes_from_total(total);
        assert!(required < budget);
        assert!(ensure_mps_workspace_budget(required, total).is_ok());
        let err = ensure_mps_workspace_budget(budget + 1, total).expect_err("over budget");
        match err {
            EngineError::InsufficientMemory {
                required,
                available,
            } => {
                assert_eq!(required, budget + 1);
                assert_eq!(available, budget);
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
