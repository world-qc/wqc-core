//! Tensor-network contraction entry point for slice execution.

use crate::engine::{ComplexResult, EngineError, Gate, SliceAssignment};

use super::boundary::BoundaryConditions;
use super::mps::MpsState;
use super::trace;

/// Contract a pruned gate list via bond-truncated MPS; returns scalar + STARK trace.
pub fn contract_slice(
    qubit_count: usize,
    gates: &[Gate],
    assignments: &[SliceAssignment],
    original_qubit_count: usize,
    state: &mut MpsState,
) -> Result<(ComplexResult, Vec<f64>), EngineError> {
    let boundary = BoundaryConditions::from_assignments(assignments)?;
    boundary.verify_compact_register(original_qubit_count, qubit_count)?;

    let trace = trace::execute_with_trace(qubit_count, gates, state)?;
    let amp = state.amplitude_at_compact_zero();

    Ok((
        ComplexResult {
            real: amp.re,
            imag: amp.im,
        },
        trace,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tn::gates::exact_bond_dim;

    #[test]
    fn mps_h_ccnot_exact_bond_produces_expected_amplitude() {
        let gates = vec![Gate::H(0), Gate::CCNOT(0, 1, 2)];
        let chi = exact_bond_dim(3);
        let mut mps = MpsState::try_new_with_bond(3, chi).expect("mps");
        let (result, _) = contract_slice(3, &gates, &[], 3, &mut mps).expect("contract");
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        assert!((result.real - inv_sqrt2).abs() < 1e-5);
        assert!(result.imag.abs() < 1e-5);
    }

    #[test]
    fn boundary_assignments_validate_compact_register() {
        let gates = vec![Gate::H(0)];
        let assignments = vec![SliceAssignment {
            edge_id: "e_1".into(),
            value: 1,
        }];
        let chi = exact_bond_dim(2);
        let mut state = MpsState::try_new_with_bond(2, chi).expect("allocate");

        contract_slice(2, &gates, &assignments, 3, &mut state).expect("2 = 3-1");
        let mut bad = MpsState::try_new_with_bond(3, chi).expect("allocate");
        assert!(contract_slice(3, &gates, &assignments, 3, &mut bad).is_err());
    }

    /// Wall clock of one χ=128 contraction. Not a pass/fail speed gate.
    ///
    /// ```text
    /// cargo test --release --features webgpu bond128_contraction_wall_time -- --ignored --nocapture
    /// WQC_TN_BACKEND=webgpu cargo test --release --features webgpu bond128_contraction_wall_time -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "manual CPU vs WebGPU wall clock"]
    fn bond128_contraction_wall_time() {
        // n>16 skips the dense statevector sample inside the trace. At n<=16 that
        // sample dominates once χ grows, and it is identical on both backends.
        let n = 20usize;
        let chi = 128usize;
        let layers = 8usize;
        let gates = brickwork(n, layers);
        let two_q = gates.iter().filter(|g| matches!(g, Gate::CNOT(..))).count();
        eprintln!(
            "circuit qubits={n} chi={chi} layers={layers} gates={} two_qubit={two_q}",
            gates.len()
        );
        let mut kept = Vec::new();
        for pass in 0..3 {
            #[cfg(feature = "webgpu")]
            {
                crate::tn::mps::GPU_TWO_Q_ACCEPTS.store(0, std::sync::atomic::Ordering::Relaxed);
                crate::tn::mps::GPU_TWO_Q_FALLBACKS.store(0, std::sync::atomic::Ordering::Relaxed);
            }
            let mut state = MpsState::try_new_with_bond(n, chi).expect("mps");
            let start = std::time::Instant::now();
            let (result, trace) = contract_slice(n, &gates, &[], n, &mut state).expect("contract");
            let ms = start.elapsed().as_millis();
            let max_bond = state
                .sites
                .iter()
                .map(|site| site.dim().2)
                .max()
                .unwrap_or(0);
            let snapped = crate::engine::snap_complex_result(result);
            #[cfg(feature = "webgpu")]
            let (accepts, fallbacks) = (
                crate::tn::mps::GPU_TWO_Q_ACCEPTS.load(std::sync::atomic::Ordering::Relaxed),
                crate::tn::mps::GPU_TWO_Q_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed),
            );
            #[cfg(not(feature = "webgpu"))]
            let (accepts, fallbacks) = (0u64, 0u64);
            eprintln!(
                "pass={pass} backend={} wall_ms={ms} max_bond={max_bond} trace_f64s={} amp=({}, {}) snap_key=({}, {}) gpu_accepts={accepts} gpu_fallbacks={fallbacks} vram_bytes={}",
                state.backend_label(),
                trace.len(),
                snapped.real,
                snapped.imag,
                (snapped.real / 1e-6).round() as i64,
                (snapped.imag / 1e-6).round() as i64,
                state.peak_vram_bytes(),
            );
            if pass > 0 {
                kept.push(ms);
            }
        }
        kept.sort();
        eprintln!("measured_ms={kept:?} median_ms={}", kept[kept.len() / 2]);
    }

    /// Same 20q / χ=128 brickwork on CPU f64 and on a webgpu device that only
    /// sends merges at or above the side threshold to the GPU.
    #[test]
    #[ignore = "20q chi=128 CPU vs GPU-merge snap"]
    fn bond128_gpu_merge_matches_cpu_snap() {
        #[cfg(feature = "webgpu")]
        {
            let Some(gpu) = crate::tn::gpu::GpuMpsDevice::try_new() else {
                eprintln!("skip bond128_gpu_merge_matches_cpu_snap: no WebGPU adapter");
                return;
            };
            let n = 20usize;
            let chi = 128usize;
            let gates = brickwork(n, 8);
            let mut cpu = MpsState::try_new_with_bond(n, chi).expect("cpu");
            let mut acc = MpsState::try_new_with_gpu(n, chi, gpu).expect("gpu");
            let cpu_start = std::time::Instant::now();
            let (cpu_amp, _) = contract_slice(n, &gates, &[], n, &mut cpu).expect("cpu");
            let cpu_ms = cpu_start.elapsed().as_millis();
            let start = std::time::Instant::now();
            let (gpu_amp, _) = contract_slice(n, &gates, &[], n, &mut acc).expect("gpu");
            let gpu_ms = start.elapsed().as_millis();
            let cpu_key = (
                (cpu_amp.real / 1e-6).round() as i64,
                (cpu_amp.imag / 1e-6).round() as i64,
            );
            let gpu_key = (
                (gpu_amp.real / 1e-6).round() as i64,
                (gpu_amp.imag / 1e-6).round() as i64,
            );
            eprintln!(
                "snap cpu={cpu_key:?} gpu={gpu_key:?} backend={} cpu_ms={cpu_ms} gpu_ms={gpu_ms} max_bond={} rsvd_kept={} rsvd_fallback={}",
                acc.backend_label(),
                acc.sites.iter().map(|site| site.dim().2).max().unwrap_or(0),
                crate::tn::mps::RSVD_KEPT.load(std::sync::atomic::Ordering::Relaxed),
                crate::tn::mps::RSVD_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed),
            );
            assert_eq!(cpu_key, gpu_key);
            assert_eq!(acc.backend_label(), "webgpu");
        }
        #[cfg(not(feature = "webgpu"))]
        {
            eprintln!("skip bond128_gpu_merge_matches_cpu_snap: webgpu feature off");
        }
    }

    fn brickwork(n: usize, layers: usize) -> Vec<Gate> {
        let mut gates = Vec::new();
        for layer in 0..layers {
            for q in 0..n {
                let angle = 1.1 + 0.05 * (q as f64) + 0.07 * (layer as f64);
                gates.push(Gate::RY(q, angle));
            }
            let mut q = layer % 2;
            while q + 1 < n {
                gates.push(Gate::CNOT(q, q + 1));
                q += 2;
            }
        }
        gates
    }

    #[test]
    fn truncated_bond_still_runs() {
        let gates = vec![Gate::H(0), Gate::CNOT(0, 1)];
        let mut mps = MpsState::try_new_with_bond(2, 4).expect("mps");
        assert!(contract_slice(2, &gates, &[], 2, &mut mps).is_ok());
    }

    #[test]
    fn reverse_site_order_matches_identity_amplitude() {
        use crate::tn::site_order::{logical_to_site_map, remap_gates, validate_site_order};

        let gates = vec![Gate::H(0), Gate::CNOT(0, 1)];
        let chi = exact_bond_dim(2);

        let mut mps_id = MpsState::try_new_with_bond(2, chi).expect("mps");
        let (r_id, _) = contract_slice(2, &gates, &[], 2, &mut mps_id).expect("id");

        let order = vec![1, 0];
        validate_site_order(&order, 2).unwrap();
        let remapped = remap_gates(&gates, &logical_to_site_map(&order)).unwrap();
        let mut mps_rev = MpsState::try_new_with_bond(2, chi).expect("mps");
        let (r_rev, _) = contract_slice(2, &remapped, &[], 2, &mut mps_rev).expect("rev");

        // |0…0⟩ is invariant under qubit permutation, so ⟨00|ψ⟩ matches.
        assert!((r_id.real - r_rev.real).abs() < 1e-8);
        assert!((r_id.imag - r_rev.imag).abs() < 1e-8);
    }

    #[test]
    fn scalar_50_and_100_qubit_product_amplitude() {
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        for n in [50usize, 100] {
            let gates = vec![Gate::H(0)];
            let mut mps = MpsState::try_new_with_bond(n, 128).expect("mps");
            let (result, _) = contract_slice(n, &gates, &[], n, &mut mps).expect("contract");
            assert!(
                (result.real - inv_sqrt2).abs() < 1e-8,
                "n={n} real {}",
                result.real
            );
            assert!(result.imag.abs() < 1e-8, "n={n} imag {}", result.imag);
        }
    }
}
