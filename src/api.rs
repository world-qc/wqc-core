//! HTTP API surface for wqc-core: compute, verify, discovery, and health.

use crate::distribution_proof::{
    build_terminal_distribution_segment, distribution_stark_status,
    distribution_stark_status_born_air, distribution_stark_status_born_air_zk,
    distribution_stark_status_born_air_zk_composed, distribution_stark_status_bound,
};
use crate::engine::{
    calculate_complex_result_hash, ComplexResult, ContractionWorkspace, SliceAssignment,
    TensorNetwork,
};
use crate::expectation::{
    calculate_expectation_result_hash, compute_expectations, validate_expectation_task,
    ExpectationResult, ObservableSpec,
};
use crate::mid_circuit::{
    extract_unitary_gates_for_proof, sample_mid_circuit_measurements_with_trace,
    uses_mid_circuit_semantics, validate_phase_c_sample_circuit,
};
use crate::noise::NoiseModel;
use crate::proof::{Proof, StarkProver};
use crate::sample::{
    calculate_sample_result_hash, sample_terminal_measurements, split_unitary_and_measures,
    OutputMode, SampleResult,
};
use crate::trajectory_proof::{
    build_trajectory_segment, distribution_stark_status_trajectory_air_zk,
    distribution_stark_status_trajectory_air_zk_composed,
    distribution_stark_status_trajectory_bound,
};
use axum::{http::StatusCode, Json};
use colored::*;
use serde::{Deserialize, Serialize};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, System};
use wqc_stark_engine::plonky3_stark::recursion::PcsMemoryPolicy;
use wqc_stark_engine::trace_spec::TRACE_WIDTH;

/// Payload received from wqc-node (matches the orchestrator's pruned sub-task shape).
#[derive(Debug, Deserialize)]
pub struct ComputeTask {
    /// Unique sub-task identifier (also used as STARK `sub_task_id`).
    pub task_id: String,
    /// Hash of the pruned tensor sub-graph; bound into zk-STARK public inputs.
    pub circuit_id: String,
    /// Executing node identity (injected by wqc-node before dispatch).
    #[serde(default)]
    pub node_id: String,
    /// Effective tensor complexity `QubitCount` (max intermediate dimension for this slice).
    pub qubit_count: usize,
    /// Global circuit size before slicing; used for engine initialization context.
    pub original_qubit_count: usize,
    /// Binary path of the slice tree (e.g. `"0"`, `"01"`); bound into public inputs.
    pub slice_id: String,
    /// Fixed classical values on cut edges (`e_0`, `e_1`, … from the orchestrator).
    pub slice_assignments: Vec<SliceAssignment>,
    /// Pruned gate list for this slice (already optimized upstream).
    pub circuit: Vec<crate::engine::Gate>,
    /// Orchestrator-recommended MPS bond dimension χ (capped by `WQC_MPS_MAX_BOND_DIM`).
    #[serde(default)]
    pub mps_max_bond_dim: Option<usize>,
    /// Optional MPS path: `site_order[site] = logical` compact qubit (identity if absent/empty).
    #[serde(default)]
    pub mps_site_order: Option<Vec<usize>>,
    /// Result mode (default: contracted scalar amplitude).
    #[serde(default)]
    pub output_mode: OutputMode,
    /// Classical register width for terminal measurements.
    #[serde(default)]
    pub classical_bit_count: Option<usize>,
    /// Shot count for `sample_counts` mode.
    #[serde(default)]
    pub shots: Option<u64>,
    /// PRNG seed for deterministic histogram reproduction.
    #[serde(default)]
    pub sample_seed: Option<u64>,
    /// Named Pauli observables for `expectation` mode.
    #[serde(default)]
    pub observables: Vec<ObservableSpec>,
    /// Optional noise model for trajectory sampling (Phase C3; not STARK-bound).
    #[serde(default)]
    pub noise_model: Option<NoiseModel>,
}

/// Successful compute response: scalar and/or sample histogram plus STARK proof.
#[derive(Debug, Serialize)]
pub struct ComputeResponse {
    pub task_id: String,
    pub status: String,
    #[serde(rename = "result_type")]
    pub result_type: String,
    pub complex_result: ComplexResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_result: Option<SampleResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expectation_result: Option<ExpectationResult>,
    pub proof: Proof,
    pub work_report: WorkReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distribution_proof: Option<crate::distribution_proof::DistributionProofStatus>,
}

/// Auditable execution metrics returned to wqc-node for D-PoUW settlement.
#[derive(Debug, Serialize)]
pub struct WorkReport {
    pub trace_rows: u64,
    pub gate_count: u32,
    pub compute_wall_ms: u64,
    pub prove_wall_ms: u64,
    pub proof_bytes: u64,
    #[serde(default)]
    pub tn_backend: String,
    #[serde(default)]
    pub vram_peak_bytes: u64,
}

#[derive(Debug, Deserialize)]
pub struct VerifyProof {
    pub proof: Proof,
}

#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub valid: bool,
    pub reason: Option<String>,
}

/// Request body for deferred leaf PCS bundle construction.
#[derive(Debug, Deserialize)]
pub struct LeafPcsRequest {
    pub proof: Proof,
}

/// Encoded leaf PCS bundle returned to wqc-node for orch follow-up delivery.
#[derive(Debug, Serialize)]
pub struct LeafPcsResponse {
    pub leaf_pcs_b64: String,
    pub bytes: u64,
}

/// Host resource snapshot for orchestrator / node capacity discovery.
#[derive(Debug, Serialize)]
pub struct SystemInfo {
    pub system_memory_used_kb: u64,
    pub system_memory_total_kb: u64,
    pub cpu_usage_percent: f32,
    pub tn_backend_requested: String,
    pub tn_backend_active: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tn_backend_note: Option<String>,
    pub mps_max_bond_dim: usize,
    /// Prove-time PCS RAM gate (`WQC_PCS_MEMORY_POLICY` on this core process).
    pub pcs_memory_policy: String,
}

fn validate_task(
    task: &ComputeTask,
    measures: &[crate::engine::MeasureParams],
) -> Result<(), String> {
    match task.output_mode {
        OutputMode::StatevectorScalar => {
            if !measures.is_empty() {
                return Err(
                    "circuit contains MEASURE gates but output_mode is not sample_counts".into(),
                );
            }
            Ok(())
        }
        OutputMode::SampleCounts => validate_sample_task(task, measures),
        OutputMode::Expectation => {
            validate_expectation_task(task.qubit_count, measures, &task.observables)
        }
    }
}

fn validate_sample_task(
    task: &ComputeTask,
    measures: &[crate::engine::MeasureParams],
) -> Result<(), String> {
    if measures.is_empty() {
        return Err("sample_counts requires at least one MEASURE gate".into());
    }

    let classical_bit_count = task
        .classical_bit_count
        .ok_or_else(|| "classical_bit_count is required for sample_counts".to_string())?;
    if classical_bit_count == 0 {
        return Err("classical_bit_count must be > 0".into());
    }

    validate_phase_c_sample_circuit(&task.circuit, classical_bit_count)
        .map_err(|e| e.to_string())?;

    let shots = task
        .shots
        .ok_or_else(|| "shots is required for sample_counts".to_string())?;
    if shots == 0 {
        return Err("shots must be >= 1".into());
    }

    if task.sample_seed.is_none() {
        return Err("sample_seed is required for sample_counts".into());
    }

    if let Some(noise) = &task.noise_model {
        if let Some(p) = noise.depolarizing_p {
            if !(0.0..=1.0).contains(&p) {
                return Err("depolarizing_p must be in [0, 1]".into());
            }
        }
        if let Some(p) = noise.readout_error {
            if !(0.0..=1.0).contains(&p) {
                return Err("readout_error must be in [0, 1]".into());
            }
        }
    }

    Ok(())
}

fn apply_mps_site_order(task: &mut ComputeTask) -> Result<(), crate::engine::EngineError> {
    let Some(order) = task.mps_site_order.as_ref() else {
        return Ok(());
    };
    if order.is_empty() {
        return Ok(());
    }
    crate::tn::site_order::validate_site_order(order, task.qubit_count)?;
    let logical_to_site = crate::tn::site_order::logical_to_site_map(order);
    task.circuit = crate::tn::site_order::remap_gates(&task.circuit, &logical_to_site)?;
    if !task.observables.is_empty() {
        task.observables =
            crate::tn::site_order::remap_observables(&task.observables, &logical_to_site)?;
    }
    Ok(())
}

// --- Handlers ---

/// PROVER ROLE: Pre-allocates workspace, runs tensor contraction, and generates a zk-STARK proof.
pub async fn handle_compute(
    Json(mut task): Json<ComputeTask>,
) -> Result<Json<ComputeResponse>, (StatusCode, String)> {
    println!(
        "{} Processing STARK-monitored task {} (slice {}) on Node {} — effective qubits {} (original {})...",
        "⚙".bright_cyan(),
        task.task_id.bright_yellow(),
        task.slice_id.dimmed(),
        task.node_id.bright_green(),
        task.qubit_count,
        task.original_qubit_count,
    );

    apply_mps_site_order(&mut task).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let measures = crate::mid_circuit::collect_measures(&task.circuit);
    validate_task(&task, &measures).map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let unitary_gates = if uses_mid_circuit_semantics(&task.circuit) {
        extract_unitary_gates_for_proof(&task.circuit)
    } else {
        let (unitary, _) = split_unitary_and_measures(&task.circuit)
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        unitary
    };

    // Step 1: MPS workspace pre-allocation (O(N · χ²); χ from orchestrator ∩ env).
    let mut workspace = ContractionWorkspace::try_allocate_with_bond(
        task.qubit_count,
        task.original_qubit_count,
        task.mps_max_bond_dim,
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Step 2: Tensor-network contraction with slice boundary conditions (unitary gates only).
    let network = TensorNetwork::from_parts(
        task.qubit_count,
        unitary_gates.clone(),
        task.slice_assignments,
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let compute_start = std::time::Instant::now();
    let (complex_result, execution_trace) = network
        .contract(&mut workspace)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let (sample_result, mid_circuit_trace) = if task.output_mode == OutputMode::SampleCounts {
        let classical_bit_count = task.classical_bit_count.unwrap_or(0);
        let shots = task.shots.unwrap_or(0);
        let seed = task.sample_seed.unwrap_or(0);
        if uses_mid_circuit_semantics(&task.circuit) {
            let (sample, trace) = sample_mid_circuit_measurements_with_trace(
                &task.circuit,
                task.qubit_count,
                classical_bit_count,
                shots,
                seed,
                task.noise_model.as_ref(),
            )
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            let bound_trace = if task.noise_model.is_none() {
                Some(trace)
            } else {
                None
            };
            (Some(sample), bound_trace)
        } else {
            let (_, terminal_measures) = split_unitary_and_measures(&task.circuit)
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            let sample = sample_terminal_measurements(
                workspace.register_mut(),
                &terminal_measures,
                classical_bit_count,
                shots,
                seed,
            )
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            (Some(sample), None)
        }
    } else {
        (None, None)
    };

    let expectation_result = if task.output_mode == OutputMode::Expectation {
        Some(
            compute_expectations(workspace.register_mut(), &task.observables)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
        )
    } else {
        None
    };

    let compute_wall_ms = compute_start.elapsed().as_millis() as u64;

    // Step 3: Bind public inputs and emit zk-STARK proof (unitary trace; hash binds to output mode).
    let (result_type, output_result_hash) = if let Some(ref sample) = sample_result {
        (
            "sample_counts".to_string(),
            calculate_sample_result_hash(sample),
        )
    } else if let Some(ref expectation) = expectation_result {
        (
            "expectation".to_string(),
            calculate_expectation_result_hash(expectation),
        )
    } else {
        (
            "statevector_scalar".to_string(),
            calculate_complex_result_hash(&complex_result),
        )
    };

    let prove_start = std::time::Instant::now();
    let distribution_segment = if task.output_mode == OutputMode::SampleCounts
        && !uses_mid_circuit_semantics(&task.circuit)
        && task.noise_model.is_none()
    {
        let classical_bit_count = task.classical_bit_count.unwrap_or(0);
        let shots = task.shots.unwrap_or(0);
        let seed = task.sample_seed.unwrap_or(0);
        let (_, terminal_measures) = split_unitary_and_measures(&task.circuit)
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        Some(
            build_terminal_distribution_segment(
                workspace.register_mut(),
                &terminal_measures,
                classical_bit_count,
                shots,
                seed,
            )
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
        )
    } else {
        None
    };

    let trajectory_segment = if let Some(trace) = mid_circuit_trace.as_ref() {
        let measures = crate::mid_circuit::collect_measures(&task.circuit);
        let measurement_spec_hash =
            crate::distribution_proof::calculate_measurement_spec_hash(&measures);
        Some(build_trajectory_segment(
            trace,
            task.qubit_count as u32,
            task.sample_seed.unwrap_or(0),
            task.shots.unwrap_or(0),
            measurement_spec_hash,
        ))
    } else {
        None
    };

    let prover = StarkProver;
    let proof = prover
        .generate_proof(
            &task.circuit_id,
            &task.task_id,
            &task.node_id,
            &task.slice_id,
            &output_result_hash,
            &execution_trace,
            distribution_segment.as_ref(),
            trajectory_segment.as_ref(),
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let prove_wall_ms = prove_start.elapsed().as_millis() as u64;

    let trace_rows = (execution_trace.len() / TRACE_WIDTH) as u64;
    let proof_bytes = proof.stark_proof_b64.len() as u64;
    let gate_count = unitary_gates.len() as u32;

    if let Some(ref sample) = sample_result {
        println!(
            "{} STARK proof generated for task {} — sample counts {:?} ({} shots)",
            "✔".green(),
            task.task_id.bright_yellow(),
            sample.counts,
            sample.shots,
        );
    } else if let Some(ref expectation) = expectation_result {
        println!(
            "{} STARK proof generated for task {} — expectation {:?}",
            "✔".green(),
            task.task_id.bright_yellow(),
            expectation.values,
        );
    } else {
        println!(
            "{} STARK proof generated for task {} — amplitude ({}, {})",
            "✔".green(),
            task.task_id.bright_yellow(),
            complex_result.real,
            complex_result.imag,
        );
    }

    Ok(Json(ComputeResponse {
        task_id: task.task_id,
        status: "success".to_string(),
        result_type,
        complex_result,
        sample_result,
        expectation_result,
        proof,
        work_report: WorkReport {
            trace_rows,
            gate_count,
            compute_wall_ms,
            prove_wall_ms,
            proof_bytes,
            tn_backend: workspace.tn_backend_label().to_string(),
            vram_peak_bytes: workspace.peak_vram_bytes(),
        },
        distribution_proof: Some(
            if trajectory_segment.as_ref().is_some_and(|s| {
                wqc_stark_engine::segment_supports_trajectory_zk(s)
                    && !s.unitary_link_digest.is_empty()
            }) {
                distribution_stark_status_trajectory_air_zk_composed()
            } else if trajectory_segment
                .as_ref()
                .is_some_and(wqc_stark_engine::segment_supports_trajectory_zk)
            {
                distribution_stark_status_trajectory_air_zk()
            } else if trajectory_segment.is_some() {
                distribution_stark_status_trajectory_bound()
            } else if distribution_segment.as_ref().is_some_and(|s| {
                wqc_stark_engine::segment_supports_born_zk(s)
                    && s.born_binding
                        .as_ref()
                        .is_some_and(|b| !b.terminal_statevector_digest.is_empty())
            }) {
                distribution_stark_status_born_air_zk_composed()
            } else if distribution_segment
                .as_ref()
                .is_some_and(wqc_stark_engine::segment_supports_born_zk)
            {
                distribution_stark_status_born_air_zk()
            } else if distribution_segment
                .as_ref()
                .is_some_and(|s| s.born_binding.is_some())
            {
                distribution_stark_status_born_air()
            } else if distribution_segment.is_some() {
                distribution_stark_status_bound()
            } else {
                distribution_stark_status()
            },
        ),
    }))
}

/// VALIDATOR ROLE: Instantly verifies a zk-STARK proof without re-running contraction.
pub async fn handle_verify(
    Json(proof): Json<VerifyProof>,
) -> Result<Json<VerifyResponse>, (StatusCode, Json<VerifyResponse>)> {
    // Stateless verification over the declared public inputs and proof transcript.
    let validator = StarkProver;
    if validator.verify_proof(&proof.proof) {
        println!(
            "{} zk-STARK proof verified instantly for remote node.",
            "★".bright_yellow()
        );
        Ok(Json(VerifyResponse {
            valid: true,
            reason: None,
        }))
    } else {
        println!(
            "{} Fraudulent zk-STARK proof or polynomial mismatch detected!",
            "✘".red()
        );
        Err((
            StatusCode::FORBIDDEN,
            Json(VerifyResponse {
                valid: false,
                reason: Some("Invalid zk-STARK transcript alignment".to_string()),
            }),
        ))
    }
}

/// Builds a standalone leaf PCS bundle from an already-proven leaf STARK.
///
/// Called by wqc-node after result delivery so `/compute` stays within the compute timeout.
pub async fn handle_leaf_pcs(
    Json(req): Json<LeafPcsRequest>,
) -> Result<Json<LeafPcsResponse>, (StatusCode, String)> {
    use base64::{engine::general_purpose, Engine as _};

    let proof_bytes = general_purpose::STANDARD
        .decode(req.proof.stark_proof_b64.as_bytes())
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("invalid stark_proof_b64: {e}"),
            )
        })?;

    println!(
        "{} Building deferred leaf PCS bundle for sub_task {} ({} proof bytes)...",
        "⚙".bright_cyan(),
        req.proof.public_inputs.sub_task_id.bright_yellow(),
        proof_bytes.len(),
    );

    let started = std::time::Instant::now();
    let encoded = wqc_stark_engine::build_encoded_leaf_pcs_bundle_from_child(&proof_bytes)
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, e))?;
    let elapsed_ms = started.elapsed().as_millis();

    println!(
        "{} Leaf PCS bundle ready ({} bytes, {} ms)",
        "★".bright_yellow(),
        encoded.len(),
        elapsed_ms,
    );

    Ok(Json(LeafPcsResponse {
        bytes: encoded.len() as u64,
        leaf_pcs_b64: general_purpose::STANDARD.encode(&encoded),
    }))
}

/// Returns supported gates for orchestrator discovery.
pub async fn get_supported_gates() -> Json<Vec<String>> {
    use strum::IntoEnumIterator;
    let gates = crate::engine::Gate::iter().map(|g| g.to_string()).collect();
    Json(gates)
}

/// Returns memory and CPU metrics for node scheduling and capacity checks.
pub async fn get_system_info() -> Json<SystemInfo> {
    let mut sys = System::new_all();
    // Refresh only what we need to keep the endpoint lightweight.
    sys.refresh_specifics(
        sysinfo::RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything()),
    );

    let tn = crate::tn::tn_engine_status();

    Json(SystemInfo {
        system_memory_used_kb: sys.used_memory() / 1024,
        system_memory_total_kb: sys.total_memory() / 1024,
        cpu_usage_percent: sys.global_cpu_info().cpu_usage(),
        tn_backend_requested: tn.requested.clone(),
        tn_backend_active: tn.active.clone(),
        tn_backend_note: tn.note.clone(),
        mps_max_bond_dim: tn.mps_max_bond_dim,
        pcs_memory_policy: PcsMemoryPolicy::from_env().as_str().to_string(),
    })
}
