//! HTTP API surface for wqc-core: compute, verify, discovery, and health.

use axum::{Json, http::StatusCode};
use colored::*;
use serde::{Deserialize, Serialize};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, System};
use wqc_stark_engine::trace_spec::TRACE_WIDTH;
use crate::engine::{
    ComplexResult, ContractionWorkspace, SliceAssignment, TensorNetwork,
    calculate_complex_result_hash,
};
use crate::proof::{Proof, StarkProver};
use crate::expectation::{
    ExpectationResult, ObservableSpec, calculate_expectation_result_hash, compute_expectations,
    validate_expectation_task,
};
use crate::distribution_proof::{
    build_terminal_distribution_segment, distribution_stark_status, distribution_stark_status_bound,
};
use crate::mid_circuit::{
    extract_unitary_gates_for_proof, sample_mid_circuit_measurements, uses_mid_circuit_semantics,
    validate_phase_c_sample_circuit,
};
use crate::noise::NoiseModel;
use crate::sample::{
    OutputMode, SampleResult, calculate_sample_result_hash, sample_terminal_measurements,
    split_unitary_and_measures,
};

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
}

fn validate_task(
    task: &ComputeTask,
    measures: &[crate::engine::MeasureParams],
) -> Result<(), String> {
    match task.output_mode {
        OutputMode::StatevectorScalar => {
            if !measures.is_empty() {
                return Err("circuit contains MEASURE gates but output_mode is not sample_counts".into());
            }
            Ok(())
        }
        OutputMode::SampleCounts => validate_sample_task(task, measures),
        OutputMode::Expectation => validate_expectation_task(task.qubit_count, measures, &task.observables),
    }
}

fn validate_sample_task(task: &ComputeTask, measures: &[crate::engine::MeasureParams]) -> Result<(), String> {
    if measures.is_empty() {
        return Err("sample_counts requires at least one MEASURE gate".into());
    }

    let classical_bit_count = task.classical_bit_count.ok_or_else(|| {
        "classical_bit_count is required for sample_counts".to_string()
    })?;
    if classical_bit_count == 0 {
        return Err("classical_bit_count must be > 0".into());
    }

    validate_phase_c_sample_circuit(&task.circuit, classical_bit_count)
        .map_err(|e| e.to_string())?;

    let shots = task.shots.ok_or_else(|| "shots is required for sample_counts".to_string())?;
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

// --- Handlers ---

/// PROVER ROLE: Pre-allocates workspace, runs tensor contraction, and generates a zk-STARK proof.
pub async fn handle_compute(Json(task): Json<ComputeTask>) -> Result<Json<ComputeResponse>, (StatusCode, String)> {
    println!(
        "{} Processing STARK-monitored task {} (slice {}) on Node {} — effective qubits {} (original {})...",
        "⚙".bright_cyan(),
        task.task_id.bright_yellow(),
        task.slice_id.dimmed(),
        task.node_id.bright_green(),
        task.qubit_count,
        task.original_qubit_count,
    );

    let measures = crate::mid_circuit::collect_measures(&task.circuit);
    validate_task(&task, &measures).map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let unitary_gates = if uses_mid_circuit_semantics(&task.circuit) {
        extract_unitary_gates_for_proof(&task.circuit)
    } else {
        let (unitary, _) =
            split_unitary_and_measures(&task.circuit).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
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
    let network = TensorNetwork::from_parts(task.qubit_count, unitary_gates.clone(), task.slice_assignments)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let compute_start = std::time::Instant::now();
    let (complex_result, execution_trace) = network
        .contract(&mut workspace)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let sample_result = if task.output_mode == OutputMode::SampleCounts {
        let classical_bit_count = task.classical_bit_count.unwrap_or(0);
        let shots = task.shots.unwrap_or(0);
        let seed = task.sample_seed.unwrap_or(0);
        Some(if uses_mid_circuit_semantics(&task.circuit) {
            sample_mid_circuit_measurements(
                &task.circuit,
                task.qubit_count,
                classical_bit_count,
                shots,
                seed,
                task.noise_model.as_ref(),
            )
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        } else {
            let (_, terminal_measures) = split_unitary_and_measures(&task.circuit)
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            sample_terminal_measurements(
                workspace.register_mut(),
                &terminal_measures,
                classical_bit_count,
                shots,
                seed,
            )
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        })
    } else {
        None
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
        ("sample_counts".to_string(), calculate_sample_result_hash(sample))
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
        let classical_bit_count = task.classical_bit_count.unwrap_or(0) as usize;
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
        distribution_proof: Some(if distribution_segment.is_some() {
            distribution_stark_status_bound()
        } else {
            distribution_stark_status()
        }),
    }))
}

/// VALIDATOR ROLE: Instantly verifies a zk-STARK proof without re-running contraction.
pub async fn handle_verify(Json(proof): Json<VerifyProof>) -> Result<Json<VerifyResponse>, (StatusCode, Json<VerifyResponse>)> {
    // Stateless verification over the declared public inputs and proof transcript.
    let validator = StarkProver;
    if validator.verify_proof(&proof.proof) {
        println!("{} zk-STARK proof verified instantly for remote node.", "★".bright_yellow());
        Ok(Json(VerifyResponse {
            valid: true,
            reason: None,
        }))
    } else {
        println!("{} Fraudulent zk-STARK proof or polynomial mismatch detected!", "✘".red());
        Err((
            StatusCode::FORBIDDEN,
            Json(VerifyResponse {
                valid: false,
                reason: Some("Invalid zk-STARK transcript alignment".to_string()),
            }),
        ))
    }
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
    })
}
