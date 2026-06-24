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
}

/// Successful compute response: one contracted scalar plus its STARK proof.
#[derive(Debug, Serialize)]
pub struct ComputeResponse {
    pub task_id: String,
    pub status: String,
    pub complex_result: ComplexResult,
    pub proof: Proof,
    pub work_report: WorkReport,
}

/// Auditable execution metrics returned to wqc-node for D-PoUW settlement.
#[derive(Debug, Serialize)]
pub struct WorkReport {
    pub trace_rows: u64,
    pub gate_count: u32,
    pub compute_wall_ms: u64,
    pub prove_wall_ms: u64,
    pub proof_bytes: u64,
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

    // Step 1: MPS workspace pre-allocation (O(N · χ²); χ from orchestrator ∩ env).
    let mut workspace = ContractionWorkspace::try_allocate_with_bond(
        task.qubit_count,
        task.original_qubit_count,
        task.mps_max_bond_dim,
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Step 2: Tensor-network contraction with slice boundary conditions.
    let network = TensorNetwork::from_parts(task.qubit_count, task.circuit.clone(), task.slice_assignments)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let compute_start = std::time::Instant::now();
    let (complex_result, execution_trace) = network
        .contract(&mut workspace)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let compute_wall_ms = compute_start.elapsed().as_millis() as u64;

    // Step 3: Bind public inputs (circuit_id, slice_id, task_id) and emit zk-STARK proof.
    let output_result_hash = calculate_complex_result_hash(&complex_result);

    let prove_start = std::time::Instant::now();
    let prover = StarkProver;
    let proof = prover
        .generate_proof(
            &task.circuit_id,
            &task.task_id,
            &task.node_id,
            &task.slice_id,
            &output_result_hash,
            &execution_trace,
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let prove_wall_ms = prove_start.elapsed().as_millis() as u64;

    let trace_rows = (execution_trace.len() / TRACE_WIDTH) as u64;
    let proof_bytes = proof.stark_proof_b64.len() as u64;
    let gate_count = task.circuit.len() as u32;

    println!(
        "{} STARK proof generated for task {} — amplitude ({}, {})",
        "✔".green(),
        task.task_id.bright_yellow(),
        complex_result.real,
        complex_result.imag,
    );

    Ok(Json(ComputeResponse {
        task_id: task.task_id,
        status: "success".to_string(),
        complex_result,
        proof,
        work_report: WorkReport {
            trace_rows,
            gate_count,
            compute_wall_ms,
            prove_wall_ms,
            proof_bytes,
        },
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

    Json(SystemInfo {
        system_memory_used_kb: sys.used_memory() / 1024,
        system_memory_total_kb: sys.total_memory() / 1024,
        cpu_usage_percent: sys.global_cpu_info().cpu_usage(),
    })
}
