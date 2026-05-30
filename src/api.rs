use axum::{Json, http::StatusCode};
use colored::*;
use serde::{Deserialize, Serialize};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, System};
use crate::engine::{QuantumRegister, Circuit, Gate};
use crate::proof::{StarkProver, Proof};

#[derive(Debug, Deserialize)]
pub struct ComputeTask {
    pub task_id: String,
    pub circuit_id: String,
    pub node_id: String,
    pub qubit_count: usize,
    pub original_qubit_count: usize,
    pub global_offset: String,
    pub circuit: Vec<Gate>,
}

#[derive(Debug, Serialize)]
pub struct ComputeResponse {
    pub task_id: String,
    pub status: String,
    pub state_vector: Vec<[f64; 2]>,
    pub proof: Proof,
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

#[derive(Debug, Serialize)]
pub struct SystemInfo {
    pub system_memory_used_kb: u64,
    pub system_memory_total_kb: u64,
    pub cpu_usage_percent: f32,
}

// --- Handlers ---

/// PROVER ROLE: Executes quantum circuit, captures execution trace, and generates an algebraic zk-STARK proof.
pub async fn handle_compute(Json(task): Json<ComputeTask>) -> Result<Json<ComputeResponse>, (StatusCode, String)> {
    println!(
        "{} Processing STARK-monitored task {} on Node {} with {} qubits...",
        "⚙".bright_cyan(),
        task.task_id.bright_yellow(),
        task.node_id.bright_green(),
        task.qubit_count
    );

    // 1. Initialize Quantum Register via memory limits configuration
    let mut register = match QuantumRegister::new(task.qubit_count) {
        Ok(reg) => reg,
        Err(e) => return Err((StatusCode::BAD_REQUEST, e.to_string())),
    };

    // 2. Build circuit instruction matrices
    let mut circuit = Circuit::new(task.qubit_count);
    for gate in task.circuit {
        if let Err(e) = circuit.add(gate) {
            return Err((StatusCode::BAD_REQUEST, e.to_string()));
        }
    }

    // 3. Execute while extracting raw polynomial execution traces
    let execution_trace = match circuit.execute_with_trace(&mut register) {
        Ok(trace) => trace,
        Err(e) => return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    };

    // 4. Format memory states back to structured float pairs
    let formatted_state: Vec<[f64; 2]> = register
        .state
        .iter()
        .map(|c| [c.re, c.im])
        .collect();

    // 5. Generate the real state hash commitment from the computed state vector
    let generated_output_hash = circuit.calculate_output_hash(&formatted_state);

    // 6. Generate Plonky3 zk-STARK proof transcript over Mersenne31 prime field
    let prover = StarkProver;
    let proof = match prover.generate_proof(
        &task.circuit_id,
        &task.task_id,
        &task.node_id,
        &generated_output_hash,
        &execution_trace,
    ) {
        Ok(p) => p,
        Err(e) => return Err((StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    println!("{} STARK Proof generated successfully for Task: {}", "✔".green(), task.task_id.bright_yellow());

    Ok(Json(ComputeResponse {
        task_id: task.task_id,
        status: "success".to_string(),
        state_vector: formatted_state,
        proof,
    }))
}

/// VALIDATOR ROLE: Instantly verifies a zk-STARK proof without re-executing any quantum state transformations.
pub async fn handle_verify(Json(proof): Json<VerifyProof>) -> Result<Json<VerifyResponse>, (StatusCode, Json<VerifyResponse>)> {
    // Stateless verification bounding the public inputs (circuit_id, task_id, output_hash)
    let validator = StarkProver;
    if validator.verify_proof(&proof.proof) {
        println!("{} zk-STARK Proof verified instantly for remote node.", "★".bright_yellow());
        Ok(Json(VerifyResponse {
            valid: true,
            reason: None,
        }))
    } else {
        println!("{} Fraudulent zk-STARK proof or polynomial mismatch detected!", "✘".red());
        Err((StatusCode::FORBIDDEN, Json(VerifyResponse {
            valid: false,
            reason: Some("Invalid zk-STARK Transcript Alignment".to_string()),
        })))
    }
}

/// Returns a list of supported gates for Orchestrator discovery.
pub async fn get_supported_gates() -> Json<Vec<String>> {
    use strum::IntoEnumIterator;
    let gates = crate::engine::Gate::iter().map(|g| g.to_string()).collect();
    Json(gates)
}

/// Returns a list of supported gates for Orchestrator discovery.
pub async fn get_system_info() -> Json<SystemInfo> {
    let mut sys = System::new_all();
    // Refresh only what we need for performance
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
