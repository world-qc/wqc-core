use crate::engine::{QuantumRegister, Circuit, Gate};
use crate::proof::{Miner, PoUWResult};
use axum::{Json, http::StatusCode};
use serde::{Deserialize, Serialize};
use colored::*;

#[derive(Deserialize)]
pub struct ComputeTask {
    pub task_id: String,
    pub qubit_count: usize,
    pub circuit: Vec<Gate>,
    pub difficulty: u32,
    pub memory_cost_kb: u32,
}

#[derive(Serialize)]
pub struct ComputeResponse {
    pub task_id: String,
    pub status: String,
    pub state_vector: Vec<[f64; 2]>,
    pub proof: PoUWResult,
}

#[derive(Deserialize)]
pub struct VerifyTask {
    pub state_vector: Vec<[f64; 2]>,
    pub proof: PoUWResult,
    pub difficulty: u32,
    pub memory_cost_kb: u32,
}

#[derive(Serialize)]
pub struct VerifyResponse {
    pub valid: bool,
    pub reason: Option<String>,
}

// --- Handlers ---

/// PROVER ROLE: Executes quantum circuit and generates a PoUW.
pub async fn handle_compute(Json(task): Json<ComputeTask>) -> Result<Json<ComputeResponse>, (StatusCode, String)> {
    // 1. Setup Register with dynamic memory guard
    let mut register = QuantumRegister::new(task.qubit_count)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Resource Guard: {}", e)))?;

    // 2. Build and Validate Circuit
    let mut circuit = Circuit::new(task.qubit_count);
    for gate in task.circuit {
        circuit.add(gate)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Circuit Error: {}", e)))?;
    }

    // 3. Execution
    circuit.execute(&mut register)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // 4. PoUW Generation (Mining)
    let miner = Miner::new(task.difficulty, task.memory_cost_kb);
    let raw_state = register.state.as_slice().unwrap();
    // Convert to [re, im] format
    let formatted_state: Vec<[f64; 2]> = raw_state.iter().map(|c| [c.re, c.im]).collect();
    let proof = miner.solve(&formatted_state);

    println!("{} Task {}: Computed & Proof Generated", "✔".green(), task.task_id);

    Ok(Json(ComputeResponse {
        task_id: task.task_id,
        status: "success".to_string(),
        state_vector: formatted_state,
        proof,
    }))
}

/// VALIDATOR ROLE: Verifies a proof without re-executing the quantum circuit.
pub async fn handle_verify(Json(task): Json<VerifyTask>) -> Result<Json<VerifyResponse>, (StatusCode, Json<VerifyResponse>)> {
    let validator = Miner::new(task.difficulty, task.memory_cost_kb);

    // Perform lightweight bit-level verification
    if validator.verify(&task.state_vector, &task.proof) {
        println!("{} Proof verified for remote node.", "★".bright_yellow());
        Ok(Json(VerifyResponse {
            valid: true,
            reason: None,
        }))
    } else {
        println!("{} Fraudulent proof detected or difficulty too low!", "✘".red());
        Err((StatusCode::FORBIDDEN, Json(VerifyResponse {
            valid: false,
            reason: Some("Invalid Proof or Insufficient Difficulty".to_string()),
        })))
    }
}

/// Returns a list of supported gates for Orchestrator discovery.
pub async fn get_supported_gates() -> Json<Vec<String>> {
    use strum::IntoEnumIterator;
    let gates = crate::engine::Gate::iter().map(|g| g.to_string()).collect();
    Json(gates)
}
