use crate::engine::{QuantumRegister, Circuit};
use crate::proof::{Miner, PoUWResult};
use axum::{Json, response::IntoResponse, http::StatusCode};
use serde::{Deserialize, Serialize};
use colored::*;
use num_complex::Complex64;

// --- Existing Compute Task Structures ---

#[derive(Deserialize)]
pub struct ComputeTask {
    pub qubit_count: usize,
    pub circuit: Vec<crate::engine::Gate>,
    pub difficulty: u32,
    pub memory_cost_kb: u32,
}

#[derive(Serialize)]
pub struct ComputeResult {
    pub state_vector: Vec<Complex64>,
    pub proof: PoUWResult,
}

// --- New Verification Task Structure ---

#[derive(Deserialize)]
pub struct VerifyTask {
    pub state_vector: Vec<Complex64>,
    pub proof: PoUWResult,
    pub difficulty: u32,
    pub memory_cost_kb: u32,
}

// --- Handlers ---

/// PROVER ROLE: Executes quantum circuit and generates a PoUW.
pub async fn handle_compute(Json(task): Json<ComputeTask>) -> Result<Json<ComputeResult>, (StatusCode, String)> {
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
    let proof = miner.solve(register.state.as_slice().unwrap());

    println!("{} Computed & Proof Generated (Difficulty: {})", "✔".green(), task.difficulty);

    Ok(Json(ComputeResult {
        state_vector: register.state.to_vec(),
        proof,
    }))
}

/// VALIDATOR ROLE: Verifies a proof without re-executing the quantum circuit.
pub async fn handle_verify(Json(task): Json<VerifyTask>) -> impl IntoResponse {
    let validator = Miner::new(task.difficulty, task.memory_cost_kb);

    // Perform lightweight bit-level verification
    if validator.verify(&task.state_vector, &task.proof) {
        println!("{} Proof verified for remote node.", "★".bright_yellow());
        (StatusCode::OK, "Verification Successful")
    } else {
        println!("{} Fraudulent proof detected or difficulty too low!", "✘".red());
        (StatusCode::FORBIDDEN, "Invalid Proof or Insufficient Difficulty")
    }
}

/// Returns a list of supported gates for Orchestrator discovery.
pub async fn get_supported_gates() -> Json<Vec<String>> {
    use strum::IntoEnumIterator;
    let gates = crate::engine::Gate::iter().map(|g| g.to_string()).collect();
    Json(gates)
}
