use crate::engine::{QuantumRegister, Circuit, Gate};
use crate::proof::{Miner, PoUWResult};
use axum::{Json, http::StatusCode};
use colored::*;
use serde::{Deserialize, Serialize};
use sysinfo::System;

const MIN_ARGON2_KB: u32 = 8;

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
    // 1. Validation: Check Argon2 parameters
    if task.memory_cost_kb < MIN_ARGON2_KB {
        return Err((StatusCode::BAD_REQUEST, format!("memory_cost_kb must be at least {} KB", MIN_ARGON2_KB)));
    }

    // 2. Pre-check memory capacity
    // Determine if the task can be accepted by checking current system-wide available memory before allocation.
    let required_memory = (1u64 << task.qubit_count) * 16;
    let mut sys = System::new_all();
    sys.refresh_memory();

    // Apply a safety factor of 0.7 (70%) to protect other processes and the OS.
    let safe_available = (sys.available_memory() as f64 * 0.7) as u64;

    if required_memory > safe_available {
        println!("{} Task {} rejected: Memory busy (Required: {}MB, Available Safe: {}MB)",
            "⚠".yellow(), task.task_id, required_memory/1024/1024, safe_available/1024/1024);
        return Err((StatusCode::SERVICE_UNAVAILABLE, "Node is busy: Insufficient memory for this task size".into()));
    }

    // 3. Setup Register with dynamic memory guard
    let mut register = QuantumRegister::new(task.qubit_count)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Resource Guard: {}", e)))?;

    // 4. Build and Validate Circuit
    let mut circuit = Circuit::new(task.qubit_count);
    for gate in task.circuit {
        circuit.add(gate)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Circuit Error: {}", e)))?;
    }

    // 5. Execution
    circuit.execute(&mut register)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // 6. PoUW Generation (Mining)
    let miner = Miner::new(task.difficulty, task.memory_cost_kb);
    let raw_state = register.state.as_slice().unwrap();
    // Convert to [re, im] format
    let formatted_state: Vec<[f64; 2]> = raw_state.iter().map(|c| [c.re, c.im]).collect();
    let proof = miner.solve(&formatted_state)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

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
    // 1. Validation
    if task.memory_cost_kb < MIN_ARGON2_KB {
        return Err((StatusCode::BAD_REQUEST, Json(VerifyResponse {
            valid: false,
            reason: Some(format!("Invalid parameter: memory_cost_kb must be at least {}", MIN_ARGON2_KB)),
        })));
    }

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
