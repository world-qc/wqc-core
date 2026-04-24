use axum::{extract::Json, http::StatusCode};
use crate::engine::{Circuit, Gate, QuantumRegister};
use crate::proof::Miner;
use sha3::{Digest, Sha3_256};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

#[derive(Deserialize)]
pub struct ComputeTask {
    pub task_id: String,
    pub qubit_count: usize,
    pub circuit: Vec<Gate>,
    pub difficulty: u32,
    pub memory_cost_kb: u32,
}

#[derive(Serialize)]
pub struct ComputeResult {
    pub task_id: String,
    pub status: String,
    pub state_hash: String,
    pub proof: Option<ProofData>,
}

#[derive(Serialize)]
pub struct ProofData {
    pub nonce: u64,
    pub proof_hash: String,
}

pub async fn handle_compute(Json(task): Json<ComputeTask>) -> Result<Json<ComputeResult>, (StatusCode, String)> {
    // 1. Setup Register & Circuit with Error Handling (Hardening phase)
    let mut register = QuantumRegister::new(task.qubit_count)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Engine Error: {}", e)))?;

    let mut circuit = Circuit::new(task.qubit_count);
    for gate in task.circuit {
        // Validation: Return 400 Bad Request if gate indices are invalid
        circuit.add(gate)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Circuit Validation Error: {}", e)))?;
    }

    // 2. Execute Quantum Computation
    circuit.execute(&mut register)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // 3. Generate State Hash (Commitment)
    let mut hasher = Sha3_256::new();
    let state_slice = register.state.as_slice()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "Memory layout error".to_string()))?;

    for val in state_slice {
        hasher.update(val.re.to_le_bytes());
        hasher.update(val.im.to_le_bytes());
    }
    let state_hash = hex::encode(hasher.finalize());

    // 4. Run PoUW (Mining)
    let miner = Miner::new(task.difficulty, task.memory_cost_kb);
    let result = miner.solve(state_slice);

    // 5. Return Response
    Ok(Json(ComputeResult {
        task_id: task.task_id,
        status: "success".to_string(),
        state_hash,
        proof: Some(ProofData {
            nonce: result.nonce,
            proof_hash: result.proof_hash,
        }),
    }))
}

/// Dynamically returns a list of all supported gates based on the Gate enum definition.
pub async fn get_supported_gates() -> Json<Vec<String>> {
    let gates = Gate::iter()
        .map(|gate| gate.to_string()) // This now respects the UPPERCASE rule
        .collect();

    Json(gates)
}
