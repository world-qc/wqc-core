use axum::{extract::Json, response::IntoResponse};
use crate::engine::{Circuit, Gate, QuantumRegister};
use crate::proof::Miner;
use sha3::{Digest, Sha3_256};
use serde::{Deserialize, Serialize};

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

pub async fn handle_compute(Json(task): Json<ComputeTask>) -> impl IntoResponse {
    // 1. Setup Register & Circuit
    let mut register = QuantumRegister::new(task.qubit_count);
    let mut circuit = Circuit::new(task.qubit_count);
    for gate in task.circuit {
        circuit.add(gate);
    }

    // 2. Execute Quantum Computation
    circuit.execute(&mut register);

    // 3. Generate State Hash (Commitment)
    let mut hasher = Sha3_256::new();
    for val in register.state.as_slice().unwrap() {
        hasher.update(val.re.to_le_bytes());
        hasher.update(val.im.to_le_bytes());
    }
    let state_hash = hex::encode(hasher.finalize());

    // 4. Run PoUW (Mining)
    let miner = Miner::new(task.difficulty, task.memory_cost_kb);
    let result = miner.solve(register.state.as_slice().unwrap());

    // 5. Return Response
    Json(ComputeResult {
        task_id: task.task_id,
        status: "success".to_string(),
        state_hash,
        proof: Some(ProofData {
            nonce: result.nonce,
            proof_hash: result.proof_hash,
        }),
    })
}
