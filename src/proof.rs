use serde::{Deserialize, Serialize};
use wqc_stark_engine::{generate_stark_proof, verify_stark_proof_core, StarkContext as EngineContext};
use base64::{Engine as _, engine::general_purpose};

/// Vision: 'The proof is the anchor of trust in a decentralized computer.'
/// STARK-based PoUW proof token payload.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Proof {
    pub public_inputs: PublicInputs,
    pub stark_proof_b64: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PublicInputs {
    pub circuit_id: String,
    pub sub_task_id: String,
    pub node_id: String,
    pub output_result_hash: String,
}

pub struct StarkProver;

impl StarkProver {
    /// PROVER ROLE: Pipes the flattened f64 execution trace stream directly
    /// into the Plonky3 Mersenne31 AIR constraint calculator.
    pub fn generate_proof(
        &self,
        circuit_id: &str,
        task_id: &str,
        node_id: &str,
        output_hash: &str,
        execution_trace: &[f64],
    ) -> Result<Proof, String> {
        if execution_trace.is_empty() {
            return Err("Cannot generate STARK proof: Execution trace stream is empty.".to_string());
        }

        // Bridge the task credentials using the real Engine's structural layout
        let context = EngineContext {
            circuit_id,
            sub_task_id: task_id,
            node_id,
            output_hash,
        };

        // Execution of the real Plonky3 multi-poly row transformation
        let proof_bytes = generate_stark_proof(&context, execution_trace);
        if proof_bytes.is_empty() {
            return Err("STARK Prover runtime error: Generated empty proof transcript.".to_string());
        }

        let stark_proof_b64: String = general_purpose::STANDARD.encode(&proof_bytes);

        Ok(Proof {
            public_inputs: PublicInputs {
                circuit_id: circuit_id.to_string(),
                sub_task_id: task_id.to_string(),
                node_id: node_id.to_string(),
                output_result_hash: output_hash.to_string(),
            },
            stark_proof_b64,
        })
    }

    /// VALIDATOR ROLE: Executes stateless validation of the remote node's proof transcript.
    /// This avoids execution trace tracking and handles validation instantly in O(1).
    pub fn verify_proof(
        &self,
        proof: &Proof,
    ) -> bool {
        let fields = [
            &proof.public_inputs.circuit_id,
            &proof.public_inputs.sub_task_id,
            &proof.public_inputs.node_id,
            &proof.public_inputs.output_result_hash,
            &proof.stark_proof_b64,
        ];
        if fields.iter().all(|s| s.is_empty()) {
            return false;
        }

        let mut result = false;
        match general_purpose::STANDARD.decode(proof.stark_proof_b64.clone()) {
            Ok(proof_bytes) => {
                // Reconstruct context mapping to mirror Orchestrator verification pipelines
                let context = EngineContext {
                    circuit_id: &proof.public_inputs.circuit_id,
                    sub_task_id: &proof.public_inputs.sub_task_id,
                    node_id: &proof.public_inputs.node_id,
                    output_hash: &proof.public_inputs.output_result_hash,
                };
                // Call the real stateless AIR evaluation pathway from wqc-stark-engine
                result = verify_stark_proof_core(&context, &proof_bytes);
            }
            Err(e) => {
                println!("Decode failed: Invalid Base64: {}", e);
            }
        }
        return result;
    }
}
