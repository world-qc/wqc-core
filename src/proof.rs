//! zk-STARK proof generation and verification (Plonky3 / Mersenne31 via `wqc-stark-engine`).

use serde::{Deserialize, Serialize};
use wqc_stark_engine::{generate_stark_proof, verify_stark_proof_core, StarkContext as EngineContext};
use base64::{engine::general_purpose, Engine as _};

/// Vision: the proof is the anchor of trust in a decentralized computer.
/// STARK-based PoUW proof token returned to wqc-node / the orchestrator.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Proof {
    pub public_inputs: PublicInputs,
    pub stark_proof_b64: String,
}

/// Immutable public inputs to which the proof transcript is cryptographically bound.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PublicInputs {
    pub circuit_id: String,
    pub sub_task_id: String,
    pub node_id: String,
    /// Binary slice path; must match the orchestrator sub-task metadata.
    pub slice_id: String,
    /// SHA3-256 of the JSON-encoded `ComplexResult` (orchestrator consensus key).
    pub output_result_hash: String,
}

pub struct StarkProver;

impl StarkProver {
    /// PROVER ROLE: pipes the flattened `f64` execution trace into the Plonky3 AIR prover.
    ///
    /// Binds `circuit_id`, `slice_id`, `task_id`, `node_id`, and the contracted scalar hash
    /// into the outward-facing `PublicInputs` envelope.
    pub fn generate_proof(
        &self,
        circuit_id: &str,
        task_id: &str,
        node_id: &str,
        slice_id: &str,
        output_hash: &str,
        execution_trace: &[f64],
    ) -> Result<Proof, String> {
        if execution_trace.is_empty() {
            return Err("Cannot generate STARK proof: execution trace stream is empty.".to_string());
        }

        let context = EngineContext {
            circuit_id,
            sub_task_id: task_id,
            node_id,
            slice_id,
            output_hash,
        };

        // Run the real Plonky3 multi-polynomial row transformation.
        let proof_bytes = generate_stark_proof(&context, execution_trace);
        if proof_bytes.is_empty() {
            return Err("STARK prover runtime error: generated empty proof transcript.".to_string());
        }

        let stark_proof_b64 = general_purpose::STANDARD.encode(&proof_bytes);

        Ok(Proof {
            public_inputs: PublicInputs {
                circuit_id: circuit_id.to_string(),
                sub_task_id: task_id.to_string(),
                node_id: node_id.to_string(),
                slice_id: slice_id.to_string(),
                output_result_hash: output_hash.to_string(),
            },
            stark_proof_b64,
        })
    }

    /// VALIDATOR ROLE: stateless verification without re-executing contraction (O(1) transcript check).
    pub fn verify_proof(&self, proof: &Proof) -> bool {
        let fields = [
            &proof.public_inputs.circuit_id,
            &proof.public_inputs.sub_task_id,
            &proof.public_inputs.node_id,
            &proof.public_inputs.slice_id,
            &proof.public_inputs.output_result_hash,
            &proof.stark_proof_b64,
        ];
        if fields.iter().any(|s| s.is_empty()) {
            return false;
        }

        match general_purpose::STANDARD.decode(proof.stark_proof_b64.clone()) {
            Ok(proof_bytes) => {
                // Reconstruct context mapping to mirror orchestrator verification pipelines.
                let context = EngineContext {
                    circuit_id: &proof.public_inputs.circuit_id,
                    sub_task_id: &proof.public_inputs.sub_task_id,
                    node_id: &proof.public_inputs.node_id,
                    slice_id: &proof.public_inputs.slice_id,
                    output_hash: &proof.public_inputs.output_result_hash,
                };
                verify_stark_proof_core(&context, &proof_bytes)
            }
            Err(e) => {
                println!("Decode failed: invalid Base64: {}", e);
                false
            }
        }
    }
}
