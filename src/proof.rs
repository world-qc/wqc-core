//! zk-STARK proof generation and verification (`wqc-stark-engine` Mersenne31 AIR commitment).

use serde::{Deserialize, Serialize};
use wqc_stark_engine::{
    append_distribution_tail, generate_plonky3_stark_proof, verify_stark_proof_core,
    DistributionSegment, StarkContext as EngineContext,
};
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
    /// PROVER ROLE: pipes the flattened `f64` execution trace into the AIR commitment prover.
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
        distribution: Option<&DistributionSegment>,
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

        let mut proof_bytes = generate_plonky3_stark_proof(&context, execution_trace)?;
        if proof_bytes.is_empty() {
            return Err("STARK prover runtime error: generated empty proof transcript.".to_string());
        }

        if let Some(segment) = distribution {
            proof_bytes = append_distribution_tail(proof_bytes, segment);
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

    /// VALIDATOR ROLE: stateless verification via embedded trace + AIR re-evaluation.
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

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::distribution_proof::build_terminal_distribution_segment;
    use crate::engine::{
        calculate_complex_result_hash, Circuit, ComplexResult, ContractionWorkspace, Gate,
        MeasureParams,
    };
    use crate::sample::{calculate_sample_result_hash, sample_terminal_measurements, split_unitary_and_measures};
    use wqc_stark_engine::split_distribution_tail;

    #[test]
    fn h_circuit_executor_trace_proves_and_verifies() {
        let mut workspace = ContractionWorkspace::try_allocate(1, 1).expect("allocate");
        let mut circuit = Circuit::new(1);
        circuit.add(Gate::H(0)).expect("add gate");

        let trace = circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");

        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        let output_hash = calculate_complex_result_hash(&ComplexResult {
            real: inv_sqrt2,
            imag: 0.0,
        });

        let prover = StarkProver;
        let proof = prover
            .generate_proof("circuit-h", "task-h", "node-1", "0", &output_hash, &trace, None)
            .expect("proof");

        assert!(prover.verify_proof(&proof));
    }

    #[test]
    fn inactive_cnot_circuit_executor_trace_proves_and_verifies() {
        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::CNOT(0, 1)).expect("add gate");

        let trace = circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");

        let output_hash = calculate_complex_result_hash(&ComplexResult {
            real: 1.0,
            imag: 0.0,
        });

        let prover = StarkProver;
        let proof = prover
            .generate_proof("circuit-cnot0", "task-cnot0", "node-1", "0", &output_hash, &trace, None)
            .expect("proof");

        assert!(prover.verify_proof(&proof));
    }

    #[test]
    fn bell_sample_counts_proof_embeds_distribution_tail() {
        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::H(0)).expect("h");
        circuit.add(Gate::CNOT(0, 1)).expect("cnot");
        circuit
            .add(Gate::MEASURE(MeasureParams { qubit: 0, cbit: 0 }))
            .expect("m0");
        circuit
            .add(Gate::MEASURE(MeasureParams { qubit: 1, cbit: 1 }))
            .expect("m1");

        let (unitary, measures) = split_unitary_and_measures(&circuit.gates).expect("split");
        let mut unitary_circuit = Circuit::new(2);
        for gate in unitary {
            unitary_circuit.add(gate).expect("gate");
        }
        let trace = unitary_circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");

        let shots = 1024u64;
        let seed = 42u64;
        let sample = sample_terminal_measurements(
            workspace.register_mut(),
            &measures,
            2,
            shots,
            seed,
        )
        .expect("sample");
        let output_hash = calculate_sample_result_hash(&sample);
        let distribution =
            build_terminal_distribution_segment(workspace.register_mut(), &measures, 2, shots, seed)
                .expect("distribution");

        let prover = StarkProver;
        let proof = prover
            .generate_proof(
                "circuit-bell",
                "task-bell",
                "node-1",
                "0",
                &output_hash,
                &trace,
                Some(&distribution),
            )
            .expect("proof");
        assert!(prover.verify_proof(&proof));

        let proof_bytes = general_purpose::STANDARD
            .decode(proof.stark_proof_b64)
            .expect("b64");
        let tail = split_distribution_tail(&proof_bytes)
            .expect("split")
            .1
            .expect("tail present");
        assert!(!tail.is_empty());

        let mut tampered = proof_bytes.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xFF;
        let bad = Proof {
            stark_proof_b64: general_purpose::STANDARD.encode(tampered),
            ..proof
        };
        assert!(!prover.verify_proof(&bad));
    }
}
