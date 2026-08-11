//! zk-STARK proof generation and verification (`wqc-stark-engine` Mersenne31 AIR commitment).

use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use wqc_stark_engine::{
    append_born_stark_tail, append_distribution_tail, append_trajectory_stark_tail,
    append_trajectory_tail, compose_unitary_born_leaf, compose_unitary_trajectory_leaf,
    generate_born_stark_proof, generate_plonky3_stark_proof, generate_trajectory_stark_bundle,
    segment_supports_born_zk, segment_supports_trajectory_zk, verify_stark_proof_core,
    BornStarkContext, DistributionSegment, StarkContext as EngineContext, TrajectorySegment,
};

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
    /// SHA3-256 hex of canonical measurement spec JSON (C2c STARK PI); empty when unbound.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub measurement_spec_hash: String,
    /// Orchestrator security tier for FRI query selection; empty → default (40).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub security_level: String,
}

/// Returns `hash` when it is a 64-char ASCII hex digest; otherwise empty (legacy test fixtures).
fn stark_pi_measurement_spec_hash(hash: &str) -> &str {
    if hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        hash
    } else {
        ""
    }
}

pub struct StarkProver;

impl StarkProver {
    /// PROVER ROLE: pipes the flattened `f64` execution trace into the AIR commitment prover.
    ///
    /// Binds `circuit_id`, `slice_id`, `task_id`, `node_id`, and the contracted scalar hash
    /// into the outward-facing `PublicInputs` envelope.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_proof(
        &self,
        circuit_id: &str,
        task_id: &str,
        node_id: &str,
        slice_id: &str,
        output_hash: &str,
        execution_trace: &[f64],
        distribution: Option<&DistributionSegment>,
        trajectory: Option<&TrajectorySegment>,
        security_level: &str,
    ) -> Result<Proof, String> {
        if execution_trace.is_empty() {
            return Err(
                "Cannot generate STARK proof: execution trace stream is empty.".to_string(),
            );
        }

        let sv_digest = distribution
            .and_then(|seg| seg.born_binding.as_ref())
            .map(|b| b.terminal_statevector_digest.as_str())
            .unwrap_or("");
        let traj_link = trajectory
            .map(|seg| seg.unitary_link_digest.as_str())
            .filter(|digest| !digest.is_empty())
            .unwrap_or("");
        let measurement_spec_hash = stark_pi_measurement_spec_hash(
            distribution
                .map(|seg| seg.measurement_spec_hash.as_str())
                .or_else(|| trajectory.map(|seg| seg.measurement_spec_hash.as_str()))
                .unwrap_or(""),
        );

        let context = EngineContext {
            circuit_id,
            sub_task_id: task_id,
            node_id,
            slice_id,
            output_hash,
            terminal_statevector_digest: if !traj_link.is_empty() {
                traj_link
            } else {
                sv_digest
            },
            measurement_spec_hash,
            security_level,
        };

        let mut proof_bytes = generate_plonky3_stark_proof(&context, execution_trace)?;
        if proof_bytes.is_empty() {
            return Err(
                "STARK prover runtime error: generated empty proof transcript.".to_string(),
            );
        }

        if let Some(segment) = distribution {
            if segment_supports_born_zk(segment) && !sv_digest.is_empty() {
                let born_ctx = BornStarkContext {
                    sub_task_id: task_id,
                    probability_digest: &segment.probability_digest,
                    terminal_statevector_digest: sv_digest,
                    security_level,
                };
                let born_inner = generate_born_stark_proof(&born_ctx, segment)?;
                proof_bytes =
                    compose_unitary_born_leaf(&context, &proof_bytes, segment, &born_inner)?;
            } else {
                proof_bytes = append_distribution_tail(proof_bytes, segment);
                if segment_supports_born_zk(segment) {
                    let born_ctx = BornStarkContext {
                        sub_task_id: task_id,
                        probability_digest: &segment.probability_digest,
                        terminal_statevector_digest: sv_digest,
                        security_level,
                    };
                    let born_proof = generate_born_stark_proof(&born_ctx, segment)?;
                    proof_bytes = append_born_stark_tail(proof_bytes, &born_proof);
                }
            }
        }

        if let Some(segment) = trajectory {
            if segment_supports_trajectory_zk(segment) && !traj_link.is_empty() {
                let bundle = generate_trajectory_stark_bundle(task_id, segment, security_level)?;
                proof_bytes =
                    compose_unitary_trajectory_leaf(&context, &proof_bytes, segment, &bundle)?;
            } else {
                proof_bytes = append_trajectory_tail(proof_bytes, segment);
                if segment_supports_trajectory_zk(segment) {
                    let bundle =
                        generate_trajectory_stark_bundle(task_id, segment, security_level)?;
                    proof_bytes = append_trajectory_stark_tail(proof_bytes, &bundle);
                }
            }
        }

        let stark_proof_b64 = general_purpose::STANDARD.encode(&proof_bytes);

        Ok(Proof {
            public_inputs: PublicInputs {
                circuit_id: circuit_id.to_string(),
                sub_task_id: task_id.to_string(),
                node_id: node_id.to_string(),
                slice_id: slice_id.to_string(),
                output_result_hash: output_hash.to_string(),
                measurement_spec_hash: measurement_spec_hash.to_string(),
                security_level: security_level.to_string(),
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
                    terminal_statevector_digest: "",
                    measurement_spec_hash: &proof.public_inputs.measurement_spec_hash,
                    security_level: &proof.public_inputs.security_level,
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
    use crate::sample::{
        calculate_sample_result_hash, sample_terminal_measurements, split_unitary_and_measures,
    };
    use base64::engine::general_purpose;
    use base64::Engine;

    #[test]
    #[ignore = "slow Plonky3 prove; local only — not run in CI"]
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
            .generate_proof(
                "circuit-h",
                "task-h",
                "node-1",
                "0",
                &output_hash,
                &trace,
                None,
                None,
                "",
            )
            .expect("proof");

        assert!(prover.verify_proof(&proof));
    }

    #[test]
    #[ignore = "slow Plonky3 prove; local only — not run in CI"]
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
            .generate_proof(
                "circuit-cnot0",
                "task-cnot0",
                "node-1",
                "0",
                &output_hash,
                &trace,
                None,
                None,
                "",
            )
            .expect("proof");

        assert!(prover.verify_proof(&proof));
    }

    #[test]
    #[ignore = "slow Plonky3 prove; local only — not run in CI"]
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
        let sample =
            sample_terminal_measurements(workspace.register_mut(), &measures, 2, shots, seed)
                .expect("sample");
        let output_hash = calculate_sample_result_hash(&sample);
        let distribution = build_terminal_distribution_segment(
            workspace.register_mut(),
            &measures,
            2,
            shots,
            seed,
        )
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
                None,
                "",
            )
            .expect("proof");
        assert!(prover.verify_proof(&proof));
        assert_eq!(
            proof.public_inputs.measurement_spec_hash,
            distribution.measurement_spec_hash
        );

        let proof_bytes = general_purpose::STANDARD
            .decode(proof.stark_proof_b64.as_bytes())
            .expect("b64");
        assert!(wqc_stark_engine::is_unitary_born_leaf_compose(&proof_bytes));

        let mut tampered = proof_bytes.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xFF;
        let bad = Proof {
            public_inputs: proof.public_inputs.clone(),
            stark_proof_b64: general_purpose::STANDARD.encode(tampered),
        };
        assert!(!prover.verify_proof(&bad));
    }

    #[test]
    #[ignore = "slow Plonky3 prove; local only — not run in CI"]
    fn mid_circuit_if_compose_proves_and_verifies() {
        use crate::engine::{Gate, IfParams};
        use crate::mid_circuit::{
            extract_unitary_gates_for_proof, sample_mid_circuit_measurements_with_trace,
        };
        use crate::trajectory_proof::build_trajectory_segment;
        use wqc_stark_engine::is_unitary_trajectory_leaf_compose;

        let gates = vec![
            Gate::H(0),
            Gate::MEASURE(MeasureParams { qubit: 0, cbit: 0 }),
            Gate::IF(IfParams {
                cbit: 0,
                value: 1,
                gate: Box::new(Gate::X(1)),
            }),
            Gate::MEASURE(MeasureParams { qubit: 1, cbit: 1 }),
        ];

        let (sample, trace) =
            sample_mid_circuit_measurements_with_trace(&gates, 2, 2, 16, 42, None)
                .expect("trajectory sample");
        let output_hash = calculate_sample_result_hash(&sample);
        let trajectory = build_trajectory_segment(&trace, 2, 42, 16, "spec-hash".into());

        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let unitary_gates = extract_unitary_gates_for_proof(&gates);
        let mut unitary_circuit = Circuit::new(2);
        for gate in unitary_gates {
            unitary_circuit.add(gate).expect("unitary gate");
        }
        let execution_trace = unitary_circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary trace");

        let prover = StarkProver;
        let proof = prover
            .generate_proof(
                "circuit-if",
                "sub-if-compose",
                "node-1",
                "0",
                &output_hash,
                &execution_trace,
                None,
                Some(&trajectory),
                "",
            )
            .expect("composed proof");

        assert!(prover.verify_proof(&proof));

        let proof_bytes = general_purpose::STANDARD
            .decode(&proof.stark_proof_b64)
            .expect("b64");
        assert!(is_unitary_trajectory_leaf_compose(&proof_bytes));
    }

    #[test]
    #[ignore = "slow Plonky3 prove; local only — not run in CI"]
    fn mid_circuit_compose_rejects_tampered_transcript() {
        use crate::engine::{Gate, IfParams};
        use crate::mid_circuit::{
            extract_unitary_gates_for_proof, sample_mid_circuit_measurements_with_trace,
        };
        use crate::trajectory_proof::build_trajectory_segment;

        let gates = vec![
            Gate::H(0),
            Gate::MEASURE(MeasureParams { qubit: 0, cbit: 0 }),
            Gate::IF(IfParams {
                cbit: 0,
                value: 1,
                gate: Box::new(Gate::X(1)),
            }),
            Gate::MEASURE(MeasureParams { qubit: 1, cbit: 1 }),
        ];

        let (sample, trace) =
            sample_mid_circuit_measurements_with_trace(&gates, 2, 2, 8, 7, None).expect("sample");
        let output_hash = calculate_sample_result_hash(&sample);
        let trajectory = build_trajectory_segment(&trace, 2, 7, 8, "spec".into());

        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut unitary_circuit = Circuit::new(2);
        for gate in extract_unitary_gates_for_proof(&gates) {
            unitary_circuit.add(gate).expect("gate");
        }
        let execution_trace = unitary_circuit
            .execute_with_trace(workspace.register_mut())
            .expect("trace");

        let prover = StarkProver;
        let proof = prover
            .generate_proof(
                "circuit-if",
                "sub-if-tamper",
                "node-1",
                "0",
                &output_hash,
                &execution_trace,
                None,
                Some(&trajectory),
                "",
            )
            .expect("proof");

        let mut proof_bytes = general_purpose::STANDARD
            .decode(proof.stark_proof_b64.clone())
            .expect("b64");
        let last = proof_bytes.len() - 1;
        proof_bytes[last] ^= 0xFF;
        let tampered = Proof {
            stark_proof_b64: general_purpose::STANDARD.encode(proof_bytes),
            ..proof.clone()
        };
        assert!(!prover.verify_proof(&tampered));

        let mut wrong_meta = proof;
        wrong_meta.public_inputs.sub_task_id = "wrong-subtask".into();
        assert!(!prover.verify_proof(&wrong_meta));
    }
}
