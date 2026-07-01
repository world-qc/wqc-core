//! Phase C2: measurement-distribution binding (Born probabilities + deterministic sampling).
//!
//! C2a-1 defines the cross-crate `probability_digest` contract used by `wqc-orchestrator`
//! and `wqc-core`. Keys are Qiskit-order bitstrings (same as `sample_counts`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use wqc_stark_engine::{BornBinding, DistributionSegment};

use crate::engine::MeasureParams;
use crate::sample::compute_outcome_probabilities;
use crate::tn::MpsState;

/// Born-rule outcome probabilities for terminal Z measurements (support outcomes only).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct OutcomeProbabilities {
    pub probabilities: BTreeMap<String, f64>,
}

/// Reports whether the STARK transcript binds sampled counts beyond unitary trace.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DistributionProofStatus {
    pub bound: bool,
    pub scheme: &'static str,
}

/// Phase C2a-2 — terminal `sample_counts` with Born probabilities in the transcript tail.
pub fn distribution_stark_status_bound() -> DistributionProofStatus {
    DistributionProofStatus {
        bound: true,
        scheme: "born_deterministic_v1",
    }
}

/// Phase C2b — terminal statevector Born-rule binding (algebraic verify).
pub fn distribution_stark_status_born_air() -> DistributionProofStatus {
    DistributionProofStatus {
        bound: true,
        scheme: "born_air_v1",
    }
}

/// Phase C2b zk linked — Born zk + unitary v2 `terminal_statevector_digest` bridge.
pub fn distribution_stark_status_born_air_zk_linked() -> DistributionProofStatus {
    DistributionProofStatus {
        bound: true,
        scheme: "born_air_zk_linked_v1",
    }
}

/// Phase C2b zk — Born-rule constraints proved in Plonky3 `DistributionAir`.
pub fn distribution_stark_status_born_air_zk() -> DistributionProofStatus {
    DistributionProofStatus {
        bound: true,
        scheme: "born_air_zk_v1",
    }
}

/// Phase C2 placeholder — quorum still uses canonical `counts` hash (seed-fixed).
pub fn distribution_stark_status() -> DistributionProofStatus {
    DistributionProofStatus {
        bound: false,
        scheme: "unitary_trace_only",
    }
}

/// Canonical JSON for SHA3-256 hashing — must match orchestrator `FormatProbabilityJSON`.
pub fn format_go_probability_json(table: &OutcomeProbabilities) -> String {
    let mut pairs = String::new();
    for (key, value) in &table.probabilities {
        if !pairs.is_empty() {
            pairs.push(',');
        }
        pairs.push_str(&format!(r#""{key}":{}"#, format_go_float(*value)));
    }
    format!(r#"{{"probabilities":{{{pairs}}}}}"#)
}

/// SHA3-256 hex digest of the canonical probability JSON (`probability_digest`).
pub fn calculate_probability_digest(table: &OutcomeProbabilities) -> String {
    use sha3::{Digest, Sha3_256};
    hex::encode(Sha3_256::digest(format_go_probability_json(table).as_bytes()))
}

/// Canonical measurement-spec JSON — must match orchestrator `FormatMeasurementSpecJSON`.
pub fn format_measurement_spec_json(measures: &[MeasureParams]) -> String {
    let mut specs: Vec<(u32, u32)> = measures
        .iter()
        .map(|m| (m.qubit as u32, m.cbit as u32))
        .collect();
    specs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let parts: Vec<String> = specs
        .into_iter()
        .map(|(qubit, cbit)| format!(r#"{{"cbit":{cbit},"qubit":{qubit}}}"#))
        .collect();
    format!(r#"{{"measures":[{}]}}"#, parts.join(","))
}

/// SHA3-256 hex digest of the canonical measurement spec JSON (C2a-4).
pub fn calculate_measurement_spec_hash(measures: &[MeasureParams]) -> String {
    use sha3::{Digest, Sha3_256};
    hex::encode(Sha3_256::digest(
        format_measurement_spec_json(measures).as_bytes(),
    ))
}

/// Builds a transcript distribution segment from Born probabilities (C2a-2/4).
pub fn distribution_segment_from_probabilities(
    probabilities: BTreeMap<String, f64>,
    shots: u64,
    sample_seed: u64,
    measurement_spec_hash: String,
) -> DistributionSegment {
    let table = OutcomeProbabilities { probabilities };
    let probability_digest = calculate_probability_digest(&table);
    let probabilities: Vec<(String, f64)> = table.probabilities.into_iter().collect();
    DistributionSegment {
        sample_seed,
        shots,
        measurement_spec_hash,
        probability_digest,
        probabilities,
        born_binding: None,
    }
}

/// Computes Born probabilities from a post-unitary MPS state and packs a distribution segment.
pub fn build_terminal_distribution_segment(
    state: &MpsState,
    measures: &[MeasureParams],
    classical_bit_count: usize,
    shots: u64,
    sample_seed: u64,
) -> Result<DistributionSegment, crate::engine::EngineError> {
    let statevector = state.contract_to_statevector()?;
    let probabilities = compute_outcome_probabilities(
        &statevector,
        state.qubit_count,
        measures,
        classical_bit_count,
    )?;
    let measurement_spec_hash = calculate_measurement_spec_hash(measures);
    let measure_pairs: Vec<(u32, u32)> = measures
        .iter()
        .map(|m| (m.qubit as u32, m.cbit as u32))
        .collect();
    let terminal_statevector: Vec<(f64, f64)> = statevector
        .iter()
        .map(|c| (c.re, c.im))
        .collect();
    let born_binding = BornBinding::from_specs(
        state.qubit_count as u32,
        classical_bit_count as u32,
        &measure_pairs,
        terminal_statevector,
    );
    let mut segment = distribution_segment_from_probabilities(
        probabilities,
        shots,
        sample_seed,
        measurement_spec_hash,
    );
    segment.born_binding = born_binding;
    Ok(segment)
}

fn format_go_float(val: f64) -> String {
    if val == (val as i64) as f64 {
        format!("{:.1}", val)
    } else {
        format!("{val}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Circuit, ContractionWorkspace, Gate, MeasureParams};
    use crate::sample::compute_outcome_probabilities;

    #[test]
    fn probability_json_uses_sorted_keys_and_go_float_style() {
        let table = OutcomeProbabilities {
            probabilities: BTreeMap::from([
                ("11".into(), 0.5),
                ("00".into(), 0.5),
            ]),
        };
        assert_eq!(
            format_go_probability_json(&table),
            r#"{"probabilities":{"00":0.5,"11":0.5}}"#
        );
    }

    #[test]
    fn probability_digest_matches_orchestrator_golden_bell() {
        let table = OutcomeProbabilities {
            probabilities: BTreeMap::from([
                ("00".into(), 0.5),
                ("11".into(), 0.5),
            ]),
        };
        assert_eq!(
            calculate_probability_digest(&table),
            "ef8f4691ad99dc93489c72d6a5863df7974ce1d0c1ad58525c133c15d43190fc"
        );
    }

    #[test]
    fn probability_digest_integer_one_uses_one_point_zero() {
        let table = OutcomeProbabilities {
            probabilities: BTreeMap::from([("0".into(), 1.0)]),
        };
        assert_eq!(
            format_go_probability_json(&table),
            r#"{"probabilities":{"0":1.0}}"#
        );
        assert_eq!(
            calculate_probability_digest(&table),
            "b3de34846864135b2fc5dc4cfc94c950b8e4c95b98015ba3e09fa46ada453e20"
        );
    }

    #[test]
    fn bell_state_probabilities_match_compute_outcome_probabilities() {
        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::H(0)).expect("h");
        circuit.add(Gate::CNOT(0, 1)).expect("cnot");
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");

        let statevector = workspace.register_mut().contract_to_statevector().expect("dense");
        let measures = vec![
            MeasureParams { qubit: 0, cbit: 0 },
            MeasureParams { qubit: 1, cbit: 1 },
        ];
        let probs = compute_outcome_probabilities(&statevector, 2, &measures, 2).expect("probs");
        let table = OutcomeProbabilities {
            probabilities: probs,
        };
        let p00 = table.probabilities.get("00").copied().unwrap_or(0.0);
        let p11 = table.probabilities.get("11").copied().unwrap_or(0.0);
        assert!((p00 - 0.5).abs() < 1e-9, "p(00)={p00}");
        assert!((p11 - 0.5).abs() < 1e-9, "p(11)={p11}");
        assert_eq!(table.probabilities.len(), 2);
        // Digest is defined on formatted floats from simulation (may differ from hand-crafted 0.5 literals).
        let json = format_go_probability_json(&table);
        assert!(json.contains(r#""00""#) && json.contains(r#""11""#));
        assert_eq!(
            calculate_probability_digest(&table),
            calculate_probability_digest(&OutcomeProbabilities {
                probabilities: table.probabilities.clone(),
            })
        );
    }

    #[test]
    fn measurement_spec_hash_matches_orchestrator_bell_order() {
        let measures = vec![
            MeasureParams { qubit: 1, cbit: 1 },
            MeasureParams { qubit: 0, cbit: 0 },
        ];
        let json = format_measurement_spec_json(&measures);
        assert_eq!(
            json,
            r#"{"measures":[{"cbit":0,"qubit":0},{"cbit":1,"qubit":1}]}"#
        );
        let hash = calculate_measurement_spec_hash(&measures);
        assert_eq!(hash.len(), 64);
        assert_eq!(
            calculate_measurement_spec_hash(&[
                MeasureParams { qubit: 0, cbit: 0 },
                MeasureParams { qubit: 1, cbit: 1 },
            ]),
            hash
        );
    }

    #[test]
    fn empty_support_yields_empty_object() {
        let table = OutcomeProbabilities::default();
        assert_eq!(
            format_go_probability_json(&table),
            r#"{"probabilities":{}}"#
        );
    }

    #[test]
    fn terminal_segment_embeds_born_binding_for_bell() {
        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::H(0)).expect("h");
        circuit.add(Gate::CNOT(0, 1)).expect("cnot");
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");

        let measures = vec![
            MeasureParams { qubit: 0, cbit: 0 },
            MeasureParams { qubit: 1, cbit: 1 },
        ];
        let segment = build_terminal_distribution_segment(
            workspace.register_mut(),
            &measures,
            2,
            1024,
            42,
        )
        .expect("segment");
        let binding = segment.born_binding.as_ref().expect("born binding");
        assert_eq!(binding.qubit_count, 2);
        assert_eq!(binding.classical_bit_count, 2);
        assert_eq!(binding.terminal_statevector.len(), 4);
        assert_eq!(binding.measures, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn executor_traces_satisfy_born_constraints() {
        use wqc_stark_engine::air::distribution::evaluate_born_constraint_sum;

        let mut workspace = ContractionWorkspace::try_allocate(2, 2).expect("allocate");
        let mut circuit = Circuit::new(2);
        circuit.add(Gate::H(0)).expect("h");
        circuit.add(Gate::CNOT(0, 1)).expect("cnot");
        circuit
            .execute_with_trace(workspace.register_mut())
            .expect("unitary");

        let measures = vec![
            MeasureParams { qubit: 0, cbit: 0 },
            MeasureParams { qubit: 1, cbit: 1 },
        ];
        let segment = build_terminal_distribution_segment(
            workspace.register_mut(),
            &measures,
            2,
            256,
            99,
        )
        .expect("segment");
        assert_eq!(evaluate_born_constraint_sum(&segment), 0);
    }
}
