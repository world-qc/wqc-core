//! Phase C2c: mid-circuit trajectory binding for noiseless `sample_counts`.

use std::collections::BTreeMap;

use wqc_stark_engine::{
    calculate_terminal_statevector_digest, calculate_trajectory_digest,
    canonicalize_terminal_statevector, z_marginal_from_statevector, TrajectoryMarginalWitness,
    TrajectoryMeasureEvent, TrajectorySegment, TrajectoryShotTrace,
};

use crate::distribution_proof::DistributionProofStatus;
use crate::mid_circuit::TrajectoryTrace;

/// Phase C2c — mid-circuit trajectory transcript binding (algebraic verify).
pub fn distribution_stark_status_trajectory_bound() -> DistributionProofStatus {
    DistributionProofStatus {
        bound: true,
        scheme: "trajectory_bound_v1",
    }
}

/// Phase C2c zk — per-MEASURE Z marginals proved in Plonky3 `DistributionAir`.
pub fn distribution_stark_status_trajectory_air_zk() -> DistributionProofStatus {
    DistributionProofStatus {
        bound: true,
        scheme: "trajectory_air_zk_shot_v1",
    }
}

/// Phase C2c zk composed — unitary + trajectory children in v3 compose + AggregationAir tail.
pub fn distribution_stark_status_trajectory_air_zk_composed() -> DistributionProofStatus {
    DistributionProofStatus {
        bound: true,
        scheme: "trajectory_air_zk_composed_shot_v1",
    }
}

/// Phase C2c zk linked — trajectory marginal zk + unitary v2 `unitary_link_digest` bridge.
pub fn distribution_stark_status_trajectory_air_zk_linked() -> DistributionProofStatus {
    DistributionProofStatus {
        bound: true,
        scheme: "trajectory_air_zk_linked_shot_v1",
    }
}

fn collect_marginal_witnesses(
    trace: &TrajectoryTrace,
    qubit_count: u32,
) -> (Vec<TrajectoryMarginalWitness>, String) {
    let mut by_key: BTreeMap<(String, u32), TrajectoryMarginalWitness> = BTreeMap::new();
    let mut unitary_link = String::new();

    for shot in &trace.traces {
        for (measure_index, m) in shot.measures.iter().enumerate() {
            let canonical = canonicalize_terminal_statevector(&m.pre_measure_statevector);
            let digest = calculate_terminal_statevector_digest(&canonical);
            let (reference_p0, reference_p1) =
                z_marginal_from_statevector(&canonical, m.qubit, qubit_count as usize)
                    .unwrap_or((m.p0, m.p1));
            if unitary_link.is_empty() && shot.shot_index == 0 && measure_index == 0 {
                unitary_link = digest.clone();
            }
            by_key
                .entry((digest.clone(), m.qubit as u32))
                .or_insert_with(|| TrajectoryMarginalWitness {
                    qubit: m.qubit as u32,
                    reference_p0,
                    reference_p1,
                    pre_measure_statevector: canonical,
                    pre_measure_statevector_digest: digest,
                });
        }
    }

    (by_key.into_values().collect(), unitary_link)
}

/// Builds a trajectory segment from a sampled mid-circuit trace.
pub fn build_trajectory_segment(
    trace: &TrajectoryTrace,
    qubit_count: u32,
    sample_seed: u64,
    shots: u64,
    measurement_spec_hash: String,
) -> TrajectorySegment {
    let (marginal_witnesses, unitary_link_digest) = collect_marginal_witnesses(trace, qubit_count);
    let traces: Vec<TrajectoryShotTrace> = trace
        .traces
        .iter()
        .map(|shot| TrajectoryShotTrace {
            shot_index: shot.shot_index,
            shot_seed: shot.shot_seed,
            final_outcome: shot.final_outcome.clone(),
            classical_bits: shot.classical_bits.clone(),
            measures: shot
                .measures
                .iter()
                .map(|m| {
                    let canonical = canonicalize_terminal_statevector(&m.pre_measure_statevector);
                    let digest = calculate_terminal_statevector_digest(&canonical);
                    TrajectoryMeasureEvent {
                        gate_index: m.gate_index as u32,
                        qubit: m.qubit as u32,
                        cbit: m.cbit as u32,
                        p0: m.p0,
                        p1: m.p1,
                        outcome: m.outcome,
                        pre_measure_statevector_digest: digest,
                    }
                })
                .collect(),
        })
        .collect();
    let trajectory_digest = calculate_trajectory_digest(&traces);
    TrajectorySegment {
        sample_seed,
        shots,
        measurement_spec_hash,
        trajectory_digest,
        qubit_count,
        unitary_link_digest,
        traces,
        marginal_witnesses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Gate, IfParams, MeasureParams};
    use crate::mid_circuit::sample_mid_circuit_measurements_with_trace;
    use wqc_stark_engine::{
        append_trajectory_stark_tail, append_trajectory_tail, decode_and_verify_trajectory_tail,
        format_trajectory_json, generate_trajectory_stark_bundle, segment_supports_trajectory_zk,
        TRAJ_V2_MARKER,
    };

    #[test]
    fn off_grid_rotation_marginals_match_canonical_witness() {
        use wqc_stark_engine::air::shot_sampling::evaluate_trajectory_shot_sampling_constraints;
        use wqc_stark_engine::air::trajectory::evaluate_trajectory_marginal_constraints;

        let gates = vec![
            Gate::RX(0, 0.3),
            Gate::MEASURE(MeasureParams { qubit: 0, cbit: 0 }),
            Gate::IF(IfParams {
                cbit: 0,
                value: 1,
                gate: Box::new(Gate::X(1)),
            }),
            Gate::MEASURE(MeasureParams { qubit: 1, cbit: 1 }),
        ];
        let (_, trace) =
            sample_mid_circuit_measurements_with_trace(&gates, 2, 2, 4, 11, None).expect("trace");
        let raw = &trace.traces[0].measures[0];
        let raw_p0: f64 = raw
            .pre_measure_statevector
            .iter()
            .enumerate()
            .filter(|(basis, _)| basis & 1 == 0)
            .map(|(_, (re, im))| re * re + im * im)
            .sum();
        assert!(
            (raw.p0 - raw_p0).abs() > 1e-5,
            "fixture must sit off the canonical grid (raw {raw_p0}, recorded {})",
            raw.p0
        );

        let segment = build_trajectory_segment(&trace, 2, 11, 4, "spec".into());
        assert!(evaluate_trajectory_marginal_constraints(&segment));
        assert!(evaluate_trajectory_shot_sampling_constraints(&segment));

        let proof = append_trajectory_tail(Vec::new(), &segment);
        let (_, tail) = wqc_stark_engine::split_trajectory_tail(&proof).expect("split");
        let (payload, marker) = tail.expect("tail");
        decode_and_verify_trajectory_tail(payload, marker).expect("verify");
    }

    #[test]
    fn trajectory_digest_matches_stark_engine_json() {
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
        let (_, trace) =
            sample_mid_circuit_measurements_with_trace(&gates, 2, 2, 2, 7, None).expect("trace");
        let segment = build_trajectory_segment(&trace, 2, 7, 2, "spec".into());
        let stark_traces: Vec<TrajectoryShotTrace> = segment.traces.clone();
        assert_eq!(
            segment.trajectory_digest,
            calculate_trajectory_digest(&stark_traces)
        );
        assert!(format_trajectory_json(&stark_traces).contains(r#""trajectory""#));
        assert_eq!(segment.traces.len(), 2);
        assert!(segment_supports_trajectory_zk(&segment));
        assert!(!segment.unitary_link_digest.is_empty());
        assert!(!segment.marginal_witnesses.is_empty());
    }

    #[test]
    #[ignore = "slow Plonky3 prove; local only — not run in CI"]
    fn if_demo_512_shots_marginal_constraints_and_zk_roundtrip() {
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
        let (_, trace) =
            sample_mid_circuit_measurements_with_trace(&gates, 2, 2, 512, 42, None).expect("trace");
        let segment = build_trajectory_segment(&trace, 2, 42, 512, "spec".into());
        assert!(segment_supports_trajectory_zk(&segment));

        let payload = {
            let proof = append_trajectory_tail(Vec::new(), &segment);
            let (_, tail) = wqc_stark_engine::split_trajectory_tail(&proof).expect("split");
            let (payload, _) = tail.expect("tail");
            payload.to_vec()
        };
        let decoded =
            decode_and_verify_trajectory_tail(&payload, TRAJ_V2_MARKER).expect("decode segment");
        assert_eq!(decoded.trajectory_digest, segment.trajectory_digest);

        let bundle = generate_trajectory_stark_bundle("sub-traj", &segment, "").expect("zk prove");
        let mut proof = append_trajectory_tail(Vec::new(), &segment);
        proof = append_trajectory_stark_tail(proof, &bundle);
        let (_, tail) = wqc_stark_engine::split_trajectory_tail(&proof).expect("split");
        let (payload, marker) = tail.expect("tail");
        decode_and_verify_trajectory_tail(payload, marker).expect("verify after tail wrap");
    }
}
