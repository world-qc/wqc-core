//! Phase C2c: mid-circuit trajectory binding for noiseless `sample_counts`.

use wqc_stark_engine::{
    calculate_trajectory_digest, TrajectoryMeasureEvent, TrajectorySegment, TrajectoryShotTrace,
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

/// Builds a trajectory segment from a sampled mid-circuit trace.
pub fn build_trajectory_segment(
    trace: &TrajectoryTrace,
    sample_seed: u64,
    shots: u64,
    measurement_spec_hash: String,
) -> TrajectorySegment {
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
                .map(|m| TrajectoryMeasureEvent {
                    gate_index: m.gate_index as u32,
                    qubit: m.qubit as u32,
                    cbit: m.cbit as u32,
                    p0: m.p0,
                    p1: m.p1,
                    outcome: m.outcome,
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
        traces,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Gate, IfParams, MeasureParams};
    use crate::mid_circuit::sample_mid_circuit_measurements_with_trace;
    use wqc_stark_engine::format_trajectory_json;

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
        let segment = build_trajectory_segment(&trace, 7, 2, "spec".into());
        let stark_traces: Vec<TrajectoryShotTrace> = segment.traces.clone();
        assert_eq!(
            segment.trajectory_digest,
            calculate_trajectory_digest(&stark_traces)
        );
        assert!(format_trajectory_json(&stark_traces).contains(r#""trajectory""#));
        assert_eq!(segment.traces.len(), 2);
    }
}
