//! v2 (sv-stark) leaves: proves a `sample_counts` sub-task with `wqc-sv-stark` when the
//! orchestrator dispatches it with `sv_leaf_rows`.
//!
//! The leaf proves that the circuit, applied to |0…0⟩, ends in a state whose digest it
//! exports, and that the counts were drawn from that state under the dispatch's seed. The
//! counts come from the leaf's own integer sampler, not from [`crate::sample`], so every v2
//! node reports the same counts for a dispatch and v1 and v2 counts differ.

use std::sync::OnceLock;
use std::time::Instant;

use axum::http::StatusCode;
use base64::{engine::general_purpose, Engine as _};
use wqc_sv_stark::born::Sampling;
use wqc_sv_stark::leaf::{self, Dispatch};
use wqc_sv_stark::{wire, SecurityLevel};

use crate::api::{ComputeResponse, ComputeTask, WorkReport};
use crate::engine::ComplexResult;
use crate::proof::{Proof, PublicInputs};
use crate::sample::{calculate_sample_result_hash, OutputMode, SampleResult};

/// `WorkReport.proof_scope` of a v2 leaf, as the orchestrator records it.
pub const PROOF_SCOPE: &str = "sv_stark_v2";

/// Memory a leaf takes to prove, about `SV_BASE_BYTES + SV_BYTES_PER_ROW × rows` (peak RSS
/// at n = 12 to 16, 20 to 100 rounding steps, on e2-standard-8 with every core proving).
/// A node sizes its `sv_max_rows` from it.
pub const SV_BYTES_PER_ROW: u64 = 4_600;
pub const SV_BASE_BYTES: u64 = 300_000_000;

type Failure = (StatusCode, String);

fn bad_request(m: impl Into<String>) -> Failure {
    (StatusCode::BAD_REQUEST, m.into())
}

fn internal(m: impl Into<String>) -> Failure {
    (StatusCode::INTERNAL_SERVER_ERROR, m.into())
}

/// Whether this core proves v2 leaves (`WQC_SV_STARK_V2`, default on). Off, it reports no
/// rate, so its node does not advertise them.
fn enabled() -> bool {
    !matches!(
        std::env::var("WQC_SV_STARK_V2").as_deref().map(str::trim),
        Ok("0" | "false" | "off")
    )
}

/// The pool v2 leaves prove on: every core (`WQC_SV_THREADS` to cap it). The global pool is
/// pinned to one thread for v1's leaf PCS, which would make a v2 leaf several times slower
/// than the time and memory its `sv_max_rows` were sized for.
fn pool() -> Result<&'static rayon::ThreadPool, Failure> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::env::var("WQC_SV_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            // Not 0: rayon would then read RAYON_NUM_THREADS, which main pins to 1.
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("sv-stark-{i}"))
            .build()
            .map_err(|e| eprintln!("sv-stark thread pool: {e}"))
            .ok()
    })
    .as_ref()
    .ok_or_else(|| internal("sv-stark thread pool unavailable"))
}

/// Proves `task` as a v2 sampling leaf of `rows` gate rows.
pub fn compute(task: ComputeTask, rows: u64) -> Result<ComputeResponse, Failure> {
    if !enabled() {
        return Err(bad_request("this core does not prove v2 leaves"));
    }
    let checked = Checked::from_task(&task, rows)?;
    let start = Instant::now();
    let (lp, counts) = pool()?.install(|| {
        let leaf = leaf::build_sampling(
            checked.level,
            task.qubit_count,
            checked.circuit.gates.clone(),
            checked.sampling.clone(),
        )
        .map_err(|e| bad_request(format!("v2 leaf: {e}")))?;
        crate::cancel::check().map_err(|e| internal(e.to_string()))?;
        let ctx = Dispatch {
            circuit_id: &task.circuit_id,
            sub_task_id: &task.task_id,
            node_id: &task.node_id,
            slice_id: &task.slice_id,
        }
        .context();
        leaf.prove_sampling(ctx, checked.seed)
            .map_err(|e| internal(format!("v2 leaf: {e}")))
    })?;
    let prove_wall_ms = start.elapsed().as_millis() as u64;

    let sv_digest: Vec<u64> = lp.statement[4..leaf::STATEMENT_LEN]
        .iter()
        .map(p3_field_sv::PrimeField64::as_canonical_u64)
        .collect();
    let sample = sample_result(&counts, checked.sampling.cbits);
    let proof_bytes = leaf::encode_proof(&lp.proof);
    let proof = Proof {
        public_inputs: PublicInputs {
            circuit_id: task.circuit_id.clone(),
            sub_task_id: task.task_id.clone(),
            node_id: task.node_id.clone(),
            slice_id: task.slice_id.clone(),
            output_result_hash: calculate_sample_result_hash(&sample),
            measurement_spec_hash: String::new(),
            security_level: task.security_level.clone(),
            sv_digest,
        },
        stark_proof_b64: general_purpose::STANDARD.encode(&proof_bytes),
    };
    println!(
        "v2 leaf proved for task {}: {} rows, {} ms, {} proof bytes",
        task.task_id,
        rows,
        prove_wall_ms,
        proof_bytes.len()
    );
    Ok(ComputeResponse {
        task_id: task.task_id,
        status: "success".to_string(),
        result_type: "sample_counts".to_string(),
        complex_result: ComplexResult {
            real: 0.0,
            imag: 0.0,
        },
        sample_result: Some(sample),
        expectation_result: None,
        work_report: WorkReport {
            trace_rows: rows,
            gate_count: checked.circuit.gates.len() as u32,
            compute_wall_ms: 0,
            prove_wall_ms,
            proof_bytes: proof.stark_proof_b64.len() as u64,
            tn_backend: String::new(),
            vram_peak_bytes: 0,
            proof_scope: PROOF_SCOPE.to_string(),
        },
        proof,
        distribution_proof: None,
    })
}

/// A dispatch checked against what a v2 sampling leaf proves.
struct Checked {
    circuit: wire::Circuit,
    sampling: Sampling,
    level: SecurityLevel,
    seed: u64,
}

impl Checked {
    fn from_task(task: &ComputeTask, rows: u64) -> Result<Self, Failure> {
        if task.output_mode != OutputMode::SampleCounts {
            return Err(bad_request("a v2 leaf is a sample_counts sub-task"));
        }
        if !task.slice_assignments.is_empty()
            || task.contraction_tree.is_some()
            || task.mps_time_split.is_some()
        {
            return Err(bad_request(
                "a v2 leaf is an unsliced sub-task without a tree or time split",
            ));
        }
        if task.noise_model.is_some() {
            return Err(bad_request("a v2 leaf has no noise model"));
        }
        let level = SecurityLevel::parse(&task.security_level).ok_or_else(|| {
            bad_request(format!("unknown security level {:?}", task.security_level))
        })?;
        let (Some(cbits), Some(shots), Some(seed)) =
            (task.classical_bit_count, task.shots, task.sample_seed)
        else {
            return Err(bad_request(
                "a v2 leaf needs classical_bit_count, shots and sample_seed",
            ));
        };
        let shots = u32::try_from(shots).map_err(|_| bad_request(format!("{shots} shots")))?;
        // The orchestrator verifies against the circuit it signed; the leaf reads it the same
        // way from this JSON (angles round-trip exactly).
        let json = serde_json::to_string(&task.circuit)
            .map_err(|e| bad_request(format!("circuit JSON: {e}")))?;
        let circuit = wire::parse_circuit(&json).map_err(|e| bad_request(e.to_string()))?;
        let want = leaf::gate_rows(task.qubit_count, &circuit.gates)
            .map_err(|e| bad_request(e.to_string()))?;
        if want as u64 != rows {
            return Err(bad_request(format!(
                "sv_leaf_rows {rows}, but the circuit has {want} gate rows"
            )));
        }
        Ok(Self {
            sampling: Sampling {
                measures: circuit.measures.clone(),
                cbits,
                shots,
            },
            circuit,
            level,
            seed,
        })
    }
}

/// The leaf's counts (indexed by register value) as a `SampleResult`: keys of `cbits`
/// characters, cbit 0 rightmost, zero counts omitted, as [`wire::counts_json`] writes them.
fn sample_result(counts: &[u64], cbits: usize) -> SampleResult {
    let counts: std::collections::BTreeMap<String, u64> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(reg, &c)| (format!("{reg:0cbits$b}"), c))
        .collect();
    SampleResult {
        shots: counts.values().sum(),
        counts,
    }
}

/// Seconds a v2 leaf takes per gate row on this machine, measured once per process by
/// proving a small sampling leaf (n = 12, 8 rounding steps, 1024 shots). Fixed costs of the
/// small leaf count as per-row time, so it errs slow. `None` when v2 is off or the
/// measurement failed.
pub fn seconds_per_row() -> Option<f64> {
    static RATE: OnceLock<Option<f64>> = OnceLock::new();
    *RATE.get_or_init(|| {
        if !enabled() {
            return None;
        }
        measure().map_err(|e| eprintln!("v2 leaf rate: {e}")).ok()
    })
}

fn measure() -> Result<f64, String> {
    const N: usize = 12;
    let mut circuit: Vec<crate::engine::Gate> = (0..8)
        .map(|q| crate::engine::Gate::RY(q, 0.3 + 0.1 * q as f64))
        .collect();
    circuit.extend(
        (0..4).map(|q| {
            crate::engine::Gate::MEASURE(crate::engine::MeasureParams { qubit: q, cbit: q })
        }),
    );
    let json = serde_json::to_string(&circuit).map_err(|e| e.to_string())?;
    let parsed = wire::parse_circuit(&json).map_err(|e| e.to_string())?;
    let rows = leaf::gate_rows(N, &parsed.gates).map_err(|e| e.to_string())?;
    let sampling = Sampling {
        measures: parsed.measures,
        cbits: 4,
        shots: 1024,
    };
    let pool = pool().map_err(|(_, m)| m)?;
    let start = Instant::now();
    pool.install(|| {
        let leaf = leaf::build_sampling(SecurityLevel::Low, N, parsed.gates, sampling)
            .map_err(|e| e.to_string())?;
        let ctx = Dispatch {
            circuit_id: "rate",
            sub_task_id: "rate",
            node_id: "rate",
            slice_id: "0",
        }
        .context();
        leaf.prove_sampling(ctx, 1).map_err(|e| e.to_string())
    })?;
    let rate = start.elapsed().as_secs_f64() / rows as f64;
    (rate.is_finite() && rate > 0.0)
        .then_some(rate)
        .ok_or_else(|| format!("rate {rate}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Gate, MeasureParams};

    fn task(rows: u64) -> ComputeTask {
        let circuit = vec![
            Gate::H(0),
            Gate::CNOT(0, 1),
            Gate::RY(1, 0.7),
            Gate::MEASURE(MeasureParams { qubit: 0, cbit: 0 }),
            Gate::MEASURE(MeasureParams { qubit: 1, cbit: 1 }),
        ];
        serde_json::from_value(serde_json::json!({
            "task_id": "sub-fixture",
            "circuit_id": "circuit-fixture",
            "node_id": "node-fixture",
            "qubit_count": 2,
            "original_qubit_count": 2,
            "slice_id": "0",
            "slice_assignments": [],
            "circuit": circuit,
            "output_mode": "sample_counts",
            "classical_bit_count": 2,
            "shots": 64,
            "sample_seed": 0xDEADBEEFu64,
            "security_level": "low",
            "sv_leaf_rows": rows,
        }))
        .expect("task")
    }

    #[test]
    fn measures_a_rate() {
        let rate = seconds_per_row().expect("rate");
        assert!(rate > 0.0 && rate < 1.0, "{rate}");
    }

    #[test]
    fn counts_match_the_wire_json() {
        let counts = [3, 0, 61, 0];
        let sample = sample_result(&counts, 2);
        assert_eq!(
            crate::sample::format_go_sample_result_json(&sample),
            wire::counts_json(&counts, 2)
        );
        assert_eq!(sample.shots, 64);
    }

    #[test]
    fn checks_the_dispatch() {
        // H and RY round (the CNOT flushes H first): two steps of 2 rows.
        assert!(Checked::from_task(&task(4), 4).is_ok());
        let err = Checked::from_task(&task(8), 8).err().expect("wrong rows");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.contains("4 gate rows"), "{}", err.1);

        let mut sliced = task(4);
        sliced.slice_assignments =
            vec![
                serde_json::from_value(serde_json::json!({"edge_id": "e_0", "value": 1}))
                    .expect("assignment"),
            ];
        assert!(Checked::from_task(&sliced, 4).is_err());
        let mut scalar = task(4);
        scalar.output_mode = OutputMode::StatevectorScalar;
        assert!(Checked::from_task(&scalar, 4).is_err());
        let mut tier = task(4);
        tier.security_level = "extreme".into();
        assert!(Checked::from_task(&tier, 4).is_err());
        let mut reset = task(4);
        reset.circuit.push(Gate::RESET(0));
        assert!(Checked::from_task(&reset, 4).is_err());
    }

    /// Proves the orchestrator's fixture dispatch; the orchestrator's FFI test verifies the
    /// same ids, seed and circuit, so the digest and counts must match the leaf's own.
    #[test]
    fn proves_a_sampling_leaf() {
        let response = compute(task(4), 4).expect("v2 leaf");
        let sample = response.sample_result.expect("counts");
        assert_eq!(sample.shots, 64);
        assert_eq!(response.result_type, "sample_counts");
        let pi = &response.proof.public_inputs;
        assert_eq!(pi.sv_digest.len(), 4);
        assert_eq!(pi.output_result_hash, calculate_sample_result_hash(&sample));
        assert_eq!(response.work_report.proof_scope, PROOF_SCOPE);

        // Verify it the way the orchestrator does: its own verifier, the node's digest.
        let json = serde_json::to_string(&task(4).circuit).expect("json");
        let c = wire::parse_circuit(&json).expect("circuit");
        let sampling = Sampling {
            measures: c.measures,
            cbits: 2,
            shots: 64,
        };
        let verifier =
            leaf::build_verifier(SecurityLevel::Low, 2, c.gates, Some(sampling)).expect("verifier");
        let counts = wire::parse_counts(
            &crate::sample::format_go_sample_result_json(&sample),
            verifier.sampling().expect("sampling"),
        )
        .expect("canonical counts");
        let proof = general_purpose::STANDARD
            .decode(&response.proof.stark_proof_b64)
            .expect("base64");
        let digest: [u64; 4] = pi.sv_digest.clone().try_into().expect("4 words");
        let dispatch = Dispatch {
            circuit_id: "circuit-fixture",
            sub_task_id: "sub-fixture",
            node_id: "node-fixture",
            slice_id: "0",
        };
        verifier
            .verify_sampling(proof.as_slice(), &dispatch, digest, 0xDEADBEEF, &counts)
            .expect("the orchestrator's check passes");
        let other = Dispatch {
            node_id: "node-other",
            ..dispatch
        };
        assert!(verifier
            .verify_sampling(proof.as_slice(), &other, digest, 0xDEADBEEF, &counts)
            .is_err());
    }
}
