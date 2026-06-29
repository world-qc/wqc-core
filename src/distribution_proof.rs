//! Phase C2 scaffolding: measurement-distribution STARK binding (not yet implemented).

use serde::{Deserialize, Serialize};

/// Reports whether the STARK transcript binds sampled counts beyond unitary trace.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DistributionProofStatus {
    pub bound: bool,
    pub scheme: &'static str,
}

/// Phase C2 placeholder — quorum still uses canonical `counts` hash (seed-fixed).
pub fn distribution_stark_status() -> DistributionProofStatus {
    DistributionProofStatus {
        bound: false,
        scheme: "unitary_trace_only",
    }
}
