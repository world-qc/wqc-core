//! Tensor-network contraction engine for `wqc-core`.
//!
//! Each gate is a complex unitary tensor; contraction updates the open wire indices of the
//! network state. The dense backend materialises the rank-N state tensor (`2^N` amplitudes);
//! bond-truncated MPS and GPU backends will plug in behind the same contract API.

pub mod boundary;
pub mod contract;
pub mod dense;
pub mod trace;

pub use contract::contract_slice;
pub use dense::DenseTnState;
