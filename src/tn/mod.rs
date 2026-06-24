//! Tensor-network contraction engine for `wqc-core`.
//!
//! Default backend: bond-truncated MPS (`O(N · χ²)` memory). Dense reference backend remains
//! in `dense.rs` for unit tests.

pub mod boundary;
pub mod contract;
pub mod dense;
pub mod gates;
pub mod mps;
pub mod trace;

pub use contract::contract_slice;
pub use gates::max_bond_dim_from_env;
pub use mps::MpsState;
