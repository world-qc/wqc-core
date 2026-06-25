//! Tensor-network contraction engine for `wqc-core`.
//!
//! Default backend: bond-truncated MPS (`O(N · χ²)` memory). Dense reference backend remains
//! in `dense.rs` for unit tests.

pub mod backend;
pub mod boundary;
pub mod contract;
pub mod dense;
pub mod gates;
pub mod mps;
pub mod trace;

#[cfg(feature = "webgpu")]
pub mod gpu;

pub use contract::contract_slice;
pub use gates::resolve_bond_dim;
pub use mps::MpsState;
