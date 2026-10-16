//! WebGPU acceleration layer for bond-truncated MPS.

mod device;
mod gemm;
mod split;

pub use device::GpuMpsDevice;
pub(crate) use gemm::{calibrate_gemm_side, gemm_min_side};
pub(crate) use split::svd_uses_gpu;
pub use split::{GpuSite, SplitOutcome};
