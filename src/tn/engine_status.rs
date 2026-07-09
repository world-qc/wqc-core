//! Cached TN engine probe (requested vs active backend, shared GPU device handle).

use std::sync::OnceLock;

#[cfg(feature = "webgpu")]
use std::sync::Arc;

use super::gates::max_bond_dim_from_env;

#[cfg(feature = "webgpu")]
use super::backend::{tn_backend_from_env, TnBackend};
#[cfg(feature = "webgpu")]
use super::gpu::GpuMpsDevice;

/// Result of a one-time TN backend probe (served via `GET /sysinfo`).
#[derive(Debug, Clone)]
pub struct TnEngineStatus {
    pub requested: String,
    pub active: String,
    pub note: Option<String>,
    pub mps_max_bond_dim: usize,
}

static TN_STATUS: OnceLock<TnEngineStatus> = OnceLock::new();

#[cfg(feature = "webgpu")]
static GPU_DEVICE: OnceLock<Option<Arc<GpuMpsDevice>>> = OnceLock::new();

fn requested_backend_from_env() -> String {
    match std::env::var("WQC_TN_BACKEND")
        .unwrap_or_else(|_| "cpu".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "webgpu" | "gpu" => "webgpu".to_string(),
        _ => "cpu".to_string(),
    }
}

fn probe_tn_engine_status() -> TnEngineStatus {
    let requested = requested_backend_from_env();
    let mps_max_bond_dim = max_bond_dim_from_env();

    if requested == "cpu" {
        return TnEngineStatus {
            requested,
            active: "cpu".to_string(),
            note: None,
            mps_max_bond_dim,
        };
    }

    #[cfg(not(feature = "webgpu"))]
    {
        TnEngineStatus {
            requested,
            active: "cpu".to_string(),
            note: Some(
                "WQC_TN_BACKEND=webgpu but wqc-core was built without --features webgpu".into(),
            ),
            mps_max_bond_dim,
        }
    }

    #[cfg(feature = "webgpu")]
    {
        if shared_gpu_device().is_some() {
            TnEngineStatus {
                requested,
                active: "webgpu".to_string(),
                note: None,
                mps_max_bond_dim,
            }
        } else {
            TnEngineStatus {
                requested,
                active: "cpu".to_string(),
                note: Some("WebGPU adapter not found; CPU MPS fallback".into()),
                mps_max_bond_dim,
            }
        }
    }
}

/// Cached probe used by `/sysinfo` and `MpsState` GPU initialization.
pub fn tn_engine_status() -> &'static TnEngineStatus {
    TN_STATUS.get_or_init(probe_tn_engine_status)
}

/// Shared WebGPU device when `active == webgpu` (initialized at most once).
#[cfg(feature = "webgpu")]
pub fn shared_gpu_device() -> Option<Arc<GpuMpsDevice>> {
    GPU_DEVICE
        .get_or_init(|| {
            if tn_backend_from_env() != TnBackend::WebGpu {
                return None;
            }
            GpuMpsDevice::try_new()
        })
        .clone()
}

#[cfg(not(feature = "webgpu"))]
#[allow(dead_code)]
pub fn shared_gpu_device() -> Option<()> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_status_is_cpu() {
        std::env::remove_var("WQC_TN_BACKEND");
        let _ = TN_STATUS.get();
        // Cannot reset OnceLock; only assert env default parses as cpu.
        assert_eq!(requested_backend_from_env(), "cpu");
    }
}
