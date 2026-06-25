//! TN execution backend selection (`WQC_TN_BACKEND`).

/// Active tensor-network execution backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TnBackend {
    Cpu,
    #[cfg(feature = "webgpu")]
    WebGpu,
}

/// Read `WQC_TN_BACKEND` (default `cpu`). `webgpu` requires the `webgpu` cargo feature.
pub fn tn_backend_from_env() -> TnBackend {
    match std::env::var("WQC_TN_BACKEND")
        .unwrap_or_else(|_| "cpu".into())
        .to_ascii_lowercase()
        .as_str()
    {
        #[cfg(feature = "webgpu")]
        "webgpu" | "gpu" => TnBackend::WebGpu,
        #[cfg(not(feature = "webgpu"))]
        "webgpu" | "gpu" => {
            eprintln!(
                "WQC_TN_BACKEND=webgpu requested but wqc-core was built without --features webgpu; using CPU"
            );
            TnBackend::Cpu
        }
        _ => TnBackend::Cpu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_backend_is_cpu() {
        std::env::remove_var("WQC_TN_BACKEND");
        assert_eq!(tn_backend_from_env(), TnBackend::Cpu);
    }
}
