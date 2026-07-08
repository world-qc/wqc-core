//! Host memory budget helpers for WQC workers.
//!
//! Reserve headroom for the OS and other processes:
//! - `< 16 GiB` total RAM → reserve 1 GiB
//! - `>= 16 GiB` total RAM → reserve 2 GiB
//!
//! `max_memory_gb = total_gib - reserve_gib` (minimum 1 GiB).

pub const HOST_MEMORY_RESERVE_THRESHOLD_GIB: u64 = 16;
pub const HOST_MEMORY_RESERVE_SMALL_GIB: u64 = 1;
pub const HOST_MEMORY_RESERVE_LARGE_GIB: u64 = 2;

const GIB: u64 = 1024 * 1024 * 1024;

/// Headroom to leave on the host (GiB).
pub fn host_memory_reserve_gib(total_gib: u64) -> u64 {
    if total_gib >= HOST_MEMORY_RESERVE_THRESHOLD_GIB {
        HOST_MEMORY_RESERVE_LARGE_GIB
    } else {
        HOST_MEMORY_RESERVE_SMALL_GIB
    }
}

/// Maximum WQC memory budget (GiB) from total physical RAM (GiB).
pub fn max_wqc_memory_gib_from_total(total_gib: u64) -> u64 {
    if total_gib == 0 {
        return 1;
    }
    total_gib
        .saturating_sub(host_memory_reserve_gib(total_gib))
        .max(1)
}

/// Maximum WQC memory budget (bytes) from total physical RAM (bytes).
pub fn max_wqc_memory_bytes_from_total(total_bytes: u64) -> u64 {
    let total_gib = total_bytes / GIB;
    max_wqc_memory_gib_from_total(total_gib).saturating_mul(GIB)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_one_gib_below_sixteen() {
        assert_eq!(host_memory_reserve_gib(2), 1);
        assert_eq!(host_memory_reserve_gib(8), 1);
        assert_eq!(host_memory_reserve_gib(15), 1);
    }

    #[test]
    fn reserve_two_gib_at_or_above_sixteen() {
        assert_eq!(host_memory_reserve_gib(16), 2);
        assert_eq!(host_memory_reserve_gib(128), 2);
    }

    #[test]
    fn max_memory_examples() {
        assert_eq!(max_wqc_memory_gib_from_total(2), 1);
        assert_eq!(max_wqc_memory_gib_from_total(4), 3);
        assert_eq!(max_wqc_memory_gib_from_total(8), 7);
        assert_eq!(max_wqc_memory_gib_from_total(16), 14);
        assert_eq!(max_wqc_memory_gib_from_total(32), 30);
        assert_eq!(max_wqc_memory_gib_from_total(64), 62);
    }
}
