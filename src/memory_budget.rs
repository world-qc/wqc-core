//! Host memory budget helpers for WQC workers.
//!
//! Reserve headroom for the OS and other processes:
//! - `< 16 GiB` total RAM → reserve 1 GiB
//! - `>= 16 GiB` total RAM → reserve 2 GiB
//!
//! `max_memory_bytes = total_bytes - reserve` (minimum 1 GiB). The total is not
//! rounded down to whole GiB first: a 3.8 GiB host gets 2.8 GiB, not 2 GiB.

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

/// Maximum WQC memory budget (bytes) from total physical RAM (bytes).
/// Subtracts the reserve from the exact total, so the fraction of a GiB is kept.
/// Matches wqc-node (same test vectors there).
pub fn max_wqc_memory_bytes_from_total(total_bytes: u64) -> u64 {
    let reserve = host_memory_reserve_gib(total_bytes / GIB).saturating_mul(GIB);
    total_bytes.saturating_sub(reserve).max(GIB)
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
        for (total, budget) in [(2, 1), (4, 3), (8, 7), (16, 14), (32, 30), (64, 62)] {
            assert_eq!(max_wqc_memory_bytes_from_total(total * GIB), budget * GIB);
        }
    }

    #[test]
    fn bytes_budget_keeps_the_fraction_of_a_gib() {
        let mib = 1024 * 1024;
        // e2-medium reports about 3.83 GiB.
        let total = 3 * GIB + 850 * mib;
        assert_eq!(max_wqc_memory_bytes_from_total(total), 2 * GIB + 850 * mib);
        assert_eq!(
            max_wqc_memory_bytes_from_total(16 * GIB + 512 * mib),
            14 * GIB + 512 * mib
        );
        assert_eq!(max_wqc_memory_bytes_from_total(8 * GIB), 7 * GIB);
    }

    #[test]
    fn bytes_budget_is_at_least_one_gib() {
        assert_eq!(max_wqc_memory_bytes_from_total(0), GIB);
        assert_eq!(max_wqc_memory_bytes_from_total(GIB + GIB / 2), GIB);
    }
}
