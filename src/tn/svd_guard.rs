//! Shared bond-truncation rule for CPU f64 SVD and the GPU Jacobi path.
//!
//! Singular values at or below [`SINGULAR_CUTOFF`] are dropped. A GPU split is
//! kept only when the cutoff and the χ boundary are far enough apart that
//! double-float noise cannot change which values are kept. Closeness among
//! values that are all kept does not change the bond.

/// Same threshold as the historical `truncated_svd_split` filter.
pub const SINGULAR_CUTOFF: f64 = 1e-12;

/// Relative gap below which the χ cut itself is ambiguous.
const CHI_REL_GAP: f64 = 1e-3;

/// Absolute band around the cutoff. Values here can flip under GPU rounding.
const CUTOFF_ABS_BAND: f64 = 1e-10;

/// How many singular values the MPS split keeps.
///
/// `values_desc` must be sorted descending. Matches the CPU rank rule:
/// count of values strictly above [`SINGULAR_CUTOFF`], at least 1, then
/// capped by `max_bond` and the matrix shape.
pub fn retained_chi(values_desc: &[f64], max_bond: usize, rows: usize, cols: usize) -> usize {
    if values_desc.is_empty() || max_bond == 0 || rows == 0 || cols == 0 {
        return 0;
    }
    let rank = values_desc
        .iter()
        .filter(|s| **s > SINGULAR_CUTOFF)
        .count()
        .max(1);
    rank.min(max_bond).min(rows).min(cols)
}

/// Round-robin Jacobi pairs for one sweep. `cols` must be even and at least 2.
///
/// Layout: `steps * (cols/2)` pairs, each pair two `u32`s `(p, q)`.
pub fn jacobi_pairs(cols: usize) -> Vec<u32> {
    assert!(cols >= 2 && cols.is_multiple_of(2));
    // Circle method: index 0 stays put, the rest rotate. Each step is a
    // perfect matching, and one sweep visits every unordered pair once.
    let mut players: Vec<usize> = (0..cols).collect();
    let mut out = Vec::with_capacity((cols - 1) * cols);
    for _step in 0..(cols - 1) {
        for i in 0..(cols / 2) {
            out.push(players[i] as u32);
            out.push(players[cols - 1 - i] as u32);
        }
        let last = players.pop().expect("cols >= 2");
        players.insert(1, last);
    }
    out
}

/// GPU singular values are too close to a decision boundary to trust.
pub fn needs_f64_fallback(values_desc: &[f64], chi: usize) -> bool {
    if values_desc.iter().any(|s| !s.is_finite() || *s < 0.0) {
        return true;
    }
    for &s in values_desc {
        if s > 0.0 && (s - SINGULAR_CUTOFF).abs() <= CUTOFF_ABS_BAND {
            return true;
        }
    }
    if chi > 0 && chi < values_desc.len() {
        let kept = values_desc[chi - 1];
        let dropped = values_desc[chi];
        let scale = kept.max(dropped).max(1e-30);
        if (kept - dropped).abs() / scale < CHI_REL_GAP {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_spectrum_is_not_a_fallback() {
        let values: Vec<f64> = (0..8).map(|i| 1.0 - 0.4 * (i as f64) / 7.0).collect();
        let chi = retained_chi(&values, 8, 8, 8);
        assert_eq!(chi, 8);
        assert!(!needs_f64_fallback(&values, chi));
    }

    #[test]
    fn cutoff_neighbour_falls_back() {
        let values = vec![1.0, 0.5, SINGULAR_CUTOFF * 1.5];
        let chi = retained_chi(&values, 8, 8, 8);
        assert!(needs_f64_fallback(&values, chi));
    }

    #[test]
    fn kept_values_may_be_close() {
        let values = vec![1.0, 1.0 - 1e-8, 0.2];
        let chi = retained_chi(&values, 8, 8, 8);
        assert_eq!(chi, 3);
        assert!(!needs_f64_fallback(&values, chi));
    }

    #[test]
    fn jacobi_schedule_covers_each_pair_once_per_sweep() {
        let cols = 6;
        let pairs = jacobi_pairs(cols);
        let per_step = cols / 2;
        let steps = cols - 1;
        assert_eq!(pairs.len(), steps * per_step * 2);
        let mut seen = std::collections::BTreeSet::new();
        for step in 0..steps {
            let mut used = vec![false; cols];
            for p in 0..per_step {
                let i = (step * per_step + p) * 2;
                let a = pairs[i] as usize;
                let b = pairs[i + 1] as usize;
                assert!(a < cols && b < cols && a != b);
                assert!(!used[a] && !used[b]);
                used[a] = true;
                used[b] = true;
                let key = if a < b { (a, b) } else { (b, a) };
                seen.insert(key);
            }
        }
        assert_eq!(seen.len(), cols * (cols - 1) / 2);
    }

    #[test]
    fn chi_boundary_gap_falls_back() {
        let values = vec![1.0, 0.5, 0.5 * (1.0 - 1e-4)];
        let chi = 2;
        assert!(needs_f64_fallback(&values, chi));
    }
}
