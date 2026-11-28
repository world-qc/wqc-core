//! Shared bond-truncation rule for CPU f64 SVD and the GPU Jacobi path.
//!
//! Singular values at or below [`SINGULAR_CUTOFF`] are dropped. A GPU split is
//! kept only when the cutoff and the χ boundary are far enough apart that
//! double-float noise cannot change which values are kept. Closeness among
//! values that are all kept does not change the bond.

use nalgebra::{DMatrix, Dyn, SVD};
use num_complex::Complex64;

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

/// An SVD whose factors reproduce the matrix.
///
/// nalgebra's complex SVD is not reliable on rank-deficient matrices: on a
/// rank-1 2×2 matrix with entries near 1e-16 it returned a top singular value
/// above the Frobenius norm and a `U` with weight on a zero row, and on 4×4
/// MPS splits its singular values missed the norm by 1e-8 to 1e-5. An MPS
/// built from such a split loses its norm and gives wrong amplitudes.
///
/// Entries below 1e-14 of the norm are set to 0 first. nalgebra's result is
/// kept when it passes [`svd_reproduces`]; otherwise the transpose is tried,
/// then a one-sided Jacobi SVD ([`jacobi_svd`]), which is accurate but
/// slower. A factorization that still fails the check is an error, never a
/// wrong state.
pub fn checked_svd(mut mat: DMatrix<Complex64>) -> Result<SVD<Complex64, Dyn, Dyn>, String> {
    let fro = mat.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    let dust = fro * 1e-14;
    for c in mat.iter_mut() {
        if c.norm() < dust {
            *c = Complex64::new(0.0, 0.0);
        }
    }
    let svd = mat.clone().svd(true, true);
    if svd_reproduces(&mat, &svd) {
        return Ok(svd);
    }
    #[cfg(test)]
    SVD_RETRIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let t = mat.transpose().svd(true, true);
    if let (Some(u_t), Some(v_t_t)) = (t.u.as_ref(), t.v_t.as_ref()) {
        let flipped = SVD {
            u: Some(v_t_t.transpose()),
            v_t: Some(u_t.transpose()),
            singular_values: t.singular_values.clone(),
        };
        if svd_reproduces(&mat, &flipped) {
            return Ok(flipped);
        }
    }
    #[cfg(test)]
    SVD_JACOBI.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let jacobi = jacobi_svd(&mat);
    if svd_reproduces(&mat, &jacobi) {
        return Ok(jacobi);
    }
    Err(format!(
        "SVD of a {}x{} matrix failed its check",
        mat.nrows(),
        mat.ncols()
    ))
}

#[cfg(test)]
pub static SVD_RETRIES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub static SVD_JACOBI: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// One-sided (Hestenes) Jacobi SVD of a complex matrix, with singular values
/// in descending order and `U` of `min(rows, cols)` columns.
///
/// Column pairs of `A` are rotated until orthogonal; then `A·V = U·Σ`. Each
/// pair is first rephased so its inner product is real, and the real 2×2
/// rotation then zeroes it. A wide matrix is done on its adjoint.
pub fn jacobi_svd(a: &DMatrix<Complex64>) -> SVD<Complex64, Dyn, Dyn> {
    if a.nrows() < a.ncols() {
        let t = jacobi_svd(&a.adjoint());
        // Aᴴ = U'ΣV'ᴴ gives A = V'ΣU'ᴴ.
        return SVD {
            u: t.v_t.as_ref().map(|v| v.adjoint()),
            v_t: t.u.as_ref().map(|u| u.adjoint()),
            singular_values: t.singular_values,
        };
    }
    let (m, n) = (a.nrows(), a.ncols());
    let mut w = a.clone();
    let mut v = DMatrix::<Complex64>::identity(n, n);
    for _sweep in 0..80 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let alpha: f64 = w.column(p).iter().map(|c| c.norm_sqr()).sum();
                let beta: f64 = w.column(q).iter().map(|c| c.norm_sqr()).sum();
                let gamma: Complex64 = w
                    .column(p)
                    .iter()
                    .zip(w.column(q).iter())
                    .map(|(x, y)| x.conj() * y)
                    .sum();
                let g = gamma.norm();
                if g <= 1e-15 * (alpha * beta).sqrt() || g == 0.0 {
                    continue;
                }
                rotated = true;
                let phase = gamma / g;
                let zeta = (beta - alpha) / (2.0 * g);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for (mat, rows) in [(&mut w, m), (&mut v, n)] {
                    for r in 0..rows {
                        let xp = mat[(r, p)];
                        let xq = mat[(r, q)] * phase.conj();
                        mat[(r, p)] = xp * c - xq * s;
                        mat[(r, q)] = (xp * s + xq * c) * phase;
                    }
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut order: Vec<(f64, usize)> = (0..n)
        .map(|j| {
            (
                w.column(j).iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt(),
                j,
            )
        })
        .collect();
    order.sort_by(|x, y| y.0.total_cmp(&x.0));
    let mut u = DMatrix::<Complex64>::zeros(m, n);
    let mut v_t = DMatrix::<Complex64>::zeros(n, n);
    let mut values = nalgebra::DVector::<f64>::zeros(n);
    for (k, (sigma, j)) in order.into_iter().enumerate() {
        values[k] = sigma;
        if sigma > 0.0 {
            for r in 0..m {
                u[(r, k)] = w[(r, j)] / sigma;
            }
        }
        for r in 0..n {
            v_t[(k, r)] = v[(r, j)].conj();
        }
    }
    SVD {
        u: Some(u),
        v_t: Some(v_t),
        singular_values: values,
    }
}

/// Relative tolerance of [`svd_reproduces`]; a correct f64 SVD is near 1e-15.
const SVD_CHECK_TOL: f64 = 1e-9;

/// Whether `svd` is a factorization of `a`: singular values finite,
/// non-negative and descending with `Σσ² = ‖A‖F²`, and `U·Σ·Vᴴ·x = A·x` for
/// two fixed probe vectors (a Freivalds check, O(rows·cols) instead of a
/// full rebuild).
fn svd_reproduces(a: &DMatrix<Complex64>, svd: &SVD<Complex64, Dyn, Dyn>) -> bool {
    let (Some(u), Some(v_t)) = (svd.u.as_ref(), svd.v_t.as_ref()) else {
        return false;
    };
    let s = &svd.singular_values;
    if s.iter().any(|x| !x.is_finite() || *x < 0.0) || s.as_slice().windows(2).any(|w| w[0] < w[1])
    {
        return false;
    }
    let fro2: f64 = a.iter().map(|c| c.norm_sqr()).sum();
    let s2: f64 = s.iter().map(|x| x * x).sum();
    let scale = fro2.sqrt().max(f64::MIN_POSITIVE);
    if (s2 - fro2).abs() > SVD_CHECK_TOL * fro2.max(f64::MIN_POSITIVE) + 1e-28 {
        return false;
    }
    let k = s.len();
    if u.ncols() < k || v_t.nrows() < k || u.nrows() != a.nrows() || v_t.ncols() != a.ncols() {
        return false;
    }
    for probe in 0..2u32 {
        let x = DMatrix::<Complex64>::from_fn(a.ncols(), 1, |j, _| {
            let t = (j as f64 + 1.0) * (0.754_877_666 + f64::from(probe) * 0.569_840_29);
            Complex64::new(t.sin(), t.cos())
        });
        let direct = a * &x;
        let mut inner = v_t.rows(0, k) * &x;
        for (row, sv) in s.iter().enumerate() {
            inner[(row, 0)] *= *sv;
        }
        let rebuilt = u.columns(0, k) * inner;
        let x_norm = x.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        let err = (direct - rebuilt)
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if err > SVD_CHECK_TOL * scale * x_norm + 1e-14 {
            return false;
        }
    }
    true
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
