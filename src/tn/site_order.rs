//! MPS site-order hints: remap compact logical qubits onto the 1D MPS path.

use crate::engine::{EngineError, Gate, IfParams, MeasureParams};
use crate::expectation::{ObservableSpec, PauliTerm};

/// Validate `site_order[site] = logical` is a permutation of `0..qubit_count`.
pub fn validate_site_order(site_order: &[usize], qubit_count: usize) -> Result<(), EngineError> {
    if site_order.len() != qubit_count {
        return Err(EngineError::ExecutionFailed(format!(
            "mps_site_order length {} != qubit_count {}",
            site_order.len(),
            qubit_count
        )));
    }
    let mut seen = vec![false; qubit_count];
    for &logical in site_order {
        if logical >= qubit_count {
            return Err(EngineError::QubitIndexOutOfBounds {
                index: logical,
                limit: qubit_count,
            });
        }
        if seen[logical] {
            return Err(EngineError::ExecutionFailed(format!(
                "mps_site_order is not a permutation (duplicate logical {logical})"
            )));
        }
        seen[logical] = true;
    }
    Ok(())
}

/// Build `logical_to_site[logical] = site` from `site_order[site] = logical`.
pub fn logical_to_site_map(site_order: &[usize]) -> Vec<usize> {
    let n = site_order.len();
    let mut inv = vec![0usize; n];
    for (site, &logical) in site_order.iter().enumerate() {
        inv[logical] = site;
    }
    inv
}

fn map_q(logical_to_site: &[usize], q: usize) -> Result<usize, EngineError> {
    logical_to_site
        .get(q)
        .copied()
        .ok_or(EngineError::QubitIndexOutOfBounds {
            index: q,
            limit: logical_to_site.len(),
        })
}

/// Remap every gate qubit index from logical → MPS site coordinates.
pub fn remap_gates(gates: &[Gate], logical_to_site: &[usize]) -> Result<Vec<Gate>, EngineError> {
    gates
        .iter()
        .map(|g| remap_gate(g, logical_to_site))
        .collect()
}

fn remap_gate(gate: &Gate, logical_to_site: &[usize]) -> Result<Gate, EngineError> {
    Ok(match gate {
        Gate::H(t) => Gate::H(map_q(logical_to_site, *t)?),
        Gate::X(t) => Gate::X(map_q(logical_to_site, *t)?),
        Gate::Y(t) => Gate::Y(map_q(logical_to_site, *t)?),
        Gate::Z(t) => Gate::Z(map_q(logical_to_site, *t)?),
        Gate::T(t) => Gate::T(map_q(logical_to_site, *t)?),
        Gate::S(t) => Gate::S(map_q(logical_to_site, *t)?),
        Gate::RX(t, th) => Gate::RX(map_q(logical_to_site, *t)?, *th),
        Gate::RY(t, th) => Gate::RY(map_q(logical_to_site, *t)?, *th),
        Gate::RZ(t, th) => Gate::RZ(map_q(logical_to_site, *t)?, *th),
        Gate::CNOT(c, t) => Gate::CNOT(map_q(logical_to_site, *c)?, map_q(logical_to_site, *t)?),
        Gate::CZ(c, t) => Gate::CZ(map_q(logical_to_site, *c)?, map_q(logical_to_site, *t)?),
        Gate::CCNOT(c1, c2, t) => Gate::CCNOT(
            map_q(logical_to_site, *c1)?,
            map_q(logical_to_site, *c2)?,
            map_q(logical_to_site, *t)?,
        ),
        Gate::MEASURE(MeasureParams { qubit, cbit }) => Gate::MEASURE(MeasureParams {
            qubit: map_q(logical_to_site, *qubit)?,
            cbit: *cbit,
        }),
        Gate::RESET(t) => Gate::RESET(map_q(logical_to_site, *t)?),
        Gate::IF(IfParams { cbit, value, gate }) => Gate::IF(IfParams {
            cbit: *cbit,
            value: *value,
            gate: Box::new(remap_gate(gate, logical_to_site)?),
        }),
    })
}

/// Remap Pauli labels so character `i` acts on the site that holds logical qubit `i`.
pub fn remap_observables(
    observables: &[ObservableSpec],
    logical_to_site: &[usize],
) -> Result<Vec<ObservableSpec>, EngineError> {
    let n = logical_to_site.len();
    observables
        .iter()
        .map(|obs| {
            let terms = obs
                .terms
                .iter()
                .map(|term| {
                    let label = remap_pauli_label(&term.label, logical_to_site, n)?;
                    Ok(PauliTerm {
                        label,
                        coeff: term.coeff.clone(),
                    })
                })
                .collect::<Result<Vec<_>, EngineError>>()?;
            Ok(ObservableSpec {
                id: obs.id.clone(),
                terms,
            })
        })
        .collect()
}

fn remap_pauli_label(
    label: &str,
    logical_to_site: &[usize],
    n: usize,
) -> Result<String, EngineError> {
    let chars: Vec<char> = label.chars().collect();
    if chars.len() != n {
        return Err(EngineError::ExecutionFailed(format!(
            "Pauli label length {} != qubit_count {n}",
            chars.len()
        )));
    }
    let mut out = vec!['I'; n];
    for (logical, ch) in chars.into_iter().enumerate() {
        let site = map_q(logical_to_site, logical)?;
        out[site] = ch;
    }
    Ok(out.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expectation::ComplexCoeff;

    #[test]
    fn validate_rejects_non_permutation() {
        assert!(validate_site_order(&[0, 0], 2).is_err());
        assert!(validate_site_order(&[0, 1, 2], 2).is_err());
    }

    #[test]
    fn remap_cnot_swaps_operands_under_reverse_order() {
        let order = vec![1, 0]; // site0=logical1, site1=logical0
        validate_site_order(&order, 2).unwrap();
        let inv = logical_to_site_map(&order);
        let gates = remap_gates(&[Gate::CNOT(0, 1)], &inv).unwrap();
        assert_eq!(gates, vec![Gate::CNOT(1, 0)]);
    }

    #[test]
    fn remap_pauli_label_reverse() {
        let order = vec![1, 0];
        let inv = logical_to_site_map(&order);
        let obs = remap_observables(
            &[ObservableSpec {
                id: "Z".into(),
                terms: vec![PauliTerm {
                    label: "ZI".into(),
                    coeff: ComplexCoeff {
                        real: 1.0,
                        imag: 0.0,
                    },
                }],
            }],
            &inv,
        )
        .unwrap();
        assert_eq!(obs[0].terms[0].label, "IZ");
    }
}
