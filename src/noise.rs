//! Phase C3: optional noise models for trajectory sampling (not STARK-bound).

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::engine::Gate;
use crate::tn::dense::DenseTnState;

/// Optional simulator noise applied during mid-circuit trajectory sampling only.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct NoiseModel {
    /// Single-qubit depolarizing probability after each 1-qubit gate (0..1).
    #[serde(default)]
    pub depolarizing_p: Option<f64>,
    /// Classical bit flip probability after each MEASURE (0..1).
    #[serde(default)]
    pub readout_error: Option<f64>,
}

impl NoiseModel {
    pub fn is_active(&self) -> bool {
        self.depolarizing_p.is_some() || self.readout_error.is_some()
    }

    pub fn apply_depolarizing(&self, state: &mut DenseTnState, qubit: usize, rng: &mut impl Rng) {
        let Some(p) = self.depolarizing_p else {
            return;
        };
        if p <= 0.0 {
            return;
        }
        if rng.gen::<f64>() >= p {
            return;
        }
        let pauli = rng.gen_range(0..3);
        let gate = match pauli {
            0 => Gate::X(qubit),
            1 => Gate::Y(qubit),
            _ => Gate::Z(qubit),
        };
        state.apply_gate(&gate);
    }

    pub fn apply_readout(&self, outcome: u8, rng: &mut impl Rng) -> u8 {
        let Some(p) = self.readout_error else {
            return outcome;
        };
        if p <= 0.0 {
            return outcome;
        }
        if rng.gen::<f64>() < p {
            1 - outcome
        } else {
            outcome
        }
    }
}
