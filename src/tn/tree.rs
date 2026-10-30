//! Sweep contraction tree for `statevector_scalar`.
//!
//! A leaf is an input `|0⟩` or one parent-circuit gate. Each internal node is
//! one step of the MPS sweep along `site_order` (empty means identity):
//! a one-qubit gate replaces that wire's site tensor, and a two-qubit gate
//! swaps sites until the two wires are adjacent, merges them, applies the
//! gate, splits them back into two site tensors, then undoes the swaps.
//! Wire numbers stay the parent circuit's. A wire no gate touches stays
//! `Ket0` and has no node; the sweep still keeps its site at `|0⟩`.
//!
//! [`contract_tree`] replays those nodes. It rejects every other shape. The
//! STARK trace is the same sweep trace as [`super::contract::contract_slice`],
//! including an even run of H, a net rotation, and a disjoint adjacent layer.
//! The call has no slice assignments. A task with no tree stays on
//! `contract_slice`.

use std::ops::Range;

use crate::engine::{Circuit, ComplexResult, EngineError, Gate};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use super::mps::MpsState;
use super::site_order::{logical_to_site_map, remap_gates, validate_site_order};
use super::trace::{execute_with_trace_using, TraceStep};

/// Operand of a sweep step: an input `|0⟩`, or one factor of an earlier node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRef {
    /// `|0⟩` on a parent-circuit wire.
    Ket0(u32),
    /// The site tensor written by unary node `nodes[i]`.
    Node(u32),
    /// Left site after swap or pair node `nodes[i]`.
    Left(u32),
    /// Right site after swap or pair node `nodes[i]`.
    Right(u32),
}

/// One step of the MPS sweep.
///
/// `Site` replaces one site. `Swap` exchanges two adjacent sites.
/// `Pair` merges two adjacent sites, applies a two-qubit gate, and splits
/// them. The `ccnot_*` steps are the same sweep inside one CCNOT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ContractionNode {
    Site {
        tensor: TensorRef,
        gate: u32,
    },
    Swap {
        left: TensorRef,
        right: TensorRef,
    },
    Pair {
        left: TensorRef,
        right: TensorRef,
        gate: u32,
        ctrl_on_left: bool,
    },
    CcnotH {
        tensor: TensorRef,
        gate: u32,
    },
    CcnotT {
        tensor: TensorRef,
        gate: u32,
    },
    CcnotTdg {
        tensor: TensorRef,
        gate: u32,
    },
    CcnotCnot {
        left: TensorRef,
        right: TensorRef,
        gate: u32,
        ctrl_on_left: bool,
    },
}

/// Sweep tree plus the MPS path the nodes were built on.
///
/// An empty `site_order` is the identity `0..qubit_count`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractionTree {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub site_order: Vec<u32>,
    pub nodes: Vec<ContractionNode>,
}

/// Checks that sit in front of [`contract_tree`] on the compute path.
pub struct SweepDispatch<'a> {
    pub scalar: bool,
    pub qubit_count: usize,
    pub original_qubit_count: usize,
    pub assignments_empty: bool,
    pub time_split: bool,
    pub mps_site_order: Option<&'a [usize]>,
    pub gates: &'a [Gate],
    pub tree: &'a ContractionTree,
}

struct BuiltSweep {
    tree: ContractionTree,
    groups: Vec<Range<usize>>,
    /// Parent wire at each MPS site before the first gate.
    wire_at: Vec<u32>,
}

/// Parent-circuit sweep. `site_order` empty, or `0..n`, is stored empty.
pub fn sweep_tree(
    qubit_count: usize,
    gates: &[Gate],
    site_order: &[u32],
) -> Result<ContractionTree, EngineError> {
    Ok(build_sweep(qubit_count, gates, site_order)?.tree)
}

/// Reject a tree task that is not a parent-register scalar sweep.
pub fn validate_sweep_dispatch(dispatch: &SweepDispatch<'_>) -> Result<(), EngineError> {
    if !dispatch.scalar {
        return Err(EngineError::ExecutionFailed(
            "contraction_tree is only valid for a scalar".into(),
        ));
    }
    if !dispatch.assignments_empty {
        return Err(EngineError::ExecutionFailed(
            "contraction_tree does not take slice assignments".into(),
        ));
    }
    if dispatch.time_split {
        return Err(EngineError::ExecutionFailed(
            "contraction_tree does not take mps_time_split".into(),
        ));
    }
    if dispatch.qubit_count != dispatch.original_qubit_count {
        return Err(EngineError::ExecutionFailed(
            "contraction_tree keeps the parent register".into(),
        ));
    }
    prepare(
        dispatch.qubit_count,
        dispatch.gates,
        dispatch.tree,
        dispatch.mps_site_order,
    )?;
    Ok(())
}

/// `⟨0…0|ψ⟩` and the same STARK trace the sweep emits today.
pub fn contract_tree(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    mps_site_order: Option<&[usize]>,
    state: &mut MpsState,
) -> Result<(ComplexResult, Vec<f64>), EngineError> {
    let built = prepare(qubit_count, gates, tree, mps_site_order)?;
    let order: Vec<usize> = expand_order(qubit_count, &built.tree.site_order)
        .into_iter()
        .map(|q| q as usize)
        .collect();
    let remapped = remap_gates(gates, &logical_to_site_map(&order))?;
    let mut live = Live::new(&built);
    let trace = execute_with_trace_using(qubit_count, &remapped, state, |state, step| {
        live.on_step(state, gates, &remapped, step)
    })?;
    if live.next_group != built.groups.len() {
        return Err(EngineError::ExecutionFailed(
            "contraction tree is not the mps sweep".into(),
        ));
    }
    let amp = state.amplitude_at_compact_zero();
    Ok((
        ComplexResult {
            real: amp.re,
            imag: amp.im,
        },
        trace,
    ))
}

/// Sweep amplitude without the STARK trace.
///
/// The state update replays every node. The 16-qubit acceptance uses this so
/// the trace sampler, which rereads a dense slice at `n ≤ 16`, is not part of
/// the check.
pub(crate) fn contract_tree_amplitude(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    state: &mut MpsState,
) -> Result<Complex64, EngineError> {
    let built = prepare(qubit_count, gates, tree, None)?;
    if state.qubit_count != qubit_count {
        return Err(EngineError::MismatchedRegister);
    }
    let mut live = Live::new(&built);
    for _ in 0..built.groups.len() {
        live.apply_next(state, gates, true)?;
    }
    Ok(state.amplitude_at_compact_zero())
}

fn prepare(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    mps_site_order: Option<&[usize]>,
) -> Result<BuiltSweep, EngineError> {
    let tree_order = canonicalize_site_order(qubit_count, &tree.site_order)?;
    if let Some(order) = mps_site_order {
        let as_u32 = order
            .iter()
            .map(|&q| {
                u32::try_from(q)
                    .map_err(|_| EngineError::ExecutionFailed("qubit index exceeds u32".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let task_order = canonicalize_site_order(qubit_count, &as_u32)?;
        if task_order != tree_order {
            return Err(EngineError::ExecutionFailed(
                "mps_site_order does not match the contraction tree".into(),
            ));
        }
    }
    let built = build_sweep(qubit_count, gates, &tree_order)?;
    if built.tree.nodes != tree.nodes {
        return Err(EngineError::ExecutionFailed(
            "contraction tree is not the mps sweep".into(),
        ));
    }
    Ok(built)
}

fn build_sweep(
    qubit_count: usize,
    gates: &[Gate],
    site_order: &[u32],
) -> Result<BuiltSweep, EngineError> {
    if qubit_count == 0 {
        return Err(EngineError::InvalidQubitCount(qubit_count));
    }
    let mut circuit = Circuit::new(qubit_count);
    for gate in gates {
        circuit.add(gate.clone())?;
    }
    let site_order = canonicalize_site_order(qubit_count, site_order)?;
    let wire_at = expand_order(qubit_count, &site_order);
    let initial_wires = wire_at.clone();
    let mut site_of = vec![0usize; qubit_count];
    for (site, &wire) in wire_at.iter().enumerate() {
        site_of[wire as usize] = site;
    }
    let mut build = SweepBuilder {
        nodes: Vec::new(),
        groups: Vec::new(),
        tensor: (0..qubit_count as u32).map(TensorRef::Ket0).collect(),
        wire_at,
        site_of,
    };
    for (index, gate) in gates.iter().enumerate() {
        let Some(wires) = unitary_wires(gate, qubit_count, index)? else {
            continue;
        };
        let gate_index = u32::try_from(index)
            .map_err(|_| EngineError::ExecutionFailed("gate index exceeds u32".into()))?;
        let start = build.nodes.len();
        match wires.as_slice() {
            [wire] => build.site_gate(*wire, gate_index),
            [ctrl, tgt] => build.two_qubit(*ctrl, *tgt, gate_index, PairKind::Gate)?,
            [c1, c2, tgt] => build.ccnot(*c1, *c2, *tgt, gate_index)?,
            _ => {
                return Err(EngineError::ExecutionFailed(format!(
                    "gate {index} is not a sweep gate"
                )))
            }
        }
        build.groups.push(start..build.nodes.len());
    }
    Ok(BuiltSweep {
        tree: ContractionTree {
            site_order,
            nodes: build.nodes,
        },
        groups: build.groups,
        wire_at: initial_wires,
    })
}

fn expand_order(qubit_count: usize, site_order: &[u32]) -> Vec<u32> {
    if site_order.is_empty() {
        (0..qubit_count as u32).collect()
    } else {
        site_order.to_vec()
    }
}

/// Empty, or an explicit identity, becomes empty. Any other permutation is kept.
fn canonicalize_site_order(
    qubit_count: usize,
    site_order: &[u32],
) -> Result<Vec<u32>, EngineError> {
    if site_order.is_empty() {
        return Ok(Vec::new());
    }
    let as_usize: Vec<usize> = site_order.iter().map(|&q| q as usize).collect();
    validate_site_order(&as_usize, qubit_count)?;
    let identity = as_usize.iter().enumerate().all(|(i, q)| i == *q);
    if identity {
        Ok(Vec::new())
    } else {
        Ok(site_order.to_vec())
    }
}

/// `None` skips MEASURE, RESET, and IF, matching the sweep's trace loop.
fn unitary_wires(
    gate: &Gate,
    qubit_count: usize,
    gate_index: usize,
) -> Result<Option<Vec<usize>>, EngineError> {
    let wires = match gate {
        Gate::H(t)
        | Gate::X(t)
        | Gate::Y(t)
        | Gate::Z(t)
        | Gate::S(t)
        | Gate::T(t)
        | Gate::RX(t, _)
        | Gate::RY(t, _)
        | Gate::RZ(t, _) => vec![*t],
        Gate::CNOT(c, t) | Gate::CZ(c, t) => vec![*c, *t],
        Gate::CCNOT(c1, c2, t) => vec![*c1, *c2, *t],
        Gate::MEASURE(_) | Gate::RESET(_) | Gate::IF(_) => return Ok(None),
    };
    let mut seen = vec![false; qubit_count];
    for &w in &wires {
        if w >= qubit_count {
            return Err(EngineError::QubitIndexOutOfBounds {
                index: w,
                limit: qubit_count,
            });
        }
        if seen[w] {
            return Err(EngineError::ExecutionFailed(format!(
                "gate {gate_index} repeats wire {w}"
            )));
        }
        seen[w] = true;
    }
    Ok(Some(wires))
}

enum PairKind {
    Gate,
    CcnotCnot,
}

struct SweepBuilder {
    nodes: Vec<ContractionNode>,
    groups: Vec<Range<usize>>,
    tensor: Vec<TensorRef>,
    wire_at: Vec<u32>,
    site_of: Vec<usize>,
}

impl SweepBuilder {
    fn site_gate(&mut self, wire: usize, gate: u32) {
        let index = self.push(ContractionNode::Site {
            tensor: self.tensor[wire],
            gate,
        });
        self.tensor[wire] = TensorRef::Node(index);
    }

    fn two_qubit(
        &mut self,
        ctrl: usize,
        tgt: usize,
        gate: u32,
        kind: PairKind,
    ) -> Result<(), EngineError> {
        let pc = self.site_of[ctrl];
        let pt = self.site_of[tgt];
        let (left, right) = if pc < pt { (pc, pt) } else { (pt, pc) };
        let ctrl_on_left = pc < pt;
        if right - left > 1 {
            for pos in (left + 1..right).rev() {
                self.swap_sites(pos);
            }
        }
        self.pair_at(left, gate, ctrl_on_left, kind);
        if right - left > 1 {
            for pos in left + 1..right {
                self.swap_sites(pos);
            }
        }
        Ok(())
    }

    /// Same decomposition as `MpsState::apply_ccnot`.
    fn ccnot(&mut self, c1: usize, c2: usize, tgt: usize, gate: u32) -> Result<(), EngineError> {
        self.decomp_unary(tgt, gate, DecompUnary::H);
        self.two_qubit(c2, tgt, gate, PairKind::CcnotCnot)?;
        self.decomp_unary(tgt, gate, DecompUnary::Tdg);
        self.two_qubit(c1, tgt, gate, PairKind::CcnotCnot)?;
        self.decomp_unary(tgt, gate, DecompUnary::T);
        self.two_qubit(c2, tgt, gate, PairKind::CcnotCnot)?;
        self.decomp_unary(tgt, gate, DecompUnary::Tdg);
        self.two_qubit(c1, tgt, gate, PairKind::CcnotCnot)?;
        self.decomp_unary(tgt, gate, DecompUnary::T);
        self.decomp_unary(tgt, gate, DecompUnary::H);
        Ok(())
    }

    fn decomp_unary(&mut self, wire: usize, gate: u32, kind: DecompUnary) {
        let tensor = self.tensor[wire];
        let node = match kind {
            DecompUnary::H => ContractionNode::CcnotH { tensor, gate },
            DecompUnary::T => ContractionNode::CcnotT { tensor, gate },
            DecompUnary::Tdg => ContractionNode::CcnotTdg { tensor, gate },
        };
        let index = self.push(node);
        self.tensor[wire] = TensorRef::Node(index);
    }

    fn swap_sites(&mut self, pos: usize) {
        let wl = self.wire_at[pos];
        let wr = self.wire_at[pos + 1];
        let index = self.push(ContractionNode::Swap {
            left: self.tensor[wl as usize],
            right: self.tensor[wr as usize],
        });
        self.tensor[wr as usize] = TensorRef::Left(index);
        self.tensor[wl as usize] = TensorRef::Right(index);
        self.wire_at[pos] = wr;
        self.wire_at[pos + 1] = wl;
        self.site_of[wr as usize] = pos;
        self.site_of[wl as usize] = pos + 1;
    }

    fn pair_at(&mut self, pos: usize, gate: u32, ctrl_on_left: bool, kind: PairKind) {
        let wl = self.wire_at[pos];
        let wr = self.wire_at[pos + 1];
        let node = match kind {
            PairKind::Gate => ContractionNode::Pair {
                left: self.tensor[wl as usize],
                right: self.tensor[wr as usize],
                gate,
                ctrl_on_left,
            },
            PairKind::CcnotCnot => ContractionNode::CcnotCnot {
                left: self.tensor[wl as usize],
                right: self.tensor[wr as usize],
                gate,
                ctrl_on_left,
            },
        };
        let index = self.push(node);
        self.tensor[wl as usize] = TensorRef::Left(index);
        self.tensor[wr as usize] = TensorRef::Right(index);
    }

    fn push(&mut self, node: ContractionNode) -> u32 {
        let index = self.nodes.len() as u32;
        self.nodes.push(node);
        index
    }
}

enum DecompUnary {
    H,
    T,
    Tdg,
}

struct Live<'a> {
    nodes: &'a [ContractionNode],
    groups: &'a [Range<usize>],
    next_group: usize,
    node_wire: Vec<Option<u32>>,
    left_wire: Vec<Option<u32>>,
    right_wire: Vec<Option<u32>>,
    wire_at: Vec<u32>,
    site_of: Vec<usize>,
}

impl<'a> Live<'a> {
    fn new(built: &'a BuiltSweep) -> Self {
        let n = built.tree.nodes.len();
        let mut site_of = vec![0usize; built.wire_at.len()];
        for (site, &wire) in built.wire_at.iter().enumerate() {
            site_of[wire as usize] = site;
        }
        Self {
            nodes: &built.tree.nodes,
            groups: &built.groups,
            next_group: 0,
            node_wire: vec![None; n],
            left_wire: vec![None; n],
            right_wire: vec![None; n],
            wire_at: built.wire_at.clone(),
            site_of,
        }
    }

    fn on_step(
        &mut self,
        state: &mut MpsState,
        gates: &[Gate],
        remapped: &[Gate],
        step: &TraceStep,
    ) -> Result<(), EngineError> {
        match step {
            TraceStep::One(_) => self.apply_next(state, gates, true),
            TraceStep::Net { len, net, .. } => {
                if *len == 1 {
                    self.apply_next(state, gates, true)
                } else {
                    // The sweep's trace emits one net rotation. Replaying each
                    // node would be the same unitary and a different trace row.
                    state.apply_gate(net)?;
                    for _ in 0..*len {
                        self.apply_next(state, gates, false)?;
                    }
                    Ok(())
                }
            }
            TraceStep::Layer { start, len } => {
                state.apply_adjacent_disjoint_gates(&remapped[*start..*start + *len])?;
                for _ in 0..*len {
                    self.apply_next(state, gates, false)?;
                }
                Ok(())
            }
        }
    }

    fn apply_next(
        &mut self,
        state: &mut MpsState,
        gates: &[Gate],
        exec: bool,
    ) -> Result<(), EngineError> {
        let range = self.groups.get(self.next_group).cloned().ok_or_else(|| {
            EngineError::ExecutionFailed("contraction tree is not the mps sweep".into())
        })?;
        self.next_group += 1;
        for index in range {
            self.replay(index, state, gates, exec)?;
        }
        Ok(())
    }

    fn replay(
        &mut self,
        index: usize,
        state: &mut MpsState,
        gates: &[Gate],
        exec: bool,
    ) -> Result<(), EngineError> {
        let node = self.nodes[index];
        match node {
            ContractionNode::Site { tensor, gate } => {
                let wire = self.wire_of(tensor)?;
                if exec {
                    let site = self.site_of_wire(wire)?;
                    state.apply_unary_at(site, &gates[gate as usize])?;
                }
                self.node_wire[index] = Some(wire);
            }
            ContractionNode::CcnotH { tensor, gate: _ } => {
                let wire = self.wire_of(tensor)?;
                if exec {
                    state.apply_unary_at(self.site_of_wire(wire)?, &Gate::H(0))?;
                }
                self.node_wire[index] = Some(wire);
            }
            ContractionNode::CcnotT { tensor, gate: _ } => {
                let wire = self.wire_of(tensor)?;
                if exec {
                    state.apply_unary_at(self.site_of_wire(wire)?, &Gate::T(0))?;
                }
                self.node_wire[index] = Some(wire);
            }
            ContractionNode::CcnotTdg { tensor, gate: _ } => {
                let wire = self.wire_of(tensor)?;
                if exec {
                    state.apply_tdg_at(self.site_of_wire(wire)?)?;
                }
                self.node_wire[index] = Some(wire);
            }
            ContractionNode::Swap { left, right } => {
                let (wl, wr) = self.adjacent(left, right)?;
                if exec {
                    state.apply_swap_at(self.site_of_wire(wl)?)?;
                }
                self.exchange(wl, wr);
                self.left_wire[index] = Some(wr);
                self.right_wire[index] = Some(wl);
            }
            ContractionNode::Pair {
                left,
                right,
                gate,
                ctrl_on_left,
            } => {
                let (wl, wr) = self.adjacent(left, right)?;
                if exec {
                    state.apply_pair_at(
                        self.site_of_wire(wl)?,
                        &gates[gate as usize],
                        ctrl_on_left,
                    )?;
                }
                self.left_wire[index] = Some(wl);
                self.right_wire[index] = Some(wr);
            }
            ContractionNode::CcnotCnot {
                left,
                right,
                gate: _,
                ctrl_on_left,
            } => {
                let (wl, wr) = self.adjacent(left, right)?;
                if exec {
                    state.apply_cnot_at(self.site_of_wire(wl)?, ctrl_on_left)?;
                }
                self.left_wire[index] = Some(wl);
                self.right_wire[index] = Some(wr);
            }
        }
        Ok(())
    }

    fn adjacent(&self, left: TensorRef, right: TensorRef) -> Result<(u32, u32), EngineError> {
        let wl = self.wire_of(left)?;
        let wr = self.wire_of(right)?;
        let pl = self.site_of_wire(wl)?;
        let pr = self.site_of_wire(wr)?;
        if pr != pl + 1 {
            return Err(EngineError::ExecutionFailed(
                "contraction tree is not the mps sweep".into(),
            ));
        }
        Ok((wl, wr))
    }

    fn exchange(&mut self, wl: u32, wr: u32) {
        let pl = self.site_of[wl as usize];
        let pr = self.site_of[wr as usize];
        self.wire_at[pl] = wr;
        self.wire_at[pr] = wl;
        self.site_of[wr as usize] = pl;
        self.site_of[wl as usize] = pr;
    }

    fn wire_of(&self, tensor: TensorRef) -> Result<u32, EngineError> {
        let wire = match tensor {
            TensorRef::Ket0(wire) => Some(wire),
            TensorRef::Node(index) => self.node_wire.get(index as usize).copied().flatten(),
            TensorRef::Left(index) => self.left_wire.get(index as usize).copied().flatten(),
            TensorRef::Right(index) => self.right_wire.get(index as usize).copied().flatten(),
        };
        wire.ok_or_else(|| {
            EngineError::ExecutionFailed("contraction tree is not the mps sweep".into())
        })
    }

    fn site_of_wire(&self, wire: u32) -> Result<usize, EngineError> {
        self.site_of.get(wire as usize).copied().ok_or_else(|| {
            EngineError::ExecutionFailed("contraction tree is not the mps sweep".into())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::IfParams;
    use crate::tn::contract_slice;
    use crate::tn::dense::DenseTnState;
    use crate::tn::gates::exact_bond_dim;

    fn ghz(n: usize) -> Vec<Gate> {
        let mut gates = vec![Gate::H(0)];
        for q in 0..n - 1 {
            gates.push(Gate::CNOT(q, q + 1));
        }
        gates
    }

    fn ry_cnot_brickwork(n: usize, depth: usize) -> Vec<Gate> {
        let mut gates = Vec::new();
        for layer in 0..depth {
            for q in 0..n {
                gates.push(Gate::RY(q, 0.17 + 0.01 * (q + layer) as f64));
            }
            let mut q = layer % 2;
            while q + 1 < n {
                gates.push(Gate::CNOT(q, q + 1));
                q += 2;
            }
        }
        gates
    }

    fn dense_zero(n: usize, gates: &[Gate]) -> Complex64 {
        let mut dense = DenseTnState::try_new(n).expect("dense");
        for gate in gates {
            dense.apply_gate(gate);
        }
        dense.state[0]
    }

    fn mps_zero(n: usize, gates: &[Gate], chi: usize) -> Complex64 {
        let mut mps = MpsState::try_new_with_bond(n, chi).expect("mps");
        for gate in gates {
            mps.apply_gate(gate).expect("apply");
        }
        mps.amplitude_at_compact_zero()
    }

    fn assert_close(got: Complex64, expected: Complex64, tol: f64) {
        assert!((got - expected).norm() < tol, "{got} vs {expected}");
    }

    #[test]
    fn sweep_tree_ghz_splits_after_each_cnot() {
        let gates = ghz(3);
        let tree = sweep_tree(3, &gates, &[]).expect("tree");
        assert!(tree.site_order.is_empty());
        assert_eq!(
            tree.nodes,
            vec![
                ContractionNode::Site {
                    tensor: TensorRef::Ket0(0),
                    gate: 0,
                },
                ContractionNode::Pair {
                    left: TensorRef::Node(0),
                    right: TensorRef::Ket0(1),
                    gate: 1,
                    ctrl_on_left: true,
                },
                ContractionNode::Pair {
                    left: TensorRef::Right(1),
                    right: TensorRef::Ket0(2),
                    gate: 2,
                    ctrl_on_left: true,
                },
            ]
        );
    }

    #[test]
    fn nonadjacent_cnot_swaps_along_site_order() {
        let gates = vec![Gate::CNOT(0, 2)];
        let tree = sweep_tree(3, &gates, &[]).expect("tree");
        assert_eq!(
            tree.nodes,
            vec![
                ContractionNode::Swap {
                    left: TensorRef::Ket0(1),
                    right: TensorRef::Ket0(2),
                },
                ContractionNode::Pair {
                    left: TensorRef::Ket0(0),
                    right: TensorRef::Left(0),
                    gate: 0,
                    ctrl_on_left: true,
                },
                ContractionNode::Swap {
                    left: TensorRef::Right(1),
                    right: TensorRef::Right(0),
                },
            ]
        );
        let reversed = sweep_tree(3, &gates, &[2, 1, 0]).expect("reversed");
        assert_ne!(tree.nodes, reversed.nodes);
        assert!(reversed
            .nodes
            .iter()
            .any(|node| matches!(node, ContractionNode::Swap { .. })));
    }

    #[test]
    fn site_order_changes_an_adjacent_cnot() {
        let gates = vec![Gate::CNOT(0, 1)];
        let identity = sweep_tree(3, &gates, &[]).expect("id");
        let flipped = sweep_tree(3, &gates, &[1, 0, 2]).expect("flip");
        assert_ne!(identity.nodes, flipped.nodes);
        assert_eq!(
            flipped.nodes,
            vec![ContractionNode::Pair {
                left: TensorRef::Ket0(1),
                right: TensorRef::Ket0(0),
                gate: 0,
                ctrl_on_left: false,
            }]
        );
    }

    #[test]
    fn identity_site_order_is_omitted_and_round_trips() {
        let gates = vec![Gate::H(0)];
        let tree = sweep_tree(1, &gates, &[0]).expect("tree");
        assert!(tree.site_order.is_empty());
        let json = serde_json::to_string(&tree).expect("json");
        assert!(!json.contains("site_order"));
        assert!(json.contains("ket0"));
        let back: ContractionTree = serde_json::from_str(&json).expect("parse");
        assert_eq!(back, tree);
    }

    #[test]
    fn ghz_tree_matches_dense_and_contract_slice() {
        let n = 3usize;
        let gates = ghz(n);
        let chi = exact_bond_dim(n);
        let tree = sweep_tree(n, &gates, &[]).expect("tree");
        let mut explicit = tree.clone();
        explicit.site_order = vec![0, 1, 2];

        let mut mps = MpsState::try_new_with_bond(n, chi).expect("mps");
        let (tree_amp, tree_trace) =
            contract_tree(n, &gates, &explicit, None, &mut mps).expect("tree");
        let mut slice = MpsState::try_new_with_bond(n, chi).expect("mps");
        let (slice_amp, slice_trace) =
            contract_slice(n, &gates, &[], n, &mut slice).expect("slice");

        let dense = dense_zero(n, &gates);
        assert_close(Complex64::new(tree_amp.real, tree_amp.imag), dense, 1e-8);
        assert_close(Complex64::new(slice_amp.real, slice_amp.imag), dense, 1e-8);
        assert_eq!(tree_trace.len(), slice_trace.len());
        for (got, expected) in tree_trace.iter().zip(&slice_trace) {
            assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
        }
    }

    #[test]
    fn reversed_site_order_keeps_parent_wires() {
        let n = 4usize;
        let gates = vec![
            Gate::H(0),
            Gate::H(1),
            Gate::CNOT(0, 2),
            Gate::CZ(3, 1),
            Gate::CNOT(2, 0),
        ];
        let order = vec![2u32, 0, 3, 1];
        let tree = sweep_tree(n, &gates, &order).expect("tree");
        assert_eq!(tree.site_order, order);
        assert_eq!(
            tree.nodes[0],
            ContractionNode::Site {
                tensor: TensorRef::Ket0(0),
                gate: 0,
            }
        );
        let identity = sweep_tree(n, &gates, &[]).expect("identity");
        assert_ne!(tree.nodes, identity.nodes);

        let chi = exact_bond_dim(n);
        let mut mps = MpsState::try_new_with_bond(n, chi).expect("mps");
        let got = contract_tree_amplitude(n, &gates, &tree, &mut mps).expect("tree");
        let mut dense = DenseTnState::try_new(n).expect("dense");
        for gate in &gates {
            dense.apply_gate(gate);
        }
        assert_close(got, dense.state[0], 1e-8);
        assert_close(got, mps_zero(n, &gates, chi), 1e-8);

        let site_state = mps.to_dense_state();
        for site_bits in 0..site_state.len() {
            let mut logical_bits = 0usize;
            for (site, &wire) in order.iter().enumerate() {
                let bit = (site_bits >> site) & 1;
                logical_bits |= bit << wire as usize;
            }
            let got = site_state[site_bits];
            let expected = dense.state[logical_bits];
            assert!(
                (got - expected).norm() < 1e-8,
                "site {site_bits}: {got} vs logical {logical_bits} {expected}"
            );
        }
    }

    #[test]
    fn non_sweep_tree_and_bad_dispatch_are_rejected() {
        let gates = vec![Gate::H(0)];
        let mut tree = sweep_tree(1, &gates, &[]).expect("tree");
        tree.nodes[0] = ContractionNode::Swap {
            left: TensorRef::Ket0(0),
            right: TensorRef::Ket0(0),
        };
        let mut mps = MpsState::try_new_with_bond(1, 2).expect("mps");
        let err = contract_tree(1, &gates, &tree, None, &mut mps).unwrap_err();
        assert!(err.to_string().contains("not the mps sweep"));

        tree = sweep_tree(1, &gates, &[]).expect("tree");
        let err = validate_sweep_dispatch(&SweepDispatch {
            scalar: true,
            qubit_count: 1,
            original_qubit_count: 1,
            assignments_empty: false,
            time_split: false,
            mps_site_order: None,
            gates: &gates,
            tree: &tree,
        })
        .unwrap_err();
        assert!(err.to_string().contains("slice assignments"));

        let err = validate_sweep_dispatch(&SweepDispatch {
            scalar: true,
            qubit_count: 1,
            original_qubit_count: 2,
            assignments_empty: true,
            time_split: false,
            mps_site_order: None,
            gates: &gates,
            tree: &tree,
        })
        .unwrap_err();
        assert!(err.to_string().contains("parent register"));

        let wide = vec![Gate::H(0), Gate::CNOT(0, 1)];
        let wide_tree = sweep_tree(2, &wide, &[]).expect("tree");
        let mut wide_mps = MpsState::try_new_with_bond(2, 4).expect("mps");
        let err = contract_tree(2, &wide, &wide_tree, Some(&[]), &mut wide_mps);
        assert!(
            err.is_ok(),
            "empty order is the identity, and so is the tree"
        );

        let flipped = sweep_tree(2, &wide, &[1, 0]).expect("flip");
        let err = contract_tree(2, &wide, &flipped, Some(&[]), &mut wide_mps).unwrap_err();
        assert!(err.to_string().contains("does not match"));

        assert!(sweep_tree(2, &[Gate::CNOT(0, 0)], &[]).is_err());
        let bad_if = Gate::IF(IfParams {
            cbit: 0,
            value: 2,
            gate: Box::new(Gate::H(0)),
        });
        let err = sweep_tree(1, &[bad_if], &[]).unwrap_err();
        assert!(err.to_string().contains("IF value"));
        assert!(sweep_tree(1, &[Gate::RESET(3)], &[]).is_err());
    }

    #[test]
    fn brickwork_16q_depth_16_matches_dense_and_mps() {
        // The full depth ranks past χ=128. χ=256 keeps the sweep exact, which
        // is the bond the time-split tests already require for this circuit.
        let n = 16usize;
        let gates = ry_cnot_brickwork(n, 16);
        let chi = 256usize;
        let tree = sweep_tree(n, &gates, &[]).expect("tree");
        let mut mps = MpsState::try_new_with_bond(n, chi).expect("mps");
        let got = contract_tree_amplitude(n, &gates, &tree, &mut mps).expect("tree");
        assert_close(got, mps_zero(n, &gates, chi), 1e-8);
        assert_close(got, dense_zero(n, &gates), 1e-8);
    }
}
