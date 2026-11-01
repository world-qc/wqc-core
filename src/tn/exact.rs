//! Exact contraction of a `Contract` tree.
//!
//! A state tensor is a dense array indexed by its open parent wires. One leg
//! has dimension 2, so `legs` wires occupy `16 << legs` bytes. A node
//! consumes two tensors and stores one result. Nothing is factored with an
//! SVD and no bond is cut.
//!
//! A gate is not a second buffer. It is applied in place once every wire it
//! touches is already open, and those wires stay open. Gates are consumed in
//! circuit order. Two state tensors with no wire in common are multiplied.
//! While that product is allocated the inputs are still alive, so the declared
//! `peak_bytes` counts the other live tensors, both inputs, and the result.
//! A gate apply does not add a second copy.

use std::collections::BTreeMap;

use crate::engine::{Circuit, EngineError, Gate};
use num_complex::Complex64;

use super::gates::{two_qubit_matrix, unary_matrix};
use super::tree::{
    canonicalize_site_order, unitary_wires, ContractionNode, ContractionTree, TensorRef,
};

/// One state tensor may have this many open wires: `2^26 × 16` = 1 GiB.
pub(crate) const MAX_OPEN_LEGS: usize = 26;
/// Live state tensors together stay within one maximum tensor.
pub(crate) const MAX_PEAK_BYTES: u64 = 1 << 30;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RefId {
    Ket(u32),
    Gate(u32),
    Node(u32),
}

enum Live {
    /// Open wires, sorted. `data` is absent while only the peak is walked.
    State {
        wires: Vec<u32>,
        data: Option<Vec<Complex64>>,
    },
    Gate {
        index: usize,
    },
}

pub(crate) fn validate(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    mps_site_order: Option<&[usize]>,
) -> Result<(), EngineError> {
    // Shape only. The node walk happens once, inside [`contract_exact`].
    declared_peak(tree)?;
    check_shape(qubit_count, gates, tree, mps_site_order)?;
    Ok(())
}

pub(crate) fn contract_exact(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    mps_site_order: Option<&[usize]>,
) -> Result<Complex64, EngineError> {
    contract_exact_at(qubit_count, gates, tree, mps_site_order, 0)
}

/// Amplitude of one computational-basis vector. Bit `q` of `bits` is wire `q`.
pub(crate) fn contract_exact_at(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    mps_site_order: Option<&[usize]>,
    bits: usize,
) -> Result<Complex64, EngineError> {
    let declared = declared_peak(tree)?;
    let (peak, amp) = walk(qubit_count, gates, tree, mps_site_order, true, bits)?;
    if declared != peak {
        return Err(EngineError::ExecutionFailed(
            "contraction tree peak does not match".into(),
        ));
    }
    Ok(amp)
}

fn declared_peak(tree: &ContractionTree) -> Result<u64, EngineError> {
    let declared = tree.peak_bytes.ok_or_else(|| {
        EngineError::ExecutionFailed("exact contraction tree requires a peak".into())
    })?;
    if declared > MAX_PEAK_BYTES {
        return Err(EngineError::ExecutionFailed(
            "contraction tree peak exceeds 1 GiB".into(),
        ));
    }
    Ok(declared)
}

fn check_shape(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    mps_site_order: Option<&[usize]>,
) -> Result<(), EngineError> {
    if qubit_count == 0 {
        return Err(EngineError::InvalidQubitCount(qubit_count));
    }
    let mut circuit = Circuit::new(qubit_count);
    for gate in gates {
        circuit.add(gate.clone())?;
    }
    let tree_order = canonicalize_site_order(qubit_count, &tree.site_order)?;
    if !tree_order.is_empty() {
        return Err(EngineError::ExecutionFailed(
            "exact contraction keeps parent wire numbers".into(),
        ));
    }
    if let Some(order) = mps_site_order {
        let as_u32 = order
            .iter()
            .map(|&q| {
                u32::try_from(q)
                    .map_err(|_| EngineError::ExecutionFailed("qubit index exceeds u32".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let task_order = canonicalize_site_order(qubit_count, &as_u32)?;
        if !task_order.is_empty() {
            return Err(EngineError::ExecutionFailed(
                "mps_site_order does not match the contraction tree".into(),
            ));
        }
    }
    Ok(())
}

fn walk(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    mps_site_order: Option<&[usize]>,
    eval: bool,
    bits: usize,
) -> Result<(u64, Complex64), EngineError> {
    check_shape(qubit_count, gates, tree, mps_site_order)?;

    let mut walk = Walk::new(qubit_count, gates, eval, tree.nodes.len());
    let mut peak = 0u64;

    for (index, node) in tree.nodes.iter().enumerate() {
        let ContractionNode::Contract { left, right } = *node else {
            return Err(EngineError::ExecutionFailed(
                "contraction tree mixes sweep steps and exact contractions".into(),
            ));
        };
        let left_id = ref_id(left)?;
        let right_id = ref_id(right)?;
        walk.materialize(left_id)?;
        walk.materialize(right_id)?;
        let left_tensor = walk.take(left_id)?;
        let right_tensor = walk.take(right_id)?;
        // An outer product allocates the result before the inputs are dropped.
        // A gate is written into the state buffer, so it adds no second copy.
        let allocating = matches!(
            (&left_tensor, &right_tensor),
            (Live::State { .. }, Live::State { .. })
        );
        let inputs = add_bytes(occupied(&left_tensor)?, occupied(&right_tensor)?)?;
        let result = combine(left_tensor, right_tensor, gates, eval)?;
        let mut during = add_bytes(live_bytes(&walk.live)?, inputs)?;
        if allocating {
            during = add_bytes(during, occupied(&result)?)?;
        }
        if during > MAX_PEAK_BYTES {
            return Err(EngineError::ExecutionFailed(
                "contraction tree peak exceeds 1 GiB".into(),
            ));
        }
        peak = peak.max(during);
        walk.live.insert(RefId::Node(index as u32), result);
    }

    for (index, gate) in gates.iter().enumerate() {
        if unitary_wires(gate, qubit_count, index)?.is_some() && !walk.consumed_gate[index] {
            return Err(EngineError::ExecutionFailed(
                "gate was not contracted".into(),
            ));
        }
    }

    let amp = if eval {
        sample(&walk.live, bits, qubit_count)?
    } else {
        Complex64::ZERO
    };
    Ok((peak, amp))
}

fn sample(
    live: &BTreeMap<RefId, Live>,
    bits: usize,
    qubit_count: usize,
) -> Result<Complex64, EngineError> {
    let mut covered = vec![false; qubit_count];
    let mut amp = Complex64::ONE;
    for tensor in live.values() {
        match tensor {
            Live::Gate { .. } => {
                return Err(EngineError::ExecutionFailed(
                    "gate was not contracted".into(),
                ));
            }
            Live::State { wires, data } => {
                for &wire in wires {
                    if let Some(slot) = covered.get_mut(wire as usize) {
                        *slot = true;
                    }
                }
                let data = data.as_ref().ok_or_else(|| {
                    EngineError::ExecutionFailed("contraction tree tensor is not ready".into())
                })?;
                amp *= component(wires, data, bits);
            }
        }
    }
    for (wire, seen) in covered.iter().enumerate() {
        if wire < usize::BITS as usize && !seen && (bits >> wire) & 1 == 1 {
            return Ok(Complex64::ZERO);
        }
    }
    Ok(amp)
}

fn component(wires: &[u32], data: &[Complex64], bits: usize) -> Complex64 {
    let mut index = 0usize;
    for (pos, &wire) in wires.iter().enumerate() {
        if wire < usize::BITS && (bits >> wire) & 1 == 1 {
            index |= 1usize << pos;
        }
    }
    data.get(index).copied().unwrap_or(Complex64::ZERO)
}

fn ref_id(tensor: TensorRef) -> Result<RefId, EngineError> {
    match tensor {
        TensorRef::Ket0(wire) => Ok(RefId::Ket(wire)),
        TensorRef::Gate(index) => Ok(RefId::Gate(index)),
        TensorRef::Node(index) => Ok(RefId::Node(index)),
        TensorRef::Left(_) | TensorRef::Right(_) => Err(EngineError::ExecutionFailed(
            "exact contraction has no split".into(),
        )),
    }
}

struct Walk<'a> {
    qubit_count: usize,
    gates: &'a [Gate],
    eval: bool,
    live: BTreeMap<RefId, Live>,
    consumed_nodes: Vec<bool>,
    consumed_gate: Vec<bool>,
    consumed_ket: Vec<bool>,
    /// First gate index that has not been consumed yet.
    next_gate: usize,
}

impl<'a> Walk<'a> {
    fn new(qubit_count: usize, gates: &'a [Gate], eval: bool, node_count: usize) -> Self {
        Self {
            qubit_count,
            gates,
            eval,
            live: BTreeMap::new(),
            consumed_nodes: vec![false; node_count],
            consumed_gate: vec![false; gates.len()],
            consumed_ket: vec![false; qubit_count],
            next_gate: 0,
        }
    }

    fn materialize(&mut self, id: RefId) -> Result<(), EngineError> {
        if self.live.contains_key(&id) {
            return Ok(());
        }
        match id {
            RefId::Ket(wire) => {
                let wire_index = wire as usize;
                if wire_index >= self.qubit_count {
                    return Err(EngineError::QubitIndexOutOfBounds {
                        index: wire_index,
                        limit: self.qubit_count,
                    });
                }
                if self.consumed_ket[wire_index] {
                    return Err(EngineError::ExecutionFailed(
                        "contraction tree reuses a tensor".into(),
                    ));
                }
                self.consumed_ket[wire_index] = true;
                let data = self.eval.then(|| vec![Complex64::ONE, Complex64::ZERO]);
                self.live.insert(
                    id,
                    Live::State {
                        wires: vec![wire],
                        data,
                    },
                );
            }
            RefId::Gate(index) => {
                let gate_index = index as usize;
                let gate = self.gates.get(gate_index).ok_or_else(|| {
                    EngineError::ExecutionFailed("contraction tree gate is missing".into())
                })?;
                if unitary_wires(gate, self.qubit_count, gate_index)?.is_none() {
                    return Err(EngineError::ExecutionFailed(format!(
                        "gate {gate_index} is not a contraction leaf"
                    )));
                }
                let expected = self.next_unitary()?;
                if gate_index != expected {
                    return Err(EngineError::ExecutionFailed(
                        "gates are out of circuit order".into(),
                    ));
                }
                self.next_gate = gate_index + 1;
                if self.consumed_gate[gate_index] {
                    return Err(EngineError::ExecutionFailed(
                        "contraction tree reuses a tensor".into(),
                    ));
                }
                self.consumed_gate[gate_index] = true;
                self.live.insert(id, Live::Gate { index: gate_index });
            }
            RefId::Node(index) => {
                // Absent from `live`: either this node has not been built, or an
                // earlier node already consumed it.
                if self
                    .consumed_nodes
                    .get(index as usize)
                    .copied()
                    .unwrap_or(false)
                {
                    return Err(EngineError::ExecutionFailed(
                        "contraction tree reuses a tensor".into(),
                    ));
                }
                return Err(EngineError::ExecutionFailed(
                    "contraction tree tensor is not ready".into(),
                ));
            }
        }
        Ok(())
    }

    fn take(&mut self, id: RefId) -> Result<Live, EngineError> {
        let tensor = self.live.remove(&id).ok_or_else(|| {
            EngineError::ExecutionFailed("contraction tree tensor is not ready".into())
        })?;
        if let RefId::Node(index) = id {
            let slot = self.consumed_nodes.get_mut(index as usize).ok_or_else(|| {
                EngineError::ExecutionFailed("contraction tree tensor is not ready".into())
            })?;
            if *slot {
                return Err(EngineError::ExecutionFailed(
                    "contraction tree reuses a tensor".into(),
                ));
            }
            *slot = true;
        }
        Ok(tensor)
    }

    fn next_unitary(&self) -> Result<usize, EngineError> {
        for index in self.next_gate..self.gates.len() {
            if unitary_wires(&self.gates[index], self.qubit_count, index)?.is_some() {
                return Ok(index);
            }
        }
        Err(EngineError::ExecutionFailed(
            "gates are out of circuit order".into(),
        ))
    }
}

fn combine(left: Live, right: Live, gates: &[Gate], eval: bool) -> Result<Live, EngineError> {
    match (left, right) {
        (Live::Gate { index }, state) | (state, Live::Gate { index }) => {
            let Live::State { wires, data } = state else {
                return Err(EngineError::ExecutionFailed(
                    "exact contraction does not contract two gates".into(),
                ));
            };
            apply_gate(&wires, data, &gates[index], eval)
        }
        (
            Live::State {
                wires: left_wires,
                data: left_data,
            },
            Live::State {
                wires: right_wires,
                data: right_data,
            },
        ) => outer(&left_wires, left_data, &right_wires, right_data, eval),
    }
}

fn apply_gate(
    wires: &[u32],
    data: Option<Vec<Complex64>>,
    gate: &Gate,
    eval: bool,
) -> Result<Live, EngineError> {
    for wire in gate_wires(gate)? {
        if !wires.contains(&wire) {
            return Err(EngineError::ExecutionFailed(
                "gate wires are not open".into(),
            ));
        }
    }
    let data = if eval {
        let mut data = data.ok_or_else(|| {
            EngineError::ExecutionFailed("contraction tree tensor is not ready".into())
        })?;
        apply_in_place(&mut data, wires, gate)?;
        Some(data)
    } else {
        None
    };
    Ok(Live::State {
        wires: wires.to_vec(),
        data,
    })
}

fn outer(
    left_wires: &[u32],
    left_data: Option<Vec<Complex64>>,
    right_wires: &[u32],
    right_data: Option<Vec<Complex64>>,
    eval: bool,
) -> Result<Live, EngineError> {
    if left_wires.iter().any(|wire| right_wires.contains(wire)) {
        return Err(EngineError::ExecutionFailed("tensors share a wire".into()));
    }
    let mut wires = Vec::with_capacity(left_wires.len() + right_wires.len());
    wires.extend_from_slice(left_wires);
    wires.extend_from_slice(right_wires);
    wires.sort_unstable();
    reject_legs(wires.len())?;
    let data = if eval {
        let left = left_data.ok_or_else(|| {
            EngineError::ExecutionFailed("contraction tree tensor is not ready".into())
        })?;
        let right = right_data.ok_or_else(|| {
            EngineError::ExecutionFailed("contraction tree tensor is not ready".into())
        })?;
        let left_pos = positions(left_wires, &wires);
        let right_pos = positions(right_wires, &wires);
        let mut data = vec![Complex64::ZERO; 1usize << wires.len()];
        for (left_index, left_amp) in left.iter().enumerate() {
            for (right_index, right_amp) in right.iter().enumerate() {
                let index = scatter(left_index, &left_pos) | scatter(right_index, &right_pos);
                data[index] = left_amp * right_amp;
            }
        }
        Some(data)
    } else {
        None
    };
    Ok(Live::State { wires, data })
}

fn apply_in_place(data: &mut [Complex64], wires: &[u32], gate: &Gate) -> Result<(), EngineError> {
    if data.len() != 1usize << wires.len() {
        return Err(EngineError::ExecutionFailed(
            "contraction tree tensor is not ready".into(),
        ));
    }
    match gate {
        Gate::H(_)
        | Gate::X(_)
        | Gate::Y(_)
        | Gate::Z(_)
        | Gate::S(_)
        | Gate::T(_)
        | Gate::RX(_, _)
        | Gate::RY(_, _)
        | Gate::RZ(_, _) => {
            let wire = gate_wires(gate)?[0];
            let bit = 1usize << position(wires, wire)?;
            let mat = unary_matrix(gate);
            let mut index = 0;
            while index < data.len() {
                if index & bit == 0 {
                    let up = index | bit;
                    let v0 = data[index];
                    let v1 = data[up];
                    data[index] = mat[0][0] * v0 + mat[0][1] * v1;
                    data[up] = mat[1][0] * v0 + mat[1][1] * v1;
                }
                index += 1;
            }
        }
        Gate::CNOT(_, _) | Gate::CZ(_, _) => {
            let wires_g = gate_wires(gate)?;
            let ctrl = 1usize << position(wires, wires_g[0])?;
            let tgt = 1usize << position(wires, wires_g[1])?;
            let mat = two_qubit_matrix(gate);
            let mut index = 0;
            while index < data.len() {
                if index & ctrl == 0 && index & tgt == 0 {
                    let i_c = index | ctrl;
                    let i_t = index | tgt;
                    let i_ct = index | ctrl | tgt;
                    let v00 = data[index];
                    let v10 = data[i_c];
                    let v01 = data[i_t];
                    let v11 = data[i_ct];
                    let amp = |co: usize, to: usize| {
                        mat[co][to][0][0] * v00
                            + mat[co][to][1][0] * v10
                            + mat[co][to][0][1] * v01
                            + mat[co][to][1][1] * v11
                    };
                    data[index] = amp(0, 0);
                    data[i_c] = amp(1, 0);
                    data[i_t] = amp(0, 1);
                    data[i_ct] = amp(1, 1);
                }
                index += 1;
            }
        }
        Gate::CCNOT(c1, c2, target) => {
            let m1 = 1usize << position(wires, *c1 as u32)?;
            let m2 = 1usize << position(wires, *c2 as u32)?;
            let mt = 1usize << position(wires, *target as u32)?;
            let mut index = 0;
            while index < data.len() {
                if index & m1 != 0 && index & m2 != 0 && index & mt == 0 {
                    data.swap(index, index | mt);
                }
                index += 1;
            }
        }
        Gate::MEASURE(_) | Gate::RESET(_) | Gate::IF(_) => {
            return Err(EngineError::ExecutionFailed(
                "gate is not a contraction leaf".into(),
            ));
        }
    }
    Ok(())
}

fn gate_wires(gate: &Gate) -> Result<Vec<u32>, EngineError> {
    let wires: Vec<u32> = match gate {
        Gate::H(t)
        | Gate::X(t)
        | Gate::Y(t)
        | Gate::Z(t)
        | Gate::S(t)
        | Gate::T(t)
        | Gate::RX(t, _)
        | Gate::RY(t, _)
        | Gate::RZ(t, _) => vec![*t as u32],
        Gate::CNOT(c, t) | Gate::CZ(c, t) => vec![*c as u32, *t as u32],
        Gate::CCNOT(c1, c2, t) => vec![*c1 as u32, *c2 as u32, *t as u32],
        Gate::MEASURE(_) | Gate::RESET(_) | Gate::IF(_) => {
            return Err(EngineError::ExecutionFailed(
                "gate is not a contraction leaf".into(),
            ));
        }
    };
    Ok(wires)
}

fn position(wires: &[u32], wire: u32) -> Result<usize, EngineError> {
    wires
        .iter()
        .position(|&open| open == wire)
        .ok_or_else(|| EngineError::ExecutionFailed("gate wires are not open".into()))
}

fn positions(src: &[u32], dst: &[u32]) -> Vec<usize> {
    src.iter()
        .map(|wire| {
            dst.iter()
                .position(|open| open == wire)
                .expect("outer wires are the union")
        })
        .collect()
}

fn scatter(index: usize, pos: &[usize]) -> usize {
    let mut out = 0;
    for (bit, &place) in pos.iter().enumerate() {
        if (index >> bit) & 1 == 1 {
            out |= 1usize << place;
        }
    }
    out
}

fn reject_legs(legs: usize) -> Result<(), EngineError> {
    if legs > MAX_OPEN_LEGS {
        return Err(EngineError::ExecutionFailed(
            "contraction tree tensor exceeds 26 legs".into(),
        ));
    }
    Ok(())
}

fn live_bytes(live: &BTreeMap<RefId, Live>) -> Result<u64, EngineError> {
    let mut sum = 0u64;
    for tensor in live.values() {
        sum = add_bytes(sum, occupied(tensor)?)?;
    }
    Ok(sum)
}

fn occupied(tensor: &Live) -> Result<u64, EngineError> {
    match tensor {
        Live::Gate { .. } => Ok(0),
        Live::State { wires, .. } => {
            reject_legs(wires.len())?;
            Ok(16u64 << wires.len())
        }
    }
}

fn add_bytes(left: u64, right: u64) -> Result<u64, EngineError> {
    left.checked_add(right)
        .ok_or_else(|| EngineError::ExecutionFailed("contraction tree peak exceeds 1 GiB".into()))
}

/// Peak of `tree` before the declared value is checked. Tests use this to
/// fill `peak_bytes`.
fn measured_peak(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    mps_site_order: Option<&[usize]>,
) -> Result<u64, EngineError> {
    Ok(walk(qubit_count, gates, tree, mps_site_order, false, 0)?.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineError;
    use crate::engine::IfParams;
    use crate::tn::dense::DenseTnState;
    use crate::tn::tree::{contract_tree, is_contraction_client_error, sweep_tree};
    use crate::tn::{contract_slice, MpsState};

    fn star_gates(n: usize) -> Vec<Gate> {
        let mut gates = vec![Gate::H(0)];
        for q in 1..n {
            gates.push(Gate::CNOT(0, q));
        }
        gates
    }

    /// Center qubit stays in one tensor. Each leaf is absorbed, then the CNOT
    /// is applied. No swap and no split.
    fn star_tree(n: usize, gates: &[Gate]) -> ContractionTree {
        let mut nodes = vec![ContractionNode::Contract {
            left: TensorRef::Ket0(0),
            right: TensorRef::Gate(0),
        }];
        for q in 1..n {
            let prev = TensorRef::Node((nodes.len() - 1) as u32);
            nodes.push(ContractionNode::Contract {
                left: prev,
                right: TensorRef::Ket0(q as u32),
            });
            let joined = TensorRef::Node((nodes.len() - 1) as u32);
            nodes.push(ContractionNode::Contract {
                left: joined,
                right: TensorRef::Gate(q as u32),
            });
        }
        let mut tree = ContractionTree {
            site_order: Vec::new(),
            nodes,
            peak_bytes: None,
        };
        tree.peak_bytes = Some(measured_peak(n, gates, &tree, None).expect("peak"));
        tree
    }

    fn assert_close(got: Complex64, expected: Complex64) {
        assert!((got - expected).norm() < 1e-8, "{got} vs {expected}");
    }

    /// Inputs of the last join are still alive while the `n`-leg result is allocated.
    fn star_peak(n: usize) -> u64 {
        if n <= 1 {
            32
        } else {
            (16u64 << (n - 1)) + (16u64 << 1) + (16u64 << n)
        }
    }

    fn dense_state(n: usize, gates: &[Gate]) -> Vec<Complex64> {
        let mut dense = DenseTnState::try_new(n).expect("dense");
        for gate in gates {
            dense.apply_gate(gate);
        }
        dense.state.to_vec()
    }

    #[test]
    fn star_16_matches_a_leaf_one_and_the_sweep_trace() {
        let n = 16usize;
        let gates = star_gates(n);
        let tree = star_tree(n, &gates);
        assert!(tree
            .nodes
            .iter()
            .all(|node| matches!(node, ContractionNode::Contract { .. })));
        assert_eq!(tree.peak_bytes, Some(star_peak(n)));
        let sweep = sweep_tree(n, &gates, &[]).expect("sweep");
        assert!(sweep
            .nodes
            .iter()
            .any(|node| matches!(node, ContractionNode::Swap { .. })));

        let dense = dense_state(n, &gates);
        // Every leaf is |1⟩, which is the GHZ term the CNOTs copy off the center.
        let ones = (1usize << n) - 1;
        assert!(dense[ones].norm() > 0.1, "leaf-|1⟩ amplitude vanished");
        let got = contract_exact_at(n, &gates, &tree, None, ones).expect("ones");
        assert_close(got, dense[ones]);

        // Schmidt rank of this star is 2. A larger χ leaves SVD dust in the
        // extra bonds, and the n≤16 trace sampler then walks those bonds.
        let chi = 2usize;
        let mut mps = MpsState::try_new_with_bond(n, chi).expect("mps");
        let (amp, trace) = contract_tree(n, &gates, &tree, None, &mut mps).expect("tree");
        assert_close(Complex64::new(amp.real, amp.imag), dense[0]);
        let mut slice = MpsState::try_new_with_bond(n, chi).expect("mps");
        let (_, slice_trace) = contract_slice(n, &gates, &[], n, &mut slice).expect("slice");
        assert_eq!(trace.len(), slice_trace.len());
        for (got, expected) in trace.iter().zip(&slice_trace) {
            assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
        }
    }

    #[test]
    fn ccnot_and_cz_change_a_nonzero_basis_amplitude() {
        let n = 4usize;
        let gates = vec![
            Gate::X(0),
            Gate::X(1),
            Gate::CCNOT(0, 1, 2),
            Gate::X(3),
            Gate::CZ(2, 3),
        ];
        let mut nodes = Vec::new();
        nodes.push(ContractionNode::Contract {
            left: TensorRef::Ket0(0),
            right: TensorRef::Ket0(1),
        });
        nodes.push(ContractionNode::Contract {
            left: TensorRef::Node(0),
            right: TensorRef::Ket0(2),
        });
        nodes.push(ContractionNode::Contract {
            left: TensorRef::Node(1),
            right: TensorRef::Ket0(3),
        });
        for gate in 0..gates.len() as u32 {
            let prev = TensorRef::Node((nodes.len() - 1) as u32);
            nodes.push(ContractionNode::Contract {
                left: prev,
                right: TensorRef::Gate(gate),
            });
        }
        let mut tree = ContractionTree {
            site_order: Vec::new(),
            nodes,
            peak_bytes: None,
        };
        tree.peak_bytes = Some(measured_peak(n, &gates, &tree, None).expect("peak"));
        // Last join is 3 legs + 1 leg while the 4-leg result is allocated.
        assert_eq!(
            tree.peak_bytes,
            Some((16u64 << 3) + (16u64 << 1) + (16u64 << 4))
        );
        let dense = dense_state(n, &gates);
        // Wires 0,1,2,3 are |1⟩. CCNOT flips wire 2, then CZ contributes the minus.
        let bits = 0b1111;
        assert!(dense[bits].norm() > 0.5, "CCNOT/CZ amplitude vanished");
        let got = contract_exact_at(n, &gates, &tree, Some(&[]), bits).expect("bits");
        assert_close(got, dense[bits]);
    }

    #[test]
    fn gates_out_of_circuit_order_are_rejected() {
        let gates = vec![Gate::H(0), Gate::X(0)];
        let tree = ContractionTree {
            site_order: Vec::new(),
            nodes: vec![ContractionNode::Contract {
                left: TensorRef::Ket0(0),
                right: TensorRef::Gate(1),
            }],
            peak_bytes: Some(0),
        };
        let err = contract_exact(1, &gates, &tree, None).unwrap_err();
        assert!(err.to_string().contains("circuit order"));
        assert!(is_contraction_client_error(&err));
    }

    #[test]
    fn untouched_wire_stays_ket0() {
        let gates = vec![Gate::H(0)];
        let mut tree = ContractionTree {
            site_order: Vec::new(),
            nodes: vec![ContractionNode::Contract {
                left: TensorRef::Ket0(0),
                right: TensorRef::Gate(0),
            }],
            peak_bytes: None,
        };
        tree.peak_bytes = Some(measured_peak(3, &gates, &tree, None).expect("peak"));
        let got = contract_exact(3, &gates, &tree, None).expect("exact");
        let dense = dense_state(3, &gates);
        assert_close(got, dense[0]);
    }

    #[test]
    fn over_budget_peak_and_legs_are_rejected() {
        let gates = star_gates(3);
        let mut tree = star_tree(3, &gates);
        tree.peak_bytes = Some(MAX_PEAK_BYTES + 1);
        let err = validate(3, &gates, &tree, None).unwrap_err();
        assert!(err.to_string().contains("exceeds 1 GiB"));

        tree.peak_bytes = Some(1);
        let err = contract_exact(3, &gates, &tree, None).unwrap_err();
        assert!(err.to_string().contains("does not match"));
        assert!(is_contraction_client_error(&err));
        assert!(!is_contraction_client_error(&EngineError::ExecutionFailed(
            "mps bond exceeded chi".into(),
        )));

        let mut wide = star_tree_unmeasured(26);
        wide.peak_bytes = Some(0);
        let err = walk(26, &star_gates(26), &wide, None, false, 0).unwrap_err();
        assert!(err.to_string().contains("exceeds 1 GiB"));

        let err = contract_exact(27, &[], &join_past_26_legs(), None).unwrap_err();
        assert!(err.to_string().contains("exceeds 26 legs"));

        let mut reordered = star_tree(2, &star_gates(2));
        reordered.site_order = vec![1, 0];
        let err = validate(2, &star_gates(2), &reordered, None).unwrap_err();
        assert!(err.to_string().contains("parent wire"));

        let err = validate(
            2,
            &star_gates(2),
            &star_tree(2, &star_gates(2)),
            Some(&[1, 0]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not match"));

        let bad_if = Gate::IF(IfParams {
            cbit: 0,
            value: 2,
            gate: Box::new(Gate::H(0)),
        });
        let err = validate(
            1,
            &[bad_if],
            &ContractionTree {
                site_order: Vec::new(),
                nodes: Vec::new(),
                peak_bytes: Some(0),
            },
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("IF value"));
        assert!(validate(
            1,
            &[Gate::RESET(3)],
            &ContractionTree {
                site_order: Vec::new(),
                nodes: Vec::new(),
                peak_bytes: Some(0),
            },
            None,
        )
        .is_err());
    }

    fn star_tree_unmeasured(n: usize) -> ContractionTree {
        let mut nodes = vec![ContractionNode::Contract {
            left: TensorRef::Ket0(0),
            right: TensorRef::Gate(0),
        }];
        for q in 1..n {
            let prev = TensorRef::Node((nodes.len() - 1) as u32);
            nodes.push(ContractionNode::Contract {
                left: prev,
                right: TensorRef::Ket0(q as u32),
            });
            let joined = TensorRef::Node((nodes.len() - 1) as u32);
            nodes.push(ContractionNode::Contract {
                left: joined,
                right: TensorRef::Gate(q as u32),
            });
        }
        ContractionTree {
            site_order: Vec::new(),
            nodes,
            peak_bytes: None,
        }
    }

    /// Two tensors of 14 and 13 wires. Their product would have 27 legs.
    fn join_past_26_legs() -> ContractionTree {
        let mut nodes = Vec::new();
        let left = join_wires(0u32..14, &mut nodes);
        let right = join_wires(14u32..27, &mut nodes);
        nodes.push(ContractionNode::Contract { left, right });
        ContractionTree {
            site_order: Vec::new(),
            nodes,
            peak_bytes: Some(0),
        }
    }

    fn join_wires(wires: impl Iterator<Item = u32>, nodes: &mut Vec<ContractionNode>) -> TensorRef {
        let mut wires = wires;
        let first = wires.next().expect("wires");
        let mut acc = TensorRef::Ket0(first);
        for wire in wires {
            nodes.push(ContractionNode::Contract {
                left: acc,
                right: TensorRef::Ket0(wire),
            });
            acc = TensorRef::Node((nodes.len() - 1) as u32);
        }
        acc
    }
}
