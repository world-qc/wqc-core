//! Network contraction of `⟨0…0|U|0…0⟩`.
//!
//! The circuit is one tensor network. Wire `w` starts at a `ket0(w)` leaf and
//! ends at a `bra0(w)` leaf. Unitary gate `i` is a `gate(i)` leaf with one
//! input and one output leg on every wire it touches. An edge is one segment
//! of a wire between two consecutive leaves on it, so every edge joins exactly
//! two leaves. The tree names leaves only. Edges follow from the circuit, and
//! MEASURE, RESET, and IF are skipped as in the exact tree.
//!
//! A node contracts two tensors over the edges they share; the result keeps
//! the rest. Two tensors that share nothing are multiplied. Every leaf is used
//! once, every node but the last is used once, and the last node has no legs.
//! Its value is the amplitude.
//!
//! One tensor may have at most 26 legs. A leaf is built when a node takes it.
//! A node first reorders each input so the shared edges come first, in the
//! left input's order, then multiplies. A reordered input is a new copy, made
//! while the original is still allocated. So while a node runs, the live node
//! results and both inputs are allocated, plus the larger of the result and
//! each input that is reordered. `peak_bytes` is the maximum of that sum.
//! An input already in that order is not copied.

use std::collections::BTreeMap;

use crate::engine::{EngineError, Gate};
use ndarray::ArrayView2;
use num_complex::Complex64;
use rayon::prelude::*;

use super::exact::{apply_in_place, check_shape, gate_wires, MAX_OPEN_LEGS, MAX_PEAK_BYTES};
use super::tree::{unitary_wires, ContractionKind, ContractionNode, ContractionTree, TensorRef};

/// Output bits a lookup table covers when legs are reordered.
const PERMUTE_LOW_BITS: usize = 12;

/// Edges of every leaf.
pub(crate) struct Network {
    /// Edge of `bra0(w)`. The edge of `ket0(w)` is `w`.
    bra: Vec<u32>,
    /// Inputs, then outputs, in the gate's wire order. `None` is not unitary.
    gate: Vec<Option<Vec<u32>>>,
}

impl Network {
    /// Edge `w` leaves `ket0(w)`. Each gate numbers its outputs next, in wire
    /// order.
    pub(crate) fn build(qubit_count: usize, gates: &[Gate]) -> Result<Self, EngineError> {
        let mut current: Vec<u32> = (0..qubit_count as u32).collect();
        let mut next = qubit_count as u32;
        let mut legs_of = Vec::with_capacity(gates.len());
        for (index, gate) in gates.iter().enumerate() {
            let Some(wires) = unitary_wires(gate, qubit_count, index)? else {
                legs_of.push(None);
                continue;
            };
            let mut legs: Vec<u32> = wires.iter().map(|&w| current[w]).collect();
            for &w in &wires {
                current[w] = next;
                legs.push(next);
                next = next.checked_add(1).ok_or_else(|| {
                    EngineError::ExecutionFailed("contraction tree has too many edges".into())
                })?;
            }
            legs_of.push(Some(legs));
        }
        Ok(Self {
            bra: current,
            gate: legs_of,
        })
    }

    pub(crate) fn qubit_count(&self) -> usize {
        self.bra.len()
    }

    pub(crate) fn gate_legs(&self, index: usize) -> Option<&[u32]> {
        self.gate.get(index).and_then(|legs| legs.as_deref())
    }

    pub(crate) fn bra_edge(&self, wire: usize) -> u32 {
        self.bra[wire]
    }
}

struct Tensor {
    legs: Vec<u32>,
    data: Option<Vec<Complex64>>,
}

impl Tensor {
    fn bytes(&self) -> u64 {
        16u64 << self.legs.len()
    }
}

pub(crate) fn validate(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    mps_site_order: Option<&[usize]>,
) -> Result<(), EngineError> {
    // No data: the walk only counts legs, so the peak is checked here.
    let declared = declared_peak(tree)?;
    let (peak, _) = walk(qubit_count, gates, tree, mps_site_order, false)?;
    if declared != peak {
        return Err(EngineError::ExecutionFailed(
            "contraction tree peak does not match".into(),
        ));
    }
    Ok(())
}

pub(crate) fn contract_network(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    mps_site_order: Option<&[usize]>,
) -> Result<Complex64, EngineError> {
    let declared = declared_peak(tree)?;
    let (peak, amp) = walk(qubit_count, gates, tree, mps_site_order, true)?;
    if declared != peak {
        return Err(EngineError::ExecutionFailed(
            "contraction tree peak does not match".into(),
        ));
    }
    Ok(amp)
}

/// Peak of `tree` before the declared value is checked.
pub(crate) fn measured_peak(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
) -> Result<u64, EngineError> {
    Ok(walk(qubit_count, gates, tree, None, false)?.0)
}

fn declared_peak(tree: &ContractionTree) -> Result<u64, EngineError> {
    let declared = tree
        .peak_bytes
        .ok_or_else(|| EngineError::ExecutionFailed("contraction tree requires a peak".into()))?;
    if declared > MAX_PEAK_BYTES {
        return Err(EngineError::ExecutionFailed(
            "contraction tree peak exceeds 1 GiB".into(),
        ));
    }
    Ok(declared)
}

fn walk(
    qubit_count: usize,
    gates: &[Gate],
    tree: &ContractionTree,
    mps_site_order: Option<&[usize]>,
    eval: bool,
) -> Result<(u64, Complex64), EngineError> {
    if tree.kind != Some(ContractionKind::Network) {
        return Err(EngineError::ExecutionFailed(
            "contraction tree is not a network".into(),
        ));
    }
    check_shape(qubit_count, gates, tree, mps_site_order)?;
    let network = Network::build(qubit_count, gates)?;
    let mut leaves = Leaves {
        network: &network,
        gates,
        eval,
        ket: vec![false; qubit_count],
        bra: vec![false; qubit_count],
        gate: vec![false; gates.len()],
    };
    let mut live: BTreeMap<usize, Tensor> = BTreeMap::new();
    let mut live_bytes = 0u64;
    let mut peak = 0u64;

    for (index, node) in tree.nodes.iter().enumerate() {
        let ContractionNode::Contract { left, right } = *node else {
            return Err(EngineError::ExecutionFailed(
                "contraction tree mixes sweep steps and network contractions".into(),
            ));
        };
        let left = take(left, index, &mut live, &mut live_bytes, &mut leaves)?;
        let right = take(right, index, &mut live, &mut live_bytes, &mut leaves)?;
        let inputs = left.bytes() + right.bytes();
        let (result, working) = contract_pair(left, right, eval)?;
        let during = live_bytes + inputs + working;
        if during > MAX_PEAK_BYTES {
            return Err(EngineError::ExecutionFailed(
                "contraction tree peak exceeds 1 GiB".into(),
            ));
        }
        peak = peak.max(during);
        live_bytes += result.bytes();
        live.insert(index, result);
    }

    if leaves.ket.iter().chain(&leaves.bra).any(|used| !used)
        || gates
            .iter()
            .enumerate()
            .any(|(i, _)| network.gate_legs(i).is_some() && !leaves.gate[i])
    {
        return Err(EngineError::ExecutionFailed(
            "contraction tree leaves a leaf out".into(),
        ));
    }
    // The last node is never taken, so one live tensor means it is the root.
    let root = match (live.len(), live.into_values().next()) {
        (1, Some(root)) if root.legs.is_empty() => root,
        _ => {
            return Err(EngineError::ExecutionFailed(
                "contraction tree does not end in one scalar".into(),
            ))
        }
    };
    let amp = if eval {
        root.data
            .as_ref()
            .and_then(|d| d.first().copied())
            .ok_or_else(|| {
                EngineError::ExecutionFailed("contraction tree tensor is not ready".into())
            })?
    } else {
        Complex64::ZERO
    };
    Ok((peak, amp))
}

fn take(
    tensor: TensorRef,
    index: usize,
    live: &mut BTreeMap<usize, Tensor>,
    live_bytes: &mut u64,
    leaves: &mut Leaves<'_>,
) -> Result<Tensor, EngineError> {
    match tensor {
        TensorRef::Node(node) => {
            let node = node as usize;
            if node >= index {
                return Err(EngineError::ExecutionFailed(
                    "contraction tree tensor is not ready".into(),
                ));
            }
            let tensor = live.remove(&node).ok_or_else(|| {
                EngineError::ExecutionFailed("contraction tree reuses a tensor".into())
            })?;
            *live_bytes -= tensor.bytes();
            Ok(tensor)
        }
        TensorRef::Ket0(wire) => leaves.ket(wire as usize),
        TensorRef::Bra0(wire) => leaves.bra(wire as usize),
        TensorRef::Gate(gate) => leaves.gate(gate as usize),
        TensorRef::Left(_) | TensorRef::Right(_) => Err(EngineError::ExecutionFailed(
            "contraction tree network has no split".into(),
        )),
    }
}

struct Leaves<'a> {
    network: &'a Network,
    gates: &'a [Gate],
    eval: bool,
    ket: Vec<bool>,
    bra: Vec<bool>,
    gate: Vec<bool>,
}

impl Leaves<'_> {
    fn ket(&mut self, wire: usize) -> Result<Tensor, EngineError> {
        mark(&mut self.ket, wire, self.network.qubit_count())?;
        Ok(self.basis_zero(wire as u32))
    }

    fn bra(&mut self, wire: usize) -> Result<Tensor, EngineError> {
        mark(&mut self.bra, wire, self.network.qubit_count())?;
        Ok(self.basis_zero(self.network.bra_edge(wire)))
    }

    fn basis_zero(&self, edge: u32) -> Tensor {
        Tensor {
            legs: vec![edge],
            data: self.eval.then(|| vec![Complex64::ONE, Complex64::ZERO]),
        }
    }

    fn gate(&mut self, index: usize) -> Result<Tensor, EngineError> {
        let gate = self.gates.get(index).ok_or_else(|| {
            EngineError::ExecutionFailed("contraction tree gate is missing".into())
        })?;
        let legs = self.network.gate_legs(index).ok_or_else(|| {
            EngineError::ExecutionFailed(format!("gate {index} is not a contraction leaf"))
        })?;
        if self.gate[index] {
            return Err(EngineError::ExecutionFailed(
                "contraction tree reuses a tensor".into(),
            ));
        }
        self.gate[index] = true;
        let data = if self.eval {
            Some(gate_tensor(gate)?)
        } else {
            None
        };
        Ok(Tensor {
            legs: legs.to_vec(),
            data,
        })
    }
}

fn mark(used: &mut [bool], wire: usize, qubit_count: usize) -> Result<(), EngineError> {
    let slot = used
        .get_mut(wire)
        .ok_or(EngineError::QubitIndexOutOfBounds {
            index: wire,
            limit: qubit_count,
        })?;
    if *slot {
        return Err(EngineError::ExecutionFailed(
            "contraction tree reuses a tensor".into(),
        ));
    }
    *slot = true;
    Ok(())
}

/// Entry `input | output << m` is `⟨output|U|input⟩` for an `m`-wire gate.
fn gate_tensor(gate: &Gate) -> Result<Vec<Complex64>, EngineError> {
    let wires = gate_wires(gate)?;
    let m = wires.len();
    let dim = 1usize << m;
    let mut data = vec![Complex64::ZERO; dim * dim];
    for input in 0..dim {
        let mut column = vec![Complex64::ZERO; dim];
        column[input] = Complex64::ONE;
        apply_in_place(&mut column, &wires, gate)?;
        for (output, amp) in column.into_iter().enumerate() {
            data[input | output << m] = amp;
        }
    }
    Ok(data)
}

/// Sum over shared edges. The result's legs are `right`'s free legs (low
/// bits), then `left`'s. Also returns the bytes allocated beyond both inputs:
/// the larger of the result and each reordered copy.
fn contract_pair(left: Tensor, right: Tensor, eval: bool) -> Result<(Tensor, u64), EngineError> {
    let shared: Vec<u32> = left
        .legs
        .iter()
        .copied()
        .filter(|leg| right.legs.contains(leg))
        .collect();
    let left_free: Vec<u32> = left
        .legs
        .iter()
        .copied()
        .filter(|leg| !shared.contains(leg))
        .collect();
    let right_free: Vec<u32> = right
        .legs
        .iter()
        .copied()
        .filter(|leg| !shared.contains(leg))
        .collect();
    let mut legs = right_free.clone();
    legs.extend_from_slice(&left_free);
    if legs.len() > MAX_OPEN_LEGS {
        return Err(EngineError::ExecutionFailed(
            "contraction tree tensor exceeds 26 legs".into(),
        ));
    }
    let left_order = [&shared[..], &left_free].concat();
    let right_order = [&shared[..], &right_free].concat();
    let mut working = 16u64 << legs.len();
    if left.legs != left_order {
        working = working.max(left.bytes());
    }
    if right.legs != right_order {
        working = working.max(right.bytes());
    }
    let data = match (eval, left.data, right.data) {
        (false, _, _) => None,
        (true, Some(left_data), Some(right_data)) => {
            let left_data = permute(left_data, &left.legs, &left_order);
            let right_data = permute(right_data, &right.legs, &right_order);
            let k = 1usize << shared.len();
            let m = 1usize << left_free.len();
            let n = 1usize << right_free.len();
            Some(matmul_transposed(&left_data, &right_data, m, k, n)?)
        }
        _ => {
            return Err(EngineError::ExecutionFailed(
                "contraction tree tensor is not ready".into(),
            ))
        }
    };
    Ok((Tensor { legs, data }, working))
}

/// `C[i][j] = Σ_k A[i][k] B[j][k]` with `A` `m × k` and `B` `n × k`, row-major.
fn matmul_transposed(
    a: &[Complex64],
    b: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<Complex64>, EngineError> {
    let shape_err = |_| EngineError::ExecutionFailed("contraction tree tensor is not ready".into());
    let a = ArrayView2::from_shape((m, k), a).map_err(shape_err)?;
    let b = ArrayView2::from_shape((n, k), b).map_err(shape_err)?;
    // With k = 1 both operands are also column-major, and `dot` may return a
    // column-major product. Its raw vector would then be the transpose.
    let product = a.dot(&b.t());
    Ok(if product.is_standard_layout() {
        product.into_raw_vec()
    } else {
        product.as_standard_layout().into_owned().into_raw_vec()
    })
}

/// Reorder legs `from` into `to`. Leg `p` of a tensor is bit `p` of its index.
fn permute(data: Vec<Complex64>, from: &[u32], to: &[u32]) -> Vec<Complex64> {
    if from == to {
        return data;
    }
    let source: Vec<usize> = to
        .iter()
        .map(|leg| {
            from.iter()
                .position(|l| l == leg)
                .expect("permuted legs are the same set")
        })
        .collect();
    let low = source.len().min(PERMUTE_LOW_BITS);
    let low_table = scatter_table(&source[..low]);
    let high_table = scatter_table(&source[low..]);
    let mut out = vec![Complex64::ZERO; data.len()];
    out.par_chunks_mut(1usize << low)
        .zip(high_table.par_iter())
        .for_each(|(chunk, &base)| {
            for (slot, &offset) in chunk.iter_mut().zip(&low_table) {
                *slot = data[base | offset];
            }
        });
    out
}

/// Entry `x` sets bit `bits[b]` for every set bit `b` of `x`.
fn scatter_table(bits: &[usize]) -> Vec<usize> {
    (0..1usize << bits.len())
        .map(|x| {
            bits.iter()
                .enumerate()
                .filter(|(b, _)| (x >> b) & 1 == 1)
                .fold(0, |acc, (_, &place)| acc | 1usize << place)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::IfParams;
    use crate::tn::dense::DenseTnState;
    use crate::tn::tree::SweepDispatch;
    use crate::tn::tree::{contract_tree, is_contraction_client_error, validate_sweep_dispatch};
    use crate::tn::MpsState;

    /// Leaves in canonical order: `ket0`, unitary gates, `bra0`.
    fn leaves(qubit_count: usize, gates: &[Gate]) -> Vec<(TensorRef, Vec<u32>)> {
        let network = Network::build(qubit_count, gates).expect("network");
        let mut out: Vec<(TensorRef, Vec<u32>)> = (0..qubit_count as u32)
            .map(|w| (TensorRef::Ket0(w), vec![w]))
            .collect();
        for index in 0..gates.len() {
            if let Some(legs) = network.gate_legs(index) {
                out.push((TensorRef::Gate(index as u32), legs.to_vec()));
            }
        }
        for w in 0..qubit_count {
            out.push((TensorRef::Bra0(w as u32), vec![network.bra_edge(w)]));
        }
        out
    }

    fn symmetric_difference(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut legs: Vec<u32> = b.iter().copied().filter(|l| !a.contains(l)).collect();
        legs.extend(a.iter().copied().filter(|l| !b.contains(l)));
        legs
    }

    fn finish(qubit_count: usize, gates: &[Gate], nodes: Vec<ContractionNode>) -> ContractionTree {
        let mut tree = ContractionTree {
            kind: Some(ContractionKind::Network),
            site_order: Vec::new(),
            nodes,
            peak_bytes: None,
        };
        // Over budget stays `None`; contracting it then fails on the missing peak.
        tree.peak_bytes = measured_peak(qubit_count, gates, &tree).ok();
        tree
    }

    /// Any order: pick a live tensor, then a neighbour that shares an edge
    /// with it if there is one.
    fn random_tree(qubit_count: usize, gates: &[Gate], seed: u64) -> ContractionTree {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = move |bound: usize| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound
        };
        let mut live = leaves(qubit_count, gates);
        let mut nodes = Vec::new();
        while live.len() > 1 {
            let i = next(live.len());
            let neighbours: Vec<usize> = (0..live.len())
                .filter(|&j| j != i && live[j].1.iter().any(|l| live[i].1.contains(l)))
                .collect();
            let j = if neighbours.is_empty() {
                (i + 1 + next(live.len() - 1)) % live.len()
            } else {
                neighbours[next(neighbours.len())]
            };
            let (hi, lo) = (i.max(j), i.min(j));
            let b = live.remove(hi);
            let a = live.remove(lo);
            let legs = symmetric_difference(&a.1, &b.1);
            nodes.push(ContractionNode::Contract {
                left: a.0,
                right: b.0,
            });
            live.push((TensorRef::Node((nodes.len() - 1) as u32), legs));
        }
        finish(qubit_count, gates, nodes)
    }

    /// Smallest growth `|result| − |left| − |right|` among pairs that share
    /// an edge, then the smaller ids. Leaves take ids in canonical order and
    /// node `i` takes `leaves + i`. Scalars join last, smallest ids first.
    pub(crate) fn greedy_tree(qubit_count: usize, gates: &[Gate]) -> ContractionTree {
        let mut slots: Vec<Option<(TensorRef, Vec<u32>)>> =
            leaves(qubit_count, gates).into_iter().map(Some).collect();
        // Edge → the two slots that hold it.
        let mut owners: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (id, slot) in slots.iter().enumerate() {
            for &leg in &slot.as_ref().expect("leaf").1 {
                owners.entry(leg).or_default().push(id);
            }
        }
        let mut nodes = Vec::new();
        let mut alive = slots.len();
        while alive > 1 {
            let mut best: Option<(i128, usize, usize)> = None;
            for pair in owners.values() {
                let (i, j) = (pair[0].min(pair[1]), pair[0].max(pair[1]));
                let a = &slots[i].as_ref().expect("live").1;
                let b = &slots[j].as_ref().expect("live").1;
                let legs = symmetric_difference(a, b).len();
                let cost = (1i128 << legs) - (1i128 << a.len()) - (1i128 << b.len());
                if best.is_none_or(|c| (cost, i, j) < c) {
                    best = Some((cost, i, j));
                }
            }
            let (i, j) = match best {
                Some((_, i, j)) => (i, j),
                None => {
                    let mut live = (0..slots.len()).filter(|&id| slots[id].is_some());
                    (live.next().expect("two"), live.next().expect("two"))
                }
            };
            let (left, a) = slots[i].take().expect("live");
            let (right, b) = slots[j].take().expect("live");
            let legs = symmetric_difference(&a, &b);
            let id = slots.len();
            for leg in a.iter().chain(&b) {
                if legs.contains(leg) {
                    for owner in owners.get_mut(leg).expect("open leg") {
                        if *owner == i || *owner == j {
                            *owner = id;
                        }
                    }
                } else {
                    owners.remove(leg);
                }
            }
            nodes.push(ContractionNode::Contract { left, right });
            slots.push(Some((TensorRef::Node((nodes.len() - 1) as u32), legs)));
            alive -= 1;
        }
        finish(qubit_count, gates, nodes)
    }

    fn dense_zero(n: usize, gates: &[Gate]) -> Complex64 {
        let mut dense = DenseTnState::try_new(n).expect("dense");
        for gate in gates {
            dense.apply_gate(gate);
        }
        dense.state[0]
    }

    fn assert_close(got: Complex64, expected: Complex64) {
        assert!((got - expected).norm() < 1e-9, "{got} vs {expected}");
    }

    fn random_circuit(n: usize, count: usize, seed: u64) -> Vec<Gate> {
        let mut state = seed;
        let mut next = move |bound: usize| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound
        };
        let mut gates = Vec::new();
        for _ in 0..count {
            let a = next(n);
            let b = (a + 1 + next(n - 1)) % n;
            let theta = next(1000) as f64 * 0.00731 + 0.1;
            let gate = match next(12) {
                0 => Gate::H(a),
                1 => Gate::X(a),
                2 => Gate::Y(a),
                3 => Gate::Z(a),
                4 => Gate::S(a),
                5 => Gate::T(a),
                6 => Gate::RX(a, theta),
                7 => Gate::RY(a, theta),
                8 => Gate::RZ(a, theta),
                9 => Gate::CNOT(a, b),
                10 => Gate::CZ(a, b),
                _ if n >= 3 => {
                    let c = (0..n).find(|&c| c != a && c != b).expect("third wire");
                    Gate::CCNOT(a, b, c)
                }
                _ => Gate::CNOT(a, b),
            };
            gates.push(gate);
        }
        gates
    }

    fn brickwork(n: usize, depth: usize) -> Vec<Gate> {
        let mut gates = Vec::new();
        for layer in 0..depth {
            for q in 0..n {
                gates.push(Gate::RY(q, 0.17 + 0.013 * (q + 3 * layer) as f64));
                gates.push(Gate::RZ(q, 0.31 + 0.007 * (2 * q + layer) as f64));
            }
            let mut q = layer % 2;
            while q + 1 < n {
                gates.push(Gate::CNOT(q, q + 1));
                q += 2;
            }
        }
        gates
    }

    #[test]
    fn any_order_matches_dense() {
        let ghz: Vec<Gate> = std::iter::once(Gate::H(0))
            .chain((0..4).map(|q| Gate::CNOT(q, q + 1)))
            .collect();
        let far = vec![
            Gate::H(0),
            Gate::CNOT(0, 4),
            Gate::RY(4, 0.4),
            Gate::CZ(4, 1),
        ];
        let mut circuits = vec![(5, ghz), (5, far)];
        for seed in 0..6u64 {
            circuits.push((6, random_circuit(6, 40, seed)));
        }
        for (n, gates) in &circuits {
            let expected = dense_zero(*n, gates);
            for seed in 0..8u64 {
                let tree = random_tree(*n, gates, seed);
                let got = contract_network(*n, gates, &tree, None).expect("network");
                assert_close(got, expected);
            }
            let got = contract_network(*n, gates, &greedy_tree(*n, gates), None).expect("greedy");
            assert_close(got, expected);
        }
    }

    /// The orchestrator's plain greedy pass emits this text for the same
    /// circuit. Its search stops at once here, so `PlanNetworkContraction`
    /// returns it too.
    const FIXTURE_JSON: &str = r#"{"kind":"network","nodes":[{"op":"contract","left":{"ket0":1},"right":{"gate":3}},{"op":"contract","left":{"bra0":0},"right":{"node":0}},{"op":"contract","left":{"gate":1},"right":{"node":1}},{"op":"contract","left":{"ket0":2},"right":{"node":2}},{"op":"contract","left":{"bra0":2},"right":{"node":3}},{"op":"contract","left":{"ket0":0},"right":{"gate":0}},{"op":"contract","left":{"gate":4},"right":{"bra0":1}},{"op":"contract","left":{"node":4},"right":{"node":5}},{"op":"contract","left":{"ket0":3},"right":{"bra0":3}},{"op":"contract","left":{"node":6},"right":{"node":7}},{"op":"contract","left":{"node":8},"right":{"node":9}}],"peak_bytes":2080}"#;

    #[test]
    fn orchestrator_fixture_is_the_greedy_tree_and_matches_dense() {
        let gates = vec![
            Gate::H(0),
            Gate::CNOT(0, 2),
            Gate::MEASURE(crate::engine::MeasureParams { qubit: 1, cbit: 0 }),
            Gate::CCNOT(0, 2, 1),
            Gate::RZ(1, 0.4),
        ];
        let tree: ContractionTree = serde_json::from_str(FIXTURE_JSON).expect("parse");
        assert_eq!(tree, greedy_tree(4, &gates));
        assert_eq!(serde_json::to_string(&tree).expect("json"), FIXTURE_JSON);
        let unitary: Vec<Gate> = gates
            .iter()
            .filter(|g| !matches!(g, Gate::MEASURE(_)))
            .cloned()
            .collect();
        let got = contract_network(4, &gates, &tree, None).expect("network");
        let expected = dense_zero(4, &unitary);
        assert!(expected.norm() > 0.1, "amplitude vanished: {expected}");
        assert_close(got, expected);
    }

    /// SHA3-256 of the orchestrator's plain greedy tree JSON for a 6-qubit
    /// depth-3 brickwork. Many pairs tie on cost there, so this pins the tie
    /// order.
    const BRICKWORK_TREE_SHA3: &str =
        "dccaf420f62676fdc7f589aa8791dd29ecd145c68a9af661dabfd443429aa2c1";

    #[test]
    fn orchestrator_brickwork_tree_hash_matches_the_greedy() {
        use sha3::{Digest, Sha3_256};
        let tree = greedy_tree(6, &brickwork(6, 3));
        let json = serde_json::to_string(&tree).expect("json");
        assert_eq!(
            hex::encode(Sha3_256::digest(json.as_bytes())),
            BRICKWORK_TREE_SHA3,
            "{json}"
        );
    }

    /// RY and RZ on every qubit of a side×side grid, then CNOTs on
    /// alternating pairs: two layers along rows, two along columns. The
    /// orchestrator's `gridCircuit`.
    fn grid(side: usize, depth: usize) -> Vec<Gate> {
        let at = |row: usize, col: usize| row * side + col;
        let mut gates = Vec::new();
        for layer in 0..depth {
            for q in 0..side * side {
                gates.push(Gate::RY(q, 0.17));
                gates.push(Gate::RZ(q, 0.31));
            }
            if layer % 4 < 2 {
                for row in 0..side {
                    for col in (layer % 2..side.saturating_sub(1)).step_by(2) {
                        gates.push(Gate::CNOT(at(row, col), at(row, col + 1)));
                    }
                }
            } else {
                for row in (layer % 2..side.saturating_sub(1)).step_by(2) {
                    for col in 0..side {
                        gates.push(Gate::CNOT(at(row, col), at(row + 1, col)));
                    }
                }
            }
        }
        gates
    }

    /// The orchestrator's searched tree for a 4×4 grid, depth 12. The search
    /// picked it over the plain greedy (2.4 MB peak against 9.0 MB).
    const GRID_TREE_JSON: &str = include_str!("network_grid_4x4_d12.json");
    const GRID_TREE_SHA3: &str = "3ac1c2130343db568ea5ee2c22b6f2f7291f0d39cd3c73800aad3be734ad01ed";

    #[test]
    fn orchestrator_searched_tree_matches_dense() {
        use sha3::{Digest, Sha3_256};
        assert_eq!(
            hex::encode(Sha3_256::digest(GRID_TREE_JSON.as_bytes())),
            GRID_TREE_SHA3
        );
        let tree: ContractionTree = serde_json::from_str(GRID_TREE_JSON).expect("parse");
        assert_ne!(tree, greedy_tree(16, &grid(4, 12)), "not a searched tree");
        let gates = grid(4, 12);
        validate(16, &gates, &tree, None).expect("declared peak");
        let got = contract_network(16, &gates, &tree, None).expect("network");
        let expected = dense_zero(16, &gates);
        assert!(expected.norm() > 0.1, "amplitude vanished: {expected}");
        assert_close(got, expected);
    }

    #[test]
    fn untouched_wire_and_skipped_ops_close_with_one() {
        let gates = vec![
            Gate::H(0),
            Gate::MEASURE(crate::engine::MeasureParams { qubit: 0, cbit: 0 }),
            Gate::IF(IfParams {
                cbit: 0,
                value: 1,
                gate: Box::new(Gate::X(1)),
            }),
            Gate::CNOT(0, 2),
        ];
        let tree = greedy_tree(4, &gates);
        let unitary: Vec<Gate> = vec![Gate::H(0), Gate::CNOT(0, 2)];
        let got = contract_network(4, &gates, &tree, None).expect("network");
        assert_close(got, dense_zero(4, &unitary));
        let one = ContractionTree {
            kind: Some(ContractionKind::Network),
            site_order: Vec::new(),
            nodes: vec![ContractionNode::Contract {
                left: TensorRef::Ket0(0),
                right: TensorRef::Bra0(0),
            }],
            peak_bytes: Some(32 + 32 + 16),
        };
        assert_close(
            contract_network(1, &[], &one, None).expect("one"),
            Complex64::ONE,
        );
    }

    #[test]
    fn far_20q_matches_dense_and_the_state_tree() {
        let n = 20usize;
        let mut gates = brickwork(n, 3);
        gates.extend([
            Gate::CNOT(0, 19),
            Gate::CCNOT(3, 11, 17),
            Gate::CZ(14, 5),
            Gate::RX(19, 0.23),
            Gate::CNOT(17, 2),
        ]);
        gates.extend(
            random_circuit(n, 30, 7)
                .into_iter()
                .filter(|g| !matches!(g, Gate::X(_) | Gate::Y(_) | Gate::CCNOT(..))),
        );
        let tree = greedy_tree(n, &gates);
        let got = contract_network(n, &gates, &tree, None).expect("network");
        let expected = dense_zero(n, &gates);
        assert!(expected.norm() > 1e-4, "amplitude vanished: {expected}");
        assert_close(got, expected);
        let state_tree = crate::tn::plan::plan_greedy(n, &gates).expect("state tree");
        let state = crate::tn::exact::contract_exact(n, &gates, &state_tree, None).expect("exact");
        assert_close(got, state);
        eprintln!(
            "far 20q: network peak {} B, state peak {} B",
            tree.peak_bytes.unwrap(),
            state_tree.peak_bytes.unwrap()
        );
    }

    #[test]
    fn dispatch_runs_the_network_and_the_sweep_trace() {
        let n = 6usize;
        let gates = brickwork(n, 4);
        let tree = greedy_tree(n, &gates);
        let json = serde_json::to_string(&tree).expect("json");
        assert!(json.starts_with(r#"{"kind":"network","nodes":["#), "{json}");
        assert!(json.contains(r#"{"bra0":"#));
        let back: ContractionTree = serde_json::from_str(&json).expect("parse");
        assert_eq!(back, tree);
        validate_sweep_dispatch(&SweepDispatch {
            scalar: true,
            qubit_count: n,
            original_qubit_count: n,
            assignments_empty: true,
            time_split: false,
            mps_site_order: Some(&[0, 1, 2, 3, 4, 5]),
            gates: &gates,
            tree: &tree,
        })
        .expect("valid");
        let mut mps = MpsState::try_new_with_bond(n, 8).expect("mps");
        let (amp, trace) = contract_tree(n, &gates, &tree, None, &mut mps).expect("tree");
        assert_close(Complex64::new(amp.real, amp.imag), dense_zero(n, &gates));
        let mut slice = MpsState::try_new_with_bond(n, 8).expect("mps");
        let (_, slice_trace) =
            crate::tn::contract_slice(n, &gates, &[], n, &mut slice).expect("slice");
        assert_eq!(trace.len(), slice_trace.len());
        for (got, expected) in trace.iter().zip(&slice_trace) {
            assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
        }
    }

    #[test]
    fn bad_trees_are_rejected() {
        let n = 3usize;
        let gates = vec![Gate::H(0), Gate::CNOT(0, 1), Gate::CCNOT(0, 1, 2)];
        let good = greedy_tree(n, &gates);
        let check = |tree: &ContractionTree, needle: &str| {
            let err = validate(n, &gates, tree, None).unwrap_err();
            assert!(err.to_string().contains(needle), "{err} lacks {needle}");
            assert!(is_contraction_client_error(&err), "{err}");
        };

        let mut tree = good.clone();
        tree.peak_bytes = Some(tree.peak_bytes.unwrap() + 16);
        check(&tree, "does not match");

        tree = good.clone();
        tree.peak_bytes = None;
        check(&tree, "requires a peak");

        tree = good.clone();
        tree.nodes.pop();
        check(&tree, "one scalar");

        tree = good.clone();
        tree.nodes.truncate(1);
        check(&tree, "leaves a leaf out");

        tree = good.clone();
        let ContractionNode::Contract { left, .. } = tree.nodes[0] else {
            unreachable!()
        };
        tree.nodes.push(ContractionNode::Contract {
            left,
            right: TensorRef::Node((tree.nodes.len() - 1) as u32),
        });
        check(&tree, "reuses a tensor");

        tree = good.clone();
        tree.nodes[0] = ContractionNode::Contract {
            left: TensorRef::Node(0),
            right: TensorRef::Ket0(0),
        };
        check(&tree, "not ready");

        tree = good.clone();
        tree.site_order = vec![2, 1, 0];
        check(&tree, "contraction tree keeps parent wire numbers");

        tree = good.clone();
        tree.nodes[0] = ContractionNode::Contract {
            left: TensorRef::Left(0),
            right: TensorRef::Ket0(0),
        };
        check(&tree, "no split");

        // Without `kind`, the same nodes run as an exact tree and fail there.
        tree = good.clone();
        tree.kind = None;
        let mut mps = MpsState::try_new_with_bond(n, 4).expect("mps");
        let err = contract_tree(n, &gates, &tree, None, &mut mps).unwrap_err();
        assert!(is_contraction_client_error(&err), "{err}");
    }

    #[test]
    fn outer_products_of_tensors_with_legs_keep_their_order() {
        // ket0(1) shares nothing with the wire-0 tensor; both have a leg.
        let gates = vec![Gate::RY(0, 0.3), Gate::RY(1, 0.5)];
        let node = |left, right| ContractionNode::Contract { left, right };
        let tree = finish(
            2,
            &gates,
            vec![
                node(TensorRef::Ket0(0), TensorRef::Gate(0)),
                node(TensorRef::Node(0), TensorRef::Ket0(1)),
                node(TensorRef::Node(1), TensorRef::Gate(1)),
                node(TensorRef::Node(2), TensorRef::Bra0(0)),
                node(TensorRef::Node(3), TensorRef::Bra0(1)),
            ],
        );
        let got = contract_network(2, &gates, &tree, None).expect("network");
        assert_close(got, dense_zero(2, &gates));
    }

    #[test]
    fn reordered_copy_counts_toward_the_peak() {
        // gate(0) legs are [in0=0, in1=1, out0=2, out1=3]. Sharing edge 1 with
        // ket0(1) puts it first, so the 4-leg gate is copied (256 B) while the
        // 3-leg result (128 B) is smaller. 256 + 32 + 256 = 544.
        let gates = vec![Gate::CNOT(0, 1)];
        let node = |left, right| ContractionNode::Contract { left, right };
        let tree = finish(
            2,
            &gates,
            vec![
                node(TensorRef::Gate(0), TensorRef::Ket0(1)),
                node(TensorRef::Node(0), TensorRef::Ket0(0)),
                node(TensorRef::Node(1), TensorRef::Bra0(0)),
                node(TensorRef::Node(2), TensorRef::Bra0(1)),
            ],
        );
        assert_eq!(tree.peak_bytes, Some(544));
        let got = contract_network(2, &gates, &tree, None).expect("network");
        assert_close(got, Complex64::ONE);
    }

    #[test]
    fn over_budget_peaks_are_rejected() {
        let n = 3usize;
        let gates = vec![Gate::H(0)];
        let mut tree = greedy_tree(n, &gates);
        tree.peak_bytes = Some(MAX_PEAK_BYTES + 1);
        let err = validate(n, &gates, &tree, None).unwrap_err();
        assert!(err.to_string().contains("peak exceeds 1 GiB"), "{err}");

        // Two 25-leg chains: no tensor passes 26 legs, but while the second
        // finishes, the first (512 MiB) is live next to it.
        let n = 50usize;
        let mut nodes = Vec::new();
        for start in [0u32, 25] {
            let mut acc = TensorRef::Ket0(start);
            for w in start + 1..start + 25 {
                nodes.push(ContractionNode::Contract {
                    left: acc,
                    right: TensorRef::Ket0(w),
                });
                acc = TensorRef::Node((nodes.len() - 1) as u32);
            }
        }
        let tree = ContractionTree {
            kind: Some(ContractionKind::Network),
            site_order: Vec::new(),
            nodes,
            peak_bytes: Some(0),
        };
        let err = validate(n, &[], &tree, None).unwrap_err();
        assert!(err.to_string().contains("peak exceeds 1 GiB"), "{err}");
    }

    #[test]
    fn wide_join_is_rejected() {
        // 27 wires joined pairwise before any bra0: the last join has 27 legs.
        let n = 27usize;
        let mut nodes = Vec::new();
        let mut acc = TensorRef::Ket0(0);
        for w in 1..n as u32 {
            nodes.push(ContractionNode::Contract {
                left: acc,
                right: TensorRef::Ket0(w),
            });
            acc = TensorRef::Node((nodes.len() - 1) as u32);
        }
        let tree = ContractionTree {
            kind: Some(ContractionKind::Network),
            site_order: Vec::new(),
            nodes,
            peak_bytes: Some(0),
        };
        let err = validate(n, &[], &tree, None).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
    }

    #[test]
    fn brickwork_50q_depth_8_fits() {
        let n = 50usize;
        let gates = brickwork(n, 8);
        let tree = greedy_tree(n, &gates);
        let peak = tree.peak_bytes.expect("peak");
        assert!(peak <= MAX_PEAK_BYTES, "peak {peak}");
        let start = std::time::Instant::now();
        let got = contract_network(n, &gates, &tree, None).expect("network");
        eprintln!(
            "50q depth 8: peak {peak} B, nodes {}, contract {:?}, amp {got}",
            tree.nodes.len(),
            start.elapsed()
        );
        // Same circuit cut in two: the MPS at an exact bond agrees.
        let mut mps = MpsState::try_new_with_bond(n, 256).expect("mps");
        for gate in &gates {
            mps.apply_gate(gate).expect("apply");
        }
        assert_close(got, mps.amplitude_at_compact_zero());
    }

    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn timing_wide_and_trace() {
        for (n, depth) in [
            (30usize, 12usize),
            (30, 16),
            (30, 24),
            (40, 20),
            (100, 8),
            (100, 12),
        ] {
            let gates = brickwork(n, depth);
            let t = std::time::Instant::now();
            let tree = greedy_tree(n, &gates);
            let plan = t.elapsed();
            let widest = widest(n, &gates, &tree);
            let Some(peak) = tree.peak_bytes else {
                eprintln!("{n}q depth {depth}: widest {widest} legs, over budget, plan {plan:?}");
                continue;
            };
            let t = std::time::Instant::now();
            let amp = contract_network(n, &gates, &tree, None).expect("network");
            eprintln!(
                "{n}q depth {depth}: widest {widest} legs, peak {} B, plan {plan:?}, contract {:?}, amp {amp}",
                peak,
                t.elapsed()
            );
        }
        for chi in [16usize, 64] {
            let n = 50usize;
            let gates = brickwork(n, 8);
            let tree = greedy_tree(n, &gates);
            let mut mps = MpsState::try_new_with_bond(n, chi).expect("mps");
            let t = std::time::Instant::now();
            let (_, trace) = contract_tree(n, &gates, &tree, None, &mut mps).expect("tree");
            eprintln!(
                "50q depth 8 with sweep trace χ={chi}: {:?}, {} trace values",
                t.elapsed(),
                trace.len()
            );
        }
    }

    fn widest(n: usize, gates: &[Gate], tree: &ContractionTree) -> usize {
        let mut legs: BTreeMap<String, Vec<u32>> = leaves(n, gates)
            .into_iter()
            .map(|(r, l)| (format!("{r:?}"), l))
            .collect();
        let mut widest = 0;
        for (i, node) in tree.nodes.iter().enumerate() {
            let ContractionNode::Contract { left, right } = node else {
                unreachable!()
            };
            let a = legs.remove(&format!("{left:?}")).expect("left");
            let b = legs.remove(&format!("{right:?}")).expect("right");
            let out = symmetric_difference(&a, &b);
            widest = widest.max(out.len());
            legs.insert(format!("{:?}", TensorRef::Node(i as u32)), out);
        }
        widest
    }
}
