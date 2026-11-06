//! Network contraction of `⟨0…0|U|0…0⟩`.
//!
//! The circuit is one tensor network. Wire `w` starts at a `ket0(w)` leaf and
//! ends at a `bra0(w)` leaf. Unitary gate `i` is a `gate(i)` leaf with one
//! input and one output leg on every wire it touches. An edge is one segment
//! of a wire between two consecutive leaves on it, so every edge joins exactly
//! two leaves. The tree names leaves only. Edges follow from the circuit, and
//! MEASURE, RESET, and IF are skipped as in the exact tree.
//!
//! A CNOT or CZ may instead enter as two halves, `control(i)` and
//! `target(i)`, joined by a bond edge. Each half has the input, output, and
//! bond legs of its wire. The control half is 1 where all three are equal;
//! the target half is the identity on bond 0 and X (CNOT) or Z (CZ) on bond 1.
//! Their contraction is the gate. A tree uses a gate whole or both halves.
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
use std::sync::OnceLock;
use std::time::Instant;

use crate::engine::{EngineError, Gate};
use ndarray::ArrayView2;
use num_complex::Complex64;
use rayon::prelude::*;

use super::exact::{apply_in_place, check_shape, gate_wires, MAX_OPEN_LEGS, MAX_PEAK_BYTES};
use super::tree::{unitary_wires, ContractionKind, ContractionNode, ContractionTree, TensorRef};

/// Output bits a lookup table covers when legs are reordered.
const PERMUTE_LOW_BITS: usize = 12;

/// Largest `min(m, n) · k` that [`matmul_small_side`] takes instead of GEMM.
const SMALL_SIDE: usize = 64;

/// Edges of every leaf.
pub(crate) struct Network {
    /// Edge of `bra0(w)`. The edge of `ket0(w)` is `w`.
    bra: Vec<u32>,
    /// Inputs, then outputs, in the gate's wire order. `None` is not unitary.
    gate: Vec<Option<Vec<u32>>>,
    /// Edges on wires. The bond of gate `i` is edge `wire_edges + i`.
    wire_edges: u32,
}

impl Network {
    /// Edge `w` leaves `ket0(w)`. Each gate numbers its outputs next, in wire
    /// order. Bonds come after every wire edge, by gate index.
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
        u32::try_from(gates.len())
            .ok()
            .and_then(|count| next.checked_add(count))
            .ok_or_else(|| {
                EngineError::ExecutionFailed("contraction tree has too many edges".into())
            })?;
        Ok(Self {
            bra: current,
            gate: legs_of,
            wire_edges: next,
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

    /// Input, output, and bond edges of one half of gate `index`. Half 0 is
    /// the first wire.
    pub(crate) fn half_legs(&self, index: usize, half: usize) -> Option<[u32; 3]> {
        let legs = self.gate_legs(index)?;
        (legs.len() == 4).then(|| [legs[half], legs[2 + half], self.wire_edges + index as u32])
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
        gate: vec![0; gates.len()],
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
        || gates.iter().enumerate().any(|(i, _)| {
            network.gate_legs(i).is_some() && !matches!(leaves.gate[i], WHOLE | HALVES)
        })
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
        TensorRef::Control(gate) => leaves.half(gate as usize, 0),
        TensorRef::Target(gate) => leaves.half(gate as usize, 1),
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
    /// `WHOLE`, or the `CONTROL` and `TARGET` bits taken so far.
    gate: Vec<u8>,
}

const WHOLE: u8 = 1;
const CONTROL: u8 = 2;
const TARGET: u8 = 4;
const HALVES: u8 = CONTROL | TARGET;

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
        self.take_gate(index, WHOLE)?;
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

    fn half(&mut self, index: usize, half: usize) -> Result<Tensor, EngineError> {
        let gate = self.gates.get(index).ok_or_else(|| {
            EngineError::ExecutionFailed("contraction tree gate is missing".into())
        })?;
        let target_is_z = match gate {
            Gate::CNOT(..) => false,
            Gate::CZ(..) => true,
            _ => {
                return Err(EngineError::ExecutionFailed(format!(
                    "gate {index} has no halves"
                )))
            }
        };
        let legs = self.network.half_legs(index, half).ok_or_else(|| {
            EngineError::ExecutionFailed(format!("gate {index} is not a contraction leaf"))
        })?;
        self.take_gate(index, if half == 0 { CONTROL } else { TARGET })?;
        let data = self.eval.then(|| half_tensor(half, target_is_z));
        Ok(Tensor {
            legs: legs.to_vec(),
            data,
        })
    }

    fn take_gate(&mut self, index: usize, part: u8) -> Result<(), EngineError> {
        let used = self.gate[index];
        if used & part != 0 {
            return Err(EngineError::ExecutionFailed(
                "contraction tree reuses a tensor".into(),
            ));
        }
        if used != 0 && (used | part) & WHOLE != 0 {
            return Err(EngineError::ExecutionFailed(
                "contraction tree mixes a gate and its halves".into(),
            ));
        }
        self.gate[index] = used | part;
        Ok(())
    }
}

/// Entry `input | output << 1 | bond << 2` of one half of a CNOT or CZ.
fn half_tensor(half: usize, target_is_z: bool) -> Vec<Complex64> {
    (0..8usize)
        .map(|index| {
            let (input, output, bond) = (index & 1, index >> 1 & 1, index >> 2);
            let value = match (half, bond) {
                (0, _) => (input == output && output == bond) as u8 as f64,
                (_, 0) => (input == output) as u8 as f64,
                _ if target_is_z => {
                    if input != output {
                        0.0
                    } else if input == 1 {
                        -1.0
                    } else {
                        1.0
                    }
                }
                _ => (input != output) as u8 as f64,
            };
            Complex64::new(value, 0.0)
        })
        .collect()
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

/// Seconds this executor takes per byte and per flop, in the planner's counts
/// (docs `work-budget.md` §1). A node turns a task's announced work into an
/// expected time with them.
#[derive(Clone, Copy, Debug)]
pub struct TnRates {
    pub seconds_per_byte: f64,
    pub seconds_per_flop: f64,
}

static TN_RATES: OnceLock<Option<TnRates>> = OnceLock::new();

/// Measured once per process; the first call takes about a second. `None`
/// when a benchmark join fails, so a node does not judge work by a guess.
pub fn tn_rates() -> Option<TnRates> {
    *TN_RATES.get_or_init(measure_rates)
}

/// The planner's `traffic` and `flops` for one join: both inputs read and the
/// result written, a reordered input copied once more; `2^(|left| + |right
/// free|)` flops.
fn join_cost(left: &[u32], right: &[u32]) -> (f64, f64) {
    let shared: Vec<u32> = left.iter().copied().filter(|l| right.contains(l)).collect();
    let free = |legs: &[u32]| -> Vec<u32> {
        legs.iter()
            .copied()
            .filter(|l| !shared.contains(l))
            .collect()
    };
    let (left_free, right_free) = (free(left), free(right));
    let bytes = |legs: usize| (16u64 << legs) as f64;
    let mut traffic =
        bytes(left.len()) + bytes(right.len()) + bytes(left_free.len() + right_free.len());
    if left != [&shared[..], &left_free].concat() {
        traffic += 2.0 * bytes(left.len());
    }
    if right != [&shared[..], &right_free].concat() {
        traffic += 2.0 * bytes(right.len());
    }
    (traffic, (1u64 << (left.len() + right_free.len())) as f64)
}

fn measure_rates() -> Option<TnRates> {
    let tensor = |legs: &[u32]| Tensor {
        legs: legs.to_vec(),
        data: Some(
            (0..1usize << legs.len())
                .map(|i| Complex64::new((i % 7) as f64 * 0.1, 0.5))
                .collect(),
        ),
    };
    let fastest = |left: &[u32], right: &[u32]| -> Option<f64> {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let (l, r) = (tensor(left), tensor(right));
            let start = Instant::now();
            if let Err(e) = contract_pair(l, r, true) {
                eprintln!("contraction rate benchmark failed: {e}");
                return None;
            }
            best = best.min(start.elapsed().as_secs_f64());
        }
        Some(best)
    };
    // A gate half absorbed into a 2^23 state whose shared leg sits mid-way, so
    // the state is reordered: almost all traffic (512 MiB), few flops.
    let state: Vec<u32> = (0..23).collect();
    let half = [100, 11];
    let (traffic, _) = join_cost(&half, &state);
    let seconds_per_byte = fastest(&half, &state)? / traffic;
    // Two 2^20 operands sharing 10 legs, both already in order: 2^30 flops.
    // Its 48 MiB of traffic stay in the time, which errs slow.
    let left: Vec<u32> = (0..10).chain(100..110).collect();
    let right: Vec<u32> = (0..10).chain(200..210).collect();
    let (_, flops) = join_cost(&left, &right);
    let seconds_per_flop = fastest(&left, &right)? / flops;
    let valid = |x: f64| x.is_finite() && x > 0.0;
    (valid(seconds_per_byte) && valid(seconds_per_flop)).then_some(TnRates {
        seconds_per_byte,
        seconds_per_flop,
    })
}

/// `C[i][j] = Σ_k A[i][k] B[j][k]` with `A` `m × k` and `B` `n × k`, row-major.
fn matmul_transposed(
    a: &[Complex64],
    b: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<Complex64>, EngineError> {
    if m.min(n) * k <= SMALL_SIDE {
        return Ok(matmul_small_side(a, b, m, k, n));
    }
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

/// `matmul_transposed` when one side is a few rows, such as a gate half
/// absorbed into a large state. The GEMM kernel runs its edge path there.
fn matmul_small_side(
    a: &[Complex64],
    b: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    let dot =
        |x: &[Complex64], y: &[Complex64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<Complex64>();
    let mut out = vec![Complex64::ZERO; m * n];
    if n <= m {
        out.par_chunks_mut(n)
            .zip(a.par_chunks(k))
            .for_each(|(row, a_row)| {
                for (slot, b_row) in row.iter_mut().zip(b.chunks(k)) {
                    *slot = dot(a_row, b_row);
                }
            });
    } else {
        for (row, a_row) in out.chunks_mut(n).zip(a.chunks(k)) {
            row.par_iter_mut()
                .zip(b.par_chunks(k))
                .for_each(|(slot, b_row)| *slot = dot(a_row, b_row));
        }
    }
    out
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

    /// `leaves` with every CNOT and CZ as its control half, then its target.
    fn split_leaves(qubit_count: usize, gates: &[Gate]) -> Vec<(TensorRef, Vec<u32>)> {
        let network = Network::build(qubit_count, gates).expect("network");
        let mut out = Vec::new();
        for (r, legs) in leaves(qubit_count, gates) {
            match r {
                TensorRef::Gate(i)
                    if matches!(gates[i as usize], Gate::CNOT(..) | Gate::CZ(..)) =>
                {
                    let i_us = i as usize;
                    out.push((
                        TensorRef::Control(i),
                        network.half_legs(i_us, 0).expect("half").to_vec(),
                    ));
                    out.push((
                        TensorRef::Target(i),
                        network.half_legs(i_us, 1).expect("half").to_vec(),
                    ));
                }
                _ => out.push((r, legs)),
            }
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
        random_tree_of(qubit_count, gates, leaves(qubit_count, gates), seed)
    }

    fn random_tree_of(
        qubit_count: usize,
        gates: &[Gate],
        mut live: Vec<(TensorRef, Vec<u32>)>,
        seed: u64,
    ) -> ContractionTree {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = move |bound: usize| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound
        };
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
        greedy_tree_of(qubit_count, gates, leaves(qubit_count, gates))
    }

    fn greedy_tree_of(
        qubit_count: usize,
        gates: &[Gate],
        leaves: Vec<(TensorRef, Vec<u32>)>,
    ) -> ContractionTree {
        let mut slots: Vec<Option<(TensorRef, Vec<u32>)>> = leaves.into_iter().map(Some).collect();
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

    /// The orchestrator's split-gate search tree for the same grid. It
    /// takes CNOTs as halves and peaks at 843776 B.
    const SPLIT_GRID_TREE_JSON: &str = include_str!("network_grid_4x4_d12_split.json");
    const SPLIT_GRID_TREE_SHA3: &str =
        "06ca39ca74c302b35a58990558022c0ee78fa637d9043d8b19f8814b2f4aa561";

    #[test]
    fn orchestrator_split_tree_matches_dense() {
        use sha3::{Digest, Sha3_256};
        assert_eq!(
            hex::encode(Sha3_256::digest(SPLIT_GRID_TREE_JSON.as_bytes())),
            SPLIT_GRID_TREE_SHA3
        );
        let tree: ContractionTree = serde_json::from_str(SPLIT_GRID_TREE_JSON).expect("parse");
        assert_eq!(
            serde_json::to_string(&tree).expect("json"),
            SPLIT_GRID_TREE_JSON
        );
        assert!(SPLIT_GRID_TREE_JSON.contains(r#"{"control":"#));
        assert_eq!(tree.peak_bytes, Some(843776));
        let gates = grid(4, 12);
        validate(16, &gates, &tree, None).expect("declared peak");
        let got = contract_network(16, &gates, &tree, None).expect("network");
        let expected = dense_zero(16, &gates);
        assert!(expected.norm() > 0.1, "amplitude vanished: {expected}");
        assert_close(got, expected);
    }

    /// The orchestrator's planned tree for the same grid: a recursive
    /// bisection over the split gates. It peaks at 262144 B.
    const BISECT_GRID_TREE_JSON: &str = include_str!("network_grid_4x4_d12_bisect.json");
    const BISECT_GRID_TREE_SHA3: &str =
        "6038f75c8c231db5e672af5b3eb22a95968a7aabd50b170029466267e364988a";

    #[test]
    fn orchestrator_bisection_tree_matches_dense() {
        use sha3::{Digest, Sha3_256};
        assert_eq!(
            hex::encode(Sha3_256::digest(BISECT_GRID_TREE_JSON.as_bytes())),
            BISECT_GRID_TREE_SHA3
        );
        let tree: ContractionTree = serde_json::from_str(BISECT_GRID_TREE_JSON).expect("parse");
        assert_eq!(
            serde_json::to_string(&tree).expect("json"),
            BISECT_GRID_TREE_JSON
        );
        assert!(BISECT_GRID_TREE_JSON.contains(r#"{"control":"#));
        assert_eq!(tree.peak_bytes, Some(262144));
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
    fn tree_trace_sweep_is_capped_at_the_trace_bond() {
        use crate::tn::tree::TREE_TRACE_BOND;
        // Above 16 qubits the trace reads site amplitudes, not a dense state.
        let n = 18usize;
        let gates = brickwork(n, 12);
        let tree = greedy_tree(n, &gates);
        let wide = 4 * TREE_TRACE_BOND;
        let mut mps = MpsState::try_new_with_bond(n, wide).expect("mps");
        let (amp, trace) = contract_tree(n, &gates, &tree, None, &mut mps).expect("tree");
        assert_close(Complex64::new(amp.real, amp.imag), dense_zero(n, &gates));
        assert_eq!(mps.max_bond_dim, TREE_TRACE_BOND);
        let trace_at = |chi: usize| {
            let mut slice = MpsState::try_new_with_bond(n, chi).expect("mps");
            crate::tn::contract_slice(n, &gates, &[], n, &mut slice)
                .expect("slice")
                .1
        };
        // The circuit needs more than the cap, so the cap changes the trace.
        assert_ne!(trace_at(wide), trace_at(TREE_TRACE_BOND));
        assert_eq!(trace, trace_at(TREE_TRACE_BOND));
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
    fn split_gates_match_dense() {
        let mut circuits = vec![(5usize, brickwork(5, 4)), (4, grid(2, 6))];
        for seed in 0..6u64 {
            circuits.push((6, random_circuit(6, 40, seed)));
        }
        circuits.push((
            3,
            vec![
                Gate::H(0),
                Gate::H(2),
                Gate::CZ(2, 0),
                Gate::CNOT(0, 2),
                Gate::CZ(0, 1),
            ],
        ));
        for (n, gates) in &circuits {
            let expected = dense_zero(*n, gates);
            for seed in 0..8u64 {
                let tree = random_tree_of(*n, gates, split_leaves(*n, gates), seed);
                let got = contract_network(*n, gates, &tree, None).expect("network");
                assert_close(got, expected);
            }
            let tree = greedy_tree_of(*n, gates, split_leaves(*n, gates));
            let got = contract_network(*n, gates, &tree, None).expect("greedy");
            assert_close(got, expected);
        }
    }

    #[test]
    fn split_and_whole_gates_mix_across_gates() {
        // CNOT 1 whole, the others split.
        let n = 4usize;
        let gates = brickwork(n, 3);
        let whole = gates
            .iter()
            .position(|g| matches!(g, Gate::CNOT(..)))
            .expect("cnot") as u32;
        let network = Network::build(n, &gates).expect("network");
        let live: Vec<(TensorRef, Vec<u32>)> = split_leaves(n, &gates)
            .into_iter()
            .filter(|(r, _)| !matches!(r, TensorRef::Target(i) if *i == whole))
            .map(|(r, legs)| match r {
                TensorRef::Control(i) if i == whole => (
                    TensorRef::Gate(i),
                    network.gate_legs(i as usize).expect("legs").to_vec(),
                ),
                _ => (r, legs),
            })
            .collect();
        let tree = random_tree_of(n, &gates, live, 3);
        let got = contract_network(n, &gates, &tree, None).expect("network");
        assert_close(got, dense_zero(n, &gates));
    }

    #[test]
    fn split_tree_json_names_the_halves() {
        let gates = vec![Gate::H(0), Gate::CNOT(0, 1)];
        let node = |left, right| ContractionNode::Contract { left, right };
        let tree = finish(
            2,
            &gates,
            vec![
                node(TensorRef::Control(1), TensorRef::Target(1)),
                node(TensorRef::Ket0(0), TensorRef::Gate(0)),
                node(TensorRef::Node(1), TensorRef::Node(0)),
                node(TensorRef::Ket0(1), TensorRef::Node(2)),
                node(TensorRef::Bra0(0), TensorRef::Node(3)),
                node(TensorRef::Bra0(1), TensorRef::Node(4)),
            ],
        );
        // Five wire edges: two kets, H's output, CNOT's two outputs. The bond
        // of gate 1 is edge 5 + 1.
        let network = Network::build(2, &gates).expect("network");
        assert_eq!(network.half_legs(1, 0), Some([2, 3, 6]));
        assert_eq!(network.half_legs(1, 1), Some([1, 4, 6]));
        let json = serde_json::to_string(&tree).expect("json");
        assert!(
            json.starts_with(r#"{"kind":"network","nodes":[{"op":"contract","left":{"control":1},"right":{"target":1}}"#),
            "{json}"
        );
        let back: ContractionTree = serde_json::from_str(&json).expect("parse");
        assert_eq!(back, tree);
        let got = contract_network(2, &gates, &tree, None).expect("network");
        assert_close(got, dense_zero(2, &gates));
    }

    #[test]
    fn bad_split_trees_are_rejected() {
        let n = 3usize;
        let gates = vec![Gate::H(0), Gate::CNOT(0, 1), Gate::CCNOT(0, 1, 2)];
        let good = greedy_tree_of(n, &gates, split_leaves(n, &gates));
        validate(n, &gates, &good, None).expect("good");
        let check = |tree: &ContractionTree, needle: &str| {
            let err = validate(n, &gates, tree, None).unwrap_err();
            assert!(err.to_string().contains(needle), "{err} lacks {needle}");
            assert!(is_contraction_client_error(&err), "{err}");
        };
        let replace = |from: TensorRef, to: TensorRef| {
            let mut tree = good.clone();
            for node in &mut tree.nodes {
                let ContractionNode::Contract { left, right } = node else {
                    unreachable!()
                };
                for side in [left, right] {
                    if *side == from {
                        *side = to;
                    }
                }
            }
            tree
        };

        check(
            &replace(TensorRef::Target(1), TensorRef::Gate(1)),
            "mixes a gate and its halves",
        );
        check(
            &replace(TensorRef::Target(1), TensorRef::Control(1)),
            "reuses a tensor",
        );
        check(
            &replace(TensorRef::Target(1), TensorRef::Control(0)),
            "gate 0 has no halves",
        );
        check(
            &replace(TensorRef::Target(1), TensorRef::Target(2)),
            "gate 2 has no halves",
        );
        check(
            &replace(TensorRef::Target(1), TensorRef::Target(9)),
            "gate is missing",
        );

        // One half only: the bond stays open, so a leaf is left out first.
        let mut tree = good.clone();
        let at = tree
            .nodes
            .iter()
            .position(|node| matches!(node, ContractionNode::Contract { left, right } if *left == TensorRef::Target(1) || *right == TensorRef::Target(1)))
            .expect("target");
        tree.nodes.truncate(at);
        check(&tree, "leaves a leaf out");

        // The exact (state) tree has no halves.
        let mut exact = crate::tn::plan::plan_greedy(n, &gates).expect("state tree");
        let ContractionNode::Contract { left, .. } = &mut exact.nodes[0] else {
            panic!("exact trees start with a contraction")
        };
        *left = TensorRef::Control(1);
        let mut mps = MpsState::try_new_with_bond(n, 4).expect("mps");
        let err = contract_tree(n, &gates, &exact, None, &mut mps).unwrap_err();
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
    fn matmul_paths_match_the_definition() {
        // Small sides on either operand, then GEMM, including k = 1 (the
        // column-major product) with both sides past SMALL_SIDE.
        for (m, k, n) in [
            (2, 2, 300),
            (300, 2, 4),
            (1, 1, 1),
            (100, 1, 80),
            (70, 4, 90),
        ] {
            let fill = |len: usize, seed: f64| -> Vec<Complex64> {
                (0..len)
                    .map(|i| Complex64::new((i as f64 * seed).sin(), (i as f64 * seed).cos()))
                    .collect()
            };
            let a = fill(m * k, 0.37);
            let b = fill(n * k, 0.61);
            let got = matmul_transposed(&a, &b, m, k, n).expect("matmul");
            for i in 0..m {
                for j in 0..n {
                    let want: Complex64 = (0..k).map(|x| a[i * k + x] * b[j * k + x]).sum();
                    assert!(
                        (got[i * n + j] - want).norm() < 1e-12,
                        "{m}x{k}x{n} at {i},{j}"
                    );
                }
            }
        }
    }

    #[test]
    fn join_cost_matches_the_planner() {
        // The orchestrator's TestNetworkJoinCountsTraffic: the left input has
        // the shared edge last and is copied. 3 × 64 + 2 × 64 bytes, 2^3 flops.
        assert_eq!(join_cost(&[1, 2], &[2, 3]), (320.0, 8.0));
        // Already in order: no copy.
        assert_eq!(join_cost(&[2, 1], &[2, 3]), (192.0, 8.0));
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "512 MiB benchmark; about a minute unoptimized"
    )]
    fn rates_are_measured() {
        let rates = tn_rates().expect("rates");
        assert!(rates.seconds_per_byte > 0.0 && rates.seconds_per_byte.is_finite());
        assert!(rates.seconds_per_flop > 0.0 && rates.seconds_per_flop.is_finite());
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

    /// Split leaves absorbed into one tensor, by highest wire (lowest when
    /// `reverse`), then leaf order.
    fn split_sweep_tree(n: usize, gates: &[Gate], reverse: bool) -> ContractionTree {
        let wires = |i: u32| gate_wires(&gates[i as usize]).expect("wires");
        let top = |r: &TensorRef| -> u32 {
            let (low, high) = match *r {
                TensorRef::Ket0(w) | TensorRef::Bra0(w) => (w, w),
                TensorRef::Control(i) => (wires(i)[0], wires(i)[0]),
                TensorRef::Target(i) => (wires(i)[1], wires(i)[1]),
                TensorRef::Gate(i) => {
                    let w = wires(i);
                    (
                        *w.iter().min().expect("wire"),
                        *w.iter().max().expect("wire"),
                    )
                }
                _ => unreachable!(),
            };
            if reverse {
                n as u32 - 1 - low
            } else {
                high
            }
        };
        let mut leaves = split_leaves(n, gates);
        leaves.sort_by_key(|(r, _)| top(r));
        let mut nodes = Vec::new();
        let mut acc = leaves[0].0;
        for (r, _) in &leaves[1..] {
            nodes.push(ContractionNode::Contract {
                left: acc,
                right: *r,
            });
            acc = TensorRef::Node((nodes.len() - 1) as u32);
        }
        finish(n, gates, nodes)
    }

    #[test]
    fn split_sweep_matches_dense() {
        let n = 10usize;
        let gates = brickwork(n, 12);
        let expected = dense_zero(n, &gates);
        for reverse in [false, true] {
            let tree = split_sweep_tree(n, &gates, reverse);
            let got = contract_network(n, &gates, &tree, None).expect("network");
            assert_close(got, expected);
        }
    }

    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn timing_split_sweep() {
        for (n, depth) in [(20usize, 32usize), (30, 32), (50, 32), (100, 32), (100, 40)] {
            let gates = brickwork(n, depth);
            let mut amps = Vec::new();
            for reverse in [false, true] {
                let tree = split_sweep_tree(n, &gates, reverse);
                let Some(peak) = tree.peak_bytes else {
                    eprintln!("{n}q depth {depth}: over budget");
                    continue;
                };
                let t = std::time::Instant::now();
                let amp = contract_network(n, &gates, &tree, None).expect("network");
                eprintln!(
                    "{n}q depth {depth} reverse={reverse}: split sweep peak {peak} B, contract {:?}, amp {amp:e}",
                    t.elapsed()
                );
                amps.push(amp);
            }
            if amps.len() == 2 {
                let scale = amps[0].norm().max(1e-300);
                eprintln!(
                    "  relative difference {:e}",
                    (amps[0] - amps[1]).norm() / scale
                );
            }
            if n <= 20 {
                eprintln!("  dense {:e}", dense_zero(n, &gates));
            }
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
