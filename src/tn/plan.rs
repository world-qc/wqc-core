//! Test double of the orchestrator's greedy exact plan.
//!
//! `PlanGreedyContraction` in `wqc-orchestrator` is the plan that ships.
//! This copy is compiled only for tests, so a planned tree can be checked
//! against a dense amplitude. The chain JSON in both tests is the same text.
//! Gates stay in circuit order. A gate is applied once every wire it touches
//! sits on one state. Joins of those states pick the fewest result legs, then
//! the younger origin gate, then the smaller minimum wire, then the smaller
//! maximum wire. When those match, the tensor that appears earlier in the
//! gate arguments is kept. The declared peak counts both inputs and the
//! result during an outer product.

use crate::engine::Gate;
use crate::tn::tree::{ContractionNode, ContractionTree, TensorRef};

const MAX_QUBITS: usize = 4096;
const MAX_GATES: usize = 100_000;
const MAX_LEGS: usize = 26;
const MAX_PEAK: u64 = 1 << 30;

fn plan_greedy(qubit_count: usize, gates: &[Gate]) -> Option<ContractionTree> {
    if qubit_count == 0 || qubit_count > MAX_QUBITS || gates.len() > MAX_GATES {
        return None;
    }
    let mut plan = Plan {
        slots: Vec::new(),
        at: vec![None; qubit_count],
        nodes: Vec::new(),
        peak: 0,
    };
    for (index, gate) in gates.iter().enumerate() {
        let Some(wires) = unitary_wires(gate, qubit_count, index)? else {
            continue;
        };
        if !plan.contract_gate(&wires, index) {
            return None;
        }
    }
    Some(ContractionTree {
        site_order: Vec::new(),
        nodes: plan.nodes,
        peak_bytes: Some(plan.peak),
    })
}

fn unitary_wires(gate: &Gate, qubit_count: usize, index: usize) -> Option<Option<Vec<u32>>> {
    match super::tree::unitary_wires(gate, qubit_count, index) {
        Ok(Some(wires)) => Some(Some(wires.into_iter().map(|w| w as u32).collect())),
        Ok(None) => Some(None),
        Err(_) => None,
    }
}

struct Slot {
    wires: Vec<u32>,
    node: Option<u32>,
    gate: Option<u32>,
    alive: bool,
}

struct Plan {
    slots: Vec<Slot>,
    at: Vec<Option<usize>>,
    nodes: Vec<ContractionNode>,
    peak: u64,
}

impl Plan {
    fn contract_gate(&mut self, wires: &[u32], gate_index: usize) -> bool {
        let mut ids = Vec::with_capacity(wires.len());
        let mut pos = Vec::<Option<usize>>::new();
        for (arg, &wire) in wires.iter().enumerate() {
            let id = self.ensure(wire);
            if id >= pos.len() {
                pos.resize(id + 1, None);
            }
            if pos[id].is_none() {
                pos[id] = Some(arg);
                ids.push(id);
            }
        }
        while ids.len() > 1 {
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best = self.pair_key(ids[0], ids[1], &pos);
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let key = self.pair_key(ids[i], ids[j], &pos);
                    if key < best {
                        best = key;
                        best_i = i;
                        best_j = j;
                    }
                }
            }
            let earlier = pos[ids[best_i]].min(pos[ids[best_j]]);
            let Some(merged) = self.join(ids[best_i], ids[best_j]) else {
                return false;
            };
            if merged >= pos.len() {
                pos.resize(merged + 1, None);
            }
            pos[merged] = earlier;
            let (lo, hi) = if best_i < best_j {
                (best_i, best_j)
            } else {
                (best_j, best_i)
            };
            ids.remove(hi);
            ids.remove(lo);
            ids.push(merged);
        }
        self.apply(ids[0], gate_index)
    }

    fn ensure(&mut self, wire: u32) -> usize {
        if let Some(id) = self.at[wire as usize] {
            return id;
        }
        let id = self.slots.len();
        self.slots.push(Slot {
            wires: vec![wire],
            node: None,
            gate: None,
            alive: true,
        });
        self.at[wire as usize] = Some(id);
        id
    }

    fn join(&mut self, mut a_id: usize, mut b_id: usize) -> Option<usize> {
        if self.slots[b_id].wires[0] < self.slots[a_id].wires[0] {
            std::mem::swap(&mut a_id, &mut b_id);
        }
        let legs = self.slots[a_id].wires.len() + self.slots[b_id].wires.len();
        if legs > MAX_LEGS {
            return None;
        }
        let inputs = add(
            leg_bytes(self.slots[a_id].wires.len())?,
            leg_bytes(self.slots[b_id].wires.len())?,
        )?;
        let during = add(add(self.others(&[a_id, b_id])?, inputs)?, leg_bytes(legs)?)?;
        if during > MAX_PEAK {
            return None;
        }
        self.peak = self.peak.max(during);
        let left = tensor_ref(&self.slots[a_id]);
        let right = tensor_ref(&self.slots[b_id]);
        let origin = younger(self.slots[a_id].gate, self.slots[b_id].gate);
        let mut wires = self.slots[a_id].wires.clone();
        wires.extend_from_slice(&self.slots[b_id].wires);
        wires.sort_unstable();
        let index = self.nodes.len() as u32;
        self.nodes.push(ContractionNode::Contract { left, right });
        self.slots[a_id].alive = false;
        self.slots[b_id].alive = false;
        let id = self.slots.len();
        self.slots.push(Slot {
            wires: wires.clone(),
            node: Some(index),
            gate: origin,
            alive: true,
        });
        for wire in wires {
            self.at[wire as usize] = Some(id);
        }
        Some(id)
    }

    fn apply(&mut self, id: usize, gate_index: usize) -> bool {
        let Some(state_bytes) = leg_bytes(self.slots[id].wires.len()) else {
            return false;
        };
        let Some(others) = self.others(&[id]) else {
            return false;
        };
        let Some(during) = add(others, state_bytes) else {
            return false;
        };
        if during > MAX_PEAK {
            return false;
        }
        self.peak = self.peak.max(during);
        let left = tensor_ref(&self.slots[id]);
        let index = self.nodes.len() as u32;
        self.nodes.push(ContractionNode::Contract {
            left,
            right: TensorRef::Gate(gate_index as u32),
        });
        self.slots[id].node = Some(index);
        self.slots[id].gate = younger(self.slots[id].gate, Some(gate_index as u32));
        true
    }

    fn pair_key(
        &self,
        a_id: usize,
        b_id: usize,
        pos: &[Option<usize>],
    ) -> (usize, u64, u32, u32, usize, usize) {
        let a = &self.slots[a_id];
        let b = &self.slots[b_id];
        let mut min_w = a.wires[0];
        let mut max_w = *a.wires.last().expect("tensor has a wire");
        if b.wires[0] < min_w {
            min_w = b.wires[0];
        }
        if *b.wires.last().expect("tensor has a wire") > max_w {
            max_w = b.wires[b.wires.len() - 1];
        }
        let pa = pos[a_id].expect("tensor is a gate argument");
        let pb = pos[b_id].expect("tensor is a gate argument");
        let (earlier, later) = if pa < pb { (pa, pb) } else { (pb, pa) };
        (
            a.wires.len() + b.wires.len(),
            origin_rank(younger(a.gate, b.gate)),
            min_w,
            max_w,
            earlier,
            later,
        )
    }

    fn others(&self, skip: &[usize]) -> Option<u64> {
        let mut sum = 0u64;
        for (index, slot) in self.slots.iter().enumerate() {
            if !slot.alive || skip.contains(&index) {
                continue;
            }
            sum = add(sum, leg_bytes(slot.wires.len())?)?;
        }
        Some(sum)
    }
}

fn tensor_ref(slot: &Slot) -> TensorRef {
    match slot.node {
        Some(index) => TensorRef::Node(index),
        None => TensorRef::Ket0(slot.wires[0]),
    }
}

fn younger(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn origin_rank(gate: Option<u32>) -> u64 {
    gate.map(u64::from).unwrap_or(u64::MAX)
}

fn leg_bytes(legs: usize) -> Option<u64> {
    if legs >= 63 {
        return None;
    }
    Some(16u64 << legs)
}

fn add(a: u64, b: u64) -> Option<u64> {
    a.checked_add(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Gate, MeasureParams};
    use crate::tn::dense::DenseTnState;
    use crate::tn::exact::contract_exact_at;
    use num_complex::Complex64;

    const CHAIN_JSON: &str = r#"{"nodes":[{"op":"contract","left":{"ket0":0},"right":{"gate":0}},{"op":"contract","left":{"node":0},"right":{"ket0":1}},{"op":"contract","left":{"node":1},"right":{"gate":1}},{"op":"contract","left":{"node":2},"right":{"ket0":2}},{"op":"contract","left":{"node":3},"right":{"gate":2}}],"peak_bytes":224}"#;

    fn assert_close(got: Complex64, expected: Complex64) {
        assert!((got - expected).norm() < 1e-8, "{got} vs {expected}");
    }

    fn dense_at(n: usize, gates: &[Gate], index: usize) -> Complex64 {
        let mut dense = DenseTnState::try_new(n).expect("dense");
        for gate in gates {
            dense.apply_gate(gate);
        }
        dense.state[index]
    }

    fn run(n: usize, gates: &[Gate], bits: usize) -> Complex64 {
        let tree = plan_greedy(n, gates).expect("tree");
        contract_exact_at(n, gates, &tree, None, bits).expect("exact")
    }

    #[test]
    fn chain_matches_the_orchestrator_json_and_dense() {
        let gates = vec![Gate::H(0), Gate::CNOT(0, 1), Gate::CNOT(1, 2)];
        let tree = plan_greedy(3, &gates).expect("tree");
        let encoded = serde_json::to_string(&tree).expect("json");
        assert_eq!(encoded, CHAIN_JSON);
        assert_close(run(3, &gates, 0), dense_at(3, &gates, 0));
        assert_close(run(3, &gates, 0b101), dense_at(3, &gates, 0b101));
    }

    #[test]
    fn younger_gate_wins_a_leg_tie_and_ccnot_matches_dense() {
        let gates = vec![Gate::H(5), Gate::H(0), Gate::H(1), Gate::CCNOT(5, 0, 1)];
        let tree = plan_greedy(6, &gates).expect("tree");
        assert_eq!(
            tree.nodes,
            vec![
                ContractionNode::Contract {
                    left: TensorRef::Ket0(5),
                    right: TensorRef::Gate(0),
                },
                ContractionNode::Contract {
                    left: TensorRef::Ket0(0),
                    right: TensorRef::Gate(1),
                },
                ContractionNode::Contract {
                    left: TensorRef::Ket0(1),
                    right: TensorRef::Gate(2),
                },
                ContractionNode::Contract {
                    left: TensorRef::Node(1),
                    right: TensorRef::Node(0),
                },
                ContractionNode::Contract {
                    left: TensorRef::Node(3),
                    right: TensorRef::Node(2),
                },
                ContractionNode::Contract {
                    left: TensorRef::Node(4),
                    right: TensorRef::Gate(3),
                },
            ]
        );
        let bits = (1usize << 5) | (1usize << 0) | (1usize << 1);
        assert_close(run(6, &gates, bits), dense_at(6, &gates, bits));
    }

    #[test]
    fn interior_pair_follows_the_gate_arguments() {
        // The span {0,5} already covers both endpoints. The two interior
        // tensors have the same number of legs, so the earlier CCNOT argument
        // chooses the partner. Both orders declare the same peak and the same
        // amplitude.
        let early = interior_tree(&[0, 1, 3]);
        let late = interior_tree(&[0, 3, 1]);
        let listed_first = interior_tree(&[1, 3, 0]);
        assert_eq!(early.nodes.len(), 9);
        assert_eq!(
            early.nodes[6],
            ContractionNode::Contract {
                left: TensorRef::Node(1),
                right: TensorRef::Node(3),
            }
        );
        assert_eq!(
            late.nodes[6],
            ContractionNode::Contract {
                left: TensorRef::Node(1),
                right: TensorRef::Node(5),
            }
        );
        assert_eq!(early.nodes[6], listed_first.nodes[6]);
        assert_eq!(early.peak_bytes, Some(1344));
        assert_eq!(early.peak_bytes, late.peak_bytes);
        let early_gates = interior_gates(&[0, 1, 3]);
        let late_gates = interior_gates(&[0, 3, 1]);
        let early_amp = contract_exact_at(6, &early_gates, &early, None, 0).expect("early");
        let late_amp = contract_exact_at(6, &late_gates, &late, None, 0).expect("late");
        assert_close(early_amp, late_amp);
        assert_close(early_amp, dense_at(6, &early_gates, 0));
        assert_close(late_amp, dense_at(6, &late_gates, 0));
    }

    fn interior_gates(ccnot: &[usize]) -> Vec<Gate> {
        vec![
            Gate::CNOT(0, 5),
            Gate::CNOT(1, 2),
            Gate::CNOT(3, 4),
            Gate::CCNOT(ccnot[0], ccnot[1], ccnot[2]),
        ]
    }

    fn interior_tree(ccnot: &[usize]) -> ContractionTree {
        plan_greedy(6, &interior_gates(ccnot)).expect("tree")
    }

    #[test]
    fn measure_keeps_the_following_gate_index() {
        let gates = vec![
            Gate::H(0),
            Gate::MEASURE(MeasureParams { qubit: 0, cbit: 0 }),
            Gate::CNOT(0, 1),
        ];
        let tree = plan_greedy(2, &gates).expect("tree");
        let mut seen = Vec::new();
        for node in &tree.nodes {
            if let ContractionNode::Contract {
                right: TensorRef::Gate(index),
                ..
            } = node
            {
                seen.push(*index);
            }
        }
        assert_eq!(seen, vec![0, 2]);
        assert_eq!(tree.peak_bytes, Some(128));
        assert_close(run(2, &gates, 0), dense_at(2, &gates, 0));
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

    #[test]
    fn brickwork_16_depth_8_matches_dense() {
        let gates = ry_cnot_brickwork(16, 8);
        let tree = plan_greedy(16, &gates).expect("tree");
        // Last product in the odd layer: 14 legs with 2 legs.
        assert_eq!(
            tree.peak_bytes,
            Some((16u64 << 14) + (16u64 << 2) + (16u64 << 16))
        );
        let dense0 = dense_at(16, &gates, 0);
        assert!(dense0.norm() > 1e-3, "all-zero amplitude vanished");
        assert_close(
            contract_exact_at(16, &gates, &tree, None, 0).expect("zero"),
            dense0,
        );
        let mid = 0b1010_1010_1010_1010;
        assert_close(
            contract_exact_at(16, &gates, &tree, None, mid).expect("mid"),
            dense_at(16, &gates, mid),
        );
    }

    #[test]
    fn brickwork_50_depth_8_has_no_tree() {
        assert!(plan_greedy(50, &ry_cnot_brickwork(50, 8)).is_none());
    }

    fn star_gates(n: usize) -> Vec<Gate> {
        let mut gates = vec![Gate::H(0)];
        for q in 1..n {
            gates.push(Gate::CNOT(0, q));
        }
        gates
    }

    fn star_peak(n: usize) -> u64 {
        if n <= 1 {
            32
        } else {
            (16u64 << (n - 1)) + (16u64 << 1) + (16u64 << n)
        }
    }

    #[test]
    fn star_16_matches_every_leaf_one() {
        let n = 16usize;
        let gates = star_gates(n);
        let tree = plan_greedy(n, &gates).expect("tree");
        assert_eq!(tree.peak_bytes, Some(star_peak(n)));
        let ones = (1usize << n) - 1;
        let dense = dense_at(n, &gates, ones);
        assert!(dense.norm() > 0.1, "leaf-|1⟩ amplitude vanished");
        assert_close(
            contract_exact_at(n, &gates, &tree, None, ones).expect("ones"),
            dense,
        );
    }

    #[test]
    fn over_budget_stars_and_caps_have_no_tree() {
        assert_eq!(
            plan_greedy(25, &star_gates(25)).expect("25").peak_bytes,
            Some(star_peak(25))
        );
        assert!(plan_greedy(26, &star_gates(26)).is_none());
        assert!(plan_greedy(40, &star_gates(40)).is_none());
        assert!(plan_greedy(MAX_QUBITS + 1, &[Gate::H(0)]).is_none());
        assert!(plan_greedy(1, &vec![Gate::H(0); MAX_GATES + 1]).is_none());
        assert_eq!(
            plan_greedy(1, &vec![Gate::H(0); MAX_GATES])
                .expect("cap")
                .nodes
                .len(),
            MAX_GATES
        );
        assert!(plan_greedy(2, &[Gate::CNOT(0, 0)]).is_none());
        assert!(plan_greedy(3, &[Gate::H(3)]).is_none());
    }

    #[test]
    fn local_gates_on_50_qubits_stay_a_small_tree() {
        let gates: Vec<Gate> = (0..50).map(Gate::H).collect();
        let tree = plan_greedy(50, &gates).expect("tree");
        assert_eq!(tree.peak_bytes, Some(50 * 32));
        assert_eq!(tree.nodes.len(), 50);
        // Each wire is its own tensor. ⟨0|H|0⟩^50 = 2^-25. A dense vector of
        // 50 qubits is not allocated.
        let amp = contract_exact_at(50, &gates, &tree, None, 0).expect("exact");
        assert_close(amp, Complex64::new(2.0_f64.powi(-25), 0.0));
    }
}
