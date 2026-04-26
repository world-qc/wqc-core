# wqc-core (The Engine)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-yellow.svg)]()

`wqc-core` is the computational heart of the World Quantum Computer (WQC) protocol.
It is a high-performance quantum circuit simulator written in Rust, optimized for **Proof of Useful Work (PoUW)** in a decentralized environment.

## Key Features
- **High-Precision Simulation**: Optimized in-place state vector manipulation for universal quantum circuits.
- **Bit-Precision PoUW**: Enhanced PoW algorithm allowing bit-level difficulty adjustment (leading zero bits) for granular network scaling.
- **Argon2id Memory Hardness**: ASIC-resistant hashing that cryptographically binds the quantum state integrity to the computation proof.
- **Lightweight Verification**: A dedicated /verify logic that allows nodes to validate remote proofs in $O(1)$ time without re-running the quantum simulation.
- **Cross-Platform Data Alignment**: Standardized [re, im] float array serialization for seamless interoperability between Rust, Go, and Python.

## Technical Architecture & Roadmap
The WQC project evolves through three strategic phases to achieve a global-scale decentralized quantum computer.

### ✅ Phase 1: Foundation (Current)
*Focus: Single-node optimization and trust protocol.*
- [x] **Universal Gate Set**: Support for H, X, Y, Z, T, S, CNOT, CZ, RX, RY, RZ and CCNOT (Toffoli).
- [x] **State Vector Engine**: Successfully simulated 30 Qubits on consumer hardware.
- [x] **Trust Anchor**: Implementation of the "Compute -> Mine -> Verify" cycle.
- [x] **API Alignment**: Unified data structures across Orchestrator and Swarm Nodes.

### 🚧 Phase 2: Scaling & Distribution (Upcoming)
*Focus: Breaking the memory wall via parallelization.*
- [ ] **Tensor Network (TN) Engine**: Transition from state vectors to TN contraction to allow circuit slicing.
- [ ] **Distributed Processing**: Splitting large-scale circuits across multiple swarm nodes.
- [ ] **Data Sharding**: Mechanisms to store and retrieve large quantum states across the P2P network.

### 🚀 Phase 3: Sovereign Network
*Focus: Privacy and hardware integration.*
- [ ] **zk-STARKs Integration**: Near-instant verification of distributed tasks via Zero-Knowledge proofs.
- [ ] **QPU Proxy**: Support for connecting physical quantum processing units to the WQC protocol.
- [ ] **Economic Layer**: Integration of the $WQC token for automated reward distribution.

## API Usage
### Compute Task (Prover Role)

Execute a circuit and generate a PoUW.

```bash
curl -X POST http://localhost:3000/compute \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-001",
    "qubit_count": 3,
    "circuit": [{ "type": "H", "params": 0 }],
    "difficulty": 10,
    "memory_cost_kb": 4096
  }'
```

### Verify Result (Validator Role)

Verify a result from another node without re-simulating the circuit.

```bash
curl -X POST http://localhost:8081/verify \
  -H "Content-Type: application/json" \
  -d '{
    "state_vector": [[0.707, 0.0], [0.707, 0.0], ...],
    "proof": { "nonce": 123, "proof_hash": "..." },
    "difficulty": 10,
    "memory_cost_kb": 4096
  }'
```

## Requirements
- **Rust**: 1.95+
- **Memory**: 16GB+ RAM (32GB+ recommended for >29 Qubits)
- **Dependencies**: num-complex (with serde feature), argon2, sha3

## License
Distributed under the GNU General Public License v3.0 (GPLv3). See `LICENSE` for more information.
