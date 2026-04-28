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

## API Reference

### Data Parameters Definition

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `task_id` | `string` | Unique identifier for the computation task. |
| `qubit_count`| `int` | Number of qubits ($N$). Memory cost scales by $2^N \times 16$ bytes. |
| `circuit` | `array` | Sequence of quantum gates (H, CNOT, CCNOT, etc.) to be applied. |
| `difficulty` | `uint32` | **Target Difficulty**: Number of leading zero **bits** required in the hash. |
| `memory_cost_kb`| `uint32` | Memory hardness parameter for Argon2id (ASIC resistance). |
| `nonce` | `uint64` | The solution found by the miner to satisfy the difficulty. |
| `proof_hash` | `string` | The resulting Argon2id hash in encoded string format. |
| `iterations` | `uint64` | **Actual Effort**: Total number of hash attempts performed to find the nonce. |

---

### 1. Compute Task (`POST /compute`)
Execute a quantum circuit and generate a cryptographic proof of work.

**Example Request:**
```json
{
  "task_id": "job-550e8400",
  "qubit_count": 5,
  "circuit": [
    { "type": "H", "params": 0 },
    { "type": "CCNOT", "params": [0, 1, 2] }
  ],
  "difficulty": 12,
  "memory_cost_kb": 4096
}
```

**Example Response:**
```json
{
  "task_id": "job-550e8400",
  "status": "success",
  "state_vector": [[0.707, 0.0], [0.0, 0.0], "..."],
  "proof": {
    "nonce": 9512,
    "proof_hash": "$argon2id$v=19$m=4096...",
    "iterations": 4102
  }
}
```
> **Audit Logic**: The Orchestrator expects `iterations` to be statistically close to $2^{difficulty}$. If `iterations` is consistently too low, the node may be flagged for potential pre-computation attacks or fraudulent reporting.

---

### 2. Verify Proof (`POST /verify`)
Audit a computation result submitted by another node. This is a stateless, high-speed operation that checks the integrity of the state vector and the validity of the PoUW hash.

**Example Request:**
```json
{
  "state_vector": [[0.707, 0.0], "..."],
  "proof": {
    "nonce": 9512,
    "proof_hash": "$argon2id$...",
    "iterations": 4102
  },
  "difficulty": 12,
  "memory_cost_kb": 4096
}
```

**Example Response:**
```json
{
  "valid": true,
  "reason": null
}
```

---

## Requirements
- **Rust**: 1.95+
- **Memory**: 16GB+ RAM (32GB+ recommended for >29 Qubits)
- **Dependencies**: num-complex (with serde feature), argon2, sha3

## License
Distributed under the GNU General Public License v3.0 (GPLv3). See `LICENSE` for more information.
