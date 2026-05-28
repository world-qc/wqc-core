# wqc-core (The Engine)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-yellow.svg)]()

`wqc-core` is the computational heart of the World Quantum Computer (WQC) protocol.
It is a high-performance quantum circuit simulator written in Rust, optimized for **Proof of Useful Work (PoUW)** in a decentralized environment.

## Key Features
- **High-Precision Simulation**: Optimized in-place state vector manipulation for universal quantum circuits.
- **Zero-Knowledge PoUW**: True Proof of Useful Work driven by zk-STARKs, mathematically proving that the quantum simulation was executed honestly according to the specified constraints.
- **Dynamic Circuit Commitments**: Cryptographically binds the exact quantum gate layout to a unique `circuit_id` (SHA3-256 hash), preventing malicious task substitution or trace tampering.
- **Succinct Non-Interactive Verification**: Leverages the FRI protocol to allow the orchestrator to validate complex simulation proofs in polylogarithmic time without re-running any quantum gates.
- **Cross-Platform Data Alignment**: Standardized float array serialization for seamless cryptographic and state interoperability between Rust, Go, and Python backends.

## Technical Architecture & Roadmap
The WQC project evolves through three strategic phases to achieve a global-scale decentralized quantum computer.

### ✅ Phase 1: Foundation (Completed)
*Focus: Single-node optimization and trust protocol.*
- [x] **Universal Gate Set**: Support for H, X, Y, Z, T, S, CNOT, CZ, RX, RY, RZ and CCNOT (Toffoli).
- [x] **State Vector Engine**: Successfully simulated 30 Qubits on consumer hardware.
- [x] **Trust Anchor**: Implementation of the "Compute -> Mine -> Verify" cycle.
- [x] **API Alignment**: Unified data structures across Orchestrator and Swarm Nodes.

### 🚧 Phase 2: Scaling & Distribution (Current)
*Focus: Breaking the memory wall via parallelization.*
- [x] **zk-STARKs Integration**: Near-instant verification of distributed tasks via Zero-Knowledge proofs.
- [ ] **Tensor Network (TN) Engine**: Transition from state vectors to TN contraction to allow circuit slicing.
- [ ] **Distributed Processing**: Splitting large-scale circuits across multiple swarm nodes.
- [ ] **Data Sharding**: Mechanisms to store and retrieve large quantum states across the P2P network.

### 🚀 Phase 3: Sovereign Network (Upcoming)
*Focus: Privacy and hardware integration.*
- [ ] **QPU Proxy**: Support for connecting physical quantum processing units to the WQC protocol.
- [ ] **Economic Layer**: Integration of the $WQC token for automated reward distribution.

## API Reference

### Data Parameters Definition

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `task_id` | `string` | Unique identifier for the computation task. |
| `circuit_id` | `string` |  |
| `node_id` | `string` |  |
| `qubit_count`| `int` | Number of qubits ($N$). Memory cost scales by $2^N \times 16$ bytes. |
| `original_qubit_count` | `int` |  |
| `global_offset` | `string` |  |
| `circuit` | `array` | Sequence of quantum gates (H, CNOT, CCNOT, etc.) to be applied. |

---

### 1. Compute Task (`POST /compute`)
Execute a quantum circuit and generate a cryptographic proof of work.

**Example Request:**
```json
{
  "task_id": "job-550e8400",
  "circuit_id": "sha3-256-hash-of-the-pruned-circuit",
  "node_id": "node-identifier-of-itself",
  "qubit_count": 5,
  "circuit": [
    { "type": "H", "params": 0 },
    { "type": "CCNOT", "params": [0, 1, 2] }
  ],
}
```

**Example Response:**
```json
{
  "task_id": "job-550e8400",
  "status": "success",
  "state_vector": [[0.707, 0.0], [0.0, 0.0], "..."],
  "proof": {
    "public_inputs": {
      "circuit_id": "sha3-256-hash-of-the-pruned-circuit",
      "sub_task_id": "job-550e8400",
      "node_id": "node-identifier-of-itself",
      "output_result_hash": "sha3-256-hash-of-the-result-state-vector",
    },
    "stark_proof_b64": "base64-encoded-zk-proof-trace",
  },
}
```

---

### 2. Verify Proof (`POST /verify`)
Audit a computation result submitted by another node. This is a stateless, high-speed operation that checks the integrity of the state vector and the validity of the PoUW hash.

**Example Request:**
```json
{
  "proof": {
    "public_inputs": {
      "circuit_id": "sha3-256-hash-of-the-pruned-circuit",
      "sub_task_id": "job-550e8400",
      "node_id": "node-identifier-of-itself",
      "output_result_hash": "sha3-256-hash-of-the-result-state-vector",
    },
    "stark_proof_b64": "base64-encoded-zk-proof-trace",
  },
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

### Error Handling & Reliability
`wqc-core` implements strict resource guarding to ensure node stability.

| HTTP Code | Situation | Description |
| :--- | :--- | :--- |
| `400 Bad Request` | Invalid Parameters | Triggered if `memory_cost_kb` is below the absolute minimum (8 KiB) or qubit indices are out of bounds. |
| `403 Forbidden` | Verification Failed | Triggered during `/verify` if the state hash does not match the proof or the difficulty requirement is not met. |
| `503 Service Unavailable` | Resource Busy | **Crucial for Orchestrators**: Triggered when the requested task exceeds 70% of the currently available system memory. The Orchestrator should re-route the task to another node. |

## Requirements & Security Policy
- **Rust**: 1.95+
- **Memory**: 16GB+ RAM (32GB+ recommended for >29 Qubits)
- **Dependencies**: num-complex (with serde feature), sha3
- **Concurrency**: The node automatically manages concurrency based on real-time RAM availability. Large-scale simulations (e.g., >29 Qubits) will lock the resource until completion to prevent OOM panics.

## License
Distributed under the GNU General Public License v3.0 (GPLv3). See `LICENSE` for more information.
