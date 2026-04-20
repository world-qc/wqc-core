# wqc-core (The Engine)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-yellow.svg)]()

`wqc-core` is the computational heart of the World Quantum Computer (WQC) protocol. It is a high-performance quantum circuit simulator written in Rust, optimized for **Proof of Useful Work (PoUW)** in a decentralized environment.

## Key Features
- **In-place State Vector Engine**: High-efficiency memory management capable of simulating **30 Qubits (16GB state vector)** on consumer-grade hardware (32GB RAM).
- **Parallel Gate Operations**: Powered by rayon and raw pointer optimizations for maximum CPU/RAM bandwidth utilization.
- **Argon2id-based PoUW**: An ASIC-resistant Proof of Useful Work algorithm that links quantum state integrity with memory-hard hashing.
- **Universal Gate Set**: Support for H, X, Y, Z, T, CNOT, and **CCNOT (Toffoli)** gates, enabling universal quantum computation.
- **Stateless JSON-RPC API**: A built-in HTTP server (Axum) for receiving circuit tasks and returning verified proofs.

## Technical Architecture
Currently, WQC-Core utilizes a **High-Performance State Vector Simulation**. To reach the 100+ Qubit milestone, we are moving towards a distributed architecture:

1. **Phase 1 (Current)**: Optimized In-place State Vector manipulation (up to 30+ Qubits).
2. **Phase 2 (Planned)**: **Tensor Network** contraction to allow circuit slicing and distributed processing across the swarm.
3. **Phase 3 (Planned)**: Integration of **zk-STARKs** for near-instant verification of large-scale distributed tasks, replacing the current re-computation verification.

## Technical Milestones
- [x] **30 Qubit Barrier**: Successfully simulated a 30-qubit circuit using in-place transformation within a 20GB memory limit (Colima/Docker).
- [x] **Universal Computation**: Implemented Toffoli gates, enabling complex algorithms like Grover's Search.
- [x] **Verification Loop**: Full "Compute -> Mine -> Verify" cycle implemented and validated.

## Requirements
- Rust 1.95+
- 16GB+ RAM (32GB recommended for >29 Qubit simulations)
- Docker & Docker Compose (for orchestrated deployment)

## API Usage
You can request quantum computations via JSON-RPC:

```bash
curl -X POST http://localhost:3000/compute \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "grover-001",
    "qubit_count": 5,
    "circuit": [
      { "type": "H", "params": 0 },
      { "type": "CCNOT", "params": [0, 1, 2] }
    ],
    "difficulty": 1,
    "memory_cost_kb": 4096
  }'
```

## License
Distributed under the GNU General Public License v3.0 (GPLv3). See `LICENSE` for more information.
