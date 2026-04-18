# wqc-core (The Engine)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-yellow.svg)]()

`wqc-core` is the heart of the World Quantum Computer (WQC) protocol. It is a high-performance computational library written in Rust, designed to handle large-scale quantum simulations through decentralized resources.

## Key Features
- **Swarm-Tensor Engine**: Advanced tensor network contraction algorithms optimized for distributed memory-bound tasks.
- **MHQH (Memory-Hard Quantum Hybrid)**: An ASIC-resistant PoUW algorithm that prioritizes memory bandwidth over raw hashing power.
- **zk-STARK Integrity**: Built-in zero-knowledge proof generation to ensure trustless computational results.

## Technical Architecture
WQC breaks down 100+ Qubit circuits into manageable "slices." This core handles:
1. Circuit decomposition into tensor networks.
2. Optimal contraction path calculation.
3. Polynomial-time verification of results.

## Requirements
- Rust 1.75+
- CUDA / Metal (Optional, for GPU acceleration)

## Installation
```bash
git clone git@github.com:world-qc/wqc-core.git
cd wqc-core
cargo build --release
```

## License
Distributed under the GNU General Public License v3.0 (GPLv3). See `LICENSE` for more information.
