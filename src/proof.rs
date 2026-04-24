use num_complex::Complex64;
use sha3::{Digest, Sha3_256};
use argon2::{
    password_hash::{PasswordHasher, SaltString},
    Argon2, Algorithm, Version, Params,
};
use serde::{Deserialize, Serialize};

/// Vision: 'The proof is the anchor of trust in a decentralized computer.'
/// PoUW (Proof of Useful Work) result structure.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PoUWResult {
    pub nonce: u64,
    pub proof_hash: String,
}

pub struct Miner {
    pub difficulty: u32,
    pub memory_cost_kb: u32,
}

impl Miner {
    /// Initialize a new Miner with network difficulty parameters.
    pub fn new(difficulty: u32, memory_cost_kb: u32) -> Self {
        Self {
            difficulty,
            memory_cost_kb,
        }
    }

    /// Internal helper to create a deterministic commitment of the quantum state.
    /// Changed return type to Vec<u8> to avoid complex generic type errors.
    fn calculate_state_hash(state_vector: &[Complex64]) -> Vec<u8> {
        let mut hasher = Sha3_256::new();
        for val in state_vector {
            hasher.update(val.re.to_le_bytes());
            hasher.update(val.im.to_le_bytes());
        }
        hasher.finalize().to_vec()
    }

    /// Check if the hash satisfies the bit-level difficulty requirement.
    /// 'difficulty' represents the required number of leading zero bits.
    pub fn check_difficulty(hash_bytes: &[u8], difficulty: u32) -> bool {
        let mut total_leading_zeros = 0;
        for &byte in hash_bytes {
            let zeros = byte.leading_zeros();
            total_leading_zeros += zeros;
            if zeros < 8 {
                // Stop at the first non-zero bit found in the current byte
                break;
            }
        }
        total_leading_zeros >= difficulty
    }

    /// Main mining function: Find a nonce that satisfies the PoUW requirements.
    pub fn solve(&self, state_vector: &[Complex64]) -> PoUWResult {
        let mut nonce = 0u64;

        // 1. Commit the quantum state
        let state_hash = Self::calculate_state_hash(state_vector);

        // 2. Setup Argon2 (Memory-hard function to prevent ASIC dominance)
        let params = Params::new(self.memory_cost_kb, 3, 4, None)
            .expect("Hardening Error: Invalid Argon2 parameters");
        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

        // Salt is derived from the state hash itself
        let salt = SaltString::encode_b64(&state_hash[..16]).unwrap();

        // 3. Iterative search for a valid nonce
        loop {
            let mut input = state_hash.clone();
            input.extend_from_slice(&nonce.to_le_bytes());

            if let Ok(hash_output) = argon2.hash_password(&input, &salt) {
                if let Some(hash_bytes) = hash_output.hash {
                    if Self::check_difficulty(hash_bytes.as_ref(), self.difficulty) {
                        return PoUWResult {
                            nonce,
                            proof_hash: hash_bytes.to_string(),
                        };
                    }
                }
            }
            nonce += 1;

            // Safety: In a real production environment, you would check
            // for cancellation tokens or timeouts here.
        }
    }

    /// Verification function: Validates the work of another node.
    /// This is a lightweight operation (O(1) Argon2 execution).
    pub fn verify(&self, state_vector: &[Complex64], proof: &PoUWResult) -> bool {
        // 1. Re-calculate state commitment
        let state_hash = Self::calculate_state_hash(state_vector);

        // 2. Setup Argon2 with identical parameters
        let params = Params::new(self.memory_cost_kb, 3, 4, None)
            .expect("Hardening Error: Failed to initialize Argon2 for verification");
        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

        let salt = SaltString::encode_b64(&state_hash[..16]).expect("Salt encoding failed");

        // 3. Re-hash using the provided nonce
        let mut input = state_hash.clone();
        input.extend_from_slice(&proof.nonce.to_le_bytes());

        if let Ok(hash_output) = argon2.hash_password(&input, &salt) {
            if let Some(hash_bytes) = hash_output.hash {
                // Verify the hash matches and difficulty is satisfied
                let hash_str = hash_bytes.to_string();
                return hash_str == proof.proof_hash &&
                       Self::check_difficulty(hash_bytes.as_ref(), self.difficulty);
            }
        }
        false
    }
}
