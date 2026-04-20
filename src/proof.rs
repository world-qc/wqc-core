use argon2::{
    password_hash::{PasswordHasher, SaltString},
    Argon2, Params, Algorithm, Version,
};
use sha3::{Digest, Sha3_256};
use num_complex::Complex64;

pub struct PoUWResult {
    pub nonce: u64,
    pub proof_hash: String,
}

pub struct Miner {
    pub difficulty: u32,
    pub memory_cost_kb: u32,
}

impl Miner {
    pub fn new(difficulty: u32, memory_cost_kb: u32) -> Self {
        Self { difficulty, memory_cost_kb }
    }

    pub fn solve(&self, state_vector: &[Complex64]) -> PoUWResult {
        let mut nonce = 0u64;

        // 1. Commit the quantum state
        let mut hasher = Sha3_256::new();
        for val in state_vector {
            hasher.update(val.re.to_le_bytes());
            hasher.update(val.im.to_le_bytes());
        }
        let state_hash = hasher.finalize();

        // 2. Setup Argon2 (Corrected for v0.5.3)
        // Note: Version::V19 is the standard for Argon2 v1.3
        let params = Params::new(self.memory_cost_kb, 3, 4, None)
            .expect("Invalid Argon2 parameters");
        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

        // 3. PoW Loop
        loop {
            let mut input = state_hash.to_vec();
            input.extend_from_slice(&nonce.to_le_bytes());

            // Create a salt from the state hash
            let salt = SaltString::encode_b64(&state_hash[..16]).unwrap();

            if let Ok(hash_output) = argon2.hash_password(&input, &salt) {
                // hash_output.hash is a field, not a method
                if let Some(hash_bytes) = hash_output.hash {
                    if hash_bytes.as_bytes()[0] == 0 {
                        return PoUWResult {
                            nonce,
                            proof_hash: hash_bytes.to_string(),
                        };
                    }
                }
            }
            nonce += 1;
        }
    }
}

pub struct Validator {
    pub difficulty: u32,
    pub memory_cost_kb: u32,
}

impl Validator {
    pub fn new(difficulty: u32, memory_cost_kb: u32) -> Self {
        Self { difficulty, memory_cost_kb }
    }

    /// Quickly verify if the provided proof is valid for the given state
    pub fn verify(&self, state_vector: &[Complex64], proof: &PoUWResult) -> bool {
        // 1. Re-calculate the state hash (The commitment)
        let mut hasher = Sha3_256::new();
        for val in state_vector {
            hasher.update(val.re.to_le_bytes());
            hasher.update(val.im.to_le_bytes());
        }
        let state_hash = hasher.finalize();

        // 2. Setup the same Argon2 parameters
        let params = Params::new(self.memory_cost_kb, 3, 4, None).unwrap();
        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

        let mut input = state_hash.to_vec();
        input.extend_from_slice(&proof.nonce.to_le_bytes());
        let salt = SaltString::encode_b64(&state_hash[..16]).unwrap();

        // 3. Check if the re-calculated hash matches and satisfies difficulty
        if let Ok(hash_output) = argon2.hash_password(&input, &salt) {
            if let Some(hash_bytes) = hash_output.hash {
                // Check if the hash matches the one reported by the miner
                if hash_bytes.to_string() != proof.proof_hash {
                    return false;
                }
                // Check if it satisfies difficulty (first byte is 0)
                if hash_bytes.as_bytes()[0] == 0 {
                    return true;
                }
            }
        }
        false
    }
}
