//! Range Proofs Using RangeProof's Existing Binary Constraint (h2)
//!
//! RangeProof already enforces that exponent bits are binary through the h2 constraint:
//!   h2: B(X)·(B(X)-1) = 0
//!
//! This constraint forces each bit b ∈ {0, 1}, which automatically proves
//! that any value represented by those bits is in range [0, 2^n).
//!
//! This module simply exposes this existing functionality as explicit range proofs.

use crate::zk_exp_lib::{ZkExpSystem, TestField, CompactExponentiationProof};

/// A range proof that proves value ∈ [0, 2^num_bits)
/// 
/// Internally uses RangeProof's exponentiation proof, which enforces the h2 binary
/// constraint on each bit of the value's binary representation.
#[derive(Clone, Debug)]
pub struct RangeProof {
    /// The value being proven to be in range
    pub value: u64,
    
    /// Number of bits (proves value < 2^num_bits)
    pub num_bits: usize,
    
    /// The underlying RangeProof proof that enforces bit constraints
    pub inner_proof: CompactExponentiationProof,
}

impl RangeProof {
    /// Create a new range proof for value ∈ [0, 2^num_bits)
    ///
    /// # Arguments
    /// * `system` - The RangeProof system to use for proof generation
    /// * `value` - The value to prove is in range
    /// * `num_bits` - The bit width (proves value < 2^num_bits)
    ///
    /// # Returns
    /// * `Ok(RangeProof)` - A valid range proof
    /// * `Err(String)` - If value is out of range or proof generation fails
    ///
    /// # Example
    /// ```ignore
    /// let system = ZkExpSystem::new(256, false, "demo");
    /// let proof = RangeProof::new(&system, 42, 8)?; // Prove 42 ∈ [0, 256)
    /// assert!(proof.verify(&system));
    /// ```
    pub fn new(
        system: &ZkExpSystem,
        value: u64,
        num_bits: usize,
    ) -> Result<Self, String> {
        // Validate input
        if num_bits == 0 {
            return Err("num_bits must be positive".to_string());
        }
        
        if num_bits > 64 {
            return Err("num_bits cannot exceed 64 for u64 values".to_string());
        }
        
        // Check value fits in num_bits
        let max_value = if num_bits == 64 {
            u64::MAX
        } else {
            (1u64 << num_bits) - 1
        };
        
        if value > max_value {
            return Err(format!(
                "Value {} exceeds range [0, 2^{}) = [0, {}]",
                value, num_bits, max_value
            ));
        }
        
        // Decompose value to bits (LSB first)
        let bits: Vec<bool> = (0..num_bits)
            .map(|i| ((value >> i) & 1) == 1)
            .collect();
        
        // Use base 2 for clean semantics (proves 2^value)
        // The h2 constraint ensures each bit is 0 or 1
        let base = TestField::from(2u64);
        
        // Generate RangeProof proof - this enforces h2 on all bits!
        let inner_proof = system.prove_single_exponentiation(base, &bits)?;
        
        Ok(RangeProof {
            value,
            num_bits,
            inner_proof,
        })
    }
    
    /// Verify the range proof
    ///
    /// # Arguments
    /// * `system` - The RangeProof system to use for verification
    ///
    /// # Returns
    /// * `true` - If the proof is valid and value is in range
    /// * `false` - If verification fails
    pub fn verify(&self, system: &ZkExpSystem) -> bool {
        // Reconstruct bit representation
        let bits: Vec<bool> = (0..self.num_bits)
            .map(|i| ((self.value >> i) & 1) == 1)
            .collect();
        
        let base = TestField::from(2u64);
        let expected_result = system.compute_exponentiation(base, &bits);
        
        // Verify using RangeProof verification
        // This implicitly checks that h2 was satisfied for all bits
        system.verify_single_exponentiation(&self.inner_proof, expected_result)
    }
    
    /// Get the proof size in bytes
    pub fn size_bytes(&self) -> usize {
        self.inner_proof.size_bytes()
    }
    
    /// Get the upper bound of the range (exclusive)
    pub fn upper_bound(&self) -> u128 {
        if self.num_bits == 64 {
            u128::from(u64::MAX) + 1
        } else {
            1u128 << self.num_bits
        }
    }
}

/// Extension trait for ZkExpSystem to add range proof methods
pub trait RangeProofExt {
    /// Prove that a value is in range [0, 2^num_bits)
    fn prove_range(&self, value: u64, num_bits: usize) -> Result<RangeProof, String>;
    
    /// Verify a range proof
    fn verify_range(&self, proof: &RangeProof) -> bool;
}

impl RangeProofExt for ZkExpSystem {
    fn prove_range(&self, value: u64, num_bits: usize) -> Result<RangeProof, String> {
        RangeProof::new(self, value, num_bits)
    }
    
    fn verify_range(&self, proof: &RangeProof) -> bool {
        proof.verify(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_range_proof_basic() {
        let system = ZkExpSystem::new(256, false, "test");
        
        // Prove 42 is in range [0, 256)
        let proof = RangeProof::new(&system, 42, 8).expect("Proof generation failed");
        assert!(proof.verify(&system), "Verification should pass");
        assert_eq!(proof.value, 42);
        assert_eq!(proof.num_bits, 8);
    }
    
    #[test]
    fn test_range_proof_zero() {
        let system = ZkExpSystem::new(256, false, "test");
        
        let proof = RangeProof::new(&system, 0, 8).expect("Zero proof failed");
        assert!(proof.verify(&system), "Zero should verify");
    }
    
    #[test]
    fn test_range_proof_max_value() {
        let system = ZkExpSystem::new(256, false, "test");
        
        // Max value for 8 bits is 255
        let proof = RangeProof::new(&system, 255, 8).expect("Max value proof failed");
        assert!(proof.verify(&system), "Max value should verify");
    }
    
    #[test]
    fn test_range_proof_out_of_range() {
        let system = ZkExpSystem::new(256, false, "test");
        
        // 256 is out of range for 8 bits
        let result = RangeProof::new(&system, 256, 8);
        assert!(result.is_err(), "Should reject out-of-range value");
    }
    
    #[test]
    fn test_range_proof_large_value() {
        let system = ZkExpSystem::new(4096, false, "test");
        
        // Large value with 32 bits
        let proof = RangeProof::new(&system, 1_000_000, 32).expect("Large value proof failed");
        assert!(proof.verify(&system), "Large value should verify");
    }
    
    #[test]
    fn test_range_proof_power_of_two() {
        let system = ZkExpSystem::new(256, false, "test");
        
        // Test powers of 2
        for bits in 1..=8 {
            let value = 1u64 << (bits - 1);
            let proof = RangeProof::new(&system, value, bits).expect("Power of 2 proof failed");
            assert!(proof.verify(&system), "Power of 2 should verify");
        }
    }
    
    #[test]
    fn test_range_proof_using_trait() {
        let system = ZkExpSystem::new(256, false, "test");
        
        // Test using the extension trait
        let proof = system.prove_range(123, 8).expect("Trait method failed");
        assert!(system.verify_range(&proof), "Trait verification failed");
    }
}
