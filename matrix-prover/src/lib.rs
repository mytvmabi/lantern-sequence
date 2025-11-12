//! zkMaP: Zero-Knowledge Matrix Multiplication Proofs
//! 
//! This library provides efficient zero-knowledge proofs for matrix
//! multiplication with dimension reduction and batching support.

pub mod kzg;
pub mod zk_matrix;
pub mod utils;

#[cfg(feature = "asvc")]
pub mod asvc;

// Re-export main types for convenience
pub use zk_matrix::{ZKMatrixProof, OptimizedBatchedZKMatrixProof, ProofMetrics};
pub use kzg::KZG;

// Type aliases for external use
pub type BLS12381Pairing = ark_bls12_381::Bls12_381;
pub type BLS12381Fr = ark_bls12_381::Fr;
pub type BLS12381G1 = ark_bls12_381::G1Projective;
pub type BLS12381G2 = ark_bls12_381::G2Projective;