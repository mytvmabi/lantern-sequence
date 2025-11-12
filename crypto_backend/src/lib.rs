// Cryptographic Backend Library
// 
// This library provides KZG polynomial commitments and range proofs
// for the RIV protocol. Based on BLS12-381 curve with 128-bit security.

pub mod kzg;
pub mod utils;
pub mod zk_exp_lib;
pub mod metrics;
pub mod range_proof;

// Re-export main types
pub use kzg::KZG;
pub use range_proof::RangeProof;
pub use zk_exp_lib::{ZkExpSystem, TestField};

// Type aliases for BLS12-381
pub type BLS12381Pairing = ark_bls12_381::Bls12_381;
pub type BLS12381Fr = ark_bls12_381::Fr;
pub type BLS12381G1 = ark_bls12_381::G1Projective;
pub type BLS12381G2 = ark_bls12_381::G2Projective;

// Python bindings (if feature enabled)
#[cfg(feature = "python")]
pub mod python_bindings;
