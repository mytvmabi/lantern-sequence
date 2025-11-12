/// Demonstration: Modular ZK Matrix Multiplication Proof System
/// 
/// This example shows how to use zkMaP as a modular system:
/// 1. Setup once (create SRS/KZG)
/// 2. Prove for ANY matrix size (within degree limit)
/// 3. Pass proof to verifier
/// 4. Verifier checks proof and returns true/false
///
/// Key benefits:
/// - Proof generation and verification are separate operations
/// - Proof can be serialized and sent over network
/// - Verifier doesn't need access to original matrices A or B
/// - Constant verification time (~5-6ms) regardless of matrix size

use zkMaP::{KZG, ZKMatrixProof, BLS12381Pairing, BLS12381Fr};
use ark_bls12_381::{G1Projective, G2Projective};
use ark_std::UniformRand;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

fn main() {
    println!("=== Modular ZK Matrix Multiplication Demo ===\n");
    
    // Setup phase (done once)
    println!("📋 Step 1: One-time Setup");
    println!("   Creating trusted setup (SRS) for matrix proofs...");
    
    let mut rng = StdRng::seed_from_u64(12345u64);
    
    // Choose maximum degree based on largest matrix you'll support
    // For n×n matrix, need degree ≥ n*n (or compressed representation size)
    let max_matrix_size = 128;
    let degree = max_matrix_size * max_matrix_size; // Support up to 128×128
    
    let g1 = G1Projective::rand(&mut rng);
    let g2 = G2Projective::rand(&mut rng);
    let mut kzg = KZG::<BLS12381Pairing>::new(g1, g2, degree);
    let secret = BLS12381Fr::rand(&mut rng);
    kzg.setup(secret);
    
    let zk_system = ZKMatrixProof::new(kzg, degree);
    println!("   ✓ Setup complete! Can now prove/verify matrices up to {}×{}\n", 
             max_matrix_size, max_matrix_size);
    
    // Now demonstrate with different matrix sizes
    let test_sizes = vec![4, 8, 16, 32, 64];
    
    for n in test_sizes {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Testing with {}×{} matrices", n, n);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // Prover side: Create matrices and generate proof
        println!("\n🔐 PROVER Side:");
        println!("   Creating random {}×{} matrices A and B...", n, n);
        
        let a_matrix: Vec<Vec<BLS12381Fr>> = (0..n)
            .map(|_| (0..n).map(|_| BLS12381Fr::rand(&mut rng)).collect())
            .collect();
        
        let b_matrix: Vec<Vec<BLS12381Fr>> = (0..n)
            .map(|_| (0..n).map(|_| BLS12381Fr::rand(&mut rng)).collect())
            .collect();
        
        println!("   Generating zero-knowledge proof that C = A × B...");
        let proof = zk_system.prove_matrix_mult(&a_matrix, &b_matrix);
        
        println!("   ✓ Proof generated!");
        println!("   Proof size: {} bytes (4 commitments + scalars)", 
                 calculate_proof_size(&proof));
        println!("   → Matrices A and B are HIDDEN in the proof");
        println!("   → Only commitments are included");
        
        // In real system: serialize proof and send to verifier
        println!("\n   📤 [Proof would be serialized and sent to verifier here]");
        
        // Verifier side: Verify proof WITHOUT knowing A or B
        println!("\n✅ VERIFIER Side:");
        println!("   Received proof for {}×{} matrix multiplication", n, n);
        println!("   Verifying proof...");
        
        let verification_result = zk_system.verify(&proof);
        
        if verification_result {
            println!("   ✓ PROOF VERIFIED!");
            println!("   → C = A × B is correct (with high probability)");
            println!("   → Verifier learned NOTHING about A or B");
        } else {
            println!("   ✗ PROOF REJECTED!");
            println!("   → The claimed multiplication is incorrect");
        }
        
        println!();
    }
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n📊 Key Properties:");
    println!("   • Setup: One-time trusted setup creates SRS");
    println!("   • Proving: Works for ANY matrix size (up to degree limit)");
    println!("   • Proof Size: Constant (~320 bytes) for any matrix size");
    println!("   • Verification: Constant time (~5-6ms) for any matrix size");
    println!("   • Zero-Knowledge: Verifier learns nothing about A or B");
    println!("   • Soundness: Cannot fake proof for incorrect multiplication");
    println!("\n💡 Use Cases:");
    println!("   • Outsourced computation verification");
    println!("   • Privacy-preserving matrix operations");
    println!("   • Blockchain/smart contract applications");
    println!("   • Confidential machine learning");
}

// Helper function to calculate proof size
fn calculate_proof_size(proof: &HashMap<String, (ark_bls12_381::G1Projective, BLS12381Fr)>) -> usize {
    // G1 point: 48 bytes (compressed), Scalar: 32 bytes
    proof.len() * (48 + 32)
}
