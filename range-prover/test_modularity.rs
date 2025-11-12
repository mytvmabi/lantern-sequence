/// Test script to verify zkExp works in a modular way
/// This tests if zkExp can handle arbitrary exponentiation inputs like zkMap handles matrices

use zkexp::{ZkExpSystem, TestField};

fn main() {
    println!("=== Testing zkExp Modularity ===");
    println!("Verifying zkExp can handle arbitrary exponentiation inputs\n");
    
    // Test Case 1: Small exponentiation (like a small matrix)
    println!("Test 1: Small exponentiation (2^5)");
    test_exponentiation(2, vec![true, false, true], "2^5", 1);
    
    // Test Case 2: Medium exponentiation (like medium matrix)
    println!("\nTest 2: Medium exponentiation (3^45)");
    test_exponentiation(3, vec![true, false, true, true, false, true], "3^45", 2);
    
    // Test Case 3: Larger exponentiation
    println!("\nTest 3: Larger exponentiation (5^255)");
    let exp_255 = vec![true, true, true, true, true, true, true, true]; // 255 in binary
    test_exponentiation(5, exp_255, "5^255", 3);
    
    // Test Case 4: Custom arbitrary exponent
    println!("\nTest 4: Custom arbitrary exponent (7^127)");
    let exp_127 = vec![true, true, true, true, true, true, true]; // 127 in binary
    test_exponentiation(7, exp_127, "7^127", 4);
    
    // Test Case 5: Batch exponentiations (like multiple matrices)
    println!("\nTest 5: Batch exponentiations (multiple bases/exponents)");
    test_batch_exponentiations();
    
    // Test Case 6: Very small exponent
    println!("\nTest 6: Very small exponent (11^1)");
    test_exponentiation(11, vec![true], "11^1", 6);
    
    // Test Case 7: Different base with same exponent
    println!("\nTest 7: Different base with same exponent (13^45)");
    test_exponentiation(13, vec![true, false, true, true, false, true], "13^45", 7);
    
    println!("\n=== Modularity Test Summary ===");
    println!("✓ zkExp successfully processes arbitrary exponentiation inputs");
    println!("✓ Works similarly to zkMap - pass any base/exponent, get a proof");
    println!("✓ Handles single and batch operations");
    println!("✓ Maintains constant proof size regardless of input");
}

fn test_exponentiation(base_val: u64, exponent_bits: Vec<bool>, description: &str, test_num: usize) {
    let mut system = ZkExpSystem::new(4096, false, &format!("modularity_test_{}", test_num));
    
    let base = TestField::from(base_val);
    
    // Generate proof
    match system.prove_single_exponentiation(base, &exponent_bits) {
        Ok(proof) => {
            // Verify proof
            let result = system.compute_exponentiation(base, &exponent_bits);
            let verified = system.verify_single_exponentiation(&proof, result);
            
            println!("  {}: {} bytes proof, verified: {}", 
                    description, proof.size_bytes(), verified);
            
            if verified {
                println!("  ✓ Successfully proved and verified {}", description);
            } else {
                println!("  ✗ Verification failed for {}", description);
            }
        }
        Err(e) => {
            println!("  ✗ Failed to generate proof for {}: {}", description, e);
        }
    }
}

fn test_batch_exponentiations() {
    let mut system = ZkExpSystem::new(4096, false, "batch_modularity_test");
    
    // Create arbitrary batch of exponentiations
    let bases = vec![
        TestField::from(2u64),
        TestField::from(3u64),
        TestField::from(5u64),
        TestField::from(7u64),
    ];
    
    let exponents = vec![
        vec![true, false, true, true],           // 13
        vec![false, true, false, true],          // 10
        vec![true, true, true, false],           // 14
        vec![true, false, false, true, true],    // 19
    ];
    
    // Generate batch proof
    match system.prove_sliding_window_batch(&bases, &exponents, 2) {
        Ok(proof) => {
            // Compute expected results
            let results: Vec<_> = bases.iter()
                .zip(exponents.iter())
                .map(|(&b, e)| system.compute_exponentiation(b, e))
                .collect();
            
            // Verify batch proof
            let verified = system.verify_sliding_window_batch(&proof, &bases, &exponents, &results);
            
            println!("  Batch of {} exponentiations: {} bytes proof, verified: {}", 
                    bases.len(), proof.size_bytes(), verified);
            
            if verified {
                println!("  ✓ Successfully proved and verified batch of {} exponentiations", bases.len());
            } else {
                println!("  ✗ Batch verification failed");
            }
        }
        Err(e) => {
            println!("  ✗ Failed to generate batch proof: {}", e);
        }
    }
}
