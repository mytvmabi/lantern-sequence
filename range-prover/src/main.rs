/// zkExp Zero-Knowledge Exponentiation Proof System
/// 
/// This module provides the main entry point for the zkExp benchmark suite,
/// implementing zero-knowledge proofs for discrete exponentiation with constant
/// verification time and proof size. The system supports multiple evaluation modes
/// including single exponentiations, batch processing, and comparative analysis.
///
/// Features:
/// - Constant-time verification independent of batch size
/// - Memory-optimized sliding window processing
/// - Comprehensive benchmarking against classical schemes
/// - Performance metrics collection and analysis
///
/// Based on the zkExp protocol using KZG polynomial commitments over BLS12-381.

/// zkExp Zero-Knowledge Exponentiation Proof System

// Core modules
mod kzg;
mod asvc;
mod utils;
mod metrics;
mod zk_exp_lib;
mod benchmark_runner;
mod single_exp_analysis;
mod backing_test;

// Baseline comparisons (feature-gated)
#[cfg(feature = "schnorr-baseline")]
pub mod schnorr_baseline;
#[cfg(feature = "bls-baseline")]
pub mod bls_baseline;
#[cfg(feature = "groth16-baseline")]
pub mod groth16_baseline;

use crate::zk_exp_lib::*;
use benchmark_runner::run_comprehensive_comparison;
use std::env;

fn main() {
    println!("zkExp Zero-Knowledge Exponentiation Proof System");
    println!("================================================");
    
    let args: Vec<String> = env::args().collect();
    let mode = if args.len() > 1 { &args[1] } else { "comprehensive" };
    
    match mode {
        "zkexp" => {
            println!("\n=== zkExp Protocol Validation ===");
            run_zkexp_validation();
        }
        "baselines" => {
            println!("\n=== Baseline Comparisons ===");
            run_comprehensive_comparison();
        }
        "comparison" => {
            println!("\n=== zkExp + Baseline Analysis ===");
            run_zkexp_validation();
            run_comprehensive_comparison();
        }
        "single-analysis" => {
            println!("\n=== Single Exponentiation Analysis ===");
            single_exp_analysis::run_single_exponentiation_analysis();
        }
        "backing-test" => {
            println!("\n=== Backing Tests vs Traditional Methods ===");
            backing_test::run_backing_test_suite();
        }
        "backing-quick" => {
            println!("\n=== Quick Validation Test ===");
            quick_backing_validation();
        }
        _ => {
            println!("\n=== Comprehensive Benchmark Suite ===");
            run_zkexp_validation();
            run_comprehensive_comparison();
        }
    }
    
    println!("\n=== Benchmark Complete ===");
    display_usage_info();
}

fn run_zkexp_validation() {
    let mut system = ZkExpSystem::new(4096, true, "zkexp_validation");
    
    // Test 1: Single exponentiation
    println!("\nTest 1: Single Exponentiation");
    let base = TestField::from(2u64);
    let exp = vec![true, false, true, true, false, true]; // 45
    
    match system.prove_single_exponentiation(base, &exp) {
        Ok(proof) => {
            let result = system.compute_exponentiation(base, &exp);
            let verified = system.verify_single_exponentiation(&proof, result);
            println!("✓ Single proof: {} bytes, verified: {}", proof.size_bytes(), verified);
        }
        Err(e) => println!("✗ Single proof failed: {}", e),
    }
    
    // Test 2: Batch exponentiation
    println!("\nTest 2: Batch Exponentiation");
    let bases = vec![TestField::from(2u64), TestField::from(3u64), TestField::from(5u64)];
    let exps = vec![
        vec![true, false, true, true],
        vec![false, true, false, true],
        vec![true, true, true, false],
    ];
    
    match system.prove_sliding_window_batch(&bases, &exps, 2) {
        Ok(proof) => {
            let results: Vec<_> = bases.iter()
                .zip(exps.iter())
                .map(|(&b, e)| system.compute_exponentiation(b, e))
                .collect();
            let verified = system.verify_sliding_window_batch(&proof, &bases, &exps, &results);
            println!("✓ Batch proof: {} bytes, verified: {}", proof.size_bytes(), verified);
        }
        Err(e) => println!("✗ Batch proof failed: {}", e),
    }
    
    // Test 3: Sliding window validation
    println!("\nTest 3: Sliding Window Validation");
    system.run_comprehensive_sliding_window_validation();
}

fn quick_backing_validation() {
    use crate::backing_test::BackingTestSuite;
    
    let mut suite = BackingTestSuite::new();
    let exp = vec![true, false, true, true];
    
    match suite.compare_single_exponentiation(
        TestField::from(2u64),
        &exp,
        4,
        "QuickValidation"
    ) {
        Ok(result) => {
            println!("✓ Correctness: {}, Verification: {}, Overhead: {:.1}x",
                    result.correctness_match,
                    result.zkexp_verification_success,
                    result.prove_overhead_factor);
        }
        Err(e) => println!("✗ Validation failed: {}", e),
    }
}

fn display_usage_info() {
    println!("\nUsage:");
    println!("  cargo run --bin enhanced_zkproofs [MODE]");
    println!("  ");
    println!("Modes:");
    println!("  zkexp              - zkExp protocol validation");
    println!("  baselines          - Baseline comparisons");
    println!("  comparison         - zkExp + baseline analysis");
    println!("  single-analysis    - Single exponentiation analysis");
    println!("  backing-test       - Backing tests vs traditional methods");
    println!("  backing-quick      - Quick validation test");
    println!("  comprehensive      - Full benchmark suite (default)");
}
