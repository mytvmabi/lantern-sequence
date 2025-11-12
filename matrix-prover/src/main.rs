//main.rs
pub mod kzg;
pub mod asvc;
pub mod utils;
pub mod zk_matrix;
pub mod benchmark;

/// Main entry point for the Zero-Knowledge Matrix Multiplication Proofs
/// 
/// This application demonstrates and benchmarks our novel approach to proving
/// matrix multiplication in zero-knowledge with optimal efficiency.
fn main() {
    println!("Zero-Knowledge Matrix Multiplication Proofs");
    println!("===========================================");
    
    // Run comprehensive benchmark suite to evaluate performance
    println!("\nRunning comprehensive benchmark suite...");
    benchmark::run_benchmark_suite();
    
    // Evaluate parallel processing capabilities
    println!("\nEvaluating parallelization efficiency...");
    benchmark::run2_parallelization_benchmark();
    
    // Analyze batch processing efficiency to demonstrate scalability
    println!("\nAnalyzing batch processing efficiency...");
    benchmark::run_batch_efficiency_benchmark();
    
    // Run specific matrix proof tests
    println!("\nRunning matrix proof verification tests...");
    //benchmark::test_matrix_proof();
    
    // Final instructions for data visualization
    println!("\nAll benchmarks complete. Generated data files are ready for analysis.");
    println!("To visualize results, run the Python script:");
    println!("python generate_plots.py");
}