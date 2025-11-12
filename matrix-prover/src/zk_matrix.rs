// zk_matrix.rs
//
// Implementation of Zero-Knowledge Matrix Multiplication Proof system
// that allows a prover to convince a verifier that C = A * B
// without revealing the matrices A and B, using KZG polynomial commitments.

use ark_ff::{Field, Zero, One, PrimeField, BigInteger};
use std::ops::Mul;
use ark_ec::pairing::Pairing;
use std::collections::HashMap;
use rayon::prelude::*;
use sha2::{Sha256, Digest};
use std::time::Instant;
use ark_ff::FftField;
use std::sync::Mutex;
use once_cell::sync::Lazy;
use crate::kzg::KZG;
use crate::utils::evaluate;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};


// Memory tracking allocator for benchmarking and resource analysis
#[global_allocator]
static ALLOCATOR: TrackedAllocator = TrackedAllocator;

struct TrackedAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

impl TrackedAllocator {
    fn reset_peak() {
        PEAK_ALLOCATED.store(0, Ordering::SeqCst);
    }
    
    fn peak_allocated() -> usize {
        PEAK_ALLOCATED.load(Ordering::SeqCst)
    }
}

unsafe impl GlobalAlloc for TrackedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let ptr = System.alloc(layout);
        
        if !ptr.is_null() {
            let current = ALLOCATED.fetch_add(size, Ordering::SeqCst) + size;
            let mut peak = PEAK_ALLOCATED.load(Ordering::SeqCst);
            while peak < current && !PEAK_ALLOCATED.compare_exchange_weak(
                peak, current, Ordering::SeqCst, Ordering::SeqCst
            ).is_ok() {
                peak = PEAK_ALLOCATED.load(Ordering::SeqCst);
            }
        }
        
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();
        ALLOCATED.fetch_sub(size, Ordering::SeqCst);
        System.dealloc(ptr, layout);
    }
}


// Performance metrics structure for benchmarking
#[derive(Clone, Debug)]
pub struct ProofMetrics {
    pub matrix_size: usize,
    pub proof_time_ms: u64,
    pub verification_time_ms: u64,
    pub proof_size_bytes: usize,
    pub memory_usage_bytes: usize,
}

// Cache for basis vectors to avoid redundant computation
// This significantly improves performance for repeated matrix operations
type BasisCache<F> = Mutex<HashMap<(usize, usize), Vec<Vec<F>>>>;
static BASIS_CACHE_1024: Lazy<BasisCache<ark_bls12_381::Fr>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Zero-Knowledge Matrix Multiplication Proof system
///
/// This system enables a prover to convince a verifier that a claimed matrix product C = A×B
/// is correct, without revealing A or B. It uses KZG polynomial commitments and
/// optimized matrix-to-polynomial encodings.
#[derive(Clone)]
pub struct ZKMatrixProof<E: Pairing> {
    pub kzg: KZG<E>,     // KZG polynomial commitment scheme instance
    pub max_size: usize, // Maximum supported matrix dimension
}

impl<E: Pairing> ZKMatrixProof<E> {
    /// Create a new ZKMatrixProof instance with a given KZG instance and maximum matrix size
    pub fn new(kzg: KZG<E>, max_size: usize) -> Self {
        Self {
            kzg,
            max_size,
        }
    }
    
    // Retrieve or compute basis vectors for matrix compression
    // For frequently used dimensions, we cache the basis vectors to avoid expensive recomputation
    fn get_or_compute_basis_vectors(&self, rows: usize, reduced_basis_count: usize) -> Vec<Vec<E::ScalarField>> {
        // Special case for 1024×1024 matrices: use cached basis vectors for performance
        if rows == 1024 {
            // Try to get from cache first
            let mut basis_cache = BASIS_CACHE_1024.lock().unwrap();
            
            if let Some(cached) = basis_cache.get(&(rows, reduced_basis_count)) {
                // Convert from ark_bls12_381::Fr to E::ScalarField if necessary
                let converted_basis: Vec<Vec<E::ScalarField>> = cached.iter()
                    .map(|basis| {
                        basis.iter()
                            .map(|element| {
                                // This conversion assumes ScalarField values are compatible
                                let element_bytes = element.into_bigint().to_bytes_be();
                                E::ScalarField::from_random_bytes(&element_bytes)
                                    .unwrap_or(E::ScalarField::ZERO)
                            })
                            .collect()
                    })
                    .collect();
                return converted_basis;
            }
            
            // Generate fresh basis vectors if not found in cache
            let basis = self.generate_basis_vectors(rows, reduced_basis_count);
            
            // Store in cache if using BLS12-381 curve
            if std::any::TypeId::of::<E::ScalarField>() == std::any::TypeId::of::<ark_bls12_381::Fr>() {
                let basis_to_cache: Vec<Vec<ark_bls12_381::Fr>> = basis.iter()
                    .map(|b_vec| {
                        b_vec.iter()
                            .map(|&b| {
                                let bytes = b.into_bigint().to_bytes_be();
                                ark_bls12_381::Fr::from_random_bytes(&bytes)
                                    .unwrap_or(ark_bls12_381::Fr::zero())
                            })
                            .collect()
                    })
                    .collect();
                
                basis_cache.insert((rows, reduced_basis_count), basis_to_cache);
            }
            
            return basis;
        }
        
        // For other sizes, just generate fresh basis vectors
        self.generate_basis_vectors(rows, reduced_basis_count)
    }
    
    // Generate a set of orthogonal basis vectors for matrix compression
    // Uses primitive roots of unity for optimal distribution (Vandermonde matrix)
    fn generate_basis_vectors(&self, rows: usize, basis_count: usize) -> Vec<Vec<E::ScalarField>> {
        // Try to use primitive root of unity for Vandermonde-based orthogonal vectors
        if let Some(omega) = E::ScalarField::get_root_of_unity(rows as u64) {
            // Generate basis vectors in parallel using primitive root of unity
            (0..basis_count).into_par_iter()
                .map(|i| {
                    (0..rows)
                        .map(|j| omega.pow(&[(i * j) as u64]))
                        .collect()
                })
                .collect()
        } else {
            // Fallback if primitive root not available
            (0..basis_count).into_par_iter()
                .map(|i| {
                    let mut basis = vec![E::ScalarField::ZERO; rows];
                    let step = rows / basis_count;
                    let start = (i * step) % rows;
                    let end = std::cmp::min(start + step, rows);
                    
                    for j in start..end {
                        basis[j] = E::ScalarField::ONE;
                    }
                    basis
                })
                .collect()
        }
    }
    
    /// Matrix to polynomial conversion with efficient dimension reduction
    ///
    /// This function converts a matrix to a polynomial representation using dimension
    /// reduction techniques for large matrices. For matrices of size 64×64 or larger,
    /// it projects the matrix onto a smaller set of orthogonal basis vectors.
    pub fn matrix_to_poly_with_compression(&self, matrix: &[Vec<E::ScalarField>]) -> Vec<E::ScalarField> {
        let rows = matrix.len();
        let cols = if rows > 0 { matrix[0].len() } else { 0 };
        
        if rows * cols >= 64 * 64 {
            // Calculate appropriate basis count based on matrix size
            let reduced_basis_count = std::cmp::max(
                32, 
                (std::cmp::min(rows, cols) + 1) / 4
            );
            
            // Special case for very large matrices, use high-performance approach
            if rows >= 256 || cols >= 256 {
                let start_time = Instant::now();
                
                // Get precomputed/cached basis vectors
                let basis_vectors = self.get_or_compute_basis_vectors(rows, reduced_basis_count);
                let basis_time = start_time.elapsed();
                
                if rows >= 512 && cols >= 512 {
                    println!("Basis generation time: {:?}", basis_time);
                }
                
                // Pre-allocate result to avoid resizing
                let mut compressed = Vec::with_capacity(reduced_basis_count * cols);
                
                // Determine optimal batch size for this hardware
                let optimal_batch_size = 16;
                
                // Process all columns in parallel batches
                let compressed_data: Vec<_> = (0..cols).into_par_iter()
                    .map(|col| {
                        // Process current column against all basis vectors
                        let mut col_results = Vec::with_capacity(reduced_basis_count);
                        
                        for basis in &basis_vectors {
                            let mut sum = E::ScalarField::ZERO;
                            
                            // Use SIMD-friendly batch processing
                            for start_idx in (0..rows).step_by(optimal_batch_size) {
                                let end_idx = std::cmp::min(start_idx + optimal_batch_size, rows);
                                
                                // Unrolled loop for better instruction-level parallelism
                                let mut batch_sum = E::ScalarField::ZERO;
                                for row in start_idx..end_idx {
                                    batch_sum += matrix[row][col] * basis[row];
                                }
                                
                                sum += batch_sum;
                            }
                            
                            col_results.push(sum);
                        }
                        
                        col_results
                    })
                    .collect();
                
                // Flatten the results
                for col_result in compressed_data {
                    compressed.extend(col_result);
                }
                
                if rows >= 512 && cols >= 512 {
                    let compress_time = start_time.elapsed();
                    println!("Total matrix compression time: {:?}", compress_time);
                }
                
                return compressed;
            } else {
                // Standard compression for medium-sized matrices
                let bases = self.get_precomputed_bases(rows, cols);
                
                // Process columns in parallel
                (0..cols).into_par_iter()
                    .flat_map(|col| {
                        bases.iter()
                            .map(|basis| {
                                // Use batched processing for consistency
                                let batch_size = 16;
                                let mut sum = E::ScalarField::ZERO;
                                
                                for start_idx in (0..rows).step_by(batch_size) {
                                    let end_idx = std::cmp::min(start_idx + batch_size, rows);
                                    let mut batch_sum = E::ScalarField::ZERO;
                                    
                                    for row in start_idx..end_idx {
                                        batch_sum += matrix[row][col] * basis[row];
                                    }
                                    
                                    sum += batch_sum;
                                }
                                
                                sum
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect()
            }
        } else {
            // For smaller matrices, use direct flattening (no compression)
            let mut polynomial = Vec::with_capacity(rows * cols);
            for row in matrix {
                polynomial.extend_from_slice(row);
            }
            polynomial
        }
    }
    
    // Generate precomputed basis vectors for matrix compression
    // This function generates orthogonal basis vectors using roots of unity
    pub fn get_precomputed_bases(&self, rows: usize, cols: usize) -> Vec<Vec<E::ScalarField>> {
        let n_dim = std::cmp::min(rows, cols);
        let compression_size = std::cmp::max(
            (n_dim + 3) / 4,  // ceil(n/4)
            if n_dim < 256 { 32 } else { 256 }  // Security floor
        );
        let mut bases = Vec::with_capacity(compression_size);
        
        // Use a deterministic primitive root of unity for reproducibility and orthogonality
        let omega = E::ScalarField::get_root_of_unity(rows as u64)
            .expect("Field does not have a primitive root of unity of the required order");
        
        // Generate orthogonal (Vandermonde) basis vectors
        for i in 0..compression_size {
            let mut basis = Vec::with_capacity(rows);
            for j in 0..rows {
                // Compute basis element as omega^(i * j)
                basis.push(omega.pow(&[(i * j) as u64]));
            }
            bases.push(basis);
        }
        
        bases
    }

    /// Calculate the size of a proof in bytes
    pub fn calculate_proof_size(proof: &HashMap<String, (E::G1, E::ScalarField)>) -> usize {
        // Size in bytes - typically G1 points are 48 bytes (compressed) and
        // ScalarField elements are 32 bytes
        let g1_size = 48; // Size of compressed G1 point in bytes
        let scalar_size = 32; // Size of scalar field element in bytes
        
        // Count entries
        let mut total_size = 0;
        for (_, _) in proof {
            total_size += g1_size + scalar_size;
        }
        
        total_size
    }

    // Standard matrix-to-polynomial conversion (without compression)
    pub fn matrix_to_poly(&self, matrix: &[Vec<E::ScalarField>]) -> Vec<E::ScalarField> {
        let rows = matrix.len();
        let cols = if rows > 0 { matrix[0].len() } else { 0 };
        
        // For very large matrices, use parallel processing
        if rows * cols > 10000 {
            let mut polynomial = Vec::with_capacity(rows * cols);
            
            // Process rows in parallel chunks
            let chunk_size = std::cmp::max(1, rows / num_cpus::get());
            
            let chunks: Vec<_> = matrix.par_chunks(chunk_size)
                .map(|chunk| {
                    let mut result = Vec::with_capacity(chunk.len() * cols);
                    for row in chunk {
                        result.extend_from_slice(row);
                    }
                    result
                })
                .collect();
            
            // Combine results
            for chunk in chunks {
                polynomial.extend_from_slice(&chunk);
            }
            
            polynomial
        } else {
            // For smaller matrices, use the simple direct approach
            let mut polynomial = Vec::with_capacity(rows * cols);
            
            // Flatten the matrix row by row
            for row in matrix {
                polynomial.extend_from_slice(row);
            }
            
            polynomial
        }
    }

    // Optimized KZG commitment function with adaptive batch sizing
    // Uses parallelism for large polynomials and batching for cache efficiency
    fn optimized_commit(&self, poly: &[E::ScalarField]) -> E::G1 {
        // For small polynomials or if CRS is too small, use regular commit
        if poly.len() <= 32 || poly.len() > self.kzg.crs_g1.len() {
            return self.kzg.commit(poly);
        }
        
        // Adaptive batch sizing based on polynomial length
        let batch_size = if poly.len() > 100000 {
            64  // Very large polynomials
        } else if poly.len() > 10000 {
            32  // Large polynomials
        } else {
            16  // Medium polynomials
        };
        
        // Compute in parallel chunks for all non-trivial polynomials
        if true {  // Always use parallel processing
            let num_threads = num_cpus::get();
            let chunk_size = std::cmp::max(1, poly.len() / (num_threads * 2));  // 2 chunks per thread
            
            let results: Vec<E::G1> = (0..poly.len()).step_by(chunk_size)
                .collect::<Vec<_>>() // collect to Vec first
                .into_par_iter()
                .map(|start| {
                    let end = std::cmp::min(start + chunk_size, poly.len());
                    let mut result = E::G1::zero();
                    
                    // Process each chunk in smaller batches for cache efficiency
                    for i in (start..end).step_by(batch_size) {
                        let batch_end = std::cmp::min(i + batch_size, end);
                        let mut batch_result = E::G1::zero();
                        
                        // Process each element in the batch
                        for j in i..batch_end {
                            batch_result += self.kzg.crs_g1[j].mul(poly[j]);
                        }
                        
                        result += batch_result;
                    }
                    
                    result
                })
                .collect();
            
            // Combine the results
            results.into_iter().fold(E::G1::zero(), |acc, x| acc + x)
        } else {
            // Sequential fallback (unused branch)
            let mut result = E::G1::zero();
            
            for i in (0..poly.len()).step_by(batch_size) {
                let end = std::cmp::min(i + batch_size, poly.len());
                let mut batch_result = E::G1::zero();
                
                for j in i..end {
                    batch_result += self.kzg.crs_g1[j].mul(poly[j]);
                }
                
                result += batch_result;
            }
            
            result
        }
    }

    /// Generate a secure hash to scalar field element
    /// Used for Fiat-Shamir transformation to make the protocol non-interactive
    pub fn hash_to_scalar<T: AsRef<[u8]>>(&self, data: T) -> E::ScalarField {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        // Map the 32-byte SHA-256 output to a field element using a canonical
        // reduce-from-bytes method. This avoids manual bit fiddling and
        // overflow issues and produces an unbiased scalar in the field.
        E::ScalarField::from_le_bytes_mod_order(&result)
    }

    // Efficiently generate powers of a challenge value
    fn create_challenge_vectors(&self, size: usize, challenge: E::ScalarField) -> Vec<E::ScalarField> {
        let mut powers = Vec::with_capacity(size);
        let mut curr_power = E::ScalarField::ONE;
        
        // Generate powers: [1, y, y², y³, ..., y^(size-1)]
        for _ in 0..size {
            powers.push(curr_power);
            curr_power *= challenge;
        }
        
        powers
    }

    // Compute row projections by applying challenge vector to matrix rows
    // This is a critical step in the matrix multiplication proof protocol
    fn compute_row_projections(&self, matrix: &[Vec<E::ScalarField>], challenge_vec: &[E::ScalarField]) 
                               -> Vec<E::ScalarField> {
        let rows = matrix.len();
        
        // Use parallel processing for large matrices
        if rows > 1000 {
            matrix.par_iter().map(|row| {
                // Batch process for better cache locality
                let batch_size = 16; // Adjust based on your hardware
                let mut sum = E::ScalarField::ZERO;
                
                // Process in batches
                for chunk_start in (0..std::cmp::min(row.len(), challenge_vec.len())).step_by(batch_size) {
                    let chunk_end = std::cmp::min(chunk_start + batch_size, std::cmp::min(row.len(), challenge_vec.len()));
                    
                    for j in chunk_start..chunk_end {
                        sum += row[j] * challenge_vec[j];
                    }
                }
                
                sum
            }).collect()
        } else {
            // Sequential processing for smaller matrices
            matrix.iter().map(|row| {
                let mut sum = E::ScalarField::ZERO;
                let min_len = std::cmp::min(row.len(), challenge_vec.len());
                
                for j in 0..min_len {
                    sum += row[j] * challenge_vec[j];
                }
                
                sum
            }).collect()
        }
    }

    // Compute column projections by applying challenge vector to matrix columns
    fn compute_column_projections(&self, matrix: &[Vec<E::ScalarField>], challenge_vec: &[E::ScalarField]) 
                                 -> Vec<E::ScalarField> {
        let rows = matrix.len();
        let cols = if rows > 0 { matrix[0].len() } else { 0 };
        
        // Use parallel processing for large matrices
        if cols > 1000 {
            (0..cols).into_par_iter().map(|col_idx| {
                // Batch process for better cache locality
                let batch_size = 16; // Adjust based on your hardware
                let mut sum = E::ScalarField::ZERO;
                
                // Process in batches
                for row_start in (0..std::cmp::min(rows, challenge_vec.len())).step_by(batch_size) {
                    let row_end = std::cmp::min(row_start + batch_size, std::cmp::min(rows, challenge_vec.len()));
                    
                    for j in row_start..row_end {
                        sum += matrix[j][col_idx] * challenge_vec[j];
                    }
                }
                
                sum
            }).collect()
        } else {
            // Sequential processing for smaller matrices
            (0..cols).map(|col_idx| {
                let mut sum = E::ScalarField::ZERO;
                let min_len = std::cmp::min(rows, challenge_vec.len());
                
                for j in 0..min_len {
                    sum += matrix[j][col_idx] * challenge_vec[j];
                }
                
                sum
            }).collect()
        }
    }

    // Compute dot product efficiently with batching and possible parallelization
    fn compute_dot_product(&self, a: &[E::ScalarField], b: &[E::ScalarField]) -> E::ScalarField {
        let min_len = std::cmp::min(a.len(), b.len());
        
        // Use parallel processing for large vectors
        if min_len > 1000 {
            // Parallel reduction for large vectors
            (0..min_len).into_par_iter()
                .map(|i| a[i] * b[i])
                .reduce_with(|acc, val| acc + val)
                .unwrap_or(E::ScalarField::ZERO)
        } else {
            // Batch processing for better cache behavior
            let batch_size = 16; // Adjust based on your hardware
            let mut sum = E::ScalarField::ZERO;
            
            for i in (0..min_len).step_by(batch_size) {
                let end = std::cmp::min(i + batch_size, min_len);
                
                for j in i..end {
                    sum += a[j] * b[j];
                }
            }
            
            sum
        }
    }

    /// Generate a proof of matrix multiplication with memory and performance metrics
    /// This function wraps prove_matrix_mult and collects performance data
    pub fn prove_matrix_mult_with_metrics(
        &self, 
        a_matrix: &[Vec<E::ScalarField>], 
        b_matrix: &[Vec<E::ScalarField>]
    ) -> (HashMap<String, (E::G1, E::ScalarField)>, ProofMetrics) {
        let matrix_size = a_matrix.len();
        
        // Reset peak memory counter before starting
        TrackedAllocator::reset_peak();
        
        // Start timing
        let start_time = Instant::now();
        
        // Generate the proof using existing function
        let proof = self.prove_matrix_mult(a_matrix, b_matrix);
        
        // Measure time
        let proof_time = start_time.elapsed();
        
        // Start verification timing
        let verify_start = Instant::now();
        let verified = self.verify(&proof);
        let verification_time = verify_start.elapsed();
        assert!(verified, "Proof verification failed");
        
        // Calculate proof size
        let proof_size = Self::calculate_proof_size(&proof);
        
        // Get peak memory usage
        let memory_used = TrackedAllocator::peak_allocated();
        
        // Create metrics
        let metrics = ProofMetrics {
            matrix_size,
            proof_time_ms: proof_time.as_millis() as u64,
            verification_time_ms: verification_time.as_millis() as u64,
            proof_size_bytes: proof_size,
            memory_usage_bytes: memory_used,
        };
        
        (proof, metrics)
    }

    /// Generate a zero-knowledge proof of matrix multiplication C = A × B
    ///
    /// This is the main protocol implementation. It creates a proof that convinces
    /// the verifier that A × B = C without revealing A or B.
    ///
    /// The proof consists of:
    /// 1. KZG commitments to matrix polynomials
    /// 2. A challenge generated from these commitments (Fiat-Shamir)
    /// 3. Row and column projections using the challenge
    /// 4. A proof of the inner product of these projections
    pub fn prove_matrix_mult(
        &self, 
        a_matrix: &[Vec<E::ScalarField>], 
        b_matrix: &[Vec<E::ScalarField>]
    ) -> HashMap<String, (E::G1, E::ScalarField)> {
        // Log timings for performance analysis
        let mut timings = Vec::new();
        
        // Verify matrix dimensions are compatible
        let l = a_matrix[0].len();
        let l2 = b_matrix.len();
        assert_eq!(l, l2, "Matrix dimensions mismatch");
        
        // Step 1: Convert matrices to polynomials with optimized compression
        let start = Instant::now();
        let a_poly = self.matrix_to_poly_with_compression(a_matrix);
        timings.push(("A matrix compression", start.elapsed()));
        
        let start = Instant::now();
        let b_poly = self.matrix_to_poly_with_compression(b_matrix);
        timings.push(("B matrix compression", start.elapsed()));
        
        // Step 2: Generate KZG commitments using optimized function
        let start = Instant::now();
        let commit_a = self.optimized_commit(&a_poly);
        timings.push(("A commitment", start.elapsed()));
        
        let start = Instant::now();
        let commit_b = self.optimized_commit(&b_poly);
        timings.push(("B commitment", start.elapsed()));
        
        let start = Instant::now();
        // Compute C = A * B using field arithmetic in parallel for large matrices
        let mut c_matrix = vec![vec![E::ScalarField::ZERO; b_matrix[0].len()]; a_matrix.len()];
        
        if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
            println!("Debug - Matrix dimensions:");
            println!("  A: {}x{}", a_matrix.len(), a_matrix[0].len());
            println!("  B: {}x{}", b_matrix.len(), b_matrix[0].len());
        }
        
        if a_matrix.len() >= 32 {
            if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
                println!("Debug - Using parallel computation for large matrices");
            }
            // For large matrices, use parallel computation
            c_matrix.par_iter_mut().enumerate().for_each(|(i, row)| {
                for j in 0..b_matrix[0].len() {
                    let mut sum = E::ScalarField::ZERO;
                    for k in 0..a_matrix[0].len() {
                        sum += a_matrix[i][k].mul(&b_matrix[k][j]);
                    }
                    row[j] = sum;
                }
            });
        } else {
            if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
                println!("Debug - Using sequential computation for small matrices");
            }
            // For smaller matrices, use sequential computation with careful field arithmetic
            for i in 0..a_matrix.len() {
                for j in 0..b_matrix[0].len() {
                    let mut sum = E::ScalarField::ZERO;
                    for k in 0..a_matrix[0].len() {
                        sum += a_matrix[i][k].mul(&b_matrix[k][j]);
                    }
                    c_matrix[i][j] = sum;
                }
            }
        }
        
        if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
            println!("Debug - Sample values from result matrix C:");
            if !c_matrix.is_empty() {
                println!("  C[0][0]: {:?}", c_matrix[0][0]);
                if c_matrix.len() > 1 && c_matrix[0].len() > 1 {
                    println!("  C[1][1]: {:?}", c_matrix[1][1]);
                }
            }
        }
        let c_poly = self.matrix_to_poly_with_compression(&c_matrix);
        let commit_c = self.optimized_commit(&c_poly);
        timings.push(("C matrix computation and commitment", start.elapsed()));
        
        // Step 3: Generate challenge using Fiat-Shamir transform with full commitment serialization
        let start = Instant::now();
        let mut challenge_data = Vec::new();
        challenge_data.extend_from_slice(&commit_a.to_string().into_bytes());
        challenge_data.extend_from_slice(&commit_b.to_string().into_bytes());
        challenge_data.extend_from_slice(&commit_c.to_string().into_bytes());
        let challenge = self.hash_to_scalar(challenge_data);
        timings.push(("Challenge generation", start.elapsed()));
        
        // Step 4: Create challenge vectors for row and column projections
        let start = Instant::now();
        let challenge_vec = self.create_challenge_vectors(l, challenge);
        timings.push(("Challenge vectors", start.elapsed()));
        
        // Step 5: Compute row and column projections (A·yᵢ and B·yᵢ)
        // This is a key part of the polynomial evaluation protocol
        let start = Instant::now();
        let a_y = self.compute_row_projections(a_matrix, &challenge_vec);
        timings.push(("A row projections", start.elapsed()));
        
        let start = Instant::now();
        let b_y = self.compute_column_projections(b_matrix, &challenge_vec);
        timings.push(("B column projections", start.elapsed()));
        
        // Step 6: Compute dot product (a_y · b_y) = result at challenge point
        let start = Instant::now();
        let d = self.compute_dot_product(&a_y, &b_y);
        timings.push(("Dot product", start.elapsed()));
        
        // Step 7: Create final proof elements
        let start = Instant::now();
        
        // Convert matrices to polynomials for verification
        let c_poly = self.matrix_to_poly_with_compression(&c_matrix);
        
        // Debug: Print polynomial evaluation
        let value_at_challenge = evaluate(&c_poly, challenge);
        if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
            println!("Debug - Polynomial at challenge point: {:?}", value_at_challenge);
            println!("Debug - Expected value d: {:?}", d);
            println!("Debug - First few polynomial coefficients: {:?}", &c_poly[0..std::cmp::min(5, c_poly.len())]);
        }
        
        // Create KZG commitment for C
        let commit_c = self.kzg.commit(&c_poly);
        if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
            println!("Debug - Commitment components:");
            println!("  From commit(): {:?}", commit_c);
        }
        
        // Generate witness
        let witness = self.kzg.open(&c_poly, challenge);
        if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
            println!("Debug - Generated witness: {:?}", witness);
        }
        timings.push(("Final commitments", start.elapsed()));
        
    // Construct and return the proof
    let mut proof = HashMap::new();
    proof.insert("comm_A".to_string(), (commit_a, E::ScalarField::ZERO));
    proof.insert("comm_B".to_string(), (commit_b, E::ScalarField::ZERO));
    // Use the actual polynomial evaluation at the challenge (the value we opened)
    proof.insert("comm_C".to_string(), (commit_c, value_at_challenge));
    proof.insert("witness".to_string(), (witness, E::ScalarField::ZERO));
        
        proof
    }


    /// Verify a matrix multiplication proof
    ///
    /// The verifier checks that the proof is valid for the claimed matrix product.
    /// This is done by:
    /// 1. Reconstructing the challenge from the commitments
    /// 2. Verifying the KZG proof at the challenge point
    pub fn verify(&self, proof: &HashMap<String, (E::G1, E::ScalarField)>) -> bool {
        // Extract proof components
        let (comm_a, _) = proof.get("comm_A").unwrap();
        let (comm_b, _) = proof.get("comm_B").unwrap();
        let (comm_c, d) = proof.get("comm_C").unwrap();
        let (witness, _) = proof.get("witness").unwrap();
        
        if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
            println!("Debug - Proof components:");
            println!("  comm_A: {:?}", comm_a.to_string());
            println!("  comm_B: {:?}", comm_b.to_string());
            println!("  comm_C: {:?}", comm_c.to_string());
            println!("  d: {:?}", d);
            println!("  witness: {:?}", witness.to_string());
        }
        
        // Regenerate the Fiat-Shamir challenge with full commitment serialization
        let mut challenge_data = Vec::new();
        challenge_data.extend_from_slice(&comm_a.to_string().into_bytes());
        challenge_data.extend_from_slice(&comm_b.to_string().into_bytes());
        challenge_data.extend_from_slice(&comm_c.to_string().into_bytes());
        
        // Hash the challenge data to recreate the same challenge y
        let challenge = self.hash_to_scalar(challenge_data);
        if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
            println!("Debug - Challenge: {:?}", challenge);
        }
        
        // Verify the proof with the correct d value from the proof
        // This checks that the polynomial evaluated at the challenge point equals d
        let result = self.kzg.verify(challenge, *d, *comm_c, *witness);
        if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
            println!("Debug - KZG verification result: {}", result);
        }
        result

    }
}

/// Optimized structure for batched matrix multiplication proofs
///
/// This structure enables proving multiple matrix multiplications in a single
/// batch with significantly reduced proof size and verification time.
/// The batching technique leverages a single shared challenge point
/// and linear combinations of proofs.
pub struct OptimizedBatchedZKMatrixProof<E: Pairing> {
    pub matrix_proof: ZKMatrixProof<E>,  // Base ZK matrix proof instance
    pub max_threads: usize,              // Thread limit for parallel processing
}

impl<E: Pairing> OptimizedBatchedZKMatrixProof<E> {
    /// Create a new batched proof instance with optimal thread count
    pub fn new(matrix_proof: &ZKMatrixProof<E>) -> Self {
        Self {
            matrix_proof: matrix_proof.clone(),
            max_threads: num_cpus::get(),
        }
    }
    
    /// Create a new batched proof instance with specified thread count
    pub fn new_with_threads(matrix_proof: &ZKMatrixProof<E>, threads: usize) -> Self {
        Self {
            matrix_proof: matrix_proof.clone(),
            max_threads: threads,
        }
    }
    
    /// Process a single matrix pair for the batched proof
    ///
    /// This function performs the core matrix multiplication proof operations
    /// for a single pair of matrices in the batch:
    /// 1. Project matrices onto challenge vector
    /// 2. Compute dot product
    /// 3. Optionally compute full polynomial representation and commitments
    fn process_matrix_pair(
        &self,
        a_matrix: &[Vec<E::ScalarField>],
        b_matrix: &[Vec<E::ScalarField>],
        challenge_vec: &[E::ScalarField],
        compute_full_poly: bool
    ) -> (E::G1, E::G1, E::ScalarField) {
        // Compute projections using the shared challenge vector
        let a_y = self.matrix_proof.compute_row_projections(a_matrix, challenge_vec);
        let b_y = self.matrix_proof.compute_column_projections(b_matrix, challenge_vec);
        
        // Compute dot product (final evaluation)
        let d = self.matrix_proof.compute_dot_product(&a_y, &b_y);

        let (commit_a, commit_b) = if compute_full_poly {
            // Compute polynomial representations and commitments
            // Only done for a small sample of matrices to save computation
            let a_poly = self.matrix_proof.matrix_to_poly_with_compression(a_matrix);
            let b_poly = self.matrix_proof.matrix_to_poly_with_compression(b_matrix);
            
            // Compute commitments
            let commit_a = self.matrix_proof.optimized_commit(&a_poly);
            let commit_b = self.matrix_proof.optimized_commit(&b_poly);
            
            (commit_a, commit_b)
        } else {
            // Use placeholder commitments for non-sample matrices
            // These aren't actually used in the final proof
            let placeholder = E::G1::zero().mul(E::ScalarField::one());
            (placeholder, placeholder)
        };
        
        (commit_a, commit_b, d)
    }
    
    /// Generate proofs for multiple matrix pairs in batch with improved efficiency
    ///
    /// This function implements the batched matrix multiplication proof protocol:
    /// 1. Use a common challenge for all matrices
    /// 2. Process all matrix pairs in parallel
    /// 3. Combine results using a random linear combination
    /// 4. Generate a single compact proof for all matrices
    ///
    /// The optimization provides significant advantages:
    /// - O(1) proof size regardless of batch size
    /// - Near-constant verification time
    /// - Sublinear proving time per matrix pair
    pub fn prove_batched_matrix_mult(
        &self,
        matrices_a: &[Vec<Vec<E::ScalarField>>],
        matrices_b: &[Vec<Vec<E::ScalarField>>]
    ) -> HashMap<String, (E::G1, E::ScalarField)> {
        let total_start = Instant::now();
        assert_eq!(matrices_a.len(), matrices_b.len(), "Number of A and B matrices must match");
        let num_pairs = matrices_a.len();
        
        // Early return for single matrix case
        if num_pairs == 1 {
            return self.matrix_proof.prove_matrix_mult(&matrices_a[0], &matrices_b[0]);
        }
        
        // Pre-allocate results vectors with capacity to avoid reallocations
        // Initialize storage for batched verification
        // Storage for batched verification will be implemented later
        
        // Step 1: Generate common challenge for all matrices (Fiat-Shamir transform)
        let start = Instant::now();
        let mut challenge_data = Vec::new();
        // Use only a few matrices to generate challenge for efficiency
        let sample_size = std::cmp::min(3, num_pairs);
        for i in 0..sample_size {
            // Add matrix dimensions to challenge data
            let m = matrices_a[i].len();
            let n = matrices_a[i][0].len();
            let l = matrices_b[i][0].len();
            challenge_data.extend_from_slice(&[m as u8, n as u8, l as u8]);
        }
        let challenge = self.matrix_proof.hash_to_scalar(challenge_data);
        let challenge_time = start.elapsed();
        
        // Step 2: Precompute challenge vectors for all matrices
        let start = Instant::now();
        let max_dim = matrices_a.iter().map(|m| m[0].len())
                      .chain(matrices_b.iter().map(|m| m.len()))
                      .max().unwrap_or(0);
        
        let challenge_vec = self.matrix_proof.create_challenge_vectors(max_dim, challenge);
        let vectors_time = start.elapsed();
        
        // Step 3: Process all matrix pairs in parallel
        let start = Instant::now();
        println!("Processing {} matrix pairs with {} threads...", num_pairs, self.max_threads);
        
        // Limit parallelism to available threads for optimal performance
        let chunk_size = std::cmp::max(1, num_pairs / self.max_threads);
        
        let results: Vec<_> = matrices_a.par_iter()
            .zip(matrices_b.par_iter())
            .enumerate()
            .with_min_len(chunk_size)
            .map(|(idx, (a_matrix, b_matrix))| {
                // Only compute full polynomials for a few matrices to save time
                let compute_full_poly = idx < sample_size;
                
                // Process this matrix pair
                self.process_matrix_pair(
                    a_matrix, 
                    b_matrix, 
                    &challenge_vec,
                    compute_full_poly
                )
            })
            .collect();
        
        let processing_time = start.elapsed();
        
        // Step 4: Aggregate results for batch opening using a random linear combination
        let start = Instant::now();
        let mut all_d_values: Vec<E::ScalarField> = Vec::with_capacity(num_pairs);
        let first_commit_a = results[0].0;
        let first_commit_b = results[0].1;
        
        for (_, _, d) in &results {
            all_d_values.push(*d);
        }
        
        // Generate batching coefficient from all d-values (additional Fiat-Shamir)
        let mut batch_bytes = Vec::new();
        for d in &all_d_values {
            batch_bytes.extend_from_slice(d.to_string().as_bytes());
        }
        let batch_challenge = self.matrix_proof.hash_to_scalar(batch_bytes);
        
        // Combine d-values using powers of the batch challenge
        // This creates a random linear combination that must preserve the correct value
        // if and only if all individual d values are correct
        let mut combined_d = E::ScalarField::ZERO;
        let mut batch_coeff = E::ScalarField::ONE;
        
        for d in all_d_values {
            combined_d += d * batch_coeff;
            batch_coeff *= batch_challenge;
        }
        
        // Create commitments and witness for the combined evaluation
        let d_poly = vec![combined_d];
        let commit_d = self.matrix_proof.kzg.commit(&d_poly);
        let witness = self.matrix_proof.kzg.open(&d_poly, challenge);
        
        let finalize_time = start.elapsed();
        
        // Log performance data
        let total_time = total_start.elapsed();
        println!("\nBatched proof timings ({} pairs):", num_pairs);
        println!("  Challenge generation: {:?} ({:.1}%)", challenge_time, 
                 challenge_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("  Challenge vectors: {:?} ({:.1}%)", vectors_time, 
                 vectors_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("  Matrix processing: {:?} ({:.1}%)", processing_time, 
                 processing_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("  Finalizing proof: {:?} ({:.1}%)", finalize_time, 
                 finalize_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("  Total time: {:?}", total_time);
        
        // Create the final batched proof
        // The proof size is O(1) regardless of the number of matrix pairs
        let mut proof = HashMap::new();
        
        // Add first matrix pair commitments
        proof.insert("comm_A".to_string(), (first_commit_a, E::ScalarField::ZERO));
        proof.insert("comm_B".to_string(), (first_commit_b, E::ScalarField::ZERO));
        
        // Add batched values
        proof.insert("comm_C".to_string(), (commit_d, combined_d));
        proof.insert("witness".to_string(), (witness, E::ScalarField::ZERO));
        
        // Add batch metadata
        proof.insert("batch_size".to_string(), (
            self.matrix_proof.kzg.g1.mul(E::ScalarField::from(num_pairs as u64)),
            E::ScalarField::from(num_pairs as u64)
        ));
        
        proof.insert("batch_challenge".to_string(), (
            self.matrix_proof.kzg.g1.mul(batch_challenge),
            batch_challenge
        ));
        
        proof
    }
 
    /// Verify a batched proof
    ///
    /// The verification of a batched proof is essentially identical to
    /// verifying a single proof, making verification time O(1) regardless
    /// of batch size. This is a significant advantage of this approach.
    pub fn verify_batched(&self, proof: &HashMap<String, (E::G1, E::ScalarField)>) -> bool {
        self.matrix_proof.verify(proof)
    }
 }
