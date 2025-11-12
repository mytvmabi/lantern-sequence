# Matrix Prover: Zero-Knowledge Matrix Multiplication Proofs



A modular zero-knowledge proof system for verifying correct matrix multiplication without revealing input matrices, built on KZG polynomial commitments over the BLS12-381 curve.zkMaP: Zero-Knowledge Succinct Non-Interactive Matrix Multiplication Proofs



## Features

This repository contains an efficient implementation of zero-knowledge proofs for matrix multiplication using KZG polynomial commitments. Our implementation, ZK-MaP (Zero-Knowledge Succinct Non-Interactive Matrix Multiplication Proofs), provides high-performance verification of matrix operations with constant-sized proofs, designed for cryptographic applications.

- **Constant-size proofs**: 48 bytes per commitment (independent of matrix size)

- **O(1) verification**: Two pairing operations regardless of matrix dimensionsOVERVIEW

- **128-bit security**: Based on discrete logarithm hardness on BLS12-381

- **Practical performance**: Proof generation in milliseconds for typical neural network layersZK-MaP enables a prover to demonstrate that C = A × B for matrices A, B, and C without revealing their contents. Built on the KZG commitment scheme, the protocol offers the following advantages:



## Technical Overview- Constant-Sized Proofs: The proof size remains fixed regardless of the matrix dimensions.

- Efficient Verification: Verification time is independent of the matrix size (constant time complexity).

The system proves that for matrices A (m×n), B (n×p), and C (m×p):- Batched Proofs: Supports efficient amortization when proving multiple operations.

```- Non-Interactive Proofs: Achieved via a Fiat-Shamir transformation.

C = A × B

```Cryptographic Foundations:

- KZG Commitments: Ensure binding and correct evaluation under the Polynomial Discrete Logarithm assumption.

using polynomial commitments and random linear checks without revealing A or B.- Fiat-Shamir Transform: Converts interactive proofs into non-interactive ones using a secure hash function.

- Trusted Setup: Although this demo uses a basic setup, production systems should adopt a secure multi-party computation (MPC) ceremony.

### Core Protocol

PERFORMANCE

1. **Setup**: Generate structured reference string (SRS) with KZG parameters

2. **Commit**: Create polynomial commitments to matrix elementsPreliminary benchmarks on a typical CPU (single-threaded) are as follows:

3. **Prove**: Generate zero-knowledge proof of correct multiplication

4. **Verify**: Check proof using pairing-based verification (constant time)Matrix Size | Proof Generation | Verification | Proof Size

-----------|------------------|--------------|---------------

### Security Properties128×128     | 243 ms           | 3.66 ms      | 320 bytes

256×256     | 859 ms           | 3.67 ms      | 320 bytes

- **Completeness**: Honest proofs always verify512×512     | 3.28 s           | 3.70 ms      | 320 bytes

- **Soundness**: Cheating proofs detected except with negligible probability1024×1024   | 12.3 s           | 3.69 ms      | 320 bytes

- **Zero-knowledge**: Proof reveals nothing beyond correctness (under random oracle model)

Verification is orders of magnitude faster than recomputation, demonstrating the efficiency of our approach.

## Building

REQUIREMENTS

This is a Rust library that can be used standalone or via Python bindings.

- Rust (2018 edition or later)

### Standalone Rust- For visualization: Python 3.6+ with matplotlib, pandas, and numpy



```bashINSTALLATION

cd matrix-prover

cargo build --releaseClone the repository:

cargo test --release

```git clone https://github.com/20code-submission-review25/anonzkp-harbor-6092



### Python Bindings

Build the project:

```bash

cd matrix-prover-pythoncargo build --release

pip install maturin

maturin develop --releaseREPOSITORY STRUCTURE

```

- src/

## Usage  - main.rs: Demonstrates core functionality and entry point for running benchmarks

  - kzg.rs: Implements KZG polynomial commitments

### Python Example  - zk_matrix.rs: Encodes matrix polynomial proofs, including proof generation and verification

  - benchmark.rs: Contains the benchmarking suite for performance and scalability tests

```python  - utils.rs: Utility functions for polynomial operations

import numpy as np- plot.py: Script for visualizing benchmark results

from matrix_prover_rust import MatrixProver

USAGE

# Initialize prover with max polynomial degree

prover = MatrixProver(degree=4096)Basic Example:



# Create matricesuse zk_matrix_proofs::{

A = np.array([[1.0, 2.0], [3.0, 4.0]])    kzg::KZG,

B = np.array([[5.0, 6.0], [7.0, 8.0]])    zk_matrix::ZKMatrixProof,

    ark_bls12_381::{Bls12_381, Fr, G1Projective as G1, G2Projective as G2},

# Generate proof    ark_std::UniformRand,

proof = prover.prove_matrix_mult(A, B)};



# Proof contains:// Initialize KZG instance

# - proof['result']: Computed result matrixlet mut rng = ark_std::test_rng();

# - proof['verified']: Boolean verification statuslet degree = 64; // Must be >= max matrix dimension squared

# - proof['num_commitments']: Number of polynomial commitmentslet mut kzg_instance = KZG::<Bls12_381>::new(

    G1::rand(&mut rng),

print(f"Result: {proof['result']}")    G2::rand(&mut rng),

print(f"Verified: {proof['verified']}")    degree

```);



## Performance// Trusted setup

let secret = Fr::rand(&mut rng);

Typical performance on modern hardware (Intel i7, 3.0GHz):kzg_instance.setup(secret);



| Matrix Size | Prove Time | Verify Time | Proof Size |// Create ZKMatrixProof instance

|-------------|-----------|-------------|------------|let zk_matrix = ZKMatrixProof::new(kzg_instance, degree);

| 16×16       | ~2ms      | ~4ms        | 384 bytes  |

| 64×64       | ~8ms      | ~4ms        | 384 bytes  |// Generate random matrices

| 256×256     | ~45ms     | ~4ms        | 384 bytes  |let a_matrix = generate_random_matrix::<Fr>(8, 8, &mut rng);

let b_matrix = generate_random_matrix::<Fr>(8, 8, &mut rng);

*Note: Verification time remains constant regardless of matrix size (O(1) property)*

// Generate proof

## Implementation Detailslet proof = zk_matrix.prove_matrix_mult(&a_matrix, &b_matrix);



- **Curve**: BLS12-381 pairing-friendly curve// Verify proof

- **Commitment scheme**: KZG polynomial commitmentslet result = zk_matrix.verify(&proof);

- **Field arithmetic**: Uses arkworks cryptography librariesassert!(result, "Verification failed");

- **Parallelization**: Rayon for multi-core proof generation

- **Serialization**: Arkworks serialize for proof portabilityBatched Proofs:



## Dependenciesuse zk_matrix_proofs::zk_matrix::OptimizedBatchedZKMatrixProof;



```toml// Create batched instance

ark-ff = "0.4.0"let batched_zk = OptimizedBatchedZKMatrixProof::new(&zk_matrix);

ark-ec = "0.4.0"

ark-bls12-381 = "0.4.0"// Generate multiple matrix pairs

ark-serialize = "0.4.2"let matrices_a = vec![

rand = "0.8.5"    generate_random_matrix::<Fr>(16, 16, &mut rng),

rayon = "1.8.0"    generate_random_matrix::<Fr>(16, 16, &mut rng),

```];

let matrices_b = vec![

## Testing    generate_random_matrix::<Fr>(16, 16, &mut rng),

    generate_random_matrix::<Fr>(16, 16, &mut rng),

```bash];

# Run unit tests

cargo test --release// Generate batched proof

let batched_proof = batched_zk.prove_batched_matrix_mult(&matrices_a, &matrices_b);

# Run integration tests

cargo test --release --test integration_proof_verify// Verify batched proof

```let result = batched_zk.verify_batched(&batched_proof);

assert!(result, "Batch verification failed");

## Security Considerations

BENCHMARKING

- **Trusted Setup**: KZG requires a one-time trusted setup ceremony. For production use, use universal setup parameters or multi-party computation for setup generation.

- **Randomness**: Proof generation requires secure randomness. Uses OS randomness via `rand` crate.Run the benchmarks:

- **Side Channels**: Constant-time field operations help mitigate timing attacks. For high-security environments, additional protections may be needed.

cargo run --release

## Limitations

This will generate CSV files with benchmark results:

- Fixed maximum polynomial degree (set at initialization)- zk_matrix_benchmark.csv: Basic performance metrics

- Trusted setup requirement (can be mitigated with universal SRS)- zk_vs_nonzk_comparison.csv: Comparison with standard matrix multiplication

- Proof generation scales linearly with matrix size (though verification does not)- batch_efficiency.csv: Efficiency of batched proofs

- parallelization_benchmark.csv: Effect of parallelization

## Integration with Neural Network Verification

Visualization:

This library is designed for verifying neural network training computations:

To visualize benchmark results, run the included Python script:

- **Layer updates**: Prove gradient×activation = weight update

- **Batch processing**: Commit to multiple examples in parallel  python3 plot.py

- **Floating-point**: Combines with interval commitment schemes for practical ML

This will generate publication-quality PDF plots:

## License- matrix_performance.pdf: Core performance metrics

- batch_efficiency.pdf: Batch processing efficiency

Apache 2.0 - See LICENSE file for details- parallel_scaling.pdf: Parallelization scaling

- zk_comparison.pdf: Comparison with non-ZK approach

## References



- KZG Commitments: Kate, Zaverucha, Goldberg (2010)

- BLS12-381 Curve: Barreto, Lynn, Scott pairing-friendly curveACKNOWLEDGMENTS

- Arkworks: Rust ecosystem for zkSNARK development

This research builds upon:

---- KZG polynomial commitments (https://www.iacr.org/archive/asiacrypt2010/6477178/6477178.pdf)

- Arkworks libraries (https://github.com/arkworks-rs)

**Note**: This is cryptographic software. Use at your own risk in production environments.
