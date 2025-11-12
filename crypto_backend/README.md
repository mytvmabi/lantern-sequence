# Cryptographic Backend

## Overview

This directory contains the Rust implementation of cryptographic primitives for the RIV protocol. The implementation provides:

1. **KZG Polynomial Commitments** on BLS12-381 curve
2. **Range Proofs** using binary constraint enforcement
3. **Python FFI Bindings** via PyO3

## Building from Source

### Prerequisites

- Rust toolchain 1.70+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Python 3.8+ with development headers

### Quick Build

```bash
bash build.sh
```

This script will:
1. Compile the Rust library in release mode
2. Build Python bindings using maturin
3. Install the Python package in development mode

### Manual Build

```bash
# Build Rust library
cargo build --release

# Build Python bindings
cd python
pip install maturin
maturin develop --release
cd ..
```

## Implementation Details

### KZG Commitments (src/kzg.rs)

Implements polynomial commitments with the following properties:

- **Commitment size**: 48 bytes (G1 element on BLS12-381)
- **Proof size**: 48 bytes (constant, independent of polynomial degree)
- **Verification**: Single pairing operation, O(1) time
- **Security**: Binding under q-SDH assumption, 128-bit security level

Key operations:
- `commit(coefficients)`: Create commitment to polynomial
- `create_proof(coefficients, point)`: Generate opening proof at point
- `verify(commitment, point, value, proof)`: Verify opening

### Range Proofs (src/range_proof.rs)

Implements range proofs for SIC using binary constraint enforcement:

For value v in [0, 2^n), decompose as v = Σ b_i * 2^i and prove:
- Each b_i ∈ {0, 1} via constraint b_i(b_i - 1) = 0
- Decomposition is correct via polynomial commitment

Properties:
- **Proof size**: ~384 bytes for 64-bit range
- **Verification**: O(1) time (constant number of pairings)
- **Non-interactive**: Uses Fiat-Shamir transform

### Utilities (src/utils.rs)

Helper functions for:
- Field arithmetic on BLS12-381 scalar field
- Polynomial evaluation and interpolation
- Serialization/deserialization

## Python Interface

The Python bindings provide three main classes:

### KZGCommitment

```python
from crypto_backend_rust import KZGCommitment

# Initialize with degree
kzg = KZGCommitment(degree=4096)

# Commit to polynomial
coeffs = [1.0, 2.0, 3.0, 4.0]
commitment = kzg.commit(coeffs)

# Create opening proof
proof = kzg.create_proof(coeffs, point=2.0)

# Verify
is_valid = kzg.verify(commitment, point=2.0, value=49.0, proof)
```

### PolynomialProver

Higher-level interface for polynomial operations:

```python
from crypto_backend_rust import PolynomialProver

prover = PolynomialProver(degree=4096)
commitment = prover.commit(coefficients)
proof_data = prover.create_proof(coefficients, evaluation_point)
```

### RangeProver

Range proof interface:

```python
from crypto_backend_rust import RangeProver

range_prover = RangeProver(degree=256)

# Prove value in [0, 2^32)
proof = range_prover.prove_range(value=12345, bit_length=32)

# Verify
is_valid = range_prover.verify_range(commitment, bit_length=32, proof)
```

## Performance

On a modern workstation (Intel Xeon, 32 GB RAM):

| Operation | Time |
|-----------|------|
| Trusted setup (degree 4096) | ~200 ms |
| Commitment generation | ~2-5 ms |
| Proof generation | ~3-8 ms |
| Verification | ~10-15 ms |
| Range proof (64-bit) | ~50 ms |
| Range verification | ~15 ms |

## Security Notes

### Trusted Setup

The KZG scheme requires a trusted setup to generate the Structured Reference String (SRS). For production use, this should be generated via multi-party computation (MPC) ceremony.

For this artifact, we use a deterministic setup for reproducibility. This is **ONLY** suitable for research/evaluation purposes.

### Assumptions

Security relies on:
1. **q-SDH assumption**: Computing (g, g^τ, ..., g^{τ^d}) → (g^{1/(τ+c)}, c) is hard
2. **Discrete log**: Computing g^x from g, x is hard in G1/G2
3. **Random Oracle Model**: For Fiat-Shamir transform (SHA-256 as hash function)

### Binding vs Hiding

**Important**: The KZG commitments used here are **binding but NOT hiding**. This means:
- ✅ Prover cannot open commitment to wrong value (binding)
- ❌ Commitment may leak information about polynomial (not hiding)

For zero-knowledge mode, additional randomization is needed (Section 5.3 in paper).

## Source Code Organization

```
crypto_backend/
├── Cargo.toml              # Rust dependencies
├── build.sh                # Build script
├── src/
│   ├── lib.rs              # Library entry point
│   ├── kzg.rs              # KZG commitment implementation
│   ├── range_proof.rs      # Range proof implementation
│   ├── utils.rs            # Utility functions
│   └── python_bindings.rs  # PyO3 Python interface
└── python/
    ├── setup.py            # Python package setup
    └── README.md           # Python usage guide
```

## Dependencies

Core cryptographic libraries (from arkworks ecosystem):
- `ark-ff`: Finite field arithmetic
- `ark-ec`: Elliptic curve operations
- `ark-bls12-381`: BLS12-381 curve implementation
- `ark-serialize`: Serialization utilities

## Testing

Run Rust tests:
```bash
cargo test
```

Run Python tests:
```bash
cd python
pytest tests/
```

## License

This code is released under Apache 2.0 License.

## References

1. Kate, Zaverucha, Goldberg. "Constant-Size Commitments to Polynomials and Their Applications" (KZG, 2010)
2. Boneh, Boyen. "Short Signatures Without Random Oracles" (q-SDH assumption, 2004)
3. Fuchsbauer, Kiltz, Loss. "The Algebraic Group Model and its Applications" (AGM, 2018)
4. Higham. "Accuracy and Stability of Numerical Algorithms" (Backward error analysis, 2002)
