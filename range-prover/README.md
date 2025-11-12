# Range/Exponentiation Prover: Zero-Knowledge Proofs for Exponentiation and Range Constraints

A cryptographic library providing zero-knowledge proofs for exponentiation statements and range constraints, built on KZG polynomial commitments over the BLS12-381 curve.

## Features

- **Constant-size proofs**: 384 bytes per proof (independent of exponent bit length)
- **O(1) verification**: Single pairing operation for verification
- **Built-in range proofs**: Prove values lie in [0, 2^n) without revealing the value
- **128-bit security**: Based on discrete logarithm hardness on BLS12-381
- **Batch operations**: Prove multiple exponentiations efficiently

## Technical Overview

The system proves exponentiation statements of the form:
```
y = g^x
```

where `x` is a secret exponent and `y` is public, without revealing `x`.

### Core Protocol

1. **Setup**: Generate structured reference string (SRS) with KZG parameters
2. **Commit**: Create polynomial commitments to exponent bits
3. **Prove**: Generate zero-knowledge proof satisfying:
   - Binary constraint: Each bit is 0 or 1
   - Exponentiation correctness: y = g^(sum of bit contributions)
4. **Verify**: Check proof using pairing-based verification (constant time)

### Security Properties

- **Completeness**: Honest proofs always verify
- **Soundness**: Cheating proofs detected except with negligible probability
- **Zero-knowledge**: Proof reveals nothing beyond the exponentiation correctness

## Building

This is a Rust library that can be used standalone or via Python bindings.

### Standalone Rust

```bash
cd range-prover
cargo build --release
cargo test --release
```

### Python Bindings

```bash
cd range-prover-python
pip install maturin
maturin develop --release
```

## Usage

### Python Example

```python
from range_prover_rust import RangeProver

# Initialize prover with max polynomial degree
prover = RangeProver(degree=4096, verbose=False)

# Prove exponentiation: 2^5 = 32
result = prover.compute_exponentiation(
    base=2.0,
    exponent_bits=[True, False, True, False, False]  # Binary: 10100 = 5
)
print(f"2^5 = {result}")  # Output: 32.0

# Prove value is in range [0, 256)
range_proof = prover.prove_range(
    value=42,
    num_bits=8  # 8 bits allows range [0, 256)
)

print(f"Range proof verified: {range_proof['verified']}")
print(f"Proof size: {range_proof['proof_size_bytes']} bytes")
```

### Range Proof Example

Range proofs leverage the binary constraint enforcement to prove a value lies within a specified range:

```python
# Prove value 127 is in [0, 256)
prover = RangeProver(degree=256)
proof = prover.prove_range(value=127, num_bits=8)

# The proof guarantees:
# 1. Value decomposes into 8 binary bits
# 2. Each bit is 0 or 1 (enforced cryptographically)
# 3. Reconstructed value equals claimed value
# 4. Therefore: 0 ≤ value < 2^8 = 256
```

## Performance

Typical performance on modern hardware (Intel i7, 3.0GHz):

| Exponent Bits | Prove Time | Verify Time | Proof Size |
|---------------|-----------|-------------|------------|
| 8 bits        | ~2ms      | ~4ms        | 384 bytes  |
| 32 bits       | ~6ms      | ~4ms        | 384 bytes  |
| 64 bits       | ~10ms     | ~4ms        | 384 bytes  |
| 256 bits      | ~35ms     | ~4ms        | 384 bytes  |

*Note: Verification time remains constant regardless of exponent size (O(1) property)*

## Range Proof Details

Range proofs use the exponentiation proof system internally:

- **Binary decomposition**: Value v decomposed into bits: v = Σ(b_i · 2^i)
- **Bit constraint**: Each b_i proven to be 0 or 1 cryptographically
- **Range guarantee**: If all bits valid, then 0 ≤ v < 2^n

**Applications:**
- Proving training hyperparameters are in valid ranges
- Verifying quantized neural network weights
- Constraining gradient magnitudes in federated learning
- Proving dataset sizes without revealing exact counts

## Implementation Details

- **Curve**: BLS12-381 pairing-friendly curve
- **Commitment scheme**: KZG polynomial commitments
- **Field arithmetic**: Uses arkworks cryptography libraries
- **Parallelization**: Rayon for multi-core proof generation
- **Binary constraints**: Enforced via polynomial constraints on bit values

## Dependencies

```toml
ark-ff = "0.4.0"
ark-ec = "0.4.0"
ark-bls12-381 = "0.4.0"
ark-serialize = "0.4.2"
rand = "0.8.5"
rayon = "1.8.0"
```

## Testing

```bash
# Run unit tests
cargo test --release

# Run range proof tests
cargo test --release range_proof

# Run benchmarks
cargo bench
```

## Security Considerations

- **Trusted Setup**: KZG requires a one-time trusted setup ceremony. For production use, use universal setup parameters or multi-party computation for setup generation.
- **Randomness**: Proof generation requires secure randomness. Uses OS randomness via `rand` crate.
- **Bit constraints**: Binary enforcement is cryptographically sound under discrete logarithm assumption.

## Limitations

- Fixed maximum polynomial degree (set at initialization)
- Trusted setup requirement (can be mitigated with universal SRS)
- Proof generation scales linearly with bit length (though verification does not)

## Integration with Neural Network Verification

This library is designed for verifying neural network training properties:

- **Gradient bounds**: Prove gradients lie within expected ranges
- **Weight quantization**: Verify quantized weights match constraints
- **Hyperparameter validation**: Prove learning rates, batch sizes are in valid ranges
- **Activation functions**: Verify intermediate activations satisfy bounds

## License

Apache 2.0 - See LICENSE file for details

## References

- KZG Commitments: Kate, Zaverucha, Goldberg (2010)
- BLS12-381 Curve: Barreto, Lynn, Scott pairing-friendly curve
- Binary Constraints: R1CS-style bit decomposition proofs
- Arkworks: Rust ecosystem for zkSNARK development

---

**Note**: This is cryptographic software. Use at your own risk in production environments.
