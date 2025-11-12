# Anonymity Statement for IEEE S&P Artifact Evaluation

## Purpose

This artifact uses **anonymized names** for the cryptographic components to preserve double-blind review. The implementation names will be finalized in the camera-ready version after acceptance.

## Naming Conventions

### Public-Facing Names (Python Interface)

For artifact evaluation, the cryptographic libraries are exposed through Python with generic names:

- **`matrix_prover_rust`**: Zero-knowledge proofs for matrix multiplication
  - Implements KZG polynomial commitments for proving `C = A × B`
  - Provides constant-size proofs with O(1) verification
  - Used by RIV protocol to verify neural network layer computations

- **`range_prover_rust`**: Zero-knowledge proofs for exponentiation and range constraints
  - Implements proofs for `y = g^x` and range membership
  - Enforces binary constraints on exponent bits
  - Used by RIV protocol for gradient bound verification

### Internal Implementation Names

The Rust implementation uses working names internally that will be updated for publication:

- Internal crate names: `zkMaP` (matrix), `range-prover-lib` (exponentiation/range)
- Internal types: `ZkExpSystem`, `ZKMatrixProof`, etc.
- These are **hidden from reviewers** through the Python API layer

## Justification

### Why Different Names?

1. **Anonymity Preservation**: The internal names reference our prior published work, which could deanonymize the submission
2. **Gradual Refactoring**: The Python interface has been fully anonymized; internal Rust will be updated for camera-ready
3. **Minimal Risk**: Reviewers interact only with the Python API, never directly with Rust internals

### What Reviewers See

When evaluating this artifact, reviewers will:

✅ **See (Public Interface)**:
- Python modules: `matrix_prover_rust`, `range_prover_rust`
- Python classes: `MatrixProver`, `RangeProver`
- Documentation using generic "matrix prover" and "range prover" terminology
- README files without prior work references

❌ **Don't See (Internal Implementation)**:
- Rust crate names (hidden in compiled binaries)
- Internal type names (not exposed through Python FFI)
- Comments in Rust source (unless they inspect .rs files directly)

## Verification

Reviewers can verify anonymization by:

```bash
# Check Python imports (should be anonymized)
grep -r "import.*prover" riv-artifact/src/
grep -r "import.*prover" riv-artifact/tests/

# Check top-level documentation (should be anonymized)
grep -i "zkmap\|zkexp" riv-artifact/README.md
grep -i "zkmap\|zkexp" riv-artifact/matrix-prover/README.md
grep -i "zkmap\|zkexp" riv-artifact/range-prover/README.md

# Should return no matches in public-facing files
```

## Camera-Ready Updates

After acceptance, we will:

1. Rename Rust crates to match final publication names
2. Update all internal types and documentation
3. Provide deanonymized GitHub repository with full history
4. Update Python module names if requested by program committee

## Contact

For questions about naming conventions or anonymization strategy, please contact the program committee through the anonymous submission system.

---

**Artifact Evaluation Committee**: This statement explains our anonymization approach. The technical implementation is complete and functional; naming is the only aspect awaiting finalization.
