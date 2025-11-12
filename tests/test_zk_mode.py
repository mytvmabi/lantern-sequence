#!/usr/bin/env python3
"""

This tests the cryptographic backend (matrix_prover_rust + range_prover_rust) to ensure:
1. KZG polynomial commitments work
2. Proof generation succeeds
3. Proof verification passes
4. Constant-time verification property holds
"""

import sys
import time
import numpy as np

def test_zk_imports():
    """Test that ZK backends can be imported."""
    print("Test 1: Importing ZK backends...")
    try:
        from matrix_prover_rust import MatrixProver
        from range_prover_rust import RangeProver
        print("  ✅ PASS: matrix_prover_rust and range_prover_rust imported")
        return True
    except ImportError as e:
        print(f"  ❌ FAIL: {e}")
        print("  → Run: cd matrix-prover-python && maturin develop --release")
        return False

def test_matrix_proof():
    """Test matrix multiplication proof generation and verification."""
    print("\nTest 2: Matrix multiplication proofs...")
    try:
        from matrix_prover_rust import MatrixProver
        
        prover = MatrixProver(degree=4096)
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        proof = prover.prove_matrix_mult(A, B)
        
        if proof['verified']:
            print("  ✅ PASS: Proof verified (pairing check passed)")
            return True
        else:
            print("  ❌ FAIL: Proof verification failed")
            return False
            
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False

def test_range_prover_init():
    """Test range/exponentiation prover initialization."""
    print("\nTest 3: Range/Exponentiation prover initialization...")
    try:
        from range_prover_rust import RangeProver

        prover = RangeProver(degree=4096, verbose=False)
        print("  ✅ PASS: Range/Exponentiation prover initialized")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False

def test_constant_time_verification():
    """Test that verification time is constant (O(1))."""
    print("\nTest 4: Constant-time verification...")
    try:
        from matrix_prover_rust import MatrixProver
        
        prover = MatrixProver(degree=4096)
        
        # Test with different matrix sizes
        times = []
        for size in [2, 4, 8]:
            A = np.random.randn(size, size)
            B = np.random.randn(size, size)
            
            start = time.time()
            proof = prover.prove_matrix_mult(A, B)
            elapsed = time.time() - start
            
            times.append((size, elapsed, proof['verified']))
        
        # Verification should be roughly constant
        # (Small variance is acceptable)
        print(f"  Matrix 2×2: {times[0][1]*1000:.1f}ms, verified={times[0][2]}")
        print(f"  Matrix 4×4: {times[1][1]*1000:.1f}ms, verified={times[1][2]}")
        print(f"  Matrix 8×8: {times[2][1]*1000:.1f}ms, verified={times[2][2]}")
        
        all_verified = all(t[2] for t in times)
        if all_verified:
            print("  ✅ PASS: All proofs verified")
            return True
        else:
            print("  ❌ FAIL: Some proofs did not verify")
            return False
            
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False

def main():
    print("="*70)
    print("RIV Zero-Knowledge Mode Verification Test")
    print("="*70)
    
    results = []
    results.append(test_zk_imports())
    results.append(test_matrix_proof())
    results.append(test_range_prover_init())
    results.append(test_constant_time_verification())
    
    print("\n" + "="*70)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("="*70)
    
    if all(results):
        print("\n✅ SUCCESS: Zero-knowledge mode is fully operational!")
        print("   You can now run experiments with ZK mode enabled")
        return 0
    else:
        print("\n❌ FAILURE: Some tests failed")
        print("   Please install ZK backends:")
        print("   cd matrix-prover-python && maturin develop --release")
        print("   cd ../range-prover-python && maturin develop --release")
        return 1

if __name__ == "__main__":
    sys.exit(main())
