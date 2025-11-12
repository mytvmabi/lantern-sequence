"""
Cryptographic Backend Wrapper

Provides Python interface to Rust-based KZG polynomial commitments
and range proofs. This module abstracts the FFI layer and provides
a clean API for the RIV protocol implementation.
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional
import struct

try:
    from crypto_backend_rust import (
        KZGCommitment,
        PolynomialProver,
        RangeProver
    )
    RUST_BACKEND_AVAILABLE = True
except ImportError:
    RUST_BACKEND_AVAILABLE = False
    import warnings
    warnings.warn(
        "Rust cryptographic backend not available. "
        "Run 'cd crypto_backend && bash build.sh' to compile."
    )


class CryptoBackend:
    """
    High-level interface to cryptographic primitives.
    
    This class provides polynomial commitments and range proofs
    using KZG commitments on BLS12-381 curve.
    
    Security: 128-bit security level under q-SDH assumption.
    """
    
    def __init__(self, degree: int = 4096, verbose: bool = False):
        """
        Initialize cryptographic backend with trusted setup.
        
        Args:
            degree: Maximum polynomial degree (default 4096)
            verbose: Enable detailed logging
        
        Note:
            The trusted setup uses a structured reference string (SRS)
            derived from a multi-party computation ceremony. The setup
            phase takes approximately 200ms for degree 4096.
        """
        if not RUST_BACKEND_AVAILABLE:
            raise RuntimeError(
                "Rust backend not compiled. "
                "Please run setup.sh or manually build crypto_backend."
            )
        
        self.degree = degree
        self.verbose = verbose
        
        # Initialize polynomial commitment system
        self.polynomial_prover = PolynomialProver(degree=degree)
        
        # Initialize range proof system
        self.range_prover = RangeProver(degree=min(degree, 256))
        
        if verbose:
            print(f"Crypto backend initialized: degree={degree}, security=128-bit")
    
    def commit_polynomial(self, coefficients: np.ndarray) -> bytes:
        """
        Create KZG commitment to polynomial.
        
        Args:
            coefficients: Polynomial coefficients (constant term first)
        
        Returns:
            Commitment (48 bytes on BLS12-381)
        
        Raises:
            ValueError: If degree exceeds maximum
        """
        if len(coefficients) > self.degree:
            raise ValueError(
                f"Polynomial degree {len(coefficients)} exceeds maximum {self.degree}"
            )
        
        # Convert to field elements
        coeffs_field = self._numpy_to_field(coefficients)
        
        # Generate commitment
        commitment = self.polynomial_prover.commit(coeffs_field)
        
        return commitment
    
    def create_evaluation_proof(
        self,
        coefficients: np.ndarray,
        point: float
    ) -> Dict[str, Any]:
        """
        Create proof that polynomial evaluates to specific value at point.
        
        This implements the KZG opening protocol:
        Given commitment C to polynomial p(X), prove p(z) = v
        
        Args:
            coefficients: Polynomial coefficients
            point: Evaluation point z
        
        Returns:
            Dictionary containing:
                - proof: KZG opening proof (48 bytes)
                - value: Evaluation result p(z)
        """
        coeffs_field = self._numpy_to_field(coefficients)
        point_field = self._float_to_field(point)
        
        # Compute evaluation
        value = np.polyval(coefficients[::-1], point)
        
        # Generate opening proof
        proof = self.polynomial_prover.create_proof(coeffs_field, point_field)
        
        return {
            'proof': proof,
            'value': value,
            'point': point
        }
    
    def verify_evaluation(
        self,
        commitment: bytes,
        point: float,
        value: float,
        proof: bytes
    ) -> bool:
        """
        Verify polynomial evaluation proof.
        
        Args:
            commitment: KZG commitment to polynomial
            point: Evaluation point
            value: Claimed evaluation result
            proof: Opening proof
        
        Returns:
            True if proof is valid, False otherwise
        """
        point_field = self._float_to_field(point)
        value_field = self._float_to_field(value)
        
        return self.polynomial_prover.verify(
            commitment,
            point_field,
            value_field,
            proof
        )
    
    def create_range_proof(
        self,
        value: int,
        bit_length: int
    ) -> Dict[str, Any]:
        """
        Create proof that value lies in range [0, 2^bit_length).
        
        This uses binary constraint enforcement: for each bit b_i,
        prove b_i(b_i - 1) = 0 which forces b_i in {0, 1}.
        
        Args:
            value: Integer value to prove range for
            bit_length: Number of bits (range is [0, 2^bit_length))
        
        Returns:
            Dictionary containing:
                - proof: Range proof data
                - commitment: Commitment to bit decomposition
        
        Raises:
            ValueError: If value exceeds range
        """
        if value < 0 or value >= (1 << bit_length):
            raise ValueError(
                f"Value {value} outside range [0, {1 << bit_length})"
            )
        
        # Create range proof
        proof_data = self.range_prover.prove_range(value, bit_length)
        
        return {
            'proof': proof_data['proof'],
            'commitment': proof_data['commitment'],
            'bit_length': bit_length
        }
    
    def verify_range_proof(
        self,
        commitment: bytes,
        bit_length: int,
        proof: bytes
    ) -> bool:
        """
        Verify range proof.
        
        Args:
            commitment: Commitment to bit decomposition
            bit_length: Number of bits in range
            proof: Range proof
        
        Returns:
            True if value is proven to be in [0, 2^bit_length)
        """
        return self.range_prover.verify_range(
            commitment,
            bit_length,
            proof
        )
    
    def batch_commit(self, matrices: Dict[str, np.ndarray]) -> Dict[str, bytes]:
        """
        Create commitments to multiple matrices efficiently.
        
        Args:
            matrices: Dictionary mapping keys to matrices
        
        Returns:
            Dictionary mapping keys to commitments
        """
        commitments = {}
        
        for key, matrix in matrices.items():
            # Flatten matrix to polynomial coefficients
            coeffs = matrix.flatten()
            commitments[key] = self.commit_polynomial(coeffs)
        
        return commitments
    
    def _numpy_to_field(self, arr: np.ndarray) -> list:
        """
        Convert numpy array to finite field elements.
        
        Uses the same scaling approach as the matrix prover for consistency:
        Scale by 1,000,000 to preserve 6 decimal places.
        
        Args:
            arr: Numpy array of floats
        
        Returns:
            List of floats (will be converted to field in Rust)
        """
        # Return as list of floats - Rust backend will handle field conversion
        # using the matrix prover approach (scale by 1M)
        return arr.tolist()
    
    def _float_to_field(self, value: float) -> float:
        """
        Convert single float to field element.
        
        Args:
            value: Float value
        
        Returns:
            Float (will be converted to field in Rust)
        """
        # Return as float - Rust backend will handle field conversion
        return float(value)


class HashCommitment:
    """
    Fallback commitment scheme using cryptographic hashing.
    
    This provides a simple commitment scheme when the full
    cryptographic backend is not available. Uses SHA-256.
    
    Note: This does not provide zero-knowledge properties,
    only binding. Used for transparent verification mode.
    """
    
    @staticmethod
    def commit(data: np.ndarray) -> bytes:
        """
        Create hash commitment to data.
        
        Args:
            data: Numpy array to commit to
        
        Returns:
            SHA-256 hash (32 bytes)
        """
        import hashlib
        
        # Serialize data deterministically
        data_bytes = data.tobytes()
        
        # Hash with domain separation
        hasher = hashlib.sha256()
        hasher.update(b"RIV_COMMITMENT:")
        hasher.update(data_bytes)
        
        return hasher.digest()
    
    @staticmethod
    def verify(commitment: bytes, data: np.ndarray) -> bool:
        """
        Verify hash commitment.
        
        Args:
            commitment: Commitment to verify against
            data: Data to check
        
        Returns:
            True if commitment matches data
        """
        expected = HashCommitment.commit(data)
        return commitment == expected
