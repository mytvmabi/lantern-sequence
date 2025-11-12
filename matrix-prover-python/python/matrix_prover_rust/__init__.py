"""Matrix Prover Python Bindings

Zero-knowledge proofs for matrix multiplication using the BLS12-381 curve.

Example:
    >>> import numpy as np
    >>> from matrix_prover_rust import MatrixProver
    >>> 
    >>> prover = MatrixProver(degree=4096)
    >>> A = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> B = np.array([[5.0, 6.0], [7.0, 8.0]])
    >>> 
    >>> proof = prover.prove_matrix_mult(A, B)
    >>> print(f"Result shape: {proof['result'].shape}")
    >>> print(f"Verified: {proof['verified']}")
"""

from .matrix_prover_rust import MatrixProver

__version__ = "0.1.0"
__all__ = ["MatrixProver"]
