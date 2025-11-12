"""Range/Exponentiation Prover Python Bindings

Provides zero-knowledge proofs for:
- Exponentiations (base^exponent = result)
- Range proofs (value ∈ [0, 2^n))

Example:
    >>> import range_prover_rust
    >>> prover = range_prover_rust.RangeProver(degree=256)
    >>> result = prover.compute_exponentiation(2.0, [True, False, True])
    >>> print(f"2^5 = {result}")
    2^5 = 32.0
"""

from .range_prover_rust import RangeProver

__all__ = ['RangeProver']
__version__ = '0.1.0'
