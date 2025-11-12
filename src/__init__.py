"""
RIV Protocol Implementation

Core modules for Retroactive Intermediate Value Verification.
"""

from .riv_protocol import RIVProtocol
from .sic_commitments import SICManager, SICVerifier
from .challenge import ChallengeGenerator
from .verification import RIVVerifier, compute_detection_probability
from .crypto_backend import CryptoBackend, HashCommitment

__version__ = "1.0.0"

__all__ = [
    'RIVProtocol',
    'SICManager',
    'SICVerifier',
    'ChallengeGenerator',
    'RIVVerifier',
    'CryptoBackend',
    'HashCommitment',
    'compute_detection_probability',
]
