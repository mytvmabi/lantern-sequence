"""
Challenge Generation Module

Implements Fiat-Shamir heuristic for non-interactive random challenges.
Ensures challenges are unpredictable before commitment phase.
"""

import hashlib
import numpy as np
from typing import List, Dict, Any, Optional


class ChallengeGenerator:
    """
    Generates cryptographic challenges using Fiat-Shamir transform.
    
    The Fiat-Shamir heuristic converts an interactive protocol into
    a non-interactive one by replacing verifier's random challenges
    with deterministic hash-based challenges.
    
    Security relies on collision-resistance of SHA-256 in the
    Random Oracle Model.
    """
    
    def __init__(self, seed: Optional[bytes] = None):
        """
        Initialize challenge generator.
        
        Args:
            seed: Initial seed (typically commitment hash)
        """
        self.transcript = hashlib.sha256()
        
        if seed is not None:
            self.transcript.update(b"CHALLENGE_SEED:")
            self.transcript.update(seed)
    
    def add_commitment(self, commitment: bytes):
        """
        Add commitment to transcript.
        
        This binds the challenge to the commitment, preventing
        the prover from choosing commitments after seeing challenges.
        
        Args:
            commitment: Commitment bytes to include
        """
        self.transcript.update(b"COMMITMENT:")
        self.transcript.update(commitment)
    
    def add_message(self, label: str, data: bytes):
        """
        Add labeled message to transcript.
        
        Args:
            label: Domain separation label
            data: Message data
        """
        self.transcript.update(label.encode())
        self.transcript.update(b":")
        self.transcript.update(data)
    
    def select_challenged_layers(
        self,
        total_layers: int,
        num_challenges: int
    ) -> List[int]:
        """
        Select random layers to challenge.
        
        Uses hash-based selection to choose k layers uniformly
        at random from L total layers.
        
        Args:
            total_layers: Total number of layers (L)
            num_challenges: Number to challenge (k)
        
        Returns:
            Sorted list of challenged layer indices
        """
        # Ensure k ≤ L
        k = min(num_challenges, total_layers)
        
        # Create challenge input
        challenge_input = self.transcript.copy()
        challenge_input.update(b"LAYER_SELECTION:")
        challenge_input.update(str(total_layers).encode())
        challenge_input.update(b":")
        challenge_input.update(str(k).encode())
        
        # Generate deterministic seed from hash
        seed_bytes = challenge_input.digest()
        seed_value = int.from_bytes(seed_bytes[:4], byteorder='big')
        
        # Use numpy random with fixed seed for reproducibility
        rng = np.random.RandomState(seed_value)
        
        # Sample k layers without replacement
        challenged = rng.choice(total_layers, size=k, replace=False)
        
        return sorted(challenged.tolist())
    
    def generate_random_vector(
        self,
        dimension: int,
        layer_idx: int
    ) -> np.ndarray:
        """
        Generate random challenge vector for a layer.
        
        Used for proving linear relations via random linear
        combinations (Schwartz-Zippel lemma).
        
        Args:
            dimension: Vector dimension
            layer_idx: Layer index for domain separation
        
        Returns:
            Random vector of specified dimension
        """
        # Create challenge input with domain separation
        challenge_input = self.transcript.copy()
        challenge_input.update(b"RANDOM_VECTOR:")
        challenge_input.update(str(layer_idx).encode())
        challenge_input.update(b":")
        challenge_input.update(str(dimension).encode())
        
        # Generate seed
        seed_bytes = challenge_input.digest()
        seed_value = int.from_bytes(seed_bytes[:4], byteorder='big')
        
        # Generate random vector
        rng = np.random.RandomState(seed_value)
        vector = rng.randn(dimension)
        
        # Normalize to unit length
        vector = vector / np.linalg.norm(vector)
        
        return vector
    
    def generate_field_element(
        self,
        label: str
    ) -> int:
        """
        Generate random field element.
        
        Args:
            label: Domain separation label
        
        Returns:
            Random integer suitable for field element
        """
        challenge_input = self.transcript.copy()
        challenge_input.update(b"FIELD_ELEMENT:")
        challenge_input.update(label.encode())
        
        # Generate 256-bit random value
        hash_output = challenge_input.digest()
        
        # Convert to integer
        return int.from_bytes(hash_output, byteorder='big')
    
    def finalize(self) -> bytes:
        """
        Finalize transcript and return digest.
        
        Returns:
            Final transcript hash
        """
        return self.transcript.digest()


def hash_to_scalar(data: bytes) -> int:
    """
    Hash arbitrary data to scalar value.
    
    Args:
        data: Input data
    
    Returns:
        Integer derived from hash
    """
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h, byteorder='big')


def generate_random_vector(dimension: int, seed: bytes) -> np.ndarray:
    """
    Generate deterministic random vector from seed.
    
    Args:
        dimension: Vector dimension
        seed: Random seed
    
    Returns:
        Random vector
    """
    seed_int = int.from_bytes(seed[:4], byteorder='big') % (2**32)
    rng = np.random.RandomState(seed_int)
    
    vector = rng.randn(dimension)
    
    # Normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector
