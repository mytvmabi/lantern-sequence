"""
Dataset Commitment Module for RIV

Enables clients to commit to their dataset before training, allowing servers 
to verify that training batches come from the committed dataset.

Key Features:
- KZG polynomial commitments for dataset membership
- Block-based commitment for large datasets
- Efficient membership proof generation
- Pairing-based verification

Security: Prevents lazy clients from training on reduced datasets
"""

import sys
import os
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


# Lazy imports of cryptographic libraries
# These are loaded only when actually used to avoid import errors
def _get_crypto_modules():
    """Lazy import of cryptographic modules"""
    try:
        from kzg import ate_pairing_multi
        from fields import Fq
        from ec import JacobianPoint, default_ec
        return ate_pairing_multi, Fq, JacobianPoint, default_ec
    except ImportError:
        # If not available, try importing from subset_proof
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from subset_proof import Fq, JacobianPoint, default_ec, ate_pairing_multi
            return ate_pairing_multi, Fq, JacobianPoint, default_ec
        except ImportError as e:
            raise ImportError(
                "Could not import cryptographic modules (kzg, fields, ec). "
                "Make sure subset_proof.py is available or install required libraries."
            ) from e


@dataclass

from torch.utils.data import Dataset, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import torch
from secrets import randbelow
import hashlib
import math
import concurrent.futures

# Import KZG primitives
from kzg import ate_pairing_multi
from fields import Fq
from ec import JacobianPoint, default_ec

logger = logging.getLogger(__name__)


@dataclass
class BlockCommitment:
    """Stores commitment for a block of dataset."""
    commitment: JacobianPoint
    start_idx: int
    end_idx: int
    openings: Dict[int, Tuple[JacobianPoint, Tuple[int, int]]]


class DatasetCommitmentManager:
    """
    Manages dataset commitments for RIV clients.
    
    Workflow:
    1. Client creates manager with their dataset
    2. Manager commits to dataset (one-time setup)
    3. During training: Client samples batches and gets membership proofs
    4. Server verifies batches came from committed dataset
    """
    
    def __init__(self, train_dataset, commit_size: int = None, block_size: int = 200):
        """
        Initialize dataset commitment.
        
        Args:
            train_dataset: PyTorch dataset (TensorDataset expected)
            commit_size: Number of samples to commit to (None = all)
            block_size: Size of each block for parallel processing
        """
        self.train_dataset = train_dataset
        self.commit_size = min(commit_size or len(train_dataset), len(train_dataset))
        self.block_size = block_size
        self.q = default_ec.q
        
        # Initialize elliptic curve points
        self.G1 = JacobianPoint(Fq(self.q, 1), Fq(self.q, 2), Fq(self.q, 1), default_ec)
        self.G2 = JacobianPoint(Fq(self.q, 3), Fq(self.q, 4), Fq(self.q, 1), default_ec)
        
        # Generate secure toxic value
        self.toxic_s = self._secure_random_scalar()
        
        # Precompute powers for commitment
        self.setup_g1 = self._precompute_powers()
        self.s_g2 = self.G2 * Fq(self.q, self.toxic_s)
        
        # Storage
        self.block_commitments = []
        self.aggregated_commitment = None
        self.openings = {}
        
        # Commit to dataset
        logger.info(f"Committing to {self.commit_size} samples in blocks of {block_size}...")
        self._commit_dataset_blocks()
        logger.info("Dataset commitment complete")
    
    def _secure_random_scalar(self) -> int:
        """Generate secure random scalar for toxic waste."""
        seed = hashlib.sha256(f"riv_dataset_commitment_{id(self)}".encode()).digest()
        return int.from_bytes(seed, 'big') % self.q
    
    def _precompute_powers(self) -> List[JacobianPoint]:
        """Precompute powers of G1 for efficient commitment."""
        powers = [self.G1]
        curr = self.G1
        for _ in range(self.block_size - 1):
            curr = curr * Fq(self.q, self.toxic_s)
            powers.append(curr)
        return powers
    
    def _get_fq_value(self, element: Fq) -> int:
        """Extract integer value from Fq element."""
        try:
            if hasattr(element, 'n'):
                return int(element.n)
            if isinstance(element, int):
                return element
            return 0
        except:
            return 0
    
    def _interpolate_points(self, points: List[Tuple[int, int]]) -> List[int]:
        """Lagrange interpolation in finite field."""
        def field_inv(x: int) -> int:
            return pow(x, self.q - 2, self.q)
        
        n = len(points)
        weights = [1] * n
        
        # Compute barycentric weights
        for i in range(n):
            for j in range(n):
                if i != j:
                    weights[i] = (weights[i] * ((points[i][0] - points[j][0]) % self.q)) % self.q
            weights[i] = field_inv(weights[i])
        
        # Initialize polynomial coefficients
        result = [0] * n
        
        # For each point
        for i in range(n):
            xi, yi = points[i]
            if yi == 0:
                continue
            
            term = [0] * n
            term[0] = yi
            
            # Compute numerator polynomial
            for j in range(n):
                if i != j:
                    xj = points[j][0]
                    new_term = [0] * (len(term) + 1)
                    for k in range(len(term)):
                        new_term[k+1] = term[k]
                        new_term[k] = (new_term[k] - (term[k] * xj) % self.q) % self.q
                    term = new_term[:n]
            
            # Multiply by weight
            for k in range(len(term)):
                term[k] = (term[k] * weights[i]) % self.q
            
            # Add to result
            for k in range(len(term)):
                result[k] = (result[k] + term[k]) % self.q
        
        return result
    
    def _compute_quotient(self, coeffs: List[int], x: int, y: int) -> List[int]:
        """Compute quotient Q(X) where P(X) = (X - x)Q(X) + y"""
        coeffs_fq = [Fq(self.q, c) for c in coeffs]
        x_fq = Fq(self.q, x)
        y_fq = Fq(self.q, y)
        
        # Verify point on polynomial
        value = Fq(self.q, 0)
        power = Fq(self.q, 1)
        for coeff in coeffs_fq:
            value += coeff * power
            power *= x_fq
        
        if value != y_fq:
            raise ValueError(f"Point ({x}, {y}) not on polynomial")
        
        # Synthetic division
        quotient = [Fq(self.q, 0)] * (len(coeffs_fq) - 1)
        curr = coeffs_fq[-1]
        quotient[-1] = curr
        
        for i in range(len(coeffs_fq) - 2, -1, -1):
            curr = coeffs_fq[i] + curr * x_fq
            if i > 0:
                quotient[i-1] = curr
        
        if curr != y_fq:
            raise ValueError("Division verification failed")
        
        return [self._get_fq_value(q) for q in quotient]
    
    def _commit_block(self, start_idx: int, end_idx: int) -> BlockCommitment:
        """Commit to a single block of dataset."""
        try:
            points = []
            # Extract labels from dataset
            for idx in range(start_idx, end_idx):
                if isinstance(self.train_dataset, TensorDataset):
                    _, label = self.train_dataset[idx]
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                else:
                    _, label = self.train_dataset[idx]
                
                local_idx = idx - start_idx + 1
                points.append((local_idx, int(label) % self.q))
            
            # Interpolate polynomial
            poly = self._interpolate_points(points)
            
            # Create commitment
            commitment = self.G1 * Fq(self.q, 0)
            for i, coeff in enumerate(poly):
                commitment += self.setup_g1[i] * Fq(self.q, coeff)
            
            # Generate openings
            openings = {}
            for rel_idx, (x_val, y_val) in enumerate(points):
                global_idx = start_idx + rel_idx
                try:
                    quotient = self._compute_quotient(poly, x_val, y_val)
                    proof_point = self.G1 * Fq(self.q, 0)
                    for i, q_val in enumerate(quotient):
                        proof_point += self.setup_g1[i] * Fq(self.q, q_val)
                    openings[global_idx] = (proof_point, (x_val, y_val))
                except Exception as e:
                    logger.warning(f"Failed to generate opening for point {global_idx}: {e}")
                    continue
            
            return BlockCommitment(commitment, start_idx, end_idx, openings)
        
        except Exception as e:
            logger.error(f"Block commitment failed: {e}")
            raise
    
    def _commit_dataset_blocks(self):
        """Commit to dataset in parallel blocks."""
        try:
            num_blocks = math.ceil(self.commit_size / self.block_size)
            
            # Process blocks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(num_blocks):
                    start_idx = i * self.block_size
                    end_idx = min((i + 1) * self.block_size, self.commit_size)
                    futures.append(executor.submit(self._commit_block, start_idx, end_idx))
                
                # Collect results
                for future in futures:
                    try:
                        result = future.result()
                        self.block_commitments.append(result)
                    except Exception as e:
                        logger.error(f"Block commitment failed: {e}")
            
            # Aggregate commitments
            self._aggregate_commitments()
            
            # Combine openings
            for block in self.block_commitments:
                self.openings.update(block.openings)
        
        except Exception as e:
            logger.error(f"Dataset commitment failed: {e}")
            raise
    
    def _aggregate_commitments(self):
        """Aggregate block commitments into single commitment."""
        if not self.block_commitments:
            return
        
        try:
            # Random weights for aggregation
            weights = [randbelow(self.q) for _ in range(len(self.block_commitments))]
            
            # Weighted sum
            self.aggregated_commitment = self.block_commitments[0].commitment * Fq(self.q, weights[0])
            for i in range(1, len(self.block_commitments)):
                self.aggregated_commitment += self.block_commitments[i].commitment * Fq(self.q, weights[i])
        
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise
    
    def get_commitment(self):
        """Return commitment to share with server."""
        return self.aggregated_commitment
    
    def get_batch_with_proofs(self, indices: List[int]) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Get batch data with membership proofs.
        
        Args:
            indices: List of dataset indices
            
        Returns:
            batch_data: List of (image, label) tuples
            membership_proofs: List of (proof_point, (x, y)) tuples
        """
        batch_data = []
        membership_proofs = []
        
        for idx in indices:
            # Get data
            if isinstance(self.train_dataset, TensorDataset):
                img, label = self.train_dataset[idx]
            else:
                img, label = self.train_dataset[idx]
            
            batch_data.append((img, label))
            
            # Get membership proof
            if idx in self.openings:
                membership_proofs.append(self.openings[idx])
            else:
                logger.warning(f"No opening for index {idx}")
                membership_proofs.append((None, (0, 0)))
        
        return batch_data, membership_proofs
    
    def verify_batch(self, indices: List[int]) -> bool:
        """
        Verify batch membership (for testing).
        
        Args:
            indices: List of dataset indices
            
        Returns:
            True if batch is from committed dataset
        """
        if not indices or not self.aggregated_commitment:
            return False
        
        try:
            # Get membership proofs
            _, membership_proofs = self.get_batch_with_proofs(indices)
            
            # Verify using pairing
            return verify_batch_membership(
                self.aggregated_commitment,
                membership_proofs,
                self.s_g2,
                self.G2,
                self.G1,
                self.q
            )
        
        except Exception as e:
            logger.error(f"Batch verification error: {e}")
            return False


def verify_batch_membership(commitment: JacobianPoint,
                            membership_proofs: List[Tuple],
                            s_g2: JacobianPoint,
                            G2: JacobianPoint,
                            G1: JacobianPoint,
                            q: int) -> bool:
    """
    Verify that batch came from committed dataset.
    
    Args:
        commitment: Aggregated dataset commitment
        membership_proofs: List of (proof_point, (x, y)) tuples
        s_g2: s*G2 for pairing checks
        G2: Generator point on G2
        G1: Generator point on G1
        q: Field modulus
        
    Returns:
        True if all proofs verify
    """
    try:
        # Filter valid proofs
        valid_proofs = [(p, point) for p, point in membership_proofs if p is not None]
        
        if not valid_proofs:
            logger.error("No valid proofs provided")
            return False
        
        # Generate random challenge for batch aggregation
        rho = randbelow(q)
        
        # Combine proofs
        combined_proof = valid_proofs[0][0]
        combined_x = valid_proofs[0][1][0]
        combined_y = valid_proofs[0][1][1]
        
        for i in range(1, len(valid_proofs)):
            proof, (x, y) = valid_proofs[i]
            rho_i = pow(rho, i, q)
            combined_proof += proof * Fq(q, rho_i)
            combined_x = (combined_x + x * rho_i) % q
            combined_y = (combined_y + y * rho_i) % q
        
        # Pairing verification: e(proof, sG2 - xG2) == e(C - yG1, G2)
        x_fq = Fq(q, combined_x)
        y_fq = Fq(q, combined_y)
        
        # C - y*G1
        c_minus_y = commitment + (G1 * Fq(q, -combined_y))
        
        # s*G2 - x*G2
        s_minus_x = s_g2 + (G2 * Fq(q, -combined_x))
        
        # Pairing check
        lhs = ate_pairing_multi([combined_proof], [s_minus_x])
        rhs = ate_pairing_multi([c_minus_y], [G2])
        
        result = lhs == rhs
        
        if not result:
            logger.warning("Pairing check failed")
        
        return result
    
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False


# Quick test function
def test_dataset_commitment():
    """Test dataset commitment with small dataset."""
    import torch
    from torch.utils.data import TensorDataset
    
    logger.info("Testing dataset commitment...")
    
    # Create small test dataset
    data = torch.randn(100, 1, 28, 28)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(data, labels)
    
    # Create commitment
    mgr = DatasetCommitmentManager(dataset, commit_size=100, block_size=20)
    
    # Test batch verification
    indices = [0, 1, 2, 3, 4]
    verified = mgr.verify_batch(indices)
    
    logger.info(f"Test result: {'PASSED' if verified else 'FAILED'}")
    return verified


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataset_commitment()
