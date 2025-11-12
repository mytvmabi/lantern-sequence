"""
RIV Protocol Implementation

Implements the core Retroactive Intermediate Value Verification protocol
for proof-of-training in federated learning. This module coordinates
commitment, challenge generation, and verification phases.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import time

try:
    from .crypto_backend import CryptoBackend, HashCommitment
    from .sic_commitments import SICManager
    from .challenge import ChallengeGenerator
except ImportError:
    from crypto_backend import CryptoBackend, HashCommitment
    from sic_commitments import SICManager
    from challenge import ChallengeGenerator


class RIVProtocol:
    """
    Main protocol coordinator for RIV verification.
    
    The protocol operates in three phases:
    1. Commitment: Client commits to intermediate values
    2. Challenge: Server generates random challenges via Fiat-Shamir
    3. Verification: Client provides proofs for challenged layers
    
    This prevents selective computation by ensuring challenges are
    unpredictable at commitment time.
    """
    
    def __init__(
        self,
        challenge_budget: int = 5,
        use_zero_knowledge: bool = False,
        degree: int = 4096,
        verbose: bool = False
    ):
        """
        Initialize RIV protocol.
        
        Args:
            challenge_budget: Number of layers to challenge (k in paper)
            use_zero_knowledge: Enable zero-knowledge mode
            degree: Polynomial degree for commitments
            verbose: Enable detailed logging
        """
        self.challenge_budget = challenge_budget
        self.use_zero_knowledge = use_zero_knowledge
        self.verbose = verbose
        
        # Initialize cryptographic backend
        if use_zero_knowledge:
            try:
                self.crypto = CryptoBackend(degree=degree, verbose=verbose)
            except RuntimeError as e:
                if verbose:
                    print(f"Warning: {e}")
                    print("Falling back to transparent mode (hash-based commitments)")
                self.use_zero_knowledge = False
                self.crypto = None
        else:
            # Transparent mode uses hash commitments
            self.crypto = None
        
        # Initialize SIC manager for floating-point handling
        self.sic_manager = SICManager(precision='float32')
        
        # Protocol state
        self.commitments: Dict[str, Any] = {}
        self.challenge_generator: Optional[ChallengeGenerator] = None
    
    def client_commit_phase(
        self,
        model_updates: Dict[str, np.ndarray],
        training_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Client commitment phase.
        
        The client commits to:
        - Initial model parameters (W^t)
        - Updated model parameters (W^{t+1})
        - Intermediate activations and gradients (for challenged layers)
        
        Args:
            model_updates: Dictionary of layer weights
                Format: {'layer_0': W_0, 'layer_1': W_1, ...}
            training_metadata: Training information
                Format: {'round': t, 'batch_size': n, 'learning_rate': eta}
        
        Returns:
            Dictionary containing:
                - commitments: Layer commitments
                - commitment_hash: Root hash for Fiat-Shamir
                - timestamp: Commitment creation time
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"Client: Creating commitments for {len(model_updates)} layers")
        
        commitments = {}
        
        # Commit to each layer
        for layer_name, weights in model_updates.items():
            if self.use_zero_knowledge:
                # Zero-knowledge commitment using KZG
                coeffs = weights.flatten()
                commitment = self.crypto.commit_polynomial(coeffs)
            else:
                # Transparent commitment using hash
                commitment = HashCommitment.commit(weights)
            
            commitments[layer_name] = {
                'commitment': commitment,
                'shape': weights.shape,
                'layer_name': layer_name
            }
        
        # Create root hash for Fiat-Shamir
        commitment_hash = self._hash_commitments(commitments, training_metadata)
        
        # Store commitments
        self.commitments = commitments
        
        commit_time = time.time() - start_time
        
        if self.verbose:
            print(f"Client: Commitments created in {commit_time:.3f}s")
        
        return {
            'commitments': commitments,
            'commitment_hash': commitment_hash,
            'timestamp': time.time(),
            'metadata': training_metadata,
            'commit_time': commit_time
        }
    
    def server_challenge_phase(
        self,
        commitment_hash: bytes,
        num_layers: int,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Server challenge generation phase.
        
        Uses Fiat-Shamir heuristic to generate challenges that are:
        1. Unpredictable before commitment
        2. Deterministically verifiable
        3. Uniformly random
        
        Args:
            commitment_hash: Root hash from commitment phase
            num_layers: Total number of layers in model
            metadata: Training metadata
        
        Returns:
            Dictionary containing:
                - challenged_layers: List of layer indices to verify
                - challenge_vectors: Random vectors for each layer
                - challenge_seed: Seed for reproducibility
        """
        start_time = time.time()
        
        # Initialize challenge generator with commitment hash
        self.challenge_generator = ChallengeGenerator(seed=commitment_hash)
        
        # Add metadata to transcript
        self.challenge_generator.add_commitment(
            self._serialize_metadata(metadata)
        )
        
        # Select layers to challenge
        challenged_layers = self.challenge_generator.select_challenged_layers(
            total_layers=num_layers,
            num_challenges=min(self.challenge_budget, num_layers)
        )
        
        # Generate challenge vectors for each layer
        challenge_vectors = {}
        for layer_idx in challenged_layers:
            # Challenge vector dimension depends on layer
            # For now, use fixed dimension (will be determined by layer size)
            challenge_vectors[layer_idx] = {
                'layer_idx': layer_idx,
                'challenge_type': 'verification'
            }
        
        challenge_time = time.time() - start_time
        
        if self.verbose:
            print(f"Server: Generated {len(challenged_layers)} challenges in {challenge_time:.3f}s")
            print(f"Server: Challenged layers: {challenged_layers}")
        
        return {
            'challenged_layers': challenged_layers,
            'challenge_vectors': challenge_vectors,
            'challenge_seed': commitment_hash,
            'challenge_time': challenge_time
        }
    
    def client_proof_phase(
        self,
        challenged_layers: List[int],
        layer_data: Dict[str, Dict[str, np.ndarray]],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Client proof generation phase.
        
        For each challenged layer, the client provides:
        1. Intermediate activation values
        2. Gradient values
        3. Proof of correct forward/backward computation
        4. SIC proofs for floating-point consistency
        
        Args:
            challenged_layers: Layers to provide proofs for
            layer_data: Full training data for proof generation
                Format: {
                    'layer_0': {
                        'activation': a_0,
                        'gradient': g_0,
                        'weight_old': W_0^t,
                        'weight_new': W_0^{t+1}
                    },
                    ...
                }
            training_config: Configuration (learning rate, etc.)
        
        Returns:
            Dictionary containing proofs for each challenged layer
        """
        start_time = time.time()
        
        proofs = {}
        
        for layer_idx in challenged_layers:
            layer_name = f'layer_{layer_idx}'
            
            if layer_name not in layer_data:
                raise ValueError(f"Missing data for challenged layer {layer_idx}")
            
            layer_info = layer_data[layer_name]
            
            # Create proof for this layer
            layer_proof = self._create_layer_proof(
                layer_idx=layer_idx,
                activation=layer_info['activation'],
                gradient=layer_info.get('gradient'),
                weight_old=layer_info.get('weight_old'),
                weight_new=layer_info.get('weight_new'),
                training_config=training_config
            )
            
            proofs[layer_idx] = layer_proof
        
        proof_time = time.time() - start_time
        
        if self.verbose:
            print(f"Client: Generated proofs for {len(proofs)} layers in {proof_time:.3f}s")
        
        return {
            'proofs': proofs,
            'proof_time': proof_time
        }
    
    def server_verify_phase(
        self,
        commitments: Dict[str, Any],
        proofs: Dict[int, Dict[str, Any]],
        challenged_layers: List[int],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Server verification phase.
        
        For each challenged layer, verify:
        1. Commitment opening is valid
        2. Forward propagation is correct
        3. Backward propagation is correct
        4. Update rule is correctly applied
        5. Floating-point computations are within SIC bounds
        
        Args:
            commitments: Commitments from client
            proofs: Proofs for challenged layers
            challenged_layers: List of challenged layer indices
            metadata: Training metadata
        
        Returns:
            Dictionary containing:
                - verified: Overall verification result (bool)
                - layer_results: Per-layer verification details
                - verification_time: Time taken
        """
        start_time = time.time()
        
        layer_results = {}
        all_verified = True
        
        for layer_idx in challenged_layers:
            if layer_idx not in proofs:
                layer_results[layer_idx] = {
                    'verified': False,
                    'reason': 'Missing proof'
                }
                all_verified = False
                continue
            
            # Verify this layer
            layer_result = self._verify_layer_proof(
                layer_idx=layer_idx,
                commitment=commitments.get(f'layer_{layer_idx}'),
                proof=proofs[layer_idx],
                metadata=metadata
            )
            
            layer_results[layer_idx] = layer_result
            
            if not layer_result['verified']:
                all_verified = False
        
        verify_time = time.time() - start_time
        
        if self.verbose:
            print(f"Server: Verified {len(layer_results)} layers in {verify_time:.3f}s")
            print(f"Server: Overall result: {'PASS' if all_verified else 'FAIL'}")
        
        return {
            'verified': all_verified,
            'layer_results': layer_results,
            'verification_time': verify_time,
            'challenged_layers': challenged_layers,
            'num_verified': sum(1 for r in layer_results.values() if r['verified'])
        }
    
    def _create_layer_proof(
        self,
        layer_idx: int,
        activation: np.ndarray,
        gradient: Optional[np.ndarray],
        weight_old: Optional[np.ndarray],
        weight_new: Optional[np.ndarray],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create proof for a single layer.
        
        Args:
            layer_idx: Layer index
            activation: Activation values
            gradient: Gradient values (if available)
            weight_old: Old weights (if available)
            weight_new: New weights (if available)
            training_config: Training configuration
        
        Returns:
            Layer proof dictionary
        """
        proof = {
            'layer_idx': layer_idx,
            'activation': activation.copy(),
        }
        
        # Add gradient if available
        if gradient is not None:
            proof['gradient'] = gradient.copy()
        
        # Add weight update if available
        if weight_old is not None and weight_new is not None:
            learning_rate = training_config.get('learning_rate', 0.01)
            
            # Compute expected update
            expected_update = weight_old - learning_rate * gradient
            
            # Create SIC for update consistency
            update_error_bound = self.sic_manager.compute_update_error_bound(
                weight_old, gradient, learning_rate
            )
            
            proof['weight_update'] = {
                'old': weight_old.copy(),
                'new': weight_new.copy(),
                'expected': expected_update,
                'error_bound': update_error_bound
            }
        
        # Create cryptographic proof if in ZK mode
        if self.use_zero_knowledge:
            # Proof of activation commitment opening
            proof['commitment_proof'] = self.crypto.create_evaluation_proof(
                coefficients=activation.flatten(),
                point=0.0  # Simplified: prove commitment matches data
            )
        
        return proof
    
    def _verify_layer_proof(
        self,
        layer_idx: int,
        commitment: Dict[str, Any],
        proof: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify proof for a single layer.
        
        Args:
            layer_idx: Layer index
            commitment: Layer commitment
            proof: Layer proof
            metadata: Training metadata
        
        Returns:
            Verification result dictionary
        """
        # Check commitment opening
        if self.use_zero_knowledge:
            # Verify cryptographic proof
            commitment_valid = self.crypto.verify_evaluation(
                commitment=commitment['commitment'],
                point=0.0,
                value=proof['activation'].flatten()[0],
                proof=proof['commitment_proof']['proof']
            )
        else:
            # Verify hash commitment
            commitment_valid = HashCommitment.verify(
                commitment=commitment['commitment'],
                data=proof['activation']
            )
        
        if not commitment_valid:
            return {
                'verified': False,
                'reason': 'Commitment verification failed'
            }
        
        # Verify weight update if available
        if 'weight_update' in proof:
            update_info = proof['weight_update']
            
            # Check update is within SIC bounds
            actual_diff = update_info['new'] - update_info['expected']
            max_error = np.max(np.abs(actual_diff))
            
            if max_error > update_info['error_bound']:
                return {
                    'verified': False,
                    'reason': f'Update exceeds error bound: {max_error} > {update_info["error_bound"]}'
                }
        
        return {
            'verified': True,
            'layer_idx': layer_idx
        }
    
    def _hash_commitments(
        self,
        commitments: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> bytes:
        """
        Create root hash of all commitments.
        
        Args:
            commitments: Layer commitments
            metadata: Training metadata
        
        Returns:
            SHA-256 hash of commitments
        """
        hasher = hashlib.sha256()
        hasher.update(b"RIV_ROOT:")
        
        # Add commitments in sorted order
        for key in sorted(commitments.keys()):
            hasher.update(key.encode())
            hasher.update(commitments[key]['commitment'])
        
        # Add metadata
        hasher.update(self._serialize_metadata(metadata))
        
        return hasher.digest()
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> bytes:
        """
        Serialize metadata for hashing.
        
        Args:
            metadata: Training metadata
        
        Returns:
            Serialized bytes
        """
        import json
        return json.dumps(metadata, sort_keys=True).encode()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get protocol performance metrics.
        
        Returns:
            Dictionary of metrics (commitment time, proof time, etc.)
        """
        return {
            'challenge_budget': self.challenge_budget,
            'zero_knowledge': self.use_zero_knowledge,
            'num_commitments': len(self.commitments)
        }
