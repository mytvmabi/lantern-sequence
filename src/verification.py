"""
Verification Module

Implements verification algorithms for RIV protocol.
Checks correctness of forward/backward propagation and weight updates.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any

try:
    from .sic_commitments import SICVerifier, SICManager
except ImportError:
    from sic_commitments import SICVerifier, SICManager


class RIVVerifier:
    """
    Verifier for RIV protocol.
    
    Implements server-side verification logic to check client proofs
    for challenged layers.
    """
    
    def __init__(self, tolerance: float = 1e-4):
        """
        Initialize verifier.
        
        Args:
            tolerance: Numerical tolerance for verification
        """
        self.tolerance = tolerance
        self.sic_verifier = SICVerifier()
        self.sic_manager = SICManager()
    
    def verify_layer_computation(
        self,
        layer_proof: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Verify complete computation for one layer.
        
        Checks:
        1. Forward propagation consistency
        2. Backward propagation consistency
        3. Weight update correctness
        4. All values within SIC bounds
        
        Args:
            layer_proof: Proof data for layer
            training_config: Training configuration
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Extract proof components
        if 'activation' not in layer_proof:
            return False, "Missing activation data"
        
        activation = layer_proof['activation']
        
        # Verify weight update if present
        if 'weight_update' in layer_proof:
            update_valid, reason = self._verify_weight_update(
                layer_proof['weight_update'],
                training_config
            )
            
            if not update_valid:
                return False, f"Weight update verification failed: {reason}"
        
        # Verify forward pass if data present
        if 'forward_data' in layer_proof:
            forward_valid, reason = self._verify_forward_pass(
                layer_proof['forward_data']
            )
            
            if not forward_valid:
                return False, f"Forward pass verification failed: {reason}"
        
        # Verify backward pass if data present
        if 'backward_data' in layer_proof:
            backward_valid, reason = self._verify_backward_pass(
                layer_proof['backward_data']
            )
            
            if not backward_valid:
                return False, f"Backward pass verification failed: {reason}"
        
        return True, "Verification passed"
    
    def _verify_weight_update(
        self,
        update_data: Dict[str, np.ndarray],
        training_config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Verify weight update correctness.
        
        Checks: W_new = W_old - η * gradient (within SIC bounds)
        
        Args:
            update_data: Update information
            training_config: Training configuration
        
        Returns:
            Tuple of (is_valid, reason)
        """
        weight_old = update_data['old']
        weight_new = update_data['new']
        gradient = update_data.get('gradient')
        
        learning_rate = training_config.get('learning_rate', 0.01)
        
        if gradient is None:
            return False, "Missing gradient data"
        
        # Compute expected update
        expected_new = weight_old - learning_rate * gradient
        
        # Compute error bound
        error_bound = self.sic_manager.compute_update_error_bound(
            weight_old, gradient, learning_rate
        )
        
        # Check consistency
        max_diff = np.max(np.abs(weight_new - expected_new))
        
        if max_diff > error_bound:
            return False, f"Update exceeds error bound: {max_diff:.6e} > {error_bound:.6e}"
        
        return True, "Update consistent"
    
    def _verify_forward_pass(
        self,
        forward_data: Dict[str, np.ndarray]
    ) -> Tuple[bool, str]:
        """
        Verify forward propagation.
        
        Checks: output = activation(weights @ input)
        
        Args:
            forward_data: Forward pass data
        
        Returns:
            Tuple of (is_valid, reason)
        """
        input_act = forward_data.get('input')
        weights = forward_data.get('weights')
        output = forward_data.get('output')
        
        if input_act is None or weights is None or output is None:
            return False, "Missing forward pass data"
        
        # Compute expected output
        expected = np.dot(weights, input_act)
        
        # Apply activation if specified
        activation_type = forward_data.get('activation_type', 'linear')
        if activation_type == 'relu':
            expected = np.maximum(0, expected)
        elif activation_type == 'sigmoid':
            expected = 1.0 / (1.0 + np.exp(-expected))
        
        # Compute error bound
        error_bound = self.sic_manager.compute_matmul_error_bound(
            weights, input_act.reshape(-1, 1)
        )
        
        # Check consistency
        max_diff = np.max(np.abs(output - expected))
        
        if max_diff > error_bound:
            return False, f"Forward pass exceeds error bound: {max_diff:.6e} > {error_bound:.6e}"
        
        return True, "Forward pass consistent"
    
    def _verify_backward_pass(
        self,
        backward_data: Dict[str, np.ndarray]
    ) -> Tuple[bool, str]:
        """
        Verify backward propagation.
        
        Checks: gradient = grad_output @ input^T
        
        Args:
            backward_data: Backward pass data
        
        Returns:
            Tuple of (is_valid, reason)
        """
        grad_output = backward_data.get('grad_output')
        input_act = backward_data.get('input')
        gradient = backward_data.get('gradient')
        
        if grad_output is None or input_act is None or gradient is None:
            return False, "Missing backward pass data"
        
        # Compute expected gradient
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, 1)
        if input_act.ndim == 1:
            input_act = input_act.reshape(-1, 1)
        
        expected = np.dot(grad_output, input_act.T)
        
        # Compute error bound
        error_bound = self.sic_manager.compute_gradient_error_bound(
            grad_output, input_act, gradient
        )
        
        # Check consistency
        max_diff = np.max(np.abs(gradient - expected))
        
        if max_diff > error_bound:
            return False, f"Backward pass exceeds error bound: {max_diff:.6e} > {error_bound:.6e}"
        
        return True, "Backward pass consistent"
    
    def verify_batch_consistency(
        self,
        layer_proofs: Dict[int, Dict[str, Any]],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify consistency across multiple layers.
        
        Args:
            layer_proofs: Proofs for all challenged layers
            training_config: Training configuration
        
        Returns:
            Verification results dictionary
        """
        results = {}
        num_verified = 0
        
        for layer_idx, proof in layer_proofs.items():
            is_valid, reason = self.verify_layer_computation(
                proof, training_config
            )
            
            results[layer_idx] = {
                'verified': is_valid,
                'reason': reason
            }
            
            if is_valid:
                num_verified += 1
        
        return {
            'results': results,
            'num_verified': num_verified,
            'num_total': len(layer_proofs),
            'all_verified': num_verified == len(layer_proofs)
        }


def compute_detection_probability(
    num_challenged: int,
    num_total_layers: int,
    adversary_ratio: float
) -> float:
    """
    Compute detection probability for free-rider.
    
    From paper: Pr[detect] ≥ 1 - (1 - q)^k
    where q = fraction of inconsistent layers
          k = number of challenged layers
    
    Args:
        num_challenged: Number of layers challenged (k)
        num_total_layers: Total layers (L)
        adversary_ratio: Fraction of data adversary computed on
    
    Returns:
        Detection probability
    """
    # If adversary computes on fraction p of data,
    # approximately (1-p) of layers will be inconsistent
    q = 1.0 - adversary_ratio
    
    # Detection probability
    prob_detect = 1.0 - (1.0 - q) ** num_challenged
    
    return prob_detect
