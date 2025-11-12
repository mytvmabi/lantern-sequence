"""
Stochastic Interval Commitments (SIC)

Handles rigorous floating-point error analysis using Higham's backward
error analysis framework. Enables verification of floating-point computations
without exact value matching.
"""

import numpy as np
from typing import Tuple, Dict, Optional


class SICManager:
    """
    Manager for Stochastic Interval Commitments.
    
    Key innovation: Instead of committing to exact floating-point values
    (which may differ due to IEEE-754 nondeterminism), commit to intervals
    that provably contain the true mathematical result.
    
    Based on Higham's backward error analysis: if the computed result satisfies
    (W + ΔW) * x = y where ||ΔW|| ≤ ε||W||, then y is the exact result for
    a slightly perturbed input.
    """
    
    def __init__(self, precision: str = 'float32'):
        """
        Initialize SIC manager.
        
        Args:
            precision: Floating-point precision ('float32' or 'float64')
        """
        self.precision = precision
        
        # Machine epsilon for error bounds
        if precision == 'float32':
            self.epsilon = np.finfo(np.float32).eps  # ~1.19e-7
        else:
            self.epsilon = np.finfo(np.float64).eps  # ~2.22e-16
    
    def compute_matmul_error_bound(
        self,
        A: np.ndarray,
        B: np.ndarray
    ) -> float:
        """
        Compute rigorous error bound for matrix multiplication.
        
        Following Higham (2002), Theorem 3.5:
        For C = A @ B with dimensions m×k and k×n:
        ||ΔC||_∞ ≤ k·ε·||A||_∞·||B||_∞ + O(ε²)
        
        Args:
            A: Left matrix (m × k)
            B: Right matrix (k × n)
        
        Returns:
            Error bound δ such that computed C satisfies
            C_computed ∈ [C_exact - δ, C_exact + δ] element-wise
        """
        # Get dimensions
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        
        k = A.shape[1]  # Inner dimension
        
        # Compute matrix norms
        norm_A = np.linalg.norm(A, ord=np.inf)
        norm_B = np.linalg.norm(B, ord=np.inf)
        
        # Error bound from Higham
        # Formula: k * ε * ||A|| * ||B|| / (1 - k*ε)
        # Simplified for k*ε << 1: k * ε * ||A|| * ||B||
        error_bound = k * self.epsilon * norm_A * norm_B
        
        # Add safety factor for accumulated rounding
        safety_factor = 1.5
        error_bound *= safety_factor
        
        return error_bound
    
    def compute_activation_error_bound(
        self,
        x: np.ndarray,
        activation_type: str = 'relu'
    ) -> float:
        """
        Compute error bound for activation function.
        
        Args:
            x: Input to activation
            activation_type: Type of activation ('relu', 'sigmoid', 'tanh')
        
        Returns:
            Error bound for activation output
        """
        if activation_type == 'relu':
            # ReLU is exact (no arithmetic operations)
            # Only error is from input
            return self.epsilon * np.max(np.abs(x))
        
        elif activation_type == 'sigmoid':
            # Sigmoid: σ(x) = 1/(1 + exp(-x))
            # Error from exponential and division
            max_x = np.max(np.abs(x))
            
            # Conservative bound accounting for exp and division
            error_bound = 5 * self.epsilon * max_x
            
            return error_bound
        
        elif activation_type == 'tanh':
            # Tanh: similar analysis to sigmoid
            max_x = np.max(np.abs(x))
            error_bound = 5 * self.epsilon * max_x
            
            return error_bound
        
        else:
            # Conservative default
            return self.epsilon * np.max(np.abs(x)) * 10
    
    def compute_gradient_error_bound(
        self,
        grad_output: np.ndarray,
        input_activation: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        Compute error bound for gradient computation.
        
        For gradient: dL/dW = grad_output @ input_activation^T
        
        Args:
            grad_output: Gradient from next layer
            input_activation: Input activation to this layer
            weights: Current layer weights
        
        Returns:
            Error bound for computed gradient
        """
        # Gradient is also a matrix multiplication
        return self.compute_matmul_error_bound(
            grad_output.reshape(-1, 1),
            input_activation.reshape(1, -1)
        )
    
    def compute_update_error_bound(
        self,
        weight: np.ndarray,
        gradient: np.ndarray,
        learning_rate: float
    ) -> float:
        """
        Compute error bound for weight update.
        
        Update rule: W_new = W_old - η * gradient
        
        Args:
            weight: Current weights
            gradient: Computed gradient
            learning_rate: Learning rate η
        
        Returns:
            Error bound for weight update
        """
        # Error from multiplication η * gradient
        mult_error = self.epsilon * learning_rate * np.max(np.abs(gradient))
        
        # Error from subtraction W - (η * gradient)
        sub_error = self.epsilon * np.max(np.abs(weight))
        
        # Total error bound
        total_error = mult_error + sub_error
        
        # Safety factor
        total_error *= 2.0
        
        return total_error
    
    def create_interval(
        self,
        value: np.ndarray,
        error_bound: float
    ) -> Dict[str, np.ndarray]:
        """
        Create interval commitment for value with error bound.
        
        Args:
            value: Computed value (center of interval)
            error_bound: Rigorous error bound (width of interval)
        
        Returns:
            Dictionary containing:
                - center: Quantized value
                - width: Error bound
                - lower: Lower bound (center - width)
                - upper: Upper bound (center + width)
        """
        return {
            'center': value.copy(),
            'width': error_bound,
            'lower': value - error_bound,
            'upper': value + error_bound
        }
    
    def verify_in_interval(
        self,
        value: np.ndarray,
        interval: Dict[str, np.ndarray]
    ) -> Tuple[bool, float]:
        """
        Verify that value lies within interval.
        
        Args:
            value: Value to verify
            interval: Interval specification
        
        Returns:
            Tuple of (is_valid, max_violation)
            - is_valid: True if value ∈ interval
            - max_violation: Maximum amount by which value exceeds bounds
        """
        lower = interval['lower']
        upper = interval['upper']
        
        # Check element-wise containment
        below_lower = lower - value
        above_upper = value - upper
        
        # Maximum violation
        max_violation = max(
            np.max(below_lower),
            np.max(above_upper),
            0.0
        )
        
        is_valid = max_violation <= 0.0
        
        return is_valid, max_violation
    
    def compute_slack(
        self,
        value: np.ndarray,
        interval: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute slack in interval (how much room is left).
        
        Args:
            value: Value in interval
            interval: Interval specification
        
        Returns:
            Slack amount (distance to nearest boundary)
        """
        center = interval['center']
        width = interval['width']
        
        # Distance from value to center
        distance = np.max(np.abs(value - center))
        
        # Slack is width minus distance
        slack = width - distance
        
        return max(slack, 0.0)
    
    def combine_error_bounds(
        self,
        bounds: list
    ) -> float:
        """
        Combine multiple error bounds conservatively.
        
        Args:
            bounds: List of individual error bounds
        
        Returns:
            Combined error bound (sum of individual bounds)
        """
        # Conservative: sum of bounds
        return sum(bounds)


class SICVerifier:
    """
    Verifier for Stochastic Interval Commitments.
    
    Checks that claimed computations satisfy SIC consistency.
    """
    
    @staticmethod
    def verify_forward_pass(
        input_activation: np.ndarray,
        weights: np.ndarray,
        claimed_output: np.ndarray,
        error_bound: float
    ) -> bool:
        """
        Verify forward pass computation is within error bounds.
        
        Args:
            input_activation: Layer input
            weights: Layer weights
            claimed_output: Claimed layer output
            error_bound: Acceptable error bound
        
        Returns:
            True if computation is consistent
        """
        # Recompute output
        expected_output = np.dot(weights, input_activation)
        
        # Check difference
        max_diff = np.max(np.abs(claimed_output - expected_output))
        
        return max_diff <= error_bound
    
    @staticmethod
    def verify_backward_pass(
        grad_output: np.ndarray,
        input_activation: np.ndarray,
        claimed_gradient: np.ndarray,
        error_bound: float
    ) -> bool:
        """
        Verify backward pass gradient computation.
        
        Args:
            grad_output: Gradient from next layer
            input_activation: Input to this layer
            claimed_gradient: Claimed gradient for weights
            error_bound: Acceptable error bound
        
        Returns:
            True if gradient is consistent
        """
        # Recompute gradient
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, 1)
        if input_activation.ndim == 1:
            input_activation = input_activation.reshape(-1, 1)
        
        expected_gradient = np.dot(grad_output, input_activation.T)
        
        # Check difference
        max_diff = np.max(np.abs(claimed_gradient - expected_gradient))
        
        return max_diff <= error_bound
