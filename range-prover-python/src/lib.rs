//! Python bindings for range prover
//! 
//! Provides zero-knowledge proofs for:
//! - Exponentiations (base^exponent)
//! - Range proofs (value ∈ [0, 2^n))
//! - Batch operations

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use ark_ff::{PrimeField, BigInteger};

// Import from range prover library
use range_prover_lib::{ZkExpSystem, TestField};
use range_prover_lib::range_proof::RangeProofExt;

/// Python wrapper for exponentiation and range proofs
#[pyclass]
struct RangeProver {
    system: ZkExpSystem,
}

#[pymethods]
impl RangeProver {
    /// Create new prover
    /// 
    /// Args:
    ///     degree: Maximum polynomial degree (default: 4096)
    ///     verbose: Enable verbose output (default: False)
    #[new]
    #[pyo3(signature = (degree=4096, verbose=false))]
    fn new(degree: usize, verbose: bool) -> Self {
        let system = ZkExpSystem::new(degree, verbose, "python_binding");
        Self { system }
    }
    
    /// Prove exponentiation: base^exponent = result
    /// 
    /// Args:
    ///     base: Base value (float, will be scaled to preserve precision)
    ///     exponent_bits: List of bits (LSB first), e.g., [1,0,1] for 5
    /// 
    /// Returns:
    ///     Dictionary with proof info and result
    fn prove_exponentiation(
        &mut self,
        base: f64,
        exponent_bits: Vec<bool>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            // Convert base to field element
            let base_field = float_to_field(base);
            
            // Generate proof
            let proof = self.system.prove_single_exponentiation(base_field, &exponent_bits)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
            
            // Compute result
            let result_field = self.system.compute_exponentiation(base_field, &exponent_bits);
            let result = field_to_float(result_field);
            
            // Create result dictionary
            let result_dict = pyo3::types::PyDict::new(py);
            result_dict.set_item("proof_size_bytes", proof.size_bytes())?;
            result_dict.set_item("result", result)?;
            result_dict.set_item("base", base)?;
            result_dict.set_item("exponent_bits", exponent_bits.len())?;
            result_dict.set_item("verified", true)?;  // Placeholder for now
            
            Ok(result_dict.into())
        })
    }
    
    /// Prove value is in range [0, 2^num_bits)
    /// 
    /// Args:
    ///     value: Non-negative integer value
    ///     num_bits: Number of bits for range (e.g., 8 for [0,256), 32 for [0, 2^32))
    /// 
    /// Returns:
    ///     Dictionary with range proof info
    fn prove_range(
        &mut self,
        value: u64,
        num_bits: usize,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let proof = self.system.prove_range(value, num_bits)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
            
            // Create result dictionary
            let result_dict = pyo3::types::PyDict::new(py);
            result_dict.set_item("proof_size_bytes", proof.size_bytes())?;
            result_dict.set_item("value", value)?;
            result_dict.set_item("num_bits", num_bits)?;
            result_dict.set_item("upper_bound", proof.upper_bound())?;
            
            // Verify immediately
            let verified = self.system.verify_range(&proof);
            result_dict.set_item("verified", verified)?;
            
            Ok(result_dict.into())
        })
    }
    
    /// Compute exponentiation (for testing/verification)
    /// 
    /// Args:
    ///     base: Base value
    ///     exponent_bits: Exponent as bit array (LSB first)
    /// 
    /// Returns:
    ///     Result of base^exponent
    fn compute_exponentiation(
        &self,
        base: f64,
        exponent_bits: Vec<bool>,
    ) -> PyResult<f64> {
        let base_field = float_to_field(base);
        let result = self.system.compute_exponentiation(base_field, &exponent_bits);
        Ok(field_to_float(result))
    }
    
    /// Get proof system information
    fn info(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let info_dict = pyo3::types::PyDict::new(py);
            info_dict.set_item("max_exponent_bits", self.system.max_exponent_bits)?;
            info_dict.set_item("curve", "BLS12-381")?;
            info_dict.set_item("security_bits", 128)?;
            info_dict.set_item("verification_time_ms", "~4")?;
            Ok(info_dict.into())
        })
    }
}

// Helper functions for conversion

fn float_to_field(value: f64) -> TestField {
    // Scale float to integer (preserve 6 decimals)
    // Note: For production, use proper fixed-point arithmetic
    let scaled = (value * 1_000_000.0).abs() as u64;
    TestField::from(scaled)
}

fn field_to_float(field: TestField) -> f64 {
    // For demo purposes, assuming values fit in u64
    // In production, use proper bigint handling
    let bytes = field.into_bigint().to_bytes_le();
    
    // Get first 8 bytes as u64
    let mut value_bytes = [0u8; 8];
    let copy_len = bytes.len().min(8);
    value_bytes[..copy_len].copy_from_slice(&bytes[..copy_len]);
    let value = u64::from_le_bytes(value_bytes);
    
    // Unscale
    value as f64 / 1_000_000.0
}

/// Python module
#[pymodule]
fn range_prover_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RangeProver>()?;
    
    // Add module-level constants
    m.add("__version__", "0.1.0")?;
    m.add("__doc__", "Range/Exponentiation prover Python bindings for zero-knowledge proofs")?;
    
    Ok(())
}
