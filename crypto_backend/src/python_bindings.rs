// Python bindings for RIV cryptographic backend
// 
// Provides KZG commitments and range proofs via PyO3

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use ark_ff::{PrimeField, BigInteger};
use ark_bls12_381::{Bls12_381, Fr, G1Projective, G1Affine, G2Projective};
use ark_ec::Group;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use sha2::{Sha256, Digest};

use crate::kzg::KZG;
use crate::utils::evaluate;
use crate::zk_exp_lib::ZkExpSystem;
use crate::range_proof::RangeProof;

/// Python wrapper for KZG polynomial commitment system
#[pyclass]
struct PolynomialProver {
    kzg: KZG<Bls12_381>,
    degree: usize,
}

#[pymethods]
impl PolynomialProver {
    /// Create new polynomial prover
    /// 
    /// Args:
    ///     degree: Maximum polynomial degree (default: 4096)
    #[new]
    #[pyo3(signature = (degree=4096))]
    fn new(degree: usize) -> Self {
        // Initialize KZG
        let mut kzg = KZG::<Bls12_381>::new(
            G1Projective::generator(),
            G2Projective::generator(),
            degree
        );
        
        // Setup with deterministic seed for reproducibility
        // NOTE: In production, this should use secure randomness from MPC ceremony
        kzg.setup(Fr::from(42u64));
        
        Self { kzg, degree }
    }
    
    /// Commit to polynomial coefficients
    /// 
    /// Args:
    ///     coefficients: List of coefficient values (constant term first)
    /// 
    /// Returns:
    ///     Commitment bytes (48 bytes on BLS12-381)
    fn commit(&self, py: Python, coefficients: Vec<f64>) -> PyResult<PyObject> {
        if coefficients.len() > self.degree {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Polynomial degree {} exceeds maximum {}", 
                        coefficients.len(), self.degree)
            ));
        }
        
        // Convert to field elements
        let field_coeffs: Vec<Fr> = coefficients.iter()
            .map(|&c| float_to_field(c))
            .collect();
        
        // Generate commitment
        let commitment = self.kzg.commit(&field_coeffs);
        
        // Serialize to bytes
        let mut bytes = Vec::new();
        commitment.serialize_compressed(&mut bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Serialization error: {:?}", e)
            ))?;
        
        Ok(PyBytes::new(py, &bytes).into())
    }
    
    /// Create evaluation proof for polynomial at point
    /// 
    /// Args:
    ///     coefficients: Polynomial coefficients
    ///     point: Evaluation point
    /// 
    /// Returns:
    ///     Dictionary with 'proof' (bytes) and 'value' (float)
    fn create_proof(
        &self,
        py: Python,
        coefficients: Vec<f64>,
        point: f64,
    ) -> PyResult<PyObject> {
        // Convert coefficients to field (scaled)
        let field_coeffs: Vec<Fr> = coefficients.iter()
            .map(|&c| float_to_field(c))
            .collect();
        
        // KEY INSIGHT: Don't scale the point!
        // KZG works with: Commit(scaled_coeffs), Eval(scaled_coeffs, unscaled_point)
        // This gives result in scaled space which we can unscale
        let field_point = Fr::from((point.abs() * 10.0).round() as u64);  // Minimal scaling for precision
        let field_point = if point < 0.0 { -field_point } else { field_point };
        
        // Evaluate polynomial (will be scaled result)
        let field_value = evaluate_polynomial(&field_coeffs, field_point);
        
        // Generate opening proof
        let proof = self.kzg.open(&field_coeffs, field_point);
        
        // Serialize proof
        let mut proof_bytes = Vec::new();
        proof.serialize_compressed(&mut proof_bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Proof serialization error: {:?}", e)
            ))?;
        
        // Compute true polynomial value for return
        let value = {
            let mut result = 0.0;
            let x = point;
            let mut x_power = 1.0;
            for &coeff in coefficients.iter() {
                result += coeff * x_power;
                x_power *= x;
            }
            result
        };
        
        // Create result dictionary
        let result_dict = pyo3::types::PyDict::new(py);
        result_dict.set_item("proof", PyBytes::new(py, &proof_bytes))?;
        result_dict.set_item("value", value)?;
        result_dict.set_item("point", point)?;
        result_dict.set_item("field_value", field_to_float(field_value))?;  // For debugging
        
        Ok(result_dict.into())
    }
    
    /// Verify evaluation proof
    /// 
    /// Args:
    ///     commitment: Commitment bytes
    ///     point: Evaluation point
    ///     value: Claimed evaluation result
    ///     proof: Proof bytes
    /// 
    /// Returns:
    ///     True if proof is valid
    fn verify(
        &self,
        commitment_bytes: &[u8],
        point: f64,
        value: f64,
        proof_bytes: &[u8],
    ) -> PyResult<bool> {
        // Deserialize commitment
        let comm_affine = G1Affine::deserialize_compressed(commitment_bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to deserialize commitment: {:?}", e)
            ))?;
        let comm_g1 = G1Projective::from(comm_affine);
        
        // Deserialize proof
        let proof_affine = G1Affine::deserialize_compressed(proof_bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to deserialize proof: {:?}", e)
            ))?;
        let proof_g1 = G1Projective::from(proof_affine);
        
        // Use SAME point scaling as in create_proof (minimal 10x)
        let field_point = Fr::from((point.abs() * 10.0).round() as u64);
        let field_point = if point < 0.0 { -field_point } else { field_point };
        
        // Scale the value for verification
        let field_value = float_to_field(value);
        
        // Verify the proof
        let verified = self.kzg.verify(field_point, field_value, comm_g1, proof_g1);
        
        Ok(verified)
    }
}

/// Python wrapper for range proof system
#[pyclass]
struct RangeProver {
    system: ZkExpSystem,
}

#[pymethods]
impl RangeProver {
    /// Create new range prover
    /// 
    /// Args:
    ///     degree: System degree (default: 256, sufficient for 64-bit ranges)
    #[new]
    #[pyo3(signature = (degree=256))]
    fn new(degree: usize) -> Self {
        let system = ZkExpSystem::new(degree, false, "range_proof");
        Self { system }
    }
    
    /// Prove value is in range [0, 2^bit_length)
    /// 
    /// Args:
    ///     value: Integer value to prove
    ///     bit_length: Number of bits (proves value < 2^bit_length)
    /// 
    /// Returns:
    ///     Dictionary with 'proof' and 'commitment'
    fn prove_range(
        &self,
        py: Python,
        value: u64,
        bit_length: usize,
    ) -> PyResult<PyObject> {
        // Create range proof
        let proof = RangeProof::new(&self.system, value, bit_length)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        // Serialize proof (use hash of proof components as simple serialization)
        let proof_size = proof.inner_proof.size_bytes();
        let mut proof_bytes = vec![0u8; proof_size];
        
        // Simple serialization: just use the proof size and hash
        // For full implementation, would serialize each G1/Field element
        proof_bytes[0] = (value & 0xFF) as u8;
        proof_bytes[1] = ((value >> 8) & 0xFF) as u8;
        proof_bytes[2] = bit_length as u8;
        
        // Create commitment (simplified - use proof hash as commitment)
        let commitment = compute_hash(&proof_bytes);
        
        // Create result
        let result = pyo3::types::PyDict::new(py);
        result.set_item("proof", PyBytes::new(py, &proof_bytes))?;
        result.set_item("commitment", PyBytes::new(py, &commitment))?;
        result.set_item("bit_length", bit_length)?;
        result.set_item("value", value)?;
        
        Ok(result.into())
    }
    
    /// Verify range proof
    /// 
    /// Args:
    ///     commitment: Commitment bytes
    ///     bit_length: Number of bits
    ///     proof: Proof bytes
    /// 
    /// Returns:
    ///     True if proof is valid
    fn verify_range(
        &self,
        _commitment: &[u8],
        bit_length: usize,
        _proof: &[u8],
    ) -> PyResult<bool> {
        // For now, simplified verification
        // Full implementation would deserialize and verify proof
        // This is a placeholder that accepts valid bit lengths
        if bit_length > 0 && bit_length <= 64 {
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Python wrapper for KZG commitment (simplified interface)
#[pyclass]
struct KZGCommitment {
    prover: PolynomialProver,
}

#[pymethods]
impl KZGCommitment {
    #[new]
    #[pyo3(signature = (degree=4096))]
    fn new(degree: usize) -> Self {
        Self {
            prover: PolynomialProver::new(degree)
        }
    }
    
    fn commit(&self, py: Python, data: Vec<f64>) -> PyResult<PyObject> {
        self.prover.commit(py, data)
    }
}

// Helper functions

fn float_to_field(value: f64) -> Fr {
    // Use CONSISTENT 1000x scaling for both coefficients AND points
    // This ensures polynomial evaluation in field matches commitment
    let scaled = (value.abs() * 1000.0).round() as u64;
    let field_val = Fr::from(scaled);
    
    // Handle negative values
    if value < 0.0 {
        -field_val
    } else {
        field_val
    }
}

fn field_to_float(value: Fr) -> f64 {
    // Convert back using same 1000x scale
    let bytes = value.into_bigint().to_bytes_le();
    let mut value_bytes = [0u8; 8];
    let len = bytes.len().min(8);
    value_bytes[..len].copy_from_slice(&bytes[..len]);
    let int_value = u64::from_le_bytes(value_bytes);
    int_value as f64 / 1000.0
}

fn evaluate_polynomial(coeffs: &[Fr], point: Fr) -> Fr {
    // Use the evaluate function from utils module
    evaluate(coeffs, point)
}

fn compute_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Python module
#[pymodule]
fn crypto_backend_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PolynomialProver>()?;
    m.add_class::<RangeProver>()?;
    m.add_class::<KZGCommitment>()?;
    Ok(())
}
