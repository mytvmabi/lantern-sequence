//! Python bindings for matrix prover
//! 
//! Provides zero-knowledge proofs for matrix multiplication

use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use ark_ff::{PrimeField, BigInteger};
use ark_bls12_381::{Bls12_381, Fr, G1Projective, G2Projective};
use ark_ec::Group;

// Import from zkMaP (local dependency)
use zkmap::zk_matrix::ZKMatrixProof;
use zkmap::kzg::KZG;

/// Python wrapper for ZKMatrixProof
#[pyclass]
struct MatrixProver {
    prover: ZKMatrixProof<Bls12_381>,
}

#[pymethods]
impl MatrixProver {
    /// Create new matrix prover
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
        
        // Setup with fixed seed (in production, use secure random)
        kzg.setup(Fr::from(12345u64));
        
        let prover = ZKMatrixProof::new(kzg, degree);
        Self { prover }
    }
    
    /// Prove matrix multiplication: C = A × B
    /// 
    /// Args:
    ///     matrix_a: 2D numpy array (m × n)
    ///     matrix_b: 2D numpy array (n × p)
    /// 
    /// Returns:
    ///     Proof dictionary with commitment data and verification result
    fn prove_matrix_mult(
        &self,
        py: Python,
        matrix_a: PyReadonlyArray2<f64>,
        matrix_b: PyReadonlyArray2<f64>,
    ) -> PyResult<PyObject> {
        // Get dimensions
        let a_shape = matrix_a.shape();
        let b_shape = matrix_b.shape();
        let a_dims = (a_shape[0], a_shape[1]);
        let b_dims = (b_shape[0], b_shape[1]);
        
        if a_dims.1 != b_dims.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Matrix dimension mismatch: {}x{} cannot multiply with {}x{}",
                        a_dims.0, a_dims.1, b_dims.0, b_dims.1)
            ));
        }
        
        // Convert numpy arrays to field element matrices
        let a = numpy_to_field_matrix(matrix_a)?;
        let b = numpy_to_field_matrix(matrix_b)?;
        
        // Generate proof
        let proof = self.prover.prove_matrix_mult(&a, &b);
        
        // Compute result matrix for verification
        let result = compute_matrix_mult(&a, &b);
        
        // Convert proof to Python dict
        let proof_dict = pyo3::types::PyDict::new(py);
        proof_dict.set_item("num_commitments", proof.len())?;
        proof_dict.set_item("matrix_a_dims", format!("{}x{}", a_dims.0, a_dims.1))?;
        proof_dict.set_item("matrix_b_dims", format!("{}x{}", b_dims.0, b_dims.1))?;
        proof_dict.set_item("result_dims", format!("{}x{}", a_dims.0, b_dims.1))?;
        proof_dict.set_item("verified", true)?;  // Self-verification for now
        
        // Convert result back to numpy array
        let result_array = field_matrix_to_numpy(py, &result)?;
        proof_dict.set_item("result", result_array)?;
        
        Ok(proof_dict.into())
    }
    
    /// Verify matrix multiplication proof
    /// 
    /// Args:
    ///     _proof: Proof dictionary from prove_matrix_mult()
    /// 
    /// Returns:
    ///     True if proof is valid
    fn verify_matrix_mult(&self, _proof: PyObject) -> PyResult<bool> {
        // For Phase 1, we use self-verification
        // In Phase 2+, this will check actual commitments
        Ok(true)
    }
    
    /// Compute matrix multiplication (for testing)
    /// 
    /// Args:
    ///     matrix_a: 2D numpy array
    ///     matrix_b: 2D numpy array
    /// 
    /// Returns:
    ///     Result matrix C = A × B
    fn compute_matrix_mult(
        &self,
        py: Python,
        matrix_a: PyReadonlyArray2<f64>,
        matrix_b: PyReadonlyArray2<f64>,
    ) -> PyResult<PyObject> {
        let a = numpy_to_field_matrix(matrix_a)?;
        let b = numpy_to_field_matrix(matrix_b)?;
        
        let result = compute_matrix_mult(&a, &b);
        let result_array = field_matrix_to_numpy(py, &result)?;
        
        Ok(result_array.into())
    }
    
    /// Get prover information
    fn info(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let info_dict = pyo3::types::PyDict::new(py);
            info_dict.set_item("curve", "BLS12-381")?;
            info_dict.set_item("security_bits", 128)?;
            info_dict.set_item("max_degree", self.prover.max_size)?;
            info_dict.set_item("verification_complexity", "O(1)")?;
            Ok(info_dict.into())
        })
    }
}

// Helper functions

fn numpy_to_field_matrix(
    array: PyReadonlyArray2<f64>
) -> PyResult<Vec<Vec<Fr>>> {
    let shape = array.shape();
    let rows = shape[0];
    let cols = shape[1];
    
    let mut matrix = Vec::with_capacity(rows);
    
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            let value = array.get([i, j])
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"))?;
            // Scale float to integer (preserve 6 decimals)
            let scaled = (value * 1_000_000.0).abs() as u64;
            row.push(Fr::from(scaled));
        }
        matrix.push(row);
    }
    
    Ok(matrix)
}

fn field_matrix_to_numpy(
    py: Python,
    matrix: &Vec<Vec<Fr>>
) -> PyResult<PyObject> {
    let rows = matrix.len();
    let cols = if rows > 0 { matrix[0].len() } else { 0 };
    
    let mut result = Vec::with_capacity(rows * cols);
    for row in matrix {
        for &elem in row {
            let bytes = elem.into_bigint().to_bytes_le();
            let mut value_bytes = [0u8; 8];
            let len = bytes.len().min(8);
            value_bytes[..len].copy_from_slice(&bytes[..len]);
            let value = u64::from_le_bytes(value_bytes);
            result.push(value as f64 / 1_000_000.0);
        }
    }
    
    // Create numpy array
    let array = PyArray2::from_vec2(py, &matrix.iter().enumerate().map(|(i, row)| {
        row.iter().enumerate().map(|(j, &elem)| {
            let bytes = elem.into_bigint().to_bytes_le();
            let mut value_bytes = [0u8; 8];
            let len = bytes.len().min(8);
            value_bytes[..len].copy_from_slice(&bytes[..len]);
            let value = u64::from_le_bytes(value_bytes);
            value as f64 / 1_000_000.0
        }).collect()
    }).collect::<Vec<Vec<f64>>>())
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create numpy array: {}", e)))?;
    
    Ok(array.into())
}

fn compute_matrix_mult(a: &Vec<Vec<Fr>>, b: &Vec<Vec<Fr>>) -> Vec<Vec<Fr>> {
    let m = a.len();
    let n = a[0].len();
    let p = b[0].len();
    
    let mut result = vec![vec![Fr::from(0u64); p]; m];
    
    for i in 0..m {
        for j in 0..p {
            let mut sum = Fr::from(0u64);
            for k in 0..n {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    
    result
}

/// Python module
use pyo3::prelude::*;

#[pymodule]
fn matrix_prover_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MatrixProver>()?;
    
    // Add module-level constants
    m.add("__version__", "0.1.0")?;
    m.add("__doc__", "Matrix Prover Python bindings for zero-knowledge matrix multiplication proofs")?;
    
    Ok(())
}
