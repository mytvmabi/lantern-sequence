
// kzg.rs (optimized)

use std::ops::Mul;
use ark_ff::Field;
use ark_ec::pairing::Pairing;
use crate::utils::{div, mul, evaluate, interpolate};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;
use std::env;

#[derive(Clone)]
pub struct KZG<E: Pairing> {
    pub g1: E::G1,
    pub g2: E::G2,
    pub g2_tau: E::G2,
    pub degree: usize,
    pub crs_g1: Vec<E::G1>,
    pub crs_g2: Vec<E::G2>,
}

impl <E:Pairing> KZG<E> {
    pub fn new(g1: E::G1, g2: E::G2, degree: usize) -> Self {
        Self {
            g1,
            g2,
            g2_tau: g2.mul(E::ScalarField::ZERO),
            degree,
            crs_g1: vec![],
            crs_g2: vec![],
        }
    }

    /// Optimized setup that uses incremental power calculation and parallelization
    pub fn setup(&mut self, secret: E::ScalarField) {
        println!("Setting up KZG with degree {}...", self.degree);
        let start_time = Instant::now();
        
        // Initialize the crs_g1 and crs_g2 vectors with proper capacity
        self.crs_g1 = Vec::with_capacity(self.degree + 1);
        self.crs_g2 = Vec::with_capacity(self.degree + 1);
        
        // Used for parallel processing
        let chunk_size = 10000; // Adjust based on available memory
        let num_chunks = (self.degree + chunk_size) / chunk_size;
        
        let g1 = Arc::new(self.g1);
        let g2 = Arc::new(self.g2);
        
        // Generate powers of secret in parallel chunks
        let mut all_crs_g1 = Vec::with_capacity(self.degree + 1);
        let mut all_crs_g2 = Vec::with_capacity(self.degree + 1);
        
        // Pre-compute all powers of secret
        let mut powers: Vec<E::ScalarField> = Vec::with_capacity(self.degree + 1);
        let mut current_power = E::ScalarField::ONE;
        powers.push(current_power);
        
        for _ in 1..=self.degree {
            current_power *= secret;
            powers.push(current_power);
        }
        
        println!("Pre-computed powers in {:.2?}", start_time.elapsed());
        
        // Process chunks in parallel
        let crs_g1_chunks: Vec<Vec<E::G1>> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start_idx = chunk_idx * chunk_size;
                let end_idx = std::cmp::min(start_idx + chunk_size, self.degree + 1);
                
                let g1_ref = Arc::clone(&g1);
                let powers_slice = &powers[start_idx..end_idx];
                
                powers_slice.iter()
                    .map(|power| g1_ref.mul(*power))
                    .collect()
            })
            .collect();
        
        // Combine chunks
        for chunk in crs_g1_chunks {
            all_crs_g1.extend(chunk);
        }
        
        // Do the same for G2 (only if needed for verification)
        // Note: For most applications, we only need a few elements in G2
        // But for the full multi-point openings, we need the full CRS in G2
        let crs_g2_chunks: Vec<Vec<E::G2>> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start_idx = chunk_idx * chunk_size;
                let end_idx = std::cmp::min(start_idx + chunk_size, self.degree + 1);
                
                let g2_ref = Arc::clone(&g2);
                let powers_slice = &powers[start_idx..end_idx];
                
                powers_slice.iter()
                    .map(|power| g2_ref.mul(*power))
                    .collect()
            })
            .collect();
        
        // Combine chunks
        for chunk in crs_g2_chunks {
            all_crs_g2.extend(chunk);
        }
        
        self.crs_g1 = all_crs_g1;
        self.crs_g2 = all_crs_g2;
        self.g2_tau = self.g2.mul(secret);
        
        println!("KZG setup completed in {:.2?}", start_time.elapsed());
    }

    // Rest of the KZG implementation remains the same
    pub fn commit(&self, poly: &[E::ScalarField]) -> E::G1 {
        let mut commitment = self.g1.mul(E::ScalarField::ZERO);
        for i in 0..std::cmp::min(poly.len(), self.crs_g1.len()) {
            commitment += self.crs_g1[i] * poly[i];
        }
        commitment
    }

    pub fn open(&self, poly: &[E::ScalarField], point: E::ScalarField) -> E::G1 {
        // evaluate the polynomial at point
        let value = evaluate(poly, point);

        // initialize denominator
        let denominator = [-point, E::ScalarField::ONE];

        // initialize numerator
        let first = poly[0] - value;
        let rest = &poly[1..];
        let temp: Vec<E::ScalarField> = std::iter::once(first).chain(rest.iter().cloned()).collect();
        let numerator: &[E::ScalarField] = &temp;

        // get quotient by dividing numerator by denominator
        let quotient = div(numerator, &denominator).unwrap();

        // calculate pi as proof (quotient multiplied by CRS)
        let mut pi = self.g1.mul(E::ScalarField::ZERO);
        for i in 0..std::cmp::min(quotient.len(), self.crs_g1.len()) {
            pi += self.crs_g1[i] * quotient[i];
        }

        // return pi
        pi
    }

    pub fn multi_open(&self, poly: &[E::ScalarField], points: &[E::ScalarField]) -> E::G1 {
        // denominator is a polynomial where all its root are points to be evaluated (zero poly)
        let mut zero_poly = vec![-points[0], E::ScalarField::ONE];
        for i in 1..points.len() {
            zero_poly = mul(&zero_poly, &[-points[i], E::ScalarField::ONE]);
        }

        // perform Lagrange interpolation on points
        let mut values = vec![];
        for i in 0..points.len() {
            values.push(evaluate(poly, points[i]));
        }
        let mut lagrange_poly = interpolate(points, &values).unwrap();
        lagrange_poly.resize(poly.len(), E::ScalarField::ZERO); // pad with zeros

        // numerator is the difference between the polynomial and the Lagrange interpolation
        let mut numerator = Vec::with_capacity(poly.len());
        for (coeff1, coeff2) in poly.iter().zip(lagrange_poly.as_slice()) {
            numerator.push(*coeff1 - coeff2);
        }

        // get quotient by dividing numerator by denominator
        let quotient = div(&numerator, &zero_poly).unwrap();

        // calculate pi as proof (quotient multiplied by CRS)
        let mut pi = self.g1.mul(E::ScalarField::ZERO);
        for i in 0..std::cmp::min(quotient.len(), self.crs_g1.len()) {
            pi += self.crs_g1[i] * quotient[i];
        }

        // return pi
        pi
    }

    pub fn verify(
        &self,
        point: E::ScalarField,
        value: E::ScalarField,
        commitment: E::G1,
        pi: E::G1
    ) -> bool {
        if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
            println!("KZG Debug - Input parameters:");
            println!("  point: {:?}", point);
            println!("  value: {:?}", value);
            println!("  commitment: {:?}", commitment.to_string());
            println!("  pi: {:?}", pi.to_string());
        }
        
        let lhs_term1 = pi;
        let lhs_term2 = self.g2_tau - self.g2.mul(point);
        let lhs = E::pairing(lhs_term1, lhs_term2);
        
        let rhs_term1 = commitment - self.g1.mul(value);
        let rhs_term2 = self.g2;
        let rhs = E::pairing(rhs_term1, rhs_term2);
        
        if std::env::var("ZKMAP_DEBUG").as_deref() == Ok("1") {
            println!("KZG Debug - Verification components:");
            println!("  LHS term1 (pi): {:?}", lhs_term1.to_string());
            println!("  LHS term2 (g2_tau - g2*point): {:?}", lhs_term2.to_string());
            println!("  RHS term1 (commitment - g1*value): {:?}", rhs_term1.to_string());
            println!("  RHS term2 (g2): {:?}", rhs_term2.to_string());
            println!("  LHS pairing: {:?}", lhs.to_string());
            println!("  RHS pairing: {:?}", rhs.to_string());
            println!("KZG Debug - Final result: {}", lhs == rhs);
        }

        lhs == rhs
    }

    pub fn verify_multi(
        &self,
        points: &[E::ScalarField],
        values: &[E::ScalarField],
        commitment: E::G1,
        pi: E::G1
    ) -> bool {
        // compute the zero polynomial
        let mut zero_poly = vec![-points[0], E::ScalarField::ONE];
        for i in 1..points.len() {
            zero_poly = mul(&zero_poly, &[-points[i], E::ScalarField::ONE]);
        }

        // compute commitment of zero polynomial in regards to crs_g2
        let mut zero_commitment = self.g2.mul(E::ScalarField::ZERO);
        for i in 0..zero_poly.len() {
            zero_commitment += self.crs_g2[i] * zero_poly[i];
        }

        // compute lagrange polynomial
        let lagrange_poly = interpolate(points, &values).unwrap();

        // compute commitment of lagrange polynomial in regards to crs_g1
        let mut lagrange_commitment = self.g1.mul(E::ScalarField::ZERO);
        for i in 0..lagrange_poly.len() {
            lagrange_commitment += self.crs_g1[i] * lagrange_poly[i];
        }

        let lhs = E::pairing(pi, zero_commitment);
        let rhs = E::pairing(commitment - lagrange_commitment, self.g2);
        lhs == rhs
    }
}

