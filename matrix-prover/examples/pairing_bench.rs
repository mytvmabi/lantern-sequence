use ark_bls12_381::{Bls12_381, G1Projective, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ff::UniformRand;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    // deterministic RNG for repeatability
    let mut rng = StdRng::seed_from_u64(42u64);

    // generate random points in G1 and G2
    let g1: G1Projective = G1Projective::rand(&mut rng);
    let g2: G2Projective = G2Projective::rand(&mut rng);

    // warm up
    for _ in 0..10 {
        let _ = Bls12_381::pairing(g1, g2);
    }

    let iterations = 200usize;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = Bls12_381::pairing(g1, g2);
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() / iterations as u128;
    println!("Pairing bench: iterations = {}, total = {:?}, avg = {} ns ({} ms)", iterations, elapsed, avg_ns, avg_ns as f64 / 1_000_000.0);
}
