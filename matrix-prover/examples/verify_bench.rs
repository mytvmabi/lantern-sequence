use zkMaP::{KZG, ZKMatrixProof, BLS12381Pairing, BLS12381Fr};
use ark_bls12_381::{G1Projective, G2Projective};
use ark_std::UniformRand;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn rand_matrix(n: usize, rng: &mut StdRng) -> Vec<Vec<BLS12381Fr>> {
    (0..n).map(|_| (0..n).map(|_| BLS12381Fr::rand(rng)).collect()).collect()
}

fn main() {
    let mut rng = StdRng::seed_from_u64(12345u64);

    // choose matrix size and iterations
    let n = 1024usize; // 1024x1024 matrix
    // keep iterations very small because SRS/setup is large for 1024x1024
    let iterations = 5usize;

    // setup: choose degree large enough for polynomial representation of an n x n matrix
    // polynomial length = n * n, so degree should be at least n*n - 1; choose degree = n*n
    let degree = n * n;
    let g1 = G1Projective::rand(&mut rng);
    let g2 = G2Projective::rand(&mut rng);
    let mut kzg = KZG::<BLS12381Pairing>::new(g1, g2, degree);
    let secret = BLS12381Fr::rand(&mut rng);
    kzg.setup(secret);

    let zk = ZKMatrixProof::new(kzg, degree);

    // prepare matrices and proof
    let a = rand_matrix(n, &mut rng);
    let b = rand_matrix(n, &mut rng);
    let proof = zk.prove_matrix_mult(&a, &b);

    // measure verification time
    let start = Instant::now();
    for _ in 0..iterations {
        let ok = zk.verify(&proof);
        if !ok { panic!("verify failed"); }
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!("Verified {} times, total {:.3}s, avg {:.3} ms", iterations, elapsed.as_secs_f64(), avg_ms);
}
