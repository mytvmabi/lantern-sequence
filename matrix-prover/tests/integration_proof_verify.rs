use zkMaP::{KZG, ZKMatrixProof, BLS12381Pairing, BLS12381Fr, BLS12381G1, BLS12381G2};
use ark_std::UniformRand;
use ark_bls12_381::{G1Projective, G2Projective};
use rand::rngs::StdRng;
use rand::SeedableRng;

// Simple integration test: generate SRS, create ZKMatrixProof, prove and verify for multiple sizes
#[test]
fn test_prove_and_verify_various_sizes() {
    // deterministic rng for tests
    let mut rng = StdRng::seed_from_u64(42u64);

    // small helper to create random matrix of size n x n
    fn rand_matrix(n: usize, rng: &mut StdRng) -> Vec<Vec<BLS12381Fr>> {
        (0..n).map(|_| (0..n).map(|_| BLS12381Fr::rand(rng)).collect()).collect()
    }

    // Build a small SRS - keep degree modest to speed tests
    let degree = 1usize << 10; // 1024
    let g1 = G1Projective::rand(&mut rng);
    let g2 = G2Projective::rand(&mut rng);
    let mut kzg = KZG::<BLS12381Pairing>::new(g1, g2, degree);
    let secret = BLS12381Fr::rand(&mut rng);
    kzg.setup(secret);

    let zk = ZKMatrixProof::new(kzg, degree);

    for &n in &[1usize, 2, 8, 16] {
        let a = rand_matrix(n, &mut rng);
        let b = rand_matrix(n, &mut rng);

        let proof = zk.prove_matrix_mult(&a, &b);
        let ok = zk.verify(&proof);
        assert!(ok, "Proof failed to verify for size {}", n);
    }
}
