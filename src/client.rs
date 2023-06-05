use tfhe::core_crypto::algorithms::*;
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::ciphertext::Degree;
use tfhe::shortint::prelude::*;

pub fn gen_lwe_sk(
    params: Parameters,
    secret_rng: &mut SecretRandomGenerator<ActivatedRandomGenerator>,
) -> LweSecretKeyOwned<u64> {
    let lwe_sk = allocate_and_generate_new_binary_lwe_secret_key(params.lwe_dimension, secret_rng);
    lwe_sk
}

pub fn gen_glwe_sk(
    params: Parameters,
    secret_rng: &mut SecretRandomGenerator<ActivatedRandomGenerator>,
) -> GlweSecretKeyOwned<u64> {
    let glwe_sk = allocate_and_generate_new_binary_glwe_secret_key(
        params.glwe_dimension,
        params.polynomial_size,
        secret_rng,
    );
    glwe_sk
}

pub struct KnnClient {
    pub key: ClientKey,
    pub params: Parameters,
    pub encryption_rng: EncryptionRandomGenerator<ActivatedRandomGenerator>,
    pub dist_delta: u64,
}

impl KnnClient {
    pub fn lwe_encrypt_with_delta(&mut self, x: u64, delta: u64) -> Ciphertext {
        let sk = self.key.get_lwe_sk_ref();
        let pt = Plaintext(x * delta);
        let ct = allocate_and_encrypt_new_lwe_ciphertext(
            sk,
            pt,
            self.params.lwe_modular_std_dev,
            &mut self.encryption_rng,
        );
        Ciphertext {
            ct,
            degree: Degree(self.params.message_modulus.0 - 1),
            message_modulus: self.params.message_modulus,
            carry_modulus: self.params.carry_modulus,
        }
    }

    pub fn lwe_encrypt_with_modulus(&mut self, x: u64, modulus: usize) -> Ciphertext {
        // we still consider the padding bit
        let delta = (1u64 << 63) / (modulus * self.params.carry_modulus.0) as u64;
        self.lwe_encrypt_with_delta(x, delta)
    }

    pub fn lwe_noise(&self, ct: &Ciphertext, expected_pt: u64) -> f64 {
        // pt = b - a*s = Delta*m + e
        let mut pt = decrypt_lwe_ciphertext(&self.key.get_lwe_sk_ref(), &ct.ct);

        // pt = pt - Delta*m = e (encoded_ptxt is Delta*m)
        let delta = self.delta();

        pt.0 = pt.0.wrapping_sub(delta * expected_pt);

        ((pt.0 as i64).abs() as f64).log2()
    }

    pub fn delta(&self) -> u64 {
        let delta =
            (1u64 << 63) / (self.params.message_modulus.0 * self.params.carry_modulus.0) as u64;
        delta
    }

    /// Create a query for a given `target`.
    /// The client needs to be mutable because we mutate the encryption RNG.
    pub fn make_query(&mut self, target: &[u64]) -> (GlweCiphertextOwned<u64>, Ciphertext) {
        let gamma = target.len();
        let n = self.params.polynomial_size.0;
        let padding = vec![0u64; n - gamma];
        let delta = self.dist_delta;
        assert!(gamma < n);

        // \sum_{i=0}^{\gamma - 1} c_i * X^i
        let pt = PlaintextList::from_container({
            let mut container = vec![];
            container.extend_from_slice(target);
            container.extend_from_slice(&padding);

            container.iter_mut().for_each(|x| {
                *x = *x * delta;
            });
            container
        });

        // X^{\gamma - 1} * (\sum_{i = 0}^{\gamma - 1} c_i^2)
        let pt2 = target.iter().map(|x| x.wrapping_mul(*x)).sum();

        // now encrypt the two plaintexts
        let mut c = GlweCiphertext::new(
            0u64,
            self.params.glwe_dimension.to_glwe_size(),
            self.params.polynomial_size,
        );
        encrypt_glwe_ciphertext(
            self.key.get_glwe_sk_ref(),
            &mut c,
            &pt,
            self.params.glwe_modular_std_dev,
            &mut self.encryption_rng,
        );
        let c2 = self.lwe_encrypt_with_delta(pt2, delta);
        (c, c2)
    }
}

/// Generate the key switching keys for `lwe_to_glwe`.
pub fn gen_ksk(
    sk: &ClientKey,
    encryption_rng: &mut EncryptionRandomGenerator<ActivatedRandomGenerator>,
) -> LwePrivateFunctionalPackingKeyswitchKeyOwned<u64> {
    let lwe_sk = sk.get_lwe_sk_ref();
    let glwe_sk = sk.get_glwe_sk_ref();
    let mut pfpksk = LwePrivateFunctionalPackingKeyswitchKey::new(
        0,
        sk.parameters.pfks_base_log,
        sk.parameters.pfks_level,
        lwe_sk.lwe_dimension(),
        sk.parameters.glwe_dimension.to_glwe_size(),
        sk.parameters.polynomial_size,
    );

    let mut last_polynomial = Polynomial::new(0, sk.parameters.polynomial_size);
    last_polynomial[0] = u64::MAX;

    par_generate_lwe_private_functional_packing_keyswitch_key(
        &lwe_sk,
        &glwe_sk,
        &mut pfpksk,
        sk.parameters.pfks_modular_std_dev,
        encryption_rng,
        |x| x.wrapping_neg(),
        &last_polynomial,
    );
    pfpksk
}
