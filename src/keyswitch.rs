use tfhe::core_crypto::algorithms::polynomial_algorithms::*;
use tfhe::core_crypto::commons::math::decomposition::SignedDecomposer;
use tfhe::core_crypto::entities::{
    GlweCiphertextOwned, LweCiphertextOwned, PlaintextList, Polynomial,
};
use tfhe::core_crypto::prelude::slice_algorithms::{
    slice_wrapping_add_scalar_mul_assign, slice_wrapping_sub_scalar_mul_assign,
};
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::prelude::Parameters;

pub struct LWEtoGLWEKeyswitchKey {
    // each inner vector is a G(R)Lev ciphertext
    inner: Vec<Vec<GlweCiphertextOwned<u64>>>,
    params: Parameters,
}

impl LWEtoGLWEKeyswitchKey {
    pub fn from_lwe_glwe_sk(
        lwe_sk: &LweSecretKeyOwned<u64>,
        glwe_sk: &GlweSecretKeyOwned<u64>,
        params: Parameters,
        encryption_rng: &mut EncryptionRandomGenerator<ActivatedRandomGenerator>,
    ) -> Self {
        let mut out: Vec<Vec<GlweCiphertextOwned<u64>>> = vec![];
        for elt in lwe_sk.as_ref().iter() {
            // elt is &u64
            let glev: Vec<GlweCiphertextOwned<u64>> = (1..=params.ks_level.0)
                .into_iter()
                .map(|level| {
                    assert!(*elt == 0 || *elt == 1);
                    let shift: usize = (u64::BITS as usize) - params.ks_base_log.0 * level;
                    let message = *elt << shift;
                    let plaintext_list = PlaintextList::from_container({
                        let mut tmp = vec![0u64; params.polynomial_size.0];
                        tmp[0] = message;
                        tmp
                    });
                    // allocate_and_trivially_encrypt_new_glwe_ciphertext(params.glwe_dimension.to_glwe_size(), &plaintext_list)
                    let mut glwe = GlweCiphertext::new(
                        0u64,
                        params.glwe_dimension.to_glwe_size(),
                        params.polynomial_size,
                    );
                    encrypt_glwe_ciphertext(
                        &glwe_sk,
                        &mut glwe,
                        &plaintext_list,
                        params.glwe_modular_std_dev,
                        encryption_rng,
                    );
                    glwe
                })
                .collect();
            out.push(glev);
        }
        LWEtoGLWEKeyswitchKey { inner: out, params }
    }
}

pub fn lwe_to_glwe_keyswitch(
    ksks: &LWEtoGLWEKeyswitchKey,
    lwe: &LweCiphertextOwned<u64>,
) -> GlweCiphertextOwned<u64> {
    // we need the key switching keys
    // which should be the same as the bootstrapping keys
    // i.e., RLWE_s'(s_i), for i \in [n].

    let params = ksks.params;
    let mut out = GlweCiphertextOwned::new(
        0u64,
        params.glwe_dimension.to_glwe_size(),
        params.polynomial_size,
    );
    let decomposer = SignedDecomposer::new(params.ks_base_log, params.ks_level);
    for (glev, a) in ksks.inner.iter().zip(lwe.get_mask().as_ref().iter()) {
        // Setup decomposition, the iterator goes from high level to low level
        let closest = decomposer.closest_representable(*a);
        let decomposer_iter = decomposer.decompose(closest);

        // since the decomposer is "reversed", we need to reverse the order of glev
        for (ksk, decomposed_a) in glev.iter().rev().zip(decomposer_iter) {
            // c[1] += < g^-1(a_i), ksk_i[1] >
            assert_eq!(out.get_mask().as_polynomial_list().polynomial_count().0, 1);
            out.get_mut_mask()
                .as_mut_polynomial_list()
                .iter_mut()
                .for_each(|mut c1| {
                    slice_wrapping_add_scalar_mul_assign(
                        // polynomial_wrapping_add_mul_const_assign(
                        &mut c1,
                        &ksk.get_mask().as_polynomial_list().get(0),
                        decomposed_a.value(),
                    );
                });

            // c[2] -= < g^-1(a_i), ksk_i[2] >
            slice_wrapping_sub_scalar_mul_assign(
                // polynomial_wrapping_sub_mul_const_assign(
                &mut out.get_mut_body().as_mut_polynomial(),
                &ksk.get_body().as_polynomial(),
                decomposed_a.value(),
            );
        }
    }

    // c[2] += b
    let b = Polynomial::from_container({
        let mut v = vec![0u64; params.polynomial_size.0];
        v[0] = *lwe.get_body().0;
        v
    });
    polynomial_wrapping_add_assign(&mut out.get_mut_body().as_mut_polynomial(), &b);
    out
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::TEST_PARAM;
    use crate::{decode, gen_glwe_sk, gen_lwe_sk};
    use tfhe::core_crypto::algorithms::glwe_encryption::decrypt_glwe_ciphertext;
    use tfhe::shortint::gen_keys;

    #[test]
    fn test_decomposition() {
        let params = TEST_PARAM;
        // test < g^{-1}(a), (s * g^0, ..., s * g^{l-1} > ~= a*s >
        let a = (1 << 40) + (1 << 42) + (1 << 50);
        let s = 3u64;
        let decomposer = SignedDecomposer::new(params.ks_base_log, params.ks_level);
        let closest = decomposer.closest_representable(a);
        let decomposer_iter = decomposer.decompose(closest);
        let out = (1..=params.ks_level.0)
            .into_iter()
            .rev()
            .zip(decomposer_iter)
            .map(|(level, term)| {
                assert_eq!(term.level().0, level);
                let shift: usize = (u64::BITS as usize) - params.ks_base_log.0 * level;
                println!("value={}, shift={}", term.value(), shift);
                (term.value() << shift) * s
            })
            .sum();
        assert_eq!(closest * s, out);
    }

    #[test]
    #[ignore]
    fn test_lwe_to_glwe() {
        let params = TEST_PARAM;
        let (client, _) = gen_keys(params);
        let mut seeder = new_seeder();
        let mut secret_rng = SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
        let mut encryption_rng = EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(
            seeder.seed(),
            seeder.as_mut(),
        );

        let lwe_sk = gen_lwe_sk(params, &mut secret_rng);
        let glwe_sk = gen_glwe_sk(params, &mut secret_rng);
        let ksk =
            LWEtoGLWEKeyswitchKey::from_lwe_glwe_sk(&lwe_sk, &glwe_sk, params, &mut encryption_rng);
        let m = 1u64;
        let ct_before = client.encrypt(m);
        assert_eq!(ksk.inner.len(), lwe_sk.as_ref().len());

        let ct_after = lwe_to_glwe_keyswitch(&ksk, &ct_before.ct);
        let mut out = PlaintextList::new(0u64, PlaintextCount(params.polynomial_size.0));
        decrypt_glwe_ciphertext(&glwe_sk, &ct_after, &mut out);
        out.as_mut()
            .iter_mut()
            .for_each(|x| *x = decode(params, *x));

        assert_eq!(
            out,
            PlaintextList::from_container({
                let mut tmp = vec![0u64; params.polynomial_size.0];
                tmp[0] = m;
                tmp
            })
        );
    }

    #[test]
    fn test_functional_keyswitch() {
        let params = TEST_PARAM;
        let (client, _) = gen_keys(params);
        let mut seeder = new_seeder();
        let mut encryption_rng = EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(
            seeder.seed(),
            seeder.as_mut(),
        );

        let mut pfpksk = LwePrivateFunctionalPackingKeyswitchKey::new(
            0,
            params.pfks_base_log,
            params.pfks_level,
            client.get_lwe_sk_ref().lwe_dimension(),
            params.glwe_dimension.to_glwe_size(),
            params.polynomial_size,
        );

        let mut last_polynomial = Polynomial::new(0, params.polynomial_size);
        last_polynomial[0] = u64::MAX;

        // setup the plaintext and encrypt
        let delta = (1_u64 << 63) / (params.message_modulus.0 * params.carry_modulus.0) as u64;
        let m = 1u64;
        let encoded_m = m * delta;
        let plaintext_list = PlaintextList::from_container(vec![encoded_m]);
        let mut lwe_ciphertext_list = LweCiphertextList::new(
            0u64,
            client.get_lwe_sk_ref().lwe_dimension().to_lwe_size(),
            LweCiphertextCount(plaintext_list.plaintext_count().0),
        );
        encrypt_lwe_ciphertext_list(
            client.get_lwe_sk_ref(),
            &mut lwe_ciphertext_list,
            &plaintext_list,
            params.lwe_modular_std_dev,
            &mut encryption_rng,
        );

        // generate ksk
        // we don't use f: x -> x.wrapping_neg(), then we need wrapping_neg during decoding
        // let shift = 1u64;
        par_generate_lwe_private_functional_packing_keyswitch_key(
            client.get_lwe_sk_ref(),
            client.get_glwe_sk_ref(),
            &mut pfpksk,
            params.pfks_modular_std_dev,
            &mut encryption_rng,
            |x| (x * 2).wrapping_neg(),
            &last_polynomial,
        );

        let mut output_glwe = GlweCiphertext::new(
            0,
            params.glwe_dimension.to_glwe_size(),
            params.polynomial_size,
        );

        // NOTE: what if we try `private_functional_keyswitch_lwe_ciphertext_into_glwe_ciphertext`?
        private_functional_keyswitch_lwe_ciphertext_list_and_pack_in_glwe_ciphertext(
            &pfpksk,
            &mut output_glwe,
            &lwe_ciphertext_list,
        );

        let mut output_plaintext = PlaintextList::new(0, PlaintextCount(params.polynomial_size.0));
        decrypt_glwe_ciphertext(
            client.get_glwe_sk_ref(),
            &output_glwe,
            &mut output_plaintext,
        );
        output_plaintext.iter_mut().for_each(|x| {
            *x.0 = decode(params, *x.0)
            // *x.0 = x.0.wrapping_neg() % ctx.params.message_modulus.0 as u64;
        });

        let expected = PlaintextList::from_container({
            let mut tmp = vec![0u64; params.polynomial_size.0];
            tmp[0] = m * 2;
            tmp
        });
        assert_eq!(output_plaintext, expected);
    }
}
