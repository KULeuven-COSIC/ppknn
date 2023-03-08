use tfhe::core_crypto::algorithms::polynomial_algorithms::*;
use tfhe::core_crypto::commons::math::decomposition::SignedDecomposer;
use tfhe::core_crypto::entities::{
    GlweCiphertextOwned, LweCiphertextOwned, PlaintextList, Polynomial,
};
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::prelude::Parameters;
use crate::context::Context;

pub struct LWEtoGLWEKeyswitchKey {
    // each inner vector is a G(R)Lev ciphertext
    inner: Vec<Vec<GlweCiphertextOwned<u64>>>,
    params: Parameters,
}

impl LWEtoGLWEKeyswitchKey {
    /*
    pub fn from_server_key(server_key: &ServerKey) -> Self {
        // conver the FourierGgswCiphertext
        // back into the standard domain and take half of the GGSW
        unimplemented!()
    }

    pub fn from_client_key(client_key: &ClientKey) -> Self {
        let params = client_key.parameters;
        let mut out: Vec<Vec<GlweCiphertextOwned<u64>>> = vec![];
        for elt in client_key.get_lwe_sk_ref().as_ref().iter() {
            // elt is &u64
            let glev: Vec<GlweCiphertextOwned<u64>> = (1..=params.ks_level.0)
                .into_iter()
                .map(|level| {
                    let shift: usize = (u64::BITS as usize) - params.ks_base_log.0 * level;
                    let message = (1 << shift) * *elt as u64;
                    let plaintext_list = PlaintextList::from_container({
                        let mut tmp = vec![0u64; params.polynomial_size.0];
                        tmp[0] = message;
                        tmp
                    });
                    // allocate_and_trivially_encrypt_new_glwe_ciphertext(param.glwe_dimension.to_glwe_size(), &plaintext_list)
                    ShortintEngine::with_thread_local_mut(|engine| {
                        engine.unchecked_glwe_encrypt(client_key, &plaintext_list)
                    })
                    .unwrap()
                })
                .collect();
            out.push(glev);
        }
        LWEtoGLWEKeyswitchKey { inner: out, params }
    }
     */

    pub fn from_lwe_glwe_sk(lwe_sk: &LweSecretKeyOwned<u64>, glwe_sk: &GlweSecretKeyOwned<u64>, ctx: &mut Context) -> Self {
        let params = ctx.params;
        let mut out: Vec<Vec<GlweCiphertextOwned<u64>>> = vec![];
        for elt in lwe_sk.as_ref().iter() {
            // elt is &u64
            let glev: Vec<GlweCiphertextOwned<u64>> = (1..=params.ks_level.0)
                .into_iter()
                .map(|level| {
                    let shift: usize = (u64::BITS as usize) - params.ks_base_log.0 * level;
                    let message = (1 << shift) * *elt as u64;
                    let plaintext_list = PlaintextList::from_container({
                        let mut tmp = vec![0u64; params.polynomial_size.0];
                        tmp[0] = message;
                        tmp
                    });
                    allocate_and_trivially_encrypt_new_glwe_ciphertext(params.glwe_dimension.to_glwe_size(), &plaintext_list)
                        /*
                    let mut glwe = GlweCiphertext::new(
                        0u64,
                        params.glwe_dimension.to_glwe_size(),
                        params.polynomial_size);
                    encrypt_glwe_ciphertext(
                        &glwe_sk,
                        &mut glwe,
                        &plaintext_list,
                        params.glwe_modular_std_dev,
                        &mut ctx.encryption_rng,
                    );
                    glwe
                         */
                })
                .collect();
            out.push(glev);
        }
        LWEtoGLWEKeyswitchKey { inner: out, params }
    }
}

pub(crate) fn polynomial_wrapping_add_mul_const_assign<Scalar, OutputCont, InputCont>(
    output: &mut Polynomial<OutputCont>,
    poly: &Polynomial<InputCont>,
    k: Scalar,
) where
    Scalar: UnsignedInteger,
    OutputCont: ContainerMut<Element = Scalar>,
    InputCont: Container<Element = Scalar>,
{
    assert_eq!(output.polynomial_size(), poly.polynomial_size());
    output
        .as_mut()
        .iter_mut()
        .zip(poly.iter())
        .for_each(|(out, p)| {
            *out = out.wrapping_add(p.wrapping_mul(k));
        });
}

pub(crate) fn polynomial_wrapping_sub_mul_const_assign<Scalar, OutputCont, InputCont>(
    output: &mut Polynomial<OutputCont>,
    poly: &Polynomial<InputCont>,
    k: Scalar,
) where
    Scalar: UnsignedInteger,
    OutputCont: ContainerMut<Element = Scalar>,
    InputCont: Container<Element = Scalar>,
{
    assert_eq!(output.polynomial_size(), poly.polynomial_size());
    output
        .as_mut()
        .iter_mut()
        .zip(poly.iter())
        .for_each(|(out, p)| {
            *out = out.wrapping_sub(p.wrapping_mul(k));
        });
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
        0,
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
            out.get_mut_mask()
                .as_mut_polynomial_list()
                .iter_mut()
                .for_each(|mut poly| {
                    polynomial_wrapping_add_mul_const_assign(
                        &mut poly,
                        &ksk.get_mask().as_polynomial_list().get(0),
                        decomposed_a.value(),
                    );
                });

            // c[2] -= < g^-1(a_i), ksk_i[2] >
            polynomial_wrapping_sub_mul_const_assign(
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
    use crate::context::*;
    use tfhe::core_crypto::algorithms::glwe_encryption::decrypt_glwe_ciphertext;
    use tfhe::shortint::parameters::PARAM_MESSAGE_1_CARRY_0;

    #[test]
    fn test_poly_arith() {
        let n = 10usize;
        {
            let mut out = Polynomial::new(1u64, PolynomialSize(n));
            let poly = Polynomial::new(2u64, PolynomialSize(n));
            polynomial_wrapping_add_mul_const_assign(&mut out, &poly, 3u64);
            assert_eq!(vec![7u64; n], out.into_container());
        }
        {
            let mut out = Polynomial::new(8u64, PolynomialSize(n));
            let poly = Polynomial::new(3u64, PolynomialSize(n));
            polynomial_wrapping_sub_mul_const_assign(&mut out, &poly, 3u64);
            assert_eq!(vec![u64::MAX; n], out.into_container());
        }
    }

    #[test]
    fn test_lwe_to_glwe() {
        // let (client_key, server_key) = gen_keys(PARAM_MESSAGE_2_CARRY_0);
        let mut ctx = Context::new(PARAM_MESSAGE_1_CARRY_0);
        let lwe_sk = ctx.gen_lwe_sk();
        let glwe_sk = ctx.gen_glwe_sk();
        let ksk = LWEtoGLWEKeyswitchKey::from_lwe_glwe_sk(&lwe_sk, &glwe_sk, &mut ctx);
        let ct = lwe_encode_encrypt(&lwe_sk, &mut ctx, 0);
        assert_eq!(ksk.inner.len(), lwe_sk.as_ref().len());

        let glwe = lwe_to_glwe_keyswitch(&ksk, &ct);
        let mut out = PlaintextList::new(
            0u64,
            PlaintextCount(ctx.params.polynomial_size.0),
        );
        decrypt_glwe_ciphertext(&glwe_sk, &glwe, &mut out);
        out.as_mut().iter_mut().for_each(|x| {
            ctx.codec.decode(x);
        });
        println!("pt: {:?}", out);
    }

    /*
    #[test]
    fn test_lwe_to_glwe2() {
        let mut ctx = Context::new(PARAM_MESSAGE_2_CARRY_0);
        let lwe_sk = ctx.gen_lwe_sk();
        let glwe_sk = ctx.gen_glwe_sk();

        let mut pfpksk = LwePrivateFunctionalPackingKeyswitchKey::new(
            0,
            ctx.params.pfks_base_log,
            ctx.params.pfks_level,
            ctx.params.lwe_dimension,
            ctx.params.glwe_dimension.to_glwe_size(),
            ctx.params.polynomial_size,
        );

        let mut last_polynomial = Polynomial::new(0, ctx.params.polynomial_size);
        last_polynomial[0] = u64::MAX;

        par_generate_lwe_private_functional_packing_keyswitch_key(
            &lwe_sk,
            &glwe_sk,
            &mut pfpksk,
            ctx.params.pfks_modular_std_dev,
            &mut ctx.encryption_rng,
            |x| x,
            &last_polynomial,
        );

        let mut output_glwe = GlweCiphertext::new(
            0,
            ctx.params.glwe_dimension.to_glwe_size(),
            ctx.params.polynomial_size,
        );
    }
     */
}
