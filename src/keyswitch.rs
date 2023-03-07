use tfhe::shortint::prelude::*;
use tfhe::core_crypto::prelude::*;
use tfhe::core_crypto::entities::{PlaintextList,GlweCiphertextOwned,LweCiphertextOwned,Polynomial};
use tfhe::core_crypto::commons::math::decomposition::SignedDecomposer;
use tfhe::core_crypto::algorithms::polynomial_algorithms::*;
use tfhe::shortint::engine::ShortintEngine;

pub struct LWEtoGLWEKeyswitchKey {
    // each inner vector is a G(R)Lev ciphertext
    inner: Vec<Vec<GlweCiphertextOwned<u64>>>,
    param: Parameters,
}

impl LWEtoGLWEKeyswitchKey {
    pub fn from_server_key(server_key: &ServerKey) -> Self {
        // conver the FourierGgswCiphertext
        // back into the standard domain and take half of the GGSW
        unimplemented!()
    }

    pub fn from_client_key(client_key: &ClientKey) -> Self {
        let param = client_key.parameters;
        let mut out: Vec<Vec<GlweCiphertextOwned<u64>>> = vec![];
        for elt in client_key.get_lwe_sk_ref().as_ref().iter() {
            // elt is u64
            let glev: Vec<GlweCiphertextOwned<u64>> = (1..=param.ks_level.0).into_iter().map(|level| {
                let shift: usize = (u64::BITS as usize) - param.ks_base_log.0 * level;
                let message = *elt * (1 << shift) as u64;
                let plaintext_list = PlaintextList::from_container({
                     let mut tmp = vec![0u64; param.polynomial_size.0];
                     tmp[0] = message;
                     tmp
                });
                ShortintEngine::with_thread_local_mut(|engine| {
                    engine.unchecked_glwe_encrypt(client_key, &plaintext_list)
                }).unwrap()
            }).collect();
            out.push(glev);
        }
        LWEtoGLWEKeyswitchKey { inner: out, param }
    }
}

fn polynomial_wrapping_add_mul_const_assign<Scalar, OutputCont, InputCont>(
    output: &mut Polynomial<OutputCont>,
    poly: &Polynomial<InputCont>,
    k: Scalar,
) where
    Scalar: UnsignedInteger,
    OutputCont: ContainerMut<Element = Scalar>,
    InputCont: Container<Element = Scalar>,
{
    output.as_mut().iter_mut().zip(poly.iter()).for_each(|(out, p)| {
        *out = out.wrapping_add(p.wrapping_mul(k));
    });
}

fn polynomial_wrapping_sub_mul_const_assign<Scalar, OutputCont, InputCont>(
    output: &mut Polynomial<OutputCont>,
    poly: &Polynomial<InputCont>,
    k: Scalar,
) where
    Scalar: UnsignedInteger,
    OutputCont: ContainerMut<Element = Scalar>,
    InputCont: Container<Element = Scalar>,
{
    output.as_mut().iter_mut().zip(poly.iter()).for_each(|(out, p)| {
        *out = out.wrapping_sub(p.wrapping_mul(k));
    });
}

pub fn lwe_to_glwe_keyswitch(
    ksks: &LWEtoGLWEKeyswitchKey,
    lwe: &LweCiphertextOwned<u64>,
) -> GlweCiphertextOwned<u64> {
    // https://docs.rs/tfhe/latest/src/tfhe/core_crypto/algorithms/lwe_private_functional_packing_keyswitch.rs.html
    // we need the key switching keys
    // which should be the same as the bootstrapping keys
    // i.e., RLWE_s'(s_i), for i \in [n].

    let param = ksks.param;
    let mut out = GlweCiphertextOwned::new(0, param.glwe_dimension.to_glwe_size(), param.polynomial_size);
    for (glev, a) in ksks.inner.iter().zip(lwe.get_mask().as_ref().iter()) {
        // Setup decomposition
        let decomposer = SignedDecomposer::new(param.ks_base_log, param.ks_level);
        let closest = decomposer.closest_representable(*a);
        let decomposer_iter = decomposer.decompose(closest);

        for (ksk, decomposed_a) in glev.iter().zip(decomposer_iter) {
            // c[1] += < g^-1(a_i), ksk_i[1] >
            out.get_mut_mask().as_mut_polynomial_list().iter_mut().for_each(|mut poly| {
                polynomial_wrapping_add_mul_const_assign(
                    &mut poly,
                    &ksk.get_mask().as_polynomial_list().get(0),
                    decomposed_a.value());
            });

            // c[2] -= < g^-1(a_i), ksk_i[2] >
            polynomial_wrapping_sub_mul_const_assign(
                &mut out.get_mut_body().as_mut_polynomial(),
                &ksk.get_body().as_polynomial(),
                decomposed_a.value());
        }
    }

    // c[2] += b
    let b = Polynomial::from_container({
        let mut v = vec![0u64; param.polynomial_size.0];
        v[0] = *lwe.get_body().0;
        v
    });
    polynomial_wrapping_add_assign(&mut out.get_mut_body().as_mut_polynomial(), &b);
    out
}

#[cfg(test)]
mod test {
    use super::*;
    use tfhe::shortint::parameters::PARAM_MESSAGE_3_CARRY_0;
    use tfhe::core_crypto::algorithms::glwe_encryption::decrypt_glwe_ciphertext;

    #[test]
    fn test_lwe_to_glwe() {
        let (client_key, server_key) = gen_keys(PARAM_MESSAGE_3_CARRY_0);
        let ksk = LWEtoGLWEKeyswitchKey::from_client_key(&client_key);
        let ct = client_key.encrypt(1);
        // println!("sk: {:?}", client_key.get_lwe_sk_ref());

        let glwe = lwe_to_glwe_keyswitch(&ksk, &ct.ct);
        let mut out = PlaintextList::new(0u64, PlaintextCount(client_key.parameters.polynomial_size.0));
        decrypt_glwe_ciphertext(client_key.get_glwe_sk_ref(), &glwe, &mut out);
        println!("pt: {:?}", out);
    }
}
