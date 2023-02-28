use tfhe::shortint::prelude::*;
use tfhe::core_crypto::entities::{GlweCiphertextOwned,Polynomial};
use tfhe::core_crypto::commons::math::decomposition::SignedDecomposer;
use tfhe::core_crypto::commons::traits::contiguous_entity_container::*;
use tfhe::core_crypto::algorithms::polynomial_algorithms::*;

pub struct LWEtoGLWEKeyswitchKey {
    // each inner vector is a G(R)Lev ciphertext
    inner: Vec<Vec<GlweCiphertextOwned<u64>>>,
}

impl LWEtoGLWEKeyswitchKey {
    pub fn from_server_key(server_key: &ServerKey) -> Self {
        // conver the FourierGgswCiphertext
        // back into the standard domain and take half of the GGSW
        unimplemented!()
    }
}

pub fn lwe_to_glwe_keyswitch(
    ksks: &LWEtoGLWEKeyswitchKey,
    lwe: &Ciphertext,
    param: &Parameters,
) -> GlweCiphertextOwned<u64> {
    // https://docs.rs/tfhe/latest/src/tfhe/core_crypto/algorithms/lwe_private_functional_packing_keyswitch.rs.html
    // we need the key switching keys
    // which should be the same as the bootstrapping keys
    // i.e., RLWE_s'(s_i), for i \in [n].

    let mut out = GlweCiphertextOwned::new(0, param.glwe_dimension.to_glwe_size(), param.polynomial_size);
    for (glev, a) in ksks.inner.iter().zip(lwe.ct.get_mask().as_ref().iter()) {
        // Setup decomposition
        let decomposer = SignedDecomposer::new(param.ks_base_log, param.ks_level);
        let closest = decomposer.closest_representable(*a);
        let decomposer_iter = decomposer.decompose(closest);

        for (ksk, decomposed_a) in glev.iter().zip(decomposer_iter) {
            // c[1] += < g^-1(a_i), ksk_i[1] >
            let decomposed_a_poly = Polynomial::from_container(vec![decomposed_a.value()]);
            out.get_mut_mask().as_mut_polynomial_list().iter_mut().for_each(|mut poly| {
                polynomial_wrapping_add_mul_assign(
                    &mut poly,
                    &ksk.get_mask().as_polynomial_list().get(0),
                    &decomposed_a_poly);
            });

            // c[2] -= < g^-1(a_i), ksk_i[2] >
            polynomial_wrapping_sub_mul_assign(
                &mut out.get_mut_body().as_mut_polynomial(),
                &ksk.get_body().as_polynomial(),
                &decomposed_a_poly);
        }
    }

    // c[2] += b
    let b = Polynomial::from_container(vec![*lwe.ct.get_body().0]);
    polynomial_wrapping_add_assign(&mut out.get_mut_body().as_mut_polynomial(), &b);
    out
}


#[cfg(test)]
mod test {
    use super::*;
    use tfhe::shortint::parameters::PARAM_MESSAGE_3_CARRY_0;
    use tfhe::shortint::engine::ShortintEngine;

    #[test]
    fn test_lwe_to_glwe() {
    }
}
