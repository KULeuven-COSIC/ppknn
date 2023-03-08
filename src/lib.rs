pub mod batcher;
pub mod codec;
pub mod comparator;
pub mod context;
pub mod keyswitch;

pub use batcher::*;
pub use comparator::*;

use crate::context::{lwe_decrypt_decode, lwe_encode_encrypt, Context};
use std::fs;
use std::io::Cursor;
use tfhe::core_crypto::prelude::polynomial_algorithms::{
    polynomial_wrapping_add_assign, polynomial_wrapping_mul,
};
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::ciphertext::Degree;
use tfhe::shortint::prelude::*;
use tfhe::shortint::server_key::Accumulator;

const DUMMY_KEY: &str = "dummy_key";

pub fn read_or_gen_keys(param: Parameters) -> (ClientKey, ServerKey) {
    match fs::read(DUMMY_KEY) {
        Ok(s) => {
            let mut serialized_data = Cursor::new(&s);
            let client_key: ClientKey = bincode::deserialize_from(&mut serialized_data).unwrap();
            let server_key: ServerKey = bincode::deserialize_from(&mut serialized_data).unwrap();
            assert_eq!(client_key.parameters, param);
            (client_key, server_key)
        }
        _ => {
            let (client_key, server_key) = gen_keys(param);
            let mut serialized_data = Vec::new();
            bincode::serialize_into(&mut serialized_data, &client_key).unwrap();
            bincode::serialize_into(&mut serialized_data, &server_key).unwrap();
            fs::write(DUMMY_KEY, serialized_data).expect("unable to write to file");
            (client_key, server_key)
        }
    }
}

pub fn enc_vec(vs: &[(u64, u64)], client_key: &ClientKey) -> Vec<EncItem> {
    vs.iter()
        .map(|v| EncItem::new(client_key.encrypt(v.0), client_key.encrypt(v.1)))
        .collect()
}

pub struct KnnServer {
    key: ServerKey,
    lwe_to_glwe_ksk: LwePrivateFunctionalPackingKeyswitchKeyOwned<u64>,
    params: Parameters,
}

impl KnnServer {
    pub fn lwe_to_glwe(&self, lwe: &LweCiphertextOwned<u64>) -> GlweCiphertextOwned<u64> {
        let mut output_glwe = GlweCiphertext::new(
            0,
            self.params.glwe_dimension.to_glwe_size(),
            self.params.polynomial_size,
        );

        private_functional_keyswitch_lwe_ciphertext_into_glwe_ciphertext(
            &self.lwe_to_glwe_ksk,
            &mut output_glwe,
            &lwe,
        );

        output_glwe
    }

    pub(crate) fn polynomial_glwe_mul(
        &self,
        glwe: &GlweCiphertextOwned<u64>,
        poly: &PolynomialOwned<u64>,
    ) -> GlweCiphertextOwned<u64> {
        let mut out = GlweCiphertextOwned::new(
            0u64,
            self.params.glwe_dimension.to_glwe_size(),
            self.params.polynomial_size,
        );
        out.get_mut_mask()
            .as_mut_polynomial_list()
            .iter_mut()
            .for_each(|mut mask| {
                polynomial_wrapping_mul(
                    &mut mask,
                    &glwe.get_mask().as_polynomial_list().get(0),
                    &poly,
                );
            });
        polynomial_wrapping_mul(
            &mut out.get_mut_body().as_mut_polynomial(),
            &glwe.get_body().as_polynomial(),
            &poly,
        );
        out
    }

    pub fn double_ct_acc(
        &self,
        left_lwe: &LweCiphertextOwned<u64>,
        right_lwe: &LweCiphertextOwned<u64>,
    ) -> Accumulator {
        // first key switch the LWE ciphertexts to GLWE
        let left_glwe = self.lwe_to_glwe(&left_lwe);
        let right_glwe = self.lwe_to_glwe(&right_lwe);

        let half_n = self.params.polynomial_size.0 / 2;
        // left polynomial has the form X^0 + ... + X^{N/2-1}
        let left_poly = Polynomial::from_container({
            vec![1u64; half_n]
                .into_iter()
                .chain(vec![0u64; half_n])
                .collect::<Vec<_>>()
        });
        // right polynomial has the form X^{N/2} + ... + X^{N-1}
        let right_poly = Polynomial::from_container({
            vec![0u64; half_n]
                .into_iter()
                .chain(vec![1u64; half_n])
                .collect::<Vec<_>>()
        });

        // create the two halves of the accumulator
        let mut left_acc = self.polynomial_glwe_mul(&left_glwe, &left_poly);
        let right_acc = self.polynomial_glwe_mul(&right_glwe, &right_poly);

        // sum the two halves into the left one
        left_acc
            .as_mut_polynomial_list()
            .iter_mut()
            .zip(right_acc.as_polynomial_list().iter())
            .for_each(|(mut left, right)| polynomial_wrapping_add_assign(&mut left, &right));

        Accumulator {
            acc: left_acc,
            degree: Degree(10),
        }
    }
}

pub struct KnnClient {
    key: ClientKey,
    ctx: Context,
}

impl KnnClient {
    pub fn lwe_encode_encrypt(&mut self, x: u64) -> LweCiphertextOwned<u64> {
        lwe_encode_encrypt(&self.key.get_lwe_sk_ref(), &mut self.ctx, x)
    }

    pub fn lwe_decrypt_decode(&self, lwe: &LweCiphertextOwned<u64>) -> u64 {
        lwe_decrypt_decode(&self.key.get_lwe_sk_ref(), &self.ctx, lwe)
    }
}

pub fn setup(params: Parameters) -> (KnnServer, KnnClient) {
    let mut ctx = Context::new(params);
    let (client_key, server_key) = gen_keys(params);
    let lwe_to_glwe_ksk = ctx.gen_ksk(client_key.get_lwe_sk_ref(), client_key.get_glwe_sk_ref());
    (
        KnnServer {
            key: server_key,
            lwe_to_glwe_ksk,
            params,
        },
        KnnClient {
            key: client_key,
            ctx,
        },
    )
}

#[cfg(test)]
mod test {
    use super::*;
    use tfhe::shortint::ciphertext::Degree;
    use tfhe::shortint::server_key::Accumulator;

    pub(crate) const TEST_PARAM: Parameters = Parameters {
        lwe_dimension: LweDimension(742),
        glwe_dimension: GlweDimension(1),
        polynomial_size: PolynomialSize(2048),
        lwe_modular_std_dev: StandardDev(0.000007069849454709433),
        glwe_modular_std_dev: StandardDev(0.00000000000000029403601535432533),
        pbs_base_log: DecompositionBaseLog(23),
        pbs_level: DecompositionLevelCount(1),
        ks_level: DecompositionLevelCount(15),
        ks_base_log: DecompositionBaseLog(2),
        pfks_level: DecompositionLevelCount(15),
        pfks_base_log: DecompositionBaseLog(2),
        pfks_modular_std_dev: StandardDev(0.00000000000000029403601535432533),
        cbs_level: DecompositionLevelCount(0),
        cbs_base_log: DecompositionBaseLog(0),
        message_modulus: MessageModulus(4),
        carry_modulus: CarryModulus(1),
    };

    #[test]
    fn test_custom_accumulator() {
        // setup a truth table that always returns 1
        // then using PBS we should always get 1
        let (server, mut client) = setup(TEST_PARAM);

        let pt = 3u64;
        let ct_before = client.lwe_encode_encrypt(pt);
        let ct_after = server.lwe_to_glwe(&ct_before);

        {
            // test the key switching
            let mut output_plaintext =
                PlaintextList::new(0, PlaintextCount(server.params.polynomial_size.0));
            decrypt_glwe_ciphertext(
                &client.key.get_glwe_sk_ref(),
                &ct_after,
                &mut output_plaintext,
            );
            output_plaintext.iter_mut().for_each(|mut x| {
                client.ctx.codec.decode(&mut x.0);
            });

            let expected = PlaintextList::from_container({
                let mut tmp = vec![0u64; server.params.polynomial_size.0];
                tmp[0] = pt;
                tmp
            });
            assert_eq!(output_plaintext, expected);
        }

        // we need to set the accumulator to be: ct_after * (X^0 + ... + X^{N-1})
        // where ct_after is an encryption of `pt`
        let poly_ones = Polynomial::from_container(vec![1u64; server.params.polynomial_size.0]);
        let glwe_acc = server.polynomial_glwe_mul(&ct_after, &poly_ones);
        let acc = Accumulator {
            acc: glwe_acc,
            degree: Degree(10), // NOTE: degree doesn't seem to matter
        };

        // now we do pbs and the result should always be `pt`
        // doesn't matter what the ct is or the encoding, so we use the shortint encrypt function
        let ct = client.key.encrypt(1);
        let res = server.key.keyswitch_programmable_bootstrap(&ct, &acc);

        assert_eq!(client.lwe_decrypt_decode(&res.ct), pt);
    }

    #[test]
    fn test_enc_sort() {
        {
            let (client_key, server_key) = gen_keys(TEST_PARAM);
            let pt_vec = vec![(1, 1), (0, 0), (2, 2), (3u64, 3u64)];
            let enc_cmp = EncCmp::boxed(
                enc_vec(&pt_vec, &client_key),
                &client_key.parameters,
                server_key,
            );

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let output = sorter.inner()[0].decrypt(&client_key);
            assert_eq!(output, (0, 0));
        }
        {
            let (client_key, server_key) = gen_keys(TEST_PARAM);
            let pt_vec = vec![(2, 2), (2, 2), (1, 1), (3u64, 3u64)];
            let enc_cmp = EncCmp::boxed(
                enc_vec(&pt_vec, &client_key),
                &client_key.parameters,
                server_key,
            );

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let output = sorter.inner()[0].decrypt(&client_key);
            assert_eq!(output, (1, 1));
        }
        {
            let (client_key, server_key) = gen_keys(TEST_PARAM);
            let pt_vec = vec![(1, 1), (2, 2), (3u64, 3u64), (0, 0)];
            let enc_cmp = EncCmp::boxed(
                enc_vec(&pt_vec, &client_key),
                &client_key.parameters,
                server_key,
            );

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let output = sorter.inner()[0].decrypt(&client_key);
            assert_eq!(output, (0, 0));
        }
    }
}
