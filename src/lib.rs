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
    pub fn lwe_to_glwe(&self, ct: &Ciphertext) -> GlweCiphertextOwned<u64> {
        let mut output_glwe = GlweCiphertext::new(
            0,
            self.params.glwe_dimension.to_glwe_size(),
            self.params.polynomial_size,
        );

        private_functional_keyswitch_lwe_ciphertext_into_glwe_ciphertext(
            &self.lwe_to_glwe_ksk,
            &mut output_glwe,
            &ct.ct,
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

    pub fn double_ct_acc(&self, left_lwe: &Ciphertext, right_lwe: &Ciphertext) -> Accumulator {
        // first key switch the LWE ciphertexts to GLWE
        let left_glwe = self.lwe_to_glwe(&left_lwe);
        let right_glwe = self.lwe_to_glwe(&right_lwe);

        let half_n = self.params.polynomial_size.0 / 2;
        let chunk_size = self.params.polynomial_size.0 / self.params.message_modulus.0;

        // left polynomial has the form X^0 + ... + X^{N/2-1}
        let left_poly = Polynomial::from_container({
            let mut tmp = vec![1u64; half_n]
                .into_iter()
                .chain(vec![0u64; half_n])
                .collect::<Vec<_>>();
            for a_i in tmp[0..chunk_size / 2].iter_mut() {
                *a_i = (*a_i).wrapping_neg();
            }
            tmp.rotate_left(chunk_size / 2);
            tmp
        });

        // right polynomial has the form X^{N/2} + ... + X^{N-1}
        let right_poly = Polynomial::from_container({
            let mut tmp = vec![0u64; half_n]
                .into_iter()
                .chain(vec![1u64; half_n])
                .collect::<Vec<_>>();
            for a_i in tmp[0..chunk_size / 2].iter_mut() {
                *a_i = (*a_i).wrapping_neg();
            }
            tmp.rotate_left(chunk_size / 2);
            tmp
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

        // TODO: what should be the degree?
        Accumulator {
            acc: left_acc,
            degree: Degree(self.params.message_modulus.0 - 1),
        }
    }

    pub fn min(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
        let acc = self.double_ct_acc(a, b);

        // TODO may need to add t/4
        let diff = self.key.unchecked_sub(b, a);
        self.key.keyswitch_programmable_bootstrap(&diff, &acc)
    }

    pub fn arg_min(
        &self,
        a: &Ciphertext,
        b: &Ciphertext,
        i: &Ciphertext,
        j: &Ciphertext,
    ) -> Ciphertext {
        let acc = self.double_ct_acc(i, j);

        // TODO may need to add t/4
        let diff = self.key.unchecked_sub(b, a);
        self.key.keyswitch_programmable_bootstrap(&diff, &acc)
    }
}

pub struct KnnClient {
    key: ClientKey,
    ctx: Context,
}

impl KnnClient {
    pub fn lwe_encode_encrypt(&mut self, x: u64) -> Ciphertext {
        let ct = lwe_encode_encrypt(&self.key.get_lwe_sk_ref(), &mut self.ctx, x);
        Ciphertext {
            ct,
            degree: Degree(self.ctx.params.message_modulus.0 - 1),
            message_modulus: self.ctx.params.message_modulus,
            carry_modulus: self.ctx.params.carry_modulus,
        }
    }

    pub fn lwe_decrypt_decode(&self, ct: &Ciphertext) -> u64 {
        lwe_decrypt_decode(&self.key.get_lwe_sk_ref(), &self.ctx, &ct.ct)
    }

    pub fn glwe_encode_encrypt(
        &mut self,
        pt: &PlaintextListOwned<u64>,
    ) -> GlweCiphertextOwned<u64> {
        let mut pt_encoded = pt.clone();
        pt_encoded.iter_mut().for_each(|mut x| {
            self.ctx.codec.encode(&mut x.0);
        });

        let mut glwe = GlweCiphertext::new(
            0u64,
            self.ctx.params.glwe_dimension.to_glwe_size(),
            self.ctx.params.polynomial_size,
        );
        encrypt_glwe_ciphertext(
            self.key.get_glwe_sk_ref(),
            &mut glwe,
            &pt_encoded,
            self.ctx.params.glwe_modular_std_dev,
            &mut self.ctx.encryption_rng,
        );
        glwe
    }

    pub fn glwe_decrypt_decode(&self, ct: &GlweCiphertextOwned<u64>) -> PlaintextListOwned<u64> {
        let mut out = PlaintextList::new(0, PlaintextCount(self.ctx.params.polynomial_size.0));
        decrypt_glwe_ciphertext(self.key.get_glwe_sk_ref(), &ct, &mut out);
        out.iter_mut().for_each(|mut x| {
            self.ctx.codec.decode(&mut x.0);
        });
        out
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
        pbs_level: DecompositionLevelCount(9),
        pbs_base_log: DecompositionBaseLog(3),
        ks_level: DecompositionLevelCount(9),
        ks_base_log: DecompositionBaseLog(3),
        pfks_level: DecompositionLevelCount(9),
        pfks_base_log: DecompositionBaseLog(3),
        pfks_modular_std_dev: StandardDev(0.00000000000000029403601535432533),
        cbs_level: DecompositionLevelCount(0),
        cbs_base_log: DecompositionBaseLog(0),
        message_modulus: MessageModulus(32),
        carry_modulus: CarryModulus(1),
    };

    #[test]
    fn test_custom_accumulator() {
        // setup a truth table that always returns the same value `pt`
        // then using PBS we should always get `pt`
        let (server, mut client) = setup(TEST_PARAM);

        let pt = 1u64;
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
        let poly_ones = Polynomial::from_container({
            let mut tmp = vec![1u64; server.params.polynomial_size.0];
            let chunk_size = server.params.polynomial_size.0 / server.params.message_modulus.0;
            for a_i in tmp[0..chunk_size / 2].iter_mut() {
                *a_i = (*a_i).wrapping_neg();
            }
            // println!("chunk_size={}", chunk_size);
            tmp.rotate_left(chunk_size / 2);
            tmp
        });
        let glwe_acc = server.polynomial_glwe_mul(&ct_after, &poly_ones);

        let acc = Accumulator {
            acc: glwe_acc,
            degree: Degree(server.params.message_modulus.0 - 1), // TODO: how to set degree?
        };

        // now we do pbs and the result should always be `pt`
        // doesn't matter what the ct is or the encoding, so we use the shortint encrypt function
        // msb(u) in X^u * T(X) must 0, so we iterate to message_modulus/2
        for x in 0u64..(server.params.message_modulus.0 / 2) as u64 {
            let ct = client.lwe_encode_encrypt(x);
            let res = server.key.keyswitch_programmable_bootstrap(&ct, &acc);
            let actual = client.lwe_decrypt_decode(&res);
            println!("x={}, actual={}, expected={}", x, actual, pt);
            assert_eq!(actual, pt);
        }
    }

    #[test]
    fn test_min() {
        let (server, mut client) = setup(TEST_PARAM);
        // remember we need an extra bit for the negative
        let a_pt = 2u64;
        let b_pt = 3u64;
        let a_ct = client.lwe_encode_encrypt(a_pt);
        let b_ct = client.lwe_encode_encrypt(b_pt);

        {
            // test sub is working correctly
            let diff = server.key.unchecked_sub(&b_ct, &a_ct);
            let actual = client.lwe_decrypt_decode(&diff);
            let expected = b_pt.wrapping_sub(a_pt) % server.params.message_modulus.0 as u64;
            assert_eq!(actual, expected);
        }

        let min_ct = server.min(&a_ct, &b_ct);
        let actual = client.lwe_decrypt_decode(&min_ct);
        let expected = a_pt.min(b_pt);
        assert_eq!(actual, expected);
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
