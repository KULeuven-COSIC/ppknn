use crate::client::KnnClient;
use crate::EncItem;
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use tfhe::core_crypto::algorithms::*;
use tfhe::core_crypto::fft_impl::c64;
use tfhe::core_crypto::fft_impl::math::fft::FftView;
use tfhe::core_crypto::fft_impl::math::polynomial::FourierPolynomial;
use tfhe::core_crypto::prelude::polynomial_algorithms::*;
use tfhe::core_crypto::prelude::slice_algorithms::*;
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::ciphertext::Degree;
use tfhe::shortint::server_key::Accumulator;
use tfhe::shortint::{gen_keys, Ciphertext, Parameters, ServerKey};

pub(crate) fn polynomial_fft_wrapping_mul<Scalar, OutputCont, LhsCont, RhsCont>(
    output: &mut Polynomial<OutputCont>,
    lhs: &Polynomial<LhsCont>,
    rhs: &Polynomial<RhsCont>,
    fft: FftView,
    stack: &mut DynStack,
) where
    Scalar: UnsignedTorus,
    OutputCont: ContainerMut<Element = Scalar>,
    LhsCont: Container<Element = Scalar>,
    RhsCont: Container<Element = Scalar>,
{
    assert_eq!(lhs.polynomial_size(), rhs.polynomial_size());
    let n = lhs.polynomial_size().0;

    let mut fourier_lhs = FourierPolynomial {
        data: vec![c64::default(); n / 2],
    };
    let mut fourier_rhs = FourierPolynomial {
        data: vec![c64::default(); n / 2],
    };

    fft.forward_as_torus(
        unsafe { fourier_lhs.as_mut_view().into_uninit() },
        lhs.as_view(),
        stack.rb_mut(),
    );
    fft.forward_as_integer(
        unsafe { fourier_rhs.as_mut_view().into_uninit() },
        rhs.as_view(),
        stack.rb_mut(),
    );

    for (a, b) in fourier_lhs.data.iter_mut().zip(fourier_rhs.data.iter()) {
        *a *= *b;
    }

    fft.backward_as_torus(
        unsafe { output.as_mut_view().into_uninit() },
        fourier_lhs.as_view(),
        stack.rb_mut(),
    );
}

pub(crate) fn setup_polymul_fft(params: Parameters) -> (Fft, GlobalMemBuffer) {
    let fft = Fft::new(params.polynomial_size);
    let fft_view = fft.as_view();

    let mem = GlobalMemBuffer::new(
        fft_view
            .forward_scratch()
            .unwrap()
            .and(fft_view.backward_scratch().unwrap()),
    );
    (fft, mem)
}

/// This structure represents the server that is executing
/// privacy preserving k-NN. It needs to be constructed
/// using the `setup` function (or other variations such as `setup_with_modulus`).
pub struct KnnServer {
    key: ServerKey,
    lwe_to_glwe_ksk: LwePrivateFunctionalPackingKeyswitchKeyOwned<u64>,
    params: Parameters,
    dist_delta: u64, // delta value for distance computation
    gamma: usize,
    data: Vec<PlaintextListOwned<u64>>,
    labels: Vec<Ciphertext>, // trivially encrypted labels
}

impl KnnServer {
    /// Compute the squared distances between the target vector given by `c` and `c2`
    /// with the model stored in the server.
    /// The precision is reduced automatically if the distance plaintext modulus
    /// does not match with the sorting plaintext modulus.
    pub fn compute_distances(
        &self,
        c: &GlweCiphertextOwned<u64>,
        c2: &Ciphertext,
    ) -> Vec<Ciphertext> {
        let (fft, mut mem) = setup_polymul_fft(self.params);
        let mut stack = DynStack::new(&mut mem);
        self.compute_distances_with_fft(c, c2, fft.as_view(), &mut stack)
    }

    /// Compute the squared distances between the target vector given by `c` and `c2`
    /// with the model stored in the server using an existing FFT context.
    /// The precision is reduced automatically if the distance plaintext modulus
    /// does not match with the sorting plaintext modulus.
    pub fn compute_distances_with_fft(
        &self,
        c: &GlweCiphertextOwned<u64>,
        c2: &Ciphertext,
        fft: FftView,
        stack: &mut DynStack,
    ) -> Vec<Ciphertext> {
        let delta = self.dist_delta;
        let mut distances: Vec<_> = self
            .data
            .iter()
            .map(|m| {
                let mut glwe = c.clone();
                // we want to compute c^2 - 2 * m * c + m^2
                // first compute m*c where c is a RLWE
                glwe.get_mut_mask()
                    .as_mut_polynomial_list()
                    .iter_mut()
                    .for_each(|mut mask| {
                        polynomial_fft_wrapping_mul(
                            &mut mask,
                            &c.get_mask().as_polynomial_list().get(0),
                            &m.as_polynomial(),
                            fft,
                            stack,
                        );
                    });
                polynomial_fft_wrapping_mul(
                    &mut glwe.get_mut_body().as_mut_polynomial(),
                    &c.get_body().as_polynomial(),
                    &m.as_polynomial(),
                    fft,
                    stack,
                );

                // sample extract the \gamma -1 th coeff
                // m_times_c = m*c
                let m_times_c = {
                    let mut lwe = self.new_ct();
                    extract_lwe_sample_from_glwe_ciphertext(
                        &glwe,
                        &mut lwe.ct,
                        MonomialDegree(self.gamma - 1),
                    );
                    lwe
                };

                // c2 = \sum_{i=0}^{\gamma-1} c_i^2
                // out <- out - m_times_c * 2
                let mut out = c2.clone();
                slice_wrapping_sub_scalar_mul_assign(
                    &mut out.ct.as_mut(),
                    &m_times_c.ct.as_ref(),
                    2,
                );

                // add \sum_{i=0}^{\gamma-1} m_i^2
                // NOTE: m2 can be pre-computed, but we won't save much
                // since distance computation is fast anyway
                let m2 = Plaintext(delta * m.iter().map(|x| *x.0 * *x.0).sum::<u64>());
                lwe_ciphertext_plaintext_add_assign(&mut out.ct, m2);
                out
            })
            .collect();

        if self.dist_delta != self.delta() {
            distances.iter_mut().for_each(|x| self.lower_precision(x));
        }
        distances
    }

    /// Compute the squared distances between the target vector given by `c` and `c2`
    /// with the model stored in the server and zip the result with the existing labels/classes.
    /// The precision is reduced automatically if the distance plaintext modulus
    /// does not match with the sorting plaintext modulus.
    pub fn compute_distances_with_labels(
        &self,
        c: &GlweCiphertextOwned<u64>,
        c2: &Ciphertext,
    ) -> Vec<EncItem> {
        let distances = self.compute_distances(c, c2);
        let enc_vec = distances
            .into_iter()
            .zip(self.labels.iter())
            .map(|(d, l)| EncItem::new(d, l.clone()))
            .collect::<Vec<_>>();
        enc_vec
    }

    /// Reduce the plaintext modulus in `ct`.
    pub fn lower_precision(&self, ct: &mut Ciphertext) {
        // we assume the original ciphertext is encoded with higher precision
        // than the TFHE parameter
        // the number of elements that gets mapped into one element in the smaller message modulus
        let delta = self.dist_delta;
        let orig_modulus = (1u64 << 63) / (delta * self.params.carry_modulus.0 as u64);
        let precision_ratio = orig_modulus / self.params.message_modulus.0 as u64;
        assert!(precision_ratio > 1);

        // we need to "recenter" the plaintext space before doing the bootstrap
        // original pt: (0, Delta, 2*Delta, ..., (precision_ratio-1)*Delta)
        // is mapped to one element (0 in this example) in the new plaintext space
        // the recentering is done by subtracting
        // half of the maximum value (precision_ratio-1)*Delta
        // from the original pt
        let shift = Plaintext(((delta * (precision_ratio as u64 - 1)) / 2).wrapping_neg());
        lwe_ciphertext_plaintext_add_assign(&mut ct.ct, shift);

        self.key.keyswitch_bootstrap_assign(ct)
    }

    /// Keyswitch from LWE to RLWE.
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

    pub(crate) fn polynomial_glwe_mul_with_fft(
        &self,
        glwe: &GlweCiphertextOwned<u64>,
        poly: &PolynomialOwned<u64>,
        fft: FftView,
        stack: &mut DynStack,
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
                polynomial_fft_wrapping_mul(
                    &mut mask,
                    &glwe.get_mask().as_polynomial_list().get(0),
                    &poly,
                    fft,
                    stack,
                );
            });
        polynomial_fft_wrapping_mul(
            &mut out.get_mut_body().as_mut_polynomial(),
            &glwe.get_body().as_polynomial(),
            &poly,
            fft,
            stack,
        );
        out
    }

    fn double_glwe_acc(
        &self,
        left_glwe: &GlweCiphertextOwned<u64>,
        right_glwe: &GlweCiphertextOwned<u64>,
        fft: FftView,
        stack: &mut DynStack,
    ) -> Accumulator {
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
        let mut left_acc = self.polynomial_glwe_mul_with_fft(&left_glwe, &left_poly, fft, stack);
        let right_acc = self.polynomial_glwe_mul_with_fft(&right_glwe, &right_poly, fft, stack);

        // sum the two halves into the left one
        left_acc
            .as_mut_polynomial_list()
            .iter_mut()
            .zip(right_acc.as_polynomial_list().iter())
            .for_each(|(mut left, right)| polynomial_wrapping_add_assign(&mut left, &right));

        Accumulator {
            acc: left_acc,
            degree: Degree(self.params.message_modulus.0 - 1),
        }
    }

    /// Output the Delta (scaling factor) used for the
    /// sorting plaintext modulus.
    pub fn delta(&self) -> u64 {
        let delta =
            (1u64 << 63) / (self.params.message_modulus.0 * self.params.carry_modulus.0) as u64;
        delta
    }

    #[allow(dead_code)]
    pub(crate) fn trivially_double_ct_acc(
        &self,
        left_value: u64,
        right_value: u64,
        fft: FftView,
        stack: &mut DynStack,
    ) -> Accumulator {
        let encode = |message: u64| -> u64 {
            //The delta is the one defined by the parameters
            let delta = self.delta();

            //The input is reduced modulus the message_modulus
            let m = message % self.params.message_modulus.0 as u64;
            m * delta
        };

        let encode_and_pad = |value| {
            PlaintextList::from_container(
                vec![encode(value)]
                    .into_iter()
                    .chain(vec![0; self.params.polynomial_size.0 - 1])
                    .collect::<Vec<_>>(),
            )
        };
        let left_encoded = encode_and_pad(left_value);
        let right_encoded = encode_and_pad(right_value);

        let mut left_glwe = GlweCiphertext::new(
            0u64,
            self.params.glwe_dimension.to_glwe_size(),
            self.params.polynomial_size,
        );
        let mut right_glwe = GlweCiphertext::new(
            0u64,
            self.params.glwe_dimension.to_glwe_size(),
            self.params.polynomial_size,
        );
        trivially_encrypt_glwe_ciphertext(&mut left_glwe, &left_encoded);
        trivially_encrypt_glwe_ciphertext(&mut right_glwe, &right_encoded);

        self.double_glwe_acc(&left_glwe, &right_glwe, fft, stack)
    }

    /// Create an accumulator from two ciphertexts
    /// such that when used in PBS, if the sign is positive,
    /// then `left_lwe` is the output, if the sign is negative, then the output is `right_lwe`.
    pub fn double_ct_acc(
        &self,
        left_lwe: &Ciphertext,
        right_lwe: &Ciphertext,
        fft: FftView,
        stack: &mut DynStack,
    ) -> Accumulator {
        // first key switch the LWE ciphertexts to GLWE
        let left_glwe = self.lwe_to_glwe(&left_lwe);
        let right_glwe = self.lwe_to_glwe(&right_lwe);

        self.double_glwe_acc(&left_glwe, &right_glwe, fft, stack)
    }

    fn special_sub(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
        // we use a raw subtract and then add by t/2 to ensure the negative
        // does not overflow into the padding bit

        let mut res = self.raw_sub(&b, &a);

        let delta = self.delta();
        let mod_over_2 = Plaintext((self.params.message_modulus.0 as u64 / 2) * delta);
        lwe_ciphertext_plaintext_add_assign(&mut res.ct, mod_over_2);

        res
    }

    /// Compute `min(a, b)` homomorphically with an existing FFT context.
    pub fn min_with_fft(
        &self,
        a: &Ciphertext,
        b: &Ciphertext,
        fft: FftView,
        stack: &mut DynStack,
    ) -> Ciphertext {
        let acc = self.double_ct_acc(a, b, fft, stack);

        let diff = self.special_sub(&b, &a);
        self.key.keyswitch_programmable_bootstrap(&diff, &acc)
    }

    /// Compute `min(a, b)` homomorphically.
    pub fn min(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
        let (fft, mut mem) = setup_polymul_fft(self.params);
        let mut stack = DynStack::new(&mut mem);
        self.min_with_fft(a, b, fft.as_view(), &mut stack)
    }

    #[allow(dead_code)]
    pub(crate) fn trivially_min(
        &self,
        a_pt: u64,
        b_pt: u64,
        a: &Ciphertext,
        b: &Ciphertext,
    ) -> Ciphertext {
        let (fft, mut mem) = setup_polymul_fft(self.params);
        let mut stack = DynStack::new(&mut mem);
        let acc = self.trivially_double_ct_acc(a_pt, b_pt, fft.as_view(), &mut stack);

        let diff = self.special_sub(&b, &a);
        self.key.keyswitch_programmable_bootstrap(&diff, &acc)
    }

    /// Execute `arg_min(a_i, b_j) = if a == min(a_i, b_j) j else i` homomorphically
    /// using an existing FFT context.
    pub fn arg_min_with_fft(
        &self,
        a: &Ciphertext,
        b: &Ciphertext,
        i: &Ciphertext,
        j: &Ciphertext,
        fft: FftView,
        stack: &mut DynStack,
    ) -> Ciphertext {
        let acc = self.double_ct_acc(i, j, fft, stack);

        let diff = self.special_sub(&b, &a);
        self.key.keyswitch_programmable_bootstrap(&diff, &acc)
    }

    /// Execute `arg_min(a_i, b_j) = if a == min(a_i, b_j) j else i` homomorphically.
    pub fn arg_min(
        &self,
        a: &Ciphertext,
        b: &Ciphertext,
        i: &Ciphertext,
        j: &Ciphertext,
    ) -> Ciphertext {
        let (fft, mut mem) = setup_polymul_fft(self.params);
        let mut stack = DynStack::new(&mut mem);
        self.arg_min_with_fft(a, b, i, j, fft.as_view(), &mut stack)
    }

    fn new_ct(&self) -> Ciphertext {
        let res = Ciphertext {
            ct: LweCiphertextOwned::new(0u64, LweSize(self.params.polynomial_size.0 + 1)),
            degree: Degree(self.params.message_modulus.0 - 1),
            message_modulus: self.params.message_modulus,
            carry_modulus: self.params.carry_modulus,
        };
        res
    }

    /// Create a trivial (noiseless) encryption of `x`, i.e., enc(x) = (0, Delta * x)
    pub fn trivially_encrypt(&self, x: u64) -> Ciphertext {
        self.trivially_encrypt_with_delta(x, self.delta())
    }

    pub fn trivially_encrypt_with_delta(&self, x: u64, delta: u64) -> Ciphertext {
        let mut out = self.new_ct();
        let pt = Plaintext(x * delta);
        trivially_encrypt_lwe_ciphertext(&mut out.ct, pt);
        out
    }

    pub fn raw_sub(&self, lhs: &Ciphertext, rhs: &Ciphertext) -> Ciphertext {
        let mut res = self.new_ct();
        slice_wrapping_sub(&mut res.ct.as_mut(), &lhs.ct.as_ref(), &rhs.ct.as_ref());
        res
    }

    pub fn raw_sub_assign(&self, lhs: &mut Ciphertext, rhs: &Ciphertext) {
        slice_wrapping_sub_assign(&mut lhs.ct.as_mut(), &rhs.ct.as_ref())
    }

    pub fn raw_add(&self, lhs: &Ciphertext, rhs: &Ciphertext) -> Ciphertext {
        let mut res = self.new_ct();
        slice_wrapping_add(&mut res.ct.as_mut(), &lhs.ct.as_ref(), &rhs.ct.as_ref());
        res
    }

    pub fn raw_add_assign(&self, lhs: &mut Ciphertext, rhs: &Ciphertext) {
        slice_wrapping_add_assign(&mut lhs.ct.as_mut(), &rhs.ct.as_ref())
    }

    pub fn set_data(&mut self, data: &[Vec<u64>]) {
        let gamma = data.iter().fold(0usize, |acc, x| acc.max(x.len()));
        let padding = vec![0u64; self.params.polynomial_size.0 - gamma];
        let data: Vec<_> = data
            .iter()
            .map(|v| {
                PlaintextList::from_container({
                    let mut v_cloned = v.clone();
                    v_cloned.reverse();
                    v_cloned.extend_from_slice(&padding);
                    v_cloned
                })
            })
            .collect();

        self.gamma = gamma;
        self.data = data;
    }

    pub fn set_labels(&mut self, labels: &[u64]) {
        // we do not lower the precision of the labels, so use the "after" delta
        let delta = self.delta();
        self.labels = labels
            .iter()
            .map(|l| self.trivially_encrypt_with_delta(*l, delta))
            .collect::<Vec<_>>();
    }
}

fn setup_with_modulus(params: Parameters, dist_modulus: u64) -> (KnnClient, KnnServer) {
    assert!(dist_modulus.is_power_of_two());

    let mut seeder = new_seeder();
    let mut encryption_rng =
        EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder.as_mut());
    let (client_key, server_key) = gen_keys(params);
    let lwe_to_glwe_ksk = crate::gen_ksk(&client_key, &mut encryption_rng);

    let dist_delta = (1u64 << 63) / (dist_modulus * params.carry_modulus.0 as u64);
    assert_eq!(dist_delta % 2, 0);
    assert!((params.message_modulus.0 as u64) < dist_delta);
    (
        KnnClient {
            key: client_key,
            params,
            encryption_rng,
            dist_delta,
        },
        KnnServer {
            key: server_key,
            lwe_to_glwe_ksk,
            params,
            dist_delta,
            gamma: 0,
            data: vec![],
            labels: vec![],
        },
    )
}

pub fn setup(params: Parameters) -> (KnnClient, KnnServer) {
    let modulus = params.message_modulus.0 as u64;
    setup_with_modulus(params, modulus)
}

/// Setup the server and client with a model, specified by `data` and `labels`.
/// The data should not be encoded.
pub fn setup_with_data(
    params: Parameters,
    data: &[Vec<u64>],
    labels: &[u64],
    dist_modulus: u64,
) -> (KnnClient, KnnServer) {
    let (client, mut server) = setup_with_modulus(params, dist_modulus);
    server.set_data(data);
    server.set_labels(labels);
    (client, server)
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::batcher::BatcherSort;
    use crate::{AsyncBatcher, AsyncEncComparator, EncCmp};
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex, RwLock};
    use tfhe::shortint::prelude::*;

    pub(crate) const TEST_PARAM: Parameters = Parameters {
        lwe_dimension: LweDimension(742),
        glwe_dimension: GlweDimension(1),
        polynomial_size: PolynomialSize(2048),
        lwe_modular_std_dev: StandardDev(0.000007069849454709433),
        glwe_modular_std_dev: StandardDev(0.00000000000000029403601535432533),
        pbs_level: DecompositionLevelCount(6),
        pbs_base_log: DecompositionBaseLog(3),
        ks_level: DecompositionLevelCount(6),
        ks_base_log: DecompositionBaseLog(3),
        pfks_level: DecompositionLevelCount(6),
        pfks_base_log: DecompositionBaseLog(3),
        pfks_modular_std_dev: StandardDev(0.00000000000000029403601535432533),
        cbs_level: DecompositionLevelCount(0),
        cbs_base_log: DecompositionBaseLog(0),
        message_modulus: MessageModulus(32),
        carry_modulus: CarryModulus(1),
    };

    fn decode(params: Parameters, x: u64) -> u64 {
        let delta = (1u64 << 63) / (params.message_modulus.0 * params.carry_modulus.0) as u64;

        //The bit before the message
        let rounding_bit = delta >> 1;

        //compute the rounding bit
        let rounding = (x & rounding_bit) << 1;

        x.wrapping_add(rounding) / delta
    }

    fn enc_vec(vs: &[(u64, u64)], client_key: &ClientKey) -> Vec<EncItem> {
        vs.iter()
            .map(|v| EncItem::new(client_key.encrypt(v.0), client_key.encrypt(v.1)))
            .collect()
    }

    fn enc_vec_async(vs: &[(u64, u64)], client_key: &ClientKey) -> Vec<Arc<Mutex<EncItem>>> {
        vs.iter()
            .map(|v| {
                Arc::new(Mutex::new(EncItem::new(
                    client_key.encrypt(v.0),
                    client_key.encrypt(v.1),
                )))
            })
            .collect()
    }

    #[test]
    fn test_tfhe_arith() {
        // testing some basic tfhe-rs operations
        let (client, server) = gen_keys(TEST_PARAM);
        {
            // computation without considering the padding bit
            // note that we cannot use unchecked_sub for this
            let ct_0 = client.encrypt_without_padding(0);
            let ct_1 = client.encrypt_without_padding(1);
            let mut res = Ciphertext {
                ct: LweCiphertextOwned::new(0u64, LweSize(client.parameters.polynomial_size.0 + 1)),
                degree: Degree(client.parameters.message_modulus.0 - 1),
                message_modulus: client.parameters.message_modulus,
                carry_modulus: client.parameters.carry_modulus,
            };
            slice_wrapping_sub(&mut res.ct.as_mut(), &ct_0.ct.as_ref(), &ct_1.ct.as_ref());
            assert_eq!(
                client.decrypt_without_padding(&res),
                client.parameters.message_modulus.0 as u64 - 1
            );
        }
        {
            // computation with the padding bit for -1
            let ct_0 = client.encrypt(0);
            let ct_1 = client.encrypt(1);
            let ct = server.unchecked_sub(&ct_0, &ct_1);
            let res = client.decrypt(&ct);
            assert_eq!(res, client.parameters.message_modulus.0 as u64 - 1);

            // check that the carry-bit is 1 also
            // let carry_msg = client.decrypt_message_and_carry(&ct);
            // assert_eq!((carry_msg ^ res), client.parameters.message_modulus.0 as u64);
        }
        {
            // computation with the padding bit for 0 - (-1)
            let ct_0 = client.encrypt(0);
            let ct_1 = client.encrypt(client.parameters.message_modulus.0 as u64 - 1);
            let res = server.unchecked_sub(&ct_0, &ct_1);
            assert_eq!(client.decrypt(&res), 1);
        }
    }

    #[test]
    fn test_custom_accumulator() {
        // setup a truth table that always returns the same value `pt`
        // then using PBS we should always get `pt`
        let (client, server) = setup(TEST_PARAM);

        let pt = 1u64;
        let ct_before = client.key.encrypt(pt);
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
            output_plaintext.iter_mut().for_each(|x| {
                *x.0 = decode(TEST_PARAM, *x.0);
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
        let (fft, mut mem) = setup_polymul_fft(TEST_PARAM);
        let mut stack = DynStack::new(&mut mem);
        let glwe_acc =
            server.polynomial_glwe_mul_with_fft(&ct_after, &poly_ones, fft.as_view(), &mut stack);

        let acc = Accumulator {
            acc: glwe_acc,
            degree: Degree(server.params.message_modulus.0 - 1),
        };

        // now we do pbs and the result should always be `pt`
        for x in 0u64..server.params.message_modulus.0 as u64 {
            let ct = client.key.encrypt(x);
            let res = server.key.keyswitch_programmable_bootstrap(&ct, &acc);
            let actual = client.key.decrypt(&res);
            println!("x={}, actual={}, expected={}", x, actual, pt);
            assert_eq!(actual, pt);
        }
    }

    #[test]
    fn test_double_ct_acc() {
        let (client, server) = setup(TEST_PARAM);
        let left = 1u64;
        let right = server.params.message_modulus.0 as u64 - 1;
        // let acc = server.trivially_double_ct_acc(left, right);
        let (fft, mut mem) = setup_polymul_fft(TEST_PARAM);
        let mut stack = DynStack::new(&mut mem);
        let acc = server.double_ct_acc(
            &client.key.encrypt(left),
            &client.key.encrypt(right),
            fft.as_view(),
            &mut stack,
        );

        let modulus = server.params.message_modulus.0;
        for x in 0u64..modulus as u64 {
            let ct = client.key.encrypt(x);
            let res = server.key.keyswitch_programmable_bootstrap(&ct, &acc);
            let actual = client.key.decrypt(&res);
            let expected = if x < modulus as u64 / 2 { left } else { right };
            println!("x={}, actual={}, expected={}", x, actual, expected);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_min() {
        let (client, server) = setup(TEST_PARAM);

        // note that we can only use half of the plaintext space since
        // the subtraction will take us to the full plaintext space
        for a_pt in 0..server.params.message_modulus.0 as u64 / 2 {
            let a_ct = client.key.encrypt(a_pt);
            for b_pt in 0..server.params.message_modulus.0 as u64 / 2 {
                let b_ct = client.key.encrypt(b_pt);
                let min_ct = server.min(&a_ct, &b_ct);
                let actual = client.key.decrypt(&min_ct);
                let expected = a_pt.min(b_pt);
                println!(
                    "a={}, b={}, actual={}, expected={}",
                    a_pt, b_pt, actual, expected
                );
                assert_eq!(actual, expected);
                {
                    let trivial_min_ct = server.trivially_min(a_pt, b_pt, &a_ct, &b_ct);
                    let trivial_actual = client.key.decrypt(&trivial_min_ct);
                    assert_eq!(trivial_actual, expected);
                }
            }
        }
    }

    #[test]
    fn test_enc_sort() {
        let (client, server) = setup(TEST_PARAM);
        let server = Rc::new(RefCell::new(server));
        {
            let pt_vec = vec![(1, 0), (0, 1), (2, 2), (3u64, 3u64)];
            let enc_cmp = EncCmp::boxed(enc_vec(&pt_vec, &client.key), TEST_PARAM, server.clone());

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let actual = sorter.inner()[0].decrypt(&client.key);
            let expected = (0u64, 1u64);
            assert_eq!(actual, expected);

            let noise = client.lwe_noise(&sorter.inner()[0].value, expected.0);
            println!("noise={}", noise);
        }
        {
            let pt_vec = vec![(2, 0), (2, 1), (1, 2), (3u64, 3u64)];
            let enc_cmp = EncCmp::boxed(enc_vec(&pt_vec, &client.key), TEST_PARAM, server.clone());

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let actual = sorter.inner()[0].decrypt(&client.key);
            let expected = (1u64, 2u64);
            assert_eq!(actual, expected);

            let noise = client.lwe_noise(&sorter.inner()[0].value, expected.0);
            println!("noise={}", noise);
        }
        {
            let pt_vec = vec![(1, 0), (2, 1), (3u64, 2u64), (0, 3)];
            let enc_cmp = EncCmp::boxed(enc_vec(&pt_vec, &client.key), TEST_PARAM, server.clone());

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let actual = sorter.inner()[0].decrypt(&client.key);
            let expected = (0u64, 3u64);
            assert_eq!(actual, expected);

            let noise = client.lwe_noise(&sorter.inner()[0].value, expected.0);
            println!("noise={}", noise);
        }
    }

    #[test]
    fn test_enc_sort_async() {
        let (client, server) = setup(TEST_PARAM);
        let server = Arc::new(RwLock::new(server));
        {
            let pt_vec = vec![(1, 0), (0, 1), (2, 2), (3u64, 3u64)];
            let ct_vec = enc_vec_async(&pt_vec, &client.key);

            let async_cmp = AsyncEncComparator::new(server.clone(), TEST_PARAM);
            let batcher = AsyncBatcher::<_, ()>::new_k(1, Arc::new(async_cmp), false);
            batcher.sort(&ct_vec);

            let actual = ct_vec[0].lock().unwrap().decrypt(&client.key);
            let expected = (0u64, 1u64);
            assert_eq!(actual, expected);

            let noise = client.lwe_noise(&ct_vec[0].lock().unwrap().value, expected.0);
            println!("noise={}", noise);
        }
        {
            let pt_vec = vec![(2, 0), (2, 1), (1, 2), (3u64, 3u64)];
            let ct_vec = enc_vec_async(&pt_vec, &client.key);

            let async_cmp = AsyncEncComparator::new(server.clone(), TEST_PARAM);
            let batcher = AsyncBatcher::<_, ()>::new_k(1, Arc::new(async_cmp), false);
            batcher.sort(&ct_vec);

            let actual = ct_vec[0].lock().unwrap().decrypt(&client.key);
            let expected = (1u64, 2u64);
            assert_eq!(actual, expected);

            let noise = client.lwe_noise(&ct_vec[0].lock().unwrap().value, expected.0);
            println!("noise={}", noise);
        }
        {
            let pt_vec = vec![(1, 0), (2, 1), (3u64, 2u64), (0, 3)];
            let ct_vec = enc_vec_async(&pt_vec, &client.key);

            let async_cmp = AsyncEncComparator::new(server.clone(), TEST_PARAM);
            let batcher = AsyncBatcher::<_, ()>::new_k(1, Arc::new(async_cmp), false);
            batcher.sort(&ct_vec);

            let actual = ct_vec[0].lock().unwrap().decrypt(&client.key);
            let expected = (0u64, 3u64);
            assert_eq!(actual, expected);

            let noise = client.lwe_noise(&ct_vec[0].lock().unwrap().value, expected.0);
            println!("noise={}", noise);
        }
    }

    #[test]
    fn test_compute_distance() {
        let (mut client, mut server) = setup(TEST_PARAM);
        {
            // distance should be 2^2 + 1 = 5
            let data = vec![vec![0, 1, 0, 0u64]];
            let target = vec![2, 0, 0, 0u64];
            server.set_data(&data);
            let (glwe, lwe) = client.make_query(&target);
            let distances = server.compute_distances(&glwe, &lwe);

            let expected = 5u64;
            assert_eq!(client.key.decrypt(&distances[0]), expected);
        }
        {
            // distance should be 2^2 = 4
            let data = vec![vec![0, 0, 1, 3u64]];
            let target = vec![0, 0, 1, 1u64];
            server.set_data(&data);
            let (glwe, lwe) = client.make_query(&target);
            let distances = server.compute_distances(&glwe, &lwe);

            let expected = 4u64;
            assert_eq!(client.key.decrypt(&distances[0]), expected);
        }
        {
            let data = vec![vec![2, 0, 0, 0u64]];
            let target = vec![4, 2, 0, 0u64];
            server.set_data(&data);
            let (glwe, lwe) = client.make_query(&target);
            let distances = server.compute_distances(&glwe, &lwe);

            let expected = 8u64;
            println!(
                "noise_in_fresh_ct={:?}",
                client.lwe_noise(&client.key.encrypt(expected), expected)
            );
            println!(
                "noise_in_distance={:?}",
                client.lwe_noise(&distances[0], expected)
            );
            assert_eq!(client.key.decrypt(&distances[0]), expected);
        }
    }

    #[test]
    fn test_lower_precision() {
        // we need bigger parameters for this test
        let final_params = Parameters {
            message_modulus: MessageModulus(32),
            carry_modulus: CarryModulus(1),
            ..PARAM_MESSAGE_2_CARRY_3
        };
        let initial_modulus = MessageModulus(64);
        let initial_params = Parameters {
            message_modulus: initial_modulus,
            ..final_params
        };
        let (mut client, server) = setup_with_modulus(final_params, initial_modulus.0 as u64);
        let final_modulus = server.params.message_modulus;
        let ratio = (initial_modulus.0 / final_modulus.0) as u64;
        for m in 0..initial_modulus.0 as u64 {
            let mut ct = client.lwe_encrypt_with_modulus(m, initial_modulus.0);

            // check for correct decryption
            let encoded = decrypt_lwe_ciphertext(client.key.get_lwe_sk_ref(), &ct.ct);
            let pt = decode(initial_params, encoded.0);
            assert_eq!(pt, m);

            // lower the precision
            server.lower_precision(&mut ct);
            let expected = m / ratio;
            let actual = client.key.decrypt(&ct);
            println!(
                "m={m}, actual={actual}, expected={expected}, noise={:.2}",
                client.lwe_noise(&ct, expected)
            );
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_compute_distance_lower_precision() {
        let final_params = Parameters {
            message_modulus: MessageModulus(32),
            carry_modulus: CarryModulus(1),
            ..PARAM_MESSAGE_2_CARRY_3
        };
        let initial_modulus = MessageModulus(64);
        let (mut client, mut server) = setup_with_modulus(final_params, initial_modulus.0 as u64);
        let final_modulus = server.params.message_modulus;
        let ratio = (initial_modulus.0 / final_modulus.0) as u64;

        for i in 0..8u64 {
            // 8*8 = 64 (initial_modulus)
            for j in 0..4 {
                // we want the maximum to be 7*7 + 3*3 < 64
                let data = vec![vec![i, 0, 0, 0u64]];
                let target = vec![0, j, 0, 0u64];
                server.set_data(&data);
                let (glwe, lwe) = client.make_query(&target);
                let distances = server.compute_distances(&glwe, &lwe);

                let expected = (j * j + i * i) / ratio;
                let actual = client.key.decrypt(&distances[0]);
                println!(
                    "i={i}, j={j}, actual={actual}, expected={expected}, noise={:.2}",
                    client.lwe_noise(&distances[0], expected)
                );
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn test_fft() {
        let fft = Fft::new(TEST_PARAM.polynomial_size);
        let fft = fft.as_view();
        let n = 2048usize;

        let mut mem = GlobalMemBuffer::new(
            fft.forward_scratch()
                .unwrap()
                .and(fft.backward_scratch().unwrap()),
        );
        let mut stack = DynStack::new(&mut mem);

        let input1 = Polynomial::from_container({
            (0..n)
                .into_iter()
                .map(|_| rand::random::<u16>() as u64)
                .collect::<Vec<_>>()
        });
        let input2 = Polynomial::from_container({
            (0..n)
                .into_iter()
                .map(|_| rand::random::<u16>() as u64)
                .collect::<Vec<_>>()
        });

        let mut actual = Polynomial::new(0u64, PolynomialSize(n));
        polynomial_fft_wrapping_mul(&mut actual, &input1, &input2, fft, &mut stack);

        let mut expected = Polynomial::new(0u64, PolynomialSize(n));
        polynomial_wrapping_mul(&mut expected, &input1, &input2);

        assert_eq!(actual, expected);
    }

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
