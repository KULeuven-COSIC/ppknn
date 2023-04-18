pub mod batcher;
pub mod comparator;
pub mod keyswitch;

pub use batcher::*;
pub use comparator::*;

use dyn_stack::{mem::GlobalMemBuffer, DynStack, ReborrowMut};
use std::fs;
use std::io::Cursor;
use tfhe::core_crypto::fft_impl::c64;
use tfhe::core_crypto::fft_impl::math::fft::FftView;
use tfhe::core_crypto::fft_impl::math::polynomial::FourierPolynomial;
use tfhe::core_crypto::prelude::polynomial_algorithms::*;
use tfhe::core_crypto::prelude::slice_algorithms::*;
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

pub fn parse_csv(f_handle: fs::File) -> (Vec<Vec<u64>>, Vec<u64>) {
    // let f_handle = fs::File::open(path).expect("csv file not found, consider using --artificial");
    let mut model_vec: Vec<Vec<u64>> = vec![];
    let mut class: Vec<u64> = vec![];
    let mut reader = csv::Reader::from_reader(f_handle);
    for res in reader.records() {
        let record = res.unwrap();
        let mut row: Vec<_> = record.iter().map(|s| s.parse().unwrap()).collect();
        let last = row.pop().unwrap();
        model_vec.push(row);
        class.push(last);
    }

    (model_vec, class)
}

pub fn enc_vec(vs: &[(u64, u64)], client_key: &ClientKey) -> Vec<EncItem> {
    vs.iter()
        .map(|v| EncItem::new(client_key.encrypt(v.0), client_key.encrypt(v.1)))
        .collect()
}

pub fn decode(params: Parameters, x: u64) -> u64 {
    let delta = (1u64 << 63) / (params.message_modulus.0 * params.carry_modulus.0) as u64;

    //The bit before the message
    let rounding_bit = delta >> 1;

    //compute the rounding bit
    let rounding = (x & rounding_bit) << 1;

    x.wrapping_add(rounding) / delta
}

pub struct KnnServer {
    key: ServerKey,
    lwe_to_glwe_ksk: LwePrivateFunctionalPackingKeyswitchKeyOwned<u64>,
    params: Parameters,
    dist_delta: u64, // delta value for distance computation
    gamma: usize,
    data: Vec<PlaintextListOwned<u64>>,
    labels: Vec<Ciphertext>, // trivially encrypted labels
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

impl KnnServer {
    pub fn compute_distances(
        &self,
        c: &GlweCiphertextOwned<u64>,
        c2: &Ciphertext,
    ) -> Vec<Ciphertext> {
        let (fft, mut mem) = setup_polymul_fft(self.params);
        let mut stack = DynStack::new(&mut mem);
        self.compute_distances_with_fft(c, c2, fft.as_view(), &mut stack)
    }

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
                // first compute 2*m*c where c is a RLWE
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
                slice_wrapping_scalar_mul_assign(
                    &mut glwe.as_mut(),
                    self.params.message_modulus.0 as u64 - 2u64,
                );
                // slice_wrapping_opposite_assign(&mut glwe.as_mut()); // combine with scalar_mul?

                // sample extract the \gamma -1 th coeff
                let mut lwe = self.new_ct();
                extract_lwe_sample_from_glwe_ciphertext(
                    &glwe,
                    &mut lwe.ct,
                    MonomialDegree(self.gamma - 1),
                );

                // add c2 = \sum_{i=0}^{\gamma-1} c_i^2 to -2*m*c
                lwe_ciphertext_add_assign(&mut lwe.ct, &c2.ct);

                // add \sum_{i=0}^{\gamma-1} m_i^2
                let m2 = Plaintext(delta * m.iter().map(|x| *x.0 * *x.0).sum::<u64>());
                lwe_ciphertext_plaintext_add_assign(&mut lwe.ct, m2);
                lwe
            })
            .collect();

        if self.dist_delta != self.delta() {
            distances.iter_mut().for_each(|x| self.lower_precision(x));
        }
        distances
    }

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
        let (fft, mut mem) = setup_polymul_fft(self.params);
        let mut stack = DynStack::new(&mut mem);
        self.polynomial_glwe_mul_with_fft(glwe, poly, fft.as_view(), &mut stack)
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
            degree: Degree(self.params.message_modulus.0 - 1),
        }
    }

    fn delta(&self) -> u64 {
        let delta =
            (1u64 << 63) / (self.params.message_modulus.0 * self.params.carry_modulus.0) as u64;
        delta
    }

    pub fn trivially_double_ct_acc(&self, left_value: u64, right_value: u64) -> Accumulator {
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

        self.double_glwe_acc(&left_glwe, &right_glwe)
    }

    pub fn double_ct_acc(&self, left_lwe: &Ciphertext, right_lwe: &Ciphertext) -> Accumulator {
        // first key switch the LWE ciphertexts to GLWE
        let left_glwe = self.lwe_to_glwe(&left_lwe);
        let right_glwe = self.lwe_to_glwe(&right_lwe);

        self.double_glwe_acc(&left_glwe, &right_glwe)
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

    pub fn min(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
        let acc = self.double_ct_acc(a, b);

        let diff = self.special_sub(&b, &a);
        self.key.keyswitch_programmable_bootstrap(&diff, &acc)
    }

    pub fn trivially_min(
        &self,
        a_pt: u64,
        b_pt: u64,
        a: &Ciphertext,
        b: &Ciphertext,
    ) -> Ciphertext {
        let acc = self.trivially_double_ct_acc(a_pt, b_pt);

        let diff = self.special_sub(&b, &a);
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

        let diff = self.special_sub(&b, &a);
        self.key.keyswitch_programmable_bootstrap(&diff, &acc)
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

    pub fn set_data(&mut self, data: Vec<Vec<u64>>) {
        let gamma = data.iter().fold(0usize, |acc, x| acc.max(x.len()));
        let padding = vec![0u64; self.params.polynomial_size.0 - gamma];
        let data: Vec<_> = data
            .into_iter()
            .map(|mut v| {
                PlaintextList::from_container({
                    v.reverse();
                    v.extend_from_slice(&padding);
                    v
                })
            })
            .collect();

        self.gamma = gamma;
        self.data = data;
    }

    pub fn set_labels(&mut self, labels: Vec<u64>) {
        self.labels = labels
            .into_iter()
            .map(|l| self.trivially_encrypt_with_delta(l, self.dist_delta))
            .collect::<Vec<_>>();
    }
}

pub struct KnnClient {
    pub key: ClientKey,
    pub params: Parameters,
    pub encryption_rng: EncryptionRandomGenerator<ActivatedRandomGenerator>,
}

impl KnnClient {
    pub fn lwe_encrypt_with_modulus(&mut self, x: u64, modulus: usize) -> Ciphertext {
        // we still consider the padding bit
        let delta = (1u64 << 63) / (modulus * self.params.carry_modulus.0) as u64;
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

    pub fn lwe_noise(&self, ct: &Ciphertext, expected_pt: u64) -> f64 {
        // pt = b - a*s = Delta*m + e
        let mut pt = decrypt_lwe_ciphertext(&self.key.get_lwe_sk_ref(), &ct.ct);

        // pt = pt - Delta*m = e (encoded_ptxt is Delta*m)
        let delta = self.delta();

        pt.0 = pt.0.wrapping_sub(delta * expected_pt);

        ((pt.0 as i64).abs() as f64).log2()
    }

    fn delta(&self) -> u64 {
        let delta =
            (1u64 << 63) / (self.params.message_modulus.0 * self.params.carry_modulus.0) as u64;
        delta
    }

    pub fn make_query(&mut self, target: &[u64]) -> (GlweCiphertextOwned<u64>, Ciphertext) {
        let gamma = target.len();
        let n = self.params.polynomial_size.0;
        let padding = vec![0u64; n - gamma];
        let delta = self.delta();
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
        let c2 = self.key.encrypt(pt2);
        (c, c2)
    }
}

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

pub fn setup(params: Parameters) -> (KnnClient, KnnServer) {
    let modulus = params.message_modulus.0 as u64;
    setup_with_modulus(params, modulus)
}

pub fn setup_with_modulus(params: Parameters, dist_modulus: u64) -> (KnnClient, KnnServer) {
    let mut seeder = new_seeder();
    let mut encryption_rng =
        EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder.as_mut());
    let (client_key, server_key) = gen_keys(params);
    let lwe_to_glwe_ksk = gen_ksk(&client_key, &mut encryption_rng);

    let dist_delta = (1u64 << 63) / (dist_modulus * params.carry_modulus.0 as u64);
    assert_eq!(dist_delta % 2, 0);
    assert!((params.message_modulus.0 as u64) < dist_delta);
    (
        KnnClient {
            key: client_key,
            params,
            encryption_rng,
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

// The data should not be encoded
pub fn setup_with_data(
    params: Parameters,
    data: Vec<Vec<u64>>,
    labels: Vec<u64>,
    dist_modulus: u64,
) -> (KnnClient, KnnServer) {
    let (client, mut server) = setup_with_modulus(params, dist_modulus);
    server.set_data(data);
    server.set_labels(labels);
    (client, server)
}

pub fn polynomial_fft_wrapping_mul<Scalar, OutputCont, LhsCont, RhsCont>(
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

#[cfg(test)]
mod test {
    use super::*;

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
        let glwe_acc = server.polynomial_glwe_mul(&ct_after, &poly_ones);

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
        let acc = server.double_ct_acc(&client.key.encrypt(left), &client.key.encrypt(right));
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
                // let min_ct = server.trivially_min(a_pt, b_pt, &a_ct, &b_ct);
                let actual = client.key.decrypt(&min_ct);
                let expected = a_pt.min(b_pt);
                println!(
                    "a={}, b={}, actual={}, expected={}",
                    a_pt, b_pt, actual, expected
                );
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn test_enc_sort() {
        {
            let (client, server) = setup(TEST_PARAM);
            let pt_vec = vec![(1, 1), (0, 0), (2, 2), (3u64, 3u64)];
            let enc_cmp = EncCmp::boxed(enc_vec(&pt_vec, &client.key), TEST_PARAM, server);

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let actual = sorter.inner()[0].decrypt(&client.key);
            let expected = (0u64, 0u64);
            assert_eq!(actual, expected);

            let noise = client.lwe_noise(&sorter.inner()[0].value, expected.0);
            println!("noise={}", noise);
        }
        {
            let (client, server) = setup(TEST_PARAM);
            let pt_vec = vec![(2, 2), (2, 2), (1, 1), (3u64, 3u64)];
            let enc_cmp = EncCmp::boxed(enc_vec(&pt_vec, &client.key), TEST_PARAM, server);

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let actual = sorter.inner()[0].decrypt(&client.key);
            let expected = (1u64, 1u64);
            assert_eq!(actual, expected);

            let noise = client.lwe_noise(&sorter.inner()[0].value, expected.0);
            println!("noise={}", noise);
        }
        {
            let (client, server) = setup(TEST_PARAM);
            let pt_vec = vec![(1, 1), (2, 2), (3u64, 3u64), (0, 0)];
            let enc_cmp = EncCmp::boxed(enc_vec(&pt_vec, &client.key), TEST_PARAM, server);

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let actual = sorter.inner()[0].decrypt(&client.key);
            let expected = (0u64, 0u64);
            assert_eq!(actual, expected);

            let noise = client.lwe_noise(&sorter.inner()[0].value, expected.0);
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
            server.set_data(data);
            let (glwe, lwe) = client.make_query(&target);
            let distances = server.compute_distances(&glwe, &lwe);

            let expected = 5u64;
            assert_eq!(client.key.decrypt(&distances[0]), expected);
        }
        {
            // distance should be 2^2 = 4
            let data = vec![vec![0, 0, 1, 3u64]];
            let target = vec![0, 0, 1, 1u64];
            server.set_data(data);
            let (glwe, lwe) = client.make_query(&target);
            let distances = server.compute_distances(&glwe, &lwe);

            let expected = 4u64;
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
}
