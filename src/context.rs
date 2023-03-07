use tfhe::core_crypto::prelude::*;
use tfhe::shortint::prelude::*;
use crate::codec::Codec;

pub struct Context {
    pub params: Parameters,
    pub encryption_rng: EncryptionRandomGenerator<ActivatedRandomGenerator>,
    pub secret_rng: SecretRandomGenerator<ActivatedRandomGenerator>,
    pub codec: Codec,
}

impl Context {
    pub fn new(params: Parameters) -> Self {
        let mut seeder = new_seeder();
        Self {
            params,
            encryption_rng: EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(
                seeder.seed(),
                seeder.as_mut(),
            ),
            secret_rng: SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed()),
            codec: Codec::new(params.message_modulus.0 as u64),
        }
    }

    pub fn gen_lwe_sk(&mut self) -> LweSecretKeyOwned<u64> {
        let lwe_sk = allocate_and_generate_new_binary_lwe_secret_key(
            self.params.lwe_dimension,
            &mut self.secret_rng,
        );
        lwe_sk
    }

    pub fn gen_glwe_sk(&mut self) -> GlweSecretKeyOwned<u64> {
        let glwe_sk = allocate_and_generate_new_binary_glwe_secret_key(
            self.params.glwe_dimension,
            self.params.polynomial_size,
            &mut self.secret_rng,
        );
        glwe_sk
    }
}

pub fn lwe_encode_encrypt(sk: &LweSecretKeyOwned<u64>, ctx: &mut Context, x: u64) -> LweCiphertextOwned<u64> {
    let mut x_copy = x;
    ctx.codec.encode(&mut x_copy);
    let pt = Plaintext(x_copy);
    allocate_and_encrypt_new_lwe_ciphertext(sk, pt, ctx.params.lwe_modular_std_dev, &mut ctx.encryption_rng)
}

pub fn lwe_decrypt_decode(sk: &LweSecretKeyOwned<u64>, ctx: &Context, ct: &LweCiphertextOwned<u64>) -> u64 {
    let mut pt = decrypt_lwe_ciphertext(sk, ct).0;
    ctx.codec.decode(&mut x);
    pt
}
