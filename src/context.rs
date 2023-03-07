use tfhe::core_crypto::prelude::*;
use tfhe::shortint::prelude::*;

pub struct Context {
    pub params: Parameters,
    pub encryption_rng: EncryptionRandomGenerator<ActivatedRandomGenerator>,
    pub secret_rng: SecretRandomGenerator<ActivatedRandomGenerator>,
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
