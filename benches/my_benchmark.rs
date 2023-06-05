use criterion::{criterion_group, criterion_main, Criterion};
use ppknn::server::setup_with_modulus;
use tfhe::shortint::prelude::*;

const PARAMS: Parameters = Parameters {
    message_modulus: MessageModulus(32),
    carry_modulus: CarryModulus(1),
    ..PARAM_MESSAGE_2_CARRY_3
};

pub fn pbs_benchmark(c: &mut Criterion) {
    let (client, server) = gen_keys(PARAMS);
    let ct = client.encrypt(1);

    c.bench_function("pbs", |b| {
        b.iter(|| {
            let _ = server.keyswitch_bootstrap(&ct);
        });
    });
}

pub fn ks_benchmark(c: &mut Criterion) {
    let dist_mod = PARAMS.message_modulus.0 * 2;
    let (client, server) = setup_with_modulus(PARAMS, dist_mod as u64);
    let ct = client.key.encrypt(1);

    c.bench_function("ks", |b| {
        b.iter(|| {
            let _ = server.lwe_to_glwe(&ct);
        });
    });
}

criterion_group!(benches, pbs_benchmark, ks_benchmark);
criterion_main!(benches);
