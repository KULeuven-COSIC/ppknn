use criterion::{criterion_group, criterion_main, Criterion};
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

criterion_group!(benches, pbs_benchmark,);
criterion_main!(benches);
