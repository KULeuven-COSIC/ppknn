use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use criterion::{criterion_group, criterion_main, Criterion};
use ppknn::server::setup_with_data;
use ppknn::{network::*, AsyncEncComparator, EncItem};
use tfhe::shortint::prelude::*;

const PARAMS: Parameters = Parameters {
    message_modulus: MessageModulus(32),
    carry_modulus: CarryModulus(1),
    ..PARAM_MESSAGE_2_CARRY_3
};

fn pbs_benchmark(c: &mut Criterion) {
    let (client, server) = gen_keys(PARAMS);
    let ct = client.encrypt(1);

    c.bench_function("pbs", |b| {
        b.iter(|| {
            let _ = server.keyswitch_bootstrap(&ct);
        });
    });
}

fn ks_benchmark(c: &mut Criterion) {
    let dist_mod = PARAMS.message_modulus.0 * 2;
    let (client, server) = setup_with_data(PARAMS, &vec![], &vec![], dist_mod as u64);
    let ct = client.key.encrypt(1);

    c.bench_function("ks", |b| {
        b.iter(|| {
            let _ = server.lwe_to_glwe(&ct);
        });
    });
}

fn network_bench(c: &mut Criterion) {
    let d = 8usize;
    // let k = 1usize;
    let pb: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "data",
        // &format!("network-{}-{}.csv", d, k),
        "test_network4.csv",
    ]
    .iter()
    .collect();
    let network = load_network(pb.as_path()).unwrap();

    let dist_mod = PARAMS.message_modulus.0 * 2;
    // data and labels not actually used if we just need to use the comparator
    let (mut client, server) = setup_with_data(PARAMS, &vec![], &vec![], dist_mod as u64);
    let server = Arc::new(RwLock::new(server));
    let cmp = AsyncEncComparator::new(server, PARAMS);

    // just create dummy elements
    let actual = (0..d).map(|_| {
        EncItem::new(
            client.lwe_encrypt_with_delta(0, 0),
            client.lwe_encrypt_with_delta(0, 0),
        )
    });
    let a_actual: Vec<_> = actual.map(|x| Arc::new(Mutex::new(x))).collect();

    c.bench_function("enc trivial network", |b| {
        b.iter(|| {
            par_run_network_trivial(&network, cmp.clone(), &a_actual);
        });
    });

    c.bench_function("enc network", |b| {
        b.iter(|| {
            par_run_network(&network, cmp.clone(), &a_actual);
        });
    });
}

criterion_group!(benches, pbs_benchmark, ks_benchmark, network_bench);
criterion_main!(benches);
