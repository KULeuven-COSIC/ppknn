use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
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
    let d = 20usize;
    let k = 3usize;
    let pb: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "data",
        &format!("network-{}-{}.csv", d, k),
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

    for n_threads in [1, 2].iter() {
        let mut group = c.benchmark_group("enc network");
        group
            .sample_size(10)
            .measurement_time(Duration::from_secs(20))
            .warm_up_time(Duration::from_secs(5));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_threads),
            &n_threads,
            |b, &t| {
                b.iter(|| {
                    do_work(*t, &network, cmp.clone(), &a_actual);
                });
            },
        );
    }
}

criterion_group!(benches, pbs_benchmark, ks_benchmark, network_bench);
criterion_main!(benches);
