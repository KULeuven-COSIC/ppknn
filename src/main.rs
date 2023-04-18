use clap::Parser;
use ppknn::*;
use std::time::Instant;
use tfhe::shortint::prelude::*;

#[derive(Parser, Debug, Clone)]
#[clap(author, version, about="Privacy preserving k nearest neighbour", long_about = None)]
pub struct Cli {
    #[arg(long, help = "path to the file containing the training/testing set")]
    pub file_name: String,

    #[arg(long, default_value_t = 100, help = "size of the model")]
    pub model_size: usize,

    #[arg(long, default_value_t = 10, help = "size of the test")]
    pub test_size: usize,

    #[arg(long, default_value_t = 3, help = "k in knn")]
    pub k: usize,

    #[clap(long, help = "print more information")]
    pub verbose: bool,
}

#[allow(dead_code)]
fn test_batcher() {
    for e in 0..10 {
        let k = 1 << e;
        let n = 1 << 10;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 7;
        let k = 2;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 7;
        let k = 3;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 10;
        let k = 2;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 10;
        let k = 3;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 177;
        let k = 3;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 10;
        let k = 5;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.merge();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 177;
        let k = 5;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 177;
        let k = 7;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 1239;
        let k = 3;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 1239;
        let k = 5;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 1239;
        let k = 7;
        let mut batcher = BatcherSort::new_k(ClearCmp::boxed(vec![0; n]), k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
}

const PARAMS: Parameters = Parameters {
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

pub fn simulate(
    params: Parameters,
    k: usize,
    data: Vec<Vec<u64>>,
    labels: Vec<u64>,
    target: Vec<u64>,
) -> (Vec<(u64, u64)>, u128, u128) {
    let (mut client, server) =
        setup_with_data(params, data, labels, params.message_modulus.0 as u64 * 2);
    let (glwe, lwe) = client.make_query(&target);

    let server_start = Instant::now();
    let distances_labels = server.compute_distances_with_labels(&glwe, &lwe);
    let dist_dur = server_start.elapsed().as_millis();
    let mut sorter = BatcherSort::new_k(EncCmp::boxed(distances_labels, params, server), k);
    sorter.sort();
    let server_dur = server_start.elapsed().as_millis();

    (
        sorter.inner()[..k]
            .iter()
            .map(|ct| ct.decrypt(&client.key))
            .collect(),
        dist_dur,
        server_dur,
    )
}

fn main() {
    // test_batcher();
    let k = 3usize;
    let data = vec![vec![0, 7, 0, 0u64]; 40];
    let target = vec![3, 0, 0, 0u64];
    let labels = vec![0u64; 40];

    let (output, dist_dur, total_dur) = simulate(PARAMS, k, data, labels, target);
    assert_eq!(output.len(), k);

    println!("dist_dur={dist_dur}ms, total_dur={total_dur}ms");
    for i in 0..k {
        println!("output[{}]={:?}", i, output[i]);
    }
}
