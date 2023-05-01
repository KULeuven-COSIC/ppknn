use clap::Parser;
use ppknn::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::rc::Rc;
use std::time::Instant;
use tfhe::shortint::prelude::*;

#[derive(Parser, Debug, Clone)]
#[clap(author, version, about="Privacy preserving k nearest neighbour", long_about = None)]
pub struct Cli {
    #[arg(
        long,
        help = "path to the file containing the training/testing set",
        required = true
    )]
    pub file_name: String,

    #[arg(long, default_value_t = 100, help = "size of the model")]
    pub model_size: usize,

    #[arg(long, default_value_t = 10, help = "size of the test")]
    pub test_size: usize,

    #[arg(short, default_value_t = 3, help = "k in knn")]
    pub k: usize,

    #[clap(
        long,
        default_value_t = 0,
        help = "compute the distance with higher message modulus"
    )]
    pub initial_modulus: u64,

    #[clap(
        long,
        default_value_t = 0,
        help = "convert feature values to binary using a threshold"
    )]
    pub binary_threshold: u64,

    #[clap(short, long, default_value_t = false, help = "print more information")]
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
    message_modulus: MessageModulus(32),
    carry_modulus: CarryModulus(1),
    ..PARAM_MESSAGE_2_CARRY_3 // ..PARAM_MESSAGE_1_CARRY_3
};

fn clear_squared_distance(xs: &[u64], ys: &[u64]) -> u64 {
    xs.iter()
        .zip(ys)
        .map(|(x, y)| {
            let out = if x > y { x - y } else { y - x };
            out * out
        })
        .sum()
}

fn clear_distances(data: &[Vec<u64>], target: &[u64]) -> Vec<u64> {
    data.iter()
        .map(|x| clear_squared_distance(x, target))
        .collect()
}

fn clear_knn(
    k: usize,
    model_vec: &[Vec<u64>],
    labels: &[u64],
    target: &[u64],
) -> (Vec<ClearItem>, u64) {
    let distances: Vec<_> = clear_distances(model_vec, target)
        .iter()
        .zip(labels.iter())
        .map(|(value, class)| ClearItem {
            value: *value,
            class: *class,
        })
        .collect();
    let max_dist = distances.iter().map(|item| item.value).max().unwrap();
    let mut batcher = BatcherSort::new_k(ClearCmp::boxed(distances), k);
    batcher.sort();
    (batcher.inner()[0..k].to_vec(), max_dist)
}

fn setup_simulation(
    params: Parameters,
    model_vec: &[Vec<u64>],
    labels: &[u64],
    initial_modulus: u64,
) -> (KnnClient, Rc<RefCell<KnnServer>>) {
    let (client, server) = setup_with_data(
        params,
        model_vec,
        labels,
        if initial_modulus == 0 {
            params.message_modulus.0 as u64
        } else {
            initial_modulus
        },
    );
    let server = Rc::new(RefCell::new(server));
    (client, server)
}

fn simulate(
    params: Parameters,
    client: &mut KnnClient,
    server: Rc<RefCell<KnnServer>>,
    k: usize,
    target: &[u64],
    verbose: bool,
) -> (Vec<(u64, u64)>, u128, u128) {
    let (glwe, lwe) = client.make_query(target);

    let server_start = Instant::now();
    let distances_labels = server.borrow().compute_distances_with_labels(&glwe, &lwe);

    if verbose {
        let distances: Vec<_> = distances_labels
            .iter()
            .take(10)
            .map(|item| {
                let value = client.key.decrypt(&item.value);
                let class = client.key.decrypt(&item.class);
                (value, class)
            })
            .collect();
        println!("[DEBUG] decrypted_distances_top10={distances:?}");
    }

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

fn majority(vs: &[u64]) -> u64 {
    assert!(vs.len() > 0);
    let max = vs
        .iter()
        .fold(HashMap::<u64, usize>::new(), |mut m, x| {
            *m.entry(*x).or_default() += 1;
            m
        })
        .into_iter()
        .max_by_key(|(_, v)| *v)
        .map(|(k, _)| k);
    max.unwrap()
}

fn main() {
    // test_batcher();
    let params = PARAMS;
    let cli = Cli::parse();
    let f_handle = fs::File::open(cli.file_name).expect("csv file not found");
    let (model_vec, model_labels, test_vec, test_labels) =
        parse_csv(f_handle, cli.model_size, cli.test_size, cli.binary_threshold);

    let (mut client, server) =
        setup_simulation(params, &model_vec, &model_labels, cli.initial_modulus);
    for (i, (target, expected_label)) in test_vec.into_iter().zip(test_labels).enumerate() {
        if cli.verbose {
            let ratio = client.delta() / client.dist_delta;
            println!("[DEBUG] target_no={i}");
            println!(
                "[DEBUG] clear_distances_top10={:?}",
                clear_distances(&model_vec, &target)
                    .into_iter()
                    .map(|d| { d / ratio })
                    .zip(model_labels.clone())
                    .take(10)
                    .collect::<Vec<_>>()
            )
        }
        let (output, dist_dur, total_dur) = simulate(
            params,
            &mut client,
            server.clone(),
            cli.k,
            &target,
            cli.verbose,
        );
        let output_labels: Vec<_> = output.iter().map(|(_, b)| *b).collect();
        let actual_label = majority(&output_labels);
        assert_eq!(output.len(), cli.k);
        println!("dist_dur={dist_dur}ms, total_dur={total_dur}ms, actual_label={actual_label}, expected_label={expected_label}");
        if actual_label != expected_label {
            println!("[WARNING] prediction error");
        }

        if cli.verbose {
            let (clear_result, max_dist) = clear_knn(cli.k, &model_vec, &model_labels, &target);
            println!(
                "[DEBUG] max_dist={max_dist}, initial_modulus={}",
                cli.initial_modulus
            );
            println!("[DEBUG] actual_full={output:?}");
            println!("[DEBUG] expected_full={clear_result:?}");
        }
    }
}
