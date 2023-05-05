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

    #[clap(
        long,
        default_value_t = false,
        help = "whether to shuffle the model and test data"
    )]
    pub no_shuffle: bool,

    #[clap(short, long, default_value_t = false, help = "print more information")]
    pub verbose: bool,
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
) -> (Vec<(u64, u64)>, u128, u128, usize, f64) {
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
    let comparisons = sorter.comparisons();

    let decrypted_k: Vec<_> = sorter.inner()[..k]
        .iter()
        .map(|ct| ct.decrypt(&client.key))
        .collect();

    let first_noise = client.lwe_noise(&sorter.inner()[0].value, decrypted_k[0].0);
    (decrypted_k, dist_dur, server_dur, comparisons, first_noise)
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
    let params = PARAMS;
    let cli = Cli::parse();
    let f_handle = fs::File::open(cli.file_name).expect("csv file not found");
    let (model_vec, model_labels, test_vec, test_labels) = parse_csv(
        f_handle,
        cli.model_size,
        cli.test_size,
        cli.binary_threshold,
        cli.no_shuffle,
    );

    let (mut client, server) =
        setup_simulation(params, &model_vec, &model_labels, cli.initial_modulus);

    let mut actual_errs = 0usize;
    let mut clear_errs = 0usize;
    for (i, (target, expected)) in test_vec.into_iter().zip(test_labels).enumerate() {
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
        let (actual_full, dist_dur, total_dur, comparisons, noise) = simulate(
            params,
            &mut client,
            server.clone(),
            cli.k,
            &target,
            cli.verbose,
        );
        let actual_labels: Vec<_> = actual_full.iter().map(|(_, b)| *b).collect();
        let actual_maj = majority(&actual_labels);
        assert_eq!(actual_full.len(), cli.k);

        let (clear_full, max_dist) = clear_knn(cli.k, &model_vec, &model_labels, &target);
        let clear_labels: Vec<_> = clear_full.iter().map(|l| l.class).collect();
        let clear_maj = majority(&clear_labels);
        println!("dist_dur={dist_dur}ms, total_dur={total_dur}ms, comparisons={comparisons}, noise={noise:.2}, \
            actual_maj={actual_maj}, clear_maj={clear_maj}, expected={expected}, clear_ok={}, enc_ok={}",
            clear_maj==expected, actual_maj==expected);

        if actual_maj != expected {
            actual_errs += 1;
        }
        if clear_maj != expected {
            clear_errs += 1;
        }

        if cli.verbose {
            if actual_maj != expected {
                println!("[WARNING] prediction error");
            }
            println!(
                "[DEBUG] max_dist={max_dist}, initial_modulus={}",
                cli.initial_modulus
            );
            println!("[DEBUG] actual_full={actual_full:?}");
            println!("[DEBUG] clear_full={clear_full:?}");
        }
    }

    println!(
        "[SUMMARY]: \
        model_size={}, \
        test_size={}, \
        actual_errs={actual_errs}, \
        clear_errs={clear_errs}, \
        actual_accuracy={:.2}, \
        clear_accuracy={:.2}",
        cli.model_size,
        cli.test_size,
        1f64 - (actual_errs as f64 / cli.test_size as f64),
        1f64 - (clear_errs as f64 / cli.test_size as f64)
    );
}
