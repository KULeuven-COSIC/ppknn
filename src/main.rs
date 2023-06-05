use clap::{Parser, ValueEnum};
use ppknn::clear_knn;
use ppknn::*;
use std::cell::RefCell;
use std::fmt::{Debug, Display, Formatter};
use std::fs;
use std::rc::Rc;
use std::time::Instant;
use tfhe::shortint::prelude::*;

const MAX_MODEL: u64 = 16;

#[derive(ValueEnum, Clone, Copy)]
enum QuantizeType {
    None,
    Binary,
    Ternary,
}

impl Display for QuantizeType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizeType::None => write!(f, "none"),
            QuantizeType::Binary => write!(f, "binary"),
            QuantizeType::Ternary => write!(f, "ternary"),
        }
    }
}

impl Debug for QuantizeType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(Parser, Debug, Clone)]
#[clap(author, version, about="Privacy preserving k nearest neighbour", long_about = None)]
struct Cli {
    #[clap(
        long,
        default_value = "",
        help = "path to the file containing the training/testing set, read from stdin if empty"
    )]
    file_name: String,

    #[clap(long, default_value_t = 100, help = "size of the model")]
    model_size: usize,

    #[clap(long, default_value_t = 10, help = "size of the test")]
    test_size: usize,

    #[arg(short, default_value_t = 3, help = "k in knn")]
    k: usize,

    #[clap(
        long,
        default_value_t = 0,
        help = "compute the distance with higher message modulus"
    )]
    initial_modulus: u64,

    #[clap(long, default_value_t = QuantizeType::None)]
    quantize_type: QuantizeType,

    #[clap(long, default_value_t = false, help = "attempt to find the best model")]
    best_model: bool,

    #[clap(long, default_value_t = 1, help = "number of repetitions")]
    repetitions: usize,

    #[clap(long, default_value_t = false, help = "use csv output")]
    csv: bool,

    #[clap(long, default_value_t = false, help = "print the csv header and exit")]
    print_header: bool,

    #[clap(short, long, default_value_t = false, help = "print more information")]
    verbose: bool,
}

fn parse_csv(f_handle: fs::File, quantize_type: QuantizeType) -> Vec<Vec<u64>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(f_handle);

    let mut rows: Vec<_> = reader
        .records()
        .map(|res| {
            let record = res.unwrap();
            record
                .iter()
                .map(|s| s.parse().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();

    match quantize_type {
        QuantizeType::None => { /* do nothing */ }
        QuantizeType::Binary => {
            let threshold = MAX_MODEL / 2;
            let f = |x| {
                assert!(x <= MAX_MODEL);
                if x < threshold {
                    0
                } else {
                    1
                }
            };
            rows.iter_mut().for_each(|row| {
                row.iter_mut().rev().skip(1).for_each(|x| {
                    *x = f(*x);
                })
            });
        }
        QuantizeType::Ternary => {
            let third = (MAX_MODEL as f64 / 3.0).ceil() as u64;
            assert_eq!(third, 6);
            let f = |x| {
                if x < third {
                    0
                } else if x >= third && x < 2 * third {
                    1
                } else {
                    2
                }
            };
            rows.iter_mut().for_each(|row| {
                row.iter_mut().rev().skip(1).for_each(|x| {
                    *x = f(*x);
                })
            });
        }
    }

    rows
}

const PARAMS: Parameters = Parameters {
    message_modulus: MessageModulus(32),
    carry_modulus: CarryModulus(1),
    ..PARAM_MESSAGE_2_CARRY_3
};

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

fn main() {
    let params = PARAMS;
    let cli = Cli::parse();

    if cli.print_header {
        println!(
            "rep,k,model_size,test_size,quantize_type,dist_dur,total_dur,comparisons,noise,\
                    actual_maj,clear_maj,expected,clear_ok,enc_ok"
        );
        return;
    }

    let csv_file_name = cli.file_name;
    if csv_file_name.is_empty() {
        unimplemented!("reading from stdin not implemented");
    }

    let f_handle = fs::File::open(csv_file_name.clone()).expect("csv file not found");
    let all_rows = parse_csv(f_handle, cli.quantize_type);

    let mut actual_errs = 0usize;
    let mut clear_errs = 0usize;

    for rep in 0..cli.repetitions {
        let (model_vec, model_labels, test_vec, test_labels) = {
            if cli.best_model {
                if cli.verbose {
                    println!("[DEBUG] finding best model");
                }
                let (model_vec, model_labels, test_vec, test_labels, acc) =
                    clear_knn::find_best_model(cli.model_size, cli.test_size, cli.k, &all_rows);
                if cli.verbose {
                    println!("[DEBUG] expected accuracy: {}", acc);
                }
                (model_vec, model_labels, test_vec, test_labels)
            } else {
                clear_knn::split_model_test(cli.model_size, cli.test_size, all_rows.clone())
            }
        };

        // sanity check
        assert_eq!(model_vec.len(), cli.model_size);
        assert_eq!(test_vec.len(), cli.test_size);
        assert_eq!(model_labels.len(), cli.model_size);
        assert_eq!(test_labels.len(), cli.test_size);

        let (mut client, server) =
            setup_simulation(params, &model_vec, &model_labels, cli.initial_modulus);

        for (i, (target, expected)) in test_vec.into_iter().zip(test_labels).enumerate() {
            if cli.verbose {
                let ratio = client.delta() / client.dist_delta;
                println!("[DEBUG] target_no={i}");
                println!(
                    "[DEBUG] clear_distances_top10={:?}",
                    clear_knn::distances(&model_vec, &target)
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
            let actual_maj = clear_knn::majority(&actual_labels);
            assert_eq!(actual_full.len(), cli.k);

            let (clear_full, max_dist) =
                clear_knn::run_knn(cli.k, &model_vec, &model_labels, &target);
            let clear_labels: Vec<_> = clear_full.iter().map(|l| l.class).collect();
            let clear_maj = clear_knn::majority(&clear_labels);
            if cli.csv {
                println!(
                    "{rep},{},{},{},{},{dist_dur},{total_dur},{comparisons},{noise:.2},\
                    {actual_maj},{clear_maj},{expected},{},{}",
                    cli.k,
                    cli.model_size,
                    cli.test_size,
                    cli.quantize_type,
                    (clear_maj == expected) as u8,
                    (actual_maj == expected) as u8
                );
            } else {
                println!(
                    "rep={rep}, k={}, model_size={}, test_size={}, quantize_type={}, \
                    dist_dur={dist_dur}ms, total_dur={total_dur}ms, comparisons={comparisons}, noise={noise:.2}, \
                    actual_maj={actual_maj}, clear_maj={clear_maj}, expected={expected}, clear_ok={}, enc_ok={}",
                    cli.k,
                    cli.model_size,
                    cli.test_size,
                    cli.quantize_type,
                    (clear_maj==expected) as u8,
                    (actual_maj==expected) as u8
                );
            }

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
    }

    if cli.verbose {
        println!(
            "[SUMMARY]: \
        k={}, \
        model_size={}, \
        test_size={}, \
        actual_errs={actual_errs}, \
        clear_errs={clear_errs}, \
        actual_accuracy={:.2}, \
        clear_accuracy={:.2}",
            cli.k,
            cli.model_size,
            cli.test_size,
            1f64 - (actual_errs as f64 / (cli.repetitions * cli.test_size) as f64),
            1f64 - (clear_errs as f64 / (cli.repetitions * cli.test_size) as f64)
        );
    }
}
