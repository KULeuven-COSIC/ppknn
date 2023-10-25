use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use clap::Parser;
use ppknn::network::*;
use ppknn::*;
use tfhe::shortint::prelude::*;

const PARAMS: Parameters = Parameters {
    message_modulus: MessageModulus(32),
    carry_modulus: CarryModulus(1),
    ..PARAM_MESSAGE_2_CARRY_3
};

#[derive(Parser, Debug, Clone)]
#[clap(author, version, about="run an encrypted sorting network", long_about = None)]
struct Cli {
    #[clap(long, default_value_t = 20, help = "input length or d")]
    input_length: usize,

    #[clap(long, default_value_t = 3, help = "output length or k")]
    output_length: usize,

    #[clap(long, default_value_t = false, help = "use trivial parallelization")]
    trivial: bool,
}

fn main() {
    let cli = Cli::parse();
    let d = cli.input_length;
    let k = cli.output_length;
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

    // start the network
    let start = Instant::now();
    if cli.trivial {
        par_run_network_trivial(&network, cmp.clone(), &a_actual);
    } else {
        par_run_network(&network, cmp.clone(), &a_actual);
    }
    let dur = start.elapsed().as_millis();
    println!("{:?}", dur);
}
