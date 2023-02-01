use bincode;
use ppknn::*;
use tfhe::shortint::prelude::*;

use std::fs;
use std::io::Cursor;
use std::time::Instant;

const DUMMY_KEY: &str = "dummy_key";

fn read_or_gen_keys() -> (ClientKey, ServerKey) {
    match fs::read(DUMMY_KEY) {
        Ok(s) => {
            let mut serialized_data = Cursor::new(&s);
            let client_key: ClientKey = bincode::deserialize_from(&mut serialized_data).unwrap();
            let server_key: ServerKey = bincode::deserialize_from(&mut serialized_data).unwrap();
            (client_key, server_key)
        }
        _ => {
            let (client_key, server_key) = gen_keys(PARAM_MESSAGE_4_CARRY_4);
            let mut serialized_data = Vec::new();
            bincode::serialize_into(&mut serialized_data, &client_key).unwrap();
            bincode::serialize_into(&mut serialized_data, &server_key).unwrap();
            fs::write(DUMMY_KEY, serialized_data).expect("unable to write to file");
            (client_key, server_key)
        }
    }
}

fn test_batcher() {
    for e in 0..10 {
        let k = 1 << e;
        let n = 1 << 10;
        let mut batcher = BatcherSort::new_k(vec![0; n], k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 7;
        let k = 2;
        let mut batcher = BatcherSort::new_k(vec![0; n], k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 7;
        let k = 3;
        let mut batcher = BatcherSort::new_k(vec![0; n], k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 10;
        let k = 2;
        let mut batcher = BatcherSort::new_k(vec![0; n], k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 10;
        let k = 3;
        let mut batcher = BatcherSort::new_k(vec![0; n], k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 177;
        let k = 5;
        let mut batcher = BatcherSort::new_k(vec![0; n], k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
    {
        let n = 177;
        let k = 7;
        let mut batcher = BatcherSort::new_k(vec![0; n], k);
        batcher.sort();
        println!("n={}, k={}, comparisons={}", n, k, batcher.comparisons());
    }
}

fn test_tfhe() {
    let keygen_start = Instant::now();
    let (client_key, server_key) = read_or_gen_keys();
    println!(
        "keygen/loading duration: {} ms",
        keygen_start.elapsed().as_millis()
    );

    let msg1 = 3;
    let msg2 = 1;
    let enc_start = Instant::now();
    let ct_1 = client_key.encrypt(msg1);
    let ct_2 = client_key.encrypt(msg2);
    println!(
        "encryption duration: {} ms",
        enc_start.elapsed().as_millis()
    );

    let gt_start = Instant::now();
    let ct_res = server_key.unchecked_greater(&ct_1, &ct_2);
    println!("gt duration: {} ms", gt_start.elapsed().as_millis());

    let modulus = client_key.parameters.message_modulus.0 as u64;
    let output = client_key.decrypt(&ct_res);
    assert_eq!(output, (msg1 > msg2) as u64 % modulus);

    // do more comparisons
    for m1 in 0..modulus {
        for m2 in 0..modulus {
            let ct_1 = client_key.encrypt(m1);
            let ct_2 = client_key.encrypt(m2);
            let ct_res = server_key.unchecked_greater(&ct_1, &ct_2);
            let output = client_key.decrypt(&ct_res);
            assert_eq!(output, (m1 > m2) as u64 % modulus as u64);
        }
    }

    println!("ok");
}

fn main() {
    test_batcher();
    // test_tfhe();
}
