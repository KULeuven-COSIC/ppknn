use ppknn::*;
use tfhe::shortint::parameters::PARAM_MESSAGE_8_CARRY_0;
use tfhe::shortint::prelude::*;

use bincode;
use std::fs;
use std::io::Cursor;

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
            let (client_key, server_key) = gen_keys(PARAM_MESSAGE_8_CARRY_0);
            let mut serialized_data = Vec::new();
            bincode::serialize_into(&mut serialized_data, &client_key).unwrap();
            bincode::serialize_into(&mut serialized_data, &server_key).unwrap();
            fs::write(DUMMY_KEY, serialized_data).expect("unable to write to file");
            (client_key, server_key)
        }
    }
}

fn main() {
    for e in 0..10 {
        let k = 1 << e;
        let n = 1 << 10;
        let mut batcher = BatcherSort::new_k(vec![0; n], k);
        batcher.sort();
        println!("{}", batcher.comparisons());
    }

    let (client_key, server_key) = read_or_gen_keys();
    println!("keygen done");

    let msg1 = 255;
    let msg2 = 2;
    let ct_1 = client_key.encrypt(msg1);
    let ct_2 = client_key.encrypt(msg2);
    println!("encryption done");

    let ct_res = server_key.unchecked_greater(&ct_1, &ct_2);
    println!("greater done");

    let modulus = client_key.parameters.message_modulus.0;
    let output = client_key.decrypt(&ct_res);
    assert_eq!(output, (msg1 > msg2) as u64 % modulus as u64);
    println!("ok")
}
