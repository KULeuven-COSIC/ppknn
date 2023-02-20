pub mod batcher;
pub mod comparator;

pub use batcher::*;
pub use comparator::*;

use tfhe::shortint::prelude::*;
use std::fs;
use std::io::Cursor;

const DUMMY_KEY: &str = "dummy_key";

pub fn read_or_gen_keys(param: &Parameters) -> (ClientKey, ServerKey) {
    match fs::read(DUMMY_KEY) {
        Ok(s) => {
            let mut serialized_data = Cursor::new(&s);
            let client_key: ClientKey = bincode::deserialize_from(&mut serialized_data).unwrap();
            let server_key: ServerKey = bincode::deserialize_from(&mut serialized_data).unwrap();
            assert_eq!(client_key.parameters, *param);
            (client_key, server_key)
        }
        _ => {
            let (client_key, server_key) = gen_keys(*param);
            let mut serialized_data = Vec::new();
            bincode::serialize_into(&mut serialized_data, &client_key).unwrap();
            bincode::serialize_into(&mut serialized_data, &server_key).unwrap();
            fs::write(DUMMY_KEY, serialized_data).expect("unable to write to file");
            (client_key, server_key)
        }
    }
}

pub fn enc_vec(vs: &[u64], client_key: &ClientKey) -> Vec<Ciphertext> {
    vs.iter().map(|x| {
        client_key.encrypt(*x)
    }).collect()
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_enc_sort() {
        {
            let (client_key, server_key) = read_or_gen_keys(&PARAM_MESSAGE_2_CARRY_2);
            let pt_vec = vec![1, 0, 2, 3u64];
            let enc_cmp = EncCmp::boxed(enc_vec(&pt_vec, &client_key), &client_key.parameters, server_key);

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let output = client_key.decrypt(&sorter.inner()[0]);
            assert_eq!(output, 0);
        }
        {
            let (client_key, server_key) = read_or_gen_keys(&PARAM_MESSAGE_2_CARRY_2);
            let pt_vec = vec![1, 2, 0, 3u64];
            let enc_cmp = EncCmp::boxed(enc_vec(&pt_vec, &client_key), &client_key.parameters, server_key);

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let output = client_key.decrypt(&sorter.inner()[0]);
            assert_eq!(output, 0);
        }
        {
            let (client_key, server_key) = read_or_gen_keys(&PARAM_MESSAGE_2_CARRY_2);
            let pt_vec = vec![2, 2, 1, 3u64];
            let enc_cmp = EncCmp::boxed(enc_vec(&pt_vec, &client_key), &client_key.parameters, server_key);

            let mut sorter = BatcherSort::new_k(enc_cmp, 1);
            sorter.sort();

            let output = client_key.decrypt(&sorter.inner()[0]);
            assert_eq!(output, 1);
        }
    }
}
