use ppknn::*;

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

/*
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
    let modulus = client_key.parameters.message_modulus.0 as u64;
    let min_acc = server_key.generate_accumulator_bivariate(|x, y| x.min(y) % modulus);

    let gt_start = Instant::now();
    let ct_res = server_key.keyswitch_programmable_bootstrap_bivariate(&ct_1, &ct_2, &min_acc);
    println!("min duration: {} ms", gt_start.elapsed().as_millis());

    let output = client_key.decrypt(&ct_res);
    assert_eq!(output, msg1.min(msg2));

    // do more comparisons
    for m1 in 0..modulus {
        for m2 in 0..modulus {
            let ct_1 = client_key.encrypt(m1);
            let ct_2 = client_key.encrypt(m2);
            let ct_res =
                server_key.keyswitch_programmable_bootstrap_bivariate(&ct_1, &ct_2, &min_acc);
            let output = client_key.decrypt(&ct_res);
            assert_eq!(output, m1.min(m2));
        }
    }

    println!("ok");
}
*/

fn main() {
    test_batcher();
}
