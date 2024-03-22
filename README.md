# ppknn

[![Rust](https://github.com/kc1212/ppknn/actions/workflows/rust.yml/badge.svg)](https://github.com/kc1212/ppknn/actions/workflows/rust.yml)

This is the code used in the paper "[Revisiting Oblivious Top-k Selection with Applications to Secure k-NN Classification](https://eprint.iacr.org/2023/852)" by Kelong Cong, Robin Geelen, Jiayi Kang and Jeongeun Park.

*WARNING:*

This is proof-of-concept implementation. It may contain bugs and security issues. Please do not use in production systems.

## Building and testing
Only tested on x84-64, on Linux.
```
cargo build --release
cargo test --release # might take a few minutes
```

## Running
Use `./target/release/ppknn -h` to see the available options.
An example is shown below.
```
./target/release/ppknn --file-name data/dummy.csv --model-size 40 --test-size 4 -k 3
```

Providing a `.csv` file for the `--file-name` argument is mandatory.
This file holds the training and testing data
and should not contain values higher than 255.

For running longer experiments,
especially to reproduce the results from the paper,
see the scripts `scripts/bench-cancer.sh` and `scripts/bench-mnist.sh`.
