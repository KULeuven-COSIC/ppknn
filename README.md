# Efficient and Secure k-NN Classification from Improved Data-Oblivious Programs and Homomorphic Encryption

[![Rust](https://github.com/kc1212/ppknn/actions/workflows/rust.yml/badge.svg)](https://github.com/kc1212/ppknn/actions/workflows/rust.yml)

This is the code used in the paper "Efficient and Secure k-NN Classification from Improved Data-Oblivious Programs and Homomorphic Encryption" by Kelong Cong, Robin Geelen, Jiayi Kang and Jeongeun Park.

*WARNING:*

This is proof-of-concept implementation. It may contain bugs and security issues. Please do not use in production systems.

## Building and testing
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

## Internals

This is an improvement over the k-NN
paper by Zuber and Sirdey [0].
We use Batcher's odd-even sorting network
as described by Knuth [1].
The algorithm is optimized to output `k`
sorted elements instead of `n`,
where `n` is the length of the array.
This modification results in much fewer comparison,
which is expensive to do homomorphically,
when `k` is low.

The homomorphic encryption scheme we use is TFHE [2]
from the [tfhe-rs](tfhe.rs) library.
Some internal data structures are not exposed
by default in tfhe-rs, so we use a
[forked version](https://github.com/kc1212/tfhe-rs/tree/expose-sk).

## References

[0] https://petsymposium.org/popets/2021/popets-2021-0020.pdf

[1] The Art of Computer Programming, Vol. 3 Sorting and Searching, Donald E. Knuth

[2] https://eprint.iacr.org/2018/421.pdf
