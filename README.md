# Privacy-preserving k-nearest neighbours

## Building and testing
```
cargo build --release
cargo test --release # might take a few minutes
```

## Running
```
./target/release/ppknn --file-name data/dummy.csv --model-size 40 --test-size 4 -k 3
```

## Internals

This is an improvement over the k-NN
paper by Zuber and Sirdey [0].
We use Batcher's odd-even sorting network
as described by Knuth [1].
The algorithm is modified to output `k`
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
