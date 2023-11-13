#!/usr/bin/env bash

set -e
set -u

# USAGE, e.g.,
# ./bench-cancer 10 3 5
# first argument is the model size, the remaining are different values of k

FILENAME=data/cancer.csv
MODELSIZE=$1
KS="${@:2}" # https://stackoverflow.com/questions/2701400/remove-first-element-from-in-bash

_CHECK=$2 # check that there is at least one value of k
INITIAL_MODULUS=32

cargo build --release

for K in $KS; do
    ./target/release/ppknn \
        --file-name "$FILENAME" \
        --model-size "$MODELSIZE" \
        --test-size 200 \
        --repetitions 1 \
        --best-model \
        --network-type=file \
        -k "$K" \
        --initial-modulus "$INITIAL_MODULUS" \
        --quantize-type binary \
        --csv
done
