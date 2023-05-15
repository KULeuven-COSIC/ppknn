#!/usr/bin/env bash

set -e
set -u

# USAGE:
# ./bench 40 3 5
# first argument is the model size, the remaining are different values of k
_CHECK=$2 # check that there is at least one value of k

FILENAME=data/mnist-8x8.csv
MODELSIZE=$1
KS="${@:2}" # https://stackoverflow.com/questions/2701400/remove-first-element-from-in-bash

for k in $KS; do
    cargo run --release -- \
        --file-name "$FILENAME" \
        --model-size "$MODELSIZE" \
        --test-size 100 \
        --repetitions 10 \
        -k "$k" \
        --initial-modulus 64  \
        --binary-threshold 8 \
        --csv
done
