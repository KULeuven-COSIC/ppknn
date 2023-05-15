#!/usr/bin/env bash

set -e
set -u

# USAGE:
# ./bench 40 3 5
# first argument is the model size, the remaining are different values of k

FILENAME=data/mnist-8x8.csv
MODELSIZE=$1
QUANTIZE_TYPE=$2
KS="${@:3}" # https://stackoverflow.com/questions/2701400/remove-first-element-from-in-bash

_CHECK=$3 # check that there is at least one value of k

for k in $KS; do
    cargo run --release -- \
        --file-name "$FILENAME" \
        --model-size "$MODELSIZE" \
        --test-size 100 \
        --repetitions 10 \
        -k "$k" \
        --initial-modulus 64  \
        --quantize-type "$QUANTIZE_TYPE" \
        --csv
done
