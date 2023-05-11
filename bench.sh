#!/usr/bin/env bash

set -e
set -u

FILENAME=data/mnist-8x8.csv
MODELSIZE=$1

for k in 3 5 7; do
    cargo run --release -- \
        --file-name "$FILENAME" \
        --model-size "$MODELSIZE" \
        --test-size 100 \
        --repetitions 10 \
        -k "$k" \
        --initial-modulus 64  \
        --binary-threshold 8
done
