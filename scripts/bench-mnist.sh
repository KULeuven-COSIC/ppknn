#!/usr/bin/env bash

set -e
set -u

# USAGE, e.g.,
# ./bench-mnist 40 ternary 3 5
# first argument is the model size, the remaining are different values of k

FILENAME=data/mnist-8x8.csv
MODELSIZE=$1
QUANTIZE_TYPE=$2
KS="${@:3}" # https://stackoverflow.com/questions/2701400/remove-first-element-from-in-bash

_CHECK=$3 # check that there is at least one value of k

if [ "$QUANTIZE_TYPE" = "binary" ]; then
    INITIAL_MODULUS=64
elif [ "$QUANTIZE_TYPE" = "ternary" ]; then
    INITIAL_MODULUS=256
fi

cargo build --release

for K in $KS; do
    ./target/release/ppknn \
        --file-name "$FILENAME" \
        --model-size "$MODELSIZE" \
        --test-size 100 \
        --repetitions 10 \
        -k "$K" \
        --initial-modulus "$INITIAL_MODULUS" \
        --quantize-type "$QUANTIZE_TYPE" \
        --csv
done
