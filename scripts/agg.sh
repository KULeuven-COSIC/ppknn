#!/usr/bin/env bash

# usage:
# ./agg.sh mnist
# the first argument is the prefix of the csv files in the current directory

set -e
set -u

PREFIX=$1

./target/release/ppknn --print-header
# echo "rep,k,model_size,test_size,quantize_type,dist_dur,total_dur,comparisons,noise,actual_maj,clear_maj,expected,clear_ok,enc_ok"
for f in "$PREFIX"*.csv; do
	cat "$f" | grep -v rep | grep -v SUMMARY
done
