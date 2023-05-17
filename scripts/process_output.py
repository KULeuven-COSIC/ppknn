#!/usr/bin/env python3

import pandas as pd
import sys

filename = sys.argv[1]
df = pd.read_csv(filename)
df = df[df['quantize_type'] == 'ternary']
# df = df.groupby(['k', 'model_size'])['total_dur', 'clear_ok', 'enc_ok'].mean()
df = df.groupby(['k', 'model_size']).agg(dur_mean=('total_dur', 'mean'), \
            clear_rate=('clear_ok', 'mean'), \
            enc_rate=('enc_ok', 'mean'), \
            count=('enc_ok', 'count'))
print(df)

# print(df.to_latex())

