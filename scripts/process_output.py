#!/usr/bin/env python3

import pandas as pd
import sys

filename = sys.argv[1]
df = pd.read_csv(filename)
# zz = df.groupby(['k']).agg(dur_mean=('total_dur', 'mean'), clear_n=('clear_ok', 'sum'), enc_n=('enc_ok', 'sum'))
df = df.groupby(['k', 'model_size'])['total_dur', 'clear_ok', 'enc_ok'].mean()
print(df)
# print(df.to_latex())

