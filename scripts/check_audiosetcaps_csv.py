#!/usr/bin/env python3
"""Check AudioSetCaps CSV structure."""

from huggingface_hub import hf_hub_download
import pandas as pd

# Download and check column names
path = hf_hub_download('baijs/AudioSetCaps', filename='Dataset/AudioSetCaps_caption.csv', repo_type='dataset')
df = pd.read_csv(path, nrows=5)

print('Columns:', list(df.columns))
print('\nFirst 3 rows:')
for i in range(min(3, len(df))):
    print(f"\nRow {i}:")
    for col, val in df.iloc[i].items():
        print(f"  {col}: {val}")
