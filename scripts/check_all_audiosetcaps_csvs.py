#!/usr/bin/env python3
"""Check all AudioSetCaps CSV file formats."""

from huggingface_hub import hf_hub_download
import pandas as pd

csv_files = [
    "AudioSetCaps_caption.csv",
    "VGGSound_AudioSetCaps_caption.csv",
    "YouTube-8M_AudioSetCaps_caption.csv",
]

for csv_file in csv_files:
    print(f"\n{'='*70}")
    print(f"File: {csv_file}")
    print('='*70)

    try:
        path = hf_hub_download('baijs/AudioSetCaps', filename=f'Dataset/{csv_file}', repo_type='dataset')
        df = pd.read_csv(path, nrows=10)

        print(f"Columns: {list(df.columns)}")
        print(f"Total rows in sample: {len(df)}")

        print("\nFirst 5 sample IDs:")
        for i in range(min(5, len(df))):
            row_id = str(df.iloc[i]['id'])
            print(f"  {i}: {row_id}")

            # Parse it
            if row_id.startswith('Y'):
                youtube_id = row_id[1:]
            else:
                youtube_id = row_id

            print(f"      -> Parsed: {youtube_id} (length: {len(youtube_id)})")

    except Exception as e:
        print(f"Error: {e}")
