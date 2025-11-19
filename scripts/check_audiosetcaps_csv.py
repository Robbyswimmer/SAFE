#!/usr/bin/env python3
"""Check AudioSetCaps CSV structure and test parsing."""

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

# Test parsing
print('\n\n=== Testing ID Parsing ===')
for i in range(min(3, len(df))):
    audio_id = str(df.iloc[i]['id'])
    print(f"\nOriginal ID: {audio_id}")

    # Remove 'Y' prefix
    if audio_id.startswith('Y'):
        audio_id = audio_id[1:]

    # Split by underscore
    parts = audio_id.split('_')
    print(f"Parts: {parts}")

    if len(parts) >= 3:
        youtube_id = '_'.join(parts[:-2])
        start_time = int(parts[-2])
        print(f"  -> youtube_id: {youtube_id}, start_time: {start_time}")
    elif len(parts) == 2:
        youtube_id = parts[0]
        start_time = int(parts[1])
        print(f"  -> youtube_id: {youtube_id}, start_time: {start_time}")
    else:
        youtube_id = audio_id
        start_time = 0
        print(f"  -> youtube_id: {youtube_id}, start_time: {start_time}")
