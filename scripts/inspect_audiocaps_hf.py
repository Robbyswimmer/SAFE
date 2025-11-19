#!/usr/bin/env python3
"""Inspect the AudioCaps HuggingFace dataset structure."""

from datasets import load_dataset

print("Loading AudioCaps dataset (streaming mode to check structure)...")
ds = load_dataset("jp1924/AudioCaps", split="train", streaming=True)

print("\n=== Dataset Features ===")
print(ds.features)

print("\n=== First Sample ===")
sample = next(iter(ds))
print(f"Available keys: {list(sample.keys())}")

for key, value in sample.items():
    if key in ["audio", "audio_10s", "audio_segment", "clip"]:
        if isinstance(value, dict):
            print(f"\n{key}: dict with keys {list(value.keys())}")
            if "array" in value:
                import numpy as np
                arr = np.array(value["array"])
                print(f"  - array shape: {arr.shape}")
                print(f"  - sampling_rate: {value.get('sampling_rate')}")
                duration = len(arr) / value.get('sampling_rate', 1)
                print(f"  - duration: {duration:.2f} seconds")
        else:
            print(f"\n{key}: {type(value).__name__}")
    elif key in ["caption", "captions", "text"]:
        print(f"\n{key}: {value}")
    else:
        print(f"{key}: {type(value).__name__} = {str(value)[:100]}")
