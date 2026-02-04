#!/usr/bin/env python3
"""Load NuScenes-QA dataset with error handling for corrupted files."""

import os
from pathlib import Path
from datasets import load_dataset

# Default data directory
DEFAULT_DATA_DIR = "/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/experiments/full_training/data"


def load_nuscenes_qa(scene_type="day", split="train"):
    """Load dataset with streaming and error handling."""
    print(f"Loading NuScenes-QA {scene_type}/{split} with streaming...")

    samples = []
    ds = load_dataset(
        'KevinNotSmile/nuscenes-qa-mini',
        scene_type,
        split=split,
        streaming=True
    )

    try:
        for s in ds:
            samples.append(s)
    except Exception as e:
        print(f"Stopped after {len(samples)} samples (error: {type(e).__name__})")

    print(f"Loaded {len(samples)} samples")
    return samples


def get_default_save_path(scene_type, split, data_dir=DEFAULT_DATA_DIR):
    """Get default save path for a given scene type and split."""
    return os.path.join(data_dir, f"nuscenes_qa_{scene_type}_{split}.pkl")


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-type", default="day", choices=["day", "night", "all"])
    parser.add_argument("--split", default="train", choices=["train", "validation", "all"])
    parser.add_argument("--save", type=str, help="Save to pickle file (default: auto-generate path)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Data directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save to file")
    args = parser.parse_args()

    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)

    # Determine what to download
    scene_types = ["day", "night"] if args.scene_type == "all" else [args.scene_type]
    splits = ["train", "validation"] if args.split == "all" else [args.split]

    for scene_type in scene_types:
        for split in splits:
            samples = load_nuscenes_qa(scene_type, split)

            if samples:
                print(f"\nFirst sample:")
                print(f"  Question: {samples[0]['question']}")
                print(f"  Answer: {samples[0]['answer']}")

            if not args.no_save:
                save_path = args.save if args.save else get_default_save_path(scene_type, split, args.data_dir)
                with open(save_path, 'wb') as f:
                    pickle.dump(samples, f)
                print(f"\nSaved to {save_path}")

            print("\n" + "="*60 + "\n")
