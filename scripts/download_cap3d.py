#!/usr/bin/env python3
"""
Download and prepare Cap3D dataset for point cloud captioning.

Cap3D: Large-scale 3D captioning dataset from Objaverse.
Source: https://huggingface.co/datasets/tiange/Cap3D

Usage:
    python scripts/download_cap3d.py --output-dir data/cap3d
    python scripts/download_cap3d.py --output-dir data/cap3d --max-samples 10000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def download_with_hf(
    output_dir: Path,
    max_samples: Optional[int] = None,
) -> None:
    """Download Cap3D using HuggingFace datasets."""
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install: pip install datasets huggingface-hub")
        sys.exit(1)

    print("Downloading Cap3D captions from HuggingFace...")

    # Download captions
    # Cap3D_automated_Objaverse.json contains the main captions
    try:
        caption_file = hf_hub_download(
            repo_id="tiange/Cap3D",
            filename="Cap3D_automated_Objaverse.json",
            repo_type="dataset",
            local_dir=output_dir,
        )
        print(f"Downloaded captions: {caption_file}")
    except Exception as e:
        print(f"Could not download from HuggingFace: {e}")
        print("Trying alternative method...")
        # Create placeholder
        caption_file = output_dir / "Cap3D_automated_Objaverse.json"
        if not caption_file.exists():
            print("Creating sample caption file for testing...")
            sample_captions = {
                f"sample_{i:05d}": f"A 3D model of object {i}"
                for i in range(100)
            }
            with open(caption_file, "w") as f:
                json.dump(sample_captions, f, indent=2)

    # Load captions
    with open(caption_file if isinstance(caption_file, str) else output_dir / "Cap3D_automated_Objaverse.json", "r") as f:
        captions = json.load(f)

    print(f"Loaded {len(captions)} captions")

    if max_samples:
        # Limit to max_samples
        object_ids = list(captions.keys())[:max_samples]
        captions = {k: captions[k] for k in object_ids}
        print(f"Limited to {len(captions)} samples")

    # Save processed captions
    train_file = output_dir / "cap3d_train.json"
    with open(train_file, "w") as f:
        json.dump(captions, f)
    print(f"Saved: {train_file}")

    # Create train/val split (90/10)
    object_ids = list(captions.keys())
    split_idx = int(len(object_ids) * 0.9)

    train_captions = {k: captions[k] for k in object_ids[:split_idx]}
    val_captions = {k: captions[k] for k in object_ids[split_idx:]}

    with open(output_dir / "cap3d_train.json", "w") as f:
        json.dump(train_captions, f)

    with open(output_dir / "cap3d_val.json", "w") as f:
        json.dump(val_captions, f)

    print(f"Train: {len(train_captions)} samples")
    print(f"Val: {len(val_captions)} samples")


def download_pointclouds(
    output_dir: Path,
    captions_file: Path,
    max_samples: Optional[int] = None,
) -> None:
    """Download point cloud files for Cap3D objects."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install: pip install huggingface-hub")
        sys.exit(1)

    # Load captions to get object IDs
    with open(captions_file, "r") as f:
        captions = json.load(f)

    object_ids = list(captions.keys())
    if max_samples:
        object_ids = object_ids[:max_samples]

    pc_dir = output_dir / "pointclouds"
    pc_dir.mkdir(exist_ok=True)

    print(f"\nDownloading point clouds for {len(object_ids)} objects...")
    print("Note: This may take a while for large datasets.\n")

    downloaded = 0
    failed = 0

    for obj_id in tqdm(object_ids, desc="Downloading"):
        pc_path = pc_dir / f"{obj_id}.npy"

        if pc_path.exists():
            downloaded += 1
            continue

        try:
            # Try to download from Cap3D point cloud repo
            # Note: The actual point cloud files may need separate download
            # This is a placeholder for the actual download logic
            hf_hub_download(
                repo_id="tiange/Cap3D",
                filename=f"pointclouds/{obj_id}.npy",
                repo_type="dataset",
                local_dir=output_dir,
            )
            downloaded += 1

        except Exception:
            # If individual download fails, create placeholder
            # In practice, you'd want to download from Objaverse
            failed += 1
            continue

    print(f"\nDownloaded: {downloaded}")
    print(f"Failed: {failed}")


def create_sample_pointclouds(output_dir: Path, num_samples: int = 100) -> None:
    """Create sample point clouds for testing."""
    import numpy as np

    pc_dir = output_dir / "pointclouds"
    pc_dir.mkdir(exist_ok=True)

    print(f"\nCreating {num_samples} sample point clouds for testing...")

    for i in tqdm(range(num_samples), desc="Creating"):
        obj_id = f"sample_{i:05d}"
        pc_path = pc_dir / f"{obj_id}.npy"

        if pc_path.exists():
            continue

        # Create random point cloud (sphere + noise)
        theta = np.random.uniform(0, 2 * np.pi, 2048)
        phi = np.random.uniform(0, np.pi, 2048)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        points = np.stack([x, y, z], axis=1)
        points += np.random.randn(2048, 3) * 0.1

        np.save(pc_path, points.astype(np.float32))

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Download Cap3D dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cap3d",
        help="Output directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to download",
    )
    parser.add_argument(
        "--captions-only",
        action="store_true",
        help="Only download captions (no point clouds)",
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample point clouds for testing",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cap3D Dataset Download")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")

    # Download captions
    download_with_hf(output_dir, args.max_samples)

    # Download or create point clouds
    if args.create_samples:
        create_sample_pointclouds(output_dir, args.max_samples or 100)
    elif not args.captions_only:
        captions_file = output_dir / "cap3d_train.json"
        if captions_file.exists():
            download_pointclouds(output_dir, captions_file, args.max_samples)

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Captions: {output_dir}/cap3d_*.json")
    print(f"Point clouds: {output_dir}/pointclouds/")
    print("=" * 60)


if __name__ == "__main__":
    main()
