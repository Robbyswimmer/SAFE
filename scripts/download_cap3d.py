#!/usr/bin/env python3
"""
Download and prepare Cap3D dataset for point cloud captioning.

Cap3D: Large-scale 3D captioning dataset from Objaverse (~660K objects).
Source: https://huggingface.co/datasets/tiange/Cap3D

Data structure on HuggingFace:
- Captions: Cap3D_automated_Objaverse_full.csv (CSV with object_id, caption)
- Point clouds: PointCloud_zips/compressed_pcs_{00-09}.zip (~15-20GB each, .ply files)

Usage:
    # Download captions only (fast)
    python scripts/download_cap3d.py --output-dir data/cap3d --captions-only

    # Download captions + first 2 point cloud shards
    python scripts/download_cap3d.py --output-dir data/cap3d --num-shards 2

    # Download everything (warning: ~150GB)
    python scripts/download_cap3d.py --output-dir data/cap3d --num-shards 10

    # Quick test with sample data
    python scripts/download_cap3d.py --output-dir data/cap3d --create-samples
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# HuggingFace dataset URLs
HF_REPO = "tiange/Cap3D"
CAPTIONS_FILE = "Cap3D_automated_Objaverse_full.csv"
POINTCLOUD_SHARDS = [f"PointCloud_zips/compressed_pcs_{i:02d}.zip" for i in range(10)]


def download_captions(output_dir: Path) -> Path:
    """Download Cap3D captions CSV from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install: pip install huggingface-hub")
        sys.exit(1)

    csv_path = output_dir / CAPTIONS_FILE
    if csv_path.exists():
        print(f"Captions already exist: {csv_path}")
        return csv_path

    print("Downloading Cap3D captions from HuggingFace...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=CAPTIONS_FILE,
            repo_type="dataset",
            local_dir=output_dir,
        )
        print(f"Downloaded captions: {downloaded}")
        return Path(downloaded)
    except Exception as e:
        print(f"Error downloading captions: {e}")
        raise


def parse_csv_to_json(csv_path: Path, output_dir: Path) -> Dict[str, str]:
    """Parse CSV captions and convert to JSON format with train/val/test splits."""
    print(f"Parsing captions from {csv_path}...")

    captions = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Skip header if present
        first_row = next(reader, None)
        if first_row and len(first_row) >= 2:
            # Check if it's a header
            if not first_row[0].startswith("0") and "caption" in first_row[1].lower():
                pass  # Skip header
            else:
                # First row is data
                obj_id, caption = first_row[0].strip(), first_row[1].strip()
                if obj_id and caption:
                    captions[obj_id] = caption

        for row in reader:
            if len(row) >= 2:
                obj_id, caption = row[0].strip(), row[1].strip()
                if obj_id and caption:
                    captions[obj_id] = caption

    print(f"Loaded {len(captions)} captions")

    # Shuffle for random splits
    object_ids = list(captions.keys())
    random.seed(42)  # Reproducible splits
    random.shuffle(object_ids)

    # 90/5/5 split
    n = len(object_ids)
    train_end = int(n * 0.90)
    val_end = int(n * 0.95)

    train_ids = object_ids[:train_end]
    val_ids = object_ids[train_end:val_end]
    test_ids = object_ids[val_end:]

    splits = {
        "train": {k: captions[k] for k in train_ids},
        "val": {k: captions[k] for k in val_ids},
        "test": {k: captions[k] for k in test_ids},
    }

    # Save split files
    for split_name, split_data in splits.items():
        split_file = output_dir / f"cap3d_{split_name}.json"
        with open(split_file, "w") as f:
            json.dump(split_data, f)
        print(f"Saved {split_name}: {len(split_data)} samples -> {split_file}")

    return captions


def download_pointcloud_shard(
    output_dir: Path,
    shard_idx: int,
) -> Optional[Path]:
    """Download a single point cloud shard from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install: pip install huggingface-hub")
        sys.exit(1)

    shard_name = f"compressed_pcs_{shard_idx:02d}.zip"
    shard_path = output_dir / "PointCloud_zips" / shard_name

    if shard_path.exists():
        print(f"Shard already exists: {shard_path}")
        return shard_path

    print(f"Downloading shard {shard_idx} (~15-20GB)...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"PointCloud_zips/{shard_name}",
            repo_type="dataset",
            local_dir=output_dir,
        )
        print(f"Downloaded: {downloaded}")
        return Path(downloaded)
    except Exception as e:
        print(f"Error downloading shard {shard_idx}: {e}")
        return None


def load_ply_file(ply_path: Path) -> Optional[np.ndarray]:
    """Load point cloud from PLY file."""
    try:
        # Try plyfile first
        from plyfile import PlyData
        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]
        x = np.array(vertex["x"])
        y = np.array(vertex["y"])
        z = np.array(vertex["z"])
        return np.stack([x, y, z], axis=1).astype(np.float32)
    except ImportError:
        pass

    try:
        # Fall back to open3d
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        return np.asarray(pcd.points).astype(np.float32)
    except ImportError:
        pass

    # Manual PLY parsing as last resort
    try:
        return _parse_ply_manual(ply_path)
    except Exception as e:
        print(f"Failed to load {ply_path}: {e}")
        return None


def _parse_ply_manual(ply_path: Path) -> np.ndarray:
    """Manually parse ASCII/binary PLY file."""
    with open(ply_path, "rb") as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode("utf-8", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse header
        n_vertices = 0
        is_binary = False
        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            if "binary" in line:
                is_binary = True

        if n_vertices == 0:
            raise ValueError("No vertices found in PLY header")

        # Read vertex data
        if is_binary:
            # Read binary data (assume float32 xyz)
            data = np.frombuffer(f.read(n_vertices * 12), dtype=np.float32)
            return data.reshape(n_vertices, 3)
        else:
            # Read ASCII data
            points = []
            for _ in range(n_vertices):
                line = f.readline().decode("utf-8").strip()
                vals = line.split()
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
            return np.array(points, dtype=np.float32)


def extract_and_convert_shard(
    shard_path: Path,
    output_dir: Path,
    num_points: int = 8192,
    max_objects: Optional[int] = None,
) -> Tuple[int, int]:
    """Extract PLY files from shard and convert to NPY format."""
    pc_dir = output_dir / "pointclouds"
    pc_dir.mkdir(exist_ok=True)

    converted = 0
    failed = 0

    print(f"Extracting and converting {shard_path.name}...")

    with zipfile.ZipFile(shard_path, "r") as zf:
        ply_files = [n for n in zf.namelist() if n.endswith(".ply")]

        if max_objects:
            ply_files = ply_files[:max_objects]

        for ply_name in tqdm(ply_files, desc="Converting"):
            # Extract object ID from filename
            obj_id = Path(ply_name).stem

            npy_path = pc_dir / f"{obj_id}.npy"
            if npy_path.exists():
                converted += 1
                continue

            try:
                # Extract to temp location
                zf.extract(ply_name, output_dir / "temp_ply")
                ply_path = output_dir / "temp_ply" / ply_name

                # Load and convert
                points = load_ply_file(ply_path)
                if points is None or len(points) == 0:
                    failed += 1
                    continue

                # Subsample if needed
                if len(points) > num_points:
                    indices = np.random.choice(len(points), num_points, replace=False)
                    points = points[indices]
                elif len(points) < num_points:
                    # Pad by repeating
                    pad_size = num_points - len(points)
                    pad_indices = np.random.choice(len(points), pad_size, replace=True)
                    points = np.concatenate([points, points[pad_indices]], axis=0)

                # Save as NPY
                np.save(npy_path, points)
                converted += 1

                # Clean up temp file
                ply_path.unlink()

            except Exception as e:
                failed += 1
                continue

    # Clean up temp directory
    temp_dir = output_dir / "temp_ply"
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    return converted, failed


def download_pointclouds(
    output_dir: Path,
    num_shards: int = 1,
    num_points: int = 8192,
    max_objects_per_shard: Optional[int] = None,
) -> None:
    """Download and convert point cloud shards."""
    print(f"\nDownloading {num_shards} point cloud shard(s)...")
    print("WARNING: Each shard is ~15-20GB. Total could be ~150GB for all 10 shards.")

    total_converted = 0
    total_failed = 0

    for shard_idx in range(min(num_shards, 10)):
        shard_path = download_pointcloud_shard(output_dir, shard_idx)
        if shard_path is None:
            continue

        converted, failed = extract_and_convert_shard(
            shard_path, output_dir, num_points, max_objects_per_shard
        )
        total_converted += converted
        total_failed += failed
        print(f"Shard {shard_idx}: converted={converted}, failed={failed}")

    print(f"\nTotal: converted={total_converted}, failed={total_failed}")


def create_sample_pointclouds(output_dir: Path, num_samples: int = 100) -> None:
    """Create sample point clouds for testing."""
    pc_dir = output_dir / "pointclouds"
    pc_dir.mkdir(exist_ok=True)

    print(f"\nCreating {num_samples} sample point clouds for testing...")

    # Also create matching sample captions
    sample_captions = {}
    shapes = ["sphere", "cube", "cylinder", "cone", "torus"]

    for i in tqdm(range(num_samples), desc="Creating"):
        obj_id = f"sample_{i:05d}"
        pc_path = pc_dir / f"{obj_id}.npy"

        shape = shapes[i % len(shapes)]
        sample_captions[obj_id] = f"A 3D model of a {shape}."

        if pc_path.exists():
            continue

        # Create random point cloud based on shape
        n_points = 8192
        if shape == "sphere":
            theta = np.random.uniform(0, 2 * np.pi, n_points)
            phi = np.random.uniform(0, np.pi, n_points)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points = np.stack([x, y, z], axis=1)
        elif shape == "cube":
            # Random points on cube surface
            face = np.random.randint(0, 6, n_points)
            u = np.random.uniform(-1, 1, n_points)
            v = np.random.uniform(-1, 1, n_points)
            points = np.zeros((n_points, 3))
            for f in range(6):
                mask = face == f
                axis = f // 2
                sign = 1 if f % 2 == 0 else -1
                other_axes = [a for a in range(3) if a != axis]
                points[mask, axis] = sign
                points[mask, other_axes[0]] = u[mask]
                points[mask, other_axes[1]] = v[mask]
        elif shape == "cylinder":
            theta = np.random.uniform(0, 2 * np.pi, n_points)
            z = np.random.uniform(-1, 1, n_points)
            x = np.cos(theta)
            y = np.sin(theta)
            points = np.stack([x, y, z], axis=1)
        elif shape == "cone":
            theta = np.random.uniform(0, 2 * np.pi, n_points)
            h = np.random.uniform(0, 1, n_points)
            r = 1 - h  # Radius decreases with height
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = h
            points = np.stack([x, y, z], axis=1)
        else:  # torus
            theta = np.random.uniform(0, 2 * np.pi, n_points)
            phi = np.random.uniform(0, 2 * np.pi, n_points)
            R, r = 1.0, 0.3
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            points = np.stack([x, y, z], axis=1)

        # Add small noise
        points += np.random.randn(n_points, 3) * 0.02
        np.save(pc_path, points.astype(np.float32))

    # Save sample captions
    for split in ["train", "val", "test"]:
        split_file = output_dir / f"cap3d_{split}.json"
        if not split_file.exists():
            # Use all samples for train, subset for val/test
            if split == "train":
                data = sample_captions
            elif split == "val":
                keys = list(sample_captions.keys())[: num_samples // 10]
                data = {k: sample_captions[k] for k in keys}
            else:
                keys = list(sample_captions.keys())[num_samples // 10 : num_samples // 5]
                data = {k: sample_captions[k] for k in keys}
            with open(split_file, "w") as f:
                json.dump(data, f)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Download Cap3D dataset for point cloud captioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download captions only (fast, ~100MB)
  python scripts/download_cap3d.py --output-dir data/cap3d --captions-only

  # Download captions + 1 point cloud shard (~20GB)
  python scripts/download_cap3d.py --output-dir data/cap3d --num-shards 1

  # Download all data (~150GB total)
  python scripts/download_cap3d.py --output-dir data/cap3d --num-shards 10

  # Create sample data for testing (no download)
  python scripts/download_cap3d.py --output-dir data/cap3d --create-samples
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cap3d",
        help="Output directory (default: data/cap3d)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of point cloud shards to download (0-10, each ~15-20GB)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=8192,
        help="Number of points per cloud after subsampling (default: 8192)",
    )
    parser.add_argument(
        "--captions-only",
        action="store_true",
        help="Only download captions (skip point clouds)",
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample data for testing (no download)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of sample point clouds to create (with --create-samples)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cap3D Dataset Download")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'sample data' if args.create_samples else 'captions only' if args.captions_only else f'{args.num_shards} shard(s)'}")

    if args.create_samples:
        # Create sample data for testing
        create_sample_pointclouds(output_dir, args.num_samples)
    else:
        # Download real captions
        csv_path = download_captions(output_dir)
        parse_csv_to_json(csv_path, output_dir)

        # Download point clouds if requested
        if not args.captions_only and args.num_shards > 0:
            download_pointclouds(
                output_dir,
                num_shards=args.num_shards,
                num_points=args.num_points,
            )

    # Print summary
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"Captions: {output_dir}/cap3d_{{train,val,test}}.json")
    print(f"Point clouds: {output_dir}/pointclouds/")

    # Show stats
    for split in ["train", "val", "test"]:
        split_file = output_dir / f"cap3d_{split}.json"
        if split_file.exists():
            with open(split_file, "r") as f:
                data = json.load(f)
            print(f"  {split}: {len(data)} samples")

    pc_dir = output_dir / "pointclouds"
    if pc_dir.exists():
        npy_files = list(pc_dir.glob("*.npy"))
        print(f"  Point clouds: {len(npy_files)} files")

    print("\nTo use with training:")
    print(f"  python train_pointcloud.py --data-path {output_dir} --dataset cap3d")


if __name__ == "__main__":
    main()
