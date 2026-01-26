#!/usr/bin/env python3
"""
Download and prepare ModelNet40 dataset for point cloud training.

Downloads the HDF5 version from:
https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip

Usage:
    python scripts/download_modelnet40.py --output-dir data/modelnet40
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# URLs
MODELNET40_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
MODELNET40_URL_BACKUP = "https://cloud.tsinghua.edu.cn/f/b3d9fe3e2a514def96d9/?dl=1"  # Backup


def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f:
            with tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=desc,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

        return True

    except Exception as e:
        print(f"Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download ModelNet40 dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/modelnet40",
        help="Output directory",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ModelNet40 Dataset Download")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    # Check if already downloaded
    train_h5 = output_dir / "modelnet40_train.h5"
    test_h5 = output_dir / "modelnet40_test.h5"

    if train_h5.exists() and test_h5.exists():
        print("Dataset already exists. Skipping download.")
        print(f"  Train: {train_h5}")
        print(f"  Test: {test_h5}")
        return

    # Download
    zip_path = output_dir / "modelnet40_ply_hdf5_2048.zip"

    print("\nDownloading ModelNet40...")
    if not download_file(MODELNET40_URL, zip_path, "ModelNet40"):
        print("Primary URL failed, trying backup...")
        if not download_file(MODELNET40_URL_BACKUP, zip_path, "ModelNet40 (backup)"):
            print("ERROR: Could not download ModelNet40")
            sys.exit(1)

    # Extract
    print("\nExtracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # List contents
        names = zf.namelist()
        print(f"Archive contains {len(names)} files")

        # Extract all
        zf.extractall(output_dir)

    # Find and rename HDF5 files
    # The archive extracts to modelnet40_ply_hdf5_2048/ subdirectory
    extracted_dir = output_dir / "modelnet40_ply_hdf5_2048"

    if extracted_dir.exists():
        # Move train files
        train_files = list(extracted_dir.glob("*train*.h5"))
        test_files = list(extracted_dir.glob("*test*.h5"))

        if train_files:
            # Merge if multiple train files
            import h5py
            import numpy as np

            all_data = []
            all_labels = []

            for tf in sorted(train_files):
                with h5py.File(tf, "r") as f:
                    all_data.append(f["data"][:])
                    all_labels.append(f["label"][:])

            data = np.concatenate(all_data, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            with h5py.File(train_h5, "w") as f:
                f.create_dataset("data", data=data)
                f.create_dataset("label", data=labels)

            print(f"Created train set: {data.shape[0]} samples")

        if test_files:
            import h5py
            import numpy as np

            all_data = []
            all_labels = []

            for tf in sorted(test_files):
                with h5py.File(tf, "r") as f:
                    all_data.append(f["data"][:])
                    all_labels.append(f["label"][:])

            data = np.concatenate(all_data, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            with h5py.File(test_h5, "w") as f:
                f.create_dataset("data", data=data)
                f.create_dataset("label", data=labels)

            print(f"Created test set: {data.shape[0]} samples")

    # Cleanup
    if not args.keep_zip:
        print("\nCleaning up...")
        zip_path.unlink()
        if extracted_dir.exists():
            import shutil
            shutil.rmtree(extracted_dir)

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Train: {train_h5}")
    print(f"Test: {test_h5}")
    print("=" * 60)


if __name__ == "__main__":
    main()
