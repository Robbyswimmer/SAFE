#!/usr/bin/env python3
"""
Download and prepare ModelNet40 dataset for point cloud training.

Sources:
- HuggingFace: https://huggingface.co/datasets/Msun/modelnet40
- Original: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
- GitHub: https://github.com/antao97/PointCloudDatasets

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

# Reliable download URLs (in order of preference)
DOWNLOAD_URLS = [
    # HuggingFace mirror (most reliable)
    "https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip?download=true",
    # Original Stanford source
    "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
]


def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        response.raise_for_status()

        # Check if we got actual content (not HTML error page)
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type and response.headers.get("content-length", "0") == "0":
            print(f"  Got HTML instead of file from {url}")
            return False

        total_size = int(response.headers.get("content-length", 0))

        # Sanity check - ModelNet40 should be ~435MB
        if total_size > 0 and total_size < 1_000_000:  # Less than 1MB is wrong
            print(f"  File too small ({total_size} bytes), likely error page")
            return False

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

        # Verify file size after download
        actual_size = dest.stat().st_size
        if actual_size < 100_000_000:  # Less than 100MB is suspicious
            print(f"  Downloaded file too small ({actual_size} bytes)")
            dest.unlink()
            return False

        return True

    except Exception as e:
        print(f"  Download failed: {e}")
        if dest.exists():
            dest.unlink()
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

    print("\nDownloading ModelNet40 (~435MB)...")

    download_success = False
    for i, url in enumerate(DOWNLOAD_URLS):
        print(f"\nTrying source {i+1}/{len(DOWNLOAD_URLS)}: {url[:60]}...")
        if download_file(url, zip_path, "ModelNet40"):
            download_success = True
            break

    if not download_success:
        print("\nERROR: Could not download ModelNet40 from any source.")
        print("\nManual download instructions:")
        print("1. Go to: https://huggingface.co/datasets/Msun/modelnet40")
        print("2. Download: modelnet40_ply_hdf5_2048.zip")
        print(f"3. Place in: {output_dir}")
        print("4. Re-run this script")
        sys.exit(1)

    # Verify it's a valid zip
    print("\nVerifying download...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Check it has expected files
            names = zf.namelist()
            h5_files = [n for n in names if n.endswith(".h5")]
            print(f"  Archive contains {len(names)} files, {len(h5_files)} HDF5 files")

            if len(h5_files) == 0:
                raise zipfile.BadZipFile("No HDF5 files in archive")
    except zipfile.BadZipFile as e:
        print(f"ERROR: Invalid zip file: {e}")
        zip_path.unlink()
        sys.exit(1)

    # Extract
    print("\nExtracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    # Find and consolidate HDF5 files
    # The archive extracts to modelnet40_ply_hdf5_2048/ subdirectory
    extracted_dir = output_dir / "modelnet40_ply_hdf5_2048"

    if extracted_dir.exists():
        import h5py
        import numpy as np

        # Merge train files
        train_files = sorted(extracted_dir.glob("*train*.h5"))
        if train_files:
            print(f"\nMerging {len(train_files)} train files...")
            all_data = []
            all_labels = []

            for tf in train_files:
                with h5py.File(tf, "r") as f:
                    all_data.append(f["data"][:])
                    all_labels.append(f["label"][:])

            data = np.concatenate(all_data, axis=0)
            labels = np.concatenate(all_labels, axis=0).squeeze()

            with h5py.File(train_h5, "w") as f:
                f.create_dataset("data", data=data)
                f.create_dataset("label", data=labels)

            print(f"  Train set: {data.shape[0]} samples, {data.shape[1]} points each")

        # Merge test files
        test_files = sorted(extracted_dir.glob("*test*.h5"))
        if test_files:
            print(f"Merging {len(test_files)} test files...")
            all_data = []
            all_labels = []

            for tf in test_files:
                with h5py.File(tf, "r") as f:
                    all_data.append(f["data"][:])
                    all_labels.append(f["label"][:])

            data = np.concatenate(all_data, axis=0)
            labels = np.concatenate(all_labels, axis=0).squeeze()

            with h5py.File(test_h5, "w") as f:
                f.create_dataset("data", data=data)
                f.create_dataset("label", data=labels)

            print(f"  Test set: {data.shape[0]} samples, {data.shape[1]} points each")

    # Cleanup
    if not args.keep_zip:
        print("\nCleaning up...")
        zip_path.unlink()
        if extracted_dir.exists():
            import shutil
            shutil.rmtree(extracted_dir)

    # Verify final files
    if train_h5.exists() and test_h5.exists():
        print("\n" + "=" * 60)
        print("Download complete!")
        print(f"  Train: {train_h5}")
        print(f"  Test: {test_h5}")
        print("=" * 60)
    else:
        print("\nERROR: Final HDF5 files not created")
        sys.exit(1)


if __name__ == "__main__":
    main()
