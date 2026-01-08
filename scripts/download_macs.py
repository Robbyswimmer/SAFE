#!/usr/bin/env python3
"""
Download MACS (Multi-Annotator Captioned Sounds) dataset for SAFE training.

MACS is a high-quality audio captioning dataset with ~3K diverse sounds,
each with multiple human-written captions from different annotators.

The dataset is hosted on Zenodo. This script downloads and processes it.

Usage:
    python scripts/download_macs.py --output-dir experiments/full_training/data/macs

On cluster:
    python scripts/download_macs.py --output-dir $PWD/experiments/full_training/data/macs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv

try:
    import requests
except ImportError:
    raise SystemExit("requests package required. Install with: pip install requests")


# MACS dataset URLs (Zenodo)
MACS_ZENODO_RECORD = "5114771"
MACS_AUDIO_URL = f"https://zenodo.org/record/{MACS_ZENODO_RECORD}/files/MACS.zip"
MACS_CAPTIONS_URL = f"https://zenodo.org/record/{MACS_ZENODO_RECORD}/files/MACS_captions.csv"


def download_file(url: str, target: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress indication."""
    print(f"   Downloading: {url}", flush=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(target, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r   Progress: {pct:.1f}% ({downloaded // 1024 // 1024}MB)", end="", flush=True)

    print(f"\n   ✓ Downloaded to {target}", flush=True)


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract zip or tar archive."""
    print(f"   Extracting: {archive_path}", flush=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    elif archive_path.suffix in (".tar", ".gz", ".tgz"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_to)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")

    print(f"   ✓ Extracted to {extract_to}", flush=True)


def parse_macs_captions(captions_csv: Path) -> Dict[str, List[str]]:
    """
    Parse MACS captions CSV file.

    Returns dict mapping filename -> list of captions
    """
    filename_to_captions: Dict[str, List[str]] = {}

    with open(captions_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # MACS CSV has columns: filename, caption, annotator_id (or similar)
            # Try common column names
            filename = (
                row.get("filename")
                or row.get("file_name")
                or row.get("audio_filename")
                or row.get("sound_id")
            )
            caption = (
                row.get("caption")
                or row.get("sentence")
                or row.get("description")
                or row.get("text")
            )

            if filename and caption:
                # Normalize filename
                filename = filename.strip()
                if not filename.endswith(".wav"):
                    filename = filename + ".wav"

                caption = caption.strip()
                if caption:
                    if filename not in filename_to_captions:
                        filename_to_captions[filename] = []
                    # Avoid duplicates
                    if caption not in filename_to_captions[filename]:
                        filename_to_captions[filename].append(caption)

    return filename_to_captions


def process_macs_dataset(
    output_dir: Path,
    temp_dir: Path,
    overwrite: bool = False,
    train_ratio: float = 0.8,
) -> Tuple[int, int]:
    """
    Download and process MACS dataset.

    Returns (processed_count, skipped_count)
    """

    # Download files
    audio_zip = temp_dir / "MACS.zip"
    captions_csv = temp_dir / "MACS_captions.csv"

    if not audio_zip.exists() or overwrite:
        download_file(MACS_AUDIO_URL, audio_zip)
    else:
        print(f"   Using cached: {audio_zip}", flush=True)

    if not captions_csv.exists() or overwrite:
        download_file(MACS_CAPTIONS_URL, captions_csv)
    else:
        print(f"   Using cached: {captions_csv}", flush=True)

    # Extract audio
    audio_extract_dir = temp_dir / "MACS_audio"
    if not audio_extract_dir.exists():
        extract_archive(audio_zip, audio_extract_dir)

    # Find audio files (may be in subdirectory)
    audio_files = list(audio_extract_dir.rglob("*.wav"))
    if not audio_files:
        audio_files = list(audio_extract_dir.rglob("*.mp3"))
    if not audio_files:
        audio_files = list(audio_extract_dir.rglob("*.flac"))

    print(f"   Found {len(audio_files)} audio files", flush=True)

    # Parse captions
    filename_to_captions = parse_macs_captions(captions_csv)
    print(f"   Parsed captions for {len(filename_to_captions)} files", flush=True)

    # Create output directories
    train_audio_dir = output_dir / "audio" / "train"
    val_audio_dir = output_dir / "audio" / "val"
    train_audio_dir.mkdir(parents=True, exist_ok=True)
    val_audio_dir.mkdir(parents=True, exist_ok=True)

    # Process files with train/val split
    train_metadata: List[Dict] = []
    val_metadata: List[Dict] = []

    processed = 0
    skipped = 0

    # Deterministic split based on filename hash
    import hashlib

    for audio_file in sorted(audio_files):
        filename = audio_file.name

        # Find captions
        captions = filename_to_captions.get(filename, [])
        if not captions:
            # Try without extension
            base_name = audio_file.stem
            for key in filename_to_captions:
                if key.startswith(base_name):
                    captions = filename_to_captions[key]
                    break

        if not captions:
            skipped += 1
            continue

        # Determine split (hash-based for reproducibility)
        file_hash = int(hashlib.md5(filename.encode()).hexdigest(), 16)
        is_train = (file_hash % 100) < (train_ratio * 100)

        if is_train:
            split = "train"
            target_dir = train_audio_dir
            metadata_list = train_metadata
        else:
            split = "val"
            target_dir = val_audio_dir
            metadata_list = val_metadata

        # Copy audio file
        target_path = target_dir / filename
        rel_path = Path("audio") / split / filename

        if overwrite or not target_path.exists():
            shutil.copy2(audio_file, target_path)

        metadata_list.append({
            "split": split,
            "sound_name": filename,
            "file_path": str(rel_path.as_posix()),
            "captions": captions,
        })
        processed += 1

    # Save metadata
    train_meta_path = output_dir / "MACS_train.json"
    val_meta_path = output_dir / "MACS_val.json"

    with open(train_meta_path, "w", encoding="utf-8") as f:
        json.dump(train_metadata, f, ensure_ascii=False, indent=2)

    with open(val_meta_path, "w", encoding="utf-8") as f:
        json.dump(val_metadata, f, ensure_ascii=False, indent=2)

    print(f"   Train: {len(train_metadata)} samples", flush=True)
    print(f"   Val: {len(val_metadata)} samples", flush=True)

    return processed, skipped


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download MACS dataset for SAFE audio captioning training"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/full_training/data/macs"),
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Temporary directory for downloads (defaults to <output-dir>/.downloads)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary download files",
    )

    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.temp_dir or (args.output_dir / ".downloads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MACS Dataset Download")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Temp directory: {temp_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print("=" * 60)

    try:
        processed, skipped = process_macs_dataset(
            output_dir=args.output_dir,
            temp_dir=temp_dir,
            overwrite=args.overwrite,
            train_ratio=args.train_ratio,
        )

        print("\n" + "=" * 60)
        print(f"✅ Download complete!")
        print(f"   Total processed: {processed} samples")
        print(f"   Skipped (no captions): {skipped} samples")
        print(f"   Location: {args.output_dir}")
        print("=" * 60)

    finally:
        if not args.keep_temp and temp_dir.exists():
            print(f"\n🧹 Cleaning up temp directory: {temp_dir}", flush=True)
            # Keep downloaded archives, only remove extracted files
            for item in temp_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)


if __name__ == "__main__":
    main()
