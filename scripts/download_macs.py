#!/usr/bin/env python3
"""
Download MACS (Multi-Annotator Captioned Soundscapes) dataset using aac-datasets.

MACS contains ~3,930 audio files from TAU Urban Acoustic Scenes with multiple
human-written captions per audio.

Usage:
    pip install aac-datasets
    python scripts/download_macs.py --output-dir experiments/full_training/data/macs

On cluster:
    python scripts/download_macs.py --output-dir $PWD/experiments/full_training/data/macs
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from aac_datasets import MACS
except ImportError:
    raise SystemExit(
        "aac-datasets package required. Install with: pip install aac-datasets"
    )


def save_audio_wav(target: Path, audio_array: np.ndarray, sample_rate: int) -> None:
    """Save audio array to WAV file."""
    import soundfile as sf

    np_array = np.asarray(audio_array, dtype=np.float32)

    if np_array.ndim == 1:
        pass  # Already correct shape
    elif np_array.ndim == 2 and np_array.shape[0] < np_array.shape[1]:
        np_array = np_array.T

    sf.write(str(target), np_array, sample_rate)


def process_macs_dataset(
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool = False,
    train_ratio: float = 0.8,
    max_samples: Optional[int] = None,
) -> Tuple[int, int]:
    """Process MACS dataset using aac-datasets."""

    print(f"\n📥 Loading MACS dataset...", flush=True)

    # Load dataset using aac-datasets
    try:
        dataset = MACS(
            root=str(cache_dir),
            download=True,
            verbose=1,
        )
    except Exception as e:
        print(f"   ⚠️  Failed to load MACS: {e}", flush=True)
        return 0, 0

    # Create output directories
    train_audio_dir = output_dir / "audio" / "train"
    val_audio_dir = output_dir / "audio" / "val"
    train_audio_dir.mkdir(parents=True, exist_ok=True)
    val_audio_dir.mkdir(parents=True, exist_ok=True)

    train_metadata: List[Dict] = []
    val_metadata: List[Dict] = []

    processed = 0
    skipped = 0

    total = len(dataset)
    if max_samples is not None:
        total = min(total, max_samples)

    import hashlib

    for idx in range(total):
        if idx % 100 == 0:
            print(f"   Processing {idx}/{total}...", flush=True)

        try:
            item = dataset[idx]

            audio_data = item.get("audio")
            if audio_data is None:
                skipped += 1
                continue

            if hasattr(audio_data, "numpy"):
                audio_array = audio_data.numpy()
            else:
                audio_array = np.asarray(audio_data, dtype=np.float32)

            sample_rate = item.get("sr", item.get("sample_rate", 44100))

            filename = item.get("fname", item.get("filename", f"macs_{idx:05d}.wav"))
            if not filename.endswith(".wav"):
                filename = filename.rsplit(".", 1)[0] + ".wav"

            # Get captions
            captions = item.get("captions", [])
            if isinstance(captions, str):
                captions = [captions]
            captions = [str(c).strip() for c in captions if str(c).strip()]

            if not captions:
                captions = [""]

            # Deterministic train/val split
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

            audio_path = target_dir / filename
            rel_path = Path("audio") / split / filename

            if overwrite or not audio_path.exists():
                save_audio_wav(audio_path, audio_array, sample_rate)

            metadata_list.append({
                "split": split,
                "sound_name": filename,
                "file_path": str(rel_path.as_posix()),
                "captions": captions,
            })
            processed += 1

        except Exception as e:
            print(f"   ⚠️  Error processing sample {idx}: {e}", flush=True)
            skipped += 1
            continue

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
        "--cache-dir",
        type=Path,
        default=None,
        help="Download cache directory (defaults to <output-dir>/.aac_cache)",
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
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to process (for testing)",
    )

    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / ".aac_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MACS Dataset Download (via aac-datasets)")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print("=" * 60)

    processed, skipped = process_macs_dataset(
        output_dir=args.output_dir,
        cache_dir=cache_dir,
        overwrite=args.overwrite,
        train_ratio=args.train_ratio,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 60)
    print(f"✅ Download complete!")
    print(f"   Total processed: {processed} samples")
    print(f"   Skipped: {skipped} samples")
    print(f"   Location: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
