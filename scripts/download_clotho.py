#!/usr/bin/env python3
"""
Download Clotho dataset using aac-datasets package and prepare for SAFE training.

Clotho is a high-quality audio captioning dataset with ~6K Freesound clips,
each with 5 human-written captions.

Usage:
    pip install aac-datasets
    python scripts/download_clotho.py --output-dir experiments/full_training/data/clotho

On cluster:
    python scripts/download_clotho.py --output-dir $PWD/experiments/full_training/data/clotho
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from aac_datasets import Clotho
    from aac_datasets.utils.download import download_file
except ImportError:
    raise SystemExit(
        "aac-datasets package required. Install with: pip install aac-datasets"
    )


def save_audio_wav(target: Path, audio_array: np.ndarray, sample_rate: int) -> None:
    """Save audio array to WAV file."""
    import soundfile as sf

    np_array = np.asarray(audio_array, dtype=np.float32)

    # Ensure correct shape for soundfile (samples,) or (samples, channels)
    if np_array.ndim == 1:
        pass  # Already correct shape
    elif np_array.ndim == 2 and np_array.shape[0] < np_array.shape[1]:
        # Assume (channels, samples) -> transpose to (samples, channels)
        np_array = np_array.T

    sf.write(str(target), np_array, sample_rate)


def process_clotho_split(
    split_name: str,
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool = False,
    max_samples: Optional[int] = None,
) -> Tuple[int, int]:
    """Process a single Clotho split using aac-datasets."""

    print(f"\n📥 Loading Clotho split: {split_name}", flush=True)

    # Map split names for aac-datasets (uses 'dev', 'val', 'eval')
    aac_split_map = {
        "development": "dev",
        "train": "dev",
        "validation": "val",
        "val": "val",
        "evaluation": "eval",
        "test": "eval",
    }
    aac_split = aac_split_map.get(split_name, split_name)

    # Map to our output convention
    output_split_map = {
        "dev": "train",
        "val": "val",
        "eval": "test",
    }
    local_split = output_split_map.get(aac_split, split_name)

    # Load dataset using aac-datasets
    try:
        dataset = Clotho(
            root=str(cache_dir),
            subset=aac_split,
            download=True,
            verbose=1,
        )
    except Exception as e:
        print(f"   ⚠️  Failed to load Clotho {aac_split}: {e}", flush=True)
        return 0, 0

    audio_dir = output_dir / "audio" / local_split
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata_entries: List[Dict] = []
    processed = 0
    skipped = 0

    total = len(dataset)
    if max_samples is not None:
        total = min(total, max_samples)

    for idx in range(total):
        if idx % 100 == 0:
            print(f"   Processing {idx}/{total}...", flush=True)

        try:
            item = dataset[idx]

            # aac-datasets returns dict with 'audio', 'captions', 'fname', etc.
            audio_data = item.get("audio")
            if audio_data is None:
                skipped += 1
                continue

            # Audio is typically a tensor or numpy array
            if hasattr(audio_data, "numpy"):
                audio_array = audio_data.numpy()
            else:
                audio_array = np.asarray(audio_data, dtype=np.float32)

            # Get sample rate (Clotho is 44.1kHz)
            sample_rate = item.get("sr", item.get("sample_rate", 44100))

            # Get filename
            filename = item.get("fname", item.get("filename", f"clotho_{local_split}_{idx:05d}.wav"))
            if not filename.endswith(".wav"):
                filename = filename.rsplit(".", 1)[0] + ".wav"

            # Get captions (aac-datasets returns list of 5 captions)
            captions = item.get("captions", [])
            if isinstance(captions, str):
                captions = [captions]
            captions = [str(c).strip() for c in captions if str(c).strip()]

            if not captions:
                captions = [""]

            # Save audio file
            audio_path = audio_dir / filename
            rel_path = Path("audio") / local_split / filename

            if overwrite or not audio_path.exists():
                save_audio_wav(audio_path, audio_array, sample_rate)

            metadata_entries.append({
                "split": local_split,
                "sound_name": filename,
                "file_path": str(rel_path.as_posix()),
                "captions": captions,
            })
            processed += 1

        except Exception as e:
            print(f"   ⚠️  Error processing sample {idx}: {e}", flush=True)
            skipped += 1
            continue

    # Save metadata JSON
    meta_path = output_dir / f"Clotho_{local_split}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_entries, f, ensure_ascii=False, indent=2)

    return processed, skipped


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download Clotho dataset for SAFE audio captioning training"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/full_training/data/clotho"),
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Download cache directory (defaults to <output-dir>/.aac_cache)",
    )
    parser.add_argument(
        "--splits",
        default="dev,val,eval",
        help="Comma-separated list of splits to download (dev, val, eval)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing audio files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split (for testing)",
    )

    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / ".aac_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print("=" * 60)
    print("Clotho Dataset Download (via aac-datasets)")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Splits: {splits}")
    print("=" * 60)

    total_processed = 0
    total_skipped = 0

    for split in splits:
        processed, skipped = process_clotho_split(
            split_name=split,
            output_dir=args.output_dir,
            cache_dir=cache_dir,
            overwrite=args.overwrite,
            max_samples=args.max_samples,
        )
        total_processed += processed
        total_skipped += skipped
        print(f"   ✓ {split}: {processed} samples saved, {skipped} skipped", flush=True)

    print("\n" + "=" * 60)
    print(f"✅ Download complete!")
    print(f"   Total: {total_processed} samples")
    print(f"   Skipped: {total_skipped} samples")
    print(f"   Location: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
