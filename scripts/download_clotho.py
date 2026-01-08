#!/usr/bin/env python3
"""
Download Clotho dataset from Hugging Face and prepare for SAFE training.

Clotho is a high-quality audio captioning dataset with ~6K Freesound clips,
each with 5 human-written captions.

Usage:
    python scripts/download_clotho.py --output-dir experiments/full_training/data/clotho

On cluster:
    python scripts/download_clotho.py --output-dir $PWD/experiments/full_training/data/clotho
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from datasets import load_dataset, Audio
except ImportError:
    raise SystemExit(
        "datasets package required. Install with: pip install datasets soundfile"
    )


def save_audio_wav(target: Path, audio_array: np.ndarray, sample_rate: int) -> None:
    """Save audio array to WAV file."""
    import soundfile as sf

    np_array = np.asarray(audio_array, dtype=np.float32)

    # Ensure 2D array (channels, samples) -> (samples, channels) for soundfile
    if np_array.ndim == 1:
        np_array = np_array.reshape(-1, 1)
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
    """Process a single Clotho split."""

    print(f"\n📥 Loading Clotho split: {split_name}", flush=True)

    # Clotho on HuggingFace: https://huggingface.co/datasets/audiofolder/clotho
    # Alternative: https://huggingface.co/datasets/d0rj/clotho
    try:
        ds = load_dataset(
            "d0rj/clotho",
            split=split_name,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"   ⚠️  Failed to load from d0rj/clotho: {e}", flush=True)
        print("   Trying alternative source...", flush=True)
        ds = load_dataset(
            "clotho",
            split=split_name,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )

    # Map split names to local convention
    split_map = {
        "development": "train",
        "validation": "val",
        "evaluation": "test",
        "test": "test",
        "train": "train",
    }
    local_split = split_map.get(split_name, split_name)

    audio_dir = output_dir / "audio" / local_split
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata_entries: List[Dict] = []
    processed = 0
    skipped = 0

    # Limit samples if requested
    total = len(ds)
    if max_samples is not None:
        total = min(total, max_samples)

    for idx in range(total):
        if idx % 100 == 0:
            print(f"   Processing {idx}/{total}...", flush=True)

        sample = ds[idx]

        try:
            # Clotho has 'audio' field with array and sampling_rate
            audio_data = sample.get("audio")
            if audio_data is None:
                skipped += 1
                continue

            if isinstance(audio_data, dict):
                audio_array = np.asarray(audio_data["array"], dtype=np.float32)
                sample_rate = audio_data["sampling_rate"]
            else:
                skipped += 1
                continue

            # Get filename - Clotho uses 'file_name' field
            filename = sample.get("file_name") or sample.get("filename") or f"clotho_{local_split}_{idx:05d}.wav"
            if not filename.endswith(".wav"):
                filename = filename.rsplit(".", 1)[0] + ".wav"

            # Collect all 5 captions (Clotho has caption_1 through caption_5)
            captions = []
            for i in range(1, 6):
                cap = sample.get(f"caption_{i}") or sample.get(f"caption{i}")
                if cap and str(cap).strip():
                    captions.append(str(cap).strip())

            # Fallback to single caption field
            if not captions:
                cap = sample.get("caption") or sample.get("text")
                if cap:
                    captions.append(str(cap).strip())

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
        help="HuggingFace cache directory (defaults to <output-dir>/.hf_cache)",
    )
    parser.add_argument(
        "--splits",
        default="development,validation,evaluation",
        help="Comma-separated list of splits to download",
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
    cache_dir = args.cache_dir or (args.output_dir / ".hf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print("=" * 60)
    print("Clotho Dataset Download")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
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
