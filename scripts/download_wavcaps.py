#!/usr/bin/env python3
"""
Download and process WavCaps dataset from HuggingFace.

This script downloads the WavCaps dataset (403,050 audio-caption pairs),
converts FLAC to WAV format, truncates to 10s maximum duration, and
generates AudioCaps-compatible JSONL metadata.

Usage:
    # Download all splits
    python scripts/download_wavcaps.py --output-dir experiments/full_training/data/wavcaps

    # Download specific subsets
    python scripts/download_wavcaps.py --output-dir experiments/full_training/data/wavcaps \
        --subsets FreeSound BBC_Sound_Effects

    # Multi-threaded processing
    python scripts/download_wavcaps.py --output-dir experiments/full_training/data/wavcaps \
        --num-workers 8

Dataset sources:
    - FreeSound: 262,300 samples (CC-BY licensed)
    - BBC Sound Effects: 31,201 samples
    - SoundBible: 1,320 samples
    - AudioSet Strongly-Labelled: 108,317 samples
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm

# Suppress warnings from datasets library
warnings.filterwarnings("ignore", category=FutureWarning)


def process_audio(
    audio_data: Dict[str, Any],
    output_path: Path,
    max_duration: float = 10.0,
    target_sr: int = 48000,
) -> bool:
    """
    Process audio: resample, truncate, convert to mono WAV.

    Args:
        audio_data: Audio dict with 'array' and 'sampling_rate' keys
        output_path: Path to save processed WAV file
        max_duration: Maximum duration in seconds (default 10s)
        target_sr: Target sample rate (default 48kHz)

    Returns:
        True if successful, False otherwise
    """
    try:
        waveform = audio_data["array"]
        original_sr = audio_data["sampling_rate"]

        # Convert to numpy if needed
        if isinstance(waveform, list):
            waveform = np.array(waveform, dtype=np.float32)
        elif torch.is_tensor(waveform):
            waveform = waveform.numpy()

        # Convert stereo to mono
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)

        # Resample if needed
        if original_sr != target_sr:
            # Use torchaudio for high-quality resampling
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            resampled = torchaudio.functional.resample(
                waveform_tensor, original_sr, target_sr
            )
            waveform = resampled.squeeze(0).numpy()

        # Truncate to max_duration
        max_samples = int(max_duration * target_sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        # Save as WAV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, waveform, target_sr)

        return True

    except Exception as e:
        print(f"Error processing {output_path}: {e}")
        return False


def process_sample(
    idx: int,
    sample: Dict[str, Any],
    output_dir: Path,
    subset_name: str,
    max_duration: float = 10.0,
) -> Dict[str, Any] | None:
    """
    Process a single WavCaps sample.

    Args:
        idx: Sample index
        sample: HuggingFace dataset sample
        output_dir: Output directory for audio files
        subset_name: Name of the subset (FreeSound, BBC, etc.)
        max_duration: Maximum audio duration

    Returns:
        Metadata dict for JSONL, or None if processing failed
    """
    try:
        # Generate unique ID
        sample_id = f"{subset_name}_{idx:06d}"

        # Audio filename
        audio_relpath = f"audio/{subset_name}/{sample_id}.wav"
        audio_path = output_dir / audio_relpath

        # Process audio
        success = process_audio(
            sample["audio"],
            audio_path,
            max_duration=max_duration,
        )

        if not success:
            return None

        # Extract caption
        caption = sample.get("caption", "")
        if not caption:
            return None

        # Create AudioCaps-compatible metadata
        metadata = {
            "id": sample_id,
            "question": "What is happening in the audio?",
            "answer": caption,
            "audio_path": audio_relpath,
            "subset": subset_name,
        }

        # Include optional fields if available
        if "duration" in sample:
            metadata["original_duration"] = sample["duration"]

        return metadata

    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return None


def download_wavcaps(
    output_dir: str | Path,
    subsets: List[str] | None = None,
    num_workers: int = 4,
    max_duration: float = 10.0,
    max_samples_per_subset: int | None = None,
) -> None:
    """
    Download and process WavCaps dataset.

    Args:
        output_dir: Output directory for processed data
        subsets: List of subsets to download (default: all)
        num_workers: Number of parallel workers
        max_duration: Maximum audio duration in seconds
        max_samples_per_subset: Limit samples per subset (for testing)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading WavCaps dataset to: {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"Max duration: {max_duration}s")
    print()

    # WavCaps has a single 'train' split with all data
    # The 'subset' field in each sample indicates the source
    print(f"Loading WavCaps dataset...")
    try:
        # Load the default configuration
        dataset = load_dataset("cvssp/WavCaps", split="train")

        # Limit samples if specified (for testing)
        if max_samples_per_subset:
            dataset = dataset.select(range(min(max_samples_per_subset, len(dataset))))

        print(f"  {len(dataset)} samples loaded")

    except Exception as e:
        print(f"Error loading WavCaps: {e}")
        return

    # Filter by subsets if specified
    if subsets is not None:
        print(f"Filtering subsets: {', '.join(subsets)}")
        filtered_samples = []
        for sample in dataset:
            # Check if sample has subset field matching requested subsets
            sample_subset = sample.get("dataset", sample.get("subset", "unknown"))
            if sample_subset in subsets:
                filtered_samples.append(sample)
        dataset = filtered_samples
        print(f"  {len(dataset)} samples after filtering")

    # Process samples with progress bar
    all_metadata: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, sample in enumerate(dataset):
            # Get subset name from sample
            subset_name = sample.get("dataset", sample.get("subset", "unknown"))
            futures.append(
                executor.submit(
                    process_sample,
                    idx,
                    sample,
                    output_dir,
                    subset_name,
                    max_duration,
                )
            )

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="  Processing samples",
        ):
            result = future.result()
            if result is not None:
                all_metadata.append(result)

    print(f"  {len(all_metadata)}/{len(dataset)} samples processed successfully")

    # Save metadata as JSONL
    metadata_path = output_dir / "wavcaps_train.jsonl"
    print(f"\nSaving metadata to: {metadata_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in all_metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"\n✓ Complete!")
    print(f"  Total samples: {len(all_metadata)}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Audio files: {output_dir / 'audio'}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and process WavCaps dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/full_training/data/wavcaps",
        help="Output directory (default: experiments/full_training/data/wavcaps)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Filter by subsets (default: all). The dataset will be checked for 'dataset' or 'subset' field",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Maximum audio duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples per subset (for testing)",
    )

    args = parser.parse_args()

    download_wavcaps(
        output_dir=args.output_dir,
        subsets=args.subsets,
        num_workers=args.num_workers,
        max_duration=args.max_duration,
        max_samples_per_subset=args.max_samples,
    )


if __name__ == "__main__":
    main()
