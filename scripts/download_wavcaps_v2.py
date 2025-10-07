#!/usr/bin/env python3
"""
Download WavCaps dataset from HuggingFace repository.

This script downloads WavCaps metadata (JSON) and optionally the audio zip files,
then processes them into the format expected by SAFE training.

Usage:
    # Download only JSON metadata (177MB)
    python scripts/download_wavcaps_v2.py --output-dir experiments/full_training/data/wavcaps --metadata-only

    # Download specific subset with audio
    python scripts/download_wavcaps_v2.py --output-dir experiments/full_training/data/wavcaps --subset FreeSound

    # Download all (WARNING: 819GB!)
    python scripts/download_wavcaps_v2.py --output-dir experiments/full_training/data/wavcaps --all-subsets

Available subsets:
    - FreeSound: 262,300 samples (~653GB)
    - BBC_Sound_Effects: 31,201 samples
    - SoundBible: 1,232 samples
    - AudioSet_SL: 108,317 samples

Note: This is for ACADEMIC USE ONLY per WavCaps license.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path
from typing import List

HF_REPO = "https://huggingface.co/datasets/cvssp/WavCaps"


def download_hf_file(repo_url: str, file_path: str, output_path: Path) -> bool:
    """Download a file from HuggingFace repository using wget or curl."""
    url = f"{repo_url}/resolve/main/{file_path}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {file_path}")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")

    try:
        # Try wget first
        result = subprocess.run(
            ["wget", "-O", str(output_path), url],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    # Fallback to curl
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", str(output_path), url],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("Error: Neither wget nor curl found. Please install one.")
        return False


def download_json_metadata(output_dir: Path, subset: str) -> Path:
    """Download JSON metadata for a subset."""
    json_file = f"{subset}.json"
    json_path = output_dir / "json_files" / subset / json_file

    if json_path.exists():
        print(f"  JSON already exists: {json_path}")
        return json_path

    success = download_hf_file(
        HF_REPO,
        f"json_files/{subset}/{json_file}",
        json_path,
    )

    if success:
        print(f"  ✓ Downloaded: {json_path}")
        return json_path
    else:
        raise RuntimeError(f"Failed to download {json_file}")


def download_zip_files(output_dir: Path, subset: str) -> List[Path]:
    """Download all zip parts for a subset."""
    zip_dir = output_dir / "Zip_files" / subset
    zip_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading zip files for {subset}...")
    print("WARNING: This may be very large (hundreds of GB)")

    # Different subsets have different zip structures
    # FreeSound has .z01 through .z121 plus .zip
    # We'll try to download sequentially until we fail

    downloaded_files = []

    # Try main .zip file
    main_zip = zip_dir / f"{subset}.zip"
    if not main_zip.exists():
        success = download_hf_file(
            HF_REPO,
            f"Zip_files/{subset}/{subset}.zip",
            main_zip,
        )
        if success:
            downloaded_files.append(main_zip)
    else:
        downloaded_files.append(main_zip)

    # Try multi-part zips (.z01, .z02, etc.)
    part_num = 1
    while True:
        part_file = zip_dir / f"{subset}.z{part_num:02d}"

        if part_file.exists():
            downloaded_files.append(part_file)
            part_num += 1
            continue

        success = download_hf_file(
            HF_REPO,
            f"Zip_files/{subset}/{subset}.z{part_num:02d}",
            part_file,
        )

        if success:
            downloaded_files.append(part_file)
            part_num += 1
        else:
            # No more parts
            break

        # Safety limit
        if part_num > 200:
            break

    if not downloaded_files:
        raise RuntimeError(f"No zip files downloaded for {subset}")

    print(f"  ✓ Downloaded {len(downloaded_files)} zip parts")
    return downloaded_files


def process_subset_to_jsonl(
    output_dir: Path,
    subset: str,
    audio_dir: Path,
) -> int:
    """Convert WavCaps JSON to SAFE-compatible JSONL format."""

    # Load WavCaps JSON
    json_path = output_dir / "json_files" / subset / f"{subset}.json"

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    print(f"\nProcessing {subset} metadata...")
    with open(json_path, "r", encoding="utf-8") as f:
        wavcaps_data = json.load(f)

    # WavCaps JSON format: {"data": [{...}, {...}]}
    if isinstance(wavcaps_data, dict) and "data" in wavcaps_data:
        samples = wavcaps_data["data"]
    elif isinstance(wavcaps_data, list):
        samples = wavcaps_data
    else:
        raise ValueError(f"Unexpected JSON format in {json_path}")

    print(f"  Found {len(samples)} samples")

    # Convert to SAFE format
    output_samples = []
    for idx, sample in enumerate(samples):
        # WavCaps fields: id, caption, audio (filename), duration, etc.
        audio_filename = sample.get("audio", "")
        if not audio_filename:
            continue

        # Caption/answer
        caption = sample.get("caption", sample.get("description", ""))

        # Build SAFE-compatible entry
        safe_sample = {
            "id": f"{subset}_{idx:06d}",
            "question": "What is happening in the audio?",
            "answer": caption,
            "audio_path": f"{subset}/{audio_filename}",
            "subset": subset,
            "wavcaps_id": sample.get("id", ""),
        }

        # Optional fields
        if "duration" in sample:
            safe_sample["duration"] = sample["duration"]

        output_samples.append(safe_sample)

    # Save as JSONL
    jsonl_path = output_dir / f"wavcaps_{subset.lower()}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in output_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"  ✓ Saved {len(output_samples)} samples to {jsonl_path}")
    return len(output_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Download WavCaps dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/full_training/data/wavcaps",
        help="Output directory",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only JSON metadata (no audio)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["FreeSound", "BBC_Sound_Effects", "SoundBible", "AudioSet_SL"],
        help="Download specific subset",
    )
    parser.add_argument(
        "--all-subsets",
        action="store_true",
        help="Download all subsets (WARNING: 819GB!)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which subsets to process
    if args.all_subsets:
        subsets = ["FreeSound", "BBC_Sound_Effects", "SoundBible", "AudioSet_SL"]
    elif args.subset:
        subsets = [args.subset]
    else:
        # Default: just metadata for all
        subsets = ["FreeSound", "BBC_Sound_Effects", "SoundBible", "AudioSet_SL"]
        args.metadata_only = True
        print("No subset specified. Downloading metadata only for all subsets.")
        print("Use --subset <name> to download audio for a specific subset.\n")

    total_samples = 0

    for subset in subsets:
        print(f"\n{'='*60}")
        print(f"Processing: {subset}")
        print(f"{'='*60}")

        # Download JSON metadata
        try:
            json_path = download_json_metadata(output_dir, subset)
        except Exception as e:
            print(f"  ✗ Failed to download metadata: {e}")
            continue

        # Download audio if requested
        if not args.metadata_only:
            try:
                zip_files = download_zip_files(output_dir, subset)

                # Extract audio
                print(f"\nExtracting audio files...")
                audio_dir = output_dir / "audio" / subset
                audio_dir.mkdir(parents=True, exist_ok=True)

                # For multi-part zips, we need to use 7z or unzip
                # This is platform-specific
                print("  (Extraction code needed - use 7z or unzip)")

            except Exception as e:
                print(f"  ✗ Failed to download/extract audio: {e}")
                continue

        # Process to JSONL
        try:
            count = process_subset_to_jsonl(
                output_dir,
                subset,
                output_dir / "audio",
            )
            total_samples += count
        except Exception as e:
            print(f"  ✗ Failed to process metadata: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Output directory: {output_dir}")

    if args.metadata_only:
        print("\nMetadata downloaded. To get audio:")
        print("  1. Download zip files manually from:")
        print(f"     {HF_REPO}/tree/main/Zip_files")
        print("  2. Extract to: {output_dir}/audio/<subset>/")
        print("  3. Audio paths in JSONL will match extracted structure")


if __name__ == "__main__":
    main()
