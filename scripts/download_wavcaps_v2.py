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

    # Extract audio from existing archives without re-downloading
    python scripts/download_wavcaps_v2.py --output-dir experiments/full_training/data/wavcaps --subset FreeSound --extract-only

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
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

HF_REPO = "https://huggingface.co/datasets/cvssp/WavCaps"


def _has_audio_files(directory: Path) -> bool:
    """Check if a directory already contains audio files."""
    if not directory.exists():
        return False

    patterns = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a")
    for pattern in patterns:
        if next(directory.rglob(pattern), None) is not None:
            return True
    return False


def _collect_split_parts(zip_dir: Path, subset: str) -> List[Path]:
    """Return ordered list of split zip parts for a subset (excluding the .zip tail)."""
    part_pattern = re.compile(rf"{re.escape(subset)}\.z(\d+)$", re.IGNORECASE)
    parts: List[tuple[int, Path]] = []

    if not zip_dir.exists():
        return []

    for candidate in zip_dir.iterdir():
        if not candidate.is_file():
            continue
        match = part_pattern.match(candidate.name)
        if match:
            parts.append((int(match.group(1)), candidate))

    parts.sort(key=lambda item: item[0])
    return [path for _, path in parts]


def _validate_split_parts(parts: List[Path]) -> None:
    """Ensure split archive parts form a contiguous sequence."""
    if not parts:
        return

    missing: List[str] = []
    current = 1
    for part in parts:
        suffix = part.name.split(".z")[-1]
        try:
            part_num = int(suffix)
        except ValueError:
            continue
        while current < part_num:
            missing.append(f"z{current:02d}")
            current += 1
        current = part_num + 1

    if missing:
        raise FileNotFoundError(
            "Missing split archive part(s): " + ", ".join(missing)
        )


def _find_subset_audio_dir(audio_root: Path, subset: str) -> Path:
    """Locate the directory holding extracted audio for a subset (case-insensitive)."""
    canonical = subset.lower()
    direct = audio_root / subset
    if direct.exists():
        return direct

    for candidate in audio_root.iterdir():
        if candidate.is_dir() and candidate.name.lower() == canonical:
            return candidate

    raise FileNotFoundError(
        f"Extracted audio directory not found for subset '{subset}' under {audio_root}"
    )


def extract_subset_audio(output_dir: Path, subset: str, *, force: bool = False) -> Path:
    """Extract (or verify) WavCaps audio for a given subset from split zip archives."""

    audio_root = output_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    zip_dir = output_dir / "Zip_files" / subset
    if not zip_dir.exists():
        raise FileNotFoundError(f"Zip directory not found for subset '{subset}': {zip_dir}")

    base_zip = zip_dir / f"{subset}.zip"
    if not base_zip.exists():
        raise FileNotFoundError(f"Main zip file missing for subset '{subset}': {base_zip}")
    base_zip = base_zip.resolve()

    parts = _collect_split_parts(zip_dir, subset)
    _validate_split_parts(parts)

    try:
        subset_dir = _find_subset_audio_dir(audio_root, subset)
    except FileNotFoundError:
        subset_dir = audio_root / subset

    if _has_audio_files(subset_dir) and not force:
        print(f"  ✓ Audio already extracted for {subset}: {subset_dir}")
        return subset_dir

    extractor = None
    if shutil.which("7z"):
        extractor = ["7z", "x", str(base_zip), f"-o{audio_root}", "-aos"]
    elif shutil.which("unzip"):
        extractor = ["unzip", "-n", str(base_zip), "-d", str(audio_root)]
    else:
        raise RuntimeError(
            "Neither '7z' nor 'unzip' is available on PATH. Install one to extract split archives."
        )

    print(f"\nExtracting {subset} audio using {' '.join(extractor[:1])}...")
    result = subprocess.run(
        extractor,
        cwd=zip_dir,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"Failed to extract {subset} archive (exit code {result.returncode}).\n"
            f"stderr: {stderr}"
        )

    subset_dir = _find_subset_audio_dir(audio_root, subset)

    if not _has_audio_files(subset_dir):
        raise RuntimeError(
            f"Extraction for {subset} completed but no audio files were found in {subset_dir}"
        )

    print(f"  ✓ Extracted audio to {subset_dir}")
    return subset_dir


def download_hf_file(repo_url: str, file_path: str, output_path: Path, resume: bool = True) -> bool:
    """
    Download a file from HuggingFace repository using wget or curl.

    Args:
        repo_url: Base repository URL
        file_path: Path within repository
        output_path: Local output path
        resume: Enable resume support for partial downloads

    Returns:
        True if download successful or file already complete, False otherwise
    """
    url = f"{repo_url}/resolve/main/{file_path}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and appears complete
    if output_path.exists():
        file_size = output_path.stat().st_size
        if file_size > 0:
            print(f"File already exists ({file_size / 1024 / 1024:.2f} MB): {output_path.name}")
            # Try to verify it's not a partial download by checking if it's readable
            try:
                # For JSON files, try to parse
                if output_path.suffix == '.json':
                    with open(output_path, 'r') as f:
                        json.load(f)
                    print(f"  ✓ Verified complete JSON file")
                    return True
                else:
                    # For zip files, just check size is reasonable
                    if file_size > 1024:  # At least 1KB
                        print(f"  ✓ File appears complete, skipping")
                        return True
            except Exception:
                print(f"  ⚠ File appears corrupted, re-downloading...")
                output_path.unlink()

    print(f"Downloading: {file_path}")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")

    # Try wget first (supports resume with -c)
    try:
        wget_args = ["wget", "-c", "-O", str(output_path), url] if resume else ["wget", "-O", str(output_path), url]
        result = subprocess.run(
            wget_args,
            capture_output=False,  # Show progress
            text=True,
        )
        if result.returncode == 0:
            print(f"  ✓ Downloaded successfully")
            return True
    except FileNotFoundError:
        pass

    # Fallback to curl (supports resume with -C -)
    try:
        curl_args = ["curl", "-L", "-C", "-", "-o", str(output_path), "-w", "%{http_code}", url] if resume else ["curl", "-L", "-o", str(output_path), "-w", "%{http_code}", url]
        result = subprocess.run(
            curl_args,
            capture_output=True,  # Capture to check HTTP code
            text=True,
        )

        # Check HTTP status code (curl prints it at the end)
        if result.returncode == 0:
            # Try to extract HTTP code from output
            http_code = result.stdout.strip().split('\n')[-1] if result.stdout else ""

            # Check if it's a valid success code
            if http_code.startswith('2'):  # 2xx success
                print(f"  ✓ Downloaded successfully")
                return True
            elif http_code == '404':
                print(f"  ✗ File not found (404)")
                # Remove the error page file if it exists
                if output_path.exists() and output_path.stat().st_size < 1024:
                    output_path.unlink()
                return False
            else:
                print(f"  ✗ HTTP error: {http_code}")
                return False
        else:
            print(f"  ✗ curl failed with return code {result.returncode}")
            return False
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
    """Download all zip parts for a subset with resume support."""
    zip_dir = output_dir / "Zip_files" / subset
    zip_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading zip files for {subset}...")
    print("WARNING: This may be very large (hundreds of GB)")
    print("Downloads support resume - you can safely interrupt and restart")

    # Progress tracking file
    progress_file = zip_dir / ".download_progress.json"
    progress = {}
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            print(f"  Resuming from previous download session...")
        except Exception:
            progress = {}

    downloaded_files = []

    # Try main .zip file
    main_zip = zip_dir / f"{subset}.zip"
    if main_zip.exists() and main_zip.stat().st_size > 0:
        print(f"  ✓ Main zip already exists: {main_zip.name}")
        downloaded_files.append(main_zip)
        progress['main_zip'] = True
    else:
        success = download_hf_file(
            HF_REPO,
            f"Zip_files/{subset}/{subset}.zip",
            main_zip,
            resume=True,
        )
        if success:
            downloaded_files.append(main_zip)
            progress['main_zip'] = True
            # Save progress
            with open(progress_file, 'w') as f:
                json.dump(progress, f)

    # Try multi-part zips (.z01, .z02, etc.)
    part_num = 1
    consecutive_failures = 0
    max_consecutive_failures = 3  # Stop after 3 consecutive 404s

    while True:
        part_file = zip_dir / f"{subset}.z{part_num:02d}"
        part_key = f"part_{part_num:02d}"

        # Check if already downloaded
        if part_file.exists() and part_file.stat().st_size > 0:
            if part_key not in progress or not progress[part_key]:
                print(f"  ✓ Part already exists: {part_file.name}")
            downloaded_files.append(part_file)
            progress[part_key] = True
            consecutive_failures = 0
            part_num += 1
            continue

        # Try to download
        success = download_hf_file(
            HF_REPO,
            f"Zip_files/{subset}/{subset}.z{part_num:02d}",
            part_file,
            resume=True,
        )

        if success:
            downloaded_files.append(part_file)
            progress[part_key] = True
            consecutive_failures = 0

            # Save progress after each successful download
            with open(progress_file, 'w') as f:
                json.dump(progress, f)

            part_num += 1
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"  No more parts found (tried {consecutive_failures} consecutive files)")
                break

        # Safety limit
        if part_num > 200:
            print(f"  Reached safety limit of 200 parts")
            break

    if not downloaded_files:
        raise RuntimeError(f"No zip files downloaded for {subset}")

    print(f"\n  ✓ Total zip parts found: {len(downloaded_files)}")
    print(f"  Progress saved to: {progress_file}")
    print(f"  You can safely interrupt and resume this download")

    return downloaded_files


def process_subset_to_jsonl(
    output_dir: Path,
    subset: str,
    audio_dir: Path,
) -> List[Dict[str, Any]]:
    """Convert WavCaps JSON metadata into SAFE-compatible JSONL entries."""

    # Load WavCaps JSON
    json_path = output_dir / "json_files" / subset / f"{subset}.json"

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    print(f"\nProcessing {subset} metadata...")

    # Verify file is valid JSON before parsing
    file_size = json_path.stat().st_size
    if file_size < 100:
        raise ValueError(f"JSON file too small ({file_size} bytes), likely corrupted: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            wavcaps_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file {json_path}: {e}")

    # WavCaps JSON format: {"data": [{...}, {...}]}
    if isinstance(wavcaps_data, dict) and "data" in wavcaps_data:
        samples = wavcaps_data["data"]
    elif isinstance(wavcaps_data, list):
        samples = wavcaps_data
    else:
        raise ValueError(f"Unexpected JSON format in {json_path}")

    print(f"  Found {len(samples)} samples")

    audio_dir = Path(audio_dir)
    if audio_dir.exists():
        if not _has_audio_files(audio_dir):
            print(f"  ⚠ Audio directory has no audio files yet: {audio_dir}")
    else:
        print(f"  ⚠ Audio directory not found (metadata-only run?): {audio_dir}")

    # Convert to SAFE format
    output_samples = []
    for idx, sample in enumerate(samples):
        # WavCaps fields: id, caption, audio (filename), duration, etc.
        audio_filename = sample.get("audio", "")
        if not audio_filename:
            continue

        # Caption/answer
        caption = sample.get("caption", sample.get("description", ""))

        rel_audio_path = Path("wavcaps") / "audio" / subset / audio_filename
        # Build SAFE-compatible entry
        safe_sample = {
            "id": f"{subset}_{idx:06d}",
            "question": "What is happening in the audio?",
            "answer": caption,
            "audio": rel_audio_path.as_posix(),
            "audio_path": rel_audio_path.as_posix(),
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
    return output_samples


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
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Skip downloading archives; only extract existing zip parts and rebuild metadata",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.extract_only and args.metadata_only:
        parser.error("--extract-only cannot be combined with --metadata-only")

    if args.extract_only and not (args.all_subsets or args.subset):
        parser.error("--extract-only requires --subset or --all-subsets to specify which archives to extract")

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
    aggregated_entries: List[Dict[str, Any]] = []

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
                if args.extract_only:
                    extract_subset_audio(output_dir, subset)
                else:
                    download_zip_files(output_dir, subset)
                    extract_subset_audio(output_dir, subset)
            except Exception as e:
                print(f"  ✗ Failed to download/extract audio: {e}")
                continue

        # Process to JSONL
        try:
            subset_entries = process_subset_to_jsonl(
                output_dir,
                subset,
                output_dir / "audio" / subset,
            )
            aggregated_entries.extend(subset_entries)
            total_samples += len(subset_entries)
        except Exception as e:
            print(f"  ✗ Failed to process metadata: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Output directory: {output_dir}")

    if aggregated_entries:
        train_jsonl_path = output_dir / "wavcaps_train.jsonl"
        with open(train_jsonl_path, "w", encoding="utf-8") as f:
            for entry in aggregated_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"  ✓ Aggregated metadata written to {train_jsonl_path}")
    else:
        print("  ⚠ No samples were processed. Check previous warnings for details.")

    if args.metadata_only:
        print("\nMetadata downloaded. To get audio:")
        print("  1. Download zip files manually from:")
        print(f"     {HF_REPO}/tree/main/Zip_files")
        print("  2. Extract to: {output_dir}/audio/<subset>/")
        print("  3. Audio paths in JSONL will match extracted structure")


if __name__ == "__main__":
    main()
