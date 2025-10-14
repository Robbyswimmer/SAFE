#!/usr/bin/env python3
"""
Fix WavCaps dataset paths by scanning actual audio files and regenerating metadata.

This script addresses the common issue where JSONL metadata references paths that
don't match the actual audio file locations (e.g., symlinked directories, FLAC instead of WAV).

Usage:
    # Scan and fix all WavCaps metadata
    python scripts/fix_wavcaps_paths.py \
        --wavcaps-dir /data/.../experiments/full_training/data/wavcaps

    # Dry run to preview changes
    python scripts/fix_wavcaps_paths.py \
        --wavcaps-dir experiments/full_training/data/wavcaps \
        --dry-run

    # Specify output for corrected metadata
    python scripts/fix_wavcaps_paths.py \
        --wavcaps-dir experiments/full_training/data/wavcaps \
        --output wavcaps_train_fixed.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Supported audio formats
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def find_all_audio_files(audio_root: Path) -> List[Path]:
    """
    Recursively find all audio files under a directory.

    Returns:
        List of Path objects for audio files
    """
    audio_files: List[Path] = []

    if not audio_root.exists():
        print(f"Warning: Audio directory does not exist: {audio_root}")
        return audio_files

    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(audio_root.rglob(f"*{ext}"))

    return sorted(audio_files)


def detect_subset_from_path(audio_path: Path) -> str:
    """
    Infer the subset name from the file path.

    Common patterns:
        - BBC_Sound_Effects_flac/file.flac -> BBC_Sound_Effects
        - FreeSound/file.wav -> FreeSound
        - AudioSet_SL/file.flac -> AudioSet_SL
    """
    path_str = str(audio_path).lower()

    if "bbc" in path_str or "sound_effects" in path_str:
        return "BBC_Sound_Effects"
    elif "freesound" in path_str:
        return "FreeSound"
    elif "soundbible" in path_str:
        return "SoundBible"
    elif "audioset" in path_str:
        return "AudioSet_SL"
    else:
        # Use parent directory name
        parent = audio_path.parent.name
        return parent if parent != "audio" else "Unknown"


def load_existing_metadata(wavcaps_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load existing JSONL metadata files to preserve captions/questions.

    Returns:
        Dict mapping audio filenames (stem) to metadata entries
    """
    metadata_map: Dict[str, Dict[str, Any]] = {}

    # Look for JSONL files in the wavcaps directory
    jsonl_files = list(wavcaps_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"Warning: No JSONL metadata files found in {wavcaps_dir}")
        return metadata_map

    print(f"Loading existing metadata from {len(jsonl_files)} JSONL file(s)...")

    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)

                        # Extract filename from audio_path
                        audio_path = entry.get("audio_path") or entry.get("audio")
                        if audio_path:
                            # Get filename stem (without extension)
                            filename_stem = Path(audio_path).stem
                            metadata_map[filename_stem] = entry

                    except json.JSONDecodeError as e:
                        print(f"  Warning: Invalid JSON at {jsonl_file}:{line_num} - {e}")
                        continue

        except Exception as e:
            print(f"  Error reading {jsonl_file}: {e}")
            continue

    print(f"  Loaded {len(metadata_map)} metadata entries")
    return metadata_map


def generate_metadata_for_audio_file(
    audio_file: Path,
    wavcaps_root: Path,
    existing_metadata: Dict[str, Dict[str, Any]],
    file_index: int,
) -> Dict[str, Any]:
    """
    Generate metadata entry for an audio file.

    Args:
        audio_file: Path to the audio file
        wavcaps_root: Root directory of the WavCaps dataset
        existing_metadata: Map of existing metadata keyed by filename stem
        file_index: Global file index for generating IDs

    Returns:
        Metadata dictionary
    """
    # Get filename stem for lookup
    filename_stem = audio_file.stem

    # Try to find existing metadata
    existing = existing_metadata.get(filename_stem)

    # Detect subset
    subset = detect_subset_from_path(audio_file)

    # Compute relative path from wavcaps root
    try:
        relative_path = audio_file.relative_to(wavcaps_root)
    except ValueError:
        # File is outside wavcaps_root, use absolute path
        relative_path = audio_file

    # Generate unique ID
    if existing and "id" in existing:
        sample_id = existing["id"]
    else:
        sample_id = f"{subset}_{file_index:06d}"

    # Extract caption/answer
    if existing:
        question = existing.get("question", "What is happening in the audio?")
        answer = existing.get("answer") or existing.get("caption", "")
    else:
        question = "What is happening in the audio?"
        answer = ""  # Will need manual annotation

    metadata = {
        "id": sample_id,
        "question": question,
        "answer": answer,
        "audio_path": str(relative_path),
        "subset": subset,
    }

    # Preserve additional fields from existing metadata
    if existing:
        for key in ["original_duration", "tags", "description"]:
            if key in existing:
                metadata[key] = existing[key]

    return metadata


def group_by_subset(metadata_entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group metadata entries by subset."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for entry in metadata_entries:
        subset = entry.get("subset", "Unknown")
        grouped[subset].append(entry)

    return dict(grouped)


def write_metadata_jsonl(
    output_path: Path,
    metadata_entries: List[Dict[str, Any]],
    dry_run: bool = False,
) -> None:
    """Write metadata entries to a JSONL file."""
    if dry_run:
        print(f"\n[DRY RUN] Would write {len(metadata_entries)} entries to {output_path}")
        # Show first 3 entries as preview
        print("  Preview of first 3 entries:")
        for i, entry in enumerate(metadata_entries[:3], 1):
            print(f"    {i}. ID: {entry['id']}, Audio: {entry['audio_path']}, Subset: {entry['subset']}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"✓ Wrote {len(metadata_entries)} entries to {output_path}")


def validate_audio_file(audio_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that an audio file can be loaded.

    Returns:
        (is_valid, error_message)
    """
    if not audio_path.exists():
        return False, "File does not exist"

    if audio_path.stat().st_size == 0:
        return False, "File is empty (0 bytes)"

    # Optionally test loading with torchaudio (requires torchaudio)
    try:
        import torchaudio
        info = torchaudio.info(str(audio_path))
        if info.num_frames == 0:
            return False, "Audio file has 0 frames"
    except ImportError:
        # torchaudio not available, skip validation
        pass
    except Exception as e:
        return False, f"Failed to load: {e}"

    return True, None


def main():
    parser = argparse.ArgumentParser(
        description="Fix WavCaps paths by scanning actual audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix all WavCaps metadata
  python scripts/fix_wavcaps_paths.py --wavcaps-dir experiments/full_training/data/wavcaps

  # Dry run to preview changes
  python scripts/fix_wavcaps_paths.py --wavcaps-dir experiments/full_training/data/wavcaps --dry-run

  # Validate audio files (slower but thorough)
  python scripts/fix_wavcaps_paths.py --wavcaps-dir experiments/full_training/data/wavcaps --validate-audio
        """
    )

    parser.add_argument(
        "--wavcaps-dir",
        type=Path,
        required=True,
        help="Path to WavCaps dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL filename (default: wavcaps_train.jsonl in wavcaps-dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    parser.add_argument(
        "--validate-audio",
        action="store_true",
        help="Validate that audio files can be loaded (requires torchaudio)",
    )
    parser.add_argument(
        "--write-per-subset",
        action="store_true",
        help="Write separate JSONL files for each subset",
    )

    args = parser.parse_args()

    wavcaps_dir = args.wavcaps_dir.resolve()
    if not wavcaps_dir.exists():
        print(f"Error: WavCaps directory does not exist: {wavcaps_dir}")
        return 1

    print(f"Scanning WavCaps directory: {wavcaps_dir}")
    print()

    # Find all audio files
    audio_root = wavcaps_dir / "audio"
    print(f"Searching for audio files in: {audio_root}")
    audio_files = find_all_audio_files(audio_root)

    if not audio_files:
        print(f"\n❌ No audio files found in {audio_root}")
        print("   Checked extensions: " + ", ".join(AUDIO_EXTENSIONS))
        return 1

    print(f"✓ Found {len(audio_files)} audio files")

    # Show format breakdown
    format_counts = defaultdict(int)
    for audio_file in audio_files:
        format_counts[audio_file.suffix] += 1

    print("\nFormat breakdown:")
    for ext, count in sorted(format_counts.items()):
        print(f"  {ext}: {count} files")

    # Load existing metadata
    print()
    existing_metadata = load_existing_metadata(wavcaps_dir)

    # Validate audio files if requested
    if args.validate_audio:
        print("\nValidating audio files (this may take a while)...")
        invalid_files: List[Tuple[Path, str]] = []

        for audio_file in audio_files:
            is_valid, error = validate_audio_file(audio_file)
            if not is_valid:
                invalid_files.append((audio_file, error or "Unknown error"))

        if invalid_files:
            print(f"\n⚠️  Found {len(invalid_files)} invalid audio files:")
            for audio_file, error in invalid_files[:10]:  # Show first 10
                print(f"    {audio_file.name}: {error}")
            if len(invalid_files) > 10:
                print(f"    ... and {len(invalid_files) - 10} more")

            # Remove invalid files from processing
            valid_files = set(audio_files) - {f for f, _ in invalid_files}
            audio_files = sorted(valid_files)
            print(f"\nContinuing with {len(audio_files)} valid files")
        else:
            print(f"✓ All {len(audio_files)} audio files are valid")

    # Generate metadata for all audio files
    print("\nGenerating metadata entries...")
    all_metadata: List[Dict[str, Any]] = []

    for idx, audio_file in enumerate(audio_files):
        metadata = generate_metadata_for_audio_file(
            audio_file,
            wavcaps_dir,
            existing_metadata,
            idx,
        )
        all_metadata.append(metadata)

    print(f"✓ Generated {len(all_metadata)} metadata entries")

    # Group by subset
    grouped = group_by_subset(all_metadata)

    print("\nSubset breakdown:")
    for subset, entries in sorted(grouped.items()):
        print(f"  {subset}: {len(entries)} samples")

    # Check for missing captions
    missing_captions = [e for e in all_metadata if not e.get("answer")]
    if missing_captions:
        print(f"\n⚠️  Warning: {len(missing_captions)} entries have missing captions")
        print("   These will need manual annotation or should be excluded from training")

    # Write output
    print()
    if args.write_per_subset:
        # Write separate files for each subset
        for subset, entries in sorted(grouped.items()):
            output_path = wavcaps_dir / f"wavcaps_{subset.lower()}.jsonl"
            write_metadata_jsonl(output_path, entries, args.dry_run)
    else:
        # Write single combined file
        if args.output:
            output_path = args.output
        else:
            output_path = wavcaps_dir / "wavcaps_train.jsonl"

        write_metadata_jsonl(output_path, all_metadata, args.dry_run)

    if not args.dry_run:
        print("\n✅ WavCaps metadata repair complete!")
        print(f"   Total samples: {len(all_metadata)}")
        print(f"   Subsets: {len(grouped)}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
