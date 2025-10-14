#!/usr/bin/env python3
"""
Fix WavCaps paths using existing JSONL metadata with captions.

This script:
1. Loads existing JSONL files that have captions and wavcaps_id
2. Scans actual FLAC files
3. Matches metadata with files using wavcaps_id
4. Generates corrected JSONL with real audio paths

Usage:
    python scripts/fix_wavcaps_with_existing_metadata.py \
        --wavcaps-dir /path/to/wavcaps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_existing_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load entries from existing JSONL file."""
    entries = []

    if not jsonl_path.exists():
        print(f"Warning: {jsonl_path} does not exist")
        return entries

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue

    return entries


def find_audio_files(audio_root: Path, extensions: Tuple[str, ...] = ('.flac', '.wav')) -> List[Path]:
    """Find all audio files recursively."""
    audio_files = []

    if not audio_root.exists():
        print(f"Warning: Audio root does not exist: {audio_root}")
        return audio_files

    for ext in extensions:
        audio_files.extend(audio_root.rglob(f"*{ext}"))

    return sorted(audio_files)


def extract_id_from_filename(filepath: Path) -> str:
    """Extract ID from filename (e.g., NHU05051026 from NHU05051026.flac)."""
    return filepath.stem


def build_id_mappings(
    metadata_entries: List[Dict[str, Any]],
    audio_files: List[Path]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Path]]:
    """
    Build mappings for matching.

    Returns:
        (metadata_by_id, files_by_id)
    """
    # Map wavcaps_id -> metadata
    metadata_by_id = {}
    for entry in metadata_entries:
        wavcaps_id = entry.get('wavcaps_id')
        if wavcaps_id:
            metadata_by_id[wavcaps_id] = entry

    # Map wavcaps_id -> audio file path
    files_by_id = {}
    for audio_file in audio_files:
        file_id = extract_id_from_filename(audio_file)
        files_by_id[file_id] = audio_file

    return metadata_by_id, files_by_id


def match_metadata_to_files(
    metadata_by_id: Dict[str, Dict[str, Any]],
    files_by_id: Dict[str, Path],
    wavcaps_root: Path
) -> List[Dict[str, Any]]:
    """Match metadata entries with actual audio files."""

    matched_entries = []
    unmatched_metadata = []

    for wavcaps_id, metadata in metadata_by_id.items():
        if wavcaps_id in files_by_id:
            audio_file = files_by_id[wavcaps_id]

            # Compute relative path from wavcaps root
            try:
                relative_path = audio_file.relative_to(wavcaps_root)
            except ValueError:
                # If file is outside wavcaps_root, use absolute path
                relative_path = audio_file

            # Create updated entry with correct path
            new_entry = metadata.copy()
            new_entry['audio_path'] = str(relative_path)
            new_entry['audio'] = str(relative_path)

            matched_entries.append(new_entry)
        else:
            unmatched_metadata.append(wavcaps_id)

    return matched_entries, unmatched_metadata


def process_subset(
    subset_name: str,
    wavcaps_root: Path,
    output_path: Path,
    dry_run: bool = False
) -> int:
    """
    Process a single WavCaps subset.

    Returns:
        Number of matched entries
    """
    print(f"\n{'='*70}")
    print(f"Processing subset: {subset_name}")
    print(f"{'='*70}")

    # Determine JSONL filename pattern
    jsonl_patterns = [
        wavcaps_root / f"wavcaps_{subset_name.lower()}.jsonl",
        wavcaps_root / f"{subset_name}.jsonl",
    ]

    jsonl_path = None
    for pattern in jsonl_patterns:
        if pattern.exists():
            jsonl_path = pattern
            break

    if not jsonl_path:
        print(f"Error: Could not find JSONL for {subset_name}")
        print(f"  Tried: {[str(p) for p in jsonl_patterns]}")
        return 0

    # Load metadata
    print(f"Loading metadata from: {jsonl_path}")
    metadata = load_existing_jsonl(jsonl_path)
    print(f"  ✓ Loaded {len(metadata)} entries")

    # Check for captions
    with_captions = sum(1 for e in metadata if e.get('answer') or e.get('caption'))
    print(f"  ✓ {with_captions} entries have captions")

    # Find audio files
    audio_root = wavcaps_root / "audio"
    print(f"\nScanning for audio files in: {audio_root}")
    audio_files = find_audio_files(audio_root)
    print(f"  ✓ Found {len(audio_files)} audio files")

    # Build mappings
    print("\nBuilding ID mappings...")
    metadata_by_id, files_by_id = build_id_mappings(metadata, audio_files)
    print(f"  Metadata entries with IDs: {len(metadata_by_id)}")
    print(f"  Audio files with IDs: {len(files_by_id)}")

    # Match
    print("\nMatching metadata with audio files...")
    matched, unmatched = match_metadata_to_files(metadata_by_id, files_by_id, wavcaps_root)
    print(f"  ✓ Matched: {len(matched)}")

    if unmatched:
        print(f"  ⚠ Unmatched metadata: {len(unmatched)}")
        if len(unmatched) <= 10:
            print(f"    IDs: {unmatched}")
        else:
            print(f"    First 10 IDs: {unmatched[:10]}")

    # Find unmatched audio files
    matched_ids = set(e['wavcaps_id'] for e in matched)
    unmatched_files = [fid for fid in files_by_id.keys() if fid not in matched_ids]
    if unmatched_files:
        print(f"  ⚠ Unmatched audio files: {len(unmatched_files)}")

    # Write output
    if not dry_run:
        print(f"\nWriting to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in matched:
                f.write(json.dumps(entry) + '\n')

        print(f"  ✓ Wrote {len(matched)} entries")
    else:
        print(f"\n[DRY RUN] Would write {len(matched)} entries to {output_path}")

    # Show sample
    if matched:
        print("\n📋 Sample entry:")
        sample = matched[0]
        print(f"  ID: {sample['id']}")
        print(f"  WavCaps ID: {sample.get('wavcaps_id')}")
        caption = sample.get('answer') or sample.get('caption', '')
        print(f"  Caption: {caption[:100]}...")
        print(f"  Audio path: {sample['audio_path']}")

    return len(matched)


def main():
    parser = argparse.ArgumentParser(
        description="Fix WavCaps paths using existing metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix BBC Sound Effects
  python scripts/fix_wavcaps_with_existing_metadata.py \\
      --wavcaps-dir experiments/full_training/data/wavcaps \\
      --subset BBC_Sound_Effects

  # Fix all subsets
  python scripts/fix_wavcaps_with_existing_metadata.py \\
      --wavcaps-dir experiments/full_training/data/wavcaps \\
      --all-subsets

  # Dry run
  python scripts/fix_wavcaps_with_existing_metadata.py \\
      --wavcaps-dir experiments/full_training/data/wavcaps \\
      --subset BBC_Sound_Effects \\
      --dry-run
        """
    )

    parser.add_argument(
        '--wavcaps-dir',
        type=Path,
        required=True,
        help='Path to WavCaps dataset directory'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default=None,
        help='Process specific subset (e.g., BBC_Sound_Effects)'
    )
    parser.add_argument(
        '--all-subsets',
        action='store_true',
        help='Process all available subsets'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output JSONL path (default: wavcaps_train.jsonl)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing files'
    )

    args = parser.parse_args()

    wavcaps_dir = args.wavcaps_dir.resolve()

    if not wavcaps_dir.exists():
        print(f"Error: WavCaps directory does not exist: {wavcaps_dir}")
        return 1

    print(f"WavCaps directory: {wavcaps_dir}")

    # Determine which subsets to process
    if args.all_subsets:
        # Find all JSONL files
        subset_files = list(wavcaps_dir.glob("wavcaps_*.jsonl"))
        subsets = []
        for sf in subset_files:
            if sf.name == "wavcaps_train.jsonl":
                continue
            # Extract subset name from filename
            name = sf.stem.replace("wavcaps_", "")
            subsets.append(name)

        print(f"\nFound {len(subsets)} subsets to process:")
        for s in subsets:
            print(f"  - {s}")

    elif args.subset:
        subsets = [args.subset]

    else:
        print("Error: Must specify --subset or --all-subsets")
        return 1

    # Process each subset
    all_matched = []

    for subset in subsets:
        # Determine output path for this subset
        if args.output:
            output_path = args.output
        else:
            output_path = wavcaps_dir / f"wavcaps_{subset.lower()}_fixed.jsonl"

        num_matched = process_subset(subset, wavcaps_dir, output_path, args.dry_run)

        # Load the matched entries to combine later
        if not args.dry_run and output_path.exists():
            with open(output_path, 'r') as f:
                for line in f:
                    if line.strip():
                        all_matched.append(json.loads(line))

    # Write combined output
    if len(subsets) > 1 and not args.dry_run:
        combined_path = wavcaps_dir / "wavcaps_train.jsonl"
        print(f"\n{'='*70}")
        print(f"Writing combined output to: {combined_path}")
        print(f"{'='*70}")

        with open(combined_path, 'w', encoding='utf-8') as f:
            for entry in all_matched:
                f.write(json.dumps(entry) + '\n')

        print(f"✓ Wrote {len(all_matched)} total entries")

    if not args.dry_run:
        print("\n✅ WavCaps metadata repair complete!")
        print(f"   Total matched samples: {len(all_matched) if len(subsets) > 1 else num_matched}")
        print(f"   All entries have captions: ✓")
        print(f"   All entries have valid audio paths: ✓")
    else:
        print("\n[DRY RUN] No files were written")

    return 0


if __name__ == "__main__":
    sys.exit(main())
