#!/usr/bin/env python3
"""
Quick cleanup script to delete existing WAV files.

This script deletes all WAV files from data/audiocaps/audio/ directories.
Use this ONLY after you've extracted features to HDF5 format.

Safety features:
- Requires confirmation before deleting
- Checks for HDF5 files first
- Reports space freed
- Can dry-run to preview

Usage:
    # Preview what will be deleted (safe)
    python scripts/cleanup_wav_files.py --dry-run

    # Delete all WAV files (requires confirmation)
    python scripts/cleanup_wav_files.py --split train

    # Delete without confirmation (USE WITH CAUTION)
    python scripts/cleanup_wav_files.py --split train --force

    # Delete all splits
    python scripts/cleanup_wav_files.py --all-splits
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def find_wav_files(audio_dir: Path) -> List[Path]:
    """Find all WAV files in directory."""
    if not audio_dir.exists():
        return []
    return list(audio_dir.glob('*.wav'))


def calculate_size(files: List[Path]) -> float:
    """Calculate total size in GB."""
    total_bytes = sum(f.stat().st_size for f in files)
    return total_bytes / (1024 ** 3)


def check_hdf5_exists(features_dir: Path, split: str) -> Tuple[bool, Path]:
    """Check if HDF5 features file exists."""
    h5_path = features_dir / f'{split}_clap_embeddings.h5'
    return h5_path.exists(), h5_path


def confirm_deletion(num_files: int, size_gb: float, split: str) -> bool:
    """Ask user for confirmation."""
    print(f"\n⚠️  DELETION CONFIRMATION")
    print(f"   Split: {split}")
    print(f"   Files: {num_files} WAV files")
    print(f"   Space: {size_gb:.2f} GB")
    print(f"\n   This action CANNOT be undone!")

    response = input(f"\nType 'DELETE {split.upper()}' to confirm: ").strip()
    return response == f'DELETE {split.upper()}'


def delete_wav_files(files: List[Path], dry_run: bool = False) -> Tuple[int, int]:
    """
    Delete WAV files.

    Returns:
        (deleted_count, failed_count)
    """
    deleted = 0
    failed = 0

    for wav_file in files:
        if dry_run:
            print(f"   [DRY-RUN] Would delete: {wav_file.name}")
            deleted += 1
        else:
            try:
                wav_file.unlink()
                deleted += 1
            except Exception as e:
                print(f"  ⚠️  Failed to delete {wav_file.name}: {e}")
                failed += 1

    return deleted, failed


def cleanup_split(
    data_root: Path,
    split: str,
    dry_run: bool = False,
    force: bool = False,
    require_hdf5: bool = True
) -> bool:
    """
    Clean up WAV files for a single split.

    Returns:
        True if successful, False otherwise
    """
    audio_dir = data_root / 'audio' / split
    features_dir = data_root / 'features'

    print(f"\n{'='*60}")
    print(f"CLEANUP: {split.upper()} split")
    print(f"{'='*60}")

    # Check if audio directory exists
    if not audio_dir.exists():
        print(f"ℹ️  No audio directory found: {audio_dir}")
        return True

    # Find WAV files
    wav_files = find_wav_files(audio_dir)

    if not wav_files:
        print(f"ℹ️  No WAV files found in {audio_dir}")
        return True

    # Calculate size
    size_gb = calculate_size(wav_files)

    print(f"📁 Audio directory: {audio_dir}")
    print(f"📊 Found: {len(wav_files)} WAV files ({size_gb:.2f} GB)")

    # Check for HDF5 features
    if require_hdf5:
        hdf5_exists, hdf5_path = check_hdf5_exists(features_dir, split)

        if not hdf5_exists:
            print(f"\n⚠️  WARNING: No HDF5 features found at {hdf5_path}")
            print(f"   You should extract features BEFORE deleting WAV files!")
            print(f"   Run: python scripts/extract_audio_features.py --split {split}")

            if not force:
                response = input("\nDelete anyway? (yes/no): ").strip().lower()
                if response != 'yes':
                    print("   Cancelled.")
                    return False
        else:
            print(f"✅ HDF5 features exist: {hdf5_path}")

    # Confirm deletion
    if dry_run:
        print(f"\n🧪 DRY-RUN MODE: No files will be deleted")
    elif not force:
        if not confirm_deletion(len(wav_files), size_gb, split):
            print("   Cancelled.")
            return False

    # Delete files
    print(f"\n🗑️  {'[DRY-RUN] ' if dry_run else ''}Deleting WAV files...")
    deleted, failed = delete_wav_files(wav_files, dry_run)

    print(f"\n📊 Results:")
    print(f"   Deleted: {deleted}/{len(wav_files)}")
    if failed > 0:
        print(f"   Failed:  {failed}")

    if not dry_run and deleted > 0:
        print(f"   💰 Freed approximately {size_gb:.2f} GB")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description='Delete WAV files after feature extraction'
    )
    parser.add_argument('--split', choices=['train', 'val', 'test'],
                        help='Dataset split to clean up')
    parser.add_argument('--all-splits', action='store_true',
                        help='Clean up all splits (train, val, test)')
    parser.add_argument('--data-root', type=str, default='data/audiocaps',
                        help='Root directory for AudioCaps data')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview what would be deleted without deleting')
    parser.add_argument('--force', action='store_true',
                        help='Skip confirmation prompts (USE WITH CAUTION)')
    parser.add_argument('--skip-hdf5-check', action='store_true',
                        help='Skip check for HDF5 features (dangerous)')

    args = parser.parse_args()

    # Validate arguments
    if not args.split and not args.all_splits:
        parser.error("Must specify --split or --all-splits")

    if args.split and args.all_splits:
        parser.error("Cannot specify both --split and --all-splits")

    # Setup
    data_root = Path(args.data_root).expanduser().resolve()

    if not data_root.exists():
        print(f"❌ Data root not found: {data_root}")
        return 1

    print(f"🧹 WAV File Cleanup Utility")
    print(f"   Data root: {data_root}")

    if args.dry_run:
        print(f"   Mode: DRY-RUN (no files will be deleted)")
    elif args.force:
        print(f"   Mode: FORCE (no confirmation required)")

    # Determine splits to process
    if args.all_splits:
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]

    # Process each split
    all_success = True
    for split in splits:
        success = cleanup_split(
            data_root,
            split,
            dry_run=args.dry_run,
            force=args.force,
            require_hdf5=not args.skip_hdf5_check
        )
        all_success = all_success and success

    # Summary
    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"DRY-RUN COMPLETE - No files were deleted")
        print(f"Remove --dry-run flag to actually delete files")
    elif all_success:
        print(f"CLEANUP COMPLETE")
        print(f"✅ All WAV files successfully deleted")
        print(f"✅ HDF5 features retained in {data_root / 'features'}")
    else:
        print(f"CLEANUP COMPLETED WITH ERRORS")
        print(f"⚠️  Some files could not be deleted")

    print(f"{'='*60}\n")

    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
