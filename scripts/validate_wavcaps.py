#!/usr/bin/env python3
"""
Validate WavCaps dataset configuration for SAFE training.

This script checks that:
1. JSONL metadata files exist and are well-formed
2. Audio file paths in metadata point to existing files
3. Audio files can be loaded by torchaudio
4. Dataset statistics match expectations

Usage:
    # Basic validation
    python scripts/validate_wavcaps.py \
        --wavcaps-dir experiments/full_training/data/wavcaps

    # Detailed validation with sample loading
    python scripts/validate_wavcaps.py \
        --wavcaps-dir experiments/full_training/data/wavcaps \
        --test-loading \
        --num-samples 10

    # Quick check (metadata only, no file validation)
    python scripts/validate_wavcaps.py \
        --wavcaps-dir experiments/full_training/data/wavcaps \
        --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        self.info.append(message)

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def print_summary(self) -> None:
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"   {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   {warning}")

        if self.info:
            print("\n📊 INFO:")
            for info in self.info:
                print(f"   {info}")

        if self.is_valid():
            if not self.warnings:
                print("\n✅ Validation passed with no issues!")
            else:
                print(f"\n⚠️  Validation passed with {len(self.warnings)} warning(s)")
        else:
            print(f"\n❌ Validation failed with {len(self.errors)} error(s)")


def find_jsonl_files(wavcaps_dir: Path) -> List[Path]:
    """Find all JSONL metadata files in the WavCaps directory."""
    jsonl_files = list(wavcaps_dir.glob("wavcaps*.jsonl"))

    # Also check for train.jsonl, val.jsonl patterns
    jsonl_files.extend(wavcaps_dir.glob("*train*.jsonl"))
    jsonl_files.extend(wavcaps_dir.glob("*val*.jsonl"))
    jsonl_files.extend(wavcaps_dir.glob("*test*.jsonl"))

    # Deduplicate
    return sorted(set(jsonl_files))


def validate_jsonl_structure(jsonl_file: Path, result: ValidationResult) -> List[Dict[str, Any]]:
    """
    Validate JSONL file structure and parse entries.

    Returns:
        List of valid metadata entries
    """
    entries: List[Dict[str, Any]] = []

    if not jsonl_file.exists():
        result.add_error(f"JSONL file not found: {jsonl_file}")
        return entries

    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    entries.append(entry)

                    # Validate required fields
                    if "audio_path" not in entry and "audio" not in entry:
                        result.add_warning(
                            f"{jsonl_file.name}:{line_num} - Missing 'audio_path' or 'audio' field"
                        )

                    if "answer" not in entry and "caption" not in entry:
                        result.add_warning(
                            f"{jsonl_file.name}:{line_num} - Missing 'answer' or 'caption' field"
                        )

                except json.JSONDecodeError as e:
                    result.add_error(
                        f"{jsonl_file.name}:{line_num} - Invalid JSON: {e}"
                    )

    except Exception as e:
        result.add_error(f"Failed to read {jsonl_file}: {e}")

    return entries


def validate_audio_paths(
    entries: List[Dict[str, Any]],
    wavcaps_dir: Path,
    result: ValidationResult,
    check_subset: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Validate that audio file paths exist.

    Args:
        entries: Metadata entries
        wavcaps_dir: Root directory for WavCaps
        result: ValidationResult to append to
        check_subset: If set, only check first N files (for speed)

    Returns:
        (num_valid, num_invalid)
    """
    num_valid = 0
    num_invalid = 0
    checked = 0

    for entry in entries:
        if check_subset is not None and checked >= check_subset:
            break

        audio_path = entry.get("audio_path") or entry.get("audio")
        if not audio_path:
            continue

        full_path = wavcaps_dir / audio_path

        if full_path.exists():
            num_valid += 1
        else:
            num_invalid += 1
            if num_invalid <= 5:  # Only show first 5 missing files
                result.add_error(f"Audio file not found: {audio_path}")

        checked += 1

    if num_invalid > 5:
        result.add_error(f"... and {num_invalid - 5} more missing audio files")

    return num_valid, num_invalid


def test_audio_loading(
    entries: List[Dict[str, Any]],
    wavcaps_dir: Path,
    result: ValidationResult,
    num_samples: int = 5,
) -> None:
    """
    Test loading a subset of audio files with torchaudio.

    Args:
        entries: Metadata entries
        wavcaps_dir: Root directory
        result: ValidationResult to append to
        num_samples: Number of samples to test
    """
    try:
        import torchaudio
    except ImportError:
        result.add_warning("torchaudio not installed - skipping audio loading tests")
        return

    print(f"\nTesting audio loading with {num_samples} sample(s)...")

    num_tested = 0
    num_success = 0
    num_failed = 0

    for entry in entries:
        if num_tested >= num_samples:
            break

        audio_path = entry.get("audio_path") or entry.get("audio")
        if not audio_path:
            continue

        full_path = wavcaps_dir / audio_path
        if not full_path.exists():
            continue

        try:
            waveform, sample_rate = torchaudio.load(str(full_path))
            num_success += 1

            # Check audio properties
            if waveform.size(-1) == 0:
                result.add_warning(f"Audio file is empty (0 samples): {audio_path}")

            print(f"  ✓ Loaded: {audio_path} ({sample_rate}Hz, {waveform.size(-1)} samples)")

        except Exception as e:
            num_failed += 1
            result.add_error(f"Failed to load {audio_path}: {e}")

        num_tested += 1

    if num_tested == 0:
        result.add_warning("No audio files available to test loading")
    else:
        result.add_info(f"Audio loading test: {num_success}/{num_tested} successful")

        if num_failed > 0:
            result.add_error(f"{num_failed} audio files failed to load")


def collect_statistics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collect dataset statistics."""
    stats: Dict[str, Any] = {
        "total_samples": len(entries),
        "subsets": Counter(),
        "missing_captions": 0,
        "missing_audio_paths": 0,
    }

    for entry in entries:
        subset = entry.get("subset", "Unknown")
        stats["subsets"][subset] += 1

        if not (entry.get("answer") or entry.get("caption")):
            stats["missing_captions"] += 1

        if not (entry.get("audio_path") or entry.get("audio")):
            stats["missing_audio_paths"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate WavCaps dataset for SAFE training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full validation
  python scripts/validate_wavcaps.py --wavcaps-dir experiments/full_training/data/wavcaps

  # Quick check (metadata only)
  python scripts/validate_wavcaps.py --wavcaps-dir experiments/full_training/data/wavcaps --quick

  # Test audio loading
  python scripts/validate_wavcaps.py --wavcaps-dir experiments/full_training/data/wavcaps --test-loading --num-samples 10
        """
    )

    parser.add_argument(
        "--wavcaps-dir",
        type=Path,
        required=True,
        help="Path to WavCaps dataset directory",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation (metadata only, no file existence checks)",
    )
    parser.add_argument(
        "--test-loading",
        action="store_true",
        help="Test loading audio files with torchaudio",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of audio samples to test loading (default: 5)",
    )

    args = parser.parse_args()

    wavcaps_dir = args.wavcaps_dir.resolve()

    print("=" * 70)
    print("WavCaps Dataset Validation")
    print("=" * 70)
    print(f"\nDataset directory: {wavcaps_dir}")

    result = ValidationResult()

    # Check directory exists
    if not wavcaps_dir.exists():
        result.add_error(f"WavCaps directory does not exist: {wavcaps_dir}")
        result.print_summary()
        return 1

    # Find JSONL files
    print("\nSearching for JSONL metadata files...")
    jsonl_files = find_jsonl_files(wavcaps_dir)

    if not jsonl_files:
        result.add_error(f"No JSONL metadata files found in {wavcaps_dir}")
        result.add_info("Expected files like: wavcaps_train.jsonl, wavcaps_*.jsonl")
        result.print_summary()
        return 1

    print(f"✓ Found {len(jsonl_files)} JSONL file(s):")
    for jsonl_file in jsonl_files:
        print(f"  - {jsonl_file.name}")

    # Validate each JSONL file
    print("\nValidating JSONL structure...")
    all_entries: List[Dict[str, Any]] = []

    for jsonl_file in jsonl_files:
        print(f"  Checking {jsonl_file.name}...")
        entries = validate_jsonl_structure(jsonl_file, result)
        all_entries.extend(entries)
        print(f"    ✓ Loaded {len(entries)} entries")

    if not all_entries:
        result.add_error("No valid metadata entries found in any JSONL file")
        result.print_summary()
        return 1

    result.add_info(f"Total metadata entries: {len(all_entries)}")

    # Collect and display statistics
    print("\nCollecting statistics...")
    stats = collect_statistics(all_entries)

    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Subsets:")
    for subset, count in stats["subsets"].most_common():
        print(f"    - {subset}: {count}")

    if stats["missing_captions"] > 0:
        result.add_warning(f"{stats['missing_captions']} samples have missing captions")

    if stats["missing_audio_paths"] > 0:
        result.add_error(f"{stats['missing_audio_paths']} samples have missing audio paths")

    # Validate audio paths exist (unless quick mode)
    if not args.quick:
        print("\nValidating audio file paths...")
        check_subset = 100 if len(all_entries) > 1000 else None

        if check_subset:
            print(f"  (Checking first {check_subset} files for speed)")

        num_valid, num_invalid = validate_audio_paths(
            all_entries,
            wavcaps_dir,
            result,
            check_subset=check_subset,
        )

        print(f"  Valid paths: {num_valid}")
        print(f"  Invalid paths: {num_invalid}")

        if num_invalid > 0:
            result.add_error(
                f"{num_invalid} audio files are missing. "
                "Run scripts/fix_wavcaps_paths.py to regenerate metadata."
            )
    else:
        result.add_info("Skipped audio file validation (--quick mode)")

    # Test audio loading
    if args.test_loading and all_entries:
        test_audio_loading(
            all_entries,
            wavcaps_dir,
            result,
            num_samples=args.num_samples,
        )

    # Print summary
    result.print_summary()

    return 0 if result.is_valid() else 1


if __name__ == "__main__":
    sys.exit(main())
