#!/usr/bin/env python3
"""
Check for data leakage between AudioCaps train and validation splits.

Validates:
1. No overlapping audio files (by filename/youtube_id)
2. No overlapping captions (exact text matches)
3. Metadata consistency
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys


def load_json_metadata(json_path: Path) -> List[Dict]:
    """Load JSON or JSONL metadata file."""
    if not json_path.exists():
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        if json_path.suffix == '.jsonl':
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
            return data
        else:
            data = json.load(f)
            # Handle wrapped format
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            return data if isinstance(data, list) else []


def load_csv_metadata(csv_path: Path) -> List[Dict]:
    """Load CSV metadata file."""
    if not csv_path.exists():
        return []

    import csv
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def extract_identifiers(metadata: List[Dict], split_name: str) -> Tuple[Set[str], Set[str], Dict[str, List[str]]]:
    """
    Extract identifiers from metadata.

    Returns:
        (youtube_ids, audio_filenames, youtube_id_to_captions)
    """
    youtube_ids = set()
    audio_files = set()
    id_to_captions = {}

    for entry in metadata:
        # Extract YouTube ID
        ytid = (
            entry.get('ytid') or
            entry.get('youtube_id') or
            entry.get('id') or
            entry.get('sound_name', '').replace('.wav', '')
        )
        if ytid:
            youtube_ids.add(str(ytid))

            # Extract captions
            captions = []
            if 'captions' in entry and isinstance(entry['captions'], list):
                captions = [str(c).strip().lower() for c in entry['captions'] if str(c).strip()]
            elif 'answer' in entry:
                answer = entry['answer']
                if isinstance(answer, list):
                    captions = [str(c).strip().lower() for c in answer if str(c).strip()]
                else:
                    captions = [str(answer).strip().lower()]
            elif 'caption' in entry:
                captions = [str(entry['caption']).strip().lower()]

            if captions:
                id_to_captions[str(ytid)] = captions

        # Extract audio filename
        sound_name = (
            entry.get('sound_name') or
            entry.get('file_path', '').split('/')[-1]
        )
        if sound_name:
            # Normalize filename (remove extension, remove timestamp suffix)
            base = sound_name.replace('.wav', '').replace('.mp3', '')
            # Remove _NNNNNN timestamp suffix if present
            import re
            base = re.sub(r'_\d{6}$', '', base)
            audio_files.add(base)

    print(f"[{split_name}] Found {len(youtube_ids)} youtube_ids, {len(audio_files)} audio files, {len(id_to_captions)} with captions")

    return youtube_ids, audio_files, id_to_captions


def check_audio_file_overlap(train_dir: Path, val_dir: Path) -> Set[str]:
    """Check for overlapping audio files on disk."""
    if not train_dir.exists() or not val_dir.exists():
        print(f"⚠️  Warning: Audio directories not found")
        print(f"   Train: {train_dir.exists()} - {train_dir}")
        print(f"   Val: {val_dir.exists()} - {val_dir}")
        return set()

    # Get all audio files (wav, mp3, flac)
    train_files = set()
    val_files = set()

    for ext in ['*.wav', '*.mp3', '*.flac']:
        train_files.update(f.stem for f in train_dir.glob(ext))
        val_files.update(f.stem for f in val_dir.glob(ext))

    # Normalize filenames (remove timestamp suffixes)
    import re
    train_normalized = {re.sub(r'_\d{6}$', '', f) for f in train_files}
    val_normalized = {re.sub(r'_\d{6}$', '', f) for f in val_files}

    overlap = train_normalized & val_normalized

    print(f"\n📁 Audio File Check:")
    print(f"   Train: {len(train_files)} files ({len(train_normalized)} unique base IDs)")
    print(f"   Val: {len(val_files)} files ({len(val_normalized)} unique base IDs)")

    return overlap


def check_caption_overlap(
    train_captions: Dict[str, List[str]],
    val_captions: Dict[str, List[str]]
) -> List[Tuple[str, str, str]]:
    """
    Check for exact caption text overlap between splits.

    Returns:
        List of (train_id, val_id, overlapping_caption)
    """
    # Build caption -> id mapping for train
    train_caption_to_ids = {}
    for ytid, captions in train_captions.items():
        for caption in captions:
            if caption not in train_caption_to_ids:
                train_caption_to_ids[caption] = []
            train_caption_to_ids[caption].append(ytid)

    # Check val captions against train
    overlaps = []
    for val_id, val_caps in val_captions.items():
        for caption in val_caps:
            if caption in train_caption_to_ids:
                for train_id in train_caption_to_ids[caption]:
                    overlaps.append((train_id, val_id, caption))

    return overlaps


def main():
    parser = argparse.ArgumentParser(description="Check for data leakage in AudioCaps splits")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("experiments/full_training/data/audiocaps"),
        help="Root directory for AudioCaps data"
    )
    parser.add_argument(
        "--check-audio-files",
        action="store_true",
        help="Check audio files on disk for overlap"
    )

    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()

    if not data_root.exists():
        print(f"❌ Data root not found: {data_root}")
        return 1

    print("="*70)
    print("AudioCaps Data Leakage Check")
    print("="*70)
    print(f"Data root: {data_root}\n")

    # Find ALL metadata files for thorough checking
    train_files = []
    val_files = []

    for name in ["train.csv", "AudioCaps_train.json", "audiocaps_train.json", "train.json"]:
        path = data_root / name
        if path.exists():
            train_files.append(path)

    for name in ["val.csv", "AudioCaps_val.json", "audiocaps_val.json", "val.json"]:
        path = data_root / name
        if path.exists():
            val_files.append(path)

    if not train_files:
        print(f"❌ No train metadata found in {data_root}")
        return 1

    if not val_files:
        print(f"❌ No val metadata found in {data_root}")
        return 1

    print(f"✓ Found {len(train_files)} train file(s): {[f.name for f in train_files]}")
    print(f"✓ Found {len(val_files)} val file(s): {[f.name for f in val_files]}\n")

    # Load all metadata files for comprehensive checking
    print("Loading metadata from all files...")

    all_train_meta = []
    for train_file in train_files:
        if train_file.suffix == '.csv':
            meta = load_csv_metadata(train_file)
        else:
            meta = load_json_metadata(train_file)
        print(f"  {train_file.name}: {len(meta)} samples")
        all_train_meta.append((train_file.name, meta))

    all_val_meta = []
    for val_file in val_files:
        if val_file.suffix == '.csv':
            meta = load_csv_metadata(val_file)
        else:
            meta = load_json_metadata(val_file)
        print(f"  {val_file.name}: {len(meta)} samples")
        all_val_meta.append((val_file.name, meta))

    print()

    # Check each train file against each val file
    print("="*70)
    print("CHECKING ALL TRAIN-VAL COMBINATIONS")
    print("="*70)

    all_results = []

    for train_name, train_meta in all_train_meta:
        for val_name, val_meta in all_val_meta:
            print(f"\n--- Checking: {train_name} vs {val_name} ---")

            # Extract identifiers
            train_ids, train_audio_files, train_captions = extract_identifiers(train_meta, train_name)
            val_ids, val_audio_files, val_captions = extract_identifiers(val_meta, val_name)

            # Check overlaps
            id_overlap = train_ids & val_ids
            file_overlap = train_audio_files & val_audio_files
            caption_overlaps = check_caption_overlap(train_captions, val_captions)

            result = {
                'train_file': train_name,
                'val_file': val_name,
                'train_count': len(train_meta),
                'val_count': len(val_meta),
                'id_overlap': len(id_overlap),
                'file_overlap': len(file_overlap),
                'caption_overlap': len(caption_overlaps),
                'id_overlap_samples': list(id_overlap)[:5] if id_overlap else [],
            }
            all_results.append(result)

            if id_overlap:
                print(f"  🚨 {len(id_overlap)} YouTube IDs overlap!")
                print(f"     Examples: {result['id_overlap_samples']}")
            else:
                print(f"  ✅ No YouTube ID overlap")

            if file_overlap:
                print(f"  🚨 {len(file_overlap)} audio filenames overlap!")
            else:
                print(f"  ✅ No audio filename overlap")

            if caption_overlaps:
                same_id_leaks = [(tid, vid, cap) for tid, vid, cap in caption_overlaps if tid == vid]
                if same_id_leaks:
                    print(f"  🚨 {len(same_id_leaks)} captions with matching IDs!")
                else:
                    print(f"  ⚠️  {len(caption_overlaps)} caption text matches (different IDs - OK)")
            else:
                print(f"  ✅ No caption overlap")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    leakage_found = False

    for result in all_results:
        if result['id_overlap'] > 0 or result['file_overlap'] > 0:
            leakage_found = True
            print(f"\n🚨 LEAKAGE: {result['train_file']} vs {result['val_file']}")
            if result['id_overlap'] > 0:
                print(f"   - {result['id_overlap']} YouTube IDs overlap")
                print(f"   - Examples: {result['id_overlap_samples']}")
            if result['file_overlap'] > 0:
                print(f"   - {result['file_overlap']} audio filenames overlap")

    if not leakage_found:
        print("\n✅ No leakage detected in any train-val combination!")
        print("\nFile sizes:")
        for result in all_results:
            print(f"  {result['train_file']}: {result['train_count']} train samples")
            print(f"  {result['val_file']}: {result['val_count']} val samples")

    # Audio file overlap (on disk)
    if args.check_audio_files:
        print("\n" + "="*70)
        print("AUDIO FILE CHECK (on disk)")
        print("="*70)

        train_audio_dir = data_root / "audio" / "train"
        val_audio_dir = data_root / "audio" / "val"

        # Also check _10s directories
        if not train_audio_dir.exists():
            train_audio_dir = data_root / "audio" / "train_10s"
        if not val_audio_dir.exists():
            val_audio_dir = data_root / "audio" / "val_10s"

        disk_overlap = check_audio_file_overlap(train_audio_dir, val_audio_dir)

        if disk_overlap:
            leakage_found = True
            print(f"\n🚨 LEAKAGE DETECTED: {len(disk_overlap)} audio files on disk in both train and val!")
            print(f"   Examples: {list(disk_overlap)[:5]}")
        else:
            print(f"\n✅ No audio file overlap on disk")

    # Summary
    print("\n" + "="*70)
    if leakage_found:
        print("❌ DATA LEAKAGE DETECTED - See issues above")
        print("="*70)
        return 1
    else:
        print("✅ NO DATA LEAKAGE FOUND")
        print("="*70)
        print("\nTrain and validation splits are properly separated.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
