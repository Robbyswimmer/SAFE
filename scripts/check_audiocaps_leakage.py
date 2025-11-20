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

    # Find metadata files (JSON or CSV)
    train_candidates = [
        data_root / "AudioCaps_train.json",
        data_root / "audiocaps_train.json",
        data_root / "train.json",
        data_root / "train.csv",
    ]
    val_candidates = [
        data_root / "AudioCaps_val.json",
        data_root / "audiocaps_val.json",
        data_root / "val.json",
        data_root / "val.csv",
    ]

    train_file = next((p for p in train_candidates if p.exists()), None)
    val_file = next((p for p in val_candidates if p.exists()), None)

    if not train_file:
        print(f"❌ Train metadata not found. Looked for:")
        for p in train_candidates:
            print(f"   - {p}")
        return 1

    if not val_file:
        print(f"❌ Val metadata not found. Looked for:")
        for p in val_candidates:
            print(f"   - {p}")
        return 1

    print(f"✓ Found train metadata: {train_file.name}")
    print(f"✓ Found val metadata: {val_file.name}\n")

    # Load metadata
    print("Loading metadata...")
    if train_file.suffix == '.csv':
        train_meta = load_csv_metadata(train_file)
    else:
        train_meta = load_json_metadata(train_file)

    if val_file.suffix == '.csv':
        val_meta = load_csv_metadata(val_file)
    else:
        val_meta = load_json_metadata(val_file)

    print(f"Train samples: {len(train_meta)}")
    print(f"Val samples: {len(val_meta)}\n")

    if not train_meta or not val_meta:
        print("❌ Failed to load metadata")
        return 1

    # Extract identifiers
    print("Extracting identifiers...")
    train_ids, train_files, train_captions = extract_identifiers(train_meta, "train")
    val_ids, val_files, val_captions = extract_identifiers(val_meta, "val")

    # Check for overlaps
    print("\n" + "="*70)
    print("LEAKAGE CHECK RESULTS")
    print("="*70)

    leakage_found = False

    # 1. YouTube ID overlap
    id_overlap = train_ids & val_ids
    if id_overlap:
        leakage_found = True
        print(f"\n🚨 LEAKAGE DETECTED: {len(id_overlap)} YouTube IDs in both train and val!")
        print(f"   Examples: {list(id_overlap)[:5]}")
    else:
        print(f"\n✅ YouTube IDs: No overlap ({len(train_ids)} train, {len(val_ids)} val)")

    # 2. Audio file overlap (metadata)
    file_overlap = train_files & val_files
    if file_overlap:
        leakage_found = True
        print(f"\n🚨 LEAKAGE DETECTED: {len(file_overlap)} audio filenames in both train and val!")
        print(f"   Examples: {list(file_overlap)[:5]}")
    else:
        print(f"✅ Audio filenames (metadata): No overlap")

    # 3. Audio file overlap (on disk)
    if args.check_audio_files:
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
            print(f"✅ Audio files (on disk): No overlap")

    # 4. Caption text overlap
    caption_overlaps = check_caption_overlap(train_captions, val_captions)

    if caption_overlaps:
        print(f"\n⚠️  WARNING: {len(caption_overlaps)} exact caption text matches found")
        print(f"   (This is expected for AudioCaps - same audio can have similar descriptions)")
        print(f"   However, if the YouTube IDs also match, this indicates leakage!")

        # Check if any overlapping captions share the same YouTube ID
        same_id_caption_leaks = [
            (tid, vid, cap) for tid, vid, cap in caption_overlaps if tid == vid
        ]

        if same_id_caption_leaks:
            leakage_found = True
            print(f"\n🚨 LEAKAGE DETECTED: {len(same_id_caption_leaks)} captions with matching YouTube IDs!")
            for tid, vid, cap in same_id_caption_leaks[:3]:
                print(f"   ID: {tid} | Caption: {cap[:60]}...")
        else:
            print(f"   ✓ No matching YouTube IDs for overlapping captions (likely coincidental)")
    else:
        print(f"\n✅ Captions: No exact text overlap")

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
