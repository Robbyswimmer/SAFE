#!/usr/bin/env python3
"""
Convert AudioCaps CSV files to JSONL format expected by SAFE datasets.
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def convert_audiocaps_csv_to_jsonl(csv_path: Path, output_path: Path, audio_dir: Path, split: str):
    """Convert AudioCaps CSV to JSONL format."""
    print(f"Converting {csv_path} to {output_path}", flush=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} entries in CSV", flush=True)

    # Expected columns: youtube_id, start_time, caption, audiocap_id
    required_cols = ['youtube_id', 'caption']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries_written = 0
    entries_skipped = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            youtube_id = row['youtube_id']
            caption = row['caption']
            audiocap_id = row.get('audiocap_id', idx)

            # Look for audio file in the audio directory
            audio_path = None
            audio_file_candidates = [
                audio_dir / split / f"{youtube_id}.wav",
                audio_dir / split / f"{youtube_id}.flac",
                audio_dir / split / f"{youtube_id}.mp3",
            ]

            for candidate in audio_file_candidates:
                if candidate.exists():
                    # Store relative path from data root
                    audio_path = str(candidate.relative_to(audio_dir.parent.parent))
                    break

            if audio_path is None:
                entries_skipped += 1
                continue

            # Create JSONL entry in expected format
            entry = {
                "id": f"audiocaps_{split}_{audiocap_id}",
                "sample_id": str(audiocap_id),
                "question": "What sounds are in this audio?",
                "answer": caption,
                "audio_path": audio_path,
                "metadata": {
                    "youtube_id": youtube_id,
                    "split": split,
                }
            }

            f.write(json.dumps(entry) + '\n')
            entries_written += 1

    print(f"Wrote {entries_written} entries to {output_path}", flush=True)
    if entries_skipped > 0:
        print(f"Skipped {entries_skipped} entries (no audio file found)", flush=True)

    return entries_written


def main():
    parser = argparse.ArgumentParser(description="Convert AudioCaps CSV to JSONL")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("experiments/full_training/data"),
        help="Root directory containing audiocaps data"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated list of splits to process"
    )

    args = parser.parse_args()

    print("Starting AudioCaps CSV to JSONL conversion", flush=True)
    print(f"Data root: {args.data_root}", flush=True)

    data_root = args.data_root.expanduser().resolve()
    audiocaps_dir = data_root / "audiocaps"
    audio_dir = audiocaps_dir / "audio"

    print(f"AudioCaps directory: {audiocaps_dir}", flush=True)
    print(f"Audio directory: {audio_dir}", flush=True)

    if not audiocaps_dir.exists():
        raise FileNotFoundError(f"AudioCaps directory not found: {audiocaps_dir}")

    splits = [s.strip() for s in args.splits.split(',')]

    total_entries = 0
    for split in splits:
        csv_path = audiocaps_dir / f"{split}.csv"
        output_path = audiocaps_dir / f"{split}.jsonl"

        if not csv_path.exists():
            print(f"Skipping {split}: CSV not found at {csv_path}", flush=True)
            continue

        entries = convert_audiocaps_csv_to_jsonl(csv_path, output_path, audio_dir, split)
        total_entries += entries
        print(flush=True)

    print(f"✓ Conversion complete! Total entries: {total_entries}", flush=True)
    print(f"\nGenerated files:", flush=True)
    for split in splits:
        jsonl_path = audiocaps_dir / f"{split}.jsonl"
        if jsonl_path.exists():
            print(f"  - {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()