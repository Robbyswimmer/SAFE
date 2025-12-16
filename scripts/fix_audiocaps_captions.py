#!/usr/bin/env python3
"""
Fix empty captions in AudioCaps train.jsonl by merging captions from train.csv.

Usage:
    cd experiments/full_training/data/audiocaps
    python /path/to/scripts/fix_audiocaps_captions.py

Or with explicit paths:
    python scripts/fix_audiocaps_captions.py \
        --csv experiments/full_training/data/audiocaps/train.csv \
        --jsonl experiments/full_training/data/audiocaps/train.jsonl \
        --output experiments/full_training/data/audiocaps/train_fixed.jsonl
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


def load_captions_from_csv(csv_path: Path) -> dict[str, list[str]]:
    """Load captions from CSV, grouped by youtube_id."""
    captions_by_ytid = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ytid = row["youtube_id"]
            caption = row["caption"].strip()
            if caption:
                captions_by_ytid[ytid].append(caption)

    print(f"Loaded {len(captions_by_ytid)} unique youtube_ids from CSV")
    print(f"Total captions: {sum(len(v) for v in captions_by_ytid.values())}")
    return dict(captions_by_ytid)


def fix_jsonl(jsonl_path: Path, output_path: Path, captions_by_ytid: dict[str, list[str]]):
    """Read JSONL, populate captions, write to output."""
    matched = 0
    unmatched = 0
    skipped_empty = 0

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            ytid = entry.get("ytid", "")
            sound_name = entry.get("sound_name", "")

            # Try to match by ytid field
            captions = None
            if ytid in captions_by_ytid:
                captions = captions_by_ytid[ytid]
            else:
                # Try extracting ytid from sound_name
                # e.g., "r1nicOVtvkQ.wav" -> "r1nicOVtvkQ"
                # e.g., "r1nicOVtvkQ_000130.wav" -> "r1nicOVtvkQ"
                base_ytid = sound_name.replace(".wav", "").split("_")[0] if sound_name else ""
                if base_ytid and base_ytid in captions_by_ytid:
                    captions = captions_by_ytid[base_ytid]

            if captions:
                entry["captions"] = captions
                matched += 1
                fout.write(json.dumps(entry) + "\n")
            else:
                unmatched += 1
                # Optionally skip entries without captions
                # Or keep them with empty captions (uncomment below)
                # entry["captions"] = [""]
                # fout.write(json.dumps(entry) + "\n")

            if line_num <= 3:
                print(f"  Sample {line_num}: ytid={ytid!r}, sound_name={sound_name!r}, captions={captions}")

    print(f"\nResults:")
    print(f"  Matched (with captions): {matched}")
    print(f"  Unmatched (skipped): {unmatched}")
    print(f"  Output written to: {output_path}")

    return matched, unmatched


def main():
    parser = argparse.ArgumentParser(description="Fix empty captions in AudioCaps JSONL")
    parser.add_argument("--csv", type=Path, default=Path("train.csv"),
                        help="Path to train.csv with captions")
    parser.add_argument("--jsonl", type=Path, default=Path("train.jsonl"),
                        help="Path to train.jsonl with empty captions")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: train_fixed.jsonl)")
    parser.add_argument("--replace", action="store_true",
                        help="Replace original file after fixing")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.jsonl.parent / "train_fixed.jsonl"

    print(f"CSV source: {args.csv}")
    print(f"JSONL input: {args.jsonl}")
    print(f"Output: {args.output}")
    print()

    # Load captions from CSV
    captions_by_ytid = load_captions_from_csv(args.csv)

    # Fix JSONL
    print(f"\nProcessing JSONL...")
    matched, unmatched = fix_jsonl(args.jsonl, args.output, captions_by_ytid)

    if matched == 0:
        print("\n⚠️  WARNING: No matches found! Check that youtube_ids align between CSV and JSONL.")
        return 1

    if args.replace:
        backup = args.jsonl.with_suffix(".jsonl.broken")
        print(f"\nBacking up original to: {backup}")
        args.jsonl.rename(backup)
        print(f"Replacing with fixed file...")
        args.output.rename(args.jsonl)
        print(f"Done! Original backed up to {backup}")
    else:
        print(f"\nTo apply the fix, run:")
        print(f"  mv {args.jsonl} {args.jsonl}.broken")
        print(f"  mv {args.output} {args.jsonl}")

    return 0


if __name__ == "__main__":
    exit(main())
