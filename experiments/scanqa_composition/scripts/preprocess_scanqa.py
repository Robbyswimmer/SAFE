#!/usr/bin/env python3
"""
Preprocess ScanQA dataset for QA composition experiments.

This script:
1. Copies ScanQA annotation files to the data directory
2. Validates that corresponding ScanNet scene data exists
3. Reports statistics on data coverage

Prerequisites:
- ScanQA downloaded from https://github.com/ATR-DBI/ScanQA
- ScanNet preprocessed using experiments/scannet_composition/scripts/preprocess_scannet.py

Usage:
    python experiments/scanqa_composition/scripts/preprocess_scanqa.py \
        --scanqa-root /path/to/ScanQA/data \
        --output-dir experiments/full_training/data
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict


def analyze_questions(samples: list) -> dict:
    """Analyze question types in the dataset."""
    question_types = defaultdict(int)

    keywords = {
        "color": ["color", "colored"],
        "count": ["how many", "number of", "count"],
        "spatial": ["next to", "near", "behind", "in front", "above", "below", "between", "left", "right"],
        "object": ["what is", "what are", "what kind", "what type"],
        "location": ["where is", "where are", "location"],
        "yes_no": ["is there", "are there", "is it", "are they", "does", "do"],
        "attribute": ["size", "shape", "material", "big", "small", "large"],
    }

    for sample in samples:
        question = sample.get("question", "").lower()
        categorized = False

        for qtype, kws in keywords.items():
            if any(kw in question for kw in kws):
                question_types[qtype] += 1
                categorized = True
                break

        if not categorized:
            question_types["other"] += 1

    return dict(question_types)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ScanQA for QA composition")
    parser.add_argument("--scanqa-root", type=str, required=True,
                        help="Path to ScanQA data directory (containing qa/ folder)")
    parser.add_argument("--output-dir", type=str, default="experiments/full_training/data",
                        help="Output directory")
    parser.add_argument("--validate-scenes", action="store_true",
                        help="Check that ScanNet scene data exists")
    args = parser.parse_args()

    scanqa_root = Path(args.scanqa_root)
    output_dir = Path(args.output_dir)

    # Create output directory
    scanqa_output = output_dir / "scanqa"
    scanqa_output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ScanQA Preprocessing")
    print("=" * 60)
    print(f"ScanQA root: {scanqa_root}")
    print(f"Output dir: {scanqa_output}")
    print("=" * 60)

    # Find QA files
    qa_dir = scanqa_root / "qa"
    if not qa_dir.exists():
        qa_dir = scanqa_root  # Maybe files are directly in root

    splits = ["train", "val", "test"]
    all_scene_ids = set()

    for split in splits:
        # Try different naming conventions
        possible_names = [
            f"ScanQA_v1.0_{split}.json",
            f"scanqa_{split}.json",
            f"{split}.json",
        ]

        qa_file = None
        for name in possible_names:
            candidate = qa_dir / name
            if candidate.exists():
                qa_file = candidate
                break

        if qa_file is None:
            print(f"Warning: No QA file found for split '{split}'")
            continue

        # Load and analyze
        with open(qa_file) as f:
            data = json.load(f)

        print(f"\n{split.upper()} split:")
        print(f"  QA pairs: {len(data)}")

        # Extract scene IDs
        scene_ids = set()
        for item in data:
            scene_id = item.get("scene_id", item.get("scan_id"))
            if scene_id:
                scene_ids.add(scene_id)
                all_scene_ids.add(scene_id)

        print(f"  Unique scenes: {len(scene_ids)}")

        # Analyze question types
        qtypes = analyze_questions(data)
        print(f"  Question types:")
        for qtype, count in sorted(qtypes.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(data)
            print(f"    {qtype}: {count} ({pct:.1f}%)")

        # Copy file to output
        output_file = scanqa_output / f"ScanQA_v1.0_{split}.json"
        shutil.copy(qa_file, output_file)
        print(f"  Saved to: {output_file}")

    # Validate against ScanNet data if requested
    if args.validate_scenes:
        print("\n" + "=" * 60)
        print("Validating ScanNet scene data")
        print("=" * 60)

        scannet_dir = output_dir / "scannet"
        pc_dir = scannet_dir / "pointclouds"
        img_dir = scannet_dir / "images"

        if not scannet_dir.exists():
            print(f"Warning: ScanNet directory not found: {scannet_dir}")
            print("Run preprocess_scannet.py first")
        else:
            scenes_with_pc = set()
            scenes_with_img = set()

            for scene_id in all_scene_ids:
                if (pc_dir / f"{scene_id}.npy").exists():
                    scenes_with_pc.add(scene_id)
                if (img_dir / f"{scene_id}.jpg").exists():
                    scenes_with_img.add(scene_id)

            scenes_with_both = scenes_with_pc & scenes_with_img

            print(f"\nScene data availability:")
            print(f"  Total scenes in ScanQA: {len(all_scene_ids)}")
            if len(all_scene_ids) > 0:
                print(f"  Scenes with point cloud: {len(scenes_with_pc)} ({100*len(scenes_with_pc)/len(all_scene_ids):.1f}%)")
                print(f"  Scenes with image: {len(scenes_with_img)} ({100*len(scenes_with_img)/len(all_scene_ids):.1f}%)")
                print(f"  Scenes with both: {len(scenes_with_both)} ({100*len(scenes_with_both)/len(all_scene_ids):.1f}%)")
            else:
                print("  No scenes found in ScanQA data")

            missing = all_scene_ids - scenes_with_both
            if missing:
                print(f"\n  Missing scenes (first 10): {list(missing)[:10]}")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Ensure ScanNet data is preprocessed:")
    print(f"   python experiments/scannet_composition/scripts/preprocess_scannet.py \\")
    print(f"     --scannet-root /path/to/scannet --output-dir {output_dir}")
    print(f"\n2. Run training:")
    print(f"   MODALITY=both sbatch experiments/scanqa_composition/scripts/train_qa.sh")


if __name__ == "__main__":
    main()
