#!/usr/bin/env python3
"""
Preprocess SQA3D dataset for QA composition experiments.

This script:
1. Copies SQA3D question + annotation files to the data directory
2. Validates that corresponding ScanNet scene data exists
3. Reports statistics on data coverage

Prerequisites:
- SQA3D downloaded from https://github.com/SilongYong/SQA3D
- ScanNet preprocessed using experiments/scannet_composition/scripts/preprocess_scannet.py

Usage:
    python experiments/sqa3d_composition/scripts/preprocess_sqa3d.py \
        --sqa3d-root /path/to/SQA3D/data \
        --output-dir experiments/full_training/data
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict


def analyze_questions(questions: list) -> dict:
    """Analyze question types in the dataset."""
    question_types = defaultdict(int)

    keywords = {
        "color": ["color", "colored"],
        "count": ["how many", "number of", "count"],
        "spatial": ["next to", "near", "behind", "in front", "above", "below",
                     "between", "left", "right", "facing", "opposite"],
        "object": ["what is", "what are", "what kind", "what type"],
        "location": ["where is", "where are", "location"],
        "yes_no": ["is there", "are there", "is it", "are they", "does", "do", "can"],
        "attribute": ["size", "shape", "material", "big", "small", "large"],
    }

    for item in questions:
        question = item.get("question", "").lower()
        categorized = False

        for qtype, kws in keywords.items():
            if any(kw in question for kw in kws):
                question_types[qtype] += 1
                categorized = True
                break

        if not categorized:
            question_types["other"] += 1

    return dict(question_types)


def find_file(directory: Path, candidates: list) -> Path | None:
    """Find first existing file from a list of candidates."""
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess SQA3D for QA composition")
    parser.add_argument("--sqa3d-root", type=str, required=True,
                        help="Path to SQA3D data directory")
    parser.add_argument("--output-dir", type=str, default="experiments/full_training/data",
                        help="Output directory")
    parser.add_argument("--validate-scenes", action="store_true",
                        help="Check that ScanNet scene data exists")
    args = parser.parse_args()

    sqa3d_root = Path(args.sqa3d_root)
    output_dir = Path(args.output_dir)

    # Create output directory
    sqa3d_output = output_dir / "sqa3d"
    sqa3d_output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SQA3D Preprocessing")
    print("=" * 60)
    print(f"SQA3D root: {sqa3d_root}")
    print(f"Output dir: {sqa3d_output}")
    print("=" * 60)

    splits = ["train", "val", "test"]
    all_scene_ids = set()

    for split in splits:
        # --- Find question file ---
        q_candidates = [
            f"v1_balanced_questions_{split}_scannetv2.json",
            f"questions/v1_balanced_questions_{split}_scannetv2.json",
            f"v1_balanced_questions_{split}.json",
            f"questions_{split}.json",
            f"sqa3d_{split}_questions.json",
        ]
        q_file = find_file(sqa3d_root, q_candidates)

        # --- Find annotation file ---
        a_candidates = [
            f"v1_balanced_sqa_annotations_{split}_scannetv2.json",
            f"annotations/v1_balanced_sqa_annotations_{split}_scannetv2.json",
            f"v1_balanced_sqa_annotations_{split}.json",
            f"annotations_{split}.json",
            f"sqa3d_{split}_annotations.json",
        ]
        a_file = find_file(sqa3d_root, a_candidates)

        if q_file is None:
            print(f"\nWarning: No question file found for split '{split}'")
            continue
        if a_file is None:
            print(f"\nWarning: No annotation file found for split '{split}'")
            continue

        # Load and analyze
        with open(q_file) as f:
            q_data = json.load(f)
        with open(a_file) as f:
            a_data = json.load(f)

        # SQA3D wraps in {"questions": [...]} / {"annotations": [...]}
        questions_list = q_data.get("questions", q_data) if isinstance(q_data, dict) else q_data
        annotations_list = a_data.get("annotations", a_data) if isinstance(a_data, dict) else a_data
        if isinstance(questions_list, dict):
            questions_list = list(questions_list.values())
        if isinstance(annotations_list, dict):
            annotations_list = list(annotations_list.values())

        print(f"\n{split.upper()} split:")
        print(f"  Questions: {len(questions_list)}")
        print(f"  Annotations: {len(annotations_list)}")

        # Extract scene IDs
        scene_ids = set()
        for item in questions_list:
            scene_id = item.get("scene_id", item.get("scan_id"))
            if scene_id:
                scene_ids.add(scene_id)
                all_scene_ids.add(scene_id)
        print(f"  Unique scenes: {len(scene_ids)}")

        # Check situation field coverage
        with_situation = sum(1 for q in questions_list if q.get("situation"))
        print(f"  With situation: {with_situation}/{len(questions_list)}")

        # Analyze question types
        qtypes = analyze_questions(questions_list)
        print(f"  Question types:")
        for qtype, count in sorted(qtypes.items(), key=lambda x: -x[1]):
            pct = 100 * count / max(len(questions_list), 1)
            print(f"    {qtype}: {count} ({pct:.1f}%)")

        # Copy files to output using canonical names
        out_q = sqa3d_output / f"v1_balanced_questions_{split}_scannetv2.json"
        out_a = sqa3d_output / f"v1_balanced_sqa_annotations_{split}_scannetv2.json"
        shutil.copy(q_file, out_q)
        shutil.copy(a_file, out_a)
        print(f"  Saved questions:    {out_q}")
        print(f"  Saved annotations:  {out_a}")

    # Copy answer_dict.json if it exists
    for name in ["answer_dict.json", "sqa3d_answer_dict.json"]:
        ad_path = sqa3d_root / name
        if ad_path.exists():
            out_ad = sqa3d_output / "answer_dict.json"
            shutil.copy(ad_path, out_ad)
            print(f"\nCopied answer dict: {out_ad}")
            break

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
            print(f"  Total scenes in SQA3D: {len(all_scene_ids)}")
            if len(all_scene_ids) > 0:
                print(f"  Scenes with point cloud: {len(scenes_with_pc)} ({100*len(scenes_with_pc)/len(all_scene_ids):.1f}%)")
                print(f"  Scenes with image: {len(scenes_with_img)} ({100*len(scenes_with_img)/len(all_scene_ids):.1f}%)")
                print(f"  Scenes with both: {len(scenes_with_both)} ({100*len(scenes_with_both)/len(all_scene_ids):.1f}%)")
            else:
                print("  No scenes found in SQA3D data")

            missing = all_scene_ids - scenes_with_both
            if missing:
                print(f"\n  Missing scenes (first 10): {sorted(missing)[:10]}")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Ensure ScanNet data is preprocessed:")
    print(f"   python experiments/scannet_composition/scripts/preprocess_scannet.py \\")
    print(f"     --scannet-root /path/to/scannet --output-dir {output_dir}")
    print(f"\n2. Run training:")
    print(f"   sbatch experiments/sqa3d_composition/scripts/train_sqa3d.sh")


if __name__ == "__main__":
    main()
