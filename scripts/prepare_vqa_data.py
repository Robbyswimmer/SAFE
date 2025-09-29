#!/usr/bin/env python3
"""
Convert VQA v2 dataset to JSONL format expected by SAFE datasets.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_vqa_questions(questions_file: Path):
    """Load VQA questions JSON file."""
    print(f"Loading questions from {questions_file}", flush=True)
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = {}
    for q in data.get('questions', []):
        question_id = q['question_id']
        questions[question_id] = {
            'image_id': q['image_id'],
            'question': q['question']
        }

    print(f"Loaded {len(questions)} questions", flush=True)
    return questions


def load_vqa_annotations(annotations_file: Path):
    """Load VQA annotations JSON file."""
    print(f"Loading annotations from {annotations_file}", flush=True)
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = {}
    for ann in data.get('annotations', []):
        question_id = ann['question_id']
        # Get most common answer
        answers = [a['answer'] for a in ann.get('answers', [])]
        annotations[question_id] = {
            'answers': answers,
            'answer': answers[0] if answers else "",
            'question_type': ann.get('question_type', ''),
            'answer_type': ann.get('answer_type', '')
        }

    print(f"Loaded {len(annotations)} annotations", flush=True)
    return annotations


def find_image_path(image_id: int, coco_dir: Path, split: str):
    """Find the COCO image path for a given image_id."""
    # VQA uses COCO images - try common patterns
    patterns = [
        f"train2014/COCO_train2014_{image_id:012d}.jpg",
        f"val2014/COCO_val2014_{image_id:012d}.jpg",
        f"COCO_train2014_{image_id:012d}.jpg",
        f"COCO_val2014_{image_id:012d}.jpg",
    ]

    for pattern in patterns:
        img_path = coco_dir / pattern
        if img_path.exists():
            # Return relative path from data root
            return str(img_path.relative_to(coco_dir.parent))

    return None


def convert_vqa_to_jsonl(questions_file: Path, annotations_file: Path,
                         output_path: Path, coco_dir: Path, split: str):
    """Convert VQA questions and annotations to JSONL format."""
    print(f"\nConverting VQA {split} split to {output_path}", flush=True)

    questions = load_vqa_questions(questions_file)
    annotations = load_vqa_annotations(annotations_file) if annotations_file.exists() else {}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries_written = 0
    entries_skipped_no_image = 0
    entries_skipped_no_annotation = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for question_id, q_data in questions.items():
            # Skip if no annotation (test set doesn't have annotations)
            if question_id not in annotations:
                entries_skipped_no_annotation += 1
                continue

            ann_data = annotations[question_id]
            image_id = q_data['image_id']

            # Find image path
            image_path = find_image_path(image_id, coco_dir, split)
            if image_path is None:
                entries_skipped_no_image += 1
                continue

            # Create JSONL entry
            entry = {
                "id": f"vqa_{split}_{question_id}",
                "sample_id": str(question_id),
                "question": q_data['question'],
                "answer": ann_data['answer'],
                "image_path": image_path,
                "metadata": {
                    "image_id": image_id,
                    "split": split,
                    "question_type": ann_data.get('question_type', ''),
                    "answer_type": ann_data.get('answer_type', ''),
                    "all_answers": ann_data.get('answers', [])
                }
            }

            f.write(json.dumps(entry) + '\n')
            entries_written += 1

            # Progress indicator
            if entries_written % 10000 == 0:
                print(f"  Processed {entries_written} entries...", flush=True)

    print(f"Wrote {entries_written} entries to {output_path}", flush=True)
    if entries_skipped_no_annotation > 0:
        print(f"Skipped {entries_skipped_no_annotation} entries (no annotation)", flush=True)
    if entries_skipped_no_image > 0:
        print(f"Skipped {entries_skipped_no_image} entries (no image file found)", flush=True)

    return entries_written


def main():
    parser = argparse.ArgumentParser(description="Convert VQA v2 to JSONL")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("experiments/full_training/data"),
        help="Root directory containing vqa and coco data"
    )

    args = parser.parse_args()

    print("Starting VQA to JSONL conversion", flush=True)
    print(f"Data root: {args.data_root}", flush=True)

    data_root = args.data_root.expanduser().resolve()
    vqa_dir = data_root / "vqa"
    coco_dir = data_root / "coco"

    print(f"VQA directory: {vqa_dir}", flush=True)
    print(f"COCO directory: {coco_dir}", flush=True)

    if not vqa_dir.exists():
        raise FileNotFoundError(f"VQA directory not found: {vqa_dir}")

    if not coco_dir.exists():
        print(f"Warning: COCO directory not found: {coco_dir}", flush=True)
        print("Images will be skipped if not found", flush=True)

    # Define file mappings for train and val splits
    splits_config = {
        'train': {
            'questions': 'v2_OpenEnded_mscoco_train2014_questions.json',
            'annotations': 'v2_mscoco_train2014_annotations.json',
        },
        'val': {
            'questions': 'v2_OpenEnded_mscoco_val2014_questions.json',
            'annotations': 'v2_mscoco_val2014_annotations.json',
        }
    }

    total_entries = 0

    for split, config in splits_config.items():
        # Find questions file (may be in subdirectories)
        questions_candidates = [
            vqa_dir / config['questions'],
            vqa_dir / 'Questions' / config['questions'],
            vqa_dir / 'questions' / config['questions'],
        ]
        questions_file = next((p for p in questions_candidates if p.exists()), None)

        # Find annotations file
        annotations_candidates = [
            vqa_dir / config['annotations'],
            vqa_dir / 'Annotations' / config['annotations'],
            vqa_dir / 'annotations' / config['annotations'],
        ]
        annotations_file = next((p for p in annotations_candidates if p.exists()), None)

        if questions_file is None:
            print(f"Skipping {split}: questions file not found", flush=True)
            print(f"  Looked for: {config['questions']}", flush=True)
            continue

        if annotations_file is None:
            print(f"Skipping {split}: annotations file not found", flush=True)
            print(f"  Looked for: {config['annotations']}", flush=True)
            continue

        output_path = vqa_dir / f"{split}.jsonl"

        entries = convert_vqa_to_jsonl(
            questions_file,
            annotations_file,
            output_path,
            coco_dir,
            split
        )
        total_entries += entries

    print(f"\n✓ Conversion complete! Total entries: {total_entries}", flush=True)
    print(f"\nGenerated files:", flush=True)
    for split in splits_config.keys():
        jsonl_path = vqa_dir / f"{split}.jsonl"
        if jsonl_path.exists():
            print(f"  - {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()