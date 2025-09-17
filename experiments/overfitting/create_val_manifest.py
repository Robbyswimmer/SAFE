"""Create a reproducible validation subset manifest from VQA v2.

The script selects a deterministic subset of questions (default 500) from the
VQA validation annotations and emits a compact manifest that we can commit to
Git for reproducible evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class VQAQuestion:
    question_id: int
    image_id: int
    question: str


@dataclass
class VQAAnnotation:
    question_id: int
    answers: List[str]


def load_questions(path: Path) -> Dict[int, VQAQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = {}
    for item in data.get("questions", []):
        q = VQAQuestion(
            question_id=item["question_id"],
            image_id=item["image_id"],
            question=item["question"].strip(),
        )
        questions[q.question_id] = q
    return questions


def load_annotations(path: Path) -> Dict[int, VQAAnnotation]:
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations = {}
    for item in data.get("annotations", []):
        ans_list = [ans["answer"].strip() for ans in item.get("answers", [])]
        annotations[item["question_id"]] = VQAAnnotation(
            question_id=item["question_id"],
            answers=ans_list,
        )
    return annotations


def majority_answer(answers: List[str]) -> str:
    if not answers:
        return ""
    counts = Counter(ans.lower() for ans in answers if ans)
    if not counts:
        return ""
    top_answer, _ = counts.most_common(1)[0]
    return top_answer


def build_manifest(
    questions: Dict[int, VQAQuestion],
    annotations: Dict[int, VQAAnnotation],
    subset_size: int,
    seed: int,
) -> Dict[str, object]:
    rng = random.Random(seed)
    available_ids = sorted(set(questions.keys()) & set(annotations.keys()))
    if len(available_ids) < subset_size:
        raise ValueError(
            f"Requested {subset_size} validation items but only {len(available_ids)} available"
        )

    chosen_ids = sorted(rng.sample(available_ids, subset_size))

    entries = []
    for qid in chosen_ids:
        q = questions[qid]
        ann = annotations[qid]
        maj = majority_answer(ann.answers)
        entries.append(
            {
                "question_id": q.question_id,
                "image_id": q.image_id,
                "image_filename": f"COCO_val2014_{q.image_id:012d}.jpg",
                "question": q.question,
                "answers": ann.answers,
                "majority_answer": maj,
            }
        )

    manifest = {
        "subset_name": "vqa_val_overfit_v1",
        "description": "VQA v2 validation subset for SAFE overfitting regression checks",
        "seed": seed,
        "source_dataset": "VQA v2 validation",
        "entries": entries,
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create VQA validation subset manifest")
    parser.add_argument("--questions", type=Path,
                        default=Path("data/vqa/v2_OpenEnded_mscoco_val2014_questions.json"),
                        help="Path to VQA questions JSON")
    parser.add_argument("--annotations", type=Path,
                        default=Path("data/vqa/v2_mscoco_val2014_annotations.json"),
                        help="Path to VQA annotations JSON")
    parser.add_argument("--subset-size", type=int, default=500,
                        help="Number of validation items to include")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for deterministic sampling")
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/overfitting/val_manifest.json"),
                        help="Output manifest path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    questions = load_questions(args.questions)
    annotations = load_annotations(args.annotations)
    manifest = build_manifest(questions, annotations, args.subset_size, args.seed)

    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote validation manifest with {len(manifest['entries'])} entries to {args.output}")


if __name__ == "__main__":
    main()
