#!/usr/bin/env python3
"""
Build text-conditioned per-question images for SQA3D.

This is a pragmatic fallback for SQA3D's free-form "situation" text when no
numeric pose/orientation is available in the annotations. For each question,
the script retrieves the most relevant frame from ScanNet scene frames using a
CLIP-style text-image similarity score, then saves:

  scannet/posed_images/{question_id}.jpg

The SQA3D dataset loader can then prefer these question-specific images over
scene-level single/multiview images.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def find_question_file(sqa3d_dir: Path, split: str) -> Path:
    candidates = [
        sqa3d_dir / f"v1_balanced_questions_{split}_scannetv2.json",
        sqa3d_dir / "questions" / f"v1_balanced_questions_{split}_scannetv2.json",
        sqa3d_dir / f"v1_balanced_questions_{split}.json",
        sqa3d_dir / f"questions_{split}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find SQA3D question file for split={split}")


def load_questions(sqa3d_dir: Path, split: str) -> List[Dict]:
    q_file = find_question_file(sqa3d_dir, split)
    with open(q_file, "r", encoding="utf-8") as f:
        q_data = json.load(f)
    questions = q_data.get("questions", q_data) if isinstance(q_data, dict) else q_data
    if isinstance(questions, dict):
        questions = list(questions.values())
    return list(questions)


def candidate_frame_roots(frames_root: Path) -> List[Path]:
    roots = [
        frames_root,
        frames_root / "scannet_frames_25k",
        frames_root / "frames_25k",
    ]
    deduped: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        if root.exists():
            deduped.append(root)
    return deduped


def scene_frame_paths(scene_id: str, frame_roots: Iterable[Path], max_candidates: int) -> List[Path]:
    for root in frame_roots:
        scene_dir = root / scene_id / "color"
        if not scene_dir.exists():
            scene_dir = root / scene_id
        if not scene_dir.exists():
            continue
        frames = sorted(scene_dir.glob("*.jpg")) + sorted(scene_dir.glob("*.png"))
        if not frames:
            continue
        if len(frames) <= max_candidates:
            return frames
        step = max(1, len(frames) // max_candidates)
        sampled = frames[::step][:max_candidates]
        if sampled:
            return sampled
    return []


def build_query(question_record: Dict) -> str:
    parts = []
    situation = str(question_record.get("situation", "")).strip()
    if situation:
        parts.append(f"Situation: {situation}")
    alt = question_record.get("alternative_situation", [])
    if isinstance(alt, list) and alt:
        parts.append(" ".join(str(x).strip() for x in alt[:2] if str(x).strip()))
    question = str(question_record.get("question", "")).strip()
    if question:
        parts.append(f"Question: {question}")
    return " ".join(parts).strip()


def load_clip(device: torch.device):
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


def unwrap_feature_tensor(value: torch.Tensor | object, preferred_attr: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value

    candidate = getattr(value, preferred_attr, None)
    if torch.is_tensor(candidate):
        return candidate

    pooler = getattr(value, "pooler_output", None)
    if torch.is_tensor(pooler):
        return pooler

    if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
        return value[0]

    raise TypeError(f"Could not unwrap feature tensor from value of type {type(value).__name__}")


@torch.no_grad()
def select_best_frame(query: str, frame_paths: List[Path], model, processor, device: torch.device, batch_size: int) -> Path | None:
    if not frame_paths:
        return None

    text_inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(device)
    text_features = unwrap_feature_tensor(model.get_text_features(**text_inputs), "text_embeds")
    text_features = F.normalize(text_features, dim=-1)

    best_score = -math.inf
    best_path: Path | None = None

    for start in range(0, len(frame_paths), batch_size):
        chunk = frame_paths[start:start + batch_size]
        images = [Image.open(path).convert("RGB") for path in chunk]
        image_inputs = processor(images=images, return_tensors="pt").to(device)
        image_features = unwrap_feature_tensor(model.get_image_features(**image_inputs), "image_embeds")
        image_features = F.normalize(image_features, dim=-1)
        scores = torch.matmul(image_features, text_features.T).squeeze(-1)
        max_idx = int(torch.argmax(scores).item())
        score = float(scores[max_idx].item())
        if score > best_score:
            best_score = score
            best_path = chunk[max_idx]

    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build text-conditioned SQA3D retrieval images")
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing sqa3d/ and scannet/")
    parser.add_argument("--frames-root", type=Path, required=True, help="Extracted ScanNet frames_25k root")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-questions", type=int, default=0, help="Limit questions for a quick run (0=all)")
    parser.add_argument("--max-candidate-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_root = args.data_root
    sqa3d_dir = data_root / "sqa3d"
    posed_dir = data_root / "scannet" / "posed_images"
    posed_dir.mkdir(parents=True, exist_ok=True)

    frame_roots = candidate_frame_roots(args.frames_root)
    if not frame_roots:
        raise FileNotFoundError(f"No valid frame roots found under {args.frames_root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, processor = load_clip(device)

    questions = load_questions(sqa3d_dir, args.split)
    if args.max_questions > 0:
        questions = questions[:args.max_questions]

    print(f"Loaded {len(questions)} questions for split={args.split}")
    print(f"Frame roots: {[str(x) for x in frame_roots]}")

    copied = 0
    missing = 0
    for record in tqdm(questions):
        question_id = str(record.get("question_id"))
        scene_id = str(record.get("scene_id", record.get("scan_id", "")))
        if not question_id or not scene_id:
            missing += 1
            continue

        out_path = posed_dir / f"{question_id}.jpg"
        if out_path.exists():
            copied += 1
            continue

        frame_paths = scene_frame_paths(scene_id, frame_roots, max_candidates=int(args.max_candidate_frames))
        if not frame_paths:
            missing += 1
            continue

        query = build_query(record)
        best = select_best_frame(
            query=query,
            frame_paths=frame_paths,
            model=model,
            processor=processor,
            device=device,
            batch_size=int(args.batch_size),
        )
        if best is None:
            missing += 1
            continue
        shutil.copy2(best, out_path)
        copied += 1

    print(f"Saved posed images: {copied}")
    print(f"Missing questions:   {missing}")
    print(f"Output dir:          {posed_dir}")


if __name__ == "__main__":
    main()
