"""
SQA3D dataset for point cloud + image QA composition experiments.

SQA3D: Situated Question Answering in 3D Scenes.
Tests whether combining point cloud and image modalities improves 3D QA.

Supports three modalities:
- Point cloud only
- Image only
- Point cloud + Image (composition)

SQA3D format differences from ScanQA:
- Separate questions + annotations files joined on question_id
- Questions in {"questions": [...]}, annotations in {"annotations": [...]}
- Answers: [{"answer": "text"}, ...] (not bare strings)
- Includes "situation" field (viewpoint context)
"""

from __future__ import annotations

import json
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from PIL import Image


class SQA3DDataset(Dataset):
    """
    SQA3D dataset for 3D question answering with point cloud + image.

    Supports three modes:
    - "pointcloud": Only point cloud input
    - "image": Only RGB image input
    - "both": Point cloud + image composition
    """

    dataset_name = "sqa3d"

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        modality: str = "both",
        num_points: int = 8192,
        image_size: int = 224,
        augment: bool = None,
        transform=None,
        max_answer_length: int = 64,
        include_situation: bool = False,
    ):
        self.data_path = Path(data_path)
        self.split = split.lower()
        self.modality = modality
        self.num_points = num_points
        self.image_size = image_size
        self.augment = augment if augment is not None else (self.split == "train")
        self.transform = transform
        self.max_answer_length = max_answer_length
        self.include_situation = include_situation

        # Load QA pairs
        self.samples = self._load_samples()

        # Build scene to point cloud/image mapping
        self.scene_data = self._build_scene_mapping()

        print(f"[SQA3D] Loaded {len(self.samples)} QA pairs ({self.split}, modality={modality})")
        print(f"[SQA3D] Unique scenes: {len(self.scene_data)}")
        if include_situation:
            print(f"[SQA3D] Situation context: enabled")

    def _load_samples(self) -> List[Dict]:
        """Load QA pairs from SQA3D JSON files (questions + annotations)."""
        sqa3d_dir = self.data_path / "sqa3d"

        # Find question file
        q_candidates = [
            sqa3d_dir / f"v1_balanced_questions_{self.split}_scannetv2.json",
            sqa3d_dir / f"questions" / f"v1_balanced_questions_{self.split}_scannetv2.json",
            sqa3d_dir / f"v1_balanced_questions_{self.split}.json",
            sqa3d_dir / f"questions_{self.split}.json",
            sqa3d_dir / f"sqa3d_{self.split}_questions.json",
        ]
        q_file = None
        for candidate in q_candidates:
            if candidate.exists():
                q_file = candidate
                break

        # Find annotation file
        a_candidates = [
            sqa3d_dir / f"v1_balanced_sqa_annotations_{self.split}_scannetv2.json",
            sqa3d_dir / f"annotations" / f"v1_balanced_sqa_annotations_{self.split}_scannetv2.json",
            sqa3d_dir / f"v1_balanced_sqa_annotations_{self.split}.json",
            sqa3d_dir / f"annotations_{self.split}.json",
            sqa3d_dir / f"sqa3d_{self.split}_annotations.json",
        ]
        a_file = None
        for candidate in a_candidates:
            if candidate.exists():
                a_file = candidate
                break

        if q_file is None:
            searched = "\n  ".join(str(c) for c in q_candidates)
            raise FileNotFoundError(
                f"SQA3D question file not found. Searched:\n  {searched}\n"
                f"Download SQA3D from: https://github.com/SilongYong/SQA3D"
            )
        if a_file is None:
            searched = "\n  ".join(str(c) for c in a_candidates)
            raise FileNotFoundError(
                f"SQA3D annotation file not found. Searched:\n  {searched}\n"
                f"Download SQA3D from: https://github.com/SilongYong/SQA3D"
            )

        print(f"[SQA3D] Loading questions from: {q_file}")
        print(f"[SQA3D] Loading annotations from: {a_file}")

        with open(q_file) as f:
            q_data = json.load(f)
        with open(a_file) as f:
            a_data = json.load(f)

        # SQA3D format: questions in {"questions": [...]}, annotations in {"annotations": [...]}
        questions_list = q_data.get("questions", q_data) if isinstance(q_data, dict) else q_data
        annotations_list = a_data.get("annotations", a_data) if isinstance(a_data, dict) else a_data

        if isinstance(questions_list, dict):
            questions_list = list(questions_list.values())
        if isinstance(annotations_list, dict):
            annotations_list = list(annotations_list.values())

        # Build annotation lookup by question_id
        ann_by_qid = {}
        for ann in annotations_list:
            qid = ann.get("question_id")
            if qid is not None:
                ann_by_qid[qid] = ann

        # Join questions + annotations
        samples = []
        skipped = 0
        for q_item in questions_list:
            qid = q_item.get("question_id")
            ann = ann_by_qid.get(qid)
            if ann is None:
                skipped += 1
                continue

            scene_id = q_item.get("scene_id", q_item.get("scan_id", ann.get("scene_id", ann.get("scan_id"))))
            question = q_item.get("question", "")
            situation = q_item.get("situation", ann.get("situation", ""))

            # Extract answers: [{"answer": "text"}, ...] or bare strings
            raw_answers = ann.get("answers", [])
            answers = []
            for a in raw_answers:
                if isinstance(a, dict):
                    answers.append(a.get("answer", str(a)))
                else:
                    answers.append(str(a))

            if not answers:
                # Fallback to single answer field
                single = ann.get("answer", q_item.get("answer", ""))
                if single:
                    answers = [str(single)]

            answer = answers[0] if answers else ""

            samples.append({
                "scene_id": scene_id,
                "question": question,
                "situation": situation,
                "answer": answer,
                "all_answers": answers,
                "question_id": str(qid),
                "question_type": ann.get("question_type", q_item.get("question_type", "unknown")),
            })

        if skipped > 0:
            print(f"[SQA3D] Skipped {skipped} questions (no matching annotation)")

        return samples

    def _build_scene_mapping(self) -> Dict[str, Dict]:
        """Build mapping from scene_id to point cloud/image paths."""
        scene_data = {}
        scannet_dir = self.data_path / "scannet"

        scene_ids = set(s["scene_id"] for s in self.samples if s.get("scene_id"))

        for scene_id in scene_ids:
            pc_path = scannet_dir / "pointclouds" / f"{scene_id}.npy"
            img_path = scannet_dir / "images" / f"{scene_id}.jpg"

            scene_data[scene_id] = {
                "pointcloud_path": pc_path if pc_path.exists() else None,
                "image_path": img_path if img_path.exists() else None,
            }

        # Filter samples to only include scenes with required data
        valid_samples = []
        for sample in self.samples:
            scene = scene_data.get(sample["scene_id"], {})

            if self.modality == "pointcloud" and scene.get("pointcloud_path") is None:
                continue
            elif self.modality == "image" and scene.get("image_path") is None:
                continue
            elif self.modality == "both":
                if scene.get("pointcloud_path") is None or scene.get("image_path") is None:
                    continue

            valid_samples.append(sample)

        removed = len(self.samples) - len(valid_samples)
        if removed > 0:
            print(f"[SQA3D] Filtered {removed} samples (missing modality data)")

        self.samples = valid_samples
        return scene_data

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pointcloud(self, pc_path: Path) -> np.ndarray:
        """Load and preprocess point cloud."""
        pc = np.load(str(pc_path))

        if len(pc) > self.num_points:
            indices = np.random.choice(len(pc), self.num_points, replace=False)
            pc = pc[indices]
        elif len(pc) < self.num_points:
            pad_size = self.num_points - len(pc)
            pad_indices = np.random.choice(len(pc), pad_size, replace=True)
            pc = np.concatenate([pc, pc[pad_indices]], axis=0)

        if pc.shape[-1] > 3:
            pc = pc[:, :3]

        centroid = pc.mean(axis=0)
        pc = pc - centroid
        max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if max_dist > 0:
            pc = pc / max_dist

        if self.augment:
            pc = self._augment_pointcloud(pc)

        return pc.astype(np.float32)

    def _augment_pointcloud(self, pc: np.ndarray) -> np.ndarray:
        """Apply point cloud augmentations."""
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation = np.array([
            [cos_t, 0, sin_t],
            [0, 1, 0],
            [-sin_t, 0, cos_t]
        ])
        pc = pc @ rotation.T

        scale = np.random.uniform(0.9, 1.1)
        pc = pc * scale

        jitter = np.random.normal(0, 0.01, pc.shape)
        pc = pc + np.clip(jitter, -0.03, 0.03)

        return pc

    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and preprocess image."""
        img = Image.open(str(image_path)).convert("RGB")

        if img.size != (self.image_size, self.image_size):
            resample = getattr(Image, 'Resampling', Image).BILINEAR
            img = img.resize((self.image_size, self.image_size), resample)

        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        sample = self.samples[idx]
        scene_id = sample["scene_id"]
        scene = self.scene_data[scene_id]

        question = sample["question"]
        if self.include_situation and sample.get("situation"):
            question = f"Situation: {sample['situation']} Question: {question}"

        result = {
            "sample_id": sample["question_id"],
            "scene_id": scene_id,
            "question": question,
            "situation": sample.get("situation", ""),
            "answers": sample["answer"],
            "all_answers": sample["all_answers"],
            "question_type": sample.get("question_type", "unknown"),
        }

        if self.modality in ["pointcloud", "both"] and scene["pointcloud_path"]:
            pc = self._load_pointcloud(scene["pointcloud_path"])
            result["pointcloud"] = torch.from_numpy(pc).float()
        else:
            result["pointcloud"] = None

        if self.modality in ["image", "both"] and scene["image_path"]:
            img = self._load_image(scene["image_path"])
            if self.transform:
                img = self.transform(img)
            result["image"] = img
        else:
            result["image"] = None

        result["audio"] = None  # Not used

        return result


def collate_sqa3d_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for SQA3D dataset."""
    if len(batch) == 0:
        return {}

    result = {
        "sample_ids": [s["sample_id"] for s in batch],
        "scene_ids": [s["scene_id"] for s in batch],
        "questions": [s["question"] for s in batch],
        "situations": [s.get("situation", "") for s in batch],
        "answers": [s["answers"] for s in batch],
        "all_answers": [s["all_answers"] for s in batch],
        "question_types": [s.get("question_type", "unknown") for s in batch],
    }

    if batch[0].get("pointcloud") is not None:
        result["pointclouds"] = torch.stack([s["pointcloud"] for s in batch])
    else:
        result["pointclouds"] = None

    if batch[0].get("image") is not None:
        if isinstance(batch[0]["image"], torch.Tensor):
            result["images"] = torch.stack([s["image"] for s in batch])
        else:
            result["images"] = [s["image"] for s in batch]
    else:
        result["images"] = None

    return result
