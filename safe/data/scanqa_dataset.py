"""
ScanQA dataset for point cloud + image QA composition experiments.

ScanQA: 41K QA pairs from 800 ScanNet scenes.
Tests whether combining point cloud and image modalities improves 3D QA.

Supports three modalities:
- Point cloud only
- Image only
- Point cloud + Image (composition)
"""

from __future__ import annotations

import json
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from PIL import Image


class ScanQADataset(Dataset):
    """
    ScanQA dataset for 3D question answering with point cloud + image.

    Supports three modes:
    - "pointcloud": Only point cloud input
    - "image": Only RGB image input
    - "both": Point cloud + image composition
    """

    dataset_name = "scanqa"

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        modality: str = "both",  # "pointcloud", "image", or "both"
        num_points: int = 8192,
        image_size: int = 224,
        augment: bool = None,
        transform=None,
        max_answer_length: int = 64,
        prefer_multiview_images: bool = True,
    ):
        """
        Initialize ScanQA dataset.

        Args:
            data_path: Root data directory containing scanqa/ and scannet/
            split: "train", "val", or "test"
            modality: Which modalities to load ("pointcloud", "image", "both")
            num_points: Number of points to sample from point cloud
            image_size: Size to resize images to
            augment: Whether to apply augmentation (default: True for train)
            transform: Optional image transform
            max_answer_length: Maximum answer length (for truncation)
        """
        self.data_path = Path(data_path)
        self.split = split.lower()
        self.modality = modality
        self.num_points = num_points
        self.image_size = image_size
        self.augment = augment if augment is not None else (self.split == "train")
        self.transform = transform
        self.max_answer_length = max_answer_length
        self.prefer_multiview_images = prefer_multiview_images

        # Load QA pairs
        self.samples = self._load_samples()

        # Build scene to point cloud/image mapping
        self.scene_data = self._build_scene_mapping()

        print(f"[ScanQA] Loaded {len(self.samples)} QA pairs ({self.split}, modality={modality})")
        print(f"[ScanQA] Unique scenes: {len(self.scene_data)}")
        if modality in {"image", "both"}:
            preferred = "multiview" if prefer_multiview_images else "single-view"
            print(f"[ScanQA] Image source: {preferred} when available")
            self._log_image_source_stats()

    def _load_samples(self) -> List[Dict]:
        """Load QA pairs from JSON."""
        # Try different file naming conventions and subdirectory layouts
        scanqa_dir = self.data_path / "scanqa"
        candidates = [
            scanqa_dir / f"ScanQA_v1.0_{self.split}.json",
            scanqa_dir / "ScanQA_v1.0" / f"ScanQA_v1.0_{self.split}.json",
            scanqa_dir / "qa" / f"ScanQA_v1.0_{self.split}.json",
            scanqa_dir / f"scanqa_{self.split}.json",
        ]
        qa_file = None
        for candidate in candidates:
            if candidate.exists():
                qa_file = candidate
                break
        if qa_file is None:
            searched = "\n  ".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"QA file not found. Searched:\n  {searched}\n"
                f"Download ScanQA from: https://github.com/ATR-DBI/ScanQA"
            )

        with open(qa_file) as f:
            data = json.load(f)

        # ScanQA format: list of dicts with scene_id, question, answers, etc.
        samples = []
        for item in data:
            scene_id = item.get("scene_id", item.get("scan_id"))
            question = item.get("question")

            # Answers can be a list or a single string
            answers = item.get("answers", item.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]

            # Use first answer as primary
            answer = answers[0] if answers else ""

            samples.append({
                "scene_id": scene_id,
                "question": question,
                "answer": answer,
                "all_answers": answers,
                "question_id": item.get("question_id", f"{scene_id}_{len(samples)}"),
                "object_ids": item.get("object_ids", []),
                "object_names": item.get("object_names", []),
            })

        return samples

    def _build_scene_mapping(self) -> Dict[str, Dict]:
        """Build mapping from scene_id to point cloud/image paths."""
        scene_data = {}
        scannet_dir = self.data_path / "scannet"

        # Get unique scene IDs from samples
        scene_ids = set(s["scene_id"] for s in self.samples)

        for scene_id in scene_ids:
            pc_path = scannet_dir / "pointclouds" / f"{scene_id}.npy"
            single_img_path = scannet_dir / "images" / f"{scene_id}.jpg"
            multiview_img_path = scannet_dir / "multiview_images" / f"{scene_id}.jpg"

            preferred_img_path = None
            if self.prefer_multiview_images and multiview_img_path.exists():
                preferred_img_path = multiview_img_path
            elif single_img_path.exists():
                preferred_img_path = single_img_path
            elif multiview_img_path.exists():
                preferred_img_path = multiview_img_path

            scene_data[scene_id] = {
                "pointcloud_path": pc_path if pc_path.exists() else None,
                "image_path": preferred_img_path,
                "single_image_path": single_img_path if single_img_path.exists() else None,
                "multiview_image_path": multiview_img_path if multiview_img_path.exists() else None,
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
            print(f"[ScanQA] Filtered {removed} samples (missing modality data)")

        self.samples = valid_samples
        return scene_data

    def _log_image_source_stats(self) -> None:
        multiview = 0
        single = 0
        missing = 0
        for scene in self.scene_data.values():
            image_path = scene.get("image_path")
            if image_path is None:
                missing += 1
                continue
            if scene.get("multiview_image_path") is not None and image_path == scene.get("multiview_image_path"):
                multiview += 1
            elif scene.get("single_image_path") is not None and image_path == scene.get("single_image_path"):
                single += 1
        print(
            f"[ScanQA] Image resolution: multiview={multiview} single={single} missing={missing}",
            flush=True,
        )
        if self.prefer_multiview_images and multiview == 0:
            print(
                "[ScanQA] Warning: no multiview images found; falling back to single scene images",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pointcloud(self, pc_path: Path) -> np.ndarray:
        """Load and preprocess point cloud."""
        pc = np.load(str(pc_path))  # (N, 3) or (N, 6) with normals

        # Subsample to target number of points
        if len(pc) > self.num_points:
            indices = np.random.choice(len(pc), self.num_points, replace=False)
            pc = pc[indices]
        elif len(pc) < self.num_points:
            # Pad by repeating
            pad_size = self.num_points - len(pc)
            pad_indices = np.random.choice(len(pc), pad_size, replace=True)
            pc = np.concatenate([pc, pc[pad_indices]], axis=0)

        # Take only xyz (first 3 dims)
        if pc.shape[-1] > 3:
            pc = pc[:, :3]

        # Normalize to unit sphere
        centroid = pc.mean(axis=0)
        pc = pc - centroid
        max_dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if max_dist > 0:
            pc = pc / max_dist

        # Apply augmentation if enabled
        if self.augment:
            pc = self._augment_pointcloud(pc)

        return pc.astype(np.float32)

    def _augment_pointcloud(self, pc: np.ndarray) -> np.ndarray:
        """Apply point cloud augmentations."""
        # Random rotation around Y-axis (up)
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation = np.array([
            [cos_t, 0, sin_t],
            [0, 1, 0],
            [-sin_t, 0, cos_t]
        ])
        pc = pc @ rotation.T

        # Random scale
        scale = np.random.uniform(0.9, 1.1)
        pc = pc * scale

        # Random jitter
        jitter = np.random.normal(0, 0.01, pc.shape)
        pc = pc + np.clip(jitter, -0.03, 0.03)

        return pc

    def _load_image(self, image_path: Path) -> Image.Image:
        """Load image and preserve native resolution for the model processor."""
        return Image.open(str(image_path)).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        sample = self.samples[idx]
        scene_id = sample["scene_id"]
        scene = self.scene_data[scene_id]

        result = {
            "sample_id": sample["question_id"],
            "scene_id": scene_id,
            "question": sample["question"],
            "answers": sample["answer"],  # Primary answer
            "all_answers": sample["all_answers"],  # All valid answers
            "object_ids": sample["object_ids"],
            "object_names": sample["object_names"],
        }

        # Load point cloud if needed
        if self.modality in ["pointcloud", "both"] and scene["pointcloud_path"]:
            pc = self._load_pointcloud(scene["pointcloud_path"])
            result["pointcloud"] = torch.from_numpy(pc).float()
        else:
            result["pointcloud"] = None

        # Load image if needed
        if self.modality in ["image", "both"] and scene["image_path"]:
            img = self._load_image(scene["image_path"])
            if self.transform:
                img = self.transform(img)
            result["image"] = img
        else:
            result["image"] = None

        result["audio"] = None  # Not used

        return result


def collate_scanqa_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for ScanQA dataset."""
    if len(batch) == 0:
        return {}

    result = {
        "sample_ids": [s["sample_id"] for s in batch],
        "scene_ids": [s["scene_id"] for s in batch],
        "questions": [s["question"] for s in batch],
        "answers": [s["answers"] for s in batch],
        "all_answers": [s["all_answers"] for s in batch],
    }

    # Stack point clouds if present
    if batch[0].get("pointcloud") is not None:
        result["pointclouds"] = torch.stack([s["pointcloud"] for s in batch])
    else:
        result["pointclouds"] = None

    # Stack images if present (assumes they're already tensors from transform)
    if batch[0].get("image") is not None:
        if isinstance(batch[0]["image"], torch.Tensor):
            result["images"] = torch.stack([s["image"] for s in batch])
        else:
            # PIL images - keep as list for processor
            result["images"] = [s["image"] for s in batch]
    else:
        result["images"] = None

    return result
