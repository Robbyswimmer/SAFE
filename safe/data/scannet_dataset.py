"""
ScanNet dataset for point cloud + image composition experiments.

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

# ScanNet scene types (13-class subset used for benchmark)
SCANNET_SCENE_TYPES = [
    "apartment",
    "bathroom",
    "bedroom",
    "bookstore",
    "classroom",
    "closet",
    "conference_room",
    "copy_room",
    "dining_room",
    "game_room",
    "hallway",
    "kitchen",
    "laundry_room",
    "living_room",
    "lobby",
    "office",
    "storage",
]


class ScanNetDataset(Dataset):
    """
    ScanNet dataset for scene classification with point cloud + image.

    Supports three modes:
    - "pointcloud": Only point cloud input
    - "image": Only RGB image input
    - "both": Point cloud + image composition
    """

    dataset_name = "scannet"
    num_classes = len(SCANNET_SCENE_TYPES)
    class_names = SCANNET_SCENE_TYPES

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        modality: str = "both",  # "pointcloud", "image", or "both"
        num_points: int = 8192,
        image_size: int = 224,
        augment: bool = None,
        transform=None,
    ):
        """
        Initialize ScanNet dataset.

        Args:
            data_path: Root data directory containing preprocessed scannet/
            split: "train" or "val"
            modality: Which modalities to load ("pointcloud", "image", "both")
            num_points: Number of points to sample from point cloud
            image_size: Size to resize images to
            augment: Whether to apply augmentation (default: True for train)
            transform: Optional image transform
        """
        self.data_path = Path(data_path)
        self.split = split.lower()
        self.modality = modality
        self.num_points = num_points
        self.image_size = image_size
        self.augment = augment if augment is not None else (self.split == "train")
        self.transform = transform

        # Load metadata
        self.samples = self._load_samples()

        print(f"[ScanNet] Loaded {len(self.samples)} samples ({self.split}, modality={modality})")

    def _load_samples(self) -> List[Dict]:
        """Load sample metadata from JSON."""
        metadata_file = self.data_path / "scannet" / f"{self.split}_samples.json"

        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}\n"
                f"Run preprocessing first: python experiments/scannet_composition/scripts/preprocess_scannet.py"
            )

        with open(metadata_file) as f:
            samples = json.load(f)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pointcloud(self, pc_path: str) -> np.ndarray:
        """Load and preprocess point cloud."""
        pc = np.load(pc_path)  # (N, 3) or (N, 6) with normals

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
        scale = np.random.uniform(0.8, 1.2)
        pc = pc * scale

        # Random jitter
        jitter = np.random.normal(0, 0.01, pc.shape)
        pc = pc + np.clip(jitter, -0.05, 0.05)

        return pc

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        img = Image.open(image_path).convert("RGB")

        # Resize if needed
        if img.size != (self.image_size, self.image_size):
            # Use Resampling.BILINEAR for PIL >= 9.1.0 compatibility
            resample = getattr(Image, 'Resampling', Image).BILINEAR
            img = img.resize((self.image_size, self.image_size), resample)

        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        sample = self.samples[idx]

        scene_id = sample["scene_id"]
        scene_type = sample["scene_type"]
        label = self.class_names.index(scene_type) if scene_type in self.class_names else -1

        result = {
            "sample_id": scene_id,
            "question": "What type of room is this?",
            "answers": scene_type.replace("_", " "),
            "label": label,
            "valid": label >= 0,
        }

        # Load point cloud if needed
        if self.modality in ["pointcloud", "both"]:
            pc_path = self.data_path / "scannet" / sample["pointcloud_path"]
            pc = self._load_pointcloud(str(pc_path))
            result["pointcloud"] = torch.from_numpy(pc).float()
        else:
            result["pointcloud"] = None

        # Load image if needed
        if self.modality in ["image", "both"]:
            img_path = self.data_path / "scannet" / sample["image_path"]
            img = self._load_image(str(img_path))
            if self.transform:
                img = self.transform(img)
            result["image"] = img
        else:
            result["image"] = None

        result["audio"] = None  # Not used

        return result


def collate_scannet_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for ScanNet dataset."""
    # Filter valid samples
    batch = [s for s in batch if s.get("valid", True)]

    if len(batch) == 0:
        return {}

    result = {
        "sample_ids": [s["sample_id"] for s in batch],
        "questions": [s["question"] for s in batch],
        "answers": [s["answers"] for s in batch],
        "labels": torch.tensor([s["label"] for s in batch], dtype=torch.long),
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
