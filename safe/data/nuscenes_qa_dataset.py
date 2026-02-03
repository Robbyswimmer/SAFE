# -*- coding: utf-8 -*-
"""
NuScenes-QA dataset for point cloud + image QA composition experiments.

NuScenes-QA: ~6K QA pairs from autonomous driving scenes (day + night).
Available on Hugging Face - NO REGISTRATION REQUIRED.

Supports three modalities:
- Point cloud only (LIDAR_TOP)
- Image only (6-view cameras)
- Point cloud + Image (composition)

Dataset: https://huggingface.co/datasets/KevinNotSmile/nuscenes-qa-mini
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset
from PIL import Image

# Hugging Face datasets for easy download
try:
    from datasets import load_dataset as hf_load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class NuScenesQADataset(Dataset):
    """
    NuScenes-QA dataset for 3D question answering with point cloud + multi-view images.

    Key features:
    - 5D LiDAR point clouds (X, Y, Z, intensity, distance)
    - 6-view RGB camera images
    - Complex multi-hop reasoning questions
    - Day and night scenes
    - NO REGISTRATION REQUIRED (via Hugging Face)

    Supports three modes:
    - "pointcloud": Only point cloud input
    - "image": Only RGB image input (uses front camera by default)
    - "both": Point cloud + image composition
    """

    dataset_name = "nuscenes_qa"

    # Answer classes in the dataset (29 classes)
    ANSWER_CLASSES = None  # Will be populated from data

    def __init__(
        self,
        data_path: Optional[str | Path] = None,  # Not used - data from HF
        split: str = "train",
        scene_type: str = "day",  # "day" or "night"
        modality: str = "both",  # "pointcloud", "image", or "both"
        num_points: int = 8192,
        image_size: int = 224,
        camera_view: str = "CAM_FRONT",  # Which camera to use for image-only
        augment: bool = None,
        transform=None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize NuScenes-QA dataset.

        Args:
            data_path: Not used (data downloaded from Hugging Face)
            split: "train" or "validation"
            scene_type: "day" or "night"
            modality: Which modalities to load ("pointcloud", "image", "both")
            num_points: Number of points to sample from point cloud
            image_size: Size to resize images to
            camera_view: Which camera view to use ("CAM_FRONT", "CAM_FRONT_LEFT", etc.)
            augment: Whether to apply augmentation (default: True for train)
            transform: Optional image transform
            cache_dir: Optional Hugging Face cache directory
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "Hugging Face datasets required. Install with: pip install datasets"
            )

        self.split = split.lower()
        if self.split == "val":
            self.split = "validation"
        self.scene_type = scene_type.lower()
        self.modality = modality
        self.num_points = num_points
        self.image_size = image_size
        self.camera_view = camera_view
        self.augment = augment if augment is not None else (self.split == "train")
        self.transform = transform

        # Camera views available
        self.camera_views = [
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
        ]

        # Load from Hugging Face
        print(f"[NuScenes-QA] Loading {self.scene_type} {self.split} from Hugging Face...")
        self.hf_dataset = hf_load_dataset(
            "KevinNotSmile/nuscenes-qa-mini",
            self.scene_type,
            split=self.split,
            cache_dir=cache_dir,
        )

        print(f"[NuScenes-QA] Loaded {len(self.hf_dataset)} samples ({self.scene_type}, {self.split}, modality={modality})")

        # Build answer vocabulary
        self._build_answer_vocab()

    def _build_answer_vocab(self):
        """Build mapping of answers to indices."""
        unique_answers = set()
        for sample in self.hf_dataset:
            unique_answers.add(sample["answer"])
        self.answer_to_idx = {a: i for i, a in enumerate(sorted(unique_answers))}
        self.idx_to_answer = {i: a for a, i in self.answer_to_idx.items()}
        self.ANSWER_CLASSES = list(self.answer_to_idx.keys())
        print(f"[NuScenes-QA] Found {len(self.ANSWER_CLASSES)} unique answer classes")

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _process_pointcloud(self, lidar_data: List) -> np.ndarray:
        """Process 5D LiDAR point cloud data."""
        # lidar_data is a list/array of 5D points (X, Y, Z, intensity, distance)
        pc = np.array(lidar_data, dtype=np.float32)

        # Handle case where data might be flattened
        if pc.ndim == 1:
            # Assume 5D points
            pc = pc.reshape(-1, 5)

        # Take only XYZ for now (first 3 dims)
        pc_xyz = pc[:, :3] if pc.shape[-1] >= 3 else pc

        # Subsample to target number of points
        n_points = len(pc_xyz)
        if n_points > self.num_points:
            indices = np.random.choice(n_points, self.num_points, replace=False)
            pc_xyz = pc_xyz[indices]
        elif n_points < self.num_points and n_points > 0:
            # Pad by repeating
            pad_size = self.num_points - n_points
            pad_indices = np.random.choice(n_points, pad_size, replace=True)
            pc_xyz = np.concatenate([pc_xyz, pc_xyz[pad_indices]], axis=0)
        elif n_points == 0:
            # Empty point cloud - fill with zeros
            pc_xyz = np.zeros((self.num_points, 3), dtype=np.float32)

        # Normalize to unit sphere
        centroid = pc_xyz.mean(axis=0)
        pc_xyz = pc_xyz - centroid
        max_dist = np.max(np.sqrt(np.sum(pc_xyz ** 2, axis=1)))
        if max_dist > 0:
            pc_xyz = pc_xyz / max_dist

        # Apply augmentation if enabled
        if self.augment:
            pc_xyz = self._augment_pointcloud(pc_xyz)

        return pc_xyz.astype(np.float32)

    def _augment_pointcloud(self, pc: np.ndarray) -> np.ndarray:
        """Apply point cloud augmentations."""
        # Random rotation around Z-axis (up in driving scenes)
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ])
        pc = pc @ rotation.T

        # Random scale
        scale = np.random.uniform(0.9, 1.1)
        pc = pc * scale

        # Random jitter
        jitter = np.random.normal(0, 0.01, pc.shape)
        pc = pc + np.clip(jitter, -0.03, 0.03)

        return pc

    def _process_image(self, image_data: List, view: str = None) -> Image.Image:
        """Process camera image data."""
        if view is None:
            view = self.camera_view

        # Image data from HF dataset - convert to PIL
        if isinstance(image_data, Image.Image):
            img = image_data
        elif isinstance(image_data, np.ndarray):
            img = Image.fromarray(image_data.astype(np.uint8))
        elif isinstance(image_data, list):
            # Might be nested or flattened
            arr = np.array(image_data, dtype=np.uint8)
            if arr.ndim == 3:
                img = Image.fromarray(arr)
            else:
                # Flatten to 3D assuming HxWxC
                # This depends on the actual format
                img = Image.fromarray(arr.reshape(-1, -1, 3))
        else:
            raise ValueError(f"Unknown image format: {type(image_data)}")

        # Convert to RGB
        img = img.convert("RGB")

        # Resize if needed
        if img.size != (self.image_size, self.image_size):
            resample = getattr(Image, 'Resampling', Image).BILINEAR
            img = img.resize((self.image_size, self.image_size), resample)

        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        sample = self.hf_dataset[idx]

        result = {
            "sample_id": sample.get("token", f"{self.scene_type}_{idx}"),
            "scene_id": sample.get("token", f"{self.scene_type}_{idx}"),
            "question": sample["question"],
            "answers": sample["answer"],  # Primary answer
            "all_answers": [sample["answer"]],  # Single answer in this dataset
            "answer_idx": self.answer_to_idx.get(sample["answer"], 0),
        }

        # Load point cloud if needed
        if self.modality in ["pointcloud", "both"]:
            lidar_data = sample.get("LIDAR_TOP")
            if lidar_data is not None:
                pc = self._process_pointcloud(lidar_data)
                result["pointcloud"] = torch.from_numpy(pc).float()
            else:
                result["pointcloud"] = None
        else:
            result["pointcloud"] = None

        # Load image if needed
        if self.modality in ["image", "both"]:
            # Get specified camera view
            img_data = sample.get(self.camera_view)
            if img_data is not None:
                img = self._process_image(img_data)
                if self.transform:
                    img = self.transform(img)
                result["image"] = img
            else:
                result["image"] = None
        else:
            result["image"] = None

        result["audio"] = None  # Not used

        return result

    def get_all_camera_views(self, idx: int) -> Dict[str, Image.Image]:
        """Get all 6 camera views for a sample."""
        sample = self.hf_dataset[idx]
        views = {}
        for view in self.camera_views:
            img_data = sample.get(view)
            if img_data is not None:
                views[view] = self._process_image(img_data, view)
        return views


def collate_nuscenes_qa_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for NuScenes-QA dataset."""
    if len(batch) == 0:
        return {}

    result = {
        "sample_ids": [s["sample_id"] for s in batch],
        "scene_ids": [s["scene_id"] for s in batch],
        "questions": [s["question"] for s in batch],
        "answers": [s["answers"] for s in batch],
        "all_answers": [s["all_answers"] for s in batch],
        "answer_idxs": torch.tensor([s["answer_idx"] for s in batch], dtype=torch.long),
    }

    # Stack point clouds if present
    if batch[0].get("pointcloud") is not None:
        pcs = [s["pointcloud"] for s in batch if s["pointcloud"] is not None]
        if pcs:
            result["pointclouds"] = torch.stack(pcs)
        else:
            result["pointclouds"] = None
    else:
        result["pointclouds"] = None

    # Stack images if present
    if batch[0].get("image") is not None:
        imgs = [s["image"] for s in batch if s["image"] is not None]
        if imgs:
            if isinstance(imgs[0], torch.Tensor):
                result["images"] = torch.stack(imgs)
            else:
                # PIL images - keep as list for processor
                result["images"] = imgs
        else:
            result["images"] = None
    else:
        result["images"] = None

    return result


def download_nuscenes_qa(cache_dir: Optional[str] = None):
    """Download NuScenes-QA dataset from Hugging Face."""
    if not HF_AVAILABLE:
        raise ImportError(
            "Hugging Face datasets required. Install with: pip install datasets"
        )

    print("Downloading NuScenes-QA dataset from Hugging Face...")
    print("This dataset is freely available - no registration required!")
    print()

    for scene_type in ["day", "night"]:
        for split in ["train", "validation"]:
            print(f"  Downloading {scene_type}/{split}...")
            hf_load_dataset(
                "KevinNotSmile/nuscenes-qa-mini",
                scene_type,
                split=split,
                cache_dir=cache_dir,
            )

    print()
    print("Download complete!")
    print("Dataset statistics:")
    print("  Day scenes: 2,229 train + 2,229 validation = 4,458 samples")
    print("  Night scenes: 659 train + 659 validation = 1,318 samples")
    print("  Total: 5,776 samples")


if __name__ == "__main__":
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download the dataset")
    parser.add_argument("--test", action="store_true", help="Test loading the dataset")
    args = parser.parse_args()

    if args.download:
        download_nuscenes_qa()

    if args.test or not args.download:
        print("\nTesting dataset loading...")
        dataset = NuScenesQADataset(split="train", scene_type="day", modality="both")
        print(f"\nSample 0:")
        sample = dataset[0]
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answers']}")
        if sample["pointcloud"] is not None:
            print(f"  Point cloud shape: {sample['pointcloud'].shape}")
        if sample["image"] is not None:
            print(f"  Image: {type(sample['image'])}")
